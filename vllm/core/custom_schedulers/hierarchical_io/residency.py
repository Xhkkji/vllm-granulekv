# SPDX-License-Identifier: Apache-2.0

"""层级/稀疏 KV 的最小驻留目录。

这个目录只记录 scheduler 已声明的 logical block 是否可被当前 layer
consumer 使用，不持有 GPU tensor、allocator block 或 GranuleKV handle。真正的
物理 block 释放仍由 BlockSpaceManager 负责；因此它可以安全地留在
Worker-local runtime 中，不改变 vLLM 原生 block table。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from vllm.core.block_reservation import BlockMapping, LogicalBlockKey

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferRequest, AsyncKVTransferState)


@dataclass
class _UnitResidency:
    request: AsyncKVTransferRequest
    # None 表示 dense consumer；此时原生 attention 仍按完整 block table
    # 访问，本目录只观测 READY 生命周期，不施加 sparse block 约束。
    requested: Optional[frozenset[int]]
    requested_by_layer: Optional[Tuple[Optional[frozenset[int]], ...]]
    resident: set[int]
    local_block_start: Optional[int]
    state: AsyncKVTransferState = AsyncKVTransferState.QUEUED
    evicted: bool = False


@dataclass
class _CorrectionPlanResidency:
    plan_id: str
    seq_group_id: str
    reservation_id: str
    num_prefix_blocks: int
    catalog: Dict[int, Tuple[Tuple[int, int], LogicalBlockKey]]
    predicted_by_layer: Dict[int, frozenset[int]]
    resident_by_layer: Dict[int, set[int]]


@dataclass(frozen=True)
class SparseKVLayerPredictionState:
    plan_id: str
    seq_group_id: str
    reservation_id: str
    num_prefix_blocks: int
    predicted_prefix_blocks: Tuple[int, ...]
    resident_prefix_blocks: Tuple[int, ...]
    # Prefix-indexed (GPU physical block, SSD logical block) pairs. A negative
    # pair denotes a block already owned by the resident allocation and is not
    # a valid correction source.
    correction_mapping_by_logical: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class SparseKVCorrectionProjection:
    plan_id: str
    seq_group_id: str
    reservation_id: str
    layer_index: int
    block_mapping: BlockMapping
    logical_blocks: Tuple[LogicalBlockKey, ...]


class PrefetchResidencyDirectory:
    """维护 prefetch unit 的 logical block 可见性。

    sparse consumer 只需要调用 ``require_layer``。它不会读取 GranuleKV，也不会
    触碰 allocator；如果某个尚未 READY 的 block 被使用，会立即报错，避免
    以后把“部分恢复”错误地当成完整 prefix 命中。
    """

    def __init__(self) -> None:
        self._units: Dict[str, _UnitResidency] = {}
        self._correction_plans: Dict[str, _CorrectionPlanResidency] = {}
        self._restore_stats = {
            "prediction_restore_requests": 0,
            "prediction_restore_blocks": 0,
            "prediction_restore_fragments": 0,
        }
        self._stats = {
            "residency_miss": 0,
            "residency_miss_blocks": 0,
        }

    def stats(self) -> Dict[str, int]:
        """Return physical residency counters for experiment reporting."""
        return dict(self._stats)

    def restore_stats(self, reset: bool = False) -> Dict[str, int]:
        result = dict(self._restore_stats)
        if reset:
            for name in self._restore_stats:
                self._restore_stats[name] = 0
        return result

    def register(self, request: AsyncKVTransferRequest) -> None:
        if request.prefetch_plan_id is None:
            return
        self._register_correction_plan(request)
        if request.prefetch_plan_id in self._correction_plans:
            self._restore_stats["prediction_restore_requests"] += 1
            layer_count = (0 if request.layer_range is None else
                           request.layer_range[1] - request.layer_range[0])
            block_layers = len(request.logical_blocks) * layer_count
            self._restore_stats["prediction_restore_blocks"] += block_layers
            self._restore_stats["prediction_restore_fragments"] += (
                block_layers * 2)
        if (request.consumer_block_indices is None
                and request.consumer_blocks_by_layer is None
                and request.consumer_num_blocks is None):
            # dense layer prefetch 由 wait_ready 屏障保证，不需要额外维护逐 block
            # Python set。长上下文下这能避免 blocks × windows 的控制面开销。
            return
        if request.request_id in self._units:
            raise RuntimeError(
                f"duplicate residency request: {request.request_id}")
        requested = (None if request.consumer_block_indices is None else
                     frozenset(request.consumer_block_indices))
        prefetch_required = requested
        if request.consumer_num_blocks is not None and requested is None:
            prefetch_required = frozenset(range(request.consumer_num_blocks))
        requested_by_layer = (
            None if request.consumer_blocks_by_layer is None else tuple(
                None if indices is None else frozenset(indices)
                for indices in request.consumer_blocks_by_layer))
        restored = frozenset(key.logical_index
                             for key in request.logical_blocks)
        # reservation mapping 只包含 SSD -> GPU 的部分；requested 中剩余的
        # block 原本已经在 HBM，可以从 plan 建立时视为 resident。
        resident = (set() if prefetch_required is None else
                    set(prefetch_required.difference(restored)))
        self._units[request.request_id] = _UnitResidency(
            request=request,
            requested=requested,
            requested_by_layer=requested_by_layer,
            resident=resident,
            local_block_start=request.consumer_local_block_start)

    def mark_pending(self, request_id: str) -> None:
        if request_id not in self._units:
            return
        unit = self._get(request_id)
        unit.state = AsyncKVTransferState.PENDING

    def mark_ready(self, request_id: str) -> None:
        if request_id not in self._units:
            return
        unit = self._get(request_id)
        unit.state = AsyncKVTransferState.READY
        # A sparse unit may carry the total logical block count for validation
        # while restoring only a subset.  The subset must remain the source of
        # truth; expanding to range(consumer_num_blocks) would make an
        # incomplete restore look fully resident.
        if unit.requested is not None:
            unit.resident.update(unit.requested)
        elif unit.request.consumer_num_blocks is not None:
            unit.resident.update(range(unit.request.consumer_num_blocks))
        else:
            unit.resident.update(
                key.logical_index for key in unit.request.logical_blocks)
        self._mark_plan_unit_ready(unit.request)

    def mark_error(self, request_id: str) -> None:
        if request_id not in self._units:
            return
        unit = self._get(request_id)
        unit.state = AsyncKVTransferState.ERROR
        unit.resident.clear()

    def require_layer(
        self,
        request_ids: Sequence[str],
        layer_index: int,
        sequence_lengths_by_request: Optional[Mapping[str, int]] = None,
        block_size: Optional[int] = None,
    ) -> Optional[Tuple[int, ...]]:
        """确认当前 layer 的 KV 已驻留，并返回 sparse consumer block 集合。

        batch 中多个请求若给出不同 sparse 集合，当前 v0 attention metadata
        还无法分别表达，直接拒绝而不是静默扩大成 dense 访问。
        """
        request_id_set = frozenset(request_ids)
        selected: Optional[Tuple[int, ...]] = None
        for unit in self._units.values():
            request = unit.request
            if (request.seq_group_id not in request_id_set
                    or request.layer_range is None
                    or not (request.layer_range[0] <= layer_index <
                            request.layer_range[1])):
                continue
            if unit.evicted:
                self._stats["residency_miss"] += 1
                if unit.requested is not None:
                    self._stats["residency_miss_blocks"] += len(
                        unit.requested)
                raise RuntimeError(
                    f"KV blocks were evicted before layer {layer_index}: "
                    f"{request.request_id}")
            if unit.state != AsyncKVTransferState.READY:
                raise RuntimeError(
                    f"KV blocks are not ready for layer {layer_index}: "
                    f"{request.request_id} state={unit.state.name}")
            layer_selected = unit.requested
            if unit.requested_by_layer is not None:
                layer_selected = unit.requested_by_layer[
                    layer_index - request.layer_range[0]]
            if (layer_selected is not None
                    and request.consumer_local_block_start is not None
                    and sequence_lengths_by_request is not None
                    and block_size is not None):
                sequence_length = sequence_lengths_by_request.get(
                    request.seq_group_id)
                if sequence_length is None or sequence_length <= 0:
                    raise RuntimeError(
                        "missing current sequence length for sparse KV "
                        f"request={request.seq_group_id}")
                if block_size <= 0:
                    raise RuntimeError("sparse KV block size must be positive")
                current_block_count = ((sequence_length + block_size - 1) //
                                       block_size)
                dynamic_tail = tuple(
                    range(request.consumer_local_block_start,
                          current_block_count))
                if dynamic_tail:
                    # Local blocks are already produced by the live request;
                    # they require no GranuleKV completion event.
                    unit.resident.update(dynamic_tail)
                    layer_selected = frozenset(layer_selected).union(
                        dynamic_tail)
            if (layer_selected is not None
                    and not layer_selected.issubset(unit.resident)):
                missing = frozenset(layer_selected).difference(unit.resident)
                self._stats["residency_miss"] += 1
                self._stats["residency_miss_blocks"] += len(missing)
                raise RuntimeError(
                    f"sparse KV residency is incomplete for layer "
                    f"{layer_index}: request={request.request_id} "
                    f"missing_blocks={tuple(sorted(missing))}")
            current = (None if layer_selected is None else
                       tuple(sorted(layer_selected)))
            if selected is not None and current != selected:
                raise RuntimeError(
                    "one batch has incompatible sparse KV access plans")
            selected = current
        return selected

    def evict_unit(self, request_id: str) -> Tuple[int, ...]:
        """标记一个已消费 sparse unit 为不可见，返回 logical block 下标。

        这里只做协议层标记，不直接释放 GPU block。等 sparse attention 提供
        allocator 回收入口后，调用方可用返回值执行物理释放；在此之前保留
        地址，避免破坏现有 dense/layerwise 正确路径。
        """
        unit = self._get(request_id)
        if unit.requested is None:
            return ()
        if unit.state != AsyncKVTransferState.READY:
            raise RuntimeError("cannot evict a sparse KV unit before READY")
        evicted = tuple(sorted(unit.resident))
        unit.resident.clear()
        unit.evicted = True
        return evicted

    def forget(self, request_id: str) -> None:
        self._units.pop(request_id, None)

    def forget_seq_groups(self, seq_group_ids: Sequence[str]) -> None:
        finished = frozenset(seq_group_ids)
        for request_id, unit in tuple(self._units.items()):
            if unit.request.seq_group_id in finished:
                del self._units[request_id]
        for plan_id, plan in tuple(self._correction_plans.items()):
            if plan.seq_group_id in finished:
                del self._correction_plans[plan_id]

    def prediction_state(
        self,
        request_ids: Sequence[str],
        layer_index: int,
    ) -> Optional[SparseKVLayerPredictionState]:
        request_id_set = frozenset(request_ids)
        matches = tuple(
            plan for plan in self._correction_plans.values()
            if plan.seq_group_id in request_id_set
            and layer_index in plan.predicted_by_layer)
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(
                "one layer matched multiple sparse correction plans")
        plan = matches[0]
        return SparseKVLayerPredictionState(
            plan_id=plan.plan_id,
            seq_group_id=plan.seq_group_id,
            reservation_id=plan.reservation_id,
            num_prefix_blocks=plan.num_prefix_blocks,
            predicted_prefix_blocks=tuple(
                sorted(plan.predicted_by_layer[layer_index])),
            resident_prefix_blocks=tuple(
                sorted(plan.resident_by_layer[layer_index])),
            correction_mapping_by_logical=tuple(
                tuple(plan.catalog[index][0]) if index in plan.catalog else
                (-1, -1)
                for index in range(plan.num_prefix_blocks)),
        )

    def project_correction(
        self,
        state: SparseKVLayerPredictionState,
        layer_index: int,
        missing_blocks: Sequence[int],
    ) -> SparseKVCorrectionProjection:
        plan = self._correction_plans.get(state.plan_id)
        if plan is None:
            raise RuntimeError("sparse correction catalog is unavailable")
        missing = tuple(sorted(set(int(index) for index in missing_blocks)))
        if not missing:
            raise ValueError("sparse correction requires missing blocks")
        resident = plan.resident_by_layer.get(layer_index)
        if resident is None:
            raise RuntimeError("sparse correction layer is not registered")
        missing = tuple(index for index in missing if index not in resident)
        if not missing:
            raise RuntimeError("sparse correction no longer has missing blocks")
        unavailable = tuple(index for index in missing
                            if index not in plan.catalog)
        if unavailable:
            raise RuntimeError(
                "missing sparse blocks have no correction mapping: "
                f"{unavailable}")
        projected = tuple(plan.catalog[index] for index in missing)
        return SparseKVCorrectionProjection(
            plan_id=plan.plan_id,
            seq_group_id=plan.seq_group_id,
            reservation_id=plan.reservation_id,
            layer_index=layer_index,
            block_mapping=tuple(item[0] for item in projected),
            logical_blocks=tuple(item[1] for item in projected),
        )

    def mark_corrected(self, projection: SparseKVCorrectionProjection) -> None:
        plan = self._correction_plans.get(projection.plan_id)
        if plan is None:
            raise RuntimeError("sparse correction plan was discarded")
        resident = plan.resident_by_layer.get(projection.layer_index)
        if resident is None:
            raise RuntimeError("sparse correction layer is not registered")
        resident.update(key.logical_index
                        for key in projection.logical_blocks)

    def _register_correction_plan(
        self,
        request: AsyncKVTransferRequest,
    ) -> None:
        plan_id = request.prefetch_plan_id
        assert plan_id is not None
        prefix_count = request.consumer_local_block_start
        if prefix_count is None:
            return
        plan = self._correction_plans.get(plan_id)
        if request.correction_block_mapping is not None:
            if plan is not None:
                raise RuntimeError("duplicate sparse correction catalog")
            catalog = {
                key.logical_index: (tuple(pair), key)
                for pair, key in zip(request.correction_block_mapping,
                                     request.correction_logical_blocks or ())
            }
            if len(catalog) != len(request.correction_block_mapping):
                raise RuntimeError(
                    "sparse correction catalog has duplicate logical blocks")
            if any(index < 0 or index >= prefix_count for index in catalog):
                raise RuntimeError(
                    "sparse correction catalog is outside immutable prefix")
            plan = _CorrectionPlanResidency(
                plan_id=plan_id,
                seq_group_id=request.seq_group_id,
                reservation_id=request.reservation_id,
                num_prefix_blocks=prefix_count,
                catalog=catalog,
                predicted_by_layer={},
                resident_by_layer={},
            )
            self._correction_plans[plan_id] = plan
        if plan is None:
            return
        if (plan.seq_group_id != request.seq_group_id
                or plan.reservation_id != request.reservation_id
                or plan.num_prefix_blocks != prefix_count):
            raise RuntimeError("inconsistent sparse correction plan metadata")
        if request.layer_range is None:
            raise RuntimeError("sparse correction plan requires layer range")
        baseline = set(range(prefix_count)).difference(plan.catalog)
        for layer_index in range(*request.layer_range):
            relative = layer_index - request.layer_range[0]
            selection = (request.consumer_block_indices
                         if request.consumer_blocks_by_layer is None else
                         request.consumer_blocks_by_layer[relative])
            if selection is None:
                continue
            predicted = frozenset(int(index) for index in selection
                                  if int(index) < prefix_count)
            if not predicted:
                raise RuntimeError(
                    "sparse correction plan has empty prefix prediction")
            plan.predicted_by_layer[layer_index] = predicted
            plan.resident_by_layer[layer_index] = set(baseline)

    def _mark_plan_unit_ready(self, request: AsyncKVTransferRequest) -> None:
        plan_id = request.prefetch_plan_id
        if plan_id is None or request.layer_range is None:
            return
        plan = self._correction_plans.get(plan_id)
        if plan is None:
            return
        restored = {key.logical_index for key in request.logical_blocks}
        for layer_index in range(*request.layer_range):
            resident = plan.resident_by_layer.get(layer_index)
            if resident is not None:
                resident.update(restored)

    def _get(self, request_id: str) -> _UnitResidency:
        try:
            return self._units[request_id]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown prefetch residency request: {request_id}") from exc
