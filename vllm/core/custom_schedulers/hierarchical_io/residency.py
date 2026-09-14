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


class PrefetchResidencyDirectory:
    """维护 prefetch unit 的 logical block 可见性。

    sparse consumer 只需要调用 ``require_layer``。它不会读取 GranuleKV，也不会
    触碰 allocator；如果某个尚未 READY 的 block 被使用，会立即报错，避免
    以后把“部分恢复”错误地当成完整 prefix 命中。
    """

    def __init__(self) -> None:
        self._units: Dict[str, _UnitResidency] = {}

    def register(self, request: AsyncKVTransferRequest) -> None:
        if request.prefetch_plan_id is None:
            return
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
                raise RuntimeError(
                    f"sparse KV residency is incomplete for layer "
                    f"{layer_index}: request={request.request_id}")
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

    def _get(self, request_id: str) -> _UnitResidency:
        try:
            return self._units[request_id]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown prefetch residency request: {request_id}") from exc
