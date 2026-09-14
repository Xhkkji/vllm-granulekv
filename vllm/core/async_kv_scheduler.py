# SPDX-License-Identifier: Apache-2.0

"""可通过参数选择的 V0 block 事务式异步 KV 调度器。"""

from __future__ import annotations

import os
import time
import hashlib
from collections import deque
from typing import Callable, Iterable, List, Optional, Set, Tuple

import vllm.envs as envs
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block.interfaces import BlockAllocator
from vllm.core.block_reservation import (BlockPrefixRestoreReservation,
                                         BlockSwapReservation)
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (Scheduler, SchedulerSwappedInOutputs,
                                 SchedulingBudget, PreemptionMode)
from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVExecutionMarker, AsyncKVSchedulePolicy, AsyncKVTransferEvent,
    AsyncKVTransferOperation, AsyncKVTransferRequest, AsyncKVTransferState)
from vllm.core.custom_schedulers.hierarchical_io import (
    HierarchicalIOConfig, HierarchicalLayerBarrierConfig,
    HierarchicalRestoreController, SparseKVAccessPlan,
    SparseKVPlanFeedback, select_prefetch_unit_blocks)
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

logger = init_logger(__name__)

ASYNC_KV_STRATEGY_NATIVE = "native"
ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT = "chunked_priority_preempt"
ASYNC_KV_STRATEGY_LONG_CONTEXT_STRESS = "long_context_stress"
ASYNC_KV_STRATEGIES = {
    ASYNC_KV_STRATEGY_NATIVE,
    ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT,
    ASYNC_KV_STRATEGY_LONG_CONTEXT_STRESS,
}


def _prefix_page_index_key(seq_group: SequenceGroup,
                           block_size: int,
                           num_prefix_blocks: int) -> Optional[str]:
    """Return one identity for the exact complete prefix stored/restored.

    A finished populate request may already contain generated suffix tokens.
    Hashing the whole sequence would therefore miss when a later request
    restores the same complete prefix.  Limit the identity to the block range
    represented by the storage lookup/reservation.
    """
    sequences = seq_group.get_seqs()
    if len(sequences) != 1:
        return None
    if block_size <= 0 or num_prefix_blocks <= 0:
        return None
    sequence = sequences[0]
    token_count = num_prefix_blocks * block_size
    token_ids = sequence.get_token_ids()
    if token_count > len(token_ids):
        raise ValueError("page-index prefix exceeds sequence token count")
    payload = repr((tuple(token_ids[:token_count]), sequence.extra_hash(),
                    block_size, num_prefix_blocks)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class AsyncKVScheduler(Scheduler):
    """带有独立异步 KV 策略扩展的原生 V0 Scheduler。

    父类继续负责 prefill、decode、请求顺序和 victim 选择；本类只把同步
    swap 动作替换为 reserve -> GranuleKV transfer -> commit/abort。实验路径通过
    ``--scheduler-cls`` 显式选择，默认原生 Scheduler 完全不受影响。
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        if not scheduler_config.chunked_prefill_enabled:
            raise ValueError(
                "AsyncKVScheduler requires chunked prefill so that "
                "asynchronous KV completion is consumed at bounded "
                "model-execution intervals")
        super().__init__(scheduler_config, cache_config, lora_config,
                         pipeline_parallel_size, output_proc_callback)
        # read 请求从 swapped 队列取出后暂存在 loading；write 请求仍保留
        # 在 swapped 队列，但在 SSD 副本提交前记录于 saving，禁止被重新
        # swap-in。两个方向共用同一个多槽状态机，容量必须与 GranuleKV request
        # table 一致；小于 1 时直接失败，避免运行中出现隐式单槽回退。
        self.async_kv_policy = AsyncKVSchedulePolicy(
            max_in_flight=envs.VLLM_GRANULEKV_MAX_IN_FLIGHT)
        self.async_kv_scheduler_strategy = os.getenv(
            "VLLM_GRANULEKV_ASYNC_SCHEDULER_STRATEGY",
            ASYNC_KV_STRATEGY_NATIVE)
        if self.async_kv_scheduler_strategy not in ASYNC_KV_STRATEGIES:
            raise ValueError(
                "unsupported VLLM_GRANULEKV_ASYNC_SCHEDULER_STRATEGY="
                f"{self.async_kv_scheduler_strategy}; expected one of "
                f"{sorted(ASYNC_KV_STRATEGIES)}")
        self.loading: dict[str, SequenceGroup] = {}
        self.saving: dict[str, SequenceGroup] = {}
        # prefix populate/restore 与请求 preemption 的 swap 生命周期分开保存。
        # 两者只复用底层 GranuleKV transfer queue，READY 后的队列迁移语义不同，
        # 因此不能通过伪造 SWAPPED 状态混在 loading/saving 中。
        self.granulekv_prefix_enabled = envs.VLLM_GRANULEKV_PREFIX_ENABLE
        if (self.granulekv_prefix_enabled
                and not cache_config.enable_prefix_caching):
            raise ValueError(
                "VLLM_GRANULEKV_PREFIX_ENABLE requires --enable-prefix-caching")
        self.prefix_loading: dict[str, SequenceGroup] = {}
        self.prefix_saving: dict[str, SequenceGroup] = {}
        self.hierarchical_io_config = HierarchicalIOConfig.from_env()
        self.hierarchical_layer_barrier_config = (
            HierarchicalLayerBarrierConfig.from_env())
        if (self.hierarchical_io_config.enabled
                and not self.granulekv_prefix_enabled):
            raise ValueError(
                "hierarchical GranuleKV I/O requires VLLM_GRANULEKV_PREFIX_ENABLE")
        if (self.hierarchical_layer_barrier_config.enabled
                and not self.hierarchical_io_config.enabled):
            raise ValueError(
                "hierarchical layer barrier requires hierarchical GranuleKV I/O")
        if (self.hierarchical_layer_barrier_config.enabled
                and not self.hierarchical_io_config.rolling.enabled):
            # forward 期间 Worker 只能轮询已提交的 GranuleKV handle，无法安全地
            # 越过 Engine/Scheduler 自己激活 queued request。因此 Step 4 的
            # 最小正确版本要求一个父 restore 的所有 window 同时 in-flight。
            # 否则模型到达第 5 个 window 时可能等待尚未 submit 的第 5 笔 I/O。
            required_windows = (
                (self.hierarchical_io_config.num_layers +
                 self.hierarchical_io_config.window_layers - 1) //
                self.hierarchical_io_config.window_layers)
            if self.async_kv_policy.max_in_flight < required_windows:
                raise ValueError(
                    "hierarchical layer barrier requires "
                    "VLLM_GRANULEKV_MAX_IN_FLIGHT >= number of layer windows "
                    f"({required_windows})")
        if (self.hierarchical_io_config.rolling.enabled
                and not self.hierarchical_layer_barrier_config.enabled):
            raise ValueError(
                "rolling hierarchical I/O requires the layer barrier")
        if self.hierarchical_io_config.consumer_enabled:
            if not self.hierarchical_layer_barrier_config.enabled:
                raise ValueError(
                    "sparse consumer requires the layer barrier")
        self.hierarchical_prefix_restores = HierarchicalRestoreController()
        # key 是父 reservation/plan id，而不是 window request id。一个请求
        # 无论拆成多少层窗口，都只占一个 scheduler sequence slot。
        self.hierarchical_prefix_loading: dict[str, SequenceGroup] = {}
        # 首窗 READY 后进入“已准入但禁止 dispatch”的显式状态。Step 2 只
        # 建立控制面边界；Step 4 接入 model layer barrier 前不能放进 running。
        self.hierarchical_prefix_admitted: Set[str] = set()
        # rolling 模式第一次向 Worker 发送完整 plan 后，后续 queued unit
        # 不再由 Engine 的普通 drain 自动激活，而由 model layer progress
        # 触发。集合只属于 scheduler 控制面，不拥有 block/data。
        self._hierarchical_rolling_staged_plans: Set[str] = set()
        # Sparse restores must never be written back as a complete dense
        # prefix: unselected logical blocks were intentionally not restored.
        self._sparse_restore_seq_group_ids: Set[str] = set()
        self._sparse_restore_plans: dict[str, SparseKVAccessPlan] = {}
        self._staged_async_kv_discards: list[str] = []
        self._prefix_restore_admission_blocked = False
        logger.info(
            "[GRANULEKV_PREFIX] phase=init enabled=%s block_size=%d",
            self.granulekv_prefix_enabled,
            cache_config.block_size,
        )
        logger.info(
            "[GRANULEKV_HIERARCHICAL] phase=init enabled=%s "
            "num_layers=%d window_layers=%d dispatch_gate=%s",
            self.hierarchical_io_config.enabled,
            self.hierarchical_io_config.num_layers,
            self.hierarchical_io_config.window_layers,
            ("layer_barrier"
             if self.hierarchical_layer_barrier_config.enabled else "closed"),
        )
        self._chunked_priority_waiting_id: Optional[str] = None
        self._chunked_priority_preempted_victims: Set[str] = set()
        self._long_context_stress_min_free_blocks = int(
            os.getenv("VLLM_GRANULEKV_LONG_CONTEXT_STRESS_MIN_FREE_BLOCKS", "128"))
        self._long_context_stress_min_victim_blocks = int(
            os.getenv("VLLM_GRANULEKV_LONG_CONTEXT_STRESS_MIN_VICTIM_BLOCKS", "64"))
        self._long_context_stress_max_preempts_per_waiting = max(
            1,
            int(
                os.getenv(
                    "VLLM_GRANULEKV_LONG_CONTEXT_STRESS_MAX_PREEMPTS_PER_WAITING",
                    "1")))
        self._long_context_stress_allow_prefill = bool(
            int(os.getenv("VLLM_GRANULEKV_LONG_CONTEXT_STRESS_ALLOW_PREFILL", "1")))
        self._long_context_stress_require_priority = bool(
            int(os.getenv("VLLM_GRANULEKV_LONG_CONTEXT_STRESS_REQUIRE_PRIORITY",
                          "1")))
        self._long_context_stress_proactive = bool(
            int(os.getenv("VLLM_GRANULEKV_LONG_CONTEXT_STRESS_PROACTIVE", "0")))
        # READY 请求先进入 running，真正被 Engine dispatch 前保留一个观测
        # 标记。这样可以区分“状态已经可运行”和“已经进入模型执行 batch”。
        self._async_kv_execution_markers: dict[
            str, AsyncKVExecutionMarker] = {}
        # active I/O abort 不能立刻释放源或目标 block：GranuleKV 仍可能对这些
        # 地址执行 DMA。先记录取消意图，等 READY/ERROR 后再统一回收。
        self._cancelled_async_kv_requests: Set[str] = set()

    @property
    def scheduler_strategy(self) -> str:
        """返回稳定的策略名称，供日志和实验结果标记使用。"""
        return f"async_kv:{self.async_kv_scheduler_strategy}"

    def accept_sparse_kv_plan_feedback(
        self, feedback: Iterable[SparseKVPlanFeedback]) -> None:
        """Cache the latest policy prediction for a stable prefix key."""
        if not envs.VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE:
            return
        for item in feedback:
            try:
                plan = SparseKVAccessPlan.from_layer_selections(
                    num_blocks=item.num_prefix_blocks,
                    block_indices_by_layer=item.block_indices_by_layer,
                    source="quest_dynamic",
                )
            except (TypeError, ValueError) as exc:
                logger.warning("discarding invalid sparse restore feedback: %s",
                               exc)
                continue
            if plan.num_layers != item.num_layers:
                logger.warning("discarding sparse restore feedback with invalid "
                               "layer count: key=%s", item.page_index_key)
                continue
            self._sparse_restore_plans[item.page_index_key] = plan

    def _peek_sparse_restore_plan(
        self, page_index_key: Optional[str], num_blocks: int
    ) -> Optional[SparseKVAccessPlan]:
        if not envs.VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE:
            return None
        if not self.hierarchical_io_config.consumer_enabled:
            return None
        if page_index_key is None:
            return None
        plan = self._sparse_restore_plans.get(page_index_key)
        if plan is None or plan.num_blocks != num_blocks:
            return None
        if plan.num_layers != self.hierarchical_io_config.num_layers:
            return None
        return plan

    def _schedule_chunked_prefill(self):
        """在原生 chunked path 前注入可选的等待感知抢占策略。

        父类仍负责 continuous batching、running/decode 优先、chunked prefill
        token budget 和最终 SchedulerOutputs。本钩子只在 priority policy 下，
        当 waiting 高优先级请求因 KV block 不足无法进入 running 时，提前把
        一个低优先级 running victim 送入现有 async swap-out 路径。
        """
        self._sort_waiting_for_prefix_restore()
        self._maybe_start_granulekv_prefix_restore()

        if (self.async_kv_scheduler_strategy
                == ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT):
            self._preempt_low_priority_running_for_waiting()
        elif (self.async_kv_scheduler_strategy
              == ASYNC_KV_STRATEGY_LONG_CONTEXT_STRESS):
            self._preempt_long_context_victims_for_waiting()
        self._sort_waiting_for_prefix_restore()
        if self._prefix_restore_admission_blocked:
            # preempt 可能刚刚释放了 HBM；在进入父类 admission 前立即重试，
            # 避免本来可 restore 的请求在同一轮退化为完整 recompute。
            self._maybe_start_granulekv_prefix_restore()
        return self._schedule_chunked_prefill_with_reserved_slots()

    def _sort_waiting_for_prefix_restore(self) -> None:
        """让 prefix admission 与 vLLM priority policy 使用同一顺序。"""
        if (self.granulekv_prefix_enabled
                and self.scheduler_config.policy == "priority"):
            self.waiting = type(self.waiting)(
                sorted(self.waiting, key=self._get_priority))

    def _num_async_read_sequence_slots(self) -> int:
        """返回 read reservation 已占用、但父类 budget 看不到的 seq 数。"""
        return sum(
            group.get_max_num_running_seqs()
            for group in (*self.loading.values(),
                          *self.prefix_loading.values(),
                          *self.hierarchical_prefix_loading.values()))

    def _schedule_chunked_prefill_with_reserved_slots(self):
        """在不复制父类调度器的前提下，为异步 read 保留 sequence slot。

        loading 请求已经持有 GPU target，READY 后会回到 running。这里只从
        本轮可见 waiting 中扣除等量 admission 容量；已有 running 和仍可
        容纳的新请求继续交给原生 chunked scheduler，因此 prefix I/O 不会
        冻结整个 waiting 队列。
        """
        reserved_slots = self._num_async_read_sequence_slots()
        if reserved_slots == 0 and not self._prefix_restore_admission_blocked:
            return super()._schedule_chunked_prefill()

        running_slots = sum(group.get_max_num_running_seqs()
                            for group in self.running)
        available_slots = max(
            0,
            self.scheduler_config.max_num_seqs - running_slots -
            reserved_slots,
        )
        if self._prefix_restore_admission_blocked:
            # 当前队首确认存在 SSD prefix，但本轮 HBM 尚不足。让 running
            # 继续推进并释放空间；不能绕过它，也不能交给父类重新计算。
            available_slots = 0

        visible_waiting = deque()
        hidden_waiting = deque()
        for seq_group in self.waiting:
            required_slots = seq_group.get_max_num_running_seqs()
            if required_slots <= available_slots and not hidden_waiting:
                visible_waiting.append(seq_group)
                available_slots -= required_slots
            else:
                # 不越过第一个放不下的请求，保持 FCFS/priority 的公平性。
                hidden_waiting.append(seq_group)

        self.waiting = type(self.waiting)(visible_waiting)
        try:
            outputs = super()._schedule_chunked_prefill()
        finally:
            # 父类可能把 preempted 请求放回 waiting；它们应位于本轮未暴露
            # 的请求之前。下一轮 prefix 前置逻辑会按 priority 再统一排序。
            self.waiting.extend(hidden_waiting)
        return outputs

    def _maybe_start_granulekv_prefix_restore(self) -> int:
        """在父类 admission 前尝试恢复一个 SSD prefix。

        每次最多预留当前可用的 sequence slot 和 GranuleKV prefix loading slot。
        pending target 已由 allocator 隔离，所以 running/swapped/其他 write
        不再阻止 restore；READY 后才把请求放入 running。
        """
        if not self.granulekv_prefix_enabled:
            return 0
        self._prefix_restore_admission_blocked = False

        max_seq_slots = self.scheduler_config.max_num_seqs - (
            sum(group.get_max_num_running_seqs() for group in self.running) +
            self._num_async_read_sequence_slots())
        max_prefix_loads = (self.async_kv_policy.max_in_flight -
                            len(self.loading) - len(self.prefix_loading) -
                            len(self.hierarchical_prefix_loading))
        if max_seq_slots <= 0 or max_prefix_loads <= 0:
            return 0

        started = 0
        for seq_group in tuple(self.waiting):
            if started >= min(max_seq_slots, max_prefix_loads):
                break
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            if len(waiting_seqs) != 1 or seq_group.is_encoder_decoder():
                break
            seq = waiting_seqs[0]
            storage_prefix_blocks, gpu_prefix_blocks = (
                self.block_manager.get_granulekv_cached_prefix_block_counts(seq))
            if storage_prefix_blocks <= gpu_prefix_blocks:
                # 当前最高优先级请求应先走原生 HBM-hit/recompute admission，
                # 不能让后面的 SSD restore 提前占用它需要的 sequence/HBM slot。
                break
            alloc_status = self.block_manager.can_reserve_granulekv_prefix_restore(
                storage_prefix_blocks, gpu_prefix_blocks)
            if alloc_status == AllocStatus.LATER:
                self._prefix_restore_admission_blocked = True
                break
            if alloc_status == AllocStatus.NEVER:
                break

            try:
                reservation = self.block_manager.reserve_granulekv_prefix_restore(
                    seq_group,
                    num_prefix_blocks=storage_prefix_blocks,
                    num_gpu_cached_blocks=gpu_prefix_blocks,
                )
            except BlockAllocator.NoFreeBlocksError:
                # allocator 的可用块数量包含可驱逐 cache；逐块预留仍可能在
                # 压力边界失败。将它视为本轮 LATER，但不能越过当前请求，也
                # 不在每个 decode step 输出 INFO 干扰性能。
                logger.debug(
                    "[GRANULEKV_PREFIX] phase=restore_deferred "
                    "seq_group_id=%s storage_prefix_blocks=%d "
                    "gpu_prefix_blocks=%d reason=no_free_blocks",
                    seq_group.request_id,
                    storage_prefix_blocks,
                    gpu_prefix_blocks,
                )
                self._prefix_restore_admission_blocked = True
                break
            self.waiting.remove(seq_group)
            if self.hierarchical_io_config.enabled:
                page_index_key = _prefix_page_index_key(
                    seq_group, self.cache_config.block_size,
                    reservation.num_prefix_blocks)
                access_plan = self._peek_sparse_restore_plan(
                    page_index_key, reservation.num_prefix_blocks)
                self._enqueue_hierarchical_prefix_restore(
                    seq_group, reservation, access_plan=access_plan)
                if access_plan is not None and page_index_key is not None:
                    self._sparse_restore_plans.pop(page_index_key, None)
            else:
                request = self._enqueue_async_kv_transfer(
                    seq_group,
                    reservation,
                    AsyncKVTransferOperation.READ,
                    prefix=True,
                )
                logger.info(
                    "[GRANULEKV_PREFIX] phase=restore_queued request_id=%s "
                    "seq_group_id=%s storage_prefix_blocks=%d "
                    "gpu_prefix_blocks=%d read_blocks=%d",
                    request.request_id,
                    seq_group.request_id,
                    storage_prefix_blocks,
                    gpu_prefix_blocks,
                    len(reservation.block_mapping),
                )
            started += 1
        return started

    def _enqueue_hierarchical_prefix_restore(
        self,
        seq_group: SequenceGroup,
        reservation: BlockPrefixRestoreReservation,
        *,
        access_plan: Optional[SparseKVAccessPlan] = None,
    ) -> None:
        """按通用 prefetch plan 投影并建立多个 GranuleKV 子请求。"""
        sequences = seq_group.get_seqs(status=SequenceStatus.WAITING)
        if len(sequences) != 1:
            raise ValueError(
                "hierarchical sparse restore requires one waiting sequence")
        sequence = sequences[0]
        sequence_block_count = (
            sequence.get_len() + self.cache_config.block_size - 1
        ) // self.cache_config.block_size
        if sequence_block_count < reservation.num_prefix_blocks:
            raise RuntimeError(
                "prefix restore has more blocks than the target sequence")
        local_tail = tuple(
            range(reservation.num_prefix_blocks, sequence_block_count))
        plan = self.hierarchical_io_config.build_plan(
            reservation.reservation_id,
            num_blocks=reservation.num_prefix_blocks,
            access_plan=access_plan,
        )
        requests = []
        selected_block_counts = []
        for unit in plan.units:
            # dense layer plan 的 block_indices 为 None，投影结果与历史逻辑
            # 完全一致。sparse-style selector 只改变 unit 的 block 选择，不
            # 复制 reservation、队列和 GranuleKV 生命周期代码。
            block_mapping, logical_blocks = select_prefetch_unit_blocks(
                unit, reservation.block_mapping, reservation.logical_blocks)
            selected_block_counts.append(len(block_mapping))
            if plan.consumer_enabled and plan.access_plan is None:
                # Dense warmup is represented as a consumer-visible all-block
                # residency set.  The I/O mapping may still omit blocks already
                # present in HBM, so residency must use the full logical prefix
                # count rather than len(block_mapping).
                consumer_block_indices = tuple(range(sequence_block_count))
                consumer_blocks_by_layer = tuple(
                    consumer_block_indices for _ in range(unit.num_layers))
            elif plan.consumer_enabled:
                # The restore reservation only owns immutable prefix blocks.
                # The suffix belongs to the live request and is filled by the
                # normal scheduler; it must not enter the SSD mapping.
                layer_selections = tuple(
                    None if selection is None else tuple(
                        sorted(set(selection).union(local_tail)))
                    for selection in (unit.consumer_blocks_by_layer or ()))
                consumer_blocks_by_layer = layer_selections
                consumer_block_indices = tuple(
                    sorted({index for selection in layer_selections
                            if selection is not None for index in selection}))
            else:
                consumer_block_indices = unit.block_indices
                consumer_blocks_by_layer = unit.consumer_blocks_by_layer
            requests.append(
                self.async_kv_policy.enqueue(
                    seq_group.request_id,
                    reservation.reservation_id,
                    AsyncKVTransferOperation.READ,
                    block_mapping,
                    logical_blocks,
                    layer_range=unit.layer_range,
                    prefetch_plan_id=plan.plan_id,
                    prefetch_unit_index=unit.index,
                    consumer_block_indices=consumer_block_indices,
                    consumer_blocks_by_layer=consumer_blocks_by_layer,
                    consumer_num_blocks=(
                        sequence_block_count if plan.consumer_enabled else
                        (None if plan.access_plan is None else
                         plan.access_plan.num_blocks)),
                    consumer_local_block_start=(
                        reservation.num_prefix_blocks
                        if plan.consumer_enabled else None),
                    sparse_page_index_key=_prefix_page_index_key(
                        seq_group, self.cache_config.block_size,
                        reservation.num_prefix_blocks),
                ))
        requests = tuple(requests)
        self.hierarchical_prefix_restores.register(
            plan, tuple(request.request_id for request in requests))
        self.hierarchical_prefix_loading[plan.plan_id] = seq_group

        logger.info(
            "[GRANULEKV_HIERARCHICAL] phase=plan_queued plan_id=%s "
            "seq_group_id=%s windows=%d blocks=%d selector=%s "
            "profiling_only=%s dynamic_restore=%s "
            "selected_blocks_per_unit=%s",
            plan.plan_id,
            seq_group.request_id,
            len(plan.units),
            len(reservation.block_mapping),
            plan.block_selector,
            plan.profiling_only,
            str(access_plan is not None).lower(),
            ",".join(str(count) for count in selected_block_counts),
        )
        for request, unit in zip(requests, plan.units):
            logger.info(
                "[GRANULEKV_HIERARCHICAL] phase=window_queued plan_id=%s "
                "request_id=%s window=%d/%d layer_range=[%d,%d) "
                "selected_blocks=%d",
                plan.plan_id,
                request.request_id,
                unit.index,
                len(plan.units),
                unit.start_layer,
                unit.end_layer,
                len(request.block_mapping),
            )

    def _preempt_long_context_victims_for_waiting(self) -> int:
        """主动制造长上下文 HBM/SSD 迁移压力的实验策略。

        这个策略只挂在 AsyncKVScheduler 的显式 env 分支下。它不改变原生
        admission / batching 代码，而是在进入父类 chunked scheduler 前，
        当高优先级 waiting 请求与长 running 请求竞争 HBM 时，主动把少量
        低优先级长 running 请求送入现有 async swap-out 路径。
        """
        if not self.waiting or not self.running:
            return 0
        if self.saving:
            return 0
        if (self.async_kv_policy.in_flight_count >=
                self.async_kv_policy.max_in_flight):
            return 0

        if self.scheduler_config.policy == "priority":
            self.waiting = type(self.waiting)(
                sorted(self.waiting, key=self._get_priority))
        waiting_head = self.waiting[0]
        if waiting_head.request_id != self._chunked_priority_waiting_id:
            self._chunked_priority_waiting_id = waiting_head.request_id
            self._chunked_priority_preempted_victims.clear()

        waiting_alloc_status = self.block_manager.can_allocate(waiting_head)
        free_blocks = self.block_manager.get_num_free_gpu_blocks()
        if waiting_alloc_status == AllocStatus.OK:
            if not self._long_context_stress_proactive:
                return 0
            if free_blocks >= self._long_context_stress_min_free_blocks:
                return 0
            if len(self.running) >= self.scheduler_config.max_num_seqs:
                return 0
        elif waiting_alloc_status == AllocStatus.NEVER:
            return 0

        preempted = 0
        selected_victims: Set[str] = set()
        while preempted < self._long_context_stress_max_preempts_per_waiting:
            victim = self._select_long_context_stress_victim(
                waiting_head, selected_victims)
            if victim is None:
                break
            selected_victims.add(victim.request_id)
            self.running.remove(victim)
            blocks_to_swap_out: List[Tuple[int, int]] = []
            preempted_mode = self._preempt(victim, blocks_to_swap_out)
            if preempted_mode == PreemptionMode.SWAP:
                self.swapped.append(victim)
            else:
                self.waiting.appendleft(victim)
            preempted += 1
            if envs.VLLM_V0_SWAP_TRACE:
                logger.info(
                    "[V0_SWAP_TRACE][AsyncKV][Scheduler] "
                    "phase=long_context_stress_preempt "
                    "victim_seq_group_id=%s waiting_seq_group_id=%s "
                    "victim_priority=%s waiting_priority=%s "
                    "victim_blocks=%d free_gpu_blocks=%d "
                    "waiting_alloc_status=%s mode=%s",
                    victim.request_id,
                    waiting_head.request_id,
                    self._get_priority(victim),
                    self._get_priority(waiting_head),
                    self._estimate_seq_group_blocks(victim),
                    free_blocks,
                    waiting_alloc_status.name,
                    preempted_mode.name,
                )
        return preempted

    def _select_long_context_stress_victim(
        self,
        waiting_head: SequenceGroup,
        selected_victims: Set[str],
    ) -> Optional[SequenceGroup]:
        candidates = []
        for seq_group in self.running:
            if seq_group.request_id in selected_victims:
                continue
            if (not self._long_context_stress_allow_prefill
                    and seq_group.is_prefill()):
                continue
            victim_blocks = self._estimate_seq_group_blocks(seq_group)
            if victim_blocks < self._long_context_stress_min_victim_blocks:
                continue
            if (self.scheduler_config.policy == "priority"
                    and self._long_context_stress_require_priority
                    and self._get_priority(seq_group) <=
                    self._get_priority(waiting_head)):
                continue
            candidates.append(seq_group)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda group:
            (self._get_priority(group), self._estimate_seq_group_blocks(group)),
        )

    def _estimate_seq_group_blocks(self, seq_group: SequenceGroup) -> int:
        return sum(seq.n_blocks for seq in seq_group.get_seqs())

    def _preempt_low_priority_running_for_waiting(self) -> int:
        if self.scheduler_config.policy != "priority":
            return 0
        if not self.waiting or not self.running:
            return 0

        # Keep waiting in the same priority order used by vLLM's native
        # priority policy: lower priority value wins, then earlier arrival.
        self.waiting = type(self.waiting)(
            sorted(self.waiting, key=self._get_priority))
        waiting_head = self.waiting[0]
        if waiting_head.request_id != self._chunked_priority_waiting_id:
            self._chunked_priority_waiting_id = waiting_head.request_id
            self._chunked_priority_preempted_victims.clear()

        candidates = [
            seq_group for seq_group in self.running
            if seq_group.request_id
            not in self._chunked_priority_preempted_victims
        ]
        if not candidates:
            return 0
        victim = max(candidates, key=self._get_priority)
        if self._get_priority(victim) <= self._get_priority(waiting_head):
            return 0

        # If the high-priority waiting request can already be admitted, let
        # the native chunked scheduler handle it without extra churn.
        if self.block_manager.can_allocate(waiting_head) == AllocStatus.OK:
            return 0

        self.running.remove(victim)
        blocks_to_swap_out: List[Tuple[int, int]] = []
        preempted_mode = self._preempt(victim, blocks_to_swap_out)
        self._chunked_priority_preempted_victims.add(victim.request_id)
        if preempted_mode == PreemptionMode.SWAP:
            self.swapped.append(victim)
        else:
            self.waiting.appendleft(victim)
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Scheduler] "
                "phase=chunked_priority_preempt victim_seq_group_id=%s "
                "waiting_seq_group_id=%s victim_priority=%s "
                "waiting_priority=%s mode=%s",
                victim.request_id,
                waiting_head.request_id,
                self._get_priority(victim),
                self._get_priority(waiting_head),
                preempted_mode.name,
            )
        return 1

    def _enqueue_async_kv_transfer(
        self,
        seq_group: SequenceGroup,
        reservation: BlockSwapReservation | BlockPrefixRestoreReservation,
        operation: AsyncKVTransferOperation,
        prefix: bool = False,
    ) -> AsyncKVTransferRequest:
        """把 block reservation 登记为等待 GranuleKV transfer slot。"""
        page_index_key = None
        if prefix:
            if isinstance(reservation, BlockPrefixRestoreReservation):
                num_prefix_blocks = reservation.num_prefix_blocks
            else:
                sequences = seq_group.get_seqs()
                if len(sequences) != 1:
                    raise ValueError(
                        "sparse page metadata requires one prefix sequence")
                num_prefix_blocks = (len(sequences[0].get_token_ids()) //
                                     self.cache_config.block_size)
            page_index_key = _prefix_page_index_key(
                seq_group, self.cache_config.block_size, num_prefix_blocks)
        request = self.async_kv_policy.enqueue(
            seq_group.request_id,
            reservation.reservation_id,
            operation,
            reservation.block_mapping,
            reservation.logical_blocks,
            sparse_page_index_key=page_index_key,
        )
        if prefix:
            lifecycle = (self.prefix_loading
                         if operation == AsyncKVTransferOperation.READ else
                         self.prefix_saving)
        else:
            lifecycle = (self.loading
                         if operation == AsyncKVTransferOperation.READ else
                         self.saving)
        lifecycle[request.request_id] = seq_group
        return request

    def has_active_async_kv_transfer(self) -> bool:
        """只要至少一笔 transfer 已提交，Engine 就需要轮询 Worker。"""
        if self.hierarchical_io_config.rolling.enabled:
            return self.async_kv_policy.has_outstanding
        return self.async_kv_policy.in_flight_count > 0

    def has_unfinished_seqs(self) -> bool:
        """把 reservation 中的请求计入 Engine 未完成判断。"""
        return (bool(self.loading) or bool(self.saving)
                or bool(self.prefix_loading) or bool(self.prefix_saving)
                or bool(self.hierarchical_prefix_loading)
                or super().has_unfinished_seqs())

    def get_num_unfinished_seq_groups(self) -> int:
        """返回原生三队列和独立 loading 请求的总数。"""
        # saving 请求通常已经在原生 swapped 队列中，只为不在原生三队列
        # 中的 loading 请求额外计数，避免统计翻倍。
        return (len(self.loading) + len(self.prefix_loading)
                + len(self.hierarchical_prefix_loading)
                + len(self.prefix_saving)
                + super().get_num_unfinished_seq_groups())

    def apply_async_kv_event(self, event: AsyncKVTransferEvent) -> None:
        """应用 Worker 返回的后端无关异步 transfer 事件。

        这里只更新控制面状态；block table 的 commit/abort 统一在下一次
        Engine 调度边界完成。
        """
        self.async_kv_policy.apply_event(event)

    def complete_ready_async_kv_transfers(self) -> None:
        """在调度边界提交已完成 reservation，并处理失败或取消。"""
        ready = self.async_kv_policy.pop_ready()
        ready_groups: List[SequenceGroup] = []
        for request in ready:
            if self.hierarchical_prefix_restores.contains_request(
                    request.request_id):
                self._complete_hierarchical_prefix_window(
                    request, succeeded=True, ready_groups=ready_groups)
                continue
            is_prefix_restore = request.request_id in self.prefix_loading
            is_prefix_store = request.request_id in self.prefix_saving
            lifecycle = (self.loading
                         if request.operation == AsyncKVTransferOperation.READ
                         else self.saving)
            if is_prefix_restore:
                lifecycle = self.prefix_loading
            elif is_prefix_store:
                lifecycle = self.prefix_saving
            seq_group = lifecycle.pop(request.request_id, None)
            if seq_group is None:
                raise RuntimeError(
                    f"missing async KV sequence group: {request.request_id}")
            if request.request_id in self._cancelled_async_kv_requests:
                self._cancelled_async_kv_requests.remove(request.request_id)
                # DMA 已经完成，此时 abort reservation 可以安全释放目标
                # block；随后释放仍由正式 block table 持有的源 block。
                if is_prefix_restore:
                    self.block_manager.abort_granulekv_prefix_restore(
                        request.reservation_id)
                else:
                    self.block_manager.abort_block_swap(
                        request.reservation_id)
                self._remove_from_swapped(seq_group)
                super()._free_finished_seq_group(seq_group)
                continue

            if is_prefix_restore:
                restored_prefix_tokens = self.block_manager.commit_granulekv_prefix_restore(
                    request.reservation_id)
                self._advance_restored_prefix_frontier(
                    seq_group, restored_prefix_tokens)
            else:
                self.block_manager.commit_block_swap(request.reservation_id)
            committed_ns = time.monotonic_ns()
            if envs.VLLM_V0_SWAP_TRACE:
                logger.info(
                    "[V0_SWAP_TRACE][AsyncKV][Scheduler] phase=commit "
                    "operation=%s request_id=%s seq_group_id=%s "
                    "commit_monotonic_ns=%d",
                    request.operation.value,
                    request.request_id,
                    seq_group.request_id,
                    committed_ns,
                )
            if is_prefix_restore:
                for seq in seq_group.get_seqs(
                        status=SequenceStatus.WAITING):
                    seq.status = SequenceStatus.RUNNING
                ready_groups.append(seq_group)
                logger.info(
                    "[GRANULEKV_PREFIX] phase=restore_ready request_id=%s "
                    "seq_group_id=%s",
                    request.request_id,
                    seq_group.request_id,
                )
            elif is_prefix_store:
                # prefix populate 已完成，正式 table 此时位于 storage；调用
                # storage 专用释放后，完整 CPU block 留在 prefix allocator
                # 的 LRU 中，后续相同 token hash 可以获取稳定的 GranuleKV block id。
                self._free_completed_granulekv_prefix_store(seq_group)
                logger.info(
                    "[GRANULEKV_PREFIX] phase=store_ready request_id=%s "
                    "seq_group_id=%s",
                    request.request_id,
                    seq_group.request_id,
                )
            elif request.operation == AsyncKVTransferOperation.READ:
                for seq in seq_group.get_seqs(
                        status=SequenceStatus.SWAPPED):
                    seq.status = SequenceStatus.RUNNING
                ready_groups.append(seq_group)
                self._async_kv_execution_markers[seq_group.request_id] = (
                    AsyncKVExecutionMarker(
                        request_id=request.request_id,
                        seq_group_id=seq_group.request_id,
                        promoted_monotonic_ns=committed_ns))
                if envs.VLLM_V0_SWAP_TRACE:
                    marker = self._async_kv_execution_markers[
                        seq_group.request_id]
                    logger.info(
                        "[V0_SWAP_TRACE][AsyncKV][Scheduler] phase=promote "
                        "operation=read request_id=%s seq_group_id=%s "
                        "promoted_monotonic_ns=%d",
                        marker.request_id,
                        marker.seq_group_id,
                        marker.promoted_monotonic_ns,
                    )

        # GranuleKV poll 失败时，异步请求已经不能再被 attention 使用。将它们
        # 标记为 ignored，并释放已经预留的 GPU block，避免 block 泄漏。
        for load in self.async_kv_policy.pop_errors():
            request = load.request
            if self.hierarchical_prefix_restores.contains_request(
                    request.request_id):
                self._complete_hierarchical_prefix_window(
                    request, succeeded=False, ready_groups=ready_groups)
                continue
            # abort 只登记“完成后丢弃”，不会中止正在进行的 DMA。后端既可能
            # 回 READY，也可能回 ERROR；两种终态都必须消费取消标记，否则
            # 控制面会留下一个永远无法再次完成的 request id。
            self._cancelled_async_kv_requests.discard(request.request_id)
            is_prefix_restore = request.request_id in self.prefix_loading
            is_prefix_store = request.request_id in self.prefix_saving
            lifecycle = (self.loading
                         if request.operation == AsyncKVTransferOperation.READ
                         else self.saving)
            if is_prefix_restore:
                lifecycle = self.prefix_loading
            elif is_prefix_store:
                lifecycle = self.prefix_saving
            seq_group = lifecycle.pop(request.request_id, None)
            if seq_group is None:
                raise RuntimeError(
                    f"missing failed async KV sequence group: "
                    f"{request.request_id}")
            if is_prefix_restore:
                self.block_manager.abort_granulekv_prefix_restore(
                    request.reservation_id)
            else:
                self.block_manager.abort_block_swap(request.reservation_id)
            self._remove_from_swapped(seq_group)
            if is_prefix_store:
                # populate 是可选缓存写；失败不能把已经正常完成的用户请求
                # 改成 ignored，只丢弃这次缓存并正常释放。
                super()._free_finished_seq_group(seq_group)
            else:
                for seq in seq_group.get_seqs():
                    if not seq.is_finished():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                super()._free_finished_seq_group(seq_group)

        # error 分支可能让父事务在本轮最后一个 window 才达到终态，因此
        # ready_groups 的统一入队必须放在 ready/error 两部分都处理完之后。
        for seq_group in reversed(ready_groups):
            if seq_group not in self.running:
                self.running.appendleft(seq_group)

    def _complete_hierarchical_prefix_window(
        self,
        request: AsyncKVTransferRequest,
        *,
        succeeded: bool,
        ready_groups: List[SequenceGroup],
    ) -> None:
        """推进一个 window，并在父事务终态执行唯一一次 commit/abort。"""
        if succeeded:
            progress = self.hierarchical_prefix_restores.mark_ready(
                request.request_id)
        else:
            progress = self.hierarchical_prefix_restores.mark_error(
                request.request_id)
        seq_group = self.hierarchical_prefix_loading.get(progress.plan_id)
        if seq_group is None:
            raise RuntimeError(
                f"missing hierarchical prefix plan: {progress.plan_id}")

        logger.info(
            "[GRANULEKV_HIERARCHICAL] phase=window_%s plan_id=%s "
            "request_id=%s seq_group_id=%s window=%d layer_range=[%d,%d)",
            "ready" if succeeded else "error",
            progress.plan_id,
            request.request_id,
            seq_group.request_id,
            progress.unit.index,
            progress.unit.start_layer,
            progress.unit.end_layer,
        )
        if (progress.first_unit_became_ready
                and (not progress.profiling_only or progress.consumer_enabled)):
            # 这是 Step 2 的 first-window-ready admission。该集合表示调度器
            # 已接受请求并可在未来交给 layer pipeline；当前不修改 RUNNING
            # 状态，确保 Step 4 之前完整 model forward 看不到半恢复 KV。
            self.hierarchical_prefix_admitted.add(progress.plan_id)
            logger.info(
                "[GRANULEKV_HIERARCHICAL] phase=first_window_admitted "
                "plan_id=%s seq_group_id=%s potential_restore_ttft_ms=%.3f "
                "dispatch_gate=%s",
                progress.plan_id,
                seq_group.request_id,
                (progress.first_unit_ready_monotonic_ns -
                 progress.plan_created_monotonic_ns) / 1.0e6,
                ("layer_barrier" if self.hierarchical_layer_barrier_config.
                 enabled else "closed"),
            )
            if self.hierarchical_layer_barrier_config.enabled:
                # Step 4 的唯一 scheduler 语义变化：首窗完成即可让 native
                # chunked prefill 调度 suffix。pending target 尚未 publish，
                # 只能由 worker 的逐层 barrier 读取，绝不成为全局 hash 命中。
                restored_prefix_tokens = (
                    self.block_manager.
                    admit_granulekv_prefix_restore_for_layer_barrier(
                        progress.plan_id))
                self._advance_restored_prefix_frontier(
                    seq_group, restored_prefix_tokens)
                for seq in seq_group.get_seqs(
                        status=SequenceStatus.WAITING):
                    seq.status = SequenceStatus.RUNNING
                ready_groups.append(seq_group)
                logger.info(
                    "[GRANULEKV_HIERARCHICAL] phase=layer_barrier_dispatch "
                    "plan_id=%s seq_group_id=%s restored_prefix_tokens=%d",
                    progress.plan_id,
                    seq_group.request_id,
                    restored_prefix_tokens,
                )

        if not progress.all_terminal:
            return

        self.hierarchical_prefix_admitted.discard(progress.plan_id)
        if (progress.all_ready and not progress.cancelled
                and progress.consumer_enabled):
            # Keep pending GPU targets private to this sequence and release the
            # temporary CPU source ownership.  This intentionally does not
            # publish a global dense prefix hash.
            self.block_manager.finalize_granulekv_prefix_working_set(
                progress.plan_id)
            self._sparse_restore_seq_group_ids.add(seq_group.request_id)
            logger.info(
                "[GRANULEKV_HIERARCHICAL] phase=sparse_restore_ready "
                "plan_id=%s seq_group_id=%s sparse_restore_ms=%.3f",
                progress.plan_id,
                seq_group.request_id,
                (time.monotonic_ns() - progress.plan_created_monotonic_ns)
                / 1.0e6,
            )
        elif (progress.all_ready and not progress.cancelled
              and not progress.profiling_only):
            if envs.VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE:
                # 环形 regions 中早期层已被后续层覆盖，只结束 reservation，
                # 不能把它发布成全局 GPU prefix cache 命中。
                restored_prefix_tokens = (
                    self.block_manager.finalize_granulekv_prefix_working_set(
                        progress.plan_id))
            else:
                restored_prefix_tokens = (
                    self.block_manager.commit_granulekv_prefix_restore(
                        progress.plan_id))
            # barrier 模式已在首窗 READY 时推进 frontier 并转为 RUNNING；
            # 全部 window 结束时仅发布 hash/SSD replica，不能重复入队。
            already_dispatched = self.hierarchical_layer_barrier_config.enabled
            if not already_dispatched:
                self._advance_restored_prefix_frontier(seq_group,
                                                       restored_prefix_tokens)
            committed_ns = time.monotonic_ns()
            if not already_dispatched:
                for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                    seq.status = SequenceStatus.RUNNING
                ready_groups.append(seq_group)
            logger.info(
                "[GRANULEKV_HIERARCHICAL] phase=full_restore_ready "
                "plan_id=%s seq_group_id=%s full_restore_ms=%.3f "
                "first_to_full_gap_ms=%.3f",
                progress.plan_id,
                seq_group.request_id,
                (committed_ns - progress.plan_created_monotonic_ns) / 1.0e6,
                (committed_ns - progress.first_unit_ready_monotonic_ns)
                / 1.0e6,
            )
        else:
            # 任一 window 失败，或者首窗准入后用户 abort，都不能发布这个
            # prefix。此处已经确认全部子 DMA 终态，释放 target 才是安全的。
            terminal_ns = time.monotonic_ns()
            self.block_manager.abort_granulekv_prefix_restore(progress.plan_id)
            if progress.profiling_only and progress.all_ready:
                # 非 dense selector 只用于验证 GranuleKV 细粒度 I/O 形态。因为只恢复
                # 了 prefix 的部分 blocks，当前还没有 sparse attention consumer
                # 与 partial-residency 语义，绝不能把完整 prefix 发布为 computed。
                # 这里把请求标记为 ignored 并释放 reservation/block table；它
                # 是 profiling 实验终点，不承诺端到端生成。
                for seq in seq_group.get_seqs():
                    if not seq.is_finished():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                super()._free_finished_seq_group(seq_group)
            else:
                for seq in seq_group.get_seqs():
                    if not seq.is_finished():
                        seq.status = (SequenceStatus.FINISHED_ABORTED
                                      if progress.cancelled else
                                      SequenceStatus.FINISHED_IGNORED)
                super()._free_finished_seq_group(seq_group)
            logger.info(
                "[GRANULEKV_HIERARCHICAL] phase=restore_aborted "
                "plan_id=%s seq_group_id=%s failed=%s cancelled=%s "
                "profiling_only=%s selector=%s profiling_restore_ms=%.3f",
                progress.plan_id,
                seq_group.request_id,
                progress.failed,
                progress.cancelled,
                progress.profiling_only,
                progress.block_selector,
                (terminal_ns - progress.plan_created_monotonic_ns) / 1.0e6,
            )

        self.hierarchical_prefix_restores.release(progress.plan_id)
        del self.hierarchical_prefix_loading[progress.plan_id]
        self._hierarchical_rolling_staged_plans.discard(progress.plan_id)

    @staticmethod
    def _advance_restored_prefix_frontier(
        seq_group: SequenceGroup,
        restored_prefix_tokens: int,
    ) -> None:
        """补齐直接从 loading 提升到 running 时的 cached-token 记账。

        原生 prefix admission 会把 ``cached + uncached`` 一起放进第一次
        ScheduledSequenceGroup；output processor 随后一次性推进 sequence 的
        computed frontier。GranuleKV restore READY 后直接进入 running，绕过了这次
        waiting admission。如果只发布 block hash 而不推进 sequence frontier，
        running scheduler 会把同一小段 suffix 重复执行，直到累计 token 数
        追上 prefix。

        全 prompt 命中是唯一例外：vLLM 必须重新执行最后一个 token 来生成
        prompt logits，所以最多推进到 ``prompt_len - 1``。这里只修改 sequence
        的逻辑计算边界；pending block 的发布仍由 block manager 事务负责。
        """
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            target = min(restored_prefix_tokens, max(0, seq.get_len() - 1))
            current = seq.get_num_computed_tokens()
            if target > current:
                seq.data.update_num_computed_tokens(target - current)

    def get_hierarchical_admitted_seq_group_ids(self) -> Tuple[str, ...]:
        """返回首窗已 READY、但 dispatch safety gate 尚未放行的请求。"""
        return tuple(
            seq_group.request_id
            for plan_id, seq_group in self.hierarchical_prefix_loading.items()
            if plan_id in self.hierarchical_prefix_admitted)

    def _free_completed_granulekv_prefix_store(
            self, seq_group: SequenceGroup) -> None:
        """完成 finished bookkeeping，并释放 storage-resident table。"""
        self._free_seq_group_cross_attn_blocks(seq_group)
        self._finished_requests_ids.append(seq_group.request_id)
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.block_manager.free_granulekv_prefix_store(seq)

    def _remove_from_swapped(self, seq_group: SequenceGroup) -> None:
        """按对象身份移除可能仍在原生 swapped 队列中的 write 请求。"""
        try:
            self.swapped.remove(seq_group)
        except ValueError:
            pass

    def consume_async_kv_execution_markers(
            self, seq_group_ids: Iterable[str]
    ) -> Tuple[AsyncKVExecutionMarker, ...]:
        """取出本轮即将 dispatch 的异步恢复请求观测标记。

        Scheduler 仍然是 sequence 状态的唯一修改者；Engine 这里只消费已经
        READY/RUNNING 的只读观测信息。若 READY 请求暂时没有被当前 budget
        选中，标记会保留到它第一次真正进入执行 batch 的轮次。
        """
        markers = []
        for seq_group_id in seq_group_ids:
            marker = self._async_kv_execution_markers.pop(seq_group_id, None)
            if marker is not None:
                markers.append(marker)
        return tuple(markers)

    def free_seq(self, seq: Sequence) -> None:
        """拦截 single-step output processor 的即时 finished free。

        V0 会在 stop checker 返回 finished 后立刻调用 ``free_seq``，早于
        ``free_finished_seq_groups``。GranuleKV prefix 必须在这里保住 GPU table，
        否则稍后的 group hook 已经没有可写入 GranuleKV 的 KV source。
        """
        if (self.granulekv_prefix_enabled and seq.is_finished()
                and seq.seq_id in self.block_manager.block_tables):
            seq_group = next(
                (group for group in self.running if any(
                    candidate is seq for candidate in group.get_seqs())),
                None,
            )
            if (seq_group is not None
                    and self._try_start_granulekv_prefix_store(seq_group)):
                return
        super().free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        """完成 group 清理时避免重复提交已经开始的 prefix store。"""
        if any(group is seq_group for group in self.prefix_saving.values()):
            return
        if self._try_start_granulekv_prefix_store(seq_group):
            return
        super()._free_finished_seq_group(seq_group)
        self._sparse_restore_seq_group_ids.discard(seq_group.request_id)

    def _try_start_granulekv_prefix_store(self,
                                    seq_group: SequenceGroup) -> bool:
        """正常释放前，用 GranuleKV 异步保存可复用的完整 prefix block。

        这是 GranuleKV prefix populate 唯一挂点，并且只存在于显式选择的
        ``AsyncKVScheduler``。原生 Scheduler 的 finished/free 行为没有改动。
        请求对客户端已经完成，但 block 资源会保留到 GranuleKV DONE；Engine 通过
        ``prefix_saving`` 继续推进空调度轮次，不会提前复用 DMA source。
        """
        if (not self.granulekv_prefix_enabled or not seq_group.is_finished()
                or envs.VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE
                or seq_group.request_id in self._sparse_restore_seq_group_ids):
            # working-set 已覆盖早期层，不能从当前 HBM regions 回写完整 KV。
            # 该验证模式直接丢弃结果，已有 SSD prefix 仍由 source replica 保留。
            return False
        seqs = seq_group.get_seqs()
        normally_finished = all(
            seq.status in (SequenceStatus.FINISHED_STOPPED,
                           SequenceStatus.FINISHED_LENGTH_CAPPED)
            for seq in seqs)
        has_full_block = any(seq.get_len() >= self.cache_config.block_size
                             for seq in seqs)
        has_block_table = all(seq.seq_id in self.block_manager.block_tables
                              for seq in seqs)
        if (not normally_finished or not has_full_block or not has_block_table
                or not self.block_manager.can_reserve_granulekv_prefix_store(
                    seq_group)):
            logger.debug(
                "[GRANULEKV_PREFIX] phase=store_skipped request_id=%s "
                "normally_finished=%s has_full_block=%s has_block_table=%s",
                seq_group.request_id,
                normally_finished,
                has_full_block,
                has_block_table,
            )
            return False

        try:
            reservation = self.block_manager.reserve_granulekv_prefix_store(seq_group)
        except Exception as exc:
            # Prefix 是可选的缓存优化；storage 紧张时保留正常完成语义，
            # 不把用户请求失败扩大成服务失败。
            logger.warning(
                "[GRANULEKV_PREFIX] skip store request_id=%s error=%s",
                seq_group.request_id,
                exc,
            )
            return False
        request = self._enqueue_async_kv_transfer(
            seq_group,
            reservation,
            AsyncKVTransferOperation.WRITE,
            prefix=True,
        )
        logger.info(
            "[GRANULEKV_PREFIX] phase=store_queued request_id=%s "
            "seq_group_id=%s write_blocks=%d reused_blocks=%d",
            request.request_id,
            seq_group.request_id,
            len(reservation.block_mapping),
            reservation.num_reused_blocks,
        )
        return True

    def abort_seq_group(
        self,
        request_id: str | Iterable[str],
        seq_id_to_seq_group=None,
    ) -> None:
        """处理 transfer 中请求的 abort，并延迟 block 释放到 I/O 结束。

        原生 Scheduler 只扫描 waiting/running/swapped 三个队列；loading
        请求不在这些队列中，因此需要在调用父类逻辑后单独检查。这里不
        尝试伪造 GranuleKV cancel 协议，而是让 resident GranuleKV 完成对应 slot 的 I/O，
        再由 ``complete_ready_async_kv_transfers`` 执行最终清理。
        """
        if isinstance(request_id, str):
            request_ids = {request_id}
        else:
            request_ids = set(request_id)
        seq_id_to_seq_group = seq_id_to_seq_group or {}

        # saving 请求虽然状态为 SWAPPED，但正式 block table 仍指向 GranuleKV
        # 正在读取的 GPU source。先从原生 swapped 队列摘出目标请求，避免
        # 父类 abort 提前 free_seq，造成 DMA 写盘期间地址被复用。
        for seq_group in tuple(self.saving.values()):
            real_request_id = seq_group.request_id
            if seq_group.request_id in seq_id_to_seq_group:
                real_request_id = seq_id_to_seq_group[
                    seq_group.request_id].group_id
            if real_request_id in request_ids:
                self._remove_from_swapped(seq_group)

        super().abort_seq_group(request_ids, seq_id_to_seq_group)
        # 层级 restore 的 group 已从 waiting 摘出，但在 full restore 前也未放
        # 入 running，因此父类看不到它。这里只标记父事务取消；所有 window
        # 仍正常走到 READY/ERROR，最后一个终态才会触发安全 abort reservation。
        for plan_id, seq_group in tuple(
                self.hierarchical_prefix_loading.items()):
            real_request_id = seq_group.request_id
            if seq_group.request_id in seq_id_to_seq_group:
                real_request_id = seq_id_to_seq_group[
                    seq_group.request_id].group_id
            if real_request_id not in request_ids:
                continue
            self.hierarchical_prefix_restores.cancel_plan(plan_id)
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.FINISHED_ABORTED
            # 未激活 unit 没有 DMA，可以立即在 Scheduler 标记 ERROR，并让
            # Worker 丢弃 staged descriptor。已激活 unit 仍自然完成，父事务
            # 等全部 unit 终态后才释放目标 block。
            queued = self.async_kv_policy.requests_for_plan(
                plan_id, state=AsyncKVTransferState.QUEUED)
            for request in queued:
                self.async_kv_policy.fail_queued(
                    request.request_id, "prefetch plan was cancelled")
                if plan_id in self._hierarchical_rolling_staged_plans:
                    self._staged_async_kv_discards.append(request.request_id)

        if self._staged_async_kv_discards:
            self.complete_ready_async_kv_transfers()

        # READY 但尚未进入执行 batch 的请求可能在这里被取消。删除纯观测
        # marker，避免一次永远不会发生的 first_execute 长期占用记录。
        for seq_group_id in tuple(self._async_kv_execution_markers):
            real_request_id = seq_group_id
            if seq_group_id in seq_id_to_seq_group:
                real_request_id = seq_id_to_seq_group[seq_group_id].group_id
            if real_request_id in request_ids:
                del self._async_kv_execution_markers[seq_group_id]

        prefix_transfers = (tuple(self.prefix_loading.items())
                            + tuple(self.prefix_saving.items()))
        for async_request_id, seq_group in (tuple(self.loading.items()) + tuple(
                self.saving.items()) + prefix_transfers):
            real_request_id = seq_group.request_id
            if seq_group.request_id in seq_id_to_seq_group:
                real_request_id = seq_id_to_seq_group[
                    seq_group.request_id].group_id
            if real_request_id not in request_ids:
                continue

            self._cancelled_async_kv_requests.add(async_request_id)
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.FINISHED_ABORTED

    def drain_async_kv_transfers_to_submit(
            self) -> Tuple[AsyncKVTransferRequest, ...]:
        """生成本轮 submit/stage 批次。

        rolling plan 的首个 unit 占用真实 GranuleKV slot；其余 unit 只把完整
        mapping 预授权给 Worker，``activate_on_submit=False``，由 barrier
        按模型进度稍后激活。普通 swap 和 rolling 关闭时仍走原队列逻辑。
        """
        if not self.hierarchical_io_config.rolling.enabled:
            return self.async_kv_policy.activate_next()

        requests: list[AsyncKVTransferRequest] = []
        rolling_plan_ids = set(self.hierarchical_prefix_loading)
        for plan_id in tuple(rolling_plan_ids):
            if plan_id in self._hierarchical_rolling_staged_plans:
                continue
            initial = self.async_kv_policy.activate_plan(
                plan_id, self.hierarchical_io_config.rolling.initial_units)
            # 没拿到完整首批 slot 时暂不 stage。否则 Worker 只有未来模板，
            # 首窗却仍是 QUEUED，首窗 READY admission 永远无法发生。
            if len(initial) < self.hierarchical_io_config.rolling.initial_units:
                continue
            requests.extend(initial)
            requests.extend(self.async_kv_policy.stage_plan(plan_id))
            self._hierarchical_rolling_staged_plans.add(plan_id)

        # 其他普通 swap 仍可使用空余 request slot，但不碰 rolling plan 的
        # queued unit；后续 unit 的激活权只在 Worker barrier。
        requests.extend(self.async_kv_policy.activate_next(
            excluded_plan_ids=tuple(rolling_plan_ids)))
        return tuple(requests)

    def drain_staged_async_kv_discards(self) -> Tuple[str, ...]:
        """返回需要 Worker 删除、且确认从未激活的 descriptor templates。"""
        request_ids = tuple(self._staged_async_kv_discards)
        self._staged_async_kv_discards.clear()
        return request_ids

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        """把原生同步 swap-out 改为 reservation + 后台 GranuleKV write。

        ``blocks_to_swap_out`` 故意保持为空，防止 Worker.execute_worker
        再执行一次同步写。源 GPU block 在 write READY 前仍由 reservation
        持有，因此本轮调度不会错误地把相同地址分给其他 sequence。
        """
        if not self.block_manager.can_reserve_swap_out(seq_group):
            raise RuntimeError(
                "Aborted due to the lack of storage swap space. Please "
                "increase the swap space to avoid this error.")
        reservation = self.block_manager.reserve_swap_out(seq_group)
        request = self._enqueue_async_kv_transfer(
            seq_group, reservation, AsyncKVTransferOperation.WRITE)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
        logger.debug(
            "[ASYNC_KV_SCHEDULER] queued operation=write request_id=%s "
            "seq_group=%s dirty_blocks=%d reused_clean_blocks=%d",
            request.request_id,
            seq_group.request_id,
            len(reservation.block_mapping),
            reservation.num_reused_blocks,
        )
        if envs.VLLM_V0_SWAP_TRACE and reservation.num_reused_blocks:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Scheduler] phase=avoid_write "
                "request_id=%s seq_group_id=%s clean_blocks=%d "
                "dirty_blocks=%d",
                request.request_id,
                seq_group.request_id,
                reservation.num_reused_blocks,
                len(reservation.block_mapping),
            )

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """提交一个异步 swap-in，但不把它加入当前计算 batch。

        原生实现会在 ``block_manager.swap_in`` 后立即把请求标记为
        RUNNING，并在当前 SchedulerOutputs 中执行。这里拆成：

        1. 检查 GPU block 是否足够；
        2. 预留目标 GPU block，正式 block table 仍指向 storage；
        3. 将逻辑请求登记为 loading；
        4. 等待 Worker/GranuleKV 返回 READY；
        5. 下一轮才进入 running。

        read/write 可以提前排队，并由统一 policy 按 GranuleKV slot 容量激活。
        """
        empty = SchedulerSwappedInOutputs.create_empty()
        # read reservation 会占用真实 GPU target，因此最多保留与后端 slot
        # 数相同的 loading 请求。write 继续留在 saving/swapped；从其后
        # 找到真正位于 storage 的 seq，避免对尚未写完的 GPU source 读回。
        if (len(self.loading) >= self.async_kv_policy.max_in_flight
                or not self.swapped):
            return empty

        saving_group_ids = {id(group) for group in self.saving.values()}
        seq_group = next((group for group in self.swapped
                          if id(group) not in saving_group_ids), None)
        if seq_group is None:
            return empty
        is_prefill = seq_group.is_prefill()
        alloc_status = self.block_manager.can_swap_in(
            seq_group,
            self._get_num_lookahead_slots(is_prefill, enable_chunking),
        )
        if alloc_status == AllocStatus.LATER:
            return empty
        if alloc_status == AllocStatus.NEVER:
            logger.warning(
                "Failing the request %s because there is not enough KV "
                "cache space for async swap-in.",
                seq_group.request_id,
            )
            self.swapped.remove(seq_group)
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.FINISHED_IGNORED
            empty.infeasible_seq_groups.append(seq_group)
            return empty

        # 异步恢复本身不消耗当前 forward 的 token budget；真正进入 running
        # 后，父类会在下一轮按普通 decode/prefill 规则更新 budget。
        self.swapped.remove(seq_group)
        reservation = self.block_manager.reserve_swap_in(seq_group)
        request = self._enqueue_async_kv_transfer(
            seq_group, reservation, AsyncKVTransferOperation.READ)
        logger.debug(
            "[ASYNC_KV_SCHEDULER] queued operation=read request_id=%s "
            "seq_group=%s blocks=%d",
            request.request_id,
            seq_group.request_id,
            len(reservation.block_mapping),
        )
        return empty
