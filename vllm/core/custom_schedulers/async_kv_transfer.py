# SPDX-License-Identifier: Apache-2.0

"""异步 KV read/write 共用的后端无关 transfer 状态机。

这个模块只描述 Scheduler 与 Worker/GranuleKV 之间的 I/O 控制面协议，不包含
prefill/decode admission、victim 选择或 block residency 策略。把它从
``vllm.core.scheduler_policy`` 拆出来，是为了让后续自定义调度器可以复用
同一套 transfer 生命周期，而不继续把新策略堆在 vLLM 原生 scheduler 顶层。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Dict, Optional, Sequence, Tuple

from vllm.core.block_reservation import BlockMapping, LogicalBlockKey


class AsyncKVTransferOperation(Enum):
    """一次 block transfer 的数据方向。"""

    READ = "read"
    WRITE = "write"


class AsyncKVTransferPriority(Enum):
    """后端无关的最小 I/O 优先级。

    restore read 位于请求恢复关键路径；write 已经拥有 GPU source，可以
留在队列中等待。应用层以后可以改变入队顺序，但不需要修改 Worker/GranuleKV。
    """

    CRITICAL_READ = 0
    DEFERRED_WRITE = 1


class AsyncKVTransferState(Enum):
    """Scheduler 观察到的异步 transfer 生命周期。"""

    QUEUED = auto()
    PENDING = auto()
    READY = auto()
    ERROR = auto()


@dataclass(frozen=True)
class AsyncKVTransferRequest:
    """Scheduler 发送给 Worker 的后端无关 block transfer 描述。"""

    request_id: str
    seq_group_id: str
    reservation_id: str
    operation: AsyncKVTransferOperation
    block_mapping: BlockMapping
    logical_blocks: Tuple[LogicalBlockKey, ...]
    priority: AsyncKVTransferPriority
    # ``None`` 保持原有“搬完整 block 的全部层”语义。层级 restore 才携带
    # 左闭右开的本地 layer range；Worker/GranuleKV 不解释 scheduler plan。
    layer_range: Optional[Tuple[int, int]] = None
    # plan/unit identity 只描述预授权关系。普通 swap 为 None；layer-window
    # restore 使用它在 Worker 中一次 stage 全计划、随后按模型进度激活 unit。
    prefetch_plan_id: Optional[str] = None
    prefetch_unit_index: Optional[int] = None
    # sparse consumer 的 logical block 集合。它和 block_mapping 不同：前者
    # 是完整 prefix 的逻辑下标，后者只包含本次 SSD -> GPU 的物理映射。
    consumer_block_indices: Optional[Tuple[int, ...]] = None
    consumer_blocks_by_layer: Optional[
        Tuple[Optional[Tuple[int, ...]], ...]] = None
    # 完整 prefix 的 logical block 数，用于 residency 区分“已在 HBM”与
    # “本次从 SSD 恢复”的集合。None 表示普通 transfer 不启用 consumer。
    consumer_num_blocks: Optional[int] = None
    # Prefix restore 后，当前 logical index 及其后的 block 属于 live
    # request，而不是 SSD mapping。Worker 会按当前 decode 长度动态扩展它。
    consumer_local_block_start: Optional[int] = None
    # Stable CPU page-index identity.  It is control-plane metadata only and
    # never reaches the GranuleKV/native descriptor.
    sparse_page_index_key: Optional[str] = None
    # False 表示这次 RPC 只把 descriptor template 放进 Worker，不占用 GranuleKV
    # request slot。真正激活时 Worker 会回传 PENDING，再由 Scheduler 更新状态。
    activate_on_submit: bool = True

    def __post_init__(self) -> None:
        if (self.consumer_num_blocks is not None
                and self.consumer_num_blocks <= 0):
            raise ValueError("consumer_num_blocks must be positive")
        if (self.consumer_block_indices is not None
                and any(left >= right for left, right in zip(
                    self.consumer_block_indices,
                    self.consumer_block_indices[1:]))):
            raise ValueError(
                "consumer block indices must be strictly increasing")
        if (self.consumer_local_block_start is not None
                and self.consumer_local_block_start < 0):
            raise ValueError("consumer_local_block_start must be non-negative")


@dataclass(frozen=True)
class AsyncKVTransferEvent:
    """Worker 返回给 Scheduler 的非阻塞完成事件。"""

    request_id: str
    state: AsyncKVTransferState
    error: Optional[str] = None


@dataclass(frozen=True)
class AsyncKVExecutionMarker:
    """记录 read READY 请求第一次真正进入模型 batch 的时间。"""

    request_id: str
    seq_group_id: str
    promoted_monotonic_ns: int


@dataclass
class PendingAsyncKVTransfer:
    """状态机内部保存的一次 transfer。"""

    request: AsyncKVTransferRequest
    state: AsyncKVTransferState = AsyncKVTransferState.QUEUED
    error: Optional[str] = None


class AsyncKVTransferQueue:
    """管理 queued -> pending -> ready/error 的轻量多槽状态机。

    Scheduler 可以提前建立多个 block reservation；``activate_next`` 只按
    ``max_in_flight`` 激活后端能够容纳的请求。所有 slot 共用一套状态表，
    completion 可以乱序返回，request identity 不依赖提交顺序。
    """

    def __init__(self, max_in_flight: int = 1) -> None:
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self.max_in_flight = max_in_flight
        self._request_counter = itertools.count()
        self._transfers: Dict[str, PendingAsyncKVTransfer] = {}

    @property
    def queued_request_ids(self) -> Tuple[str, ...]:
        return self._ids_in_state(AsyncKVTransferState.QUEUED)

    @property
    def pending_request_ids(self) -> Tuple[str, ...]:
        return self._ids_in_state(AsyncKVTransferState.PENDING)

    @property
    def ready_request_ids(self) -> Tuple[str, ...]:
        return self._ids_in_state(AsyncKVTransferState.READY)

    @property
    def in_flight_count(self) -> int:
        return len(self.pending_request_ids)

    @property
    def available_capacity(self) -> int:
        """返回当前还可提交给后端的 logical transfer slot 数。"""
        return self.max_in_flight - self.in_flight_count

    @property
    def has_outstanding(self) -> bool:
        return bool(self._transfers)

    def enqueue(
        self,
        seq_group_id: str,
        reservation_id: str,
        operation: AsyncKVTransferOperation,
        block_mapping: Sequence[Tuple[int, int]],
        logical_blocks: Sequence[LogicalBlockKey] = (),
        priority: Optional[AsyncKVTransferPriority] = None,
        layer_range: Optional[Tuple[int, int]] = None,
        prefetch_plan_id: Optional[str] = None,
        prefetch_unit_index: Optional[int] = None,
        consumer_block_indices: Optional[Sequence[int]] = None,
        consumer_blocks_by_layer: Optional[
            Sequence[Optional[Sequence[int]]]] = None,
        consumer_num_blocks: Optional[int] = None,
        consumer_local_block_start: Optional[int] = None,
        sparse_page_index_key: Optional[str] = None,
    ) -> AsyncKVTransferRequest:
        """登记 reservation，但暂不占用 Worker/GranuleKV request slot。"""
        if priority is None:
            priority = (AsyncKVTransferPriority.CRITICAL_READ
                        if operation == AsyncKVTransferOperation.READ else
                        AsyncKVTransferPriority.DEFERRED_WRITE)
        request_id = f"async-kv-{next(self._request_counter)}"
        request = AsyncKVTransferRequest(
            request_id=request_id,
            seq_group_id=seq_group_id,
            reservation_id=reservation_id,
            operation=operation,
            block_mapping=tuple(tuple(pair) for pair in block_mapping),
            logical_blocks=tuple(logical_blocks),
            priority=priority,
            layer_range=layer_range,
            prefetch_plan_id=prefetch_plan_id,
            prefetch_unit_index=prefetch_unit_index,
            consumer_block_indices=(None if consumer_block_indices is None else
                                    tuple(consumer_block_indices)),
            consumer_blocks_by_layer=(
                None if consumer_blocks_by_layer is None else tuple(
                    None if indices is None else tuple(indices)
                    for indices in consumer_blocks_by_layer)),
            consumer_num_blocks=consumer_num_blocks,
            consumer_local_block_start=consumer_local_block_start,
            sparse_page_index_key=sparse_page_index_key,
        )
        self._transfers[request_id] = PendingAsyncKVTransfer(request=request)
        return request

    def activate_next(
        self,
        *,
        excluded_plan_ids: Sequence[str] = (),
        limit: Optional[int] = None,
    ) -> Tuple[AsyncKVTransferRequest, ...]:
        """优先激活 critical read，再用剩余槽位处理 deferred write。

        已经 IN_FLIGHT 的 write 不会被伪取消；后到 read 只越过仍为 QUEUED
        的 write。这样不复制两套队列状态机，同时保持同优先级 FIFO。
        """
        capacity = self.available_capacity
        if limit is not None:
            if limit < 0:
                raise ValueError("activation limit must be non-negative")
            capacity = min(capacity, limit)
        if capacity <= 0:
            return ()
        excluded = frozenset(excluded_plan_ids)
        activated = []
        for priority in AsyncKVTransferPriority:
            for transfer in self._transfers.values():
                if (transfer.state != AsyncKVTransferState.QUEUED
                        or transfer.request.priority != priority
                        or transfer.request.prefetch_plan_id in excluded):
                    continue
                transfer.state = AsyncKVTransferState.PENDING
                activated.append(transfer.request)
                if len(activated) == capacity:
                    return tuple(activated)
        return tuple(activated)

    def activate_plan(self, plan_id: str, count: int) -> Tuple[
            AsyncKVTransferRequest, ...]:
        """激活一个 plan 的前 ``count`` 个 queued unit。"""
        if count < 0:
            raise ValueError("plan activation count must be non-negative")
        if self.available_capacity < count:
            return ()
        capacity = count
        activated = []
        for transfer in self._transfers.values():
            if (capacity == 0
                    or transfer.state != AsyncKVTransferState.QUEUED
                    or transfer.request.prefetch_plan_id != plan_id):
                continue
            transfer.state = AsyncKVTransferState.PENDING
            activated.append(transfer.request)
            capacity -= 1
        return tuple(activated)

    def stage_plan(self, plan_id: str) -> Tuple[AsyncKVTransferRequest, ...]:
        """返回尚未激活的 unit，并标记 RPC 只做 Worker 侧 stage。"""
        return tuple(
            replace(transfer.request, activate_on_submit=False)
            for transfer in self._transfers.values()
            if transfer.request.prefetch_plan_id == plan_id
            and transfer.state == AsyncKVTransferState.QUEUED)

    def requests_for_plan(
        self,
        plan_id: str,
        *,
        state: Optional[AsyncKVTransferState] = None,
    ) -> Tuple[AsyncKVTransferRequest, ...]:
        """按 unit 入队顺序返回一个预授权 plan 的请求。"""
        return tuple(
            transfer.request for transfer in self._transfers.values()
            if transfer.request.prefetch_plan_id == plan_id
            and (state is None or transfer.state == state))

    def apply_event(self, event: AsyncKVTransferEvent) -> None:
        """应用 active request 的 PENDING/READY/ERROR 事件。"""
        transfer = self._get_transfer(event.request_id)
        # rolling activation 发生在 model forward 内，Scheduler 下一次运行时
        # 才收到 Worker 的 PENDING 通知。因此只有带 plan identity 的 QUEUED
        # 请求允许由该事件切到 PENDING；普通 transfer 仍必须由 activate_next。
        if (transfer.state == AsyncKVTransferState.QUEUED
                and event.state == AsyncKVTransferState.PENDING
                and transfer.request.prefetch_plan_id is not None):
            transfer.state = AsyncKVTransferState.PENDING
            return
        if transfer.state != AsyncKVTransferState.PENDING:
            raise RuntimeError(
                f"async KV request is not active: {event.request_id}")
        if event.state == AsyncKVTransferState.PENDING:
            return
        if event.state not in (AsyncKVTransferState.READY,
                               AsyncKVTransferState.ERROR):
            raise ValueError(f"unsupported async KV event: {event.state}")
        transfer.state = event.state
        transfer.error = event.error

    def pop_ready(self) -> Tuple[AsyncKVTransferRequest, ...]:
        """按入队顺序消费已经成功完成的请求。"""
        ready = tuple(
            transfer.request for transfer in self._transfers.values()
            if transfer.state == AsyncKVTransferState.READY)
        for request in ready:
            del self._transfers[request.request_id]
        return ready

    def pop_errors(self) -> Tuple[PendingAsyncKVTransfer, ...]:
        """按入队顺序消费失败请求及其错误信息。"""
        errors = tuple(
            transfer for transfer in self._transfers.values()
            if transfer.state == AsyncKVTransferState.ERROR)
        for transfer in errors:
            del self._transfers[transfer.request.request_id]
        return errors

    def cancel_queued(self, request_id: str) -> AsyncKVTransferRequest:
        """取消尚未提交给 Worker 的请求；active I/O 不能在此处撤销。"""
        transfer = self._get_transfer(request_id)
        if transfer.state != AsyncKVTransferState.QUEUED:
            raise RuntimeError(
                f"cannot cancel active async KV request: {request_id}")
        del self._transfers[request_id]
        return transfer.request

    def fail_queued(self, request_id: str, error: str) -> None:
        """把从未提交的 unit 变成可由统一 error 路径消费的终态。"""
        transfer = self._get_transfer(request_id)
        if transfer.state != AsyncKVTransferState.QUEUED:
            raise RuntimeError(
                f"cannot fail active async KV request: {request_id}")
        transfer.state = AsyncKVTransferState.ERROR
        transfer.error = error

    def _ids_in_state(
        self,
        state: AsyncKVTransferState,
    ) -> Tuple[str, ...]:
        return tuple(
            request_id for request_id, transfer in self._transfers.items()
            if transfer.state == state)

    def _get_transfer(self, request_id: str) -> PendingAsyncKVTransfer:
        transfer = self._transfers.get(request_id)
        if transfer is None:
            raise KeyError(f"unknown async KV request: {request_id}")
        return transfer


# 旧类名保留为别名，避免当前 AsyncKVScheduler 和历史测试一次性大改。
AsyncKVSchedulePolicy = AsyncKVTransferQueue
