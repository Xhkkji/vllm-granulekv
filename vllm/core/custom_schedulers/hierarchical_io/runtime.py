# SPDX-License-Identifier: Apache-2.0

"""Worker-local prefetch plan runtime。

Scheduler 预留 block 并一次下发完整 plan；本模块只决定已经预授权的 unit
何时占用 GranuleKV slot。它不能分配/释放 block，也不能发布 prefix hash。这样即使
激活发生在 model forward 内，事务所有权仍然完整留在 Scheduler。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import (Any, Callable, Dict, Iterable, Mapping, Optional, Sequence,
                    Tuple)

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferRequest, AsyncKVTransferState)

from .plan import RollingPrefetchConfig
from .residency import PrefetchResidencyDirectory


SubmitCallback = Callable[[AsyncKVTransferRequest, Any], AsyncKVTransferEvent]
PollCallback = Callable[[AsyncKVTransferRequest], AsyncKVTransferEvent]


@dataclass(frozen=True)
class PrefetchRuntimeTrace:
    """一条 worker 物理时间线事件，不使用 scheduler publish 时间。"""

    phase: str
    plan_id: str
    request_id: str
    unit_index: int
    monotonic_ns: int
    layer_index: Optional[int] = None
    wait_ns: int = 0
    lead_units: int = 0


@dataclass
class _UnitState:
    request: AsyncKVTransferRequest
    mapping: Any
    active: bool = False
    terminal: Optional[AsyncKVTransferEvent] = None
    activated_monotonic_ns: Optional[int] = None
    ready_monotonic_ns: Optional[int] = None
    terminal_reported: bool = False
    consumed: bool = False


@dataclass
class _PlanState:
    plan_id: str
    units: Dict[int, _UnitState]
    lead_units: int


class RollingPrefetchRuntime:
    """实现 stage -> activate -> poll -> wait_ready 的最小通用协议。

    后续 unit 首次由 Worker 激活时，会先向 Scheduler 回传一个 PENDING 事件，
    再回传 READY/ERROR。Scheduler 因而能把自己的 QUEUED 状态安全推进到
    PENDING，但不参与 forward 热路径中的激活时机。
    """

    def __init__(self, config: RollingPrefetchConfig) -> None:
        self.config = config
        self._plans: Dict[Tuple[int, str], _PlanState] = {}
        self._events: list[AsyncKVTransferEvent] = []
        self._traces: list[PrefetchRuntimeTrace] = []
        self.residency = PrefetchResidencyDirectory()

    def submit_or_stage(
        self,
        virtual_engine: int,
        request: AsyncKVTransferRequest,
        mapping: Any,
        submit: SubmitCallback,
    ) -> Tuple[AsyncKVTransferEvent, ...]:
        """登记 unit；首批 unit 立即 submit，其余只保存在 Worker。"""
        plan_id, unit_index = self._identity(request)
        plan_key = (virtual_engine, plan_id)
        plan = self._plans.setdefault(
            plan_key,
            _PlanState(plan_id=plan_id,
                       units={},
                       lead_units=self.config.lead_units))
        if unit_index in plan.units:
            raise RuntimeError(
                f"duplicate prefetch unit: {plan_id}/{unit_index}")
        unit = _UnitState(request=request, mapping=mapping)
        plan.units[unit_index] = unit
        self.residency.register(request)
        if not request.activate_on_submit:
            return ()

        event = self._activate(unit, submit, report_activation=False,
                               layer_index=None, lead_units=plan.lead_units)
        # 初始 unit 已由 Scheduler 在发送 RPC 前标记为 PENDING，因此这里直接
        # 返回底层 submit 事件，沿用现有 Engine apply_event 路径。
        if event.state != AsyncKVTransferState.PENDING:
            unit.terminal_reported = True
        return (event, )

    def poll_units(
        self,
        virtual_engine: int,
        poll: PollCallback,
    ) -> Tuple[AsyncKVTransferEvent, ...]:
        """非阻塞推进当前 engine 的 active units，并回收待上报事件。"""
        self._progress(virtual_engine, poll)
        events = tuple(self._events)
        self._events.clear()
        for plan_key, plan in tuple(self._plans.items()):
            if plan_key[0] != virtual_engine:
                continue
            request_ids = {event.request_id for event in events}
            for unit in plan.units.values():
                if (unit.terminal is not None
                        and unit.request.request_id in request_ids):
                    unit.terminal_reported = True
            # 外部 poll 只发生在 forward 边界。全部 terminal 且已经上报后即可
            # 删除 worker-local 模板；Scheduler 仍独立完成 commit/abort。
            if (plan.units and all(unit.terminal is not None
                                   and unit.terminal_reported
                                   for unit in plan.units.values())):
                # transfer 模板已经完成职责，但 long prompt 可能还会经历多个
                # chunked-prefill iteration。residency 必须保留到 request 真正
                # finished/abort，不能随 I/O completion 一起提前删除。
                del self._plans[plan_key]
        return events

    def wait_ready(
        self,
        virtual_engine: int,
        request_ids: Sequence[str],
        consumer_index: int,
        submit: SubmitCallback,
        poll: PollCallback,
        *,
        max_active: int,
        sleep_seconds: float = 0.0001,
    ) -> None:
        """滚动激活未来 unit，并等待当前 consumer 对应 unit READY。

        ``consumer_index`` 当前是 layer index。激活只查看 Scheduler 已经 stage
    的模板；若 GranuleKV slot 暂满，则先 poll 已激活 unit，槽位释放后再继续。
        """
        request_id_set = frozenset(request_ids)
        matching = tuple(
            plan for (engine, _), plan in self._plans.items()
            if engine == virtual_engine and any(
                unit.request.seq_group_id in request_id_set
                and unit.request.layer_range is not None
                and unit.request.layer_range[0] <= consumer_index <
                unit.request.layer_range[1] for unit in plan.units.values()))
        if not matching:
            return

        for plan in matching:
            current = next(
                unit for unit in plan.units.values()
                if unit.request.seq_group_id in request_id_set
                and unit.request.layer_range is not None
                and unit.request.layer_range[0] <= consumer_index <
                unit.request.layer_range[1])
            current_index = self._identity(current.request)[1]
            if current.consumed:
                raise RuntimeError(
                    "layer working-set KV was already evicted; this minimal "
                    "validation path only supports one model forward")
            self._activate_until(virtual_engine, plan, request_id_set,
                                 current_index + plan.lead_units, consumer_index,
                                 submit, poll, max_active, sleep_seconds)
            wait_started_ns = time.monotonic_ns()
            while current.terminal is None:
                if not current.active:
                    self._activate_until(virtual_engine, plan, request_id_set,
                                         current_index, consumer_index, submit,
                                         poll, max_active, sleep_seconds)
                    if not current.active and current.terminal is None:
                        time.sleep(sleep_seconds)
                        continue
                self._progress(virtual_engine, poll)
                if current.terminal is None:
                    time.sleep(sleep_seconds)

            wait_ns = time.monotonic_ns() - wait_started_ns
            if current.terminal.state == AsyncKVTransferState.ERROR:
                raise RuntimeError(
                    "prefetch unit failed before consumer "
                    f"{consumer_index}: {current.terminal.error}")

            # 最小反馈策略：当前 unit 仍产生超过目标的等待，说明固定 lead
            # 不够；只扩大本 plan 的未来窗口，不修改全局 scheduler 优先级。
            target_ns = int(self.config.target_slack_ms * 1.0e6)
            if (target_ns > 0 and wait_ns > target_ns
                    and plan.lead_units < self.config.max_lead_units):
                plan.lead_units += 1
            self._traces.append(
                PrefetchRuntimeTrace(
                    phase="barrier_ready",
                    plan_id=plan.plan_id,
                    request_id=current.request.request_id,
                    unit_index=current_index,
                    monotonic_ns=time.monotonic_ns(),
                    layer_index=consumer_index,
                    wait_ns=wait_ns,
                    lead_units=plan.lead_units,
                ))
            self._activate_until(virtual_engine, plan, request_id_set,
                                 current_index + plan.lead_units, consumer_index,
                                 submit, poll, max_active, sleep_seconds)

    def release_layer(
        self,
        virtual_engine: int,
        request_ids: Sequence[str],
        layer_index: int,
    ) -> None:
        """在 window 最后一层完成后允许后续 prefetch 覆盖其 GPU region。

        SSD 上的 prefix 是干净副本，因此逐出不需要 writeback。这里只记录消费
        完成并建立防重用检查；真正释放 HBM 的动作是后续 unit 写入同一环形
        region。普通 layerwise 路径关闭 working-set 时保持原来的常驻语义。
        """
        if not self.config.working_set_enabled:
            return
        request_id_set = frozenset(request_ids)
        for (engine, plan_id), plan in self._plans.items():
            if engine != virtual_engine:
                continue
            for unit_index, unit in plan.units.items():
                request = unit.request
                if (request.seq_group_id not in request_id_set
                        or request.layer_range is None
                        or layer_index + 1 != request.layer_range[1]):
                    continue
                if unit.terminal is None or unit.terminal.state != (
                        AsyncKVTransferState.READY):
                    raise RuntimeError(
                        "cannot evict layer working-set unit before READY")
                if unit.consumed:
                    raise RuntimeError("layer working-set unit consumed twice")
                unit.consumed = True
                self._traces.append(
                    PrefetchRuntimeTrace(
                        phase="working_set_evicted",
                        plan_id=plan_id,
                        request_id=request.request_id,
                        unit_index=unit_index,
                        monotonic_ns=time.monotonic_ns(),
                        layer_index=layer_index,
                        lead_units=plan.lead_units,
                    ))

    def discard_units(self, virtual_engine: int,
                      request_ids: Iterable[str]) -> None:
        """丢弃从未激活的模板；active DMA 仍必须自然到达终态。"""
        discarded = frozenset(request_ids)
        for plan_key, plan in tuple(self._plans.items()):
            if plan_key[0] != virtual_engine:
                continue
            for unit_index, unit in tuple(plan.units.items()):
                if unit.request.request_id not in discarded:
                    continue
                if unit.active or unit.terminal is not None:
                    raise RuntimeError(
                        "cannot discard an active prefetch unit")
                self.residency.forget(unit.request.request_id)
                del plan.units[unit_index]
            if not plan.units:
                del self._plans[plan_key]

    def pop_traces(self) -> Tuple[PrefetchRuntimeTrace, ...]:
        traces = tuple(self._traces)
        self._traces.clear()
        return traces

    def require_resident_layer(
        self,
        request_ids: Sequence[str],
        layer_index: int,
        sequence_lengths_by_request: Optional[Mapping[str, int]] = None,
        block_size: Optional[int] = None,
    ) -> Optional[Tuple[int, ...]]:
        """供 attention/model runner 在消费 sparse KV 前执行一致性校验。"""
        return self.residency.require_layer(
            request_ids,
            layer_index,
            sequence_lengths_by_request,
            block_size,
        )

    def residency_stats(self) -> Dict[str, int]:
        """Return physical residency miss counters for experiment reporting."""
        return self.residency.stats()

    def forget_seq_groups(self, seq_group_ids: Sequence[str]) -> None:
        """在 vLLM 通知 request finished/abort 时回收 residency 元数据。"""
        self.residency.forget_seq_groups(seq_group_ids)

    def _activate_until(
        self,
        virtual_engine: int,
        plan: _PlanState,
        request_ids: frozenset[str],
        last_unit_index: int,
        consumer_index: int,
        submit: SubmitCallback,
        poll: PollCallback,
        max_active: int,
        sleep_seconds: float,
    ) -> None:
        due = tuple(
            unit for index, unit in sorted(plan.units.items())
            if index <= last_unit_index
            and unit.request.seq_group_id in request_ids
            and not unit.active and unit.terminal is None)
        for unit in due:
            while self._active_count(virtual_engine) >= max_active:
                self._progress(virtual_engine, poll)
                if self._active_count(virtual_engine) >= max_active:
                    time.sleep(sleep_seconds)
            self._activate(unit, submit, report_activation=True,
                           layer_index=consumer_index,
                           lead_units=plan.lead_units)

    def _activate(
        self,
        unit: _UnitState,
        submit: SubmitCallback,
        *,
        report_activation: bool,
        layer_index: Optional[int],
        lead_units: int,
    ) -> AsyncKVTransferEvent:
        unit.active = True
        self.residency.mark_pending(unit.request.request_id)
        unit.activated_monotonic_ns = time.monotonic_ns()
        if report_activation:
            self._events.append(
                AsyncKVTransferEvent(unit.request.request_id,
                                     AsyncKVTransferState.PENDING))
        event = submit(unit.request, unit.mapping)
        plan_id, unit_index = self._identity(unit.request)
        self._traces.append(
            PrefetchRuntimeTrace(
                phase="activated",
                plan_id=plan_id,
                request_id=unit.request.request_id,
                unit_index=unit_index,
                monotonic_ns=unit.activated_monotonic_ns,
                layer_index=layer_index,
                lead_units=lead_units,
            ))
        if event.state != AsyncKVTransferState.PENDING:
            self._finish(unit, event)
            if report_activation:
                self._events.append(event)
        return event

    def _progress(self, virtual_engine: int, poll: PollCallback) -> None:
        for (engine, _), plan in tuple(self._plans.items()):
            if engine != virtual_engine:
                continue
            for unit in plan.units.values():
                if not unit.active or unit.terminal is not None:
                    continue
                event = poll(unit.request)
                if event.state == AsyncKVTransferState.PENDING:
                    continue
                self._finish(unit, event)
                self._events.append(event)

    def _finish(self, unit: _UnitState,
                event: AsyncKVTransferEvent) -> None:
        unit.active = False
        unit.terminal = event
        unit.ready_monotonic_ns = time.monotonic_ns()
        if event.state == AsyncKVTransferState.READY:
            self.residency.mark_ready(unit.request.request_id)
        else:
            self.residency.mark_error(unit.request.request_id)
        plan_id, unit_index = self._identity(unit.request)
        self._traces.append(
            PrefetchRuntimeTrace(
                phase=("physical_ready" if event.state ==
                       AsyncKVTransferState.READY else "physical_error"),
                plan_id=plan_id,
                request_id=unit.request.request_id,
                unit_index=unit_index,
                monotonic_ns=unit.ready_monotonic_ns,
            ))

    def _active_count(self, virtual_engine: int) -> int:
        return sum(
            1 for (engine, _), plan in self._plans.items()
            if engine == virtual_engine for unit in plan.units.values()
            if unit.active and unit.terminal is None)

    @staticmethod
    def _identity(request: AsyncKVTransferRequest) -> Tuple[str, int]:
        if (request.prefetch_plan_id is None
                or request.prefetch_unit_index is None):
            raise ValueError("prefetch request has no plan/unit identity")
        return request.prefetch_plan_id, request.prefetch_unit_index
