# SPDX-License-Identifier: Apache-2.0

"""层级 restore 父事务与 prefetch unit 子请求的生命周期。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .plan import PrefetchPlan, PrefetchUnit


@dataclass(frozen=True)
class HierarchicalRestoreProgress:
    """一个子请求到达终态后，父 restore 的完整状态快照。"""

    plan_id: str
    unit: PrefetchUnit
    first_unit_became_ready: bool
    first_unit_ready: bool
    all_terminal: bool
    all_ready: bool
    failed: bool
    cancelled: bool
    first_unit_ready_monotonic_ns: Optional[int]
    plan_created_monotonic_ns: int
    profiling_only: bool
    block_selector: str
    consumer_enabled: bool


@dataclass
class _RestoreState:
    plan: PrefetchPlan
    request_ids: Tuple[str, ...]
    terminal_request_ids: set[str]
    ready_request_ids: set[str]
    failed: bool = False
    cancelled: bool = False
    first_unit_ready_monotonic_ns: Optional[int] = None


class HierarchicalRestoreController:
    """聚合多个 unit completion，但不管理 block/sequence 所有权。

    一个父 reservation 会产生多个 GranuleKV 子请求。只有全部子请求成功后，
    scheduler 才能发布 block hash；任一子请求失败或用户取消时，也必须等
    所有 active DMA 到达终态后才能 abort target。这个类集中维护该屏障，
    避免错误处理逻辑散落在 scheduler 的 ready/error 两条分支中。
    """

    def __init__(self) -> None:
        self._states: Dict[str, _RestoreState] = {}
        self._request_to_plan: Dict[str, Tuple[str, int]] = {}

    def register(self, plan: PrefetchPlan,
                 request_ids: Sequence[str]) -> None:
        ids = tuple(request_ids)
        if plan.plan_id in self._states:
            raise ValueError(f"duplicate hierarchical plan: {plan.plan_id}")
        if len(ids) != len(plan.units) or len(set(ids)) != len(ids):
            raise ValueError("one unique request_id is required per unit")
        if any(request_id in self._request_to_plan for request_id in ids):
            raise ValueError("hierarchical request_id is already registered")

        self._states[plan.plan_id] = _RestoreState(
            plan=plan,
            request_ids=ids,
            terminal_request_ids=set(),
            ready_request_ids=set(),
        )
        for index, request_id in enumerate(ids):
            self._request_to_plan[request_id] = (plan.plan_id, index)

    def contains_request(self, request_id: str) -> bool:
        return request_id in self._request_to_plan

    def plan_id_for_request(self, request_id: str) -> str:
        return self._request_to_plan[request_id][0]

    def mark_ready(
        self,
        request_id: str,
        now_monotonic_ns: Optional[int] = None,
    ) -> HierarchicalRestoreProgress:
        state, unit = self._mark_terminal(request_id)
        state.ready_request_ids.add(request_id)
        became_ready = False
        if (unit.index == 0
                and state.first_unit_ready_monotonic_ns is None):
            state.first_unit_ready_monotonic_ns = (
                time.monotonic_ns()
                if now_monotonic_ns is None else now_monotonic_ns)
            became_ready = True
        return self._snapshot(state, unit, became_ready)

    def mark_error(self, request_id: str) -> HierarchicalRestoreProgress:
        state, unit = self._mark_terminal(request_id)
        state.failed = True
        return self._snapshot(state, unit, False)

    def cancel_plan(self, plan_id: str) -> None:
        self._states[plan_id].cancelled = True

    def release(self, plan_id: str) -> PrefetchPlan:
        """在全部子请求终态后移除父事务，并清理反向索引。"""
        state = self._states[plan_id]
        if len(state.terminal_request_ids) != len(state.request_ids):
            raise RuntimeError("cannot release active hierarchical restore")
        del self._states[plan_id]
        for request_id in state.request_ids:
            del self._request_to_plan[request_id]
        return state.plan

    def _mark_terminal(self,
                       request_id: str) -> Tuple[_RestoreState, PrefetchUnit]:
        plan_id, unit_index = self._request_to_plan[request_id]
        state = self._states[plan_id]
        if request_id in state.terminal_request_ids:
            raise RuntimeError(
                f"hierarchical request is already terminal: {request_id}")
        state.terminal_request_ids.add(request_id)
        return state, state.plan.units[unit_index]

    @staticmethod
    def _snapshot(
        state: _RestoreState,
        unit: PrefetchUnit,
        first_unit_became_ready: bool,
    ) -> HierarchicalRestoreProgress:
        all_terminal = (len(state.terminal_request_ids)
                        == len(state.request_ids))
        all_ready = (all_terminal and not state.failed and
                     len(state.ready_request_ids) == len(state.request_ids))
        return HierarchicalRestoreProgress(
            plan_id=state.plan.plan_id,
            unit=unit,
            first_unit_became_ready=first_unit_became_ready,
            first_unit_ready=(state.first_unit_ready_monotonic_ns is not None),
            all_terminal=all_terminal,
            all_ready=all_ready,
            failed=state.failed,
            cancelled=state.cancelled,
            first_unit_ready_monotonic_ns=(
                state.first_unit_ready_monotonic_ns),
            plan_created_monotonic_ns=state.plan.created_monotonic_ns,
            profiling_only=state.plan.profiling_only,
            block_selector=state.plan.block_selector,
            consumer_enabled=state.plan.consumer_enabled,
        )
