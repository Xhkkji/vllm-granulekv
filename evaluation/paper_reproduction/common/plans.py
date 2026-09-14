"""Composition helpers for selector, prefetcher, and request scheduler.

This module deliberately wraps the production ``SparseKVAccessPlan`` and
``PrefetchPlan`` types.  It does not define a second transfer state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferPriority, )
from vllm.core.custom_schedulers.hierarchical_io.plan import (
    PrefetchBlockSelectorConfig, PrefetchPlan, SparseKVAccessPlan,
    build_layer_restore_plan, select_prefetch_unit_blocks, )


class Selector(Protocol):
    name: str

    def select(self, num_layers: int,
               num_blocks: int) -> Optional[SparseKVAccessPlan]:
        """Return a production access plan; ``None`` means dense."""


class Prefetcher(Protocol):
    name: str

    def build(self, plan_id: str, num_layers: int, num_blocks: int,
              access_plan: Optional[SparseKVAccessPlan]) -> PrefetchPlan:
        """Build a production prefetch plan without submitting I/O."""


class RequestScheduler(Protocol):
    name: str

    def order(self, items: Sequence["RequestWorkItem"]) -> Tuple[
            "RequestWorkItem", ...]:
        """Order request descriptions; it does not own transfer state."""


@dataclass(frozen=True)
class RequestWorkItem:
    """A scheduler-only description of one planned request."""

    request_id: str
    useful_bytes: int
    estimated_service_us: float = 0.0
    ready_deadline_us: float = 0.0
    priority: AsyncKVTransferPriority = AsyncKVTransferPriority.CRITICAL_READ


@dataclass(frozen=True)
class StrategyPlan:
    """One experiment plan composed from three orthogonal dimensions."""

    selector: str
    prefetcher: str
    scheduler: str
    prefetch_plan: PrefetchPlan
    request_order: Tuple[RequestWorkItem, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "prefetcher": self.prefetcher,
            "scheduler": self.scheduler,
            "prefetch_plan_id": self.prefetch_plan.plan_id,
            "block_selector": self.prefetch_plan.block_selector,
            "profiling_only": self.prefetch_plan.profiling_only,
            "consumer_enabled": self.prefetch_plan.consumer_enabled,
            "units": [{
                "index": unit.index,
                "layer_range": list(unit.layer_range),
                "block_indices": (None if unit.block_indices is None else
                                   list(unit.block_indices)),
            } for unit in self.prefetch_plan.units],
            "request_order": [item.request_id for item in self.request_order],
            "metadata": dict(self.metadata),
        }


def enqueue_plan_requests(
    queue: Any,
    strategy_plan: StrategyPlan,
    *,
    seq_group_id: str,
    reservation_id: str,
    block_mapping: Sequence[tuple[int, int]],
    logical_blocks: Sequence[Any],
    operation: Any,
) -> Tuple[Any, ...]:
    """Register plan units in the existing AsyncKVTransferQueue.

    This helper only projects block selections and registers requests. The
    caller remains responsible for stage/activate/query/complete/cancel.
    """
    requests = []
    for unit in strategy_plan.prefetch_plan.units:
        mapping, keys = select_prefetch_unit_blocks(
            unit, block_mapping, logical_blocks)
        requests.append(queue.enqueue(
            seq_group_id=seq_group_id,
            reservation_id=reservation_id,
            operation=operation,
            block_mapping=mapping,
            logical_blocks=keys,
            layer_range=unit.layer_range,
            prefetch_plan_id=strategy_plan.prefetch_plan.plan_id,
            prefetch_unit_index=unit.index,
            consumer_blocks_by_layer=unit.consumer_blocks_by_layer,
            consumer_num_blocks=(
                None if strategy_plan.prefetch_plan.access_plan is None else
                strategy_plan.prefetch_plan.access_plan.num_blocks),
        ))
    return tuple(requests)


class LayerWisePrefetcher:
    """Use the production layer-window planner for every strategy."""

    name = "layer_wise"

    def __init__(self, window_layers: int) -> None:
        if window_layers <= 0:
            raise ValueError("window_layers must be positive")
        self.window_layers = window_layers

    def build(self, plan_id: str, num_layers: int, num_blocks: int,
              access_plan: Optional[SparseKVAccessPlan]) -> PrefetchPlan:
        selector = PrefetchBlockSelectorConfig(policy="dense")
        return build_layer_restore_plan(
            plan_id=plan_id,
            num_layers=num_layers,
            num_blocks=num_blocks,
            window_layers=self.window_layers,
            block_selector=selector,
            access_plan=access_plan,
        )


class ImmediatePrefetcher(LayerWisePrefetcher):
    """A one-window plan representing synchronous/on-demand consumption."""

    name = "on_demand"

    def __init__(self) -> None:
        super().__init__(window_layers=1)


def compose_plan(*,
                 plan_id: str,
                 selector: Selector,
                 prefetcher: Prefetcher,
                 scheduler: RequestScheduler,
                 num_layers: int,
                 num_blocks: int,
                 request_items: Sequence[RequestWorkItem] = (),
                 metadata: Optional[Mapping[str, Any]] = None,
                 ) -> StrategyPlan:
    access_plan = selector.select(num_layers, num_blocks)
    prefetch_plan = prefetcher.build(plan_id, num_layers, num_blocks,
                                     access_plan)
    return StrategyPlan(
        selector=selector.name,
        prefetcher=prefetcher.name,
        scheduler=scheduler.name,
        prefetch_plan=prefetch_plan,
        request_order=scheduler.order(request_items),
        metadata={} if metadata is None else dict(metadata),
    )
