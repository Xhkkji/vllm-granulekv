"""Quest adapter with an explicit boundary around the selector only."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ...common.selectors import ExplicitSelector, TailSelector
from vllm.core.custom_schedulers.hierarchical_io.plan import SparseKVAccessPlan

from .selector import QuestPageSelector


class QuestSelector:
    """Translate Quest block choices into the shared sparse access plan.

    ``selected_blocks_by_layer`` is the preferred input from a Quest
    implementation. The deterministic tail proxy exists only for a smoke test
    and is labeled as an approximation in the resulting source field.
    """

    name = "quest"

    def __init__(self,
                 block_budget: int = 0,
                 selected_blocks_by_layer: Optional[
                     Sequence[Sequence[int]]] = None,
                 page_size: int = 1) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self._dynamic = QuestPageSelector()
        self._dynamic_selections: dict[int, tuple[int, ...]] = {}
        self._selected = (None if selected_blocks_by_layer is None else
                          ExplicitSelector(selected_blocks_by_layer, self.name))
        self._proxy = (None if block_budget <= 0 else
                       TailSelector(block_budget, "quest_proxy"))
        # A selector without static choices is a valid dynamic-only adapter.
        # Calling the legacy ``select`` method in that mode gives a targeted
        # error while ``select_from_query`` remains available.

    def select(self, num_layers: int, num_blocks: int):
        if self._selected is not None:
            return self._selected.select(num_layers, num_blocks)
        if self._proxy is None:
            raise ValueError(
                "QuestSelector has no static choices; use select_from_query "
                "or build_access_plan_from_queries")
        return self._proxy.select(num_layers, num_blocks)

    def select_from_query(
        self,
        query: torch.Tensor,
        page_representatives: torch.Tensor,
        block_budget: Optional[int] = None,
        layer_index: int = 0,
    ) -> tuple[int, ...]:
        """Select logical pages for one layer and remember the layer result."""
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        budget = self._proxy.block_budget if block_budget is None and self._proxy is not None else block_budget
        if budget is None or budget <= 0:
            raise ValueError("a positive block_budget is required")
        selection = self._dynamic.select_from_query(query, page_representatives,
                                                    budget).selected_blocks
        self._dynamic_selections[layer_index] = selection
        return selection

    def build_access_plan_from_queries(
        self,
        queries_by_layer: Sequence[torch.Tensor],
        page_representatives_by_layer: Sequence[torch.Tensor],
        num_blocks: int,
        block_budget: Optional[int] = None,
    ) -> SparseKVAccessPlan:
        """Build the shared per-layer access plan from dynamic selections."""
        if len(queries_by_layer) != len(page_representatives_by_layer):
            raise ValueError("one query and representative tensor is required per layer")
        if not queries_by_layer:
            raise ValueError("at least one layer is required")
        selections = tuple(
            self.select_from_query(query, representatives, block_budget, layer)
            for layer, (query, representatives) in enumerate(
                zip(queries_by_layer, page_representatives_by_layer)))
        if any(block >= num_blocks for selection in selections for block in selection):
            raise ValueError("selected page is outside num_blocks")
        return SparseKVAccessPlan(
            num_layers=len(selections),
            num_blocks=num_blocks,
            block_indices_by_layer=selections,
            source="quest_query_aware",
        )
