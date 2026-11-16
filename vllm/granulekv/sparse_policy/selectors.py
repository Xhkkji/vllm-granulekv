"""Small selector primitives used by paper adapters.

The output is the production ``SparseKVAccessPlan``.  These selectors are
controlled approximations for mechanism validation, not claims of faithful
reimplementation of a paper's CUDA attention kernel.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from vllm.core.custom_schedulers.hierarchical_io.plan import (
    SparseKVAccessPlan, )


def _validate_selection(selection: Sequence[int], num_blocks: int) -> Tuple[int,
                                                                          ...]:
    result = tuple(selection)
    if not result:
        raise ValueError("selector must select at least one block")
    if any(index < 0 or index >= num_blocks for index in result):
        raise ValueError("selector returned a block outside the prefix")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError("selector output must be strictly increasing")
    return result


class DenseSelector:
    name = "dense"

    def select(self, num_layers: int,
               num_blocks: int) -> Optional[SparseKVAccessPlan]:
        if num_layers <= 0 or num_blocks <= 0:
            raise ValueError("dense selector requires positive dimensions")
        return None


class ExplicitSelector:
    """Apply an already computed per-layer block selection."""

    def __init__(self, selections: Sequence[Sequence[int]], name: str) -> None:
        self.name = name
        self._selections = tuple(tuple(selection) for selection in selections)

    def select(self, num_layers: int,
               num_blocks: int) -> SparseKVAccessPlan:
        if len(self._selections) != num_layers:
            raise ValueError("one selection is required for every layer")
        selections = tuple(_validate_selection(selection, num_blocks)
                           for selection in self._selections)
        return SparseKVAccessPlan(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_indices_by_layer=selections,
            source=self.name,
        )


class TailSelector:
    """Deterministic sparse proxy for a recent/local block policy."""

    def __init__(self, block_budget: int, name: str) -> None:
        if block_budget <= 0:
            raise ValueError("block_budget must be positive")
        self.block_budget = block_budget
        self.name = name

    def select(self, num_layers: int,
               num_blocks: int) -> SparseKVAccessPlan:
        count = min(self.block_budget, num_blocks)
        selected = tuple(range(num_blocks - count, num_blocks))
        return SparseKVAccessPlan(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_indices_by_layer=(selected,) * num_layers,
            source=self.name,
        )
