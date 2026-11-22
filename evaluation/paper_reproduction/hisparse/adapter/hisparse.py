"""HiSparse-style plan adapter.

Only block hit/miss selection is represented here. Residency updates and
correction reads must use the existing request lifecycle at integration time.
"""

from __future__ import annotations

from typing import Sequence

from vllm.granulekv.sparse_policy.selectors import ExplicitSelector, TailSelector


class HiSparseSelector:
    name = "hisparse"

    def __init__(self, block_budget: int = 0,
                 selections: Sequence[Sequence[int]] = ()) -> None:
        self._delegate = (ExplicitSelector(selections, self.name)
                          if selections else TailSelector(block_budget,
                                                          "hisparse_proxy"))

    def select(self, num_layers: int, num_blocks: int):
        return self._delegate.select(num_layers, num_blocks)
