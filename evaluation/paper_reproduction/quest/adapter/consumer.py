"""Restricted Quest decode consumer for selector validation.

This is a reference consumer, not a vLLM attention backend.  It closes the
selector-to-consumer loop for batch-size-one decode while keeping physical KV
mapping and GranuleKV ownership outside the experiment directory.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from vllm.core.custom_schedulers.hierarchical_io.barrier import (
    get_active_sparse_kv_blocks, )
from vllm.core.custom_schedulers.hierarchical_io.plan import (
    SparseKVAccessPlan, )

from .reference import selected_attention


def _decode_query(query: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if query.ndim == 3:
        if query.shape[0] != 1 or query.shape[1] <= 0:
            raise ValueError("decode query must have shape [1, heads, dim]")
        return query[0], True
    if query.ndim != 2 or query.shape[0] <= 0 or query.shape[1] <= 0:
        raise ValueError("decode query must have shape [heads, dim]")
    return query, False


class QuestDecodeOnlyConsumer:
    """Use one sparse plan for a batch-size-one decode reference forward.

    The consumer does not select blocks and does not submit I/O.  The caller
    supplies the active window blocks, or installs them with the shared
    ``activate_sparse_kv_blocks`` context used by the model-side bridge.
    """

    def __init__(self, access_plan: SparseKVAccessPlan, page_size: int) -> None:
        if not isinstance(access_plan, SparseKVAccessPlan):
            raise TypeError("access_plan must be a SparseKVAccessPlan")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if access_plan.is_dense:
            raise ValueError("Quest decode consumer requires a sparse access plan")
        self.access_plan = access_plan
        self.page_size = page_size

    def blocks_for_layer(self, layer_index: int) -> Tuple[int, ...]:
        selected = self.access_plan.blocks_for_layer(layer_index)
        if selected is None:
            raise RuntimeError("Quest decode consumer cannot consume a dense layer")
        return selected

    def _validate_active(
        self,
        layer_index: int,
        active_blocks: Optional[Sequence[int]],
    ) -> Tuple[int, ...]:
        selected = self.blocks_for_layer(layer_index)
        if active_blocks is None:
            raise RuntimeError(
                "no active sparse KV blocks were exposed for decode layer")
        active = frozenset(int(index) for index in active_blocks)
        if not set(selected).issubset(active):
            missing = tuple(sorted(set(selected).difference(active)))
            raise RuntimeError(
                f"selected Quest blocks are not resident for layer "
                f"{layer_index}: missing={missing}")
        return selected

    def attend(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer_index: int,
        *,
        active_blocks: Optional[Sequence[int]],
    ) -> torch.Tensor:
        """Compute selected-page attention after validating residency.

        Input caches use ``[heads, tokens, head_dim]``.  Batch size one may be
        supplied as ``[1, heads, head_dim]`` for the query; the output keeps
        that batch dimension when it was present.
        """
        normalized_query, had_batch = _decode_query(query)
        selected = self._validate_active(layer_index, active_blocks)
        output = selected_attention(normalized_query, key_cache, value_cache,
                                    selected, self.page_size)
        return output.unsqueeze(0) if had_batch else output

    def attend_from_context(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        """Consume the block set installed by the shared layer bridge."""
        return self.attend(query, key_cache, value_cache, layer_index,
                           active_blocks=get_active_sparse_kv_blocks())

