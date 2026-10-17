"""Independent PyTorch reference for SolidAttention block selection."""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def reference_solidattention_selection(
    query: torch.Tensor,
    representatives: torch.Tensor,
    prefix_blocks: int,
    block_budget: int,
    init_blocks: int = 2,
    local_blocks: int = 2,
    logical_block_indices: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    """Return the sorted init/local/per-head-top-k block union."""
    if query.ndim == 3:
        if query.shape[0] != 1:
            raise ValueError("reference selector only supports batch size one")
        query = query[0]
    if query.ndim != 2:
        raise ValueError("query must have shape [heads, head_dim]")
    if representatives.ndim not in (3, 4):
        raise ValueError("representatives must have three or four dimensions")
    if prefix_blocks < 0 or block_budget <= 0:
        raise ValueError("prefix_blocks and block_budget must be positive")
    logical = (tuple(range(representatives.shape[0]))
               if logical_block_indices is None else
               tuple(int(index) for index in logical_block_indices))
    if len(logical) != representatives.shape[0]:
        raise ValueError("logical indices must match representative count")
    if any(index < 0 for index in logical) or any(
            left >= right for left, right in zip(logical, logical[1:])):
        raise ValueError("logical indices must be sorted and unique")
    if any(index >= prefix_blocks for index in logical):
        raise ValueError("logical index is outside immutable prefix")

    init_end = min(init_blocks, prefix_blocks)
    local_start = max(init_end, prefix_blocks - local_blocks)
    fixed = set(range(init_end)) | set(range(local_start, prefix_blocks))
    candidate_positions = [
        position for position, logical_index in enumerate(logical)
        if init_end <= logical_index < local_start
    ]
    if not candidate_positions:
        return tuple(sorted(fixed))

    selected_representatives = representatives.index_select(
        0, torch.tensor(candidate_positions, device=representatives.device))
    if selected_representatives.ndim == 4:
        kv_heads = selected_representatives.shape[2]
        if query.shape[0] % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        grouped = query.reshape(kv_heads, query.shape[0] // kv_heads,
                                query.shape[-1])
        scores = (
            torch.einsum("kqd,pkd->kqp", grouped.clamp_min(0),
                         selected_representatives[:, 1])
            + torch.einsum("kqd,pkd->kqp", grouped.clamp_max(0),
                           selected_representatives[:, 0]))
        scores = scores.reshape(query.shape[0], len(candidate_positions))
    else:
        kv_heads = selected_representatives.shape[1]
        if query.shape[0] % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        grouped = query.reshape(kv_heads, query.shape[0] // kv_heads,
                                query.shape[-1]).abs()
        scores = torch.einsum("kqd,pkd->kqp", grouped,
                              selected_representatives).reshape(
                                  query.shape[0], len(candidate_positions))

    count = min(block_budget, len(candidate_positions))
    topk_positions = torch.argsort(scores,
                                   dim=-1,
                                   descending=True,
                                   stable=True)[:, :count]
    for position in torch.unique(topk_positions.reshape(-1), sorted=True):
        fixed.add(logical[candidate_positions[int(position)]])
    return tuple(sorted(fixed))
