"""Pure PyTorch Quest-style page selection.

The selector only returns logical page/block indices.  It deliberately does
not know about physical KV mappings, GranuleKV handles, or attention kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def _normalize_query(query: torch.Tensor) -> torch.Tensor:
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if query.ndim == 3:
        if query.shape[0] != 1:
            raise ValueError("Quest selector only supports batch size 1")
        query = query[0]
    if query.ndim != 2 or query.shape[0] <= 0 or query.shape[1] <= 0:
        raise ValueError("query must have shape [heads, head_dim]")
    return query


def _normalize_representatives(
    page_representatives: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if not isinstance(page_representatives, torch.Tensor):
        raise TypeError("page_representatives must be a torch.Tensor")
    if page_representatives.ndim not in (3, 4):
        raise ValueError(
            "page_representatives must be query-conditioned [pages, heads, "
            "dim] or a min/max envelope [pages, 2, heads, dim]")
    if page_representatives.shape[-1] <= 0:
        raise ValueError("page representatives must have a non-empty dimension")

    if page_representatives.ndim == 4:
        if page_representatives.shape[1] != 2:
            raise ValueError(
                "min/max page representatives require an envelope axis of 2")
        kv_heads = page_representatives.shape[2]
        if kv_heads <= 0 or num_heads % kv_heads != 0:
            raise ValueError("page representatives do not match query heads")
        if num_heads != kv_heads:
            page_representatives = page_representatives.repeat_interleave(
                num_heads // kv_heads, dim=2)
        if page_representatives.shape[0] <= 0:
            raise ValueError(
                "page representatives must contain at least one page")
        return page_representatives

    # The canonical layout is [pages, kv_heads, dim].  A query may have more
    # heads than the KV cache (GQA/MQA), so expand each KV head over its query
    # group before scoring.  Head-major input is accepted only when the
    # canonical axis cannot describe a valid KV-head grouping.
    if (page_representatives.shape[1] > 0
            and num_heads % page_representatives.shape[1] == 0):
        representatives = page_representatives
    elif (page_representatives.shape[0] > 0
          and num_heads % page_representatives.shape[0] == 0):
        representatives = page_representatives.transpose(0, 1)
    else:
        raise ValueError("page representatives do not match query heads")
    if num_heads != representatives.shape[1]:
        representatives = representatives.repeat_interleave(
            num_heads // representatives.shape[1], dim=1)
    if representatives.shape[0] <= 0:
        raise ValueError("page representatives must contain at least one page")
    return representatives


def quest_page_scores(
    query: torch.Tensor,
    page_representatives: torch.Tensor,
) -> torch.Tensor:
    """Return Quest's query-aware score for every head and page.

    A query-independent envelope has shape ``[pages, 2, kv_heads, dim]`` and
    stores per-page key minima/maxima.  The legacy 3-D form is a
    query-conditioned maximum after the query-sign transform.  GQA/MQA KV
    heads are expanded before scoring.  The return shape is ``[heads, pages]``.
    """
    query = _normalize_query(query)
    representatives = _normalize_representatives(page_representatives,
                                                  query.shape[0])
    if representatives.shape[-1] != query.shape[1]:
        raise ValueError("query and page representatives have different dimensions")
    if representatives.ndim == 4:
        lower = representatives[:, 0]
        upper = representatives[:, 1]
        chosen = torch.where(query.unsqueeze(0) >= 0, upper, lower)
        return torch.einsum("hd,phd->hp", query, chosen)
    positive_query = query.abs()
    return torch.einsum("hd,phd->hp", positive_query, representatives)


def topk_pages_by_head(scores: torch.Tensor,
                       block_budget: int) -> torch.Tensor:
    """Return stable per-head top-k page indices with shape ``[heads, k]``."""
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError("scores must have shape [heads, pages]")
    if scores.shape[0] <= 0 or scores.shape[1] <= 0:
        raise ValueError("scores must contain at least one head and page")
    if block_budget <= 0:
        raise ValueError("block_budget must be positive")
    count = min(block_budget, scores.shape[1])
    # Stable sorting makes ties deterministic across repeated selector calls.
    ordering = torch.argsort(scores, dim=-1, descending=True, stable=True)
    return ordering[:, :count]


def union_topk_pages(scores: torch.Tensor,
                     block_budget: int) -> Tuple[int, ...]:
    """Select each head's top-k pages and return their sorted union."""
    return tuple(int(index) for index in union_topk_pages_tensor(
        scores, block_budget).tolist())


def union_topk_pages_tensor(scores: torch.Tensor,
                            block_budget: int) -> torch.Tensor:
    """Return the sorted union of per-head top-k pages on the input device."""
    topk = topk_pages_by_head(scores, block_budget)
    return torch.unique(topk.reshape(-1), sorted=True)


@dataclass(frozen=True)
class QuestPageSelection:
    """Inspectable result for a single query/page selection."""

    selected_blocks: Tuple[int, ...]
    selected_by_head: Tuple[Tuple[int, ...], ...]
    scores: torch.Tensor


class QuestPageSelector:
    """State-free query-aware selector used by the Quest adapter."""

    def select_from_query(self, query: torch.Tensor,
                          page_representatives: torch.Tensor,
                          block_budget: int) -> QuestPageSelection:
        scores = quest_page_scores(query, page_representatives)
        topk = topk_pages_by_head(scores, block_budget)
        selected_by_head = tuple(tuple(int(index) for index in row.tolist())
                                 for row in topk)
        return QuestPageSelection(
            selected_blocks=tuple(sorted({int(index)
                                          for index in topk.reshape(-1).tolist()})),
            selected_by_head=selected_by_head,
            scores=scores,
        )

    def select_tensor_from_query(self, query: torch.Tensor,
                                 page_representatives: torch.Tensor,
                                 block_budget: int) -> torch.Tensor:
        """Select pages without converting the result to Python objects."""
        return union_topk_pages_tensor(
            quest_page_scores(query, page_representatives), block_budget)
