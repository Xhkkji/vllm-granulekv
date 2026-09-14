"""Metrics for Quest selector-only experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from .reference import dense_attention, exact_page_scores, selected_attention
from .selector import union_topk_pages


@dataclass(frozen=True)
class QuestSelectionMetrics:
    total_blocks: int
    selected_blocks: int
    selected_ratio: float
    block_budget: int
    exact_blocks: int
    selector_recall: float
    attention_output_error: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _relative_error(reference: torch.Tensor, actual: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(actual - reference) / denominator).item())


def evaluate_selection(query: torch.Tensor, key_cache: torch.Tensor,
                       value_cache: torch.Tensor, page_size: int,
                       selected_blocks: Sequence[int],
                       block_budget: int) -> QuestSelectionMetrics:
    """Compare a selected-page output with exact dense attention."""
    scores = exact_page_scores(query, key_cache, page_size)
    exact_blocks_tuple = union_topk_pages(scores, block_budget)
    selected_tuple = tuple(sorted(set(int(block) for block in selected_blocks)))
    exact_set = set(exact_blocks_tuple)
    selected_set = set(selected_tuple)
    recall = (len(selected_set & exact_set) / len(exact_set)
              if exact_set else 1.0)
    dense = dense_attention(query, key_cache, value_cache)
    sparse = selected_attention(query, key_cache, value_cache, selected_tuple,
                                page_size)
    total_blocks = (key_cache.shape[1] + page_size - 1) // page_size
    return QuestSelectionMetrics(
        total_blocks=total_blocks,
        selected_blocks=len(selected_tuple),
        selected_ratio=len(selected_tuple) / max(1, total_blocks),
        block_budget=block_budget,
        exact_blocks=len(exact_set),
        selector_recall=recall,
        attention_output_error=_relative_error(dense, sparse),
    )


def attention_output_error(reference: torch.Tensor,
                           actual: torch.Tensor) -> float:
    """Public relative L2 error helper for standalone tests."""
    if reference.shape != actual.shape:
        raise ValueError("attention outputs must have the same shape")
    return _relative_error(reference, actual)
