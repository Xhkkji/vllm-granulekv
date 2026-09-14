"""CPU-resident Quest page representative index."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence

import torch


class QuestPageIndex:
    """Keep page metadata independent from GPU KV-cache lifetime."""

    def __init__(self) -> None:
        self._representatives: Dict[str, Dict[int, Dict[int, torch.Tensor]]] = (
            defaultdict(dict))

    def register(self, request_id: str, layer_index: int,
                 page_representatives: torch.Tensor,
                 logical_block_indices: Optional[Sequence[int]] = None) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if not isinstance(page_representatives, torch.Tensor):
            raise TypeError("page_representatives must be a tensor")
        if page_representatives.ndim not in (3, 4) or page_representatives.shape[
                0] <= 0:
            raise ValueError(
                "page_representatives must have shape [pages, heads, dim] or "
                "[pages, 2, heads, dim]")
        if (page_representatives.ndim == 4
                and page_representatives.shape[1] != 2):
            raise ValueError("page envelope axis must contain min and max")
        if logical_block_indices is None:
            logical_indices = tuple(range(page_representatives.shape[0]))
        else:
            logical_indices = tuple(int(index) for index in logical_block_indices)
            if len(logical_indices) != page_representatives.shape[0]:
                raise ValueError("logical_block_indices must match page count")
            if any(index < 0 for index in logical_indices):
                raise ValueError("logical block indices must be non-negative")
            if len(set(logical_indices)) != len(logical_indices):
                raise ValueError("logical block indices must be unique")
        # Page metadata is deliberately copied to CPU so eviction cannot make
        # the selector depend on a released GPU KV tensor.
        layer_pages = self._representatives[request_id].setdefault(
            layer_index, {})
        for row, logical_index in zip(page_representatives, logical_indices):
            layer_pages[logical_index] = row.detach().to(device="cpu").clone()

    def get(self, request_id: str,
            layer_index: int) -> torch.Tensor:
        pages_by_layer = self._representatives.get(request_id)
        pages = None if pages_by_layer is None else pages_by_layer.get(
            layer_index)
        if not pages:
            raise KeyError(
                f"no Quest page representatives for {request_id}/layer "
                f"{layer_index}")
        max_index = max(pages)
        missing = [index for index in range(max_index + 1)
                   if index not in pages]
        if missing:
            raise KeyError(
                f"missing Quest page representatives at logical blocks "
                f"{missing}")
        return torch.stack([pages[index] for index in range(max_index + 1)])

    def discard(self, request_id: str) -> None:
        self._representatives.pop(request_id, None)

    def __contains__(self, key: tuple[str, int]) -> bool:
        request_id, layer_index = key
        return layer_index in self._representatives.get(request_id, {})
