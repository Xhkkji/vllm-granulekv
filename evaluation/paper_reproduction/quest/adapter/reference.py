"""Reference-only Quest calculations and attention comparisons.

These functions are CPU/GPU PyTorch utilities for selector validation.  They
do not replace, or connect to, a vLLM attention backend.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .selector import _normalize_query, quest_page_scores


def _canonical_kv(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise ValueError("KV tensor must have shape [heads, tokens, dim]")
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
        raise ValueError("KV tensor dimensions must be positive")
    return tensor


def build_page_representatives(
    key_cache: torch.Tensor,
    page_size: int,
    query: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build ``[pages, heads, dim]`` page representatives.

    With a query, the representative is the exact query-conditioned maximum
    of ``key * sign(query)`` used by the official PyTorch reference.  Without
    a query, this returns a query-independent min/max envelope with shape
    ``[pages, 2, heads, dim]`` suitable for the CPU page index.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if not isinstance(key_cache, torch.Tensor) or key_cache.ndim != 3:
        raise ValueError("key_cache must have shape [heads, tokens, dim]")
    keys = _canonical_kv(key_cache)
    if query is not None:
        normalized_query = _normalize_query(query)
        if normalized_query.shape[1] != keys.shape[2]:
            raise ValueError("query and key_cache dimension mismatch")
        if normalized_query.shape[0] % keys.shape[0] != 0:
            raise ValueError(
                "query heads must be a multiple of KV heads for GQA/MQA")
        keys = keys.repeat_interleave(normalized_query.shape[0] // keys.shape[0],
                                      dim=0)
        sign = torch.where(normalized_query > 0, 1, -1).to(keys.dtype)
        keys = keys * sign[:, None, :]
    pages = []
    for start in range(0, keys.shape[1], page_size):
        page = keys[:, start:start + page_size]
        if query is None:
            pages.append(torch.stack((page.amin(dim=1), page.amax(dim=1))))
        else:
            pages.append(page.amax(dim=1))
    return torch.stack(pages, dim=0)


def exact_page_scores(query: torch.Tensor, key_cache: torch.Tensor,
                      page_size: int) -> torch.Tensor:
    """Compute exact Quest page scores directly from raw page keys."""
    representatives = build_page_representatives(key_cache, page_size, query)
    return quest_page_scores(query, representatives)


def dense_attention(query: torch.Tensor, key_cache: torch.Tensor,
                    value_cache: torch.Tensor) -> torch.Tensor:
    """Compute one decode attention output per head using all tokens."""
    query = _normalize_query(query)
    keys = _canonical_kv(key_cache)
    values = _canonical_kv(value_cache)
    if keys.shape != values.shape or keys.shape[2] != query.shape[1]:
        raise ValueError("query, key_cache and value_cache shapes do not match")
    if query.shape[0] % keys.shape[0] != 0:
        raise ValueError(
            "query heads must be a multiple of KV heads for GQA/MQA")
    if query.shape[0] != keys.shape[0]:
        repeat = query.shape[0] // keys.shape[0]
        keys = keys.repeat_interleave(repeat, dim=0)
        values = values.repeat_interleave(repeat, dim=0)
    logits = torch.einsum("hd,htd->ht", query, keys) / query.shape[1]**0.5
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("ht,htd->hd", weights, values)


def selected_attention(query: torch.Tensor, key_cache: torch.Tensor,
                       value_cache: torch.Tensor, selected_blocks: Sequence[int],
                       page_size: int) -> torch.Tensor:
    """Compute attention after loading only the selected logical pages."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    keys = _canonical_kv(key_cache)
    values = _canonical_kv(value_cache)
    pages = tuple(sorted(set(int(block) for block in selected_blocks)))
    if not pages:
        raise ValueError("selected_blocks must not be empty")
    if any(block < 0 or block * page_size >= keys.shape[1] for block in pages):
        raise ValueError("selected block is outside key cache")
    token_indices = [index for block in pages for index in range(
        block * page_size, min((block + 1) * page_size, keys.shape[1]))]
    index_tensor = torch.tensor(token_indices, device=keys.device)
    return dense_attention(query, keys.index_select(1, index_tensor),
                           values.index_select(1, index_tensor))
