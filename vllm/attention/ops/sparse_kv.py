# SPDX-License-Identifier: Apache-2.0
"""Generic sparse KV metadata helpers.

This is user code added for the GranuleKV sparse-attention experiments.  It is
not part of the upstream vLLM attention implementation and deliberately does
not own allocator or KV-cache state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch


_SPARSE_STATS: dict[str, int] = {
    "calls": 0,
    "full_blocks": 0,
    "selected_blocks": 0,
    "selected_tokens": 0,
    "metadata_build_calls": 0,
    "metadata_build_blocks": 0,
    "gpu_selection_calls": 0,
    "attention_selection_calls": 0,
    "attention_selection_refreshes": 0,
    "attention_selection_cache_hits": 0,
    "attention_selected_blocks": 0,
}


def sparse_kv_stats() -> dict[str, float | int]:
    """Return process-local sparse consumer counters."""
    payload: dict[str, float | int] = dict(_SPARSE_STATS)
    payload["selected_ratio"] = payload["selected_blocks"] / max(
        1, payload["full_blocks"])
    return payload


def reset_sparse_kv_stats() -> None:
    """Reset counters after benchmark warmup."""
    for name in _SPARSE_STATS:
        _SPARSE_STATS[name] = 0


def record_sparse_kv_selection(refresh: bool) -> None:
    """Record whether an attention selection refreshed or reused its cache."""
    if not refresh:
        _SPARSE_STATS["attention_selection_cache_hits"] += 1
        return
    _SPARSE_STATS["attention_selection_refreshes"] += 1


def record_sparse_kv_attention(full_blocks: int, selected_blocks: int,
                               selected_tokens: int) -> None:
    """Record one selected-list attention call without changing selection."""
    if full_blocks <= 0 or selected_blocks <= 0 or selected_tokens <= 0:
        raise ValueError("sparse attention statistics must be positive")
    _SPARSE_STATS["calls"] += 1
    _SPARSE_STATS["full_blocks"] += full_blocks
    _SPARSE_STATS["selected_blocks"] += selected_blocks
    _SPARSE_STATS["selected_tokens"] += selected_tokens


@dataclass(frozen=True)
class SparseKVSelection:
    """Logical KV blocks selected for one decode attention call."""

    logical_block_indices: Tuple[int, ...]
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not self.logical_block_indices:
            raise ValueError("sparse KV selection must not be empty")
        if any(index < 0 for index in self.logical_block_indices):
            raise ValueError("sparse KV logical block index must be non-negative")
        if any(left >= right for left, right in zip(
                self.logical_block_indices, self.logical_block_indices[1:])):
            raise ValueError("sparse KV logical blocks must be sorted and unique")


@dataclass(frozen=True)
class SparseKVDeviceSelection:
    """GPU-resident logical block list for a selected-list attention kernel.

    Only the first ``count`` entries of ``logical_block_indices`` are valid.
    The count is taken from tensor shape metadata by the producer, so consuming
    code never needs to read a CUDA scalar back to the host.
    """

    logical_block_indices: torch.Tensor
    count: int


def validate_sparse_kv_device_selection(
        selection: SparseKVDeviceSelection) -> SparseKVDeviceSelection:
    """Validate only metadata that is available without synchronizing CUDA."""
    if not isinstance(selection, SparseKVDeviceSelection):
        raise TypeError("attention selector must return SparseKVDeviceSelection")
    indices = selection.logical_block_indices
    if not isinstance(indices, torch.Tensor) or not indices.is_cuda:
        raise ValueError("selected logical blocks must be a CUDA tensor")
    if indices.ndim != 1:
        raise ValueError("selected logical blocks must be one-dimensional")
    if indices.dtype != torch.int32:
        raise ValueError("selected logical blocks must have dtype int32")
    if not indices.is_contiguous():
        raise ValueError("selected logical blocks must be contiguous")
    if not isinstance(selection.count, int) or isinstance(selection.count, bool):
        raise TypeError("selected logical block count must be a Python int")
    if selection.count <= 0 or selection.count > indices.numel():
        raise ValueError("selected logical block count is outside the buffer")
    _SPARSE_STATS["attention_selection_calls"] += 1
    _SPARSE_STATS["attention_selected_blocks"] += selection.count
    return selection


def _normalize_selection(selected_logical_blocks: Sequence[int],
                         num_logical_blocks: int) -> Tuple[int, ...]:
    selected = tuple(sorted({int(index) for index in selected_logical_blocks}))
    if not selected:
        raise ValueError("selected_logical_blocks must not be empty")
    if selected[-1] >= num_logical_blocks:
        raise IndexError(
            "selected logical block is outside the sequence block table")
    return selected


def build_selected_decode_block_table(
    block_table: torch.Tensor,
    selected_logical_blocks: Sequence[int] | torch.Tensor,
    sequence_length: int,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    """Compact a batch-one physical block table for sparse decode.

    ``selected_logical_blocks`` indexes the original logical sequence order;
    ``block_table`` contains physical ids.  The returned table keeps the
    selected physical ids in logical order, so the existing paged-attention
    kernel can consume it without changing allocator state.  The second
    result is the number of valid KV tokens represented by the compact table.
    """
    if not isinstance(block_table, torch.Tensor):
        raise TypeError("block_table must be a torch.Tensor")
    if block_table.ndim not in (1, 2):
        raise ValueError("block_table must have shape [blocks] or [1, blocks]")
    if block_table.ndim == 2 and block_table.shape[0] != 1:
        raise ValueError("sparse decode currently supports batch size 1")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    num_logical_blocks = (sequence_length + block_size - 1) // block_size
    if block_table.shape[-1] < num_logical_blocks:
        raise ValueError("block_table is shorter than sequence_length")
    if isinstance(selected_logical_blocks, torch.Tensor):
        if selected_logical_blocks.ndim != 1:
            raise ValueError("selected logical blocks must be one-dimensional")
        if selected_logical_blocks.numel() == 0:
            raise ValueError("selected_logical_blocks must not be empty")
        indices = torch.unique(
            selected_logical_blocks.to(device=block_table.device,
                                       dtype=torch.long),
            sorted=True,
        )
        # GPU Quest selection always retains the newest block.  This computes
        # the compact sequence length without reading a GPU scalar on host.
        tail_tokens = sequence_length - (num_logical_blocks - 1) * block_size
        selected_tokens = indices.numel() * block_size - (block_size - tail_tokens)
    else:
        selected = _normalize_selection(selected_logical_blocks,
                                        num_logical_blocks)
        selected_tokens = 0
        last_logical_block = num_logical_blocks - 1
        for logical_index in selected:
            selected_tokens += (sequence_length - logical_index * block_size
                                if logical_index == last_logical_block else
                                block_size)
        indices = torch.tensor(selected,
                               dtype=torch.long,
                               device=block_table.device)
    compact = block_table.index_select(-1, indices)
    return compact, selected_tokens


def build_page_representatives_from_paged_key_cache(
    key_cache: torch.Tensor,
    physical_block_ids: Sequence[int] | torch.Tensor,
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a query-independent min/max envelope from paged keys.

    The helper understands the layouts returned by
    ``PagedAttention.split_kv_cache`` for regular (non-FP8) KV caches.  It
    returns ``[pages, 2, kv_heads, head_dim]``, where axis 1 contains the
    per-dimension minimum and maximum.  By default the result is copied to
    CPU; ``output_device`` keeps the result on a requested device for a fast
    selector.  The caller decides which logical block ids those pages
    represent and when to publish them.
    """
    if not isinstance(key_cache, torch.Tensor) or key_cache.ndim not in (4, 5):
        raise ValueError(
            "split paged key cache must have shape [blocks, heads, tokens, dim] "
            "or [blocks, heads, dim_chunks, tokens, chunk]")
    if isinstance(physical_block_ids, torch.Tensor):
        if physical_block_ids.ndim != 1 or physical_block_ids.numel() == 0:
            raise ValueError("physical_block_ids must be a non-empty vector")
        block_ids = physical_block_ids.to(device=key_cache.device,
                                          dtype=torch.long)
    else:
        block_ids = torch.tensor(tuple(int(block_id)
                                      for block_id in physical_block_ids),
                                 dtype=torch.long,
                                 device=key_cache.device)
    if block_ids.numel() == 0:
        raise ValueError("physical_block_ids must not be empty")
    if not block_ids.is_cuda:
        if any(block_id < 0 or block_id >= key_cache.shape[0]
               for block_id in block_ids.tolist()):
            raise IndexError("physical KV block id is outside key cache")

    selected = key_cache.index_select(0, block_ids)
    if selected.ndim == 5:
        # [pages, heads, dim_chunks, block_size, chunk] ->
        # [pages, heads, block_size, head_dim]
        selected = selected.permute(0, 1, 3, 2, 4).flatten(3)
    lower = selected.detach().amin(dim=2)
    upper = selected.detach().amax(dim=2)
    _SPARSE_STATS["metadata_build_calls"] += 1
    _SPARSE_STATS["metadata_build_blocks"] += int(block_ids.numel())
    result = torch.stack((lower, upper), dim=1)
    return (result.to(device="cpu") if output_device is None else
            result.to(device=output_device))


def select_and_compact_decode_blocks(
    *,
    block_table: torch.Tensor,
    key_cache: torch.Tensor,
    query: torch.Tensor,
    sequence_length: int,
    block_size: int,
    request_id: str,
    layer_index: int,
    block_budget: int,
    select_blocks: Callable[..., Tuple[int, ...]],
    register_page_representatives: Callable[..., None],
    select_blocks_device: Optional[Callable[..., torch.Tensor]] = None,
    resident_blocks: Optional[Sequence[int]] = None,
    selected_blocks_override: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, int, Tuple[int, ...] | torch.Tensor]:
    """Register Quest-style metadata, select blocks, and compact one table.

    This is the common resident/sparse consumer path.  It has no ownership of
    allocator state or I/O handles; callers provide the policy callbacks.
    ``resident_blocks`` is optional because resident-only mode treats the
    complete current vLLM table as available, while hierarchical mode passes
    the blocks confirmed by its layer barrier.
    """
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError("sparse decode requires a batch-one block table")
    num_blocks = (sequence_length + block_size - 1) // block_size
    if num_blocks <= 0 or block_table.shape[1] < num_blocks:
        raise ValueError("block table is shorter than sequence_length")
    if selected_blocks_override is None:
        if select_blocks_device is not None:
            selected = select_blocks_device(
                request_id,
                layer_index,
                query,
                key_cache,
                block_table[0, :num_blocks],
                sequence_length,
                block_size,
                block_budget,
            )
            if not isinstance(selected, torch.Tensor):
                raise TypeError("GPU sparse selector must return a tensor")
            _SPARSE_STATS["gpu_selection_calls"] += 1
        else:
            physical_ids = tuple(int(value) for value in
                                block_table[0, :num_blocks].detach().cpu().tolist())
            representatives = build_page_representatives_from_paged_key_cache(
                key_cache, physical_ids)
            register_page_representatives(
                request_id,
                layer_index,
                representatives,
                tuple(range(num_blocks)),
            )
            selected = tuple(select_blocks(
                request_id,
                layer_index,
                query.detach().to(device="cpu"),
                torch.empty(0),
                num_blocks,
                block_budget,
            ))
    else:
        # A dynamic restore plan is already the frozen result of policy
        # selection. Recomputing it in a new request context can select blocks
        # that the current restore deliberately did not load.
        selected = tuple(int(index) for index in selected_blocks_override)
    if isinstance(selected, torch.Tensor):
        if selected.ndim != 1 or selected.numel() == 0:
            raise ValueError("GPU sparse selection must be a non-empty vector")
        resident = (tuple(range(num_blocks)) if resident_blocks is None else
                    tuple(int(index) for index in resident_blocks))
        if resident != tuple(range(num_blocks)):
            raise RuntimeError(
                "GPU sparse selection requires all current blocks resident")
    else:
        selected = _normalize_selection(selected, num_blocks)
        resident = (tuple(range(num_blocks)) if resident_blocks is None else
                    tuple(int(index) for index in resident_blocks))
        missing = tuple(sorted(set(selected).difference(resident)))
        if missing:
            raise RuntimeError(
                "sparse KV policy selected non-resident blocks: "
                f"layer={layer_index} missing={missing}")
    _SPARSE_STATS["calls"] += 1
    _SPARSE_STATS["full_blocks"] += num_blocks
    selected_count = (selected.numel() if isinstance(selected, torch.Tensor)
                      else len(selected))
    _SPARSE_STATS["selected_blocks"] += selected_count
    if isinstance(selected, torch.Tensor):
        tail_tokens = sequence_length - (num_blocks - 1) * block_size
        selected_tokens = selected_count * block_size - (block_size - tail_tokens)
    else:
        selected_tokens = sum(
            sequence_length - index * block_size if index == num_blocks - 1
            else block_size for index in selected)
    _SPARSE_STATS["selected_tokens"] += selected_tokens
    compact, selected_tokens = build_selected_decode_block_table(
        block_table, selected, sequence_length, block_size)
    return compact, selected_tokens, selected
