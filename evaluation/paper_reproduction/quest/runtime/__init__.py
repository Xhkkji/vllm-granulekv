"""Lazy SolidAttention CUDA runtime.

The extension is loaded only when the explicit runtime experiment switch is
enabled.  Ordinary Quest/SolidAttention imports therefore do not trigger a
compiler or change the existing attention path.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import torch


_EXTENSION: Any = None


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    try:
        _EXTENSION = importlib.import_module("._solidattention_runtime",
                                             __name__)
        return _EXTENSION
    except ImportError:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SolidAttention CUDA runtime is unavailable without CUDA")
        from torch.utils.cpp_extension import CUDA_HOME, load

        if CUDA_HOME is None:
            raise RuntimeError("nvcc is required to build SolidAttention runtime")
        source = Path(__file__).with_name("solidattention_runtime.cu")
        _EXTENSION = load(
            name="_solidattention_runtime",
            sources=[str(source)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=os.getenv("VLLM_GRANULEKV_SPARSE_RUNTIME_VERBOSE",
                             "0") == "1",
        )
        return _EXTENSION


def score_pages(query: torch.Tensor,
                representatives: torch.Tensor,
                output: torch.Tensor | None = None) -> torch.Tensor:
    """Compute SolidAttention page scores on the query's CUDA device."""
    if query.ndim == 3:
        if query.shape[0] != 1:
            raise ValueError("runtime query must be batch one")
        query = query[0]
    if output is None:
        output = torch.empty((query.shape[0], representatives.shape[0]),
                             dtype=torch.float32, device=query.device)
    _load_extension().score_pages(query.contiguous(),
                                  representatives.contiguous(), output)
    return output


def select_pages(scores: torch.Tensor, num_blocks: int, init_blocks: int,
                 local_blocks: int, block_budget: int,
                 output: torch.Tensor | None = None,
                 counts: torch.Tensor | None = None):
    """Merge fixed/local/suffix pages with per-head GPU top-k pages."""
    from vllm.attention.ops.sparse_kv import SparseKVHeadSelection

    prefix_count = max(0, int(num_blocks) - 1)
    init_count = min(int(init_blocks), prefix_count)
    local_start = max(init_count, prefix_count - int(local_blocks))
    dynamic_count = min(int(block_budget),
                        max(0, local_start - init_count))
    max_selected = (init_count + dynamic_count + prefix_count - local_start +
                    (int(num_blocks) - prefix_count))
    if output is None or output.shape != (scores.shape[0], max_selected):
        output = torch.empty((scores.shape[0], max_selected),
                             dtype=torch.int32, device=scores.device)
    if counts is None or counts.shape != (scores.shape[0],):
        counts = torch.empty((scores.shape[0],),
                             dtype=torch.int32, device=scores.device)
    _load_extension().select_pages(
        scores.contiguous(), output, counts, int(num_blocks),
        int(init_blocks), int(local_blocks), int(block_budget))
    return SparseKVHeadSelection(output, counts)


def decode(output: torch.Tensor, query: torch.Tensor, key_cache: torch.Tensor,
           value_cache: torch.Tensor, block_table: torch.Tensor,
           logical_indices: torch.Tensor, counts: torch.Tensor,
           sequence_length: int, block_size: int, scale: float,
           num_kv_heads: int) -> None:
    """Run direct per-head sparse decode over vLLM's native KV layout."""
    _load_extension().decode(output, query, key_cache, value_cache,
                             block_table, logical_indices, counts,
                             int(sequence_length), int(block_size),
                             float(scale), int(num_kv_heads))


__all__ = ["decode", "score_pages", "select_pages"]
