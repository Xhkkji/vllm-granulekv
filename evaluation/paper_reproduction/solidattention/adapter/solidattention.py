"""SolidAttention-style block selection on the shared sparse boundary."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

import vllm.envs as envs
from vllm.attention.ops.sparse_kv import (
    SparseKVDeviceSelection,
    build_page_representatives_from_paged_key_cache, )
from vllm.attention.ops.sparse_kv import record_sparse_kv_selection

from ...common.selectors import ExplicitSelector, TailSelector


class SolidAttentionSelector:
    name = "solidattention"

    def __init__(self, block_budget: int = 0,
                 selections: Optional[Sequence[Sequence[int]]] = None) -> None:
        self._delegate = (ExplicitSelector(selections, self.name)
                          if selections is not None else
                          TailSelector(block_budget, "solidattention_proxy"))

    def select(self, num_layers: int, num_blocks: int):
        return self._delegate.select(num_layers, num_blocks)


@dataclass
class _AttentionSelectionState:
    buffer: torch.Tensor
    fixed_blocks: torch.Tensor
    candidate_blocks: torch.Tensor
    prefix_count: int
    num_blocks: int
    count: int
    mapping_token: int
    ready: bool


@dataclass(frozen=True)
class _PredictionRepresentativeState:
    logical_block_indices: Tuple[int, ...]
    representatives: torch.Tensor


class SolidAttentionPolicy:
    """Minimal SolidAttention-style policy for the shared sparse bridge.

    The policy keeps fixed initial/local blocks and selects the remaining
    immutable prefix blocks with a GPU query/page score.  It owns only policy
    metadata; physical mappings, I/O requests and allocator state stay in the
    existing vLLM/GranuleKV integration.
    """

    name = "solidattention"

    def __init__(self, init_blocks: int = 2, local_blocks: int = 2) -> None:
        if init_blocks < 0 or local_blocks < 0:
            raise ValueError("init_blocks and local_blocks must be non-negative")
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self._previous_queries: Dict[str, Dict[int, torch.Tensor]] = defaultdict(
            dict)
        self._representatives: Dict[
            str, Dict[int, Tuple[int, torch.Tensor]]] = defaultdict(dict)
        # Prediction metadata is keyed by the stable prefix identity.  The
        # resident-only attention path uses the request-keyed GPU cache above;
        # it never consults this control-plane state.
        self._page_index_keys: Dict[str, str] = {}
        self._prediction_representatives: Dict[
            str, Dict[int, _PredictionRepresentativeState]] = defaultdict(dict)
        self._seen: set[tuple[str, int]] = set()
        self._attention_selection_buffers: Dict[
            str, Dict[int, _AttentionSelectionState]] = defaultdict(dict)

    @staticmethod
    def _normalize_query(query: torch.Tensor) -> torch.Tensor:
        if query.ndim == 3:
            if query.shape[0] != 1:
                raise ValueError("SolidAttention only supports batch size 1")
            query = query[0]
        if query.ndim != 2:
            raise ValueError("query must have shape [heads, head_dim]")
        return query

    @staticmethod
    def _score_pages(query: torch.Tensor,
                     representatives: torch.Tensor) -> torch.Tensor:
        """Score pages using the representative envelope without host copies."""
        query = SolidAttentionPolicy._normalize_query(query)
        if representatives.ndim == 4:
            if representatives.shape[1] != 2:
                raise ValueError("representative envelope must have axis 2")
            kv_heads = representatives.shape[2]
            if query.shape[0] % kv_heads != 0:
                raise ValueError("representatives do not match query heads")
            lower, upper = representatives[:, 0], representatives[:, 1]
            queries_per_kv = query.shape[0] // kv_heads
            grouped_query = query.reshape(kv_heads, queries_per_kv,
                                          query.shape[-1])
            positive = grouped_query.clamp_min(0)
            negative = grouped_query.clamp_max(0)
            scores = (torch.einsum("kqd,pkd->kqp", positive, upper) +
                      torch.einsum("kqd,pkd->kqp", negative, lower))
            return scores.reshape(query.shape[0], representatives.shape[0])
        if representatives.ndim != 3:
            raise ValueError("representatives must have shape [pages, heads, dim]")
        if representatives.shape[1] <= 0 or query.shape[0] % representatives.shape[1] != 0:
            raise ValueError("representatives do not match query heads")
        kv_heads = representatives.shape[1]
        queries_per_kv = query.shape[0] // kv_heads
        grouped_query = query.reshape(kv_heads, queries_per_kv,
                                      query.shape[-1]).abs()
        scores = torch.einsum("kqd,pkd->kqp", grouped_query, representatives)
        return scores.reshape(query.shape[0], representatives.shape[0])

    def _fixed_and_candidates(self, prefix_count: int,
                              device: torch.device) -> tuple[torch.Tensor,
                                                              torch.Tensor]:
        init_end = min(self.init_blocks, prefix_count)
        local_start = max(init_end, prefix_count - self.local_blocks)
        init = torch.arange(init_end, dtype=torch.long, device=device)
        local = torch.arange(local_start, prefix_count, dtype=torch.long,
                             device=device)
        candidates = torch.arange(init_end, local_start, dtype=torch.long,
                                  device=device)
        return torch.cat((init, local)), candidates

    @staticmethod
    def _union_topk(scores: torch.Tensor, candidates: torch.Tensor,
                    budget: int) -> torch.Tensor:
        if candidates.numel() == 0 or budget <= 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        count = min(budget, candidates.numel())
        topk = torch.topk(scores, count, dim=-1).indices
        mask = torch.zeros(candidates.numel(),
                           dtype=torch.bool,
                           device=scores.device)
        mask.scatter_(0, topk.reshape(-1), True)
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        return candidates.index_select(0, selected)

    def _select_prefix(self, query: torch.Tensor,
                       representatives: torch.Tensor, prefix_count: int,
                       block_budget: int,
                       fixed_and_candidates: Optional[tuple[torch.Tensor,
                                                           torch.Tensor]] = None
                       ) -> torch.Tensor:
        prefix_count = min(prefix_count, representatives.shape[0])
        if prefix_count <= 0:
            return torch.empty(0, dtype=torch.long, device=query.device)
        if fixed_and_candidates is None:
            fixed, candidates = self._fixed_and_candidates(prefix_count,
                                                           query.device)
        else:
            fixed, candidates = fixed_and_candidates
        if candidates.numel() == 0:
            return fixed
        candidate_representatives = representatives.index_select(0, candidates)
        scores = self._score_pages(query, candidate_representatives)
        dynamic = self._union_topk(scores, candidates, block_budget)
        mask = torch.zeros(prefix_count, dtype=torch.bool, device=query.device)
        mask.scatter_(0, torch.cat((fixed, dynamic)), True)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _select_prefix_with_logical_indices(
        self,
        query: torch.Tensor,
        representatives: torch.Tensor,
        logical_block_indices: Tuple[int, ...],
        prefix_count: int,
        block_budget: int,
    ) -> torch.Tensor:
        """Select a prefix when metadata carries explicit logical indices.

        This helper is used by the control-plane prediction/check path.  The
        resident GPU selector keeps its existing dense tensor path untouched.
        """
        if representatives.shape[0] != len(logical_block_indices):
            raise ValueError(
                "SolidAttention representative count does not match logical "
                "block indices")
        if any(index < 0 for index in logical_block_indices) or any(
                left >= right for left, right in zip(
                    logical_block_indices, logical_block_indices[1:])):
            raise ValueError(
                "SolidAttention logical block indices must be sorted and "
                "unique")
        if any(index >= prefix_count for index in logical_block_indices):
            raise ValueError(
                "SolidAttention representative is outside immutable prefix")
        if query.device != representatives.device:
            query = query.to(device=representatives.device)

        prefix_count = max(0, prefix_count)
        if prefix_count == 0:
            return torch.empty(0, dtype=torch.long, device=query.device)
        init_end = min(self.init_blocks, prefix_count)
        local_start = max(init_end, prefix_count - self.local_blocks)
        fixed = torch.cat((
            torch.arange(init_end, dtype=torch.long, device=query.device),
            torch.arange(local_start,
                         prefix_count,
                         dtype=torch.long,
                         device=query.device),
        ))
        candidate_logical = tuple(
            index for index in logical_block_indices
            if init_end <= index < local_start)
        if not candidate_logical:
            return fixed
        candidate_positions = tuple(
            position for position, index in enumerate(logical_block_indices)
            if init_end <= index < local_start)
        positions = torch.tensor(candidate_positions,
                                  dtype=torch.long,
                                  device=representatives.device)
        candidates = torch.tensor(candidate_logical,
                                  dtype=torch.long,
                                  device=representatives.device)
        scores = self._score_pages(
            query, representatives.index_select(0, positions))
        dynamic = self._union_topk(scores, candidates, block_budget)
        mask = torch.zeros(prefix_count,
                           dtype=torch.bool,
                           device=representatives.device)
        mask.scatter_(0, torch.cat((fixed.to(mask.device), dynamic)), True)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _get_prediction_representatives(
        self,
        request_id: str,
        layer_index: int,
        num_prefix_blocks: int,
    ) -> Optional[tuple[Tuple[int, ...], torch.Tensor]]:
        page_index_key = self._page_index_keys.get(request_id, request_id)
        state = self._prediction_representatives.get(page_index_key,
                                                     {}).get(layer_index)
        if state is None:
            cached = self._representatives.get(page_index_key,
                                               {}).get(layer_index)
            if cached is None:
                return None
            cached_count, representatives = cached
            logical_block_indices = tuple(range(cached_count))
        else:
            logical_block_indices = state.logical_block_indices
            representatives = state.representatives
        positions = tuple(
            position for position, index in enumerate(logical_block_indices)
            if index < num_prefix_blocks)
        if not positions:
            return None
        if len(positions) != len(logical_block_indices):
            position_tensor = torch.tensor(positions,
                                           dtype=torch.long,
                                           device=representatives.device)
            representatives = representatives.index_select(0, position_tensor)
            logical_block_indices = tuple(logical_block_indices[position]
                                          for position in positions)
        return logical_block_indices, representatives

    def _cached_representatives(self, request_id: str, layer_index: int,
                                key_cache: torch.Tensor,
                                physical_block_ids: torch.Tensor,
                                prefix_count: int,
                                device: torch.device) -> torch.Tensor:
        cached_count, representatives = self._representatives[request_id].get(
            layer_index, (0, None))
        if cached_count > prefix_count:
            cached_count, representatives = 0, None
        if cached_count < prefix_count:
            new_ids = physical_block_ids[cached_count:prefix_count]
            new_representatives = build_page_representatives_from_paged_key_cache(
                key_cache, new_ids, output_device=device)
            representatives = (new_representatives
                               if representatives is None else
                               torch.cat((representatives, new_representatives)))
            cached_count = prefix_count
            self._representatives[request_id][layer_index] = (
                cached_count, representatives)
        if representatives is None:
            return torch.empty((0, 2, 0, 0), device=device)
        if representatives.device != device:
            representatives = representatives.to(device=device)
            self._representatives[request_id][layer_index] = (
                cached_count, representatives)
        return representatives[:prefix_count]

    @staticmethod
    def _mapping_token(physical_block_ids: torch.Tensor) -> int:
        # The resident decode mapping is append-only.  vLLM may reallocate
        # the block-table storage while growing it, even though existing
        # logical-to-physical entries are unchanged.  Use the logical view
        # length as the stable growth token; selected-list mode is disabled
        # for dynamic restore/remapping paths.
        return int(physical_block_ids.numel())

    def _prepare_device_selection(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        sequence_length: int,
        block_size: int,
        block_budget: int,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        query = self._normalize_query(query)
        if sequence_length <= 0 or block_size <= 0 or block_budget <= 0:
            raise ValueError("sequence and block parameters must be positive")
        num_blocks = (sequence_length + block_size - 1) // block_size
        if physical_block_ids.ndim != 1 or physical_block_ids.numel() < num_blocks:
            raise ValueError("physical block ids do not cover the sequence")
        prefix_count = max(0, num_blocks - 1)
        representatives = self._cached_representatives(
            request_id, layer_index, key_cache, physical_block_ids,
            prefix_count, query.device)
        return query, num_blocks, prefix_count, representatives

    def observe_query(self, request_id: str, layer_index: int,
                      query: torch.Tensor) -> None:
        self._previous_queries[request_id][layer_index] = (
            self._normalize_query(query).detach().clone())

    def register_page_representatives(
        self,
        request_id: str,
        layer_index: int,
        page_representatives: torch.Tensor,
        logical_block_indices: Optional[Sequence[int]] = None,
    ) -> None:
        if not isinstance(page_representatives, torch.Tensor):
            raise TypeError("page_representatives must be a tensor")
        if page_representatives.ndim not in (3, 4):
            raise ValueError(
                "SolidAttention representatives must have 3 or 4 dimensions")
        logical = (tuple(range(page_representatives.shape[0]))
                   if logical_block_indices is None else
                   tuple(int(index) for index in logical_block_indices))
        if len(logical) != page_representatives.shape[0]:
            raise ValueError(
                "logical block indices must match representative pages")
        if any(index < 0 for index in logical) or any(
                left >= right for left, right in zip(logical, logical[1:])):
            raise ValueError(
                "SolidAttention logical block indices must be sorted and "
                "unique")
        representatives = page_representatives.detach()
        self._representatives[request_id][layer_index] = (
            len(logical), representatives)
        self._prediction_representatives[request_id][layer_index] = (
            _PredictionRepresentativeState(logical, representatives))

    def bind_page_index_key(self, request_id: str, page_index_key: str) -> None:
        if not request_id or not page_index_key:
            raise ValueError("request_id and page_index_key must not be empty")
        # This mapping is only read by prediction/validation.  The resident
        # selected-list path remains request-local and does not touch it.
        self._page_index_keys[request_id] = page_index_key

    def _select_blocks_device_impl(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        sequence_length: int,
        block_size: int,
        block_budget: int,
    ) -> torch.Tensor:
        query, num_blocks, prefix_count, representatives = (
            self._prepare_device_selection(
                request_id, layer_index, query, key_cache, physical_block_ids,
                sequence_length, block_size, block_budget))

        key = (request_id, layer_index)
        if key not in self._seen:
            self._seen.add(key)
            return torch.arange(num_blocks, dtype=torch.long, device=query.device)

        selected_prefix = self._select_prefix(query, representatives,
                                               prefix_count, block_budget)
        suffix = torch.arange(prefix_count, num_blocks, dtype=torch.long,
                              device=query.device)
        return torch.cat((selected_prefix, suffix))

    def select_blocks_device(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        sequence_length: int,
        block_size: int,
        block_budget: int,
    ) -> torch.Tensor:
        return self._select_blocks_device_impl(
            request_id, layer_index, query, key_cache, physical_block_ids,
            sequence_length, block_size, block_budget)

    def select_attention_blocks_device(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        sequence_length: int,
        block_size: int,
        block_budget: int,
    ) -> SparseKVDeviceSelection:
        """Return a reusable GPU logical-block buffer for selected attention."""
        key = (request_id, layer_index)
        mapping_token = self._mapping_token(physical_block_ids)
        state = self._attention_selection_buffers[request_id].get(layer_index)
        if state is not None and state.mapping_token != mapping_token:
            # A longer logical view means that a new live block became
            # immutable.  Keep the request's warmup state and append only the
            # new representative; the selection itself is refreshed below.
            state = None

        query, num_blocks, prefix_count, representatives = (
            self._prepare_device_selection(
                request_id, layer_index, query, key_cache, physical_block_ids,
                sequence_length, block_size, block_budget))
        reuse = envs.VLLM_GRANULEKV_SPARSE_SELECTION_REUSE_ENABLE
        if (reuse and state is not None and state.ready
                and state.prefix_count == prefix_count
                and state.num_blocks == num_blocks
                and state.buffer.device == query.device):
            record_sparse_kv_selection(refresh=False)
            return SparseKVDeviceSelection(state.buffer, state.count)

        if (state is not None and state.prefix_count == prefix_count
                and state.fixed_blocks.device == query.device):
            fixed_and_candidates = (state.fixed_blocks, state.candidate_blocks)
        else:
            fixed_and_candidates = self._fixed_and_candidates(
                prefix_count, query.device)

        if key not in self._seen:
            self._seen.add(key)
            selected = torch.arange(num_blocks, dtype=torch.long,
                                    device=query.device)
            ready = False
        else:
            selected_prefix = self._select_prefix(query, representatives,
                                                   prefix_count, block_budget,
                                                   fixed_and_candidates)
            suffix = torch.arange(prefix_count, num_blocks, dtype=torch.long,
                                  device=query.device)
            selected = torch.cat((selected_prefix, suffix))
            ready = True
        record_sparse_kv_selection(refresh=True)
        count = selected.shape[0]
        if state is None or state.buffer.numel() < count or state.buffer.device != selected.device:
            capacity = count if state is None else max(count, state.buffer.numel() * 2)
            buffer = torch.empty(capacity,
                                 dtype=torch.int32,
                                 device=selected.device)
        else:
            buffer = state.buffer
        buffer[:count].copy_(selected)
        self._attention_selection_buffers[request_id][layer_index] = (
            _AttentionSelectionState(buffer, fixed_and_candidates[0],
                                     fixed_and_candidates[1], prefix_count,
                                     num_blocks, count, mapping_token, ready))
        return SparseKVDeviceSelection(buffer, count)

    def select_blocks(self, request_id: str, layer_index: int,
                      query: torch.Tensor,
                      page_representatives: Optional[torch.Tensor],
                      num_blocks: int,
                      block_budget: int) -> tuple[int, ...]:
        query = self._normalize_query(query)
        if num_blocks <= 0 or block_budget <= 0:
            raise ValueError("num_blocks and block_budget must be positive")
        representatives = page_representatives
        if representatives is None or representatives.numel() == 0:
            representatives = self._representatives[request_id].get(
                layer_index, (0, None))[1]
        if representatives is None:
            return tuple(range(num_blocks))
        key = (request_id, layer_index)
        if key not in self._seen:
            self._seen.add(key)
            return tuple(range(num_blocks))
        prefix_count = max(0, min(num_blocks - 1, representatives.shape[0]))
        selected = self._select_prefix(query, representatives, prefix_count,
                                       block_budget)
        suffix = torch.arange(prefix_count, num_blocks, dtype=torch.long,
                              device=selected.device)
        return tuple(torch.cat((selected, suffix)).tolist())

    def select_blocks_for_residency_check(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        num_blocks: int,
        block_budget: int,
    ) -> tuple[int, ...]:
        """Select the current working set without request warmup.

        Dynamic restore uses this control-plane check to detect a prediction
        miss before the compact compatibility consumer reads KV.  It is not
        used by resident-only selected-list attention.
        """
        if num_blocks <= 0 or block_budget <= 0:
            raise ValueError("num_blocks and block_budget must be positive")
        prefix_count = max(0, num_blocks - 1)
        metadata = self._get_prediction_representatives(
            request_id, layer_index, prefix_count)
        if metadata is None:
            raise RuntimeError(
                "SolidAttention prediction validation lacks prefix metadata: "
                f"request={request_id} layer={layer_index}")
        logical, representatives = metadata
        selected_prefix = self._select_prefix_with_logical_indices(
            self._normalize_query(query), representatives, logical,
            prefix_count, block_budget)
        suffix = torch.arange(prefix_count,
                              num_blocks,
                              dtype=torch.long,
                              device=selected_prefix.device)
        return tuple(torch.cat((selected_prefix, suffix)).tolist())

    def predict_restore_blocks(self, request_id: str, layer_index: int,
                               num_prefix_blocks: int,
                               block_budget: int) -> Optional[Tuple[int, ...]]:
        history = self._previous_queries.get(request_id, {}).get(layer_index)
        if history is None:
            return None
        metadata = self._get_prediction_representatives(
            request_id, layer_index, num_prefix_blocks)
        if metadata is None:
            return None
        logical, representatives = metadata
        selected = self._select_prefix_with_logical_indices(
            history, representatives, logical, num_prefix_blocks, block_budget)
        return tuple(int(index) for index in selected.tolist())

    def discard(self, request_id: str) -> None:
        page_index_key = self._page_index_keys.pop(request_id, None)
        self._previous_queries.pop(request_id, None)
        self._representatives.pop(request_id, None)
        # Stable-prefix metadata is shared by requests using the same page
        # key.  Request-local metadata is always removed; the stable entry is
        # retained until a later prefix registration replaces it.
        if page_index_key is None or page_index_key == request_id:
            self._prediction_representatives.pop(request_id, None)
        self._attention_selection_buffers.pop(request_id, None)
        self._seen = {key for key in self._seen if key[0] != request_id}
