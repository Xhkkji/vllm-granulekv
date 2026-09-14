"""Quest policy plugin for the generic sparse KV policy hook."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence

import torch

from vllm.attention.ops.sparse_kv import (
    build_page_representatives_from_paged_key_cache, )

from .page_index import QuestPageIndex
from .selector import QuestPageSelector


class QuestPolicy:
    """上一 decode query 预测下一次 decode 的 logical KV pages.

    The first call for a request/layer is an explicit dense warmup.  This keeps
    missing history from silently becoming an incorrect sparse selection.
    """

    name = "quest"

    def __init__(self, page_index: Optional[QuestPageIndex] = None) -> None:
        self.page_index = page_index or QuestPageIndex()
        self.selector = QuestPageSelector()
        self._previous_queries: Dict[str, Dict[int, torch.Tensor]] = defaultdict(
            dict)
        self._page_index_keys: Dict[str, str] = {}
        self._restore_prefix_blocks: Dict[str, int] = {}
        self._last_selection: Dict[str, Dict[int, tuple[int, ...]]] = defaultdict(
            dict)
        self._gpu_page_representatives: Dict[
            str, Dict[int, tuple[int, torch.Tensor]]] = defaultdict(dict)
        self._metrics = {"warmup": 0, "selected": 0, "correction": 0}

    def observe_query(self, request_id: str, layer_index: int,
                      query: torch.Tensor) -> None:
        if query.ndim == 3:
            if query.shape[0] != 1:
                raise ValueError("QuestPolicy only supports batch size 1")
            query = query[0]
        if query.ndim != 2:
            raise ValueError("Quest query must have shape [heads, head_dim]")
        self._previous_queries[request_id][layer_index] = query.detach().clone()

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
        """Select Quest blocks while keeping the decode path on the GPU.

        This is an optional adapter capability.  It does not expose physical
        mappings or own any allocator state; the bridge still performs the
        logical-to-physical projection.
        """
        if query.ndim == 3:
            if query.shape[0] != 1:
                raise ValueError("QuestPolicy only supports batch size 1")
            query = query[0]
        if query.ndim != 2:
            raise ValueError("Quest query must have shape [heads, head_dim]")
        if sequence_length <= 0 or block_size <= 0 or block_budget <= 0:
            raise ValueError("sequence and block parameters must be positive")
        num_blocks = (sequence_length + block_size - 1) // block_size
        if physical_block_ids.ndim != 1 or physical_block_ids.shape[0] < num_blocks:
            raise ValueError("physical block ids do not cover the sequence")

        history = self._previous_queries.get(request_id, {}).get(layer_index)
        if history is None:
            self._metrics["warmup"] += 1
            return torch.arange(num_blocks, dtype=torch.long, device=query.device)
        if history.device != query.device:
            history = history.to(device=query.device)
            self._previous_queries[request_id][layer_index] = history

        # The last logical block is the live suffix.  Only immutable prefix
        # pages are indexed and scored; the suffix is appended below.
        prefix_count = max(0, num_blocks - 1)
        page_count = prefix_count
        prefix_limit = self._restore_prefix_blocks.get(request_id)
        if prefix_limit is not None:
            page_count = min(page_count, prefix_limit)

        selected = torch.empty(0, dtype=torch.long, device=query.device)
        if page_count > 0:
            cached_count, representatives = self._gpu_page_representatives[
                request_id].get(layer_index, (0, None))
            if cached_count > prefix_count:
                cached_count = 0
                representatives = None
            if cached_count < prefix_count:
                new_ids = physical_block_ids[cached_count:prefix_count]
                new_representatives = (
                    build_page_representatives_from_paged_key_cache(
                        key_cache,
                        new_ids,
                        output_device=query.device))
                representatives = (new_representatives if representatives is None
                                   else torch.cat((representatives,
                                                   new_representatives), dim=0))
                cached_count = prefix_count
                self._gpu_page_representatives[request_id][layer_index] = (
                    cached_count, representatives)
            representatives = representatives[:page_count]
            selected = self.selector.select_tensor_from_query(
                history, representatives, min(block_budget, page_count))

        suffix = torch.arange(page_count,
                              num_blocks,
                              dtype=torch.long,
                              device=query.device)
        selected = torch.unique(torch.cat((selected, suffix)), sorted=True)
        self._metrics["selected"] += int(selected.numel())
        return selected

    def register_page_representatives(
        self,
        request_id: str,
        layer_index: int,
        page_representatives: torch.Tensor,
        logical_block_indices: Optional[Sequence[int]] = None,
    ) -> None:
        page_index_key = self._page_index_keys.get(request_id, request_id)
        self.page_index.register(page_index_key, layer_index,
                                  page_representatives,
                                  logical_block_indices)

    def predict_restore_blocks(
        self,
        request_id: str,
        layer_index: int,
        num_prefix_blocks: int,
        block_budget: int,
    ) -> Optional[tuple[int, ...]]:
        """Predict the next restore's immutable prefix blocks.

        This reuses the same selector as ``select_blocks``.  Unlike the
        consumer path, the input block count is the prefix count, so no live
        suffix can enter the restore plan.
        """
        if num_prefix_blocks <= 0 or block_budget <= 0:
            raise ValueError(
                "num_prefix_blocks and block_budget must be positive")
        history = self._previous_queries.get(request_id, {}).get(layer_index)
        if history is None:
            return None
        try:
            selection, page_count = self._select_prefix_blocks(
                request_id, layer_index, None, num_prefix_blocks, block_budget)
            if page_count != num_prefix_blocks:
                return None
            return selection
        except KeyError:
            return None

    def bind_page_index_key(self, request_id: str, page_index_key: str) -> None:
        if not request_id or not page_index_key:
            raise ValueError("request_id and page_index_key must not be empty")
        self._page_index_keys[request_id] = page_index_key

    def set_restore_prefix_blocks(self, request_id: str,
                                  num_prefix_blocks: int) -> None:
        """Record the immutable prefix boundary for the consumer path."""
        if not request_id or num_prefix_blocks <= 0:
            raise ValueError("invalid sparse restore prefix context")
        self._restore_prefix_blocks[request_id] = num_prefix_blocks

    def select_blocks(self, request_id: str, layer_index: int,
                      query: torch.Tensor,
                      page_representatives: Optional[torch.Tensor], num_blocks: int,
                      block_budget: int) -> tuple[int, ...]:
        if num_blocks <= 0 or block_budget <= 0:
            raise ValueError("num_blocks and block_budget must be positive")
        history = self._previous_queries.get(request_id, {}).get(layer_index)
        if history is None:
            self._metrics["warmup"] += 1
            return tuple(range(num_blocks))
        selection, page_count = self._select_prefix_blocks(
            request_id, layer_index, page_representatives, num_blocks,
            block_budget)
        selected = set(selection)
        # The page index covers only immutable prefix blocks.  Blocks after it
        # are the live request suffix and have no SSD representative.
        selected.update(range(page_count, num_blocks))
        # The newest logical block contains the current tail and is always
        # retained, even when its representative score is low.
        selected.add(num_blocks - 1)
        result = tuple(sorted(selected))
        if any(index >= num_blocks for index in result):
            raise ValueError("Quest selection is outside num_blocks")
        self._last_selection[request_id][layer_index] = result
        self._metrics["selected"] += len(result)
        self._metrics["correction"] += int(num_blocks - 1 not in selection)
        return result

    def _select_prefix_blocks(
        self,
        request_id: str,
        layer_index: int,
        page_representatives: Optional[torch.Tensor],
        num_blocks: int,
        block_budget: int,
    ) -> tuple[tuple[int, ...], int]:
        history = self._previous_queries.get(request_id, {}).get(layer_index)
        if history is None:
            return tuple(range(num_blocks)), num_blocks
        representatives = page_representatives
        if representatives is None or representatives.numel() == 0:
            page_index_key = self._page_index_keys.get(request_id, request_id)
            representatives = self.page_index.get(page_index_key, layer_index)
        if representatives.ndim not in (3, 4):
            raise ValueError(
                "Quest page representatives must be conditioned maxima or "
                "min/max envelopes")
        query_heads = history.shape[0]
        if representatives.ndim == 4:
            page_count = representatives.shape[0]
        elif (representatives.shape[1] > 0
              and query_heads % representatives.shape[1] == 0):
            page_count = representatives.shape[0]
        elif (representatives.shape[0] > 0
              and query_heads % representatives.shape[0] == 0):
            page_count = representatives.shape[1]
        else:
            page_count = 0
        if page_count <= 0:
            raise ValueError("Quest page representatives must contain a page")
        if history.device != representatives.device:
            history = history.to(device=representatives.device)
        prefix_limit = self._restore_prefix_blocks.get(request_id)
        if prefix_limit is not None:
            page_count = min(page_count, prefix_limit)
        page_count = min(page_count, num_blocks)
        if (representatives.ndim == 4
                or (representatives.shape[1] > 0
                    and query_heads % representatives.shape[1] == 0)):
            representatives = representatives[:page_count]
        else:
            representatives = representatives[:, :page_count, :]
        prefix_budget = min(block_budget, page_count)
        selection = self.selector.select_from_query(
            history, representatives, prefix_budget).selected_blocks
        result = tuple(sorted(set(selection)))
        if any(index < 0 or index >= page_count for index in result):
            raise ValueError("Quest selection is outside num_blocks")
        return result, page_count

    def metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def discard(self, request_id: str) -> None:
        self._previous_queries.pop(request_id, None)
        self._last_selection.pop(request_id, None)
        self._restore_prefix_blocks.pop(request_id, None)
        self._gpu_page_representatives.pop(request_id, None)
        self.page_index.discard(self._page_index_keys.pop(request_id,
                                                          request_id))
