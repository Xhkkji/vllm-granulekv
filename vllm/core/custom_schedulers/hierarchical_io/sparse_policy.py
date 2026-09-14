# SPDX-License-Identifier: Apache-2.0
"""Paper-independent sparse KV policy boundary.

The policy chooses logical blocks and observes model queries.  It does not own
GranuleKV requests, physical block allocation, layer barriers, or attention
kernels.  Paper implementations are loaded by module path so vLLM core never
imports a Quest/HiSparse/SolidAttention package directly.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from contextvars import ContextVar
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import msgspec
import torch


class SparseKVPlanFeedback(msgspec.Struct,
                           omit_defaults=True,
                           array_like=True):
    """Small serializable control-plane result for a future restore."""

    page_index_key: str
    num_layers: int
    num_prefix_blocks: int
    block_indices_by_layer: Tuple[Tuple[int, ...], ...]


class SparseKVPolicy(Protocol):
    name: str

    def select_blocks(
        self,
        request_id: str,
        layer_index: int,
        query: torch.Tensor,
        page_representatives: Optional[torch.Tensor],
        num_blocks: int,
        block_budget: int,
    ) -> tuple[int, ...]:
        ...

    def observe_query(self, request_id: str, layer_index: int,
                      query: torch.Tensor) -> None:
        ...

    def register_page_representatives(
        self,
        request_id: str,
        layer_index: int,
        page_representatives: torch.Tensor,
        logical_block_indices: Optional[Sequence[int]] = None,
    ) -> None:
        ...

    def bind_page_index_key(self, request_id: str, page_index_key: str) -> None:
        ...

    def predict_restore_blocks(
        self,
        request_id: str,
        layer_index: int,
        num_prefix_blocks: int,
        block_budget: int,
    ) -> Optional[Tuple[int, ...]]:
        ...


@dataclass
class _SparseRestoreContext:
    page_index_key: str
    num_prefix_blocks: int
    layers: set[int] = field(default_factory=set)


_SPARSE_RESTORE_CONTEXTS: Dict[str, _SparseRestoreContext] = {}


def register_sparse_restore_context(
    request_id: str,
    page_index_key: str,
    num_prefix_blocks: int,
    layer_range: Optional[Tuple[int, int]],
) -> None:
    """Remember only the metadata needed to build the next-step plan."""
    if not request_id or not page_index_key or num_prefix_blocks <= 0:
        return
    context = _SPARSE_RESTORE_CONTEXTS.get(request_id)
    if context is None:
        context = _SparseRestoreContext(page_index_key, num_prefix_blocks)
        _SPARSE_RESTORE_CONTEXTS[request_id] = context
    elif (context.page_index_key != page_index_key
          or context.num_prefix_blocks != num_prefix_blocks):
        raise RuntimeError("inconsistent sparse restore context")
    if layer_range is not None:
        start, end = layer_range
        context.layers.update(range(start, end))
    policy = get_sparse_kv_policy()
    set_prefix = (None if policy is None else
                  getattr(policy, "set_restore_prefix_blocks", None))
    if callable(set_prefix):
        set_prefix(request_id, num_prefix_blocks)


def build_sparse_restore_plan_feedback(
    request_id: str,
    block_budget: int,
) -> Optional[SparseKVPlanFeedback]:
    """Build a future restore plan from the policy's current history."""
    policy = get_sparse_kv_policy()
    context = _SPARSE_RESTORE_CONTEXTS.get(request_id)
    predictor = (None if policy is None else
                 getattr(policy, "predict_restore_blocks", None))
    if context is None or not callable(predictor):
        return None
    selections = []
    for layer_index in sorted(context.layers):
        selection = predictor(request_id, layer_index,
                              context.num_prefix_blocks, block_budget)
        if selection is None:
            return None
        normalized = tuple(sorted(set(int(index) for index in selection)))
        if (not normalized
                or normalized[0] < 0
                or normalized[-1] >= context.num_prefix_blocks):
            raise RuntimeError("sparse restore prediction is outside prefix")
        selections.append(normalized)
    if not selections:
        return None
    return SparseKVPlanFeedback(
        page_index_key=context.page_index_key,
        num_layers=len(selections),
        num_prefix_blocks=context.num_prefix_blocks,
        block_indices_by_layer=tuple(selections),
    )


def discard_sparse_restore_context(request_ids: Sequence[str]) -> None:
    for request_id in request_ids:
        _SPARSE_RESTORE_CONTEXTS.pop(request_id, None)


def load_sparse_kv_policy(module_spec: Optional[str] = None,
                          **kwargs: Any) -> SparseKVPolicy:
    """Load ``module:Class`` from the configured experiment plugin."""
    spec = module_spec or os.getenv("VLLM_GRANULEKV_SPARSE_POLICY_MODULE")
    if not spec or ":" not in spec:
        raise ValueError(
            "sparse policy must use module:Class syntax; set "
            "VLLM_GRANULEKV_SPARSE_POLICY_MODULE")
    module_name, class_name = spec.split(":", 1)
    if not module_name or not class_name:
        raise ValueError("sparse policy module specification is incomplete")
    policy_type = getattr(importlib.import_module(module_name), class_name)
    policy = policy_type(**kwargs)
    for method in ("select_blocks", "observe_query"):
        if not callable(getattr(policy, method, None)):
            raise TypeError(f"sparse policy does not implement {method}()")
    return policy


_ACTIVE_POLICY: ContextVar[Optional[SparseKVPolicy]] = ContextVar(
    "granulekv_sparse_kv_policy", default=None)
_DEFAULT_POLICY: Optional[SparseKVPolicy] = None


def configure_sparse_kv_policy(policy: Optional[SparseKVPolicy]) -> None:
    """Set the process-local policy used by the generic attention hook."""
    global _DEFAULT_POLICY
    _DEFAULT_POLICY = policy


def get_sparse_kv_policy() -> Optional[SparseKVPolicy]:
    policy = _ACTIVE_POLICY.get() or _DEFAULT_POLICY
    if policy is None and os.getenv("VLLM_GRANULEKV_SPARSE_POLICY_MODULE"):
        policy = load_sparse_kv_policy()
        configure_sparse_kv_policy(policy)
    return policy


def observe_sparse_query(layer_index: int, query: torch.Tensor,
                         request_id: str = "default") -> None:
    """Forward a query to an enabled policy; otherwise remain a no-op."""
    policy = get_sparse_kv_policy()
    if policy is not None:
        policy.observe_query(request_id, layer_index, query)


def select_sparse_blocks(
    request_id: str,
    layer_index: int,
    query: torch.Tensor,
    page_representatives: Optional[torch.Tensor],
    num_blocks: int,
    block_budget: int,
) -> tuple[int, ...]:
    """Select logical blocks through the configured policy plugin."""
    policy = get_sparse_kv_policy()
    if policy is None:
        raise RuntimeError("no sparse KV policy is configured")
    return policy.select_blocks(request_id, layer_index, query,
                                page_representatives, num_blocks, block_budget)


def register_sparse_page_representatives(
    page_index_key: str,
    layer_index: int,
    page_representatives: torch.Tensor,
    logical_block_indices: Optional[Sequence[int]] = None,
) -> None:
    """Publish CPU page metadata to the configured policy, if enabled."""
    policy = get_sparse_kv_policy()
    if policy is None:
        return
    register = getattr(policy, "register_page_representatives", None)
    if not callable(register):
        raise TypeError(
            f"sparse policy {policy.name!r} does not implement "
            "register_page_representatives()")
    register(page_index_key, layer_index, page_representatives,
             logical_block_indices)


def bind_sparse_page_index_key(request_id: str, page_index_key: str) -> None:
    """Bind a live request id to its stable prefix metadata key."""
    policy = get_sparse_kv_policy()
    if policy is None:
        return
    bind = getattr(policy, "bind_page_index_key", None)
    if not callable(bind):
        raise TypeError(
            f"sparse policy {policy.name!r} does not implement "
            "bind_page_index_key()")
    bind(request_id, page_index_key)


class SparseKVPolicyRuntime:
    """Small request/layer policy facade for tests and model-runner adapters."""

    def __init__(self, policy: SparseKVPolicy) -> None:
        self.policy = policy

    def observe_query(self, request_id: str, layer_index: int,
                      query: torch.Tensor) -> None:
        self.policy.observe_query(request_id, layer_index, query)

    def select_blocks(self, request_id: str, layer_index: int,
                      query: torch.Tensor,
                      page_representatives: torch.Tensor, num_blocks: int,
                      block_budget: int) -> tuple[int, ...]:
        return self.policy.select_blocks(request_id, layer_index, query,
                                         page_representatives, num_blocks,
                                         block_budget)

    def predict_restore_blocks(self, request_id: str, layer_index: int,
                               num_prefix_blocks: int,
                               block_budget: int) -> Optional[Tuple[int, ...]]:
        predictor = getattr(self.policy, "predict_restore_blocks", None)
        if not callable(predictor):
            return None
        return predictor(request_id, layer_index, num_prefix_blocks,
                         block_budget)
