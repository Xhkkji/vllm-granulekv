# SPDX-License-Identifier: Apache-2.0

"""层级 GranuleKV restore 的 model-forward 层屏障。

这个模块故意不依赖 Scheduler、Worker 或具体模型。它只在一次 model
forward 的动态作用域中保存一个很小的回调：模型进入第 N 层前调用
``wait_for_local_layer(N)``，回调负责确认属于当前请求的 N 所在 window
已经完成 SSD -> HBM DMA。

这样模型代码不知道 GranuleKV request id，Worker 也不需要依赖 Qwen2 的实现。
默认没有激活 session，入口立即返回，因此普通 vLLM 路径不改变数据语义。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple


LayerWaitCallback = Callable[[int, Sequence[str], int], Optional[Tuple[int, ...]]]
LayerReleaseCallback = Callable[[int, Sequence[str], int], None]


@dataclass(frozen=True)
class HierarchicalLayerBarrierConfig:
    """Step 4 的显式开关，默认保持 Step 1/2 的 full-restore 行为。"""

    enabled: bool = False

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "HierarchicalLayerBarrierConfig":
        values = environ if environ is not None else os.environ
        return cls(enabled=bool(int(values.get(
            "VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER", "0"))))


@dataclass(frozen=True)
class _LayerBarrierSession:
    """一次 forward 对应的 worker-local barrier 上下文。"""

    callback: LayerWaitCallback
    release_callback: Optional[LayerReleaseCallback]
    virtual_engine: int
    request_ids: tuple[str, ...]
    sequence_lengths_by_request: Optional[Dict[str, int]]
    block_size: Optional[int]


_ACTIVE_LAYER_BARRIER: ContextVar[Optional[_LayerBarrierSession]] = ContextVar(
    "granulekv_hierarchical_layer_barrier", default=None)
_ACTIVE_SPARSE_KV_BLOCKS: ContextVar[Optional[Tuple[int, ...]]] = ContextVar(
    "granulekv_sparse_kv_blocks", default=None)


@contextmanager
def activate_layer_barrier(
    callback: LayerWaitCallback,
    *,
    virtual_engine: int,
    request_ids: Sequence[str],
    release_callback: Optional[LayerReleaseCallback] = None,
    sequence_lengths_by_request: Optional[Mapping[str, int]] = None,
    block_size: Optional[int] = None,
) -> Iterator[None]:
    """在 model forward 的动态范围内安装 worker-local 回调。

    request id 来自该 batch 的 ``ModelInput``。连续 batching 时一个 forward
    可能包含普通请求和层级 restore 请求；callback 必须仅处理这组 id 中自己
    管理的 restore window，其他请求自然是 no-op。
    """
    token = _ACTIVE_LAYER_BARRIER.set(
        _LayerBarrierSession(callback=callback,
                             release_callback=release_callback,
                             virtual_engine=virtual_engine,
                             request_ids=tuple(request_ids),
                             sequence_lengths_by_request=(
                                 None if sequence_lengths_by_request is None else
                                 {str(request_id): int(length)
                                  for request_id, length in
                                  sequence_lengths_by_request.items()}),
                             block_size=block_size))
    try:
        yield
    finally:
        _ACTIVE_LAYER_BARRIER.reset(token)


def wait_for_local_layer(layer_index: int) -> Optional[Tuple[int, ...]]:
    """在 attention 使用该层 KV 前确认对应 window READY。

    没有活动 session 时直接返回。该情况包括开关关闭、decode batch 以及不属于
    GranuleKV hierarchical restore 的普通请求，因此模型代码不需要区分这些路径。
    """
    session = _ACTIVE_LAYER_BARRIER.get()
    if session is None:
        return None
    return session.callback(session.virtual_engine, session.request_ids,
                            layer_index)


def release_local_layer(layer_index: int) -> None:
    """通知 runtime 当前层消费完成，window 末尾可逐出对应 KV region。"""
    session = _ACTIVE_LAYER_BARRIER.get()
    if session is None or session.release_callback is None:
        return
    session.release_callback(session.virtual_engine, session.request_ids,
                             layer_index)


@contextmanager
def activate_sparse_kv_blocks(
    block_indices: Optional[Sequence[int]],
) -> Iterator[None]:
    """把当前 layer 的 sparse block 集合暴露给 attention backend。

    该接口只传递访问约束，不修改 block table，也不选择具体 sparse kernel。
    ``None`` 明确表示 dense attention。后续 xFormers/FlashAttention 的 sparse
    consumer 可以调用 ``get_active_sparse_kv_blocks`` 取得同一份 logical
    block 集合，而无需依赖 Qwen2 或 scheduler 类型。
    """
    selected = (None if block_indices is None else tuple(block_indices))
    token = _ACTIVE_SPARSE_KV_BLOCKS.set(selected)
    try:
        yield
    finally:
        _ACTIVE_SPARSE_KV_BLOCKS.reset(token)


def get_active_sparse_kv_blocks() -> Optional[Tuple[int, ...]]:
    """返回当前 layer 允许 attention 访问的 logical KV blocks。"""
    return _ACTIVE_SPARSE_KV_BLOCKS.get()


def get_active_layer_request_ids() -> Tuple[str, ...]:
    """Return request ids for the current model-forward barrier session."""
    session = _ACTIVE_LAYER_BARRIER.get()
    return () if session is None else session.request_ids


def get_active_layer_sequence_lengths() -> Optional[Dict[str, int]]:
    """Return current model sequence lengths for the active barrier session."""
    session = _ACTIVE_LAYER_BARRIER.get()
    if session is None or session.sequence_lengths_by_request is None:
        return None
    return dict(session.sequence_lengths_by_request)
