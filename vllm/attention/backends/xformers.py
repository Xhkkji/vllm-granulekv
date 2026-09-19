# SPDX-License-Identifier: Apache-2.0
"""Attention layer with xFormers and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton 可能在某些环境下不可用
    triton = None
    tl = None
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (
    CommonAttentionState, CommonMetadataBuilder,
    get_num_prefill_decode_query_kv_tokens, get_seq_len_block_table_args,
    is_all_cross_attn_metadata_set, is_all_encoder_attn_metadata_set)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.attention.ops.sparse_kv import (
    build_selected_decode_block_table, select_and_compact_decode_blocks,
    record_sparse_kv_attention, validate_sparse_kv_device_selection)
from vllm.core.custom_schedulers.hierarchical_io import (
    get_active_layer_request_ids, get_active_sparse_kv_blocks,
    get_sparse_kv_policy, register_sparse_page_representatives,
    select_sparse_blocks)
from vllm.logger import init_logger

logger = init_logger(__name__)


if triton is not None:

    @triton.jit
    def _gather_packed_key_cache_kernel(
        key_cache_ptr,
        block_ids_ptr,
        block_offsets_ptr,
        dst_token_positions_ptr,
        out_ptr,
        total_elements,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        pack_size: tl.constexpr,
        stride_cache_block,
        stride_cache_head,
        stride_cache_d_outer,
        stride_cache_token,
        stride_cache_pack,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        """从 vLLM packed key cache 直接 gather 到连续 prefix key。

        当前 fallback 最重的一段并不是 `gather_cache` 本身，而是前面的：

        ```text
        key_cache.permute(...).contiguous().view(...)
        ```

        这一步会把 packed key cache 重新排成“token-major 连续布局”，
        代价约 0.6ms / layer。这里直接按 packed layout 寻址，把 prefix token
        scatter/gather 成连续输出，从而绕开这次整块重排。
        """
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden_per_token = num_kv_heads * head_size
        token_idx = offsets // hidden_per_token
        hidden_idx = offsets % hidden_per_token
        kv_head = hidden_idx // head_size
        head_offset = hidden_idx % head_size

        block_id = tl.load(block_ids_ptr + token_idx, mask=mask, other=0)
        block_offset = tl.load(block_offsets_ptr + token_idx, mask=mask, other=0)
        dst_token_idx = tl.load(dst_token_positions_ptr + token_idx,
                                mask=mask,
                                other=0)
        d_outer = head_offset // pack_size
        d_inner = head_offset % pack_size

        src_offsets = (
            block_id * stride_cache_block +
            kv_head * stride_cache_head +
            d_outer * stride_cache_d_outer +
            block_offset * stride_cache_token +
            d_inner * stride_cache_pack
        )
        dst_offsets = (
            dst_token_idx * stride_out_token +
            kv_head * stride_out_head +
            head_offset * stride_out_dim
        )
        values = tl.load(key_cache_ptr + src_offsets, mask=mask)
        tl.store(out_ptr + dst_offsets, values, mask=mask)


    @triton.jit
    def _gather_packed_value_cache_kernel(
        value_cache_ptr,
        block_ids_ptr,
        block_offsets_ptr,
        dst_token_positions_ptr,
        out_ptr,
        total_elements,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        stride_cache_block,
        stride_cache_head,
        stride_cache_dim,
        stride_cache_token,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        """从 vLLM packed value cache 直接 gather 到连续 prefix value。"""
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden_per_token = num_kv_heads * head_size
        token_idx = offsets // hidden_per_token
        hidden_idx = offsets % hidden_per_token
        kv_head = hidden_idx // head_size
        head_offset = hidden_idx % head_size

        block_id = tl.load(block_ids_ptr + token_idx, mask=mask, other=0)
        block_offset = tl.load(block_offsets_ptr + token_idx, mask=mask, other=0)
        dst_token_idx = tl.load(dst_token_positions_ptr + token_idx,
                                mask=mask,
                                other=0)

        src_offsets = (
            block_id * stride_cache_block +
            kv_head * stride_cache_head +
            head_offset * stride_cache_dim +
            block_offset * stride_cache_token
        )
        dst_offsets = (
            dst_token_idx * stride_out_token +
            kv_head * stride_out_head +
            head_offset * stride_out_dim
        )
        values = tl.load(value_cache_ptr + src_offsets, mask=mask)
        tl.store(out_ptr + dst_offsets, values, mask=mask)


    @triton.jit
    def _scatter_contiguous_kv_tokens_kernel(
        src_ptr,
        dst_token_positions_ptr,
        out_ptr,
        total_elements,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        stride_src_token,
        stride_src_head,
        stride_src_dim,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        """把连续 query KV 直接 scatter 到 full KV buffer。

        prefix 侧我们已经收敛成：

        ```text
        paged cache -> full workspace
        ```

        query 侧如果继续保留 Python `for segment in ...: copy_()`，就会出现：

        - 控制面已经在 GPU plan 里了
        - 但真正的数据写入还在逐段从 Python 发起

        这会让后续 persistent kernel / service CTA 很难自然接进来。

        因此这里引入一个非常薄的通用 scatter kernel：

        - `src_ptr` 是已经连续的 query key/value
        - `dst_token_positions_ptr[token]` 给出这个 query token 在最终 full KV
          序列中的目标 token 下标
        - kernel 负责把 `(token, head, dim)` 元素直接写到最终 full buffer

        这样 query 侧就和 prefix 侧一样，变成“GPU plan + GPU kernel”语义。
        """
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden_per_token = num_kv_heads * head_size
        token_idx = offsets // hidden_per_token
        hidden_idx = offsets % hidden_per_token
        kv_head = hidden_idx // head_size
        head_offset = hidden_idx % head_size

        dst_token_idx = tl.load(dst_token_positions_ptr + token_idx,
                                mask=mask,
                                other=0)
        src_offsets = (
            token_idx * stride_src_token +
            kv_head * stride_src_head +
            head_offset * stride_src_dim
        )
        dst_offsets = (
            dst_token_idx * stride_out_token +
            kv_head * stride_out_head +
            head_offset * stride_out_dim
        )
        values = tl.load(src_ptr + src_offsets, mask=mask)
        tl.store(out_ptr + dst_offsets, values, mask=mask)


    @triton.jit
    def _compose_single_request_packed_key_cache_kernel(
        key_cache_ptr,
        query_key_ptr,
        context_block_ids_ptr,
        context_block_offsets_ptr,
        out_ptr,
        total_elements,
        context_tokens,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        pack_size: tl.constexpr,
        stride_cache_block,
        stride_cache_head,
        stride_cache_d_outer,
        stride_cache_token,
        stride_cache_pack,
        stride_query_token,
        stride_query_head,
        stride_query_dim,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        """单请求场景下，直接把 prefix key + query key 一次性写入 full KV。

        当前真实主线几乎都是：

        - 单个 prefill request
        - prefix 命中一整段连续上下文
        - query 自身已经是连续 token-major 张量

        在这种场景里，原有流程虽然已经很轻，但仍然会拆成：

        1. prefix packed gather kernel
        2. query scatter kernel

        这里把这两步合成“一次遍历 full token 空间”的 compose kernel：

        - `token < context_tokens` 时，从 paged/paked cache 读取 prefix
        - `token >= context_tokens` 时，从连续 query tensor 读取 query

        这样 key 侧就从两次 kernel 缩成一次 kernel，更贴近后续 persistent
        consumer 想要的“按最终消费布局直接产出”语义。
        """
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden_per_token = num_kv_heads * head_size
        token_idx = offsets // hidden_per_token
        hidden_idx = offsets % hidden_per_token
        kv_head = hidden_idx // head_size
        head_offset = hidden_idx % head_size

        is_prefix = token_idx < context_tokens
        safe_prefix_idx = tl.where(is_prefix, token_idx, 0)

        block_id = tl.load(context_block_ids_ptr + safe_prefix_idx,
                           mask=mask,
                           other=0)
        block_offset = tl.load(context_block_offsets_ptr + safe_prefix_idx,
                               mask=mask,
                               other=0)
        d_outer = head_offset // pack_size
        d_inner = head_offset % pack_size

        prefix_offsets = (
            block_id * stride_cache_block +
            kv_head * stride_cache_head +
            d_outer * stride_cache_d_outer +
            block_offset * stride_cache_token +
            d_inner * stride_cache_pack
        )

        query_token_idx = token_idx - context_tokens
        query_offsets = (
            query_token_idx * stride_query_token +
            kv_head * stride_query_head +
            head_offset * stride_query_dim
        )

        prefix_values = tl.load(key_cache_ptr + prefix_offsets, mask=mask)
        query_values = tl.load(query_key_ptr + query_offsets,
                               mask=mask & (~is_prefix))
        values = tl.where(is_prefix, prefix_values, query_values)

        dst_offsets = (
            token_idx * stride_out_token +
            kv_head * stride_out_head +
            head_offset * stride_out_dim
        )
        tl.store(out_ptr + dst_offsets, values, mask=mask)


    @triton.jit
    def _compose_single_request_packed_value_cache_kernel(
        value_cache_ptr,
        query_value_ptr,
        context_block_ids_ptr,
        context_block_offsets_ptr,
        out_ptr,
        total_elements,
        context_tokens,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        stride_cache_block,
        stride_cache_head,
        stride_cache_dim,
        stride_cache_token,
        stride_query_token,
        stride_query_head,
        stride_query_dim,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        """单请求场景下，一次性把 prefix value + query value 写入 full KV。"""
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden_per_token = num_kv_heads * head_size
        token_idx = offsets // hidden_per_token
        hidden_idx = offsets % hidden_per_token
        kv_head = hidden_idx // head_size
        head_offset = hidden_idx % head_size

        is_prefix = token_idx < context_tokens
        safe_prefix_idx = tl.where(is_prefix, token_idx, 0)

        block_id = tl.load(context_block_ids_ptr + safe_prefix_idx,
                           mask=mask,
                           other=0)
        block_offset = tl.load(context_block_offsets_ptr + safe_prefix_idx,
                               mask=mask,
                               other=0)

        prefix_offsets = (
            block_id * stride_cache_block +
            kv_head * stride_cache_head +
            head_offset * stride_cache_dim +
            block_offset * stride_cache_token
        )

        query_token_idx = token_idx - context_tokens
        query_offsets = (
            query_token_idx * stride_query_token +
            kv_head * stride_query_head +
            head_offset * stride_query_dim
        )

        prefix_values = tl.load(value_cache_ptr + prefix_offsets, mask=mask)
        query_values = tl.load(query_value_ptr + query_offsets,
                               mask=mask & (~is_prefix))
        values = tl.where(is_prefix, prefix_values, query_values)

        dst_offsets = (
            token_idx * stride_out_token +
            kv_head * stride_out_head +
            head_offset * stride_out_dim
        )
        tl.store(out_ptr + dst_offsets, values, mask=mask)


@dataclass(frozen=True)
class _XFormersPrefixFallbackSegment:
    """描述一条请求在 fallback 中的 prefix/query 拼接区间。

    `xformers` fallback 当前仍然需要把 prefix KV 从 paged cache gather 成连续
    tensor，再和本轮 query 对应的新 K/V 拼成完整连续序列。

    这一步真正随层变化的数据只有：

    - 当前层的 `prefix_key/prefix_value`
    - 当前层的 `key/value`

    但“每个请求应该从 prefix 连续缓冲区取哪一段、再从 query 连续缓冲区取哪一段，
    以及它们在最终 full KV 序列里应该落到哪里”这一层控制面其实只和 metadata
    有关，和 layer 无关。因此把它提前收成 plan，后续每层只复用这份区间描述即可，
    避免重复做 Python 切片规划。
    """

    prefix_start: int
    prefix_end: int
    query_start: int
    query_end: int
    full_prefix_start: int
    full_prefix_end: int
    full_query_start: int
    full_query_end: int


@dataclass(frozen=True)
class _XFormersPrefixFallbackPlan:
    """prefix fallback 的跨层可复用控制面计划。

    这里刻意只缓存“稳定的控制面”，不缓存任何层相关的 KV 数据：

    - `query_lens/context_lens/kv_lens`
    - `cu_context_lens`
    - `attn_bias`
    - prefix/query 的拼接区间

    这样做的原因是：

    1. prefix KV 数据本身是逐层不同的，不能跨层直接复用；
    2. 但这些长度、区间和 bias 在同一个 prefill batch 内是稳定不变的，
       完全没必要每层都重新构造一遍。
    """

    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    segments: tuple[_XFormersPrefixFallbackSegment, ...]
    cu_context_lens: torch.Tensor
    attn_bias: AttentionBias
    total_context_tokens: int
    total_query_tokens: int
    total_kv_tokens: int
    context_block_ids: torch.Tensor
    context_block_offsets: torch.Tensor
    context_compact_positions: torch.Tensor
    context_full_positions: torch.Tensor
    query_full_positions: torch.Tensor


@dataclass(frozen=True)
class _XFormersPrefixFallbackProfile:
    """一次 xformers prefix fallback 的细粒度阶段统计。"""

    prefix_mode: str
    query_mode: str
    cache_view_ms: float
    gather_key_ms: float
    gather_value_ms: float
    prefix_copy_ms: float
    query_copy_ms: float
    xformers_forward_ms: float
    total_ms: float


@dataclass(frozen=True)
class _XFormersPrefixGatherProfile:
    """一次 prefix KV gather 的细粒度阶段统计。"""

    mode: str
    cache_view_ms: float
    gather_key_ms: float
    gather_value_ms: float


@dataclass(frozen=True)
class _XFormersPrefixWorkspaceBackendChoice:
    """描述当前 prefix fallback 选中的 workspace 组织 backend。"""

    prefix_backend: str
    query_backend: str


@dataclass
class _XFormersPrefixFallbackWorkspace:
    """prefix fallback 的跨层可复用 scratch buffer。

    当前 fallback 仍然需要把 prefix/query 组织成一段完整连续的 KV 序列后再交给
    xFormers。旧实现每层都会：

    1. 生成很多小 slice
    2. 用 Python list 收集
    3. 再 `torch.cat()` 出新的 `full_key/full_value`

    这会引入额外分配和拼接开销。这里把“最终 full KV 缓冲区”也缓存下来，
    每层只做原地填充，从而把控制面和数据面都收得更紧一些。
    """

    full_key: torch.Tensor
    full_value: torch.Tensor


def _copy_runtime_dense_prefix_attachment(
    *,
    src_metadata: "XFormersMetadata",
    dst_metadata: "XFormersMetadata",
) -> None:
    """把 runtime dense-prefix attachment 从父 metadata 传给派生 metadata。

    当前单请求 runtime fast path 会把“已经按旧两次搬运语义还原好的 prefix chunk
    tensors”挂在 `XFormersMetadata` 对象上，供 xformers fallback 直接消费。

    但 xformers forward 真正拿到的常常是 `attn_metadata.prefill_metadata`
    返回出来的派生对象，而不是最初那份父 metadata。因此如果这里不显式透传：

    - adapter 已经把 dense prefix 数据挂好了
    - 但进入 prefill 子 metadata 后这份 attachment 会丢失
    - xformers fallback 又会退回旧的 paged-cache gather 语义

    这样就达不到“把 paged-KV 消费问题隔离出去”的目标。

    这里刻意只复制当前 dense-prefix consume 主线真正需要的几个动态字段，
    不去泛化成“复制所有私有属性”，避免再次把历史实验缓存也一起耦回来。
    """
    for attr_name in (
            "_granulekv_dense_prefix_chunk_tensors",
            "_granulekv_dense_prefix_context_tokens",
            "_granulekv_dense_prefix_source",
    ):
        if hasattr(src_metadata, attr_name):
            setattr(dst_metadata, attr_name, getattr(src_metadata, attr_name))


class XFormersBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "XFORMERS"

    @staticmethod
    def get_impl_cls() -> Type["XFormersImpl"]:
        return XFormersImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return XFormersMetadata

    @staticmethod
    def get_builder_cls() -> Type["XFormersMetadataBuilder"]:
        return XFormersMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class XFormersMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for XFormersbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["XFormersMetadata"] = None
    _cached_decode_metadata: Optional["XFormersMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    encoder_seq_start_loc: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        self.encoder_attn_bias: Optional[List[AttentionBias]] = None
        self.cross_attn_bias: Optional[List[AttentionBias]] = None
        # prefix fallback 只在 V100/Turing 等当前无法稳定走 paged prefix
        # Triton kernel 的环境里使用。这里把“与层无关的控制面计划”缓存到
        # metadata 上，允许同一个 prefill batch 的所有 layer 复用。
        self._cached_prefix_fallback_plan: Optional[
            _XFormersPrefixFallbackPlan] = None
        self._cached_prefix_fallback_plan_key: Optional[Tuple[Any, ...]] = None
        # prefix fallback 的 scratch buffer 也按 prefill batch 缓存在 metadata 上。
        # 原因和 plan 一样：它只和本轮 batch 的稳定形状有关，和 layer 无关，
        # 因此同一批次所有层都可以复用，避免每层重新分配 full KV 缓冲区。
        self._cached_prefix_fallback_workspace: Optional[
            _XFormersPrefixFallbackWorkspace] = None
        self._cached_prefix_fallback_workspace_key: Optional[
            Tuple[Any, ...]] = None
        # 仅服务于 “sm<80 + no-alibi 时尝试借道 alibi kernel” 这个实验开关。
        # 这组全 0 slope 只和 head 数 / device / dtype 有关，因此可以和其它
        # prefix 控制面缓存一样，按当前 prefill batch 复用。
        self._cached_zero_alibi_slopes: Optional[torch.Tensor] = None
        self._cached_zero_alibi_slopes_key: Optional[Tuple[Any, ...]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional["XFormersMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = XFormersMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        _copy_runtime_dense_prefix_attachment(
            src_metadata=self,
            dst_metadata=self._cached_prefill_metadata,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["XFormersMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = XFormersMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)

        # Batch may be composed of prefill|decodes, adjust query start indices
        # to refer to the start of decodes when the two are split apart.
        # E.g. in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata


def _get_attn_bias(
    attn_metadata: XFormersMetadata,
    attn_type: str,
) -> Optional[AttentionBias]:
    '''
    Extract appropriate attention bias from attention metadata
    according to attention type.

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention

    Returns:
    * Appropriate attention bias value given the attention type
    '''

    if (attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY):
        return attn_metadata.attn_bias
    elif attn_type == AttentionType.ENCODER:
        return attn_metadata.encoder_attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        return attn_metadata.cross_attn_bias
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def _set_attn_bias(
    attn_metadata: XFormersMetadata,
    attn_bias: List[Optional[AttentionBias]],
    attn_type: str,
) -> None:
    '''
    Update appropriate attention bias field of attention metadata,
    according to attention type.

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention
    * attn_bias: The desired attention bias value
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention
    '''

    if (attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY):
        attn_metadata.attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER:
        attn_metadata.encoder_attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        attn_metadata.cross_attn_bias = attn_bias
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


class XFormersMetadataBuilder(CommonMetadataBuilder[XFormersMetadata]):

    _metadata_cls = XFormersMetadata


class XFormersImpl(AttentionImpl[XFormersMetadata]):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "XFormers does not support block-sparse attention.")
        if logits_soft_cap is not None:
            logger.warning_once("XFormers does not support logits soft cap. "
                                "Outputs may be slightly off.")
        if use_irope:
            logger.warning_once(
                "Using irope in XFormers is not supported yet, it will fall"
                " back to global attention for long context.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

        self.attn_type = attn_type

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: "XFormersMetadata",
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        For decoder-only models: query, key and value must be non-None.

        For encoder/decoder models:
        * XFormersImpl.forward() may be invoked for both self- and cross-
          attention layers.
        * For self-attention: query, key and value must be non-None.
        * For cross-attention:
            * Query must be non-None
            * During prefill, key and value must be non-None; key and value
              get cached for use during decode.
            * During decode, key and value may be None, since:
              (1) key and value tensors were cached during prefill, and
              (2) cross-attention key and value tensors do not grow during
                  decode
        
        A note on how the attn_type (attention type enum) argument impacts
        attention forward() behavior:
    
            * DECODER: normal decoder-only behavior;
                use decoder self-attention block table
            * ENCODER: no KV caching; pass encoder sequence
                attributes (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len) to kernel, in lieu of decoder
                sequence attributes (seq_lens/seq_lens_tensor/max_seq_len).
                Used for encoder branch of encoder-decoder models.
            * ENCODER_ONLY: no kv_caching, uses the normal attention 
                attributes (seq_lens/seq_lens_tensor/max_seq_len).
            * ENCODER_DECODER: cross-attention behavior;
                use cross-attention block table for caching KVs derived
                from encoder hidden states; since KV sequence lengths
                will match encoder sequence lengths, pass encoder sequence
                attributes to kernel (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len)
    
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        attn_type = self.attn_type
        # Check that appropriate attention metadata attributes are
        # selected for the desired attention type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")

        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        # Self-attention vs. cross-attention will impact
        # which KV cache memory-mapping & which
        # seqlen datastructures we utilize

        if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):

                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                PagedAttention.write_to_paged_cache(
                    key, value, key_cache, value_cache, updated_slot_mapping,
                    self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        (num_prefill_query_tokens, num_prefill_kv_tokens,
        num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        if key is not None and value is not None:
            key = key[:num_prefill_kv_tokens]
            value = value[:num_prefill_kv_tokens]

        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                # normal attention.
                # block tables are empty if the prompt does not have a cached
                # prefix.
                out = self._run_memory_efficient_xformers_forward(
                    query, key, value, prefill_meta, attn_type=attn_type)
                assert out.shape == output[:num_prefill_query_tokens].shape
                output[:num_prefill_query_tokens] = out
            else:
                assert attn_type != AttentionType.ENCODER_ONLY, (
                    "Encoder-only models should not have prefix attention.")

                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.max_query_len is not None

                # 这条日志在每一层 prefix attention 都会打印一次，长期打开会对
                # 主线性能和日志可读性都造成干扰。这里收敛成：
                #
                # - 平时默认不打
                # - 只有在显式打开 fallback profile 时，才一起输出这份上下文
                #
                # 这样既保留联调时需要的输入形状信息，又避免稳定跑分时每层都刷
                # 一条大日志。
                if envs.VLLM_GRANULEKV_XFORMERS_PREFIX_FALLBACK_PROFILE:
                    # 这里是 prefix fallback 的观测日志，不应该再去读取 CUDA tensor
                    # 的具体值。
                    #
                    # 原先这里直接打印：
                    # - `prefill_meta.query_start_loc.tolist()`
                    # - `prefill_meta.seq_lens_tensor.tolist()`
                    #
                    # 这会强制当前线程在 xFormers forward 入口做一次设备同步。
                    # 一旦上游某个异步 CUDA kernel 已经出错，异常就会在这里提前
                    # 暴露，看起来像“xformers 日志报错”，但实际上只是同步点把
                    # 更早的 fault 揭出来。
                    #
                    # 为了让调试信息继续可用、同时不再人为引入新的同步暴露点，
                    # 这里收敛成只打印 shape / Python 侧标量。
                    logger.info(
                        "[XFORMERS_PREFIX] query_shape=%s key_shape=%s "
                        "value_shape=%s block_tables_shape=%s "
                        "query_start_loc_shape=%s seq_lens=%s "
                        "seq_lens_tensor_shape=%s "
                        "num_prefills=%d num_prefill_tokens=%d "
                        "max_query_len=%s max_prefill_seq_len=%s",
                        tuple(query.shape),
                        None if key is None else tuple(key.shape),
                        None if value is None else tuple(value.shape),
                        tuple(prefill_meta.block_tables.shape),
                        tuple(prefill_meta.query_start_loc.shape),
                        prefill_meta.seq_lens,
                        None if prefill_meta.seq_lens_tensor is None else
                        tuple(prefill_meta.seq_lens_tensor.shape),
                        prefill_meta.num_prefills,
                        prefill_meta.num_prefill_tokens,
                        prefill_meta.max_query_len,
                        prefill_meta.max_prefill_seq_len,
                    )

                # V100 + V0 + xFormers 的 prefix kernel 在当前环境会触发
                # LLVM/layout 崩溃。
                #
                # 之前保留过一条 `zero_alibi` 实验分支，试图借道另一套 Triton
                # kernel；但这条路径在当前主线里已经确认没有稳定价值，只会额外
                # 增加 prefix 入口分叉。
                #
                # 因此这里直接收束为当前唯一仍保留的 V100 主线：
                # - 继续保留 prefix hit 与 paged-KV 语义
                # - attention 计算统一走稳定的 xFormers fallback
                device_capability = torch.cuda.get_device_capability(
                    query.device)
                if device_capability < (8, 0) and self.alibi_slopes is None:
                    out = self._run_prefix_attention_fallback(
                        layer=layer,
                        query=query,
                        key=key,
                        value=value,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        prefill_meta=prefill_meta,
                    )
                else:
                    # prefix-enabled attention
                    # TODO(Hai) this triton kernel has regression issue (broke)
                    # to deal with different data types between KV and FP8 KV
                    # cache, to be addressed separately.
                    out = PagedAttention.forward_prefix(
                        query,
                        key,
                        value,
                        self.kv_cache_dtype,
                        key_cache,
                        value_cache,
                        prefill_meta.block_tables,
                        prefill_meta.query_start_loc,
                        prefill_meta.seq_lens_tensor,
                        prefill_meta.max_query_len,
                        self.alibi_slopes,
                        self.sliding_window,
                        layer._k_scale,
                        layer._v_scale,
                    )
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata.")

            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = get_seq_len_block_table_args(decode_meta, False, attn_type)

            block_size = value_cache.shape[3]
            sequence_length = int(max_seq_len_arg)
            selected_attention = None
            selected_attention_seq_len = None
            active_sparse_blocks = get_active_sparse_kv_blocks()
            if (active_sparse_blocks is None
                    and envs.VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE):
                if block_tables_arg is None:
                    raise RuntimeError(
                        "resident sparse decode requires a block table")
                active_sparse_blocks = tuple(
                    range((int(sequence_length) + block_size - 1) // block_size))
            if active_sparse_blocks is not None:
                if attn_type != AttentionType.DECODER:
                    raise RuntimeError(
                        "sparse KV decode is only supported for decoder "
                        "self-attention")
                if attn_metadata.num_prefill_tokens != 0:
                    raise RuntimeError(
                        "sparse KV decode does not support mixed prefill/decode")
                if (decode_query.shape[0] != 1 or block_tables_arg is None
                        or block_tables_arg.shape[0] != 1):
                    raise RuntimeError(
                        "sparse KV decode requires batch size one")
                if decode_meta.use_cuda_graph:
                    raise RuntimeError(
                        "sparse KV decode does not support CUDA graph capture")
                if self.kv_cache_dtype.startswith("fp8"):
                    raise RuntimeError(
                        "sparse KV decode does not support FP8 KV cache")
                if (envs.VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET > 0
                        and get_sparse_kv_policy() is not None):
                    layer_name = getattr(layer, "layer_name", "")
                    layer_index = self._layer_index_from_name(layer_name)
                    request_ids = get_active_layer_request_ids()
                    if layer_index is None or len(request_ids) > 1:
                        raise RuntimeError(
                            "sparse policy selection requires one named layer "
                            "request")
                    request_id = (request_ids[0] if request_ids else "default")
                    policy = get_sparse_kv_policy()
                    gpu_selector = None
                    attention_selector = None
                    if (envs.VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE
                            and not envs.VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE
                            and envs.VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE
                            and not envs.VLLM_GRANULEKV_ENABLE
                            and policy is not None):
                        gpu_selector = getattr(policy, "select_blocks_device", None)
                        if envs.VLLM_GRANULEKV_SPARSE_SELECTED_BLOCKS_ENABLE:
                            attention_selector = getattr(
                                policy, "select_attention_blocks_device", None)
                    if callable(attention_selector):
                        selection = validate_sparse_kv_device_selection(
                            attention_selector(
                                request_id,
                                layer_index,
                                decode_query,
                                key_cache,
                                block_tables_arg[0, :((sequence_length +
                                                       block_size - 1) //
                                                      block_size)],
                                sequence_length,
                                block_size,
                                envs.VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET,
                            ))
                        tail_tokens = sequence_length - (
                            (sequence_length + block_size - 1) // block_size -
                            1) * block_size
                        selected_attention = selection
                        selected_attention_seq_len = (
                            selection.count * block_size -
                            (block_size - tail_tokens))
                        record_sparse_kv_attention(
                            (sequence_length + block_size - 1) // block_size,
                            selection.count,
                            selected_attention_seq_len)
                    else:
                        block_tables_arg, selected_seq_len, active_sparse_blocks = (
                            select_and_compact_decode_blocks(
                                block_table=block_tables_arg,
                                key_cache=key_cache,
                                query=decode_query,
                                sequence_length=sequence_length,
                                block_size=block_size,
                                request_id=request_id,
                                layer_index=layer_index,
                                block_budget=envs.VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET,
                                select_blocks=select_sparse_blocks,
                                register_page_representatives=(
                                    register_sparse_page_representatives),
                                select_blocks_device=gpu_selector,
                                resident_blocks=active_sparse_blocks,
                                selected_blocks_override=(
                                    active_sparse_blocks
                                    if (envs.VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE
                                        and active_sparse_blocks is not None) else
                                    None),
                            ))
                else:
                    block_tables_arg, selected_seq_len = (
                        build_selected_decode_block_table(
                            block_tables_arg,
                            active_sparse_blocks,
                            sequence_length,
                            block_size,
                        ))
                if selected_attention is None:
                    seq_lens_arg = torch.tensor(
                        [selected_seq_len],
                        dtype=seq_lens_arg.dtype,
                        device=seq_lens_arg.device,
                    )
                    max_seq_len_arg = selected_seq_len

            if selected_attention is not None:
                output[num_prefill_query_tokens:] = (
                    PagedAttention.forward_decode_selected(
                        decode_query,
                        key_cache,
                        value_cache,
                        block_tables_arg,
                        seq_lens_arg,
                        selected_attention.logical_block_indices,
                        selected_attention.count,
                        selected_attention_seq_len,
                        self.kv_cache_dtype,
                        self.num_kv_heads,
                        self.scale,
                        self.alibi_slopes,
                        layer._k_scale,
                        layer._v_scale,
                    ))
            else:
                output[num_prefill_query_tokens:] = PagedAttention.forward_decode(
                    decode_query,
                    key_cache,
                    value_cache,
                    block_tables_arg,
                    seq_lens_arg,
                    max_seq_len_arg,
                    self.kv_cache_dtype,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                    layer._k_scale,
                    layer._v_scale,
                )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    @staticmethod
    def _layer_index_from_name(layer_name: str) -> Optional[int]:
        parts = layer_name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part == "layers" and parts[index + 1].isdigit():
                return int(parts[index + 1])
        return None

    def _run_prefix_attention_fallback(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        prefill_meta: XFormersMetadata,
    ) -> torch.Tensor:
        """绕开不稳定的 prefix kernel，但保留 prefix hit 与 paged KV 语义。"""
        assert prefill_meta.block_tables is not None
        plan = self._get_prefix_fallback_plan(
            prefill_meta=prefill_meta,
            device=query.device,
            block_size=int(value_cache.shape[-1]),
        )

        workspace = self._get_prefix_fallback_workspace(
            prefill_meta=prefill_meta,
            plan=plan,
            key_dtype=key.dtype,
            value_dtype=value.dtype,
            device=query.device,
        )
        full_key = workspace.full_key
        full_value = workspace.full_value
        profile_enabled = envs.VLLM_GRANULEKV_XFORMERS_PREFIX_FALLBACK_PROFILE
        backend_choice = self._select_prefix_workspace_backends(
            layer=layer,
            prefill_meta=prefill_meta,
            plan=plan,
            key_cache=key_cache,
            value_cache=value_cache,
            query_key=key,
            query_value=value,
            full_key=full_key,
            full_value=full_value,
        )
        gather_profile, prefix_copy_ms = (
            self._materialize_prefix_kv_into_workspace(
                layer=layer,
                prefill_meta=prefill_meta,
                prefix_backend=backend_choice.prefix_backend,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=prefill_meta.block_tables,
                plan=plan,
                full_key=full_key,
                full_value=full_value,
                profile_enabled=profile_enabled,
            ))
        query_mode, query_copy_ms = self._materialize_query_kv_into_workspace(
            query_backend=backend_choice.query_backend,
            query_key=key,
            query_value=value,
            plan=plan,
            full_key=full_key,
            full_value=full_value,
            profile_enabled=profile_enabled,
        )

        original_query = query
        if self.num_kv_heads != self.num_heads:
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            full_key = full_key[:, :, None, :].expand(
                full_key.shape[0],
                self.num_kv_heads,
                self.num_queries_per_kv,
                full_key.shape[-1],
            )
            full_value = full_value[:, :, None, :].expand(
                full_value.shape[0],
                self.num_kv_heads,
                self.num_queries_per_kv,
                full_value.shape[-1],
            )

        # xFormers fallback 每层都会执行一次；默认性能口径不逐层打印，
        # 避免 Python logging 干扰端到端延迟。需要 profile 时由
        # VLLM_GRANULEKV_XFORMERS_PREFIX_FALLBACK_PROFILE=1 恢复细粒度日志。
        if profile_enabled:
            logger.info(
                "[XFORMERS_PREFIX_FALLBACK] query_lens=%s context_lens=%s "
                "kv_lens=%s total_query_tokens=%d total_kv_tokens=%d",
                list(plan.query_lens),
                list(plan.context_lens),
                list(plan.kv_lens),
                query.shape[0],
                plan.total_kv_tokens,
            )

        stage_start = None
        stage_end = None
        if profile_enabled:
            stage_start, stage_end = self._new_cuda_event_pair()
            stage_start.record()
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            full_key.unsqueeze(0),
            full_value.unsqueeze(0),
            attn_bias=plan.attn_bias,
            p=0.0,
            scale=self.scale,
        )
        self._verify_xformers_attention_output_against_reference(
            layer=layer,
            plan=plan,
            query=query,
            full_key=full_key,
            full_value=full_value,
            xformers_out=out,
        )
        if profile_enabled and stage_end is not None:
            stage_end.record()
            xformers_forward_ms = self._cuda_elapsed_ms(stage_start, stage_end)
            profile = _XFormersPrefixFallbackProfile(
                prefix_mode=gather_profile.mode,
                query_mode=query_mode,
                cache_view_ms=gather_profile.cache_view_ms,
                gather_key_ms=gather_profile.gather_key_ms,
                gather_value_ms=gather_profile.gather_value_ms,
                prefix_copy_ms=prefix_copy_ms,
                query_copy_ms=query_copy_ms,
                xformers_forward_ms=xformers_forward_ms,
                total_ms=(gather_profile.cache_view_ms +
                          gather_profile.gather_key_ms +
                          gather_profile.gather_value_ms +
                          prefix_copy_ms + query_copy_ms +
                          xformers_forward_ms),
            )
            logger.info(
                "[XFORMERS_PREFIX_FALLBACK_PROFILE] prefix_mode=%s "
                "query_mode=%s cache_view_ms=%.3f gather_key_ms=%.3f "
                "gather_value_ms=%.3f prefix_copy_ms=%.3f "
                "query_copy_ms=%.3f xformers_forward_ms=%.3f total_ms=%.3f",
                profile.prefix_mode,
                profile.query_mode,
                profile.cache_view_ms,
                profile.gather_key_ms,
                profile.gather_value_ms,
                profile.prefix_copy_ms,
                profile.query_copy_ms,
                profile.xformers_forward_ms,
                profile.total_ms,
            )
        return out.view_as(original_query)

    def _select_prefix_workspace_backends(
        self,
        *,
        layer: AttentionLayer,
        prefill_meta: XFormersMetadata,
        plan: _XFormersPrefixFallbackPlan,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query_key: torch.Tensor,
        query_value: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> _XFormersPrefixWorkspaceBackendChoice:
        """为当前 prefix fallback 选择显式 workspace backend。"""
        del layer, prefill_meta

        # 2026-07 当前主线明确收敛到：
        #
        #   GPU persistent service
        #     -> 直接把 prefix 写进最终 paged KV cache
        #   xFormers fallback
        #     -> 只从 paged KV cache 消费 prefix
        #
        # 也就是说，attention 侧的权威 prefix 来源应该是“最终 paged cache”，
        # 而不是另外再挂一份 dense chunk tensor attachment 让前台自行解释。
        #
        # 之前临时接入过 `dense_prefix_workspace_consume`，目的是把
        # “两次搬运旧语义”当作一个 correctness anchor 接回 attention。
        # 但真实端到端日志已经表明：
        #
        # - 路径确实走到了 dense consume
        # - request 不再卡住
        # - 但模型输出仍然错误
        #
        # 这说明问题已经收敛成“dense consume 这条前台解释语义与真实主线不一致”，
        # 而不是 GPU 后台 poll / direct placement 没跑通。
        #
        # 为了把主线重新收束到你要求的语义：
        #
        # - GranuleKV 负责存
        # - GPU service 负责取 + 放到 paged cache + 发布可消费状态
        # - attention 只消费 paged cache
        #
        # 这里直接停止默认选择 dense backend，统一退回 paged-cache consume
        # 逻辑。这样可以避免 attention 再额外依赖一套旁路 dense attachment
        # 解释，从而把“GPU 写什么 / attention 读什么”的数据面重新收成同一份
        # paged KV cache。
        forced_prefix_backend = envs.VLLM_GRANULEKV_XFORMERS_PREFIX_BACKEND
        if forced_prefix_backend not in (
                "auto",
                "packed_direct_to_workspace",
                "gather_then_copy",
        ):
            logger.warning(
                "[XFORMERS_PREFIX_BACKEND] invalid=%s fallback=auto",
                forced_prefix_backend,
            )
            forced_prefix_backend = "auto"

        # 这里保留一个很薄的诊断开关，用来把“paged cache 写入是否正确”
        # 和“xFormers 从 paged cache 读取是否正确”拆开验证：
        #
        # - auto / packed_direct_to_workspace：
        #     走当前主线的 Triton packed 直读路径，性能更好；
        # - gather_then_copy：
        #     强制回到 vLLM 已有 gather_cache 语义，再 copy 进 xFormers
        #     workspace。它更慢，但 ABI 更保守，适合判断 packed gather 是否读错。
        #
        # 注意：这个开关只影响 attention 侧如何“消费”已经写好的 paged KV cache，
        # 不改变 GranuleKV submit、GPU persistent poll、direct placement 写入语义。
        if forced_prefix_backend == "gather_then_copy":
            prefix_backend = "gather_then_copy"
        elif forced_prefix_backend == "packed_direct_to_workspace":
            prefix_backend = (
                "packed_direct_to_workspace"
                if self._can_use_packed_prefix_gather(key_cache, value_cache)
                else "gather_then_copy")
        else:
            prefix_backend = (
                "packed_direct_to_workspace"
                if self._can_use_packed_prefix_gather(key_cache, value_cache)
                else "gather_then_copy")
        forced_query_backend = envs.VLLM_GRANULEKV_XFORMERS_QUERY_BACKEND
        if forced_query_backend not in (
                "auto",
                "direct_scatter",
                "segment_copy",
        ):
            logger.warning(
                "[XFORMERS_PREFIX_QUERY_BACKEND] invalid=%s fallback=auto",
                forced_query_backend,
            )
            forced_query_backend = "auto"

        if forced_query_backend == "segment_copy":
            # 调试定位用的保守路径：完全绕过 Triton direct scatter，
            # 回到逐 segment `copy_()`。如果这条路径输出恢复，就说明错误在
            # direct scatter kernel 或 query_full_positions 映射；如果仍然错，
            # 根因就应继续往 xFormers bias / query slice / metadata 语义查。
            query_backend = "segment_copy"
        elif forced_query_backend == "direct_scatter":
            query_backend = (
                "direct_scatter" if self._can_use_direct_query_scatter(
                    query_key, query_value, full_key, full_value) else
                "segment_copy")
        else:
            query_backend = (
                "direct_scatter" if self._can_use_direct_query_scatter(
                    query_key, query_value, full_key, full_value) else
                "segment_copy")
        if envs.VLLM_GRANULEKV_XFORMERS_PREFIX_FALLBACK_PROFILE:
            logger.info(
                "[XFORMERS_PREFIX_BACKEND] forced=%s selected_prefix=%s "
                "forced_query=%s selected_query=%s",
                forced_prefix_backend,
                prefix_backend,
                forced_query_backend,
                query_backend,
            )
        return _XFormersPrefixWorkspaceBackendChoice(
            prefix_backend=prefix_backend,
            query_backend=query_backend,
        )

    def _materialize_prefix_kv_into_workspace(
        self,
        *,
        layer: AttentionLayer,
        prefill_meta: XFormersMetadata,
        prefix_backend: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        profile_enabled: bool,
    ) -> tuple[_XFormersPrefixGatherProfile, float]:
        """按显式 prefix backend，把 prefix KV 组织进 full workspace。"""
        if prefix_backend == "dense_prefix_workspace_consume":
            gather_profile = self._fill_prefix_kv_from_runtime_dense_prefix_into_full_buffer(
                layer=layer,
                prefill_meta=prefill_meta,
                plan=plan,
                full_key=full_key,
                full_value=full_value,
            )
            return gather_profile, 0.0
        if prefix_backend == "packed_direct_to_workspace":
            gather_profile = self._fill_prefix_kv_from_packed_cache_into_full_buffer(
                key_cache=key_cache,
                value_cache=value_cache,
                plan=plan,
                full_key=full_key,
                full_value=full_value,
                profile_enabled=profile_enabled,
            )
            return gather_profile, 0.0
        if prefix_backend == "gather_then_copy":
            prefix_key, prefix_value, gather_profile = (
                self._gather_prefix_kv_from_cache(
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_tables=block_tables,
                    plan=plan,
                    allow_packed_fast_path=False,
                ))
            self._verify_prefix_gather_against_dense_reference(
                layer=layer,
                prefill_meta=prefill_meta,
                plan=plan,
                prefix_key=prefix_key,
                prefix_value=prefix_value,
            )
            prefix_copy_start = prefix_copy_end = None
            prefix_copy_ms = 0.0
            if profile_enabled:
                prefix_copy_start, prefix_copy_end = self._new_cuda_event_pair()
                prefix_copy_start.record()
            self._copy_prefix_kv_into_full_buffer(
                prefix_key=prefix_key,
                prefix_value=prefix_value,
                plan=plan,
                full_key=full_key,
                full_value=full_value,
            )
            if profile_enabled and prefix_copy_end is not None:
                prefix_copy_end.record()
                prefix_copy_ms = self._cuda_elapsed_ms(prefix_copy_start,
                                                       prefix_copy_end)
            return gather_profile, prefix_copy_ms
        raise ValueError(f"Unknown prefix workspace backend: {prefix_backend}")

    def _materialize_query_kv_into_workspace(
        self,
        *,
        query_backend: str,
        query_key: torch.Tensor,
        query_value: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        profile_enabled: bool,
    ) -> tuple[str, float]:
        """按显式 query backend，把 query KV 组织进 full workspace。"""
        query_copy_start = query_copy_end = None
        if profile_enabled:
            query_copy_start, query_copy_end = self._new_cuda_event_pair()
            query_copy_start.record()

        if query_backend == "direct_scatter":
            self._scatter_query_kv_into_full_buffer_direct(
                query_key=query_key,
                query_value=query_value,
                plan=plan,
                full_key=full_key,
                full_value=full_value,
            )
        elif query_backend == "segment_copy":
            self._copy_query_kv_into_full_buffer_by_segments(
                query_key=query_key,
                query_value=query_value,
                plan=plan,
                full_key=full_key,
                full_value=full_value,
            )
        else:
            raise ValueError(f"Unknown query workspace backend: {query_backend}")

        query_copy_ms = 0.0
        if profile_enabled and query_copy_end is not None:
            query_copy_end.record()
            query_copy_ms = self._cuda_elapsed_ms(query_copy_start,
                                                  query_copy_end)
        return query_backend, query_copy_ms

    def _get_prefix_fallback_workspace(
        self,
        *,
        prefill_meta: XFormersMetadata,
        plan: _XFormersPrefixFallbackPlan,
        key_dtype: torch.dtype,
        value_dtype: torch.dtype,
        device: torch.device,
    ) -> _XFormersPrefixFallbackWorkspace:
        """按当前 prefill batch 的稳定形状复用 full KV scratch buffer。

        这里缓存的是“最终要交给 xFormers 的连续 full KV 缓冲区”。
        它的形状只取决于：

        - 当前 batch 的 `total_kv_tokens`
        - `num_kv_heads / head_size`
        - `dtype / device`

        因此完全可以跨层复用。后续每层只需要把 prefix/query 对应的数据原地填进
        去，而不必重新分配一份新的 full KV 张量。
        """
        workspace_key = (
            device.type,
            device.index,
            int(plan.total_kv_tokens),
            int(self.num_kv_heads),
            int(self.head_size),
            key_dtype,
            value_dtype,
        )
        cached_workspace = prefill_meta._cached_prefix_fallback_workspace
        if (cached_workspace is not None
                and prefill_meta._cached_prefix_fallback_workspace_key ==
                workspace_key):
            return cached_workspace

        workspace = _XFormersPrefixFallbackWorkspace(
            full_key=torch.empty(
                (plan.total_kv_tokens, self.num_kv_heads, self.head_size),
                dtype=key_dtype,
                device=device,
            ),
            full_value=torch.empty(
                (plan.total_kv_tokens, self.num_kv_heads, self.head_size),
                dtype=value_dtype,
                device=device,
            ),
        )
        prefill_meta._cached_prefix_fallback_workspace = workspace
        prefill_meta._cached_prefix_fallback_workspace_key = workspace_key
        return workspace

    def _get_runtime_dense_prefix_chunk_tensors(
        self,
        *,
        prefill_meta: XFormersMetadata,
    ) -> Optional[tuple[torch.Tensor, ...]]:
        """读取挂在 metadata 上的 dense prefix chunk tensors。

        这份 attachment 来自 adapter/storage 侧基于 live request pages 的
        materialize 结果，语义上等价于此前已经验证过的“两次搬运”旧路径：

        ```text
        GranuleKV pages
          -> chunk tensor [2, num_layers, tokens, hidden]
          -> xformers dense prefix workspace
        ```

        这里故意只做非常薄的只读解析：

        - 不做新的 GranuleKV 读取
        - 不做新的 paged-KV 解释
        - 只检查 attachment 是否存在、形状是否像期望的 chunk tensor
        """
        dense_prefix_chunk_tensors = getattr(
            prefill_meta,
            "_granulekv_dense_prefix_chunk_tensors",
            None,
        )
        if dense_prefix_chunk_tensors is None:
            return None
        tensors = tuple(dense_prefix_chunk_tensors)
        if not tensors:
            return None
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                return None
            if tensor.ndim != 4 or tensor.shape[0] != 2:
                return None
        return tensors

    def _try_get_runtime_dense_prefix_layer_index(
        self,
        *,
        layer: AttentionLayer,
        dense_prefix_chunk_tensors: tuple[torch.Tensor, ...],
    ) -> Optional[int]:
        """把当前 attention layer 解析成 dense prefix tensor 对应的 layer 下标。

        dense prefix attachment 目前保存的是完整 chunk tensor：

        ```text
        [2, num_layers, tokens, hidden]
        ```

        xformers fallback 在每一层 forward 时，只需要取当前层那一个
        `[2, tokens, hidden]` 切片。因此这里要把 `layer.layer_name`
        解析成稳定的 decoder layer index。

        当前主线模型（Qwen2 等）里的 attention layer name 形如：

        ```text
        model.layers.17.self_attn.attn
        ```

        因此只要定位 `.layers.` 后面的那段数字即可。若未来遇到别的命名方式，
        这里会保守返回 `None`，让 selector 自动退回旧 backend，而不会误读层号。
        """
        if not dense_prefix_chunk_tensors:
            return None
        num_layers = int(dense_prefix_chunk_tensors[0].shape[1])
        if num_layers <= 0:
            return None
        layer_name = getattr(layer, "layer_name", "")
        if ".layers." not in layer_name:
            return 0 if num_layers == 1 else None
        suffix = layer_name.split(".layers.", 1)[1]
        layer_index_text = suffix.split(".", 1)[0]
        if not layer_index_text.isdigit():
            return 0 if num_layers == 1 else None
        layer_index = int(layer_index_text)
        if layer_index < 0 or layer_index >= num_layers:
            return None
        return layer_index

    def _fill_prefix_kv_from_runtime_dense_prefix_into_full_buffer(
        self,
        *,
        layer: AttentionLayer,
        prefill_meta: XFormersMetadata,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> _XFormersPrefixGatherProfile:
        """直接消费已经 materialize 完成的 dense prefix chunk tensors。

        这条路径的定位非常明确：它就是把此前“已经验证能跑通”的两次搬运语义，
        显式收成 xformers fallback 的一条独立 consume backend。

        也就是说，这里不再依赖：

        - paged KV cache 当前是否已经被正确写好
        - packed key/value cache 的在线解释是否完全正确

        而是直接吃一份已经还原好的 dense chunk tensor：

        ```text
        [2, num_layers, tokens, hidden]
          -> 取当前 layer
          -> reshape 成 [tokens, num_kv_heads, head_size]
          -> 顺序写入 full workspace 的 prefix 段
        ```

        当前这条 backend 只服务“单请求 runtime fast path”主线，因此要求：

        - `plan.query_lens` 只有一个 request
        - dense prefix token 总数恰好等于当前 context_len

        一旦不满足，就直接报错，让上层尽快知道 attachment/build 阶段有问题，
        而不是静默混回 paged-KV 语义里继续产出错误结果。
        """
        dense_prefix_chunk_tensors = self._get_runtime_dense_prefix_chunk_tensors(
            prefill_meta=prefill_meta,
        )
        if dense_prefix_chunk_tensors is None:
            raise RuntimeError(
                "dense_prefix_workspace_consume selected without "
                "runtime dense prefix chunk tensors")
        if len(plan.query_lens) != 1:
            raise RuntimeError(
                "dense_prefix_workspace_consume currently requires "
                f"single-request plan, got {len(plan.query_lens)} requests")

        layer_index = self._try_get_runtime_dense_prefix_layer_index(
            layer=layer,
            dense_prefix_chunk_tensors=dense_prefix_chunk_tensors,
        )
        if layer_index is None:
            raise RuntimeError(
                "failed to resolve attention layer index for "
                f"dense_prefix_workspace_consume: layer_name={layer.layer_name}")

        expected_context_tokens = int(
            getattr(
                prefill_meta,
                "_granulekv_dense_prefix_context_tokens",
                plan.total_context_tokens,
            ))
        if expected_context_tokens != int(plan.total_context_tokens):
            raise RuntimeError(
                "runtime dense prefix token count mismatches plan context: "
                f"attachment={expected_context_tokens} "
                f"plan={int(plan.total_context_tokens)}")

        copied_tokens = 0
        for chunk_tensor in dense_prefix_chunk_tensors:
            layer_chunk = chunk_tensor[:, layer_index]
            if layer_chunk.shape[-1] != self.num_kv_heads * self.head_size:
                raise RuntimeError(
                    "runtime dense prefix hidden size mismatch: "
                    f"got={int(layer_chunk.shape[-1])} "
                    f"expected={int(self.num_kv_heads * self.head_size)}")
            chunk_tokens = int(layer_chunk.shape[1])
            chunk_key = layer_chunk[0].reshape(chunk_tokens, self.num_kv_heads,
                                               self.head_size)
            chunk_value = layer_chunk[1].reshape(chunk_tokens,
                                                 self.num_kv_heads,
                                                 self.head_size)
            full_key[copied_tokens:copied_tokens + chunk_tokens].copy_(
                chunk_key,
                non_blocking=False,
            )
            full_value[copied_tokens:copied_tokens + chunk_tokens].copy_(
                chunk_value,
                non_blocking=False,
            )
            copied_tokens += chunk_tokens

        if copied_tokens != expected_context_tokens:
            raise RuntimeError(
                "runtime dense prefix copied token count mismatch: "
                f"copied={copied_tokens} expected={expected_context_tokens}")

        return _XFormersPrefixGatherProfile(
            mode="dense_prefix_workspace_consume",
            cache_view_ms=0.0,
            gather_key_ms=0.0,
            gather_value_ms=0.0,
        )

    def _verify_prefix_gather_against_dense_reference(
        self,
        *,
        layer: AttentionLayer,
        prefill_meta: XFormersMetadata,
        plan: _XFormersPrefixFallbackPlan,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
    ) -> None:
        """抽样校验 xFormers 从 paged KV cache gather 出来的 prefix KV。

        当前端到端错误已经被收敛到最后消费段：

        ```text
        GranuleKV live pages 解码出的 dense chunk 是对的
          -> official write repair / runtime direct placement 写入 paged KV
          -> xFormers fallback 从 paged KV gather prefix
          -> full workspace + attention
        ```

        因此这里把 `prefix_key/prefix_value` 与 dense chunk reference 做少量
        token/head/dim 抽样比较。这个函数只在
        `VLLM_GRANULEKV_XFORMERS_VERIFY_PREFIX_GATHER=1` 时运行，并且只读现有
        attachment，不会发起新的 GranuleKV I/O，也不会改变正式数据通路。
        """
        if not envs.VLLM_GRANULEKV_XFORMERS_VERIFY_PREFIX_GATHER:
            return

        dense_prefix_chunk_tensors = self._get_runtime_dense_prefix_chunk_tensors(
            prefill_meta=prefill_meta,
        )
        if dense_prefix_chunk_tensors is None:
            logger.warning(
                "[XFORMERS_PREFIX_GATHER_VERIFY_SKIP] reason=no_dense_reference "
                "layer=%s total_context_tokens=%d",
                getattr(layer, "layer_name", "<unknown>"),
                int(plan.total_context_tokens),
            )
            return

        layer_index = self._try_get_runtime_dense_prefix_layer_index(
            layer=layer,
            dense_prefix_chunk_tensors=dense_prefix_chunk_tensors,
        )
        if layer_index is None:
            logger.warning(
                "[XFORMERS_PREFIX_GATHER_VERIFY_SKIP] "
                "reason=layer_index_unresolved layer=%s",
                getattr(layer, "layer_name", "<unknown>"),
            )
            return

        total_context_tokens = int(plan.total_context_tokens)
        if total_context_tokens <= 0:
            logger.info(
                "[XFORMERS_PREFIX_GATHER_VERIFY_OK] layer=%d samples=0 "
                "reason=empty_prefix",
                layer_index,
            )
            return
        if int(prefix_key.shape[0]) < total_context_tokens or int(
                prefix_value.shape[0]) < total_context_tokens:
            logger.error(
                "[XFORMERS_PREFIX_GATHER_VERIFY_FAIL] layer=%d "
                "reason=prefix_shape_too_small key_shape=%s value_shape=%s "
                "expected_tokens=%d",
                layer_index,
                tuple(prefix_key.shape),
                tuple(prefix_value.shape),
                total_context_tokens,
            )
            return

        chunk_token_ranges: list[tuple[int, int, torch.Tensor]] = []
        cursor = 0
        for chunk_tensor in dense_prefix_chunk_tensors:
            if layer_index >= int(chunk_tensor.shape[1]):
                logger.error(
                    "[XFORMERS_PREFIX_GATHER_VERIFY_FAIL] layer=%d "
                    "reason=dense_layer_out_of_range chunk_shape=%s",
                    layer_index,
                    tuple(chunk_tensor.shape),
                )
                return
            layer_chunk = chunk_tensor[:, layer_index]
            hidden = int(layer_chunk.shape[-1])
            expected_hidden = int(self.num_kv_heads * self.head_size)
            if hidden != expected_hidden:
                logger.error(
                    "[XFORMERS_PREFIX_GATHER_VERIFY_FAIL] layer=%d "
                    "reason=dense_hidden_mismatch got=%d expected=%d",
                    layer_index,
                    hidden,
                    expected_hidden,
                )
                return
            chunk_tokens = int(layer_chunk.shape[1])
            chunk_token_ranges.append((cursor, cursor + chunk_tokens,
                                       layer_chunk))
            cursor += chunk_tokens

        if cursor < total_context_tokens:
            logger.error(
                "[XFORMERS_PREFIX_GATHER_VERIFY_FAIL] layer=%d "
                "reason=dense_reference_too_short dense_tokens=%d "
                "plan_context_tokens=%d",
                layer_index,
                cursor,
                total_context_tokens,
            )
            return

        token_candidates = {
            0,
            1,
            2,
            127,
            128,
            total_context_tokens - 2,
            total_context_tokens - 1,
        }
        # chunk 边界最容易暴露 layout / block table 错位问题，额外抽样边界两侧。
        for start, end, _ in chunk_token_ranges:
            token_candidates.update((start, start + 1, end - 2, end - 1))
        token_samples = sorted(
            token for token in token_candidates
            if 0 <= token < total_context_tokens)
        head_samples = sorted({
            head
            for head in (0, 1, int(self.num_kv_heads) - 1)
            if 0 <= head < int(self.num_kv_heads)
        })
        dim_samples = sorted({
            dim
            for dim in (0, 1, 2, 3, 7, 15, 63, 127, int(self.head_size) - 1)
            if 0 <= dim < int(self.head_size)
        })

        failures: list[str] = []
        checked = 0
        max_abs_diff = 0.0
        chunk_cursor = 0
        for token in token_samples:
            while (chunk_cursor + 1 < len(chunk_token_ranges)
                   and token >= chunk_token_ranges[chunk_cursor][1]):
                chunk_cursor += 1
            chunk_start, chunk_end, layer_chunk = chunk_token_ranges[
                chunk_cursor]
            if not (chunk_start <= token < chunk_end):
                failures.append(
                    f"token={token} missing_dense_chunk range=({chunk_start},{chunk_end})"
                )
                continue
            local_token = token - chunk_start
            dense_key = layer_chunk[0].reshape(
                int(layer_chunk.shape[1]),
                int(self.num_kv_heads),
                int(self.head_size),
            )
            dense_value = layer_chunk[1].reshape(
                int(layer_chunk.shape[1]),
                int(self.num_kv_heads),
                int(self.head_size),
            )
            for head in head_samples:
                for dim in dim_samples:
                    expected_key = dense_key[local_token, head, dim]
                    actual_key = prefix_key[token, head, dim]
                    expected_value = dense_value[local_token, head, dim]
                    actual_value = prefix_value[token, head, dim]
                    key_diff = float(
                        torch.abs(actual_key.float() -
                                  expected_key.float()).item())
                    value_diff = float(
                        torch.abs(actual_value.float() -
                                  expected_value.float()).item())
                    max_abs_diff = max(max_abs_diff, key_diff, value_diff)
                    checked += 2
                    if key_diff > 0.0 and len(failures) < 8:
                        failures.append(
                            "kind=key token=%d head=%d dim=%d expected=%r actual=%r diff=%.6g"
                            % (
                                token,
                                head,
                                dim,
                                float(expected_key.float().item()),
                                float(actual_key.float().item()),
                                key_diff,
                            ))
                    if value_diff > 0.0 and len(failures) < 8:
                        failures.append(
                            "kind=value token=%d head=%d dim=%d expected=%r actual=%r diff=%.6g"
                            % (
                                token,
                                head,
                                dim,
                                float(expected_value.float().item()),
                                float(actual_value.float().item()),
                                value_diff,
                            ))

        if failures:
            logger.error(
                "[XFORMERS_PREFIX_GATHER_VERIFY_FAIL] layer=%d checked=%d "
                "max_abs_diff=%.6g first_failures=%s",
                layer_index,
                checked,
                max_abs_diff,
                " | ".join(failures),
            )
            return

        logger.info(
            "[XFORMERS_PREFIX_GATHER_VERIFY_OK] layer=%d checked=%d "
            "tokens=%s heads=%s dims=%s max_abs_diff=%.6g",
            layer_index,
            checked,
            token_samples,
            head_samples,
            dim_samples,
            max_abs_diff,
        )

    def _verify_xformers_attention_output_against_reference(
        self,
        *,
        layer: AttentionLayer,
        plan: _XFormersPrefixFallbackPlan,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        xformers_out: torch.Tensor,
    ) -> None:
        """用小样本 PyTorch attention 校验 xFormers fallback 最终输出。

        前一个 `prefix gather` 校验只能证明：

        ```text
        paged KV cache -> prefix_key/prefix_value
        ```

        是正确的。若模型输出仍然异常，剩下的疑点就在：

        ```text
        prefix KV + query KV -> full workspace
        attention bias / bottom-right causal mask
        xFormers memory_efficient_attention_forward
        ```

        这里直接用同一份 `query/full_key/full_value` 手写少量 reference：

        - 对每个 sampled query token，按 `context_len + local_query_idx`
          截断可见 KV；
        - 对少量 head 做 `softmax(q @ k * scale) @ v`；
        - 和 xFormers 输出逐 head 向量比较。

        这条逻辑只在
        `VLLM_GRANULEKV_XFORMERS_VERIFY_ATTENTION_OUTPUT=1` 时运行，不改变正式路径。
        """
        if not envs.VLLM_GRANULEKV_XFORMERS_VERIFY_ATTENTION_OUTPUT:
            return

        layer_index = self._try_parse_layer_index_for_logging(layer)
        query_flat = self._flatten_prefix_attention_tensor(query)
        key_flat = self._flatten_prefix_attention_tensor(full_key)
        value_flat = self._flatten_prefix_attention_tensor(full_value)
        out_tensor = xformers_out.squeeze(0)
        out_flat = self._flatten_prefix_attention_tensor(out_tensor)

        expected_query_shape = (
            int(plan.total_query_tokens),
            int(self.num_heads),
            int(self.head_size),
        )
        expected_kv_shape = (
            int(plan.total_kv_tokens),
            int(self.num_heads),
            int(self.head_size),
        )
        if tuple(query_flat.shape) != expected_query_shape:
            logger.error(
                "[XFORMERS_ATTENTION_REF_VERIFY_FAIL] layer=%s "
                "reason=query_shape_mismatch got=%s expected=%s",
                layer_index,
                tuple(query_flat.shape),
                expected_query_shape,
            )
            return
        if tuple(key_flat.shape) != expected_kv_shape or tuple(
                value_flat.shape) != expected_kv_shape:
            logger.error(
                "[XFORMERS_ATTENTION_REF_VERIFY_FAIL] layer=%s "
                "reason=kv_shape_mismatch key=%s value=%s expected=%s",
                layer_index,
                tuple(key_flat.shape),
                tuple(value_flat.shape),
                expected_kv_shape,
            )
            return
        if tuple(out_flat.shape) != expected_query_shape:
            logger.error(
                "[XFORMERS_ATTENTION_REF_VERIFY_FAIL] layer=%s "
                "reason=out_shape_mismatch got=%s expected=%s",
                layer_index,
                tuple(out_flat.shape),
                expected_query_shape,
            )
            return

        failures: list[str] = []
        checked = 0
        max_abs_diff = 0.0
        # 输出向量是 fp16 路径和 reference fp32 softmax 的比较，允许少量数值误差。
        atol = 5e-2
        rtol = 5e-2

        for segment in plan.segments:
            query_len = int(segment.query_end - segment.query_start)
            context_len = int(segment.prefix_end - segment.prefix_start)
            if query_len <= 0:
                continue
            query_offsets = sorted({
                offset
                for offset in (0, 1, query_len - 2, query_len - 1)
                if 0 <= offset < query_len
            })
            head_samples = sorted({
                head
                for head in (0, 1, int(self.num_heads) - 1)
                if 0 <= head < int(self.num_heads)
            })
            for local_query_offset in query_offsets:
                query_index = int(segment.query_start + local_query_offset)
                # BlockDiagonalMask.make_causal_from_bottomright() 对 prefix
                # prefill 的语义是：第 i 个 query token 可以看到全部 prefix，
                # 以及 query 段里直到自身为止的 K/V。
                allowed_kv_end = int(segment.full_prefix_start + context_len +
                                     local_query_offset + 1)
                for head in head_samples:
                    q_vec = query_flat[query_index, head].float()
                    k_mat = key_flat[
                        int(segment.full_prefix_start):allowed_kv_end,
                        head,
                    ].float()
                    v_mat = value_flat[
                        int(segment.full_prefix_start):allowed_kv_end,
                        head,
                    ].float()
                    scores = torch.matmul(k_mat, q_vec) * float(self.scale)
                    weights = torch.softmax(scores, dim=0)
                    reference = torch.matmul(weights, v_mat)
                    actual = out_flat[query_index, head].float()
                    diff_tensor = torch.abs(actual - reference)
                    token_max_diff = float(diff_tensor.max().item())
                    ref_norm = float(torch.abs(reference).max().item())
                    tolerance = atol + rtol * ref_norm
                    max_abs_diff = max(max_abs_diff, token_max_diff)
                    checked += 1
                    if token_max_diff > tolerance and len(failures) < 8:
                        max_dim = int(torch.argmax(diff_tensor).item())
                        failures.append(
                            "query=%d local=%d head=%d dim=%d "
                            "expected=%r actual=%r diff=%.6g tol=%.6g "
                            "visible_kv=%d" % (
                                query_index,
                                local_query_offset,
                                head,
                                max_dim,
                                float(reference[max_dim].item()),
                                float(actual[max_dim].item()),
                                token_max_diff,
                                tolerance,
                                allowed_kv_end -
                                int(segment.full_prefix_start),
                            ))

        if failures:
            logger.error(
                "[XFORMERS_ATTENTION_REF_VERIFY_FAIL] layer=%s checked=%d "
                "max_abs_diff=%.6g first_failures=%s",
                layer_index,
                checked,
                max_abs_diff,
                " | ".join(failures),
            )
            return

        logger.info(
            "[XFORMERS_ATTENTION_REF_VERIFY_OK] layer=%s checked=%d "
            "max_abs_diff=%.6g atol=%.3g rtol=%.3g",
            layer_index,
            checked,
            max_abs_diff,
            atol,
            rtol,
        )
        self._verify_xformers_attention_full_output_against_reference(
            layer_index=layer_index,
            plan=plan,
            query_flat=query_flat,
            key_flat=key_flat,
            value_flat=value_flat,
            out_flat=out_flat,
            atol=atol,
            rtol=rtol,
        )

    def _verify_xformers_attention_full_output_against_reference(
        self,
        *,
        layer_index: str,
        plan: _XFormersPrefixFallbackPlan,
        query_flat: torch.Tensor,
        key_flat: torch.Tensor,
        value_flat: torch.Tensor,
        out_flat: torch.Tensor,
        atol: float,
        rtol: float,
    ) -> None:
        """对指定层做完整 attention 输出 reference 对比。

        前面的 `XFORMERS_ATTENTION_REF_VERIFY_OK` 是低成本抽样，只能说明：

        ```text
        少量 query token / head 的 xFormers 输出
        与手写 bottom-right causal reference 一致
        ```

        现在 GranuleKV 读回、paged KV 写入、prefix gather 都已经通过全量/强抽样
        校验，若模型输出仍然乱码，就需要确认“整层 attention 输出”是否真的
        完整一致。

        这条诊断支线只在显式打开时运行：

        - `VLLM_GRANULEKV_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL=1`
        - `VLLM_GRANULEKV_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL_LAYER=N`

        默认只查第 0 层。实现上按 segment 构造 fp32 reference，避免继续把
        所有层都拉进昂贵的 debug 路径；它不会改变正式 forward 输出。
        """
        if not envs.VLLM_GRANULEKV_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL:
            return

        target_layer = int(
            envs.VLLM_GRANULEKV_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL_LAYER)
        if target_layer >= 0:
            try:
                current_layer = int(layer_index)
            except ValueError:
                current_layer = -999999
            if current_layer != target_layer:
                return

        max_abs_diff = 0.0
        worst_summary = ""
        compared_tokens = 0
        compared_heads = int(self.num_heads)
        compared_dims = int(self.head_size)

        for segment in plan.segments:
            query_len = int(segment.query_end - segment.query_start)
            context_len = int(segment.prefix_end - segment.prefix_start)
            kv_len = context_len + query_len
            if query_len <= 0 or kv_len <= 0:
                continue

            q_segment = query_flat[
                int(segment.query_start):int(segment.query_end)].float()
            k_segment = key_flat[
                int(segment.full_prefix_start):int(segment.full_query_end)
            ].float()
            v_segment = value_flat[
                int(segment.full_prefix_start):int(segment.full_query_end)
            ].float()
            actual_segment = out_flat[
                int(segment.query_start):int(segment.query_end)].float()

            # 组织成 [heads, query_tokens, dim] / [heads, kv_tokens, dim]，
            # 这样可以一次性算完整层所有 head 的 reference。
            q_by_head = q_segment.permute(1, 0, 2).contiguous()
            k_by_head = k_segment.permute(1, 0, 2).contiguous()
            v_by_head = v_segment.permute(1, 0, 2).contiguous()

            scores = torch.matmul(
                q_by_head,
                k_by_head.transpose(1, 2),
            ) * float(self.scale)
            query_offsets = torch.arange(
                query_len,
                device=scores.device,
                dtype=torch.int64,
            )
            kv_offsets = torch.arange(
                kv_len,
                device=scores.device,
                dtype=torch.int64,
            )
            # BlockDiagonalMask.make_causal_from_bottomright() 的 prefix
            # partial-prefill 语义：第 i 个 query 能看到全部 prefix，以及
            # query 段内 [0, i] 的 K/V。
            visible = kv_offsets.unsqueeze(0) <= (
                int(context_len) + query_offsets).unsqueeze(1)
            scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            reference_by_head = torch.matmul(weights, v_by_head)
            reference_segment = reference_by_head.permute(1, 0, 2).contiguous()

            diff_tensor = torch.abs(actual_segment - reference_segment)
            segment_max = float(diff_tensor.max().item())
            compared_tokens += query_len
            if segment_max > max_abs_diff:
                max_abs_diff = segment_max
                flat_index = int(torch.argmax(diff_tensor).item())
                token_idx = flat_index // (int(self.num_heads) *
                                           int(self.head_size))
                rem = flat_index % (int(self.num_heads) * int(self.head_size))
                head_idx = rem // int(self.head_size)
                dim_idx = rem % int(self.head_size)
                actual_value = float(
                    actual_segment[token_idx, head_idx, dim_idx].item())
                expected_value = float(
                    reference_segment[token_idx, head_idx, dim_idx].item())
                worst_summary = (
                    "segment_query_start=%d local_query=%d global_query=%d "
                    "head=%d dim=%d actual=%r expected=%r" % (
                        int(segment.query_start),
                        token_idx,
                        int(segment.query_start) + token_idx,
                        head_idx,
                        dim_idx,
                        actual_value,
                        expected_value,
                    ))

        tolerance = atol + rtol * max(
            1.0,
            float(torch.abs(out_flat.float()).max().item()),
        )
        if max_abs_diff > tolerance:
            logger.error(
                "[XFORMERS_ATTENTION_FULL_VERIFY_FAIL] layer=%s "
                "tokens=%d heads=%d dims=%d max_abs_diff=%.6g "
                "tolerance=%.6g %s",
                layer_index,
                compared_tokens,
                compared_heads,
                compared_dims,
                max_abs_diff,
                tolerance,
                worst_summary,
            )
            return

        logger.info(
            "[XFORMERS_ATTENTION_FULL_VERIFY_OK] layer=%s tokens=%d "
            "heads=%d dims=%d max_abs_diff=%.6g tolerance=%.6g",
            layer_index,
            compared_tokens,
            compared_heads,
            compared_dims,
            max_abs_diff,
            tolerance,
        )

    @staticmethod
    def _flatten_prefix_attention_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """把 xFormers GQA/MHA 张量统一看成 `[tokens, heads, head_size]`。

        当前 Qwen2.5-7B 在 xFormers fallback 中会把 GQA 表达成：

        ```text
        [tokens, num_kv_heads, num_queries_per_kv, head_size]
        ```

        而普通 MHA 则是：

        ```text
        [tokens, num_heads, head_size]
        ```

        reference 校验只关心“按最终 head 展开后的语义”，因此这里把两种形态
        收敛成同一个 token-major 视图。
        """
        if tensor.ndim == 4:
            return tensor.reshape(
                int(tensor.shape[0]),
                int(tensor.shape[1] * tensor.shape[2]),
                int(tensor.shape[3]),
            )
        if tensor.ndim == 3:
            return tensor
        raise RuntimeError(
            "unexpected prefix attention tensor shape for reference verify: "
            f"{tuple(tensor.shape)}")

    @staticmethod
    def _try_parse_layer_index_for_logging(layer: AttentionLayer) -> str:
        """尽量从 layer name 提取层号，仅用于诊断日志。"""
        layer_name = getattr(layer, "layer_name", "")
        if ".layers." not in layer_name:
            return layer_name or "<unknown>"
        suffix = layer_name.split(".layers.", 1)[1]
        layer_index_text = suffix.split(".", 1)[0]
        return layer_index_text if layer_index_text.isdigit() else layer_name

    def _copy_prefix_kv_into_full_buffer(
        self,
        *,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> None:
        """把连续 prefix KV 按 plan 中的 prefix 段落回 full KV buffer。

        这一步只在非 packed-direct gather 的兼容路径下需要。
        因为这条路径仍然先得到连续的 `prefix_key/prefix_value` 中间张量，
        所以还需要再做一次 prefix -> full buffer 的 copy。
        """
        for segment in plan.segments:
            if segment.full_prefix_end > segment.full_prefix_start:
                full_key[segment.full_prefix_start:segment.full_prefix_end].copy_(
                    prefix_key[segment.prefix_start:segment.prefix_end],
                    non_blocking=False,
                )
                full_value[
                    segment.full_prefix_start:segment.full_prefix_end].copy_(
                        prefix_value[segment.prefix_start:segment.prefix_end],
                        non_blocking=False,
                    )

    def _copy_query_kv_into_full_buffer_by_segments(
        self,
        *,
        query_key: torch.Tensor,
        query_value: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> None:
        """把本轮 query 对应的新 KV 写入 full KV buffer 的 query 段。

        这条 helper 只承载 `segment_copy` backend。
        更直接的 query 写入已经单独收成 `direct_scatter` backend。
        """
        for segment in plan.segments:
            if segment.full_query_end > segment.full_query_start:
                full_key[segment.full_query_start:segment.full_query_end].copy_(
                    query_key[segment.query_start:segment.query_end],
                    non_blocking=False,
                )
                full_value[
                    segment.full_query_start:segment.full_query_end].copy_(
                        query_value[segment.query_start:segment.query_end],
                        non_blocking=False,
                    )

    def _scatter_query_kv_into_full_buffer_direct(
        self,
        *,
        query_key: torch.Tensor,
        query_value: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> None:
        """把 query KV 直接 scatter 到最终 full KV buffer。

        这一步的设计目标与 prefix 侧 `packed_direct_to_workspace` 完全一致：

        - 不再按 segment 在 Python 层逐段 `copy_`
        - 直接利用 plan 中预先算好的 `query_full_positions`
        - 让 query 侧也收敛成“GPU resident 映射 + 单次 kernel 写入”

        这样后面如果要把 xformers fallback 进一步接到 persistent kernel /
        service CTA，就不需要再把 query 写入这一步重新拆一遍。
        """
        if plan.total_query_tokens == 0:
            return

        total_elements = int(plan.total_query_tokens * self.num_kv_heads *
                             self.head_size)
        block = 256

        _scatter_contiguous_kv_tokens_kernel[(triton.cdiv(total_elements,
                                                          block), )](
            query_key,
            plan.query_full_positions,
            full_key,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            stride_src_token=query_key.stride(0),
            stride_src_head=query_key.stride(1),
            stride_src_dim=query_key.stride(2),
            stride_out_token=full_key.stride(0),
            stride_out_head=full_key.stride(1),
            stride_out_dim=full_key.stride(2),
            BLOCK_SIZE=block,
        )
        _scatter_contiguous_kv_tokens_kernel[(triton.cdiv(total_elements,
                                                          block), )](
            query_value,
            plan.query_full_positions,
            full_value,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            stride_src_token=query_value.stride(0),
            stride_src_head=query_value.stride(1),
            stride_src_dim=query_value.stride(2),
            stride_out_token=full_value.stride(0),
            stride_out_head=full_value.stride(1),
            stride_out_dim=full_value.stride(2),
            BLOCK_SIZE=block,
        )

    def _get_prefix_fallback_plan(
        self,
        *,
        prefill_meta: XFormersMetadata,
        device: torch.device,
        block_size: int,
    ) -> _XFormersPrefixFallbackPlan:
        """构造并缓存 prefix fallback 的跨层控制面计划。

        当前 V100 路线下，每一层都会走一次 `_run_prefix_attention_fallback()`。
        如果不把控制面收起来，每层都会重复做：

        - 解析 `query_start_loc/context_lens_tensor`
        - 构造 `kv_lens`
        - 生成 `cu_context_lens`
        - 构造 `BlockDiagonalMask`
        - 规划 prefix/query 拼接区间

        这些步骤都不依赖层内 KV 数据，因此完全可以按 prefill batch 只做一次。
        """
        assert prefill_meta.context_lens_tensor is not None
        assert prefill_meta.query_start_loc is not None

        # 单请求 prefix fallback 是当前最主流、也最需要稳定的热路径。
        #
        # 这里优先直接复用 Python 侧已经现成可得的标量语义：
        # - `num_prefill_tokens`
        # - `seq_lens`
        #
        # 避免再对 CUDA 上的 `query_start_loc/context_lens_tensor` 做 `.tolist()`，
        # 否则会在 xformers forward 入口平白引入一次设备同步，把更早的异步 fault
        # 提前暴露成“看起来像是这行 Python 报错”。
        if prefill_meta.num_prefills == 1 and len(prefill_meta.seq_lens) == 1:
            query_lens = (int(prefill_meta.num_prefill_tokens), )
            context_lens = (
                int(prefill_meta.seq_lens[0]) - int(prefill_meta.num_prefill_tokens),
            )
        else:
            query_lens = tuple(
                int(x) for x in (
                    prefill_meta.query_start_loc[1:] -
                    prefill_meta.query_start_loc[:-1]).tolist())
            context_lens = tuple(
                int(x) for x in prefill_meta.context_lens_tensor.tolist())

        device_key = (device.type, device.index)
        plan_key = (
            device_key,
            self.sliding_window,
            int(block_size),
            int(prefill_meta.num_prefills),
            query_lens,
            context_lens,
        )
        cached_plan = prefill_meta._cached_prefix_fallback_plan
        if (cached_plan is not None
                and prefill_meta._cached_prefix_fallback_plan_key == plan_key):
            return cached_plan

        kv_lens = tuple(
            context_len + query_len
            for context_len, query_len in zip(context_lens, query_lens))
        total_context_tokens = int(sum(context_lens))
        total_query_tokens = int(sum(query_lens))

        # 注意：`cu_context_lens` 必须和上面已经确定的 `context_lens`
        # 使用同一套语义。
        #
        # 这点对当前 GranuleKV + LMCache rebuild 路径尤其重要：为了避免 CUDA 同步，
        # 单请求热路径会优先从 Python 侧 `seq_lens / num_prefill_tokens`
        # 推出 `context_lens`。如果这里又回头对
        # `prefill_meta.context_lens_tensor` 做 cumsum，就可能出现：
        #
        #   plan 日志显示 context_lens=[1024]
        #   gather_cache 实际按 context_lens_tensor 的旧值/异步值读
        #
        # 这种不一致不会影响 packed-direct/compose 路径，却会让
        # `gather_then_copy` 读错 prefix，最终表现为模型输出乱码。因此这里直接
        # 从已经规范化好的 Python tuple 构造 GPU 上的 cu-seqlens，让 plan 中
        # 所有 prefix 长度字段保持单一权威来源。
        cu_context_lens_cpu = [0]
        running_context_tokens = 0
        for context_len in context_lens:
            running_context_tokens += int(context_len)
            cu_context_lens_cpu.append(running_context_tokens)
        cu_context_lens = torch.tensor(
            cu_context_lens_cpu,
            dtype=torch.int32,
            device=device,
        )

        assert prefill_meta.block_tables is not None
        # 这里把“prefix 第 i 个 token 对应哪个 physical block / block 内 offset”
        # 直接预展开成两张小表。这样后面的 Triton gather kernel 就不需要再做：
        # - batch 维 binary search
        # - block_table 行寻址
        # - token -> block/token_offset 的重复推导
        #
        # 这些映射只和当前 prefill metadata 有关，和 layer 无关，因此放进
        # fallback plan 里跨层复用最合适。
        # prefix fallback 当前真实热路径大多是：
        #
        # - 单个 prefill request
        # - 一段连续 prefix hit
        #
        # 对这类场景，如果先把整张 block table 搬回 CPU，再按 token 做 Python
        # for-loop 去展开 block_id / block_offset / full_position，控制面会平白
        # 多一次 host roundtrip。
        #
        # 因此这里对“单请求”主场景做一个专门 fast path：
        #
        # - token offset 直接在 GPU 上用 `torch.arange` 生成
        # - block_id 直接在 GPU 上从 `block_tables[0]` gather
        # - compact/full position 也直接用向量方式生成
        #
        # 多请求场景仍保留保守的 CPU 展开逻辑，避免为了当前主线之外的情形把代
        # 码复杂度抬得太高。
        if prefill_meta.num_prefills == 1 and total_context_tokens > 0:
            single_context_len = int(context_lens[0])
            token_offsets = torch.arange(
                single_context_len,
                dtype=torch.int32,
                device=device,
            )
            block_indices = torch.div(
                token_offsets,
                int(block_size),
                rounding_mode="floor",
            ).to(torch.long)
            block_row = prefill_meta.block_tables[0].to(
                device=device,
                dtype=torch.int32,
            )
            context_block_ids = block_row.index_select(
                0, block_indices).contiguous()
            context_block_offsets = torch.remainder(
                token_offsets, int(block_size)).contiguous()
            context_compact_positions = torch.arange(
                single_context_len,
                dtype=torch.int32,
                device=device,
            )
            # 单请求下，prefix 总是落在 full KV 的起点，因此 full position
            # 与 token offset 相同。
            context_full_positions = token_offsets.contiguous()
        else:
            context_block_ids_list: list[int] = []
            context_block_offsets_list: list[int] = []
            context_compact_positions_list: list[int] = []
            context_full_positions_list: list[int] = []
            block_tables_cpu = prefill_meta.block_tables.to("cpu")
            full_kv_cursor = 0
            for request_idx, (context_len, query_len) in enumerate(
                    zip(context_lens, query_lens)):
                if context_len <= 0:
                    full_kv_cursor += query_len
                    continue
                block_row = block_tables_cpu[request_idx]
                for token_offset in range(context_len):
                    block_index = token_offset // int(block_size)
                    context_block_ids_list.append(
                        int(block_row[block_index].item()))
                    context_block_offsets_list.append(
                        token_offset % int(block_size))
                    context_compact_positions_list.append(
                        len(context_compact_positions_list))
                    context_full_positions_list.append(full_kv_cursor +
                                                       token_offset)
                full_kv_cursor += context_len + query_len

            context_block_ids = torch.tensor(
                context_block_ids_list,
                dtype=torch.int32,
                device=device,
            )
            context_block_offsets = torch.tensor(
                context_block_offsets_list,
                dtype=torch.int32,
                device=device,
            )
            context_compact_positions = torch.tensor(
                context_compact_positions_list,
                dtype=torch.int32,
                device=device,
            )
            context_full_positions = torch.tensor(
                context_full_positions_list,
                dtype=torch.int32,
                device=device,
            )

        # query 侧也显式收成 GPU resident 映射。
        #
        # 当前 `query_key/query_value` 本身已经是连续张量，因此不需要像 prefix
        # 那样再额外保存 block_id / block_offset；但我们仍然需要一张稳定的小表，
        # 表达：
        #
        #   第 i 个 query token
        #     -> 在最终 full KV 序列里应该落到哪个 token 位置
        #
        # 后续无论是：
        # - 当前这版 xformers fallback 的 direct scatter
        # - 还是再下一步 persistent kernel / service CTA
        #
        # 都可以直接复用这张映射，而不必每层再按 segment 重新做 Python 切片。
        query_full_positions_list: list[int] = []

        segments: list[_XFormersPrefixFallbackSegment] = []
        prefix_cursor = 0
        query_cursor = 0
        full_kv_cursor = 0
        for context_len, query_len in zip(context_lens, query_lens):
            full_prefix_start = full_kv_cursor
            full_prefix_end = full_prefix_start + context_len
            full_query_start = full_prefix_end
            full_query_end = full_query_start + query_len
            segment = _XFormersPrefixFallbackSegment(
                prefix_start=prefix_cursor,
                prefix_end=prefix_cursor + context_len,
                query_start=query_cursor,
                query_end=query_cursor + query_len,
                full_prefix_start=full_prefix_start,
                full_prefix_end=full_prefix_end,
                full_query_start=full_query_start,
                full_query_end=full_query_end,
            )
            segments.append(segment)
            if query_len > 0:
                query_full_positions_list.extend(
                    range(full_query_start, full_query_end))
            prefix_cursor += context_len
            query_cursor += query_len
            full_kv_cursor = full_query_end

        query_full_positions = torch.tensor(
            query_full_positions_list,
            dtype=torch.int32,
            device=device,
        )

        attn_bias = BlockDiagonalMask.from_seqlens(
            q_seqlen=list(query_lens),
            kv_seqlen=list(kv_lens),
            device=device,
        ).make_causal_from_bottomright()
        if self.sliding_window is not None:
            attn_bias = attn_bias.make_local_attention_from_bottomright(
                self.sliding_window)

        plan = _XFormersPrefixFallbackPlan(
            query_lens=query_lens,
            context_lens=context_lens,
            kv_lens=kv_lens,
            segments=tuple(segments),
            cu_context_lens=cu_context_lens,
            attn_bias=attn_bias,
            total_context_tokens=total_context_tokens,
            total_query_tokens=total_query_tokens,
            total_kv_tokens=full_kv_cursor,
            context_block_ids=context_block_ids,
            context_block_offsets=context_block_offsets,
            context_compact_positions=context_compact_positions,
            context_full_positions=context_full_positions,
            query_full_positions=query_full_positions,
        )
        prefill_meta._cached_prefix_fallback_plan = plan
        prefill_meta._cached_prefix_fallback_plan_key = plan_key
        # plan build 通常每个 prefill 只打印一次，但仍属于 xFormers fallback
        # 诊断信息；默认关闭，避免 performance run 的日志口径混入 profile。
        if envs.VLLM_GRANULEKV_XFORMERS_PREFIX_FALLBACK_PROFILE:
            logger.info(
                "[XFORMERS_PREFIX_FALLBACK_PLAN_BUILD] num_prefills=%d "
                "query_lens=%s context_lens=%s total_context_tokens=%d",
                prefill_meta.num_prefills,
                list(query_lens),
                list(context_lens),
                total_context_tokens,
            )
        return plan

    def _gather_prefix_kv_from_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        allow_packed_fast_path: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, _XFormersPrefixGatherProfile]:
        """把 prefix 命中的 paged KV 真正 gather 回连续张量。"""
        profile_enabled = envs.VLLM_GRANULEKV_XFORMERS_PREFIX_FALLBACK_PROFILE
        block_size = int(value_cache.shape[-1])

        # 这里的 `allow_packed_fast_path` 很重要：
        #
        # - auto / 性能主线允许它为 True，优先用我们自己的 packed direct gather；
        # - 但当外部显式设置
        #     VLLM_GRANULEKV_XFORMERS_PREFIX_BACKEND=gather_then_copy
        #   时，调用方要传 False，强制走 vLLM 既有 `gather_cache` 语义。
        #
        # 否则会出现日志里 selected_prefix=gather_then_copy，但 helper 内部又
        # 悄悄切回 packed_direct 的假诊断，无法判断错误到底在 paged cache 写端
        # 还是 packed gather 读端。
        if (allow_packed_fast_path
                and self._can_use_packed_prefix_gather(key_cache, value_cache)):
            return self._gather_prefix_kv_from_packed_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                plan=plan,
                profile_enabled=profile_enabled,
            )

        cache_view_start = cache_view_end = None
        if profile_enabled:
            cache_view_start, cache_view_end = self._new_cuda_event_pair()
            cache_view_start.record()
        key_src_cache = key_cache.permute(0, 3, 1, 2, 4).contiguous().view(
            key_cache.shape[0],
            block_size,
            self.num_kv_heads,
            self.head_size,
        )
        value_src_cache = value_cache.permute(0, 3, 1, 2).contiguous()
        if profile_enabled and cache_view_end is not None:
            cache_view_end.record()

        gathered_key = torch.empty(
            (plan.total_context_tokens, self.num_kv_heads, self.head_size),
            dtype=key_src_cache.dtype,
            device=key_src_cache.device,
        )
        gathered_value = torch.empty(
            (plan.total_context_tokens, self.num_kv_heads, self.head_size),
            dtype=value_src_cache.dtype,
            device=value_src_cache.device,
        )

        if plan.total_context_tokens == 0:
            cache_view_ms = 0.0
            if (profile_enabled and cache_view_start is not None
                    and cache_view_end is not None):
                cache_view_ms = self._cuda_elapsed_ms(cache_view_start,
                                                      cache_view_end)
            return gathered_key, gathered_value, _XFormersPrefixGatherProfile(
                mode="view_plus_gather",
                cache_view_ms=cache_view_ms,
                gather_key_ms=0.0,
                gather_value_ms=0.0,
            )

        gather_key_start = gather_key_end = None
        if profile_enabled:
            gather_key_start, gather_key_end = self._new_cuda_event_pair()
            gather_key_start.record()
        ops.gather_cache(
            src_cache=key_src_cache,
            dst=gathered_key,
            block_table=block_tables,
            cu_seq_lens=plan.cu_context_lens,
            batch_size=len(plan.query_lens),
        )
        if profile_enabled and gather_key_end is not None:
            gather_key_end.record()

        gather_value_start = gather_value_end = None
        if profile_enabled:
            gather_value_start, gather_value_end = self._new_cuda_event_pair()
            gather_value_start.record()
        ops.gather_cache(
            src_cache=value_src_cache,
            dst=gathered_value,
            block_table=block_tables,
            cu_seq_lens=plan.cu_context_lens,
            batch_size=len(plan.query_lens),
        )
        cache_view_ms = 0.0
        gather_key_ms = 0.0
        gather_value_ms = 0.0
        if profile_enabled and gather_value_end is not None:
            gather_value_end.record()
            cache_view_ms = self._cuda_elapsed_ms(cache_view_start,
                                                  cache_view_end)
            gather_key_ms = self._cuda_elapsed_ms(gather_key_start,
                                                  gather_key_end)
            gather_value_ms = self._cuda_elapsed_ms(gather_value_start,
                                                    gather_value_end)
        return gathered_key, gathered_value, _XFormersPrefixGatherProfile(
            mode="view_plus_gather",
            cache_view_ms=cache_view_ms,
            gather_key_ms=gather_key_ms,
            gather_value_ms=gather_value_ms,
        )

    def _gather_prefix_kv_from_packed_cache(
        self,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        profile_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, _XFormersPrefixGatherProfile]:
        """直接从 packed paged cache gather prefix KV，绕开中间 cache_view 拷贝。

        这条路径的目标非常明确：

        - 不是改变 fallback 的高层语义
        - 只是把 `permute(...).contiguous()` 这层布局重排去掉

        因此输出仍然保持：

        ```text
        gathered_{key,value}: [total_context_tokens, num_kv_heads, head_size]
        ```

        上层 `concat + xformers_forward` 完全不需要改。
        """
        gathered_key = torch.empty(
            (plan.total_context_tokens, self.num_kv_heads, self.head_size),
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
        gathered_value = torch.empty(
            (plan.total_context_tokens, self.num_kv_heads, self.head_size),
            dtype=value_cache.dtype,
            device=value_cache.device,
        )
        if plan.total_context_tokens == 0:
            return gathered_key, gathered_value, _XFormersPrefixGatherProfile(
                mode="packed_direct",
                cache_view_ms=0.0,
                gather_key_ms=0.0,
                gather_value_ms=0.0,
            )

        total_elements = int(plan.total_context_tokens * self.num_kv_heads *
                             self.head_size)
        block = 256
        key_start = key_end = None
        value_start = value_end = None
        if profile_enabled:
            key_start, key_end = self._new_cuda_event_pair()
            key_start.record()
        _gather_packed_key_cache_kernel[(triton.cdiv(total_elements, block), )](
            key_cache,
            plan.context_block_ids,
            plan.context_block_offsets,
            plan.context_compact_positions,
            gathered_key,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            pack_size=int(key_cache.shape[-1]),
            stride_cache_block=key_cache.stride(0),
            stride_cache_head=key_cache.stride(1),
            stride_cache_d_outer=key_cache.stride(2),
            stride_cache_token=key_cache.stride(3),
            stride_cache_pack=key_cache.stride(4),
            stride_out_token=gathered_key.stride(0),
            stride_out_head=gathered_key.stride(1),
            stride_out_dim=gathered_key.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and key_end is not None:
            key_end.record()

        if profile_enabled:
            value_start, value_end = self._new_cuda_event_pair()
            value_start.record()
        _gather_packed_value_cache_kernel[(triton.cdiv(total_elements, block), )](
            value_cache,
            plan.context_block_ids,
            plan.context_block_offsets,
            plan.context_compact_positions,
            gathered_value,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            stride_cache_block=value_cache.stride(0),
            stride_cache_head=value_cache.stride(1),
            stride_cache_dim=value_cache.stride(2),
            stride_cache_token=value_cache.stride(3),
            stride_out_token=gathered_value.stride(0),
            stride_out_head=gathered_value.stride(1),
            stride_out_dim=gathered_value.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and value_end is not None:
            value_end.record()

        gather_key_ms = 0.0
        gather_value_ms = 0.0
        if (profile_enabled and key_start is not None and key_end is not None
                and value_start is not None and value_end is not None):
            gather_key_ms = self._cuda_elapsed_ms(key_start, key_end)
            gather_value_ms = self._cuda_elapsed_ms(value_start, value_end)
        return gathered_key, gathered_value, _XFormersPrefixGatherProfile(
            mode="packed_direct",
            cache_view_ms=0.0,
            gather_key_ms=gather_key_ms,
            gather_value_ms=gather_value_ms,
        )

    def _fill_prefix_kv_from_packed_cache_into_full_buffer(
        self,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        profile_enabled: bool,
    ) -> _XFormersPrefixGatherProfile:
        """把 packed paged cache 中的 prefix KV 直接填入最终 full KV buffer。

        这是当前主热路径真正想要的形态：

        - 不再先生成 `gathered_key/gathered_value`
        - 也不再把 prefix 从中间张量二次 copy 到 workspace
        - Triton gather kernel 直接根据 `context_full_positions`
          把每个 prefix token scatter 到 full KV 中正确的位置

        这样 prefix 侧就只保留“一次从 paged cache 取数并写入最终目标”的必要搬运。
        """
        if plan.total_context_tokens == 0:
            return _XFormersPrefixGatherProfile(
                mode="packed_direct_to_workspace",
                cache_view_ms=0.0,
                gather_key_ms=0.0,
                gather_value_ms=0.0,
            )

        total_elements = int(plan.total_context_tokens * self.num_kv_heads *
                             self.head_size)
        block = 256
        key_start = key_end = None
        value_start = value_end = None
        if profile_enabled:
            key_start, key_end = self._new_cuda_event_pair()
            key_start.record()
        _gather_packed_key_cache_kernel[(triton.cdiv(total_elements, block), )](
            key_cache,
            plan.context_block_ids,
            plan.context_block_offsets,
            plan.context_full_positions,
            full_key,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            pack_size=int(key_cache.shape[-1]),
            stride_cache_block=key_cache.stride(0),
            stride_cache_head=key_cache.stride(1),
            stride_cache_d_outer=key_cache.stride(2),
            stride_cache_token=key_cache.stride(3),
            stride_cache_pack=key_cache.stride(4),
            stride_out_token=full_key.stride(0),
            stride_out_head=full_key.stride(1),
            stride_out_dim=full_key.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and key_end is not None:
            key_end.record()

        if profile_enabled:
            value_start, value_end = self._new_cuda_event_pair()
            value_start.record()
        _gather_packed_value_cache_kernel[(triton.cdiv(total_elements, block), )](
            value_cache,
            plan.context_block_ids,
            plan.context_block_offsets,
            plan.context_full_positions,
            full_value,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            stride_cache_block=value_cache.stride(0),
            stride_cache_head=value_cache.stride(1),
            stride_cache_dim=value_cache.stride(2),
            stride_cache_token=value_cache.stride(3),
            stride_out_token=full_value.stride(0),
            stride_out_head=full_value.stride(1),
            stride_out_dim=full_value.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and value_end is not None:
            value_end.record()

        gather_key_ms = 0.0
        gather_value_ms = 0.0
        if (profile_enabled and key_start is not None and key_end is not None
                and value_start is not None and value_end is not None):
            gather_key_ms = self._cuda_elapsed_ms(key_start, key_end)
            gather_value_ms = self._cuda_elapsed_ms(value_start, value_end)
        return _XFormersPrefixGatherProfile(
            mode="packed_direct_to_workspace",
            cache_view_ms=0.0,
            gather_key_ms=gather_key_ms,
            gather_value_ms=gather_value_ms,
        )

    def _compose_single_request_packed_kv_into_full_buffer(
        self,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query_key: torch.Tensor,
        query_value: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        profile_enabled: bool,
    ) -> _XFormersPrefixGatherProfile:
        """单请求场景下，把 prefix/query KV 一次性 compose 到最终 full buffer。

        这条路径是对当前主实验口径做的进一步专门化：

        - request 数固定为 1
        - prefix 在 full KV 开头连续铺开
        - query 在 full KV 尾部连续追加

        因此我们不必再分成：

        1. packed prefix gather
        2. query scatter

        而是可以直接按“最终 full token 位置”遍历，把 prefix/query 数据一次性
        写到 full KV。这样做的直接收益是：

        - key 侧从 2 个 kernel 收成 1 个 kernel
        - value 侧从 2 个 kernel 收成 1 个 kernel
        - 语义上更接近后续 persistent consumer 想要的最终产物
        """
        total_full_tokens = int(plan.total_kv_tokens)
        if total_full_tokens == 0:
            return _XFormersPrefixGatherProfile(
                mode="single_request_packed_compose",
                cache_view_ms=0.0,
                gather_key_ms=0.0,
                gather_value_ms=0.0,
            )

        total_elements = int(total_full_tokens * self.num_kv_heads *
                             self.head_size)
        block = 256
        context_tokens = int(plan.total_context_tokens)
        key_start = key_end = None
        value_start = value_end = None

        if profile_enabled:
            key_start, key_end = self._new_cuda_event_pair()
            key_start.record()
        _compose_single_request_packed_key_cache_kernel[
            (triton.cdiv(total_elements, block), )
        ](
            key_cache,
            query_key,
            plan.context_block_ids,
            plan.context_block_offsets,
            full_key,
            total_elements,
            context_tokens,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            pack_size=int(key_cache.shape[-1]),
            stride_cache_block=key_cache.stride(0),
            stride_cache_head=key_cache.stride(1),
            stride_cache_d_outer=key_cache.stride(2),
            stride_cache_token=key_cache.stride(3),
            stride_cache_pack=key_cache.stride(4),
            stride_query_token=query_key.stride(0),
            stride_query_head=query_key.stride(1),
            stride_query_dim=query_key.stride(2),
            stride_out_token=full_key.stride(0),
            stride_out_head=full_key.stride(1),
            stride_out_dim=full_key.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and key_end is not None:
            key_end.record()

        if profile_enabled:
            value_start, value_end = self._new_cuda_event_pair()
            value_start.record()
        _compose_single_request_packed_value_cache_kernel[
            (triton.cdiv(total_elements, block), )
        ](
            value_cache,
            query_value,
            plan.context_block_ids,
            plan.context_block_offsets,
            full_value,
            total_elements,
            context_tokens,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            stride_cache_block=value_cache.stride(0),
            stride_cache_head=value_cache.stride(1),
            stride_cache_dim=value_cache.stride(2),
            stride_cache_token=value_cache.stride(3),
            stride_query_token=query_value.stride(0),
            stride_query_head=query_value.stride(1),
            stride_query_dim=query_value.stride(2),
            stride_out_token=full_value.stride(0),
            stride_out_head=full_value.stride(1),
            stride_out_dim=full_value.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and value_end is not None:
            value_end.record()

        gather_key_ms = 0.0
        gather_value_ms = 0.0
        if (profile_enabled and key_start is not None and key_end is not None
                and value_start is not None and value_end is not None):
            gather_key_ms = self._cuda_elapsed_ms(key_start, key_end)
            gather_value_ms = self._cuda_elapsed_ms(value_start, value_end)
        return _XFormersPrefixGatherProfile(
            mode="single_request_packed_compose",
            cache_view_ms=0.0,
            gather_key_ms=gather_key_ms,
            gather_value_ms=gather_value_ms,
        )

    def _can_use_single_request_packed_compose(
        self,
        *,
        plan: _XFormersPrefixFallbackPlan,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query_key: torch.Tensor,
        query_value: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> bool:
        """判断是否可以走“单请求一次性 compose”快路径。

        这里刻意只在最核心、最稳定的口径下打开：

        - 只有 1 个 prefill request
        - prefix 与 query 都存在
        - prefix 侧能走 packed direct gather
        - query 侧能走 direct GPU scatter

        这样我们先把当前主线打薄，而不会把多请求等复杂语义也一起卷进来。

        但在当前 V100 + xFormers prefix fallback 主线里，这条 compose 快路径
        已经表现出更高的不稳定性：

        - GranuleKV runtime direct retrieve 已经完成
        - runtime metadata rebuild 也已经完成
        - 但在真正进入 xFormers fallback 消费前，就可能触发非法访存

        从最近几轮日志看，炸点更像落在：

        - `_compose_single_request_packed_key_cache_kernel`
        - `_compose_single_request_packed_value_cache_kernel`

        而不是更后面的 `xops.memory_efficient_attention_forward()`。

        因此这里先把 compose 快路径从当前主线中拿掉，只保留：

        - prefix: packed gather -> full workspace
        - query: direct scatter / segment copy

        当前实现里我们刻意不再保留那段“满足若干条件后再打开 compose”的旧判定
        代码，因为这会制造一种错觉，好像主线随时可能重新启用 compose。
        实际上当前主线语义已经非常明确：

        - compose 在 V100/xFormers fallback 主线上就是关闭的
        - 需要恢复时，应在后续独立提交里显式重开并重新验证
        """
        # 当前 prefix fallback 真正会走到这里的主要环境就是 V100 / sm<80。
        # 这条 compose 快路径在该环境下已经多次触发非法访存，因此先显式关闭，
        # 避免它继续把 GranuleKV 主线与 xFormers 消费侧问题耦在一起。
        return False

    @staticmethod
    def _can_use_packed_prefix_gather(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> bool:
        """判断当前 fallback 是否可以直接走 packed direct gather。

        当前先只在最稳定、也是我们真实主线正在使用的配置上启用：

        - Triton 可用
        - key/value cache 都在 CUDA 上
        - dtype 不是 fp8/uint8

        如果这些条件不满足，就自动退回原来的：

        ```text
        cache_view + gather_cache
        ```
        """
        if triton is None:
            return False
        if (not key_cache.is_cuda) or (not value_cache.is_cuda):
            return False
        if key_cache.dtype == torch.uint8 or value_cache.dtype == torch.uint8:
            return False
        return True

    @staticmethod
    def _can_use_direct_query_scatter(
        query_key: torch.Tensor,
        query_value: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> bool:
        """判断 query 侧是否可以直接走 GPU scatter。

        这里的语义和 `_can_use_packed_prefix_gather()` 保持一致：

        - 优先在当前稳定主线已覆盖的 CUDA + Triton 场景打开
        - 如果环境不满足，就自动退回到原来的 Python `copy_` 兼容路径

        这样我们可以先把数据面收薄，而不强迫所有运行环境一次性切到新路径。
        """
        if triton is None:
            return False
        tensors = (query_key, query_value, full_key, full_value)
        if any(not tensor.is_cuda for tensor in tensors):
            return False
        if any(tensor.dtype == torch.uint8 for tensor in tensors):
            return False
        return True

    @staticmethod
    def _new_cuda_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        """为 fallback profile 创建一对计时 event。"""
        return (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))

    @staticmethod
    def _cuda_elapsed_ms(start: torch.cuda.Event,
                         end: torch.cuda.Event) -> float:
        """读取同一条 stream 上一对 event 的耗时。"""
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: XFormersMetadata,
        attn_type: str = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        See https://facebookresearch.github.io/xformers/components/ops.html
        for API spec.

        Args:
            output: shape = [num_prefill_tokens, num_heads, head_size]
            query: shape = [num_prefill_tokens, num_heads, head_size]
            key: shape = [num_prefill_tokens, num_kv_heads, head_size]
            value: shape = [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        """

        original_query = query
        if self.num_kv_heads != self.num_heads:
            # GQA/MQA requires the shape [B, M, G, H, K].
            # Note that the output also has the same shape (which is different
            # from a spec from the doc).
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            key = key[:, :,
                      None, :].expand(key.shape[0], self.num_kv_heads,
                                      self.num_queries_per_kv, key.shape[-1])
            value = value[:, :,
                          None, :].expand(value.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          value.shape[-1])

        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        attn_bias = _get_attn_bias(attn_metadata, attn_type)
        if attn_bias is None:
            if self.alibi_slopes is None:

                # Cross attention block of decoder branch of encoder-decoder
                # model uses seq_lens for dec / encoder_seq_lens for enc
                if (attn_type == AttentionType.ENCODER_DECODER):
                    assert attn_metadata.seq_lens is not None
                    assert attn_metadata.encoder_seq_lens is not None

                    # Cross-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.seq_lens,
                        attn_metadata.encoder_seq_lens,
                        device=query.device)

                # Encoder branch of encoder-decoder model uses
                # attn_metadata.encoder_seq_lens
                elif attn_type == AttentionType.ENCODER:

                    assert attn_metadata.encoder_seq_lens is not None

                    # Encoder self-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.encoder_seq_lens, device=query.device)

                # Self-attention block of encoder-only model just
                # uses the seq_lens directly.
                elif attn_type == AttentionType.ENCODER_ONLY:
                    assert attn_metadata.seq_lens is not None

                    # Encoder self-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.seq_lens, device=query.device)

                # Self-attention block of decoder branch just
                # uses the seq_lens directly
                elif attn_type == AttentionType.DECODER:
                    assert attn_metadata.seq_lens is not None

                    # Decoder self-attention mask is causal
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        attn_metadata.seq_lens, device=query.device)
                else:
                    raise ValueError("Unknown AttentionType: %s", attn_type)

                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                attn_bias = [attn_bias]
            else:
                assert attn_type == AttentionType.DECODER
                assert attn_metadata.seq_lens is not None
                attn_bias = _make_alibi_bias(self.alibi_slopes,
                                             self.num_kv_heads, query.dtype,
                                             attn_metadata.seq_lens)

            _set_attn_bias(attn_metadata, attn_bias, attn_type)

        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_bias[0],
                p=0.0,
                scale=self.scale)
            return out.view_as(original_query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        assert attn_metadata.seq_lens is not None
        output = torch.empty_like(original_query)
        start = 0
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            end = start + seq_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_bias[i],
                p=0.0,
                scale=self.scale)
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += seq_len
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[AttentionBias]:
    attn_biases: List[AttentionBias] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            seq_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases
