"""Correctness tests for the selected-logical-block paged attention ops."""

import math

import pytest
import torch

from vllm.attention.ops.paged_attn import PagedAttention


def test_single_token_prefill_metadata_is_a_local_view():
    from vllm.attention.backends.xformers import XFormersMetadata
    metadata = XFormersMetadata(
        num_prefills=1, num_prefill_tokens=1, num_decode_tokens=0,
        slot_mapping=torch.tensor([31]), seq_lens_tensor=torch.tensor([32]),
        max_prefill_seq_len=32, max_decode_seq_len=0, use_cuda_graph=False,
        multi_modal_placeholder_index_maps=None, enable_kv_scales_calculation=False,
        block_tables=torch.tensor([[1, 0]]))
    view = metadata.single_token_sparse_prefill()
    assert view is not metadata
    assert metadata.num_prefills == 1
    assert view.num_prefills == 0 and view.num_decode_tokens == 1
    assert view.max_decode_seq_len == 32
    assert view.block_tables is metadata.block_tables
    assert view.slot_mapping is metadata.slot_mapping
    metadata.num_prefill_tokens = 2
    assert metadata.single_token_sparse_prefill() is metadata


def _reference_attention(query: torch.Tensor, key_cache: torch.Tensor,
                         value_cache: torch.Tensor, block_table: torch.Tensor,
                         sequence_length: int, selected: list[int],
                         scale: float) -> torch.Tensor:
    _, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[3]
    queries_per_kv = num_heads // num_kv_heads
    positions = torch.cat([
        torch.arange(index * block_size,
                     min(sequence_length, (index + 1) * block_size),
                     device=query.device) for index in selected
    ])
    output = torch.empty_like(query)
    for head in range(num_heads):
        kv_head = head // queries_per_kv
        keys = []
        values = []
        for logical_index in selected:
            valid = min(block_size,
                        sequence_length - logical_index * block_size)
            if valid <= 0:
                continue
            physical_index = int(block_table[0, logical_index].item())
            key = key_cache[physical_index, kv_head].permute(1, 0,
                                                              2).reshape(
                                                                  block_size,
                                                                  head_size)
            value = value_cache[physical_index, kv_head].transpose(0, 1)
            keys.append(key[:valid])
            values.append(value[:valid])
        key = torch.cat(keys).float()
        value = torch.cat(values).float()
        logits = query[0, head].float() @ key.transpose(0, 1) * scale
        weights = torch.softmax(logits, dim=-1)
        output[0, head] = (weights[:, None] * value).sum(0).to(query.dtype)
    assert positions.numel() == sum(
        min(block_size, sequence_length - index * block_size)
        for index in selected)
    return output


@pytest.mark.parametrize("sequence_length,selected", [
    (50, [0, 2, 3]),
    (9000, list(range(520)) + [562]),
])
def test_selected_paged_attention_matches_reference(sequence_length, selected):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    block_size = 16
    num_blocks = (sequence_length + block_size - 1) // block_size
    num_heads, num_kv_heads, head_size = 28, 4, 32
    x = 16 // 2
    key_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        head_size // x,
        block_size,
        x,
        device="cuda",
        dtype=torch.float16,
    )
    value_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        head_size,
        block_size,
        device="cuda",
        dtype=torch.float16,
    )
    block_table = torch.arange(num_blocks, device="cuda",
                               dtype=torch.int32).flip(0).view(1, -1)
    selected = sorted(set(selected))
    selected_tensor = torch.tensor(selected, device="cuda", dtype=torch.int32)
    selected_seq_len = sum(
        min(block_size, sequence_length - index * block_size)
        for index in selected)
    query = torch.randn(1, num_heads, head_size, device="cuda",
                        dtype=torch.float16)
    seq_lens = torch.tensor([sequence_length], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_size)
    k_scale = v_scale = torch.tensor(1.0, device="cuda")
    table_before = block_table.clone()

    output = PagedAttention.forward_decode_selected(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        selected_tensor,
        len(selected),
        selected_seq_len,
        "auto",
        num_kv_heads,
        scale,
        None,
        k_scale,
        v_scale,
    )
    reference = _reference_attention(query, key_cache, value_cache,
                                     block_table, sequence_length, selected,
                                     scale)
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(block_table, table_before)


@pytest.mark.parametrize("sequence_length,num_heads,num_kv_heads,head_size", [
    (257, 8, 2, 32),
    (32769, 28, 4, 128),
])
def test_selected_all_blocks_matches_dense_paged_attention(
        sequence_length, num_heads, num_kv_heads, head_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    block_size = 16
    num_blocks = (sequence_length + block_size - 1) // block_size
    x = 16 // 2
    key_cache = torch.randn(num_blocks,
                            num_kv_heads,
                            head_size // x,
                            block_size,
                            x,
                            device="cuda",
                            dtype=torch.float16)
    value_cache = torch.randn(num_blocks,
                              num_kv_heads,
                              head_size,
                              block_size,
                              device="cuda",
                              dtype=torch.float16)
    block_table = torch.randperm(num_blocks, device="cuda",
                                 dtype=torch.int64).to(torch.int32).view(1, -1)
    selected = torch.arange(num_blocks, device="cuda", dtype=torch.int32)
    query = torch.randn(1, num_heads, head_size, device="cuda",
                        dtype=torch.float16)
    seq_lens = torch.tensor([sequence_length], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_size)
    k_scale = v_scale = torch.tensor(1.0, device="cuda")

    selected_output = PagedAttention.forward_decode_selected(
        query, key_cache, value_cache, block_table, seq_lens, selected,
        num_blocks, sequence_length, "auto", num_kv_heads, scale, None,
        k_scale, v_scale)
    dense_output = PagedAttention.forward_decode(
        query, key_cache, value_cache, block_table, seq_lens, sequence_length,
        "auto", num_kv_heads, scale, None, k_scale, v_scale)

    torch.testing.assert_close(selected_output,
                               dense_output,
                               rtol=2e-2,
                               atol=2e-2)


@pytest.mark.parametrize("sequence_length", [257, 32769])
def test_paired_layout_selected_attention_matches_dense(sequence_length):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    block_size = 16
    num_blocks = (sequence_length + block_size - 1) // block_size
    num_heads, num_kv_heads, head_size = 28, 4, 128
    x = 16 // 2
    elements = block_size * num_kv_heads * head_size
    storage = torch.randn(2 * num_blocks * elements, device="cuda",
                          dtype=torch.float16)
    backing = storage.as_strided(
        (2, num_blocks, elements), (elements, 2 * elements, 1))
    key_cache = backing[0].view(num_blocks, num_kv_heads,
                                head_size // x, block_size, x)
    value_cache = backing[1].view(num_blocks, num_kv_heads, head_size,
                                  block_size)
    block_table = torch.arange(num_blocks, device="cuda",
                               dtype=torch.int32).flip(0).view(1, -1)
    query = torch.randn(1, num_heads, head_size, device="cuda",
                        dtype=torch.float16)
    seq_lens = torch.tensor([sequence_length], device="cuda", dtype=torch.int32)
    selected = torch.arange(num_blocks, device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(head_size)
    k_scale = v_scale = torch.tensor(1.0, device="cuda")

    selected_output = PagedAttention.forward_decode_selected(
        query, key_cache, value_cache, block_table, seq_lens, selected,
        num_blocks, sequence_length, "auto", num_kv_heads, scale, None,
        k_scale, v_scale)
    dense_output = PagedAttention.forward_decode(
        query, key_cache, value_cache, block_table, seq_lens, sequence_length,
        "auto", num_kv_heads, scale, None, k_scale, v_scale)
    torch.testing.assert_close(selected_output, dense_output,
                               rtol=2e-2, atol=2e-2)
    contiguous_dense = PagedAttention.forward_decode(
        query, key_cache.contiguous(), value_cache.contiguous(), block_table,
        seq_lens, sequence_length, "auto", num_kv_heads, scale, None,
        k_scale, v_scale)
    torch.testing.assert_close(dense_output, contiguous_dense, rtol=0, atol=0)
    subset = list(range(0, num_blocks, 3))
    sparse = PagedAttention.forward_decode_selected(
        query, key_cache, value_cache, block_table, seq_lens,
        torch.tensor(subset, device="cuda", dtype=torch.int32), len(subset),
        sum(min(block_size, sequence_length - i * block_size) for i in subset),
        "auto", num_kv_heads, scale, None, k_scale, v_scale)
    reference = _reference_attention(query, key_cache, value_cache, block_table,
                                     sequence_length, subset, scale)
    torch.testing.assert_close(sparse, reference, rtol=2e-2, atol=2e-2)
