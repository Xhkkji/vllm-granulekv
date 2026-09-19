"""Correctness tests for the selected-logical-block paged attention ops."""

import math

import pytest
import torch

from vllm.attention.ops.paged_attn import PagedAttention


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
