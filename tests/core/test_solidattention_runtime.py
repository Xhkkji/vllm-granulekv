"""CUDA correctness tests for the resident SolidAttention runtime."""

import pytest
import torch

from evaluation.paper_reproduction.quest.runtime import decode
from evaluation.paper_reproduction.solidattention.adapter import (
    SolidAttentionPolicy, )


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


def _reference_decode(query, key_cache, value_cache, block_table, indices,
                      sequence_length, block_size, num_kv_heads, scale):
    _, num_heads, head_dim = query.shape
    output = torch.empty_like(query)
    group_size = num_heads // num_kv_heads
    for head in range(num_heads):
        kv_head = head // group_size
        keys = []
        values = []
        logical_pages = (indices[head].tolist()
                         if isinstance(indices[head], torch.Tensor)
                         else indices[head])
        for logical in logical_pages:
            physical = int(block_table[0, logical])
            token_count = (sequence_length - logical * block_size
                           if logical == (sequence_length + block_size - 1) //
                           block_size - 1 else block_size)
            keys.append(key_cache[physical, kv_head].permute(1, 0, 2).
                        reshape(block_size, head_dim)[:token_count])
            values.append(value_cache[physical, kv_head, :, :token_count].T)
        keys = torch.cat(keys).float()
        values = torch.cat(values).float()
        logits = query[0, head].float() @ keys.T * scale
        output[0, head] = torch.softmax(logits, dim=-1) @ values
    return output


def test_solidattention_runtime_gqa_and_partial_last_block():
    _cuda_or_skip()
    torch.manual_seed(17)
    device = torch.device("cuda")
    num_heads, num_kv_heads, head_dim = 28, 4, 32
    block_size, sequence_length = 16, 38
    pack = 16 // 2
    key_cache = torch.randn(
        12,
        num_kv_heads,
        head_dim // pack,
        block_size,
        pack,
        device=device,
        dtype=torch.float16,
    )
    value_cache = torch.randn(
        12,
        num_kv_heads,
        head_dim,
        block_size,
        device=device,
        dtype=torch.float16,
    )
    block_table = torch.tensor([[7, 2, 9]], device=device, dtype=torch.int32)
    indices = torch.tensor(
        [[0, 2], [1, 2], [0, 2], [0, 1], [1, 2], [0, 1], [0, 2],
         [0, 1], [1, 2], [0, 2], [0, 1], [0, 2], [1, 2], [0, 1],
         [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [0, 2], [1, 2],
         [0, 1], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1], [0, 2]],
        device=device,
        dtype=torch.int32,
    )
    counts = torch.full((num_heads,), 2, device=device, dtype=torch.int32)
    query = torch.randn(1, num_heads, head_dim, device=device,
                        dtype=torch.float16)
    output = torch.empty_like(query)
    cache_snapshot = (key_cache.clone(), value_cache.clone(),
                      block_table.clone())

    decode(output, query, key_cache, value_cache, block_table, indices, counts,
           sequence_length, block_size, head_dim**-0.5, num_kv_heads)
    reference = _reference_decode(query, key_cache, value_cache, block_table,
                                  indices[:, :2].tolist(), sequence_length,
                                  block_size, num_kv_heads, head_dim**-0.5)
    torch.testing.assert_close(output.float(), reference.float(),
                               atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(key_cache, cache_snapshot[0])
    torch.testing.assert_close(value_cache, cache_snapshot[1])
    torch.testing.assert_close(block_table, cache_snapshot[2])


def test_solidattention_policy_returns_per_head_gpu_pages_and_reuses_buffers():
    _cuda_or_skip()
    torch.manual_seed(23)
    device = torch.device("cuda")
    heads, kv_heads, dim, block_size, sequence_length = 28, 4, 32, 16, 70
    key_cache = torch.randn(12, kv_heads, dim // 8, block_size, 8,
                            device=device, dtype=torch.float16)
    physical_ids = torch.tensor([7, 2, 9, 1, 6], device=device,
                                dtype=torch.int32)
    query = torch.randn(1, heads, dim, device=device, dtype=torch.float16)
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)

    warmup = policy.select_blocks_per_head_device(
        "request", 0, query, key_cache, physical_ids, sequence_length,
        block_size, 2)
    selected = policy.select_blocks_per_head_device(
        "request", 0, query, key_cache, physical_ids, sequence_length,
        block_size, 2)
    state = policy._runtime_buffers["request"][0]

    assert warmup.logical_block_indices.shape == (heads, 5)
    assert selected.logical_block_indices.is_cuda
    assert selected.counts.is_cuda
    assert selected.logical_block_indices.dtype == torch.int32
    assert selected.counts.unique().tolist() == [5]
    assert torch.equal(
        selected.logical_block_indices,
        torch.sort(selected.logical_block_indices, dim=1).values)
    assert selected.logical_block_indices[:, 0].eq(0).all()
    assert selected.logical_block_indices[:, -1].eq(4).all()
    score_ptr = state["scores"].data_ptr()

    policy.select_blocks_per_head_device(
        "request", 0, query, key_cache, physical_ids, sequence_length,
        block_size, 2)
    assert state["scores"].data_ptr() == score_ptr
    policy.discard("request")
    assert "request" not in policy._runtime_buffers
