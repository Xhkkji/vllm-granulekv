"""Tests for the SolidAttention policy adapter."""

import pytest
import torch

from evaluation.paper_reproduction.solidattention.adapter import (
    SolidAttentionPolicy, )


def _representatives(num_pages: int = 8) -> torch.Tensor:
    lower = torch.full((num_pages, 2, 2), -1.0)
    upper = torch.full((num_pages, 2, 2), 1.0)
    return torch.stack((lower, upper), dim=1)


def test_solidattention_cpu_warmup_and_fixed_blocks():
    policy = SolidAttentionPolicy(init_blocks=2, local_blocks=2)
    reps = _representatives()
    query = torch.ones(2, 2)

    assert policy.select_blocks("r", 0, query, reps, 8, 1) == tuple(range(8))
    selected = policy.select_blocks("r", 0, query, reps, 8, 1)
    assert selected[:2] == (0, 1)
    assert {5, 6, 7}.issubset(selected)
    assert len(selected) == 6
    assert selected == tuple(sorted(set(selected)))


def test_solidattention_cpu_keeps_live_suffix_outside_prefix_metadata():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.observe_query("r", 0, torch.ones(2, 2))
    policy.select_blocks("r", 0, torch.ones(2, 2), _representatives(3), 6, 1)
    selected = policy.select_blocks("r", 0, torch.ones(2, 2),
                                    _representatives(3), 6, 1)
    assert selected[-3:] == (3, 4, 5)
    assert max(selected) < 6


def test_solidattention_gpu_path_caches_prefix_representatives():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    query = torch.ones(2, 4, device="cuda")
    key_cache = torch.randn(8, 2, 4, 4, device="cuda")
    physical_ids = torch.arange(8, dtype=torch.long, device="cuda")

    first = policy.select_blocks_device("r", 0, query, key_cache, physical_ids,
                                        96, 16, 2)
    second = policy.select_blocks_device("r", 0, query, key_cache, physical_ids,
                                         96, 16, 2)
    assert first.device.type == "cuda"
    assert second.device.type == "cuda"
    assert torch.equal(second, torch.unique(second, sorted=True))
    assert second[-1].item() == 5


def test_solidattention_attention_buffer_is_reused_and_grows():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    query = torch.ones(2, 4, device="cuda")
    key_cache = torch.randn(16, 2, 4, 4, device="cuda")
    physical_ids = torch.arange(16, dtype=torch.long, device="cuda")

    first = policy.select_attention_blocks_device("r", 0, query, key_cache,
                                                 physical_ids, 24, 4, 2)
    second = policy.select_attention_blocks_device("r", 0, query, key_cache,
                                                  physical_ids, 24, 4, 2)
    assert first.logical_block_indices.dtype == torch.int32
    assert first.logical_block_indices.is_contiguous()
    assert first.count == 6
    assert second.count <= first.count
    assert first.logical_block_indices.data_ptr() == (
        second.logical_block_indices.data_ptr())

    grown = policy.select_attention_blocks_device("r", 0, query, key_cache,
                                                  physical_ids, 48, 4, 8)
    assert grown.count > first.count
    assert grown.logical_block_indices.numel() >= grown.count
    policy.discard("r")
    assert "r" not in policy._attention_selection_buffers


def test_solidattention_reuses_selection_within_block(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(
        "vllm.envs.VLLM_GRANULEKV_SPARSE_SELECTION_REUSE_ENABLE", True)
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    query = torch.ones(1, 2, 4, device="cuda")
    key_cache = torch.randn(16, 2, 4, 4, device="cuda")
    physical_ids = torch.arange(16, dtype=torch.long, device="cuda")

    warmup = policy.select_attention_blocks_device(
        "r", 0, query, key_cache, physical_ids, 24, 4, 2)
    refreshed = policy.select_attention_blocks_device(
        "r", 0, -query, key_cache, physical_ids, 25, 4, 2)
    reused = policy.select_attention_blocks_device(
        "r", 0, query * 3, key_cache, physical_ids, 26, 4, 2)

    assert warmup.count == 6
    assert refreshed.count == reused.count
    assert refreshed.logical_block_indices.data_ptr() == (
        reused.logical_block_indices.data_ptr())
    assert torch.equal(refreshed.logical_block_indices[:refreshed.count],
                       reused.logical_block_indices[:reused.count])


def test_solidattention_refreshes_when_sequence_enters_new_block(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setattr(
        "vllm.envs.VLLM_GRANULEKV_SPARSE_SELECTION_REUSE_ENABLE", True)
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    query = torch.ones(1, 2, 4, device="cuda")
    key_cache = torch.randn(16, 2, 4, 4, device="cuda")
    physical_ids = torch.arange(16, dtype=torch.long, device="cuda")

    policy.select_attention_blocks_device(
        "r", 0, query, key_cache, physical_ids, 24, 4, 2)
    within = policy.select_attention_blocks_device(
        "r", 0, query, key_cache, physical_ids, 26, 4, 2)
    within_last = within.logical_block_indices[within.count - 1].item()
    crossed = policy.select_attention_blocks_device(
        "r", 0, query, key_cache, physical_ids, 29, 4, 2)

    assert within_last == 6
    assert crossed.logical_block_indices[crossed.count - 1].item() == 7
    assert policy._attention_selection_buffers["r"][0].num_blocks == 8


def test_solidattention_gqa_score_matches_expanded_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    query = torch.randn(28, 8, device="cuda")
    representatives = torch.randn(7, 2, 4, 8, device="cuda")
    lower, upper = representatives[:, 0], representatives[:, 1]
    expanded = representatives.repeat_interleave(7, dim=2)
    chosen = torch.where(query.unsqueeze(0) >= 0, expanded[:, 1],
                         expanded[:, 0])
    expected = torch.einsum("hd,phd->hp", query, chosen)
    actual = SolidAttentionPolicy._score_pages(query, representatives)
    torch.testing.assert_close(actual, expected)


def test_solidattention_restore_prediction_is_prefix_only():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.observe_query("r", 0, torch.ones(2, 2))
    policy.register_page_representatives("r", 0, _representatives(8))
    prediction = policy.predict_restore_blocks("r", 0, 8, 2)
    assert prediction is not None
    assert prediction == tuple(sorted(set(prediction)))
    assert min(prediction) >= 0
    assert max(prediction) < 8
