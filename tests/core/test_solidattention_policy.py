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


def test_solidattention_restore_prediction_is_prefix_only():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.observe_query("r", 0, torch.ones(2, 2))
    policy.register_page_representatives("r", 0, _representatives(8))
    prediction = policy.predict_restore_blocks("r", 0, 8, 2)
    assert prediction is not None
    assert prediction == tuple(sorted(set(prediction)))
    assert min(prediction) >= 0
    assert max(prediction) < 8
