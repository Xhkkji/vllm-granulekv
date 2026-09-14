"""Tests for the Quest plugin and generic policy boundary."""

import pytest
import torch

from evaluation.paper_reproduction.quest.adapter import QuestPageIndex, QuestPolicy
from vllm.core.custom_schedulers.hierarchical_io import (
    SparseKVPolicyRuntime, build_sparse_restore_plan_feedback,
    configure_sparse_kv_policy, discard_sparse_restore_context,
    load_sparse_kv_policy, register_sparse_restore_context)


def test_quest_gpu_selection_caches_prefix_metadata():
    from vllm.attention.ops.sparse_kv import reset_sparse_kv_stats, sparse_kv_stats

    policy = QuestPolicy()
    try:
        query = torch.ones(2, 4, device="cuda")
        key_cache = torch.arange(5 * 2 * 4 * 4,
                                 dtype=torch.float32,
                                 device="cuda").reshape(5, 2, 4, 4)
        physical_ids = torch.arange(5, dtype=torch.int32, device="cuda")
    except RuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    policy.observe_query("gpu", 0, query)
    reset_sparse_kv_stats()

    first = policy.select_blocks_device("gpu", 0, query, key_cache,
                                        physical_ids, 16, 4, 1)
    second = policy.select_blocks_device("gpu", 0, query, key_cache,
                                         physical_ids, 16, 4, 1)
    assert first.device.type == "cuda"
    assert torch.equal(first, second)
    assert sparse_kv_stats()["metadata_build_calls"] == 1
    assert sparse_kv_stats()["metadata_build_blocks"] == 3

    policy.select_blocks_device("gpu", 0, query, key_cache,
                                physical_ids, 20, 4, 1)
    assert sparse_kv_stats()["metadata_build_calls"] == 2
    assert sparse_kv_stats()["metadata_build_blocks"] == 4
    policy.discard("gpu")


def test_quest_policy_uses_dense_warmup_and_forces_latest_block():
    policy = QuestPolicy()
    query = torch.ones(2, 4)
    reps = torch.ones(4, 2, 4)
    assert policy.select_blocks("r", 0, query, reps, 4, 1) == (0, 1, 2, 3)
    policy.observe_query("r", 0, query)
    selected = policy.select_blocks("r", 0, query, reps, 4, 1)
    assert selected[-1] == 3
    assert len(selected) <= 2


def test_page_index_is_cpu_owned_and_discardable():
    index = QuestPageIndex()
    reps = torch.ones(3, 2, 4)
    index.register("r", 0, reps)
    assert index.get("r", 0).device.type == "cpu"
    assert ("r", 0) in index
    index.discard("r")
    with pytest.raises(KeyError):
        index.get("r", 0)


def test_policy_reads_bound_min_max_page_index():
    policy = QuestPolicy()
    envelope = torch.tensor([
        [[[-1.0, -4.0]], [[2.0, 3.0]]],
        [[[-3.0, 1.0]], [[1.0, 5.0]]],
        [[[-2.0, -2.0]], [[4.0, 2.0]]],
    ])
    policy.register_page_representatives("prefix-key", 0, envelope)
    policy.bind_page_index_key("live-request", "prefix-key")
    policy.observe_query("live-request", 0, torch.tensor([[2.0, -1.0]]))
    selected = policy.select_blocks("live-request", 0, torch.empty(0),
                                    torch.empty(0), 3, 1)
    # Page 2 has the largest envelope score; the newest page is also page 2.
    assert selected == (2, )


def test_policy_plugin_loads_without_core_importing_quest():
    policy = load_sparse_kv_policy(
        "evaluation.paper_reproduction.quest.adapter.policy:QuestPolicy")
    runtime = SparseKVPolicyRuntime(policy)
    query = torch.ones(2, 4)
    reps = torch.ones(3, 2, 4)
    runtime.observe_query("r", 0, query)
    assert runtime.select_blocks("r", 0, query, reps, 3, 1)[-1] == 2


def test_quest_policy_rejects_batch_two():
    with pytest.raises(ValueError, match="batch size 1"):
        QuestPolicy().observe_query("r", 0, torch.ones(2, 2, 4))


def test_quest_policy_keeps_unindexed_live_suffix():
    policy = QuestPolicy()
    policy.observe_query("r", 0, torch.ones(2, 4))
    selected = policy.select_blocks("r", 0, torch.ones(2, 4),
                                    torch.ones(2, 2, 4), 3, 1)
    assert selected[-1] == 2


def test_quest_policy_expands_gqa_kv_heads():
    policy = QuestPolicy()
    query = torch.ones(4, 4)
    representatives = torch.ones(3, 2, 4)
    policy.observe_query("gqa", 0, query)
    assert policy.select_blocks("gqa", 0, query, representatives, 3,
                                1) == (0, 2)


def test_policy_keeps_live_suffix_when_page_index_covers_prefix_only():
    policy = QuestPolicy()
    policy.register_page_representatives(
        "prefix-only", 0, torch.ones(3, 2, 4))
    policy.bind_page_index_key("live", "prefix-only")
    policy.observe_query("live", 0, torch.ones(2, 4))
    selected = policy.select_blocks("live", 0, torch.empty(0),
                                    torch.empty(0), 5, 1)
    assert selected[-2:] == (3, 4)


def test_page_index_can_register_partial_logical_ranges():
    index = QuestPageIndex()
    index.register("r", 0, torch.tensor([[[1.0]], [[3.0]]]),
                   logical_block_indices=(1, 3))
    with pytest.raises(KeyError, match="missing"):
        index.get("r", 0)
    index.register("r", 0, torch.tensor([[[0.0]], [[2.0]]]),
                   logical_block_indices=(0, 2))
    assert index.get("r", 0).shape[0] == 4


def test_restore_prediction_is_prefix_only_and_uses_current_query_history():
    policy = QuestPolicy()
    policy.bind_page_index_key("request", "prefix")
    for layer in range(2):
        policy.register_page_representatives("request", layer,
                                             torch.ones(4, 2, 4))
        policy.observe_query("request", layer, torch.ones(2, 4))

    for layer in range(2):
        prediction = policy.predict_restore_blocks("request", layer, 4, 1)
        assert prediction
        assert max(prediction) < 4


def test_restore_prediction_ignores_live_suffix_representatives():
    policy = QuestPolicy()
    policy.bind_page_index_key("request", "prefix")
    policy.register_page_representatives("request", 0,
                                         torch.ones(5, 2, 4))
    policy.observe_query("request", 0, torch.ones(2, 4))
    prediction = policy.predict_restore_blocks("request", 0, 4, 1)
    assert prediction
    assert max(prediction) < 4


def test_consumer_selection_uses_restore_prefix_boundary():
    policy = QuestPolicy()
    policy.bind_page_index_key("request", "prefix")
    policy.set_restore_prefix_blocks("request", 4)
    policy.observe_query("request", 0, torch.ones(2, 4))
    selected = policy.select_blocks("request", 0, torch.empty(0),
                                    torch.ones(5, 2, 4), 5, 1)
    assert selected == (0, 4)


def test_sparse_restore_feedback_compiles_one_plan_per_request():
    policy = QuestPolicy()
    configure_sparse_kv_policy(policy)
    try:
        policy.bind_page_index_key("request", "prefix")
        for layer in range(2):
            policy.register_page_representatives("request", layer,
                                                 torch.ones(4, 2, 4))
            policy.observe_query("request", layer, torch.ones(2, 4))
            register_sparse_restore_context("request", "prefix", 4,
                                            (layer, layer + 1))
        feedback = build_sparse_restore_plan_feedback("request", 1)
        assert feedback is not None
        assert feedback.page_index_key == "prefix"
        assert feedback.num_layers == 2
        assert all(max(selection) < 4
                   for selection in feedback.block_indices_by_layer)
    finally:
        discard_sparse_restore_context(("request", ))
        configure_sparse_kv_policy(None)
