"""Tests for the SolidAttention policy adapter."""

import pytest
import torch

from evaluation.paper_reproduction.solidattention.adapter import (
    SolidAttentionPolicy, )
from evaluation.paper_reproduction.solidattention.adapter.reference import (
    reference_solidattention_selection, )
from vllm.core.custom_schedulers.hierarchical_io import (
    build_sparse_restore_plan_feedback, configure_sparse_kv_policy,
    discard_sparse_restore_context, register_sparse_restore_context)


def _representatives(num_pages: int = 8) -> torch.Tensor:
    lower = torch.full((num_pages, 2, 2), -1.0)
    upper = torch.full((num_pages, 2, 2), 1.0)
    return torch.stack((lower, upper), dim=1)


@pytest.mark.parametrize("computed,tokens", [(160, 1), (163, 1), (163, 37)])
def test_complete_history_selection_and_partial_current_union(computed, tokens):
    policy = SolidAttentionPolicy(1, 1)
    policy.register_page_representatives("page", 0, _representatives(4))
    policy.bind_page_index_key("r", "page")
    policy._representatives["r"][0] = (10, _representatives(10))
    actual = policy.select_actual_blocks_device(
        "r", 0, torch.ones(tokens, 2, 2), 4, computed,
        computed + tokens, 16, 1)
    assert actual.history_blocks == (0, 1, 9) + ((10,) if computed % 16 else ())
    indices = actual.attention.logical_block_indices[:actual.attention.count].tolist()
    assert indices == [0, 1, 9, *range(10, (computed + tokens + 15) // 16)]
    assert len(indices) == len(set(indices))


@pytest.mark.parametrize("heads", [2, 8])
@pytest.mark.parametrize("complete", [1, 3, 12])
def test_complete_history_chunk_one_matches_single_query_reference(heads, complete):
    torch.manual_seed(29)
    reps = torch.randn(complete, 2, 2, 2)
    query = torch.randn(heads, 2)
    policy = SolidAttentionPolicy(2, 2)
    policy.register_page_representatives("r", 0, reps)
    expected = reference_solidattention_selection(
        query, reps, complete, 2, init_blocks=2, local_blocks=2) + (complete,)
    single = policy.select_actual_blocks_device("r", 0, query, 1,
                                               complete * 16, complete * 16 + 1, 16, 2)
    chunk = policy.select_actual_blocks_device("r", 0, query.unsqueeze(0), 1,
                                              complete * 16, complete * 16 + 1, 16, 2)
    assert single.history_blocks == chunk.history_blocks
    assert chunk.attention.logical_block_indices[:chunk.attention.count].tolist() == list(expected)


def test_history_metadata_seals_only_new_full_blocks_and_is_request_local(monkeypatch):
    import evaluation.paper_reproduction.solidattention.adapter.solidattention as module
    policy = SolidAttentionPolicy(1, 1)
    source = _representatives(4)
    policy.register_page_representatives("page", 0, source)
    for request in ("a", "b"):
        policy.bind_page_index_key(request, "page")
        policy.select_actual_blocks_device(request, 0, torch.ones(2, 2),
                                          4, 64, 65, 16, 1)
    calls = []
    builder = module.build_page_representatives_from_paged_key_cache
    def capture(cache, ids, **kwargs):
        calls.append(ids.tolist())
        return builder(cache, ids, **kwargs)
    monkeypatch.setattr(module, "build_page_representatives_from_paged_key_cache", capture)
    key = torch.randn(8, 2, 16, 2)
    ids = torch.tensor([5, 1, 6, 2, 7, 0, 4, 3])
    for length in (79, 80, 80, 95, 96):
        policy.update_history_representatives("a", 0, key, ids, length, 16)
    assert calls == [[7], [0]]
    assert policy._representatives["a"][0][0] == 6
    assert policy._representatives["b"][0][0] == 4
    shared = policy._prediction_representatives["page"][0].representatives
    assert shared.data_ptr() == source.data_ptr()
    torch.testing.assert_close(shared, source)
    policy.discard("a")
    assert "a" not in policy._representatives
    assert "page" in policy._prediction_representatives


def test_history_metadata_missing_after_reuse_is_explicit_error():
    policy = SolidAttentionPolicy()
    policy.register_page_representatives("page", 0, _representatives(4))
    policy.bind_page_index_key("r", "page")
    with pytest.raises(RuntimeError, match="not sealed"):
        policy.select_actual_blocks_device("r", 0, torch.ones(2, 2),
                                          4, 160, 161, 16, 1)


def test_last_cached_prompt_token_is_partial_history_until_recomputed():
    policy = SolidAttentionPolicy(1, 1)
    policy.register_page_representatives("page", 0, _representatives(4))
    policy.bind_page_index_key("r", "page")
    actual = policy.select_actual_blocks_device("r", 0, torch.ones(2, 2),
                                               4, 63, 64, 16, 1)
    assert actual.history_blocks == (0, 1, 2, 3)
    assert policy._representatives["r"][0][0] == 3
    key = torch.randn(4, 2, 16, 2)
    policy.update_history_representatives("r", 0, key, torch.arange(4), 64, 16)
    assert policy._representatives["r"][0][0] == 4
    torch.testing.assert_close(policy._representatives["r"][0][1][3],
                               torch.stack((key[3].amin(1), key[3].amax(1))))


def test_validation_host_copy_checks_selected_history_and_partial_version():
    policy = SolidAttentionPolicy(1, 1)
    policy.configure_validation("r", "record", dict(_validation_metadata(), prefix_blocks=8))
    policy.register_page_representatives("r", 0, _representatives(8))
    policy.select_actual_blocks_device("r", 0, torch.ones(2, 2), 8, 127, 128, 16, 1)
    policy.record_validation_host_copy("r", 0, 127, 16, "actual", ((0, 10), (6, 16)))
    policy.record_validation_host_copy("r", 0, 127, 16, "partial_tail", ((7, 17),))
    with pytest.raises(RuntimeError, match="selected read set"):
        policy.record_validation_host_copy("r", 0, 127, 16, "actual", ((4, 14),))
    with pytest.raises(RuntimeError, match="selected read set"):
        policy.record_validation_host_copy("r", 0, 127, 16, "prefetch", ((7, 17),))
    assert policy.validation_report("r")["host_copy_duplicate_blocks"] == 0


def test_short_history_only_selects_partial_current_and_seals_at_boundary():
    policy = SolidAttentionPolicy()
    key = torch.randn(2, 2, 16, 2)
    ids = torch.arange(2)
    actual = policy.select_prefill_blocks_device("r", 0, torch.ones(2, 2),
                                                key, ids, 8, 7, 16, 1)
    assert actual.history_blocks == (0,)
    assert actual.attention.logical_block_indices[:actual.attention.count].tolist() == [0]
    policy.update_history_representatives("r", 0, key, ids, 16, 16)
    assert policy._representatives["r"][0][1].shape == (1, 2, 2, 2)


def test_history_trace_records_final_selection_and_replays_boundaries():
    policy = SolidAttentionPolicy(1, 1)
    metadata = dict(_validation_metadata(), prefix_blocks=4)
    policy.configure_validation("r", "record", metadata)
    policy.register_page_representatives("r", 0, _representatives(10))
    actual = policy.select_actual_blocks_device("r", 0, torch.ones(2, 2),
                                               4, 163, 164, 16, 1)
    report = policy.validation_report("r")
    record = report["records"][0]
    assert record["attention_blocks"] == [0, 1, 9, 10]
    assert record["selected_history_blocks"] == list(actual.history_blocks)
    assert record["selected_prefix_blocks"] == [0, 1]
    policy.configure_validation("r", "replay", metadata,
                                report["records"], report["metadata"])
    replay = policy.select_actual_blocks_device("r", 0, -torch.ones(2, 2),
                                               4, 163, 164, 16, 1)
    assert replay.history_blocks == actual.history_blocks
    policy.configure_validation("r", "replay", metadata,
                                report["records"], report["metadata"])
    with pytest.raises(RuntimeError, match="complete-history semantics"):
        policy.select_actual_blocks_device("r", 0, torch.ones(2, 2),
                                           4, 162, 163, 16, 1)


def test_history_selector_rejects_legacy_prefix_only_trace():
    policy = SolidAttentionPolicy(1, 1)
    metadata = _validation_metadata()
    record = {"decode_step": 0, "layer_index": 0, "prefix_blocks": 3,
              "selected_prefix_blocks": [0, 2], "attention_blocks": [0, 2, 3]}
    policy.register_page_representatives("r", 0, _representatives(4))
    policy.configure_validation("r", "replay", metadata, [record], metadata)
    with pytest.raises(RuntimeError, match="complete-history semantics"):
        policy.select_actual_blocks_device("r", 0, torch.ones(2, 2),
                                           3, 64, 65, 16, 1)


@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("heads", [2, 8])
@pytest.mark.parametrize("tokens", [1, 7])
def test_prefill_scores_are_per_head_chunk_max(envelope, heads, tokens):
    torch.manual_seed(19)
    reps = torch.randn(12, 2, 4)
    if envelope:
        reps = torch.stack((reps - 1, reps + 1), dim=1)
    query = torch.randn(tokens, heads, 4)
    policy = SolidAttentionPolicy(1, 1)
    expected = torch.stack([policy._score_pages(q, reps) for q in query]).amax(0)
    torch.testing.assert_close(policy._score_pages(query, reps), expected)
    policy.register_page_representatives("r", 0, reps)
    for budget in (1, 12, 32):
        mask = policy._topk_mask(expected[:, 1:-1],
                                 torch.arange(1, 11), budget)
        chosen = (1 + torch.nonzero(mask).flatten()).tolist()
        padded = policy._select_actual_prefix_fixed(
            "r", 0, query, 12, budget)
        assert padded[padded >= 0].tolist() == [0, *chosen, 11]


def test_prefill_stable_ties_do_not_select_every_query_union():
    policy = SolidAttentionPolicy(1, 1)
    policy.register_page_representatives("r", 0, _representatives(12))
    query = torch.ones(7, 8, 2)
    selected = policy._select_actual_prefix_fixed("r", 0, query, 12, 2)
    assert selected[selected >= 0].tolist() == [0, 1, 2, 11]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_resident_prefill_uses_immutable_boundary_and_explicit_control(monkeypatch):
    from vllm import envs
    monkeypatch.setattr(envs, "VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE", False)
    policy = SolidAttentionPolicy(1, 1)
    metadata = dict(_validation_metadata(), prefix_blocks=8)
    policy.configure_validation("request", "record", metadata)
    # Complete generated history participates; the newest complete page stays local.
    key = torch.randn(12, 2, 1, 16, 2, device="cuda")
    table = torch.arange(12, device="cuda")
    query = torch.randn(7, 8, 2, device="cuda")
    monkeypatch.setattr(torch.Tensor, "tolist",
                        lambda _: pytest.fail("selection used CUDA tolist"))
    actual = policy.select_prefill_blocks_device(
        "request", 0, query, key, table, 167, 160, 16, 1)
    assert actual.history_blocks[0] == 0
    assert actual.history_blocks[-1] == 9
    assert all(block < 10 for block in actual.history_blocks)
    assert actual.attention.count == len(actual.history_blocks) + 1
    assert "default" in policy._selection_control_buffers


@pytest.mark.parametrize("budget", [1, 4, 32])
@pytest.mark.parametrize("heads", [2, 8])
@pytest.mark.parametrize("tied", [False, True])
def test_fixed_shape_actual_matches_reference_and_caches_indices(budget, heads, tied):
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    logical = (0, 2, 3, 5, 7)
    torch.manual_seed(11)
    reps = (_representatives(len(logical)) if tied else
            torch.randn(len(logical), 2, 2, 2))
    policy.register_page_representatives("page", 0, reps, logical)
    policy.bind_page_index_key("r", "page")
    query = torch.ones(heads, 2)
    expected = policy._select_prefix_with_logical_indices(
        query, reps, logical, 8, budget)
    padded = policy._select_actual_prefix_fixed("r", 0, query, 8, budget)
    assert padded.shape == (8,)
    assert torch.equal(padded[:expected.numel()], expected)
    assert (padded[expected.numel():] == -1).all()
    cached = policy._actual_metadata["r", 0]
    policy._select_actual_prefix_fixed("r", 0, -query, 8, budget)
    assert policy._actual_metadata["r", 0] is cached
    policy.register_page_representatives("page", 0, reps * 2, logical)
    policy._select_actual_prefix_fixed("r", 0, query, 8, budget)
    assert policy._actual_metadata["r", 0] is not cached
    policy.discard("r")
    assert not policy._actual_metadata


def test_metadata_seed_uses_stable_page_after_populate_cleanup():
    policy = SolidAttentionPolicy(1, 1)
    policy.bind_page_index_key("populate", "full-page")
    policy.register_page_representatives("full-page", 0, _representatives())
    policy.discard("populate")
    assert policy.seed_page_metadata_from_request("populate", "short-page", 4) == 0
    assert policy.seed_page_metadata_from_request("full-page", "short-page", 4) == 1
    policy.bind_page_index_key("measured", "short-page")
    logical, reps = policy._get_prediction_representatives("measured", 0, 4)
    assert logical == (0, 1, 2, 3)
    assert reps.shape[0] == 4


def test_fixed_shape_cache_isolates_layer_prefix_and_page():
    policy = SolidAttentionPolicy(1, 1)
    query = torch.ones(2, 2)
    for page in ("a", "b"):
        for layer in (0, 1):
            policy.register_page_representatives(page, layer, _representatives())
    policy.bind_page_index_key("r", "a")
    policy._select_actual_prefix_fixed("r", 0, query, 8, 1)
    old = policy._actual_metadata["r", 0]
    policy._select_actual_prefix_fixed("r", 1, query, 8, 1)
    assert policy._actual_metadata["r", 0] is old
    policy._select_actual_prefix_fixed("r", 0, query, 4, 1)
    assert policy._actual_metadata["r", 0] is not old
    old = policy._actual_metadata["r", 0]
    policy.bind_page_index_key("r", "b")
    policy._select_actual_prefix_fixed("r", 0, query, 4, 1)
    assert policy._actual_metadata["r", 0] is not old


def test_prediction_reuses_actual_selection_without_retaining_query(monkeypatch):
    monkeypatch.setattr(
        'vllm.envs.VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE', True)
    monkeypatch.setattr('vllm.envs.VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS', 0)
    policy = SolidAttentionPolicy(1, 1)
    policy.register_page_representatives("r", 0, _representatives())
    query = torch.ones(2, 2)
    policy._actual_predictions["r", 0] = (8, 2, (0, 2, 7))
    policy.observe_query("r", 0, query)
    assert "r" not in policy._previous_queries
    assert policy.predict_restore_blocks("r", 0, 8, 2) == (0, 2, 7)
    assert policy.predict_restore_blocks("r", 0, 8, 3) is None
    policy.register_page_representatives("r", 0, _representatives())
    assert ("r", 0) not in policy._actual_predictions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_actual_has_no_nonzero_and_feedback_does_not_rescore(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE", True)
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_HOST_BACKING_ENABLE", True)
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS", 0)
    policy = SolidAttentionPolicy(1, 1)
    reps = _representatives(12)
    policy.register_page_representatives("page", 0, reps)
    policy.bind_page_index_key("r", "page")
    query = torch.ones(2, 2, device="cuda")
    expected = tuple(policy._select_prefix_with_logical_indices(
        query.cpu(), reps, tuple(range(12)), 12, 2).tolist())
    monkeypatch.setattr(torch, "nonzero", lambda *a, **kw: pytest.fail("dynamic shape sync"))
    tolist = torch.Tensor.tolist

    def cpu_tolist(tensor):
        assert not tensor.is_cuda, "async selection called CUDA tolist"
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", cpu_tolist)
    actual = policy.select_actual_blocks_device("r", 0, query, 12, 192, 193, 16, 2)
    assert actual.history_blocks == expected
    assert actual.attention.logical_block_indices[:actual.attention.count].cpu().tolist() == [*expected, 12]
    policy.observe_query("r", 0, query)
    monkeypatch.setattr(policy, "_score_pages", lambda *a: pytest.fail("duplicate scoring"))
    assert policy.predict_restore_blocks("r", 0, 12, 2) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_history_metadata_stays_on_gpu_across_requests():
    policy = SolidAttentionPolicy(1, 1)
    policy.register_page_representatives("page", 0, _representatives(12))
    shared = None
    for request in ("a", "b"):
        policy.bind_page_index_key(request, "page")
        policy.select_actual_blocks_device(
            request, 0, torch.ones(2, 2, device="cuda"), 12, 191, 192, 16, 2)
        state = policy._prediction_representatives["page"][0]
        assert state.representatives.is_cuda
        if shared is not None:
            assert state is shared
        shared = state
        private = policy._representatives[request][0][1]
        assert private.data_ptr() == state.representatives.data_ptr()
        assert private.shape[0] == 11
        policy.discard(request)
    assert policy._prediction_representatives["page"][0] is shared


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_prefix_prediction_reuses_only_complete_actual_history(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE", True)
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS", 0)
    torch.manual_seed(37)
    policy = SolidAttentionPolicy(1, 1)
    reps = torch.randn(12, 2, 2, 2)
    query = torch.randn(2, 2, device="cuda")
    policy.register_page_representatives("page", 0, reps)
    policy.bind_page_index_key("r", "page")
    actual = policy.select_actual_blocks_device("r", 0, query, 12, 191, 192, 16, 2)
    policy.observe_query("r", 0, query)
    monkeypatch.setattr(policy, "_score_pages", lambda *a: pytest.fail("duplicate scoring"))
    monkeypatch.setattr(torch, "nonzero", lambda *a, **kw: pytest.fail("dynamic shape sync"))
    assert policy.predict_restore_blocks("r", 0, 12, 2) == tuple(
        block for block in actual.history_blocks if block < 11)
    assert actual.history_blocks[-1] == 11


@pytest.mark.parametrize("tokens", [1, 4])
def test_chunk_feedback_uses_actual_immutable_subset_and_cleans_up(monkeypatch, tokens):
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE", True)
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS", 0)
    policy = SolidAttentionPolicy(1, 1)
    policy.register_page_representatives("page", 0, _representatives(12))
    policy.bind_page_index_key("r", "page")
    assert policy.predict_restore_blocks("r", 0, 8, 2) is None
    policy._representatives["r"][0] = (12, _representatives(12))
    query = torch.randn(tokens, 2, 2)
    actual = policy.select_actual_blocks_device("r", 0, query, 8, 192, 192 + tokens, 16, 2)
    policy.observe_query("r", 0, query[-1:])
    monkeypatch.setattr(policy, "_score_pages", lambda *a: pytest.fail("duplicate scoring"))
    assert policy.predict_restore_blocks("r", 0, 8, 2) == tuple(
        block for block in actual.history_blocks if block < 8)
    assert "r" not in policy._previous_queries
    policy.discard("r")
    assert policy.predict_restore_blocks("r", 0, 8, 2) is None


def test_empty_actual_prediction_is_valid_feedback(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE", True)
    monkeypatch.setattr("vllm.envs.VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS", 0)
    policy = SolidAttentionPolicy(0, 0)
    configure_sparse_kv_policy(policy)
    try:
        register_sparse_restore_context("empty", "prefix", 4, (0, 1))
        assert build_sparse_restore_plan_feedback("empty", 1) is None
        policy._actual_predictions["empty", 0] = (4, 1, ())
        feedback = build_sparse_restore_plan_feedback("empty", 1)
        assert feedback.block_indices_by_layer == ((),)
    finally:
        discard_sparse_restore_context(("empty",))
        configure_sparse_kv_policy(None)


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
    assert first.count <= 6
    assert second.count <= first.count
    assert first.logical_block_indices.data_ptr() == (
        second.logical_block_indices.data_ptr())

    grown = policy.select_attention_blocks_device("r", 0, query, key_cache,
                                                  physical_ids, 48, 4, 8)
    assert grown.count > first.count
    assert grown.logical_block_indices.numel() >= grown.count
    policy.discard("r")
    assert "r" not in policy._dynamic_selection_buffers


def test_resident_decode_reuses_buffer_but_scores_each_query(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    query = torch.ones(1, 2, 4, device="cuda")
    key_cache = torch.randn(16, 2, 4, 4, device="cuda")
    physical_ids = torch.arange(16, dtype=torch.long, device="cuda")

    scores = []
    score = policy._score_pages
    monkeypatch.setattr(policy, "_score_pages",
                        lambda *args: scores.append(args[0]) or score(*args))
    first = policy.select_attention_blocks_device(
        "r", 0, -query, key_cache, physical_ids, 25, 4, 2)
    current = policy.select_attention_blocks_device(
        "r", 0, query, key_cache, physical_ids, 26, 4, 2)
    assert len(scores) == 2
    assert first.logical_block_indices.data_ptr() == current.logical_block_indices.data_ptr()
    representatives = policy._representatives["r"][0][1]
    expected = torch.cat((policy._select_prefix(query[0], representatives, 6, 2),
                          torch.tensor([6], device=query.device)))
    torch.testing.assert_close(current.logical_block_indices[:current.count].long(), expected)


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
    assert policy._representatives["r"][0][0] == 7


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


def test_solidattention_prediction_uses_stable_page_key():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.bind_page_index_key("request", "prefix")
    policy.register_page_representatives(
        "prefix", 0, _representatives(4), tuple(range(4)))
    policy.observe_query("request", 0, torch.ones(2, 2))

    prediction = policy.predict_restore_blocks("request", 0, 4, 1)

    assert prediction is not None
    assert prediction == tuple(sorted(set(prediction)))
    assert set(prediction).issubset(set(range(4)))
    policy.discard("request")
    assert "request" not in policy._page_index_keys


def test_solidattention_actual_selector_returns_prefix_only():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.register_page_representatives(
        "prefix", 0, _representatives(4), tuple(range(4)))
    policy.bind_page_index_key("request-actual", "prefix")

    selected = policy.select_actual_prefix_blocks(
        "request-actual", 0, torch.ones(2, 2), 4, 1)

    assert selected == tuple(sorted(set(selected)))
    assert 0 in selected and 3 in selected
    assert all(index < 4 for index in selected)


def test_solidattention_actual_device_selection_includes_suffix_once():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.register_page_representatives(
        "prefix", 0, _representatives(4).cuda(), tuple(range(4)))
    policy.bind_page_index_key("request-device", "prefix")

    selected = policy.select_actual_blocks_device(
        "request-device", 0, torch.ones(2, 2, device="cuda"), 4, 64, 96, 16, 1)

    indices = selected.attention.logical_block_indices[
        :selected.attention.count]
    assert indices.dtype == torch.int32
    assert indices.is_contiguous()
    assert selected.history_blocks == tuple(indices[:len(
        selected.history_blocks)].cpu().tolist())
    assert indices[-2:].cpu().tolist() == [4, 5]
    assert selected.attention.count == len(selected.history_blocks) + 2


def test_solidattention_prediction_guard_does_not_change_actual(monkeypatch):
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    representatives = torch.randn(12, 2, 2, 2)
    policy.bind_page_index_key("request-guard", "prefix")
    policy.register_page_representatives(
        "prefix", 0, representatives, tuple(range(12)))
    query = torch.randn(2, 2)
    policy.observe_query("request-guard", 0, query)
    actual_before = policy.select_actual_prefix_blocks(
        "request-guard", 0, query, 12, 1)

    monkeypatch.setattr(
        "vllm.envs.VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS", 0)
    prediction_without_guard = policy.predict_restore_blocks(
        "request-guard", 0, 12, 1)
    monkeypatch.setattr(
        "vllm.envs.VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS", 2)
    prediction_with_guard = policy.predict_restore_blocks(
        "request-guard", 0, 12, 1)
    actual_after = policy.select_actual_prefix_blocks(
        "request-guard", 0, query, 12, 1)

    assert prediction_without_guard is not None
    assert prediction_with_guard is not None
    assert set(prediction_without_guard).issubset(prediction_with_guard)
    assert len(prediction_with_guard) >= len(prediction_without_guard)
    assert actual_after == actual_before


def test_solidattention_prediction_respects_logical_representative_indices():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.bind_page_index_key("request", "prefix")
    policy.register_page_representatives(
        "prefix", 0, _representatives(4), (0, 2, 3, 4))
    policy.observe_query("request", 0, torch.ones(2, 2))

    prediction = policy.predict_restore_blocks("request", 0, 5, 1)

    assert prediction is not None
    assert set(prediction).issubset({0, 2, 3, 4})
    assert max(prediction) < 5


def test_solidattention_residency_check_bypasses_request_warmup():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.bind_page_index_key("request", "prefix")
    policy.register_page_representatives(
        "prefix", 0, _representatives(4), tuple(range(4)))

    selected = policy.select_blocks_for_residency_check(
        "request", 0, torch.ones(2, 2), 5, 1)

    assert selected != tuple(range(5))
    assert selected[-1] == 4


def test_solidattention_feedback_compiles_per_layer_prefix_plan():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    configure_sparse_kv_policy(policy)
    try:
        policy.bind_page_index_key("request", "prefix")
        for layer_index in range(2):
            policy.register_page_representatives(
                "prefix", layer_index, _representatives(4), tuple(range(4)))
            policy.observe_query("request", layer_index, torch.ones(2, 2))
            register_sparse_restore_context("request", "prefix", 4,
                                            (layer_index, layer_index + 1))

        feedback = build_sparse_restore_plan_feedback("request", 1)

        assert feedback is not None
        assert feedback.page_index_key == "prefix"
        assert feedback.num_layers == 2
        assert feedback.source == "solidattention_dynamic"
        assert all(max(selection) < 4
                   for selection in feedback.block_indices_by_layer)
    finally:
        discard_sparse_restore_context(("request", ))
        configure_sparse_kv_policy(None)


@pytest.mark.parametrize("query_heads,kv_heads,budget", [
    (4, 4, 1),
    (8, 2, 2),
    (8, 1, 8),
])
def test_solidattention_selector_matches_independent_reference(
        query_heads, kv_heads, budget):
    torch.manual_seed(11)
    policy = SolidAttentionPolicy(init_blocks=2, local_blocks=2)
    query = torch.randn(query_heads, 4)
    representatives = torch.randn(10, 2, kv_heads, 4)

    actual = policy._select_prefix(query, representatives, 10, budget)
    expected = reference_solidattention_selection(
        query, representatives, 10, budget, init_blocks=2, local_blocks=2)

    assert tuple(actual.tolist()) == expected


def test_solidattention_selector_reference_handles_ties_and_logical_gaps():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    query = torch.ones(2, 2)
    representatives = torch.ones(5, 2, 2, 2)
    logical = (0, 2, 4, 5, 7)

    actual = policy._select_prefix_with_logical_indices(
        query, representatives, logical, 8, 2)
    expected = reference_solidattention_selection(
        query,
        representatives,
        8,
        2,
        init_blocks=1,
        local_blocks=1,
        logical_block_indices=logical,
    )

    assert tuple(actual.tolist()) == expected
    assert actual.tolist() == sorted(set(actual.tolist()))


def _validation_metadata(prompt: str = "prompt"):
    return {
        "model": "Qwen2.5-7B-Instruct",
        "prompt_sha256": prompt,
        "block_size": 16,
        "prefix_blocks": 3,
        "num_layers": 2,
        "page_identity": "page",
        "capture_trace": True,
        "capture_layer_data": True,
        "retain_report": True,
    }


def test_solidattention_validation_all_blocks_and_request_steps():
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.configure_validation("request", "all_blocks",
                                _validation_metadata())

    first = policy._apply_validation_selection(
        "request", 0, torch.tensor([0, 2]), 3, 4, 1, 3)
    second_layer = policy._apply_validation_selection(
        "request", 1, torch.tensor([0, 2]), 3, 4, 1, 3)
    next_step = policy._apply_validation_selection(
        "request", 0, torch.tensor([0, 2]), 3, 4, 1, 3)
    report = policy.validation_report("request")

    assert first.tolist() == [0, 1, 2]
    assert second_layer.tolist() == [0, 1, 2]
    assert next_step.tolist() == [0, 1, 2]
    assert [(item["decode_step"], item["layer_index"])
            for item in report["records"]] == [(0, 0), (0, 1), (1, 0)]
    assert all(item["attention_blocks"] == [0, 1, 2, 3]
               for item in report["records"])


def test_resident_validation_keeps_trace_prefix_boundary(monkeypatch):
    policy = SolidAttentionPolicy()
    policy.configure_validation("request", "record", _validation_metadata())
    monkeypatch.setattr(policy, "_cached_representatives",
                        lambda request, layer, cache, ids, count, device:
                        torch.zeros(count, 2, 1, 4))
    for length in (12, 13):
        result = policy._prepare_device_selection(
            "request", 0, torch.ones(1, 1, 4), torch.empty(4, 1, 1, 4, 4),
            torch.arange((length + 3) // 4), length, 4, 1)
        assert result[2] == 3
    result = policy._prepare_device_selection(
        "normal", 0, torch.ones(1, 1, 4), torch.empty(4, 1, 1, 4, 4),
        torch.arange(3), 12, 4, 1)
    assert result[2] == 2


def test_solidattention_validation_record_replay_checks_kv_and_attention():
    metadata = _validation_metadata()
    recorder = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    recorder.configure_validation("record", "record", metadata)
    recorder._apply_validation_selection(
        "record", 0, torch.tensor([0, 2]), 3, 4, 1, 3)
    key_cache = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    value_cache = key_cache + 100
    block_table = torch.arange(4, dtype=torch.int32).view(1, -1)
    output = torch.tensor([[1.0, 2.0]])
    recorder.record_validation_attention("record", 0, key_cache, value_cache,
                                         block_table, output)
    trace = recorder.validation_report("record")

    replay = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    replay.configure_validation("replay", "replay", metadata,
                                trace["records"], trace["metadata"])
    chosen = replay._apply_validation_selection(
        "replay", 0, torch.tensor([0, 1]), 3, 4, 1, 3)
    replay.record_validation_attention("replay", 0, key_cache, value_cache,
                                       block_table, output)
    report = replay.validation_report("replay")

    assert chosen.tolist() == [0, 2]
    assert report["replay_selection_mismatches"] == 0
    assert report["kv_fingerprint_mismatches"] == 0
    assert report["attention_output_mismatches"] == 0

    replay_bad = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    replay_bad.configure_validation("replay-bad", "replay", metadata,
                                    trace["records"], trace["metadata"])
    replay_bad._apply_validation_selection(
        "replay-bad", 0, torch.tensor([0, 2]), 3, 4, 1, 3)
    changed_key = key_cache.clone()
    changed_key[2, 0, 0] += 1
    replay_bad.record_validation_attention("replay-bad", 0, changed_key,
                                           value_cache, block_table,
                                           output + 0.1)
    bad_report = replay_bad.validation_report("replay-bad")
    assert bad_report["kv_fingerprint_mismatches"] == 1
    assert bad_report["attention_output_mismatches"] == 1
    assert bad_report["first_mismatch_step"] == 0
    assert bad_report["first_mismatch_layer"] == 0


def test_solidattention_replay_rejects_metadata_mismatch():
    metadata = _validation_metadata()
    record = {
        "decode_step": 0,
        "layer_index": 0,
        "prefix_blocks": 3,
        "selected_prefix_blocks": [0, 2],
        "attention_blocks": [0, 2, 3],
        "block_budget": 1,
        "page_identity": "page",
        "representative_count": 3,
    }
    mismatched = dict(metadata, block_size=32)
    policy = SolidAttentionPolicy()
    with pytest.raises(ValueError, match="block_size"):
        policy.configure_validation("request", "replay", mismatched, [record],
                                    metadata)
    with pytest.raises(ValueError, match="position cache"):
        policy.configure_validation(
            "request", "replay", dict(metadata, position_cache_length=33792),
            [record], metadata)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_backing_selection_control_uses_query_not_cpu_synchronize(monkeypatch):
    monkeypatch.setenv("VLLM_GRANULEKV_HOST_BACKING_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE", "1")
    real_event = torch.cuda.Event

    class QueryOnlyEvent:
        def __init__(self, **kwargs):
            self.event = real_event(**kwargs)

        def record(self, stream):
            self.event.record(stream)

        def query(self):
            return self.event.query()

        def synchronize(self):
            pytest.fail("selection control must use event query")

    monkeypatch.setattr(torch.cuda, "Event", QueryOnlyEvent)
    policy = SolidAttentionPolicy()
    assert policy._selection_indices_to_host(
        torch.tensor([0, 3, 7], device="cuda")) == (0, 3, 7)


def test_solidattention_all_block_validation_compares_dense_attention():
    metadata = dict(_validation_metadata(), compare_dense_attention=True)
    policy = SolidAttentionPolicy(init_blocks=1, local_blocks=1)
    policy.configure_validation("request", "all_blocks", metadata)
    policy._apply_validation_selection(
        "request", 0, torch.tensor([0, 2]), 3, 4, 1, 3)
    key_cache = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    value_cache = key_cache + 100
    block_table = torch.arange(4, dtype=torch.int32).view(1, -1)
    selected_output = torch.tensor([[1.0, 2.0]])

    policy.record_validation_attention(
        "request", 0, key_cache, value_cache, block_table, selected_output,
        selected_output + 0.5)
    report = policy.validation_report("request")

    assert report["attention_output_mismatches"] == 1
    assert report["all_block_attention_rel_error"] > 0
    assert report["all_block_attention_cosine"] < 1


def test_validation_defers_comparison_without_waiting(monkeypatch):
    class Ready:
        done = False

        def query(self):
            return self.done

    ready = Ready()
    policy = SolidAttentionPolicy()
    monkeypatch.setattr(policy, "_validation_event", lambda _: ready)
    monkeypatch.setattr(policy, "_selection_indices_to_host",
                        lambda _: pytest.fail("validation added a selection wait"))
    policy.configure_validation("request", "record", _validation_metadata())
    policy._apply_validation_selection(
        "request", 0, torch.tensor([0, 2]), 3, 4, 1, 3)
    cache = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    output = torch.tensor([[1., 2.]])
    policy.record_validation_attention("request", 0, cache, cache,
                                       torch.arange(4).view(1, -1), output)
    cache.zero_()
    output.zero_()
    with pytest.raises(RuntimeError, match="snapshot pending"):
        policy.validation_report("request")
    ready.done = True
    report = policy.validation_report("request")
    assert report["attention_checks"] == 1
    record = report["records"][0]
    assert record["physical_blocks"] == [0, 2, 3]
    torch.testing.assert_close(policy._tensor_from_payload(record["attention_output"]),
                               torch.tensor([[1., 2.]]))
    assert report == policy.validation_report("request")


def test_validation_off_does_not_allocate_snapshots(monkeypatch):
    policy = SolidAttentionPolicy()
    monkeypatch.setattr(policy, "_validation_snapshot",
                        lambda _: pytest.fail("off mode allocated a snapshot"))
    selected = torch.tensor([0, 2])
    assert policy._apply_validation_selection(
        "request", 0, selected, 3, 4, 1, 3) is selected


def test_validation_fingerprint_ignores_unused_live_slots():
    policy = SolidAttentionPolicy()
    metadata = dict(_validation_metadata(), prefix_blocks=1)
    policy.configure_validation("record", "record", metadata)
    policy._apply_validation_selection(
        "record", 0, torch.tensor([0]), 1, 2, 1, 1)
    key = torch.zeros(2, 1, 1, 16, 8)
    value = torch.zeros(2, 1, 8, 16)
    table = torch.arange(2).view(1, -1)
    output = torch.ones(1, 8)
    policy.record_validation_attention("record", 0, key, value, table,
                                       output, sequence_length=17)
    trace = policy.validation_report("record")
    policy.configure_validation("replay", "replay", metadata,
                                trace["records"], trace["metadata"])
    policy._apply_validation_selection(
        "replay", 0, torch.tensor([0]), 1, 2, 1, 1)
    key[1, ..., 1:, :] = 123
    value[1, ..., 1:] = 456
    policy.record_validation_attention("replay", 0, key, value, table,
                                       output, sequence_length=17)
    assert policy.validation_report("replay")["kv_fingerprint_mismatches"] == 0


def test_replay_does_not_score_or_require_page_representatives(monkeypatch):
    policy = SolidAttentionPolicy()
    metadata = _validation_metadata()
    record = {"decode_step": 0, "layer_index": 0, "prefix_blocks": 3,
              "selected_prefix_blocks": [0, 2], "attention_blocks": [0, 2, 3]}
    policy.configure_validation("request", "replay", metadata, [record], metadata)
    monkeypatch.setattr(policy, "_select_prefix_with_logical_indices",
                        lambda *args: pytest.fail("replay recomputed selection"))
    actual = policy.select_actual_prefix_blocks("request", 0, torch.ones(1, 2),
                                                3, 1)
    assert actual == (0, 2)


def test_validation_logits_deferred_and_compared():
    metadata = _validation_metadata()
    policy = SolidAttentionPolicy()
    table = torch.arange(4).view(1, -1)
    cache = torch.zeros(4, 1, 4)
    output = torch.ones(1, 2)
    logits = torch.tensor([[1., 3., 2.]])
    policy.configure_validation("record", "record", metadata)
    policy._apply_validation_selection("record", 0, torch.tensor([0, 2]), 3, 4, 1, 3)
    policy.record_validation_attention("record", 0, cache, cache, table, output)
    policy.record_validation_logits("record", logits)
    trace = policy.validation_report("record")
    assert trace["logit_checks"] == 1
    policy.configure_validation("replay", "replay", metadata, trace["records"], metadata)
    policy._apply_validation_selection("replay", 0, torch.tensor([0]), 3, 4, 1, 3)
    policy.record_validation_attention("replay", 0, cache, cache, table, output)
    policy.record_validation_logits("replay", torch.tensor([[1., 2., 4.]]))
    report = policy.validation_report("replay")
    assert report["logit_mismatches"] == 1
    assert report["logit_rel_error"] > 0
    assert not report["greedy_token_match"]


def test_validation_mapping_error_reports_physical_block():
    metadata = _validation_metadata()
    policy = SolidAttentionPolicy()
    cache = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    output = torch.ones(1, 2)
    policy.configure_validation("record", "record", metadata)
    policy._apply_validation_selection("record", 0, torch.tensor([0, 2]), 3, 4, 1, 3)
    policy.record_validation_attention("record", 0, cache, cache,
                                       torch.arange(4).view(1, -1), output)
    trace = policy.validation_report("record")
    policy.configure_validation("replay", "replay", metadata, trace["records"], metadata)
    policy._apply_validation_selection("replay", 0, torch.tensor([0]), 3, 4, 1, 3)
    policy.record_validation_attention("replay", 0, cache, cache,
                                       torch.tensor([[1, 0, 2, 3]]), output)
    report = policy.validation_report("replay")
    assert report["first_mismatch"] == {
        "decode_step": 0, "layer_index": 0,
        "page_identity": metadata["page_identity"],
        "logical_block": 0, "physical_block": 1}
