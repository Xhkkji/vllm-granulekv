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
        "request-device", 0, torch.ones(2, 2, device="cuda"), 4, 6, 1)

    indices = selected.attention.logical_block_indices[
        :selected.attention.count]
    assert indices.dtype == torch.int32
    assert indices.is_contiguous()
    assert selected.prefix_blocks == tuple(indices[:len(
        selected.prefix_blocks)].cpu().tolist())
    assert indices[-2:].cpu().tolist() == [4, 5]
    assert selected.attention.count == len(selected.prefix_blocks) + 2


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
