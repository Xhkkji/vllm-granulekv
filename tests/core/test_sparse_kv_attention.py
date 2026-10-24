"""Tests for the generic user-owned sparse KV metadata bridge."""

import pytest
import torch

from vllm.attention.ops.sparse_kv import (
    SparseKVSelection, build_page_representatives_from_paged_key_cache,
    build_selected_decode_block_table, build_sparse_kv_prediction_gap,
    reset_sparse_kv_stats,
    select_and_compact_decode_blocks, sparse_kv_stats,
    validate_sparse_kv_prediction)


@pytest.mark.parametrize("with_output", [False, True])
def test_host_window_selection_precedes_single_query_metadata(monkeypatch, with_output):
    from types import SimpleNamespace
    import vllm.attention.layer as module
    from vllm.core.custom_schedulers.hierarchical_io import (
        activate_layer_barrier, activate_sparse_kv_blocks)

    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_HOST_BACKING_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE", False)
    converted = object()
    original = SimpleNamespace(num_prefills=1, num_prefill_tokens=1, num_decode_tokens=0,
                               single_token_sparse_prefill=lambda: converted)
    seen = []
    layer = SimpleNamespace(num_heads=1, head_size=2, kv_cache=[None],
                            impl=SimpleNamespace(forward=lambda *args, **kw: seen.append(args[5])))
    name = "model.layers.4.self_attn.attn"
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(
        attn_metadata=original, no_compile_layers={name: layer}, virtual_engine=0))
    monkeypatch.setattr(module, "wait_for_kv_layer_from_connector", lambda *a: None)
    monkeypatch.setattr(module, "maybe_save_kv_layer_to_connector", lambda *a: None)
    monkeypatch.setattr(module, "_observe_decode_query", lambda *a: None)
    tensor = torch.ones(1, 1, 2)
    with activate_layer_barrier(lambda *a: None, virtual_engine=0,
                               request_ids=("req",),
                               correction_callback=lambda *a: (0, 4)):
        with activate_sparse_kv_blocks(None):
            if with_output:
                module.unified_attention_with_output(tensor, tensor, tensor, tensor, name)
            else:
                module.unified_attention(tensor, tensor, tensor, name)
    assert seen == [converted]


def test_compact_block_table_accepts_gpu_selection():
    try:
        table = torch.tensor([[11, 22, 33, 44]],
                             dtype=torch.int32,
                             device="cuda")
        selected = torch.tensor([2, 0, 2, 3],
                                dtype=torch.long,
                                device="cuda")
    except RuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    compact, tokens = build_selected_decode_block_table(table, selected, 14, 4)
    assert compact.tolist() == [[11, 33, 44]]
    assert tokens == 10
    assert table.tolist() == [[11, 22, 33, 44]]


def test_logical_blocks_map_to_physical_blocks_without_mutating_table():
    table = torch.tensor([[11, 22, 33, 44]], dtype=torch.int32)
    compact, tokens = build_selected_decode_block_table(table, (2, 0, 2), 14,
                                                        4)
    assert compact.tolist() == [[11, 33]]
    assert tokens == 8
    assert table.tolist() == [[11, 22, 33, 44]]


def test_partial_last_block_is_counted_once():
    table = torch.tensor([5, 6, 7], dtype=torch.int32)
    compact, tokens = build_selected_decode_block_table(table, (2, ), 10, 4)
    assert compact.tolist() == [7]
    assert tokens == 2


def test_dynamic_restore_selection_uses_frozen_plan():
    table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    key_cache = torch.arange(4 * 1 * 4 * 1,
                             dtype=torch.float32).reshape(4, 1, 4, 1)
    selected_table, tokens, selected = select_and_compact_decode_blocks(
        block_table=table,
        key_cache=key_cache,
        query=torch.ones(1, 1, 1),
        sequence_length=14,
        block_size=4,
        request_id="request",
        layer_index=0,
        block_budget=1,
        select_blocks=lambda *args: (1, ),
        register_page_representatives=lambda *args: None,
        resident_blocks=(0, 2, 3),
        selected_blocks_override=(0, 2, 3),
    )
    assert selected == (0, 2, 3)
    assert selected_table.tolist() == [[0, 2, 3]]
    assert tokens == 10


def test_gpu_selector_callback_keeps_tensor_selection_in_bridge():
    table = torch.tensor([[1, 3, 0, 2]], dtype=torch.int32)
    metadata = []
    key_cache = torch.arange(4 * 1 * 4 * 1,
                             dtype=torch.float32).reshape(4, 1, 4, 1)
    selected_table, tokens, selected = select_and_compact_decode_blocks(
        block_table=table,
        key_cache=key_cache,
        query=torch.ones(1, 1, 1),
        sequence_length=14,
        block_size=4,
        request_id="request",
        layer_index=0,
        block_budget=1,
        select_blocks=lambda *args: (_ for _ in ()).throw(
            AssertionError("CPU selector should not be called")),
        register_page_representatives=lambda *args: metadata.append(args),
        select_blocks_device=lambda *args: torch.tensor(
            [0, 2, 3], dtype=torch.long),
        resident_blocks=(0, 1, 2, 3),
    )
    assert isinstance(selected, torch.Tensor)
    assert selected.tolist() == [0, 2, 3]
    assert selected_table.tolist() == [[1, 0, 2]]
    assert tokens == 10
    assert len(metadata) == 1
    assert metadata[0][:2] == ("request", 0)


def test_mapping_rejects_batch_two_and_invalid_selection():
    with pytest.raises(ValueError, match="batch size 1"):
        build_selected_decode_block_table(torch.ones(2, 3, dtype=torch.int32),
                                          (0, ), 4, 4)
    with pytest.raises(IndexError, match="outside"):
        build_selected_decode_block_table(torch.ones(1, 2, dtype=torch.int32),
                                          (2, ), 8, 4)
    with pytest.raises(ValueError, match="must not be empty"):
        build_selected_decode_block_table(torch.ones(1, 2, dtype=torch.int32),
                                          (), 8, 4)


def test_selection_dataclass_requires_sorted_unique_indices():
    assert SparseKVSelection((0, 2), "test").logical_block_indices == (0, 2)
    with pytest.raises(ValueError, match="sorted and unique"):
        SparseKVSelection((2, 1), "test")


def test_prediction_validation_records_miss_without_dense_fallback():
    reset_sparse_kv_stats()
    with pytest.raises(RuntimeError, match="prediction_miss"):
        validate_sparse_kv_prediction(
            predicted_prefix_blocks=(0, 2, 3),
            actual_prefix_blocks=(0, 1, 3),
            resident_blocks=(0, 2, 3),
            num_prefix_blocks=4,
            request_id="request",
            layer_index=2,
        )
    stats = sparse_kv_stats()
    assert stats["prediction_calls"] == 1
    assert stats["prediction_miss"] == 1
    assert stats["prediction_miss_blocks"] == 1
    assert stats["prediction_wasted_blocks"] == 1


def test_prediction_gap_keeps_control_plane_sets_distinct():
    reset_sparse_kv_stats()
    gap = build_sparse_kv_prediction_gap(
        predicted_prefix_blocks=(0, 2),
        actual_prefix_blocks=(1, 2),
        resident_blocks=(0, 2, 3),
        num_prefix_blocks=4,
        request_id="request-gap",
        layer_index=3,
    )
    assert gap.predicted_prefix_blocks == (0, 2)
    assert gap.actual_prefix_blocks == (1, 2)
    assert gap.resident_prefix_blocks == (0, 2, 3)
    assert gap.hit_blocks == (2, )
    assert gap.missing_blocks == (1, )
    assert gap.wasted_blocks == (0, )
    stats = sparse_kv_stats()
    assert stats["prediction_checks"] == 1
    assert stats["prediction_gap_events"] == 1
    assert stats["prediction_recall"] == 0.5


def test_host_correction_does_not_inflate_prediction_recall():
    reset_sparse_kv_stats()
    gap = build_sparse_kv_prediction_gap(
        predicted_prefix_blocks=(0, 2), actual_prefix_blocks=(1, 2),
        resident_blocks=(0, 1, 2), num_prefix_blocks=4,
        request_id="r", layer_index=0)
    assert gap.missing_blocks == ()
    assert sparse_kv_stats()["prediction_recall"] == 0.5
    assert sparse_kv_stats()["prediction_miss_blocks"] == 1
    assert sparse_kv_stats()["prediction_miss"] == 1
    assert sparse_kv_stats()["prediction_gap_events"] == 0
    assert sparse_kv_stats()["resident_coverage"] == 1.0


def test_page_representatives_from_four_dimensional_key_cache():
    # [blocks, heads, tokens, dim]
    key_cache = torch.tensor([
        [[[1.0, -2.0], [3.0, 4.0]]],
        [[[-5.0, 6.0], [7.0, -8.0]]],
    ])
    envelope = build_page_representatives_from_paged_key_cache(
        key_cache, (1, 0))
    assert envelope.shape == (2, 2, 1, 2)
    torch.testing.assert_close(envelope[0, 0], torch.tensor([[-5.0, -8.0]]))
    torch.testing.assert_close(envelope[0, 1], torch.tensor([[7.0, 6.0]]))
    torch.testing.assert_close(envelope[1, 0], torch.tensor([[1.0, -2.0]]))
    torch.testing.assert_close(envelope[1, 1], torch.tensor([[3.0, 4.0]]))


def test_page_representatives_from_packed_five_dimensional_key_cache():
    # [blocks, heads, dim_chunks, tokens, chunk]
    key_cache = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]],
                                [[-1.0, 8.0], [5.0, -2.0]]]]])
    envelope = build_page_representatives_from_paged_key_cache(
        key_cache, (0, ))
    assert envelope.shape == (1, 2, 1, 4)
    torch.testing.assert_close(envelope[0, 0],
                               torch.tensor([[1.0, 2.0, -1.0, -2.0]]))
    torch.testing.assert_close(envelope[0, 1],
                               torch.tensor([[3.0, 4.0, 5.0, 8.0]]))
