"""Tests for the generic user-owned sparse KV metadata bridge."""

import pytest
import torch

from vllm.attention.ops.sparse_kv import (
    SparseKVSelection, build_page_representatives_from_paged_key_cache,
    build_selected_decode_block_table, reset_sparse_kv_stats,
    select_and_compact_decode_blocks, sparse_kv_stats,
    validate_sparse_kv_prediction)


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
    table = torch.tensor([[11, 22, 33, 44]], dtype=torch.int32)
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
        register_page_representatives=lambda *args: (_ for _ in ()).throw(
            AssertionError("CPU metadata path should not be called")),
        select_blocks_device=lambda *args: torch.tensor(
            [0, 2, 3], dtype=torch.long),
        resident_blocks=(0, 1, 2, 3),
    )
    assert isinstance(selected, torch.Tensor)
    assert selected.tolist() == [0, 2, 3]
    assert selected_table.tolist() == [[11, 33, 44]]
    assert tokens == 10


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
