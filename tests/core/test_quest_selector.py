"""Unit tests for the pure PyTorch Quest selector boundary."""

import pytest
import torch

from evaluation.paper_reproduction.common.plans import LayerWisePrefetcher
from evaluation.paper_reproduction.quest.adapter import (
    QuestDecodeOnlyConsumer, QuestSelector, build_page_representatives,
    dense_attention,
    evaluate_selection, exact_page_scores, quest_page_scores,
    selected_attention, union_topk_pages)


def test_gpu_selector_keeps_selection_on_device():
    try:
        query = torch.tensor([[2.0, -3.0], [-1.0, 4.0]], device="cuda")
        representatives = torch.tensor([
            [[1.0, 2.0], [3.0, 1.0]],
            [[2.0, 1.0], [1.0, 2.0]],
        ], device="cuda")
    except RuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    selected = QuestPageSelector().select_tensor_from_query(
        query, representatives, 1)
    assert selected.device.type == "cuda"
    assert selected.tolist() == [0, 1]


def test_page_score_matches_reference_formula():
    query = torch.tensor([[2.0, -3.0], [-1.0, 4.0]])
    representatives = torch.tensor([
        [[1.0, 2.0], [3.0, 1.0]],
        [[2.0, 1.0], [1.0, 2.0]],
    ])
    expected = torch.tensor([[8.0, 7.0], [7.0, 9.0]])
    torch.testing.assert_close(quest_page_scores(query, representatives),
                               expected)


def test_query_independent_min_max_envelope_matches_direct_page_bound():
    query = torch.tensor([[2.0, -3.0]])
    keys = torch.tensor([[[1.0, 2.0], [4.0, -5.0], [-2.0, 3.0],
                          [6.0, 1.0]]])
    envelope = build_page_representatives(keys, page_size=2)
    assert envelope.shape == (2, 2, 1, 2)
    expected = exact_page_scores(query, keys, page_size=2)
    torch.testing.assert_close(quest_page_scores(query, envelope), expected)


def test_topk_union_is_sorted_and_has_stable_ties():
    scores = torch.tensor([[1.0, 3.0, 3.0, 0.0], [4.0, 2.0, 2.0, 1.0]])
    assert union_topk_pages(scores, 2) == (0, 1, 2)
    assert union_topk_pages(scores, 20) == (0, 1, 2, 3)


def test_dynamic_selector_builds_layer_plan_and_window_union():
    torch.manual_seed(3)
    query = torch.randn(2, 4)
    keys = torch.randn(2, 12, 4)
    representatives = build_page_representatives(keys, 3, query)
    selector = QuestSelector(page_size=3)
    plan = selector.build_access_plan_from_queries(
        (query, query), (representatives, representatives), 4, 1)
    assert plan.source == "quest_query_aware"
    assert len(plan.block_indices_by_layer) == 2
    prefetch = LayerWisePrefetcher(1).build("quest", 2, 4, plan)
    assert [unit.block_indices for unit in prefetch.units] == [
        plan.block_indices_by_layer[0], plan.block_indices_by_layer[1]
    ]


def test_exact_scores_and_attention_metrics_are_finite():
    torch.manual_seed(4)
    query = torch.randn(2, 4)
    keys = torch.randn(2, 12, 4)
    values = torch.randn(2, 12, 4)
    scores = exact_page_scores(query, keys, 3)
    selected = union_topk_pages(scores, 2)
    dense = dense_attention(query, keys, values)
    sparse = selected_attention(query, keys, values, selected, 3)
    assert dense.shape == sparse.shape == (2, 4)
    metrics = evaluate_selection(query, keys, values, 3, selected, 2)
    assert metrics.selector_recall == pytest.approx(1.0)
    assert metrics.attention_output_error >= 0.0


def test_invalid_dynamic_inputs_fail_explicitly():
    query = torch.ones(2, 4)
    reps = torch.ones(3, 2, 4)
    with pytest.raises(ValueError, match="positive"):
        QuestSelector().select_from_query(query, reps, 0, 0)
    with pytest.raises(ValueError, match="different dimensions"):
        quest_page_scores(query, torch.ones(3, 2, 5))
    with pytest.raises(ValueError, match="batch size"):
        quest_page_scores(torch.ones(2, 2, 4), reps)


def test_same_input_produces_same_selection():
    torch.manual_seed(11)
    query = torch.randn(3, 8)
    reps = torch.rand(9, 3, 8)
    selector = QuestSelector()
    assert selector.select_from_query(query, reps, 3, 1) == selector.select_from_query(
        query, reps, 3, 1)


def test_decode_consumer_uses_plan_and_active_sparse_context():
    from vllm.core.custom_schedulers.hierarchical_io.barrier import (
        activate_sparse_kv_blocks, )

    torch.manual_seed(12)
    query = torch.randn(1, 2, 4)
    keys = torch.randn(2, 12, 4)
    values = torch.randn_like(keys)
    representatives = build_page_representatives(keys, 3, query)
    selector = QuestSelector(page_size=3)
    plan = selector.build_access_plan_from_queries(
        (query, ), (representatives, ), 4, 1)
    consumer = QuestDecodeOnlyConsumer(plan, page_size=3)
    selected = consumer.blocks_for_layer(0)
    with activate_sparse_kv_blocks(selected):
        output = consumer.attend_from_context(query, keys, values, 0)
    assert output.shape == (1, 2, 4)

    with activate_sparse_kv_blocks(()) if not selected else activate_sparse_kv_blocks(
            selected[:-1]):
        with pytest.raises(RuntimeError, match="not resident"):
            consumer.attend_from_context(query, keys, values, 0)
