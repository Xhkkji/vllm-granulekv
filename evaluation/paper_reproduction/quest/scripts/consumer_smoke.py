"""Run the restricted batch-one decode consumer against a shared plan."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation.paper_reproduction.common.plans import LayerWisePrefetcher
from evaluation.paper_reproduction.quest.adapter import (
    QuestDecodeOnlyConsumer, QuestSelector, build_page_representatives,
    dense_attention, )
from vllm.core.custom_schedulers.hierarchical_io.barrier import (
    activate_sparse_kv_blocks, )


def main() -> None:
    torch.manual_seed(17)
    layers, heads, tokens, dim = 4, 3, 32, 8
    page_size, budget = 4, 2
    query = torch.randn(1, heads, dim)
    keys = torch.randn(heads, tokens, dim)
    values = torch.randn_like(keys)
    representatives = build_page_representatives(keys, page_size, query)
    selector = QuestSelector(page_size=page_size)
    plan = selector.build_access_plan_from_queries(
        (query, ) * layers, (representatives, ) * layers,
        num_blocks=tokens // page_size, block_budget=budget)
    prefetch = LayerWisePrefetcher(2).build("quest-consumer", layers,
                                             tokens // page_size, plan)
    consumer = QuestDecodeOnlyConsumer(plan, page_size)
    layer = 0
    active = prefetch.units[0].block_indices
    with activate_sparse_kv_blocks(active):
        sparse = consumer.attend_from_context(query, keys, values, layer)
    dense = dense_attention(query, keys, values).unsqueeze(0)
    payload = {
        "strategy": "quest_decode_only_reference",
        "layer": layer,
        "selected_blocks": list(consumer.blocks_for_layer(layer)),
        "active_window_blocks": list(active or ()),
        "output_shape": list(sparse.shape),
        "dense_output_shape": list(dense.shape),
        "relative_output_difference": float(
            (torch.linalg.vector_norm(sparse - dense) /
             torch.linalg.vector_norm(dense).clamp_min(1e-12)).item()),
        "plan_window_ranges": [list(unit.layer_range) for unit in prefetch.units],
    }
    output = Path(
        "evaluation/paper_reproduction/quest/results/consumer_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

