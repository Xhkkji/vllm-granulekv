"""Deterministic Quest selector and shared-plan smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation.paper_reproduction.common.plans import LayerWisePrefetcher
from evaluation.paper_reproduction.quest.adapter import QuestSelector
from evaluation.paper_reproduction.quest.adapter.reference import (
    build_page_representatives, )


def main() -> None:
    torch.manual_seed(7)
    layers, heads, tokens, dim = 4, 3, 24, 8
    page_size, budget = 4, 2
    query = torch.randn(heads, dim)
    keys = torch.randn(heads, tokens, dim)
    representatives = build_page_representatives(keys, page_size, query)
    selector = QuestSelector(page_size=page_size)
    first = selector.select_from_query(query, representatives, budget, 0)
    second = selector.select_from_query(query, representatives, budget, 0)
    if first != second or not first:
        raise AssertionError("Quest selection is not deterministic")

    selections = tuple(first for _ in range(layers))
    plan = QuestSelector(selected_blocks_by_layer=selections).select(
        layers, representatives.shape[0])
    prefetch = LayerWisePrefetcher(2).build(
        "quest-selector-smoke", layers, representatives.shape[0], plan)
    if [unit.layer_range for unit in prefetch.units] != [(0, 2), (2, 4)]:
        raise AssertionError("unexpected layer-window plan")
    payload = {
        "strategy": "quest",
        "selected_blocks": list(first),
        "num_pages": representatives.shape[0],
        "window_ranges": [list(unit.layer_range) for unit in prefetch.units],
        "source": plan.source,
    }
    output = Path("evaluation/paper_reproduction/quest/results/selector_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
