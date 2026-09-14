"""Synthetic Quest selector evaluation with shared metric fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.paper_reproduction.quest.adapter import (
    QuestSelector, evaluate_selection, )
from evaluation.paper_reproduction.quest.adapter.reference import (
    build_page_representatives, )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--block-budget", type=int, default=0)
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path("evaluation/paper_reproduction/quest/results/selector_eval.json"))
    args = parser.parse_args()
    if args.layers <= 0 or args.heads <= 0 or args.tokens <= 0 or args.head_dim <= 0:
        raise ValueError("layers, heads, tokens, and head_dim must be positive")
    if args.block_budget <= 0 and args.token_budget <= 0:
        args.block_budget = 16
    elif args.block_budget <= 0:
        if args.token_budget <= 0:
            raise ValueError("token-budget must be positive")
        args.block_budget = (args.token_budget + args.page_size - 1) // args.page_size
    elif args.token_budget > 0:
        raise ValueError("use either block-budget or token-budget")
    torch.manual_seed(args.seed)
    query = torch.randn(args.heads, args.head_dim)
    key_cache = torch.randn(args.heads, args.tokens, args.head_dim)
    value_cache = torch.randn_like(key_cache)
    representatives = build_page_representatives(key_cache, args.page_size, query)
    selector = QuestSelector(page_size=args.page_size)
    selected = selector.select_from_query(query, representatives,
                                          args.block_budget, 0)
    metrics = evaluate_selection(query, key_cache, value_cache,
                                 args.page_size, selected, args.block_budget)
    payload = {
        "strategy": "quest_style_selector",
        "seed": args.seed,
        "layers": args.layers,
        "heads": args.heads,
        "tokens": args.tokens,
        "page_size": args.page_size,
        "block_budget": args.block_budget,
        "token_budget": (args.token_budget if args.token_budget > 0 else
                          args.block_budget * args.page_size),
        **metrics.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
