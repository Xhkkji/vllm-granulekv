#!/usr/bin/env python3
"""Offline SolidAttention prediction oracle.

The oracle keeps all KV pages available and compares a prediction made from
the current query with the next query's actual SolidAttention working set.  It
does not start vLLM, GranuleKV, or any SSD transfer path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vllm.attention.ops.sparse_kv import (
    build_page_representatives_from_paged_key_cache,
    reset_sparse_kv_stats,
    sparse_kv_stats,
)

from evaluation.paper_reproduction.solidattention.adapter import (
    SolidAttentionPolicy, )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-blocks", type=int, default=512)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--block-budget", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def main() -> None:
    args = _parse_args()
    if min(args.prefix_blocks, args.layers, args.block_size, args.block_budget,
           args.steps, args.query_heads, args.kv_heads, args.head_dim) <= 0:
        raise ValueError("oracle dimensions and budgets must be positive")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query-heads must be divisible by kv-heads")

    torch.manual_seed(args.seed)
    device = _device(args.device)
    policy = SolidAttentionPolicy(init_blocks=2, local_blocks=2)
    request_id = "solidattention-oracle-request"
    page_key = "solidattention-oracle-prefix"
    policy.bind_page_index_key(request_id, page_key)

    key_cache = torch.randn(
        args.prefix_blocks,
        args.kv_heads,
        args.block_size,
        args.head_dim,
        device=device,
    )
    physical_ids = torch.arange(args.prefix_blocks,
                                dtype=torch.long,
                                device=device)
    representatives = build_page_representatives_from_paged_key_cache(
        key_cache, physical_ids, output_device=torch.device("cpu"))
    for layer_index in range(args.layers):
        policy.register_page_representatives(
            page_key,
            layer_index,
            representatives,
            tuple(range(args.prefix_blocks)),
        )

    # The full KV table is the oracle residency set.  The extra block is the
    # current live suffix and is excluded from prediction statistics.
    num_blocks = args.prefix_blocks + 1
    queries = torch.randn(args.steps + 1,
                          args.layers,
                          args.query_heads,
                          args.head_dim)
    totals = {
        "predicted_blocks": 0,
        "actual_blocks": 0,
        "hit_blocks": 0,
        "miss_blocks": 0,
        "wasted_blocks": 0,
    }
    per_step = []
    warmup_blocks = num_blocks
    for step in range(args.steps):
        step_totals = dict(totals)
        for layer_index in range(args.layers):
            policy.observe_query(request_id, layer_index,
                                 queries[step, layer_index])
            predicted = policy.predict_restore_blocks(
                request_id, layer_index, args.prefix_blocks,
                args.block_budget)
            if predicted is None:
                raise RuntimeError(
                    "SolidAttention oracle produced no prediction after "
                    "query history was recorded")
            actual = policy.select_blocks_for_residency_check(
                request_id,
                layer_index,
                queries[step + 1, layer_index],
                num_blocks,
                args.block_budget,
            )
            actual_prefix = {index for index in actual
                             if index < args.prefix_blocks}
            predicted_set = set(predicted)
            hit = predicted_set.intersection(actual_prefix)
            miss = actual_prefix.difference(predicted_set)
            wasted = predicted_set.difference(actual_prefix)
            totals["predicted_blocks"] += len(predicted_set)
            totals["actual_blocks"] += len(actual_prefix)
            totals["hit_blocks"] += len(hit)
            totals["miss_blocks"] += len(miss)
            totals["wasted_blocks"] += len(wasted)
        per_step.append({
            "step": step,
            "predicted_blocks": totals["predicted_blocks"] -
            step_totals["predicted_blocks"],
            "actual_blocks": totals["actual_blocks"] -
            step_totals["actual_blocks"],
            "hit_blocks": totals["hit_blocks"] - step_totals["hit_blocks"],
            "miss_blocks": totals["miss_blocks"] - step_totals["miss_blocks"],
            "wasted_blocks": totals["wasted_blocks"] -
            step_totals["wasted_blocks"],
        })

    totals["prediction_recall"] = (totals["hit_blocks"] /
                                    max(1, totals["actual_blocks"]))
    payload = {
        "strategy": "solidattention_prediction_oracle",
        "device": str(device),
        "prefix_blocks": args.prefix_blocks,
        "num_blocks_with_live_suffix": num_blocks,
        "layers": args.layers,
        "block_size": args.block_size,
        "block_budget": args.block_budget,
        "steps": args.steps,
        "seed": args.seed,
        "dense_warmup_blocks": warmup_blocks,
        "prediction": totals,
        "per_step": per_step,
        "sparse_stats": sparse_kv_stats(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    reset_sparse_kv_stats()
    main()
