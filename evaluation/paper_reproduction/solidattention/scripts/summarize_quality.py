#!/usr/bin/env python3
"""Compare Dense and SolidAttention quality runs into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import runpy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--longbench-metrics", type=Path,
                        help="native LongBench metrics.py; score local TriviaQA with QA F1")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def results_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in payload["results"]}


def main() -> None:
    args = parse_args()
    dense = load(args.dense)
    dense_results = results_by_id(dense)
    qa_f1 = (runpy.run_path(str(args.longbench_metrics))["qa_f1_score"]
             if args.longbench_metrics else None)
    rows = []
    for path in args.sparse:
        sparse = load(path)
        sparse_results = results_by_id(sparse)
        if set(sparse_results) != set(dense_results):
            raise ValueError(f"quality case ids differ: {path}")
        if (sparse.get("prefix_fraction", 0) != dense.get("prefix_fraction", 0)
                or sparse.get("max_num_batched_tokens") != dense.get("max_num_batched_tokens")):
            raise ValueError(f"quality prefix/chunk configuration differs: {path}")
        if any(sparse_results[key].get("prompt_sha256") !=
               dense_results[key].get("prompt_sha256") for key in dense_results
               if sparse_results[key].get("prompt_sha256") is not None
               and dense_results[key].get("prompt_sha256") is not None):
            raise ValueError(f"quality prompts differ: {path}")
        if sparse.get("prefix_fraction"):
            for key, item in sparse_results.items():
                if (item["prefix_tokens"] != dense_results[key]["prefix_tokens"]
                        or item["cached_tokens"] != item["prefix_tokens"]
                        or item.get("prefill_selected_calls", 0) <= 0):
                    raise ValueError(f"quality prefill coverage missing: {path} {key}")
        token_matches = sum(
            sparse_results[key]["output_tokens"] ==
            dense_results[key]["output_tokens"] for key in dense_results)
        dense_score = float(dense["accuracy"])
        sparse_score = float(sparse["accuracy"])
        metric = "answer_substring_accuracy"
        if sparse["task"] == "jsonl" and qa_f1 is not None:
            def score(results):
                return sum(max((qa_f1(
                    item["output_text"].lstrip('\n').split('\n')[0], answer)
                    for answer in item["answers"]), default=0.0)
                    for item in results.values()) / len(results)
            dense_score = score(dense_results)
            sparse_score = score(sparse_results)
            metric = "longbench_triviaqa_qa_f1"
        all_blocks = bool(sparse.get("all_blocks"))
        score_drop_points = (dense_score - sparse_score) * 100.0
        passkey_regressions = sum(
            bool(dense_results[key]["correct"])
            and not bool(sparse_results[key]["correct"])
            for key in dense_results)
        stats = sparse.get("sparse_stats") or {}
        quality_pass = (token_matches == len(dense_results)
                        if all_blocks else
                        (sparse["task"] != "passkey" or passkey_regressions == 0)
                        and score_drop_points <= 1.0)
        if sparse["task"] == "jsonl" and qa_f1 is None:
            quality_pass = False
        prefix_full = sum(item.get("prefill_full_prefix_layer_blocks", 0)
                          for item in sparse_results.values())
        history_full = sum(item.get("prefill_complete_history_candidates", 0)
                           for item in sparse_results.values())
        full_blocks = sum(item.get("sparse_stats", {}).get("full_blocks", 0)
                          for item in sparse_results.values())
        selected_blocks = sum(item.get("sparse_stats", {}).get("selected_blocks", 0)
                              for item in sparse_results.values())
        rows.append({
            "source": str(path),
            "task": sparse["task"],
            "block_budget": sparse["block_budget"],
            "all_blocks": all_blocks,
            "samples": len(dense_results),
            "metric": metric,
            "prefix_fraction": sparse.get("prefix_fraction", 0),
            "prefill_selected_prefix_ratio": (
                sum(item["prefill_selected_prefix_layer_blocks"]
                    for item in sparse_results.values()) / prefix_full
                if prefix_full else None),
            "prefill_complete_history_candidates": history_full,
            "prefill_selected_history_layer_blocks": sum(
                item.get("prefill_selected_history_layer_blocks", 0)
                for item in sparse_results.values()),
            "prefill_selected_generated_history_blocks": sum(
                item.get("prefill_selected_generated_history_blocks", 0)
                for item in sparse_results.values()),
            "selected_block_ratio": (selected_blocks / full_blocks if full_blocks
                                     else stats.get("selected_ratio")),
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "score_drop_points": score_drop_points,
            "greedy_token_match_rate": token_matches / len(dense_results),
            "dense_correct_sparse_failures": passkey_regressions,
            "attention_mass_coverage": None,
            "attention_output_rel_error": None,
            "attention_output_cosine": None,
            "logit_kl": None,
            "quality_gate": "pass" if quality_pass else "fail",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
