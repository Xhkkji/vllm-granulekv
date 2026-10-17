#!/usr/bin/env python3
"""Compare Dense and SolidAttention quality runs into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def results_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in payload["results"]}


def main() -> None:
    args = parse_args()
    dense = load(args.dense)
    dense_results = results_by_id(dense)
    rows = []
    for path in args.sparse:
        sparse = load(path)
        sparse_results = results_by_id(sparse)
        if set(sparse_results) != set(dense_results):
            raise ValueError(f"quality case ids differ: {path}")
        token_matches = sum(
            sparse_results[key]["output_tokens"] ==
            dense_results[key]["output_tokens"] for key in dense_results)
        dense_score = float(dense["accuracy"])
        sparse_score = float(sparse["accuracy"])
        all_blocks = bool(sparse.get("all_blocks"))
        score_drop_points = (dense_score - sparse_score) * 100.0
        passkey_regressions = sum(
            bool(dense_results[key]["correct"])
            and not bool(sparse_results[key]["correct"])
            for key in dense_results)
        stats = sparse.get("sparse_stats") or {}
        quality_pass = (token_matches == len(dense_results)
                        if all_blocks else
                        passkey_regressions == 0
                        and score_drop_points <= 1.0)
        rows.append({
            "source": str(path),
            "task": sparse["task"],
            "block_budget": sparse["block_budget"],
            "all_blocks": all_blocks,
            "samples": len(dense_results),
            "selected_block_ratio": stats.get("selected_ratio"),
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
