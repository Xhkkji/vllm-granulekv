#!/usr/bin/env python3
"""Classify all-block and fixed-selection SolidAttention validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-summary", type=Path, required=True)
    parser.add_argument("--all-block-summary", type=Path, required=True)
    parser.add_argument("--replay-summary", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (int(record["prefix_tokens"]), int(record["suffix_tokens"]),
            int(record["repetition"]))


def records(payload: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    return {key(record): record for record in payload["records"]}


def main() -> None:
    args = parse_args()
    dense = records(load(args.dense_summary))
    all_blocks = records(load(args.all_block_summary))
    if set(dense) != set(all_blocks):
        raise ValueError("Dense and all-block runs cover different sweep points")

    all_block_cases = []
    for point in sorted(dense):
        matches = (dense[point]["generated_token_ids"] ==
                   all_blocks[point]["generated_token_ids"])
        sample = all_blocks[point]
        report = sample.get("validation_report") or {}
        expected = int(report.get("metadata", {}).get("num_layers", 0)) * max(
            0, int(sample.get("generated_tokens", 0)) - 1)
        checks = int(report.get("attention_checks", 0))
        valid = (matches and expected > 0 and checks == expected
                 and int(report.get("attention_output_mismatches", 0)) == 0
                 and int(sample.get("sparse_stats", {}).get("attention_selected_calls", 0)) == checks
                 and int(sample.get("sparse_stats", {}).get("attention_compact_calls", 0)) == 0)
        all_block_cases.append({
            "prefix_tokens": point[0],
            "suffix_tokens": point[1],
            "repetition": point[2],
            "greedy_token_match": matches,
            "attention_checks": checks,
            "expected_attention_checks": expected,
            "valid": valid,
        })

    replay_runs = []
    execution_valid = bool(args.replay_summary)
    mismatch_names = (
        "selector_reference_mismatches",
        "replay_selection_mismatches",
        "replay_unconsumed_records",
        "kv_fingerprint_mismatches",
        "attention_output_mismatches",
        "logit_mismatches",
    )
    for path in args.replay_summary:
        payload = load(path)
        cases = []
        execution_valid &= bool(payload["records"])
        for record in payload["records"]:
            report = record.get("validation_report") or {}
            counters = {name: int(report.get(name, 0) or 0)
                        for name in mismatch_names}
            expected = int(report.get("metadata", {}).get("num_layers", 0)) * max(
                0, int(record.get("generated_tokens", 0)) - 1)
            valid = (report.get("mode") == "replay" and expected > 0
                     and int(report.get("attention_checks", 0)) == expected
                     and record.get("greedy_token_match") is True
                     and record.get("validation_status") == "valid"
                     and all(int(record.get("sparse_stats", {}).get(name, 0)) == 0
                             for name in ("residency_miss", "correction_errors",
                                          "duplicate_ssd_read_blocks",
                                          "host_coverage_missing_blocks"))
                     and all(value == 0 for value in counters.values()))
            execution_valid &= valid
            cases.append({
                "prefix_tokens": record["prefix_tokens"],
                "suffix_tokens": record["suffix_tokens"],
                "repetition": record["repetition"],
                "status": "valid" if valid else "invalid_for_comparison",
                "counters": counters,
                "first_mismatch_step": report.get("first_mismatch_step"),
                "first_mismatch_layer": report.get("first_mismatch_layer"),
                "first_mismatch": report.get("first_mismatch"),
            })
        replay_runs.append({"source": str(path), "cases": cases})

    all_block_valid = bool(all_block_cases) and all(case["valid"]
                                                   for case in all_block_cases)
    result = {
        "schema_version": 1,
        "all_block_dense_parity": {
            "status": "pass" if all_block_valid else "fail",
            "cases": all_block_cases,
        },
        "fixed_selection_data_path_parity": {
            "status": ("not_run" if not args.replay_summary else
                       "pass" if execution_valid else "fail"),
            "runs": replay_runs,
        },
        "valid_for_performance_comparison": all_block_valid and execution_valid,
        "classification": (
            "valid" if all_block_valid and execution_valid else
            "all_block_attention_or_mapping_error" if not all_block_valid else
            "data_path_validation_not_run" if not args.replay_summary else
            "kv_restore_residency_or_async_timing_error"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) +
                           "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
