#!/usr/bin/env python3
"""Aggregate and validate the SolidAttention dynamic-path ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CASES = ("A_dense_full", "B_exact_compact", "C_exact_selected",
         "D_union_selected", "E1_guard_1", "E2_guard_2", "E3_guard_4")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    return parser.parse_args()


def _row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    stats = payload.get("sparse_stats") or {}
    return {
        "case": name,
        "restore_mode": payload.get("prediction_restore_mode"),
        "guard_blocks": payload.get("prediction_guard_blocks"),
        "selected_list": payload.get("selected_list_enabled"),
        "end_to_end_ms": payload.get("end_to_end_ms"),
        "p50_ms": payload.get("third_elapsed_p50_ms"),
        "p95_ms": payload.get("third_elapsed_p95_ms"),
        "decode_ms_per_token": payload.get("decode_ms_per_token"),
        "attention_selected_calls": stats.get("attention_selected_calls", 0),
        "attention_compact_calls": stats.get("attention_compact_calls", 0),
        "actual_selection_ms": stats.get("actual_selection_ms", 0.0),
        "prediction_restore_requests": stats.get(
            "prediction_restore_requests", 0),
        "prediction_restore_blocks": stats.get("prediction_restore_blocks", 0),
        "prediction_restore_bytes": stats.get("prediction_restore_bytes", 0),
        "prediction_recall": stats.get("prediction_recall", 0.0),
        "prediction_gap_events": stats.get("prediction_gap_events", 0),
        "correction_requests": stats.get("correction_requests", 0),
        "correction_blocks": stats.get("correction_blocks", 0),
        "correction_bytes": stats.get("correction_bytes", 0),
        "correction_submit_ms": stats.get("correction_submit_ms", 0.0),
        "correction_wait_ms": stats.get("correction_wait_ms", 0.0),
        "correction_total_ms": stats.get("correction_total_ms", 0.0),
        "residency_miss": stats.get("residency_miss", 0),
        "correction_errors": stats.get("correction_errors", 0),
    }


def main() -> None:
    root = _args().result_root
    payloads = {
        case: json.loads((root / case / "summary.json").read_text())
        for case in CASES
    }
    rows = [_row(case, payloads[case]) for case in CASES]

    sparse_tokens = payloads["B_exact_compact"]["third_token_ids"]
    tokens_match = all(payloads[case]["third_token_ids"] == sparse_tokens
                       for case in CASES[2:])
    if not tokens_match:
        raise RuntimeError("B-E sparse ablation outputs do not match")
    for case in CASES[1:]:
        row = next(item for item in rows if item["case"] == case)
        if row["residency_miss"] or row["correction_errors"]:
            raise RuntimeError(f"{case} reported a residency/correction error")
    for case in CASES[2:]:
        row = next(item for item in rows if item["case"] == case)
        if (row["attention_selected_calls"] <= 0
                or row["attention_compact_calls"] != 0):
            raise RuntimeError(f"{case} did not use only selected-list attention")

    summary = {
        "sparse_outputs_match": tokens_match,
        "cases": rows,
    }
    (root / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    with (root / "ablation_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
