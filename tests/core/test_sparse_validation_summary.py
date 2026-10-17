import json
import sys

from evaluation.paper_reproduction.solidattention.scripts import summarize_correctness


def test_missing_checks_and_missing_replay_cannot_pass(tmp_path, monkeypatch):
    sample = {"prefix_tokens": 32768, "suffix_tokens": 0, "repetition": 0,
              "generated_tokens": 1, "generated_token_ids": [323]}
    for name in ("dense", "all_blocks"):
        (tmp_path / (name + ".json")).write_text(json.dumps({"records": [sample]}))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["summarize_correctness",
        "--dense-summary", str(tmp_path / "dense.json"),
        "--all-block-summary", str(tmp_path / "all_blocks.json"),
        "--output", str(output)])
    summarize_correctness.main()
    summary = json.loads(output.read_text())
    assert summary["all_block_dense_parity"]["status"] == "fail"
    assert summary["fixed_selection_data_path_parity"]["status"] == "not_run"
    assert not summary["valid_for_performance_comparison"]
