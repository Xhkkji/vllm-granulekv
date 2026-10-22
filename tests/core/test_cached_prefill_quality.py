"""Runner checks without loading a model or changing production execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = (Path(__file__).resolve().parents[2] / "evaluation/paper_reproduction"
        / "quest/scripts/quality_eval.py")
spec = importlib.util.spec_from_file_location("cached_prefill_quality", path)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


@pytest.mark.parametrize("length,fraction,expected", [
    (8192, 0.5, 4096), (16384, 0.5, 8192), (32768, 0.5, 16384),
    (8193, 0.5, 4096), (8192, 0, 0),
])
def test_prefix_is_complete_blocks(length, fraction, expected):
    assert quality._prefix_length(length, fraction, 16) == expected


def test_short_prefix_rejected():
    with pytest.raises(ValueError, match="complete blocks"):
        quality._prefix_length(20, 0.5, 16)


def test_cached_quality_configuration_does_not_enable_native_io(monkeypatch):
    monkeypatch.setattr(quality.os, "environ", dict(quality.os.environ))
    args = SimpleNamespace(prefix_fraction=0.5, all_blocks=True,
                           mode="solidattention")
    quality._configure(args, 32)
    import os
    assert os.environ["VLLM_GRANULEKV_ENABLE"] == "0"
    assert os.environ["VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER"] == "1"
    assert os.environ["VLLM_GRANULEKV_SPARSE_VALIDATION_MODE"] == "off"


@pytest.mark.parametrize("cached", [32, 16])
def test_runner_verifies_actual_cache_hit(cached):
    result = SimpleNamespace(request_id="quality-case", finished=True,
                             num_cached_tokens=cached)
    class Engine:
        def add_request(self, *args):
            self.pending = True

        def has_unfinished_requests(self):
            return self.pending

        def step(self):
            self.pending = False
            return [result]

    llm = SimpleNamespace(reset_prefix_cache=lambda _: True,
                          generate=lambda *args, **kwargs: None,
                          llm_engine=Engine())
    args = SimpleNamespace(prefix_fraction=0.5, page_size=16, mode="dense")
    case = {"id": "case", "prompt_token_ids": list(range(64))}
    if cached == 32:
        actual, coverage = quality._run_cached_case(llm, case, None, args)
        assert actual is result
        assert coverage == {"prefix_tokens": 32, "cached_tokens": 32}
    else:
        with pytest.raises(RuntimeError, match="prefix coverage differs"):
            quality._run_cached_case(llm, case, None, args)


@pytest.mark.parametrize("task,all_blocks,expected", [
    ("passkey", False, "fail"), ("passkey", True, "fail"),
    ("jsonl", False, "pass"),
])
def test_summary_uses_task_metric_and_all_block_tokens(
        monkeypatch, tmp_path, task, all_blocks, expected):
    import csv
    import json
    summary_path = (path.parents[2] / "solidattention/scripts/summarize_quality.py")
    summary_spec = importlib.util.spec_from_file_location("quality_summary", summary_path)
    summary = importlib.util.module_from_spec(summary_spec)
    summary_spec.loader.exec_module(summary)
    dense = {
        "task": task, "accuracy": 1.0, "prefix_fraction": 0.5,
        "max_num_batched_tokens": 1024,
        "results": [{"id": "a", "prefix_tokens": 32, "cached_tokens": 32,
                     "correct": True, "output_tokens": [1], "output_text": "answer",
                     "answers": ["answer"]}],
    }
    sparse = {**dense, "accuracy": 0.0, "all_blocks": all_blocks,
              "block_budget": 32, "results": [{
                  **dense["results"][0], "correct": False, "output_tokens": [2],
                  "prefill_selected_calls": 28,
                  "prefill_selected_prefix_layer_blocks": 28,
                  "prefill_full_prefix_layer_blocks": 56,
              }]}
    dense_path, sparse_path = tmp_path / "dense.json", tmp_path / "sparse.json"
    dense_path.write_text(json.dumps(dense))
    sparse_path.write_text(json.dumps(sparse))
    output = tmp_path / "quality.csv"
    monkeypatch.setattr(summary, "parse_args", lambda: SimpleNamespace(
        dense=dense_path, sparse=[sparse_path], output=output,
        longbench_metrics=Path("metrics.py") if task == "jsonl" else None))
    monkeypatch.setattr(summary.runpy, "run_path", lambda _: {
        "qa_f1_score": lambda text, answer: float(text == answer)})
    summary.main()
    row = next(csv.DictReader(output.open()))
    assert row["quality_gate"] == expected
    assert float(row["prefill_selected_prefix_ratio"]) == 0.5
    if task == "jsonl":
        assert row["metric"] == "longbench_triviaqa_qa_f1"
