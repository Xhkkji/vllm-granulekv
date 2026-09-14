#!/usr/bin/env python3
"""Resident-only Quest quality evaluation for Passkey and local JSONL data."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("dense", "quest"), required=True)
    parser.add_argument("--task", choices=("passkey", "jsonl"), required=True)
    parser.add_argument("--contexts", default="8192")
    parser.add_argument("--depths", default="0,25,50,75,100")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--block-budget", type=int, default=0)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _configure(args: argparse.Namespace, block_budget: int) -> None:
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_GRANULEKV_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER"] = "0"
    os.environ["VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET"] = str(block_budget)
    if args.mode == "quest":
        os.environ["VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE"] = "1"
        os.environ["VLLM_GRANULEKV_SPARSE_POLICY_MODULE"] = (
            "evaluation.paper_reproduction.quest.adapter.policy:QuestPolicy")
    else:
        os.environ["VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE"] = "0"
        os.environ.pop("VLLM_GRANULEKV_SPARSE_POLICY_MODULE", None)


def _passkey_prompt(tokenizer: Any, context_length: int, depth: int,
                    value: int) -> tuple[list[int], int]:
    marker = tokenizer.encode(
        f"The pass key is {value}. Remember it. {value} is the pass key.",
        add_special_tokens=False)
    question = tokenizer.encode(
        "What is the pass key? The pass key is", add_special_tokens=False)
    garbage_seed = tokenizer.encode(
        "The grass is green. The sky is blue. The sun is yellow. ",
        add_special_tokens=False)
    remaining = context_length - len(marker) - len(question)
    if remaining <= 0:
        raise ValueError("context-length is too short for passkey prompt")
    garbage = (garbage_seed * ((remaining + len(garbage_seed) - 1) //
                               len(garbage_seed)))[:remaining]
    split = min(remaining, max(0, int(remaining * depth / 100)))
    return garbage[:split] + marker + garbage[split:] + question, value


def _load_cases(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    if args.task == "jsonl":
        if args.manifest is None:
            raise ValueError("--manifest is required for jsonl task")
        cases = []
        with args.manifest.open() as stream:
            for index, line in enumerate(stream):
                if index >= args.limit:
                    break
                row = json.loads(line)
                prompt = row.get("prompt") or row.get("input")
                if not prompt:
                    continue
                cases.append({
                    "id": row.get("_id", str(index)),
                    "prompt_token_ids": tokenizer.encode(
                        prompt, add_special_tokens=False),
                    "answers": row.get("answers", []),
                })
        return cases

    rng = random.Random(args.seed)
    cases = []
    contexts = [int(value) for value in args.contexts.split(",") if value]
    depths = [int(value) for value in args.depths.split(",") if value]
    for context_length in contexts:
        for depth in depths:
            for sample in range(args.samples):
                value = rng.randint(1, 50000)
                prompt, answer = _passkey_prompt(tokenizer, context_length,
                                                 depth, value)
                cases.append({
                    "id": f"ctx{context_length}_depth{depth}_sample{sample}",
                    "prompt_token_ids": prompt,
                    "answers": [str(answer)],
                    "context_length": context_length,
                    "depth": depth,
                })
    return cases


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _run_case(llm: Any, case: dict[str, Any], sampling: Any) -> dict[str, Any]:
    started = time.perf_counter()
    result = llm.generate([{"prompt_token_ids": case["prompt_token_ids"]}],
                          sampling, use_tqdm=False)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    text = result.outputs[0].text
    normalized = _normalize(text)
    answers = [_normalize(str(answer)) for answer in case.get("answers", [])]
    return {
        "id": case["id"],
        "prompt_tokens": len(case["prompt_token_ids"]),
        "output_tokens": list(result.outputs[0].token_ids),
        "output_text": text,
        "correct": any(answer and answer in normalized for answer in answers),
        "elapsed_ms": elapsed_ms,
        "context_length": case.get("context_length"),
        "depth": case.get("depth"),
    }


def main() -> None:
    args = _args()
    if args.page_size <= 0 or args.decode_tokens <= 0:
        raise ValueError("page-size and decode-tokens must be positive")
    if args.mode == "quest":
        if args.block_budget <= 0:
            if args.token_budget <= 0:
                raise ValueError("Quest mode requires a positive budget")
            args.block_budget = (args.token_budget + args.page_size - 1) // args.page_size
    else:
        args.block_budget = 1
    _configure(args, args.block_budget)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cases = _load_cases(args, tokenizer)
    if not cases:
        raise ValueError("no quality cases were loaded")
    max_prompt = max(len(case["prompt_token_ids"]) for case in cases)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        device="cuda",
        dtype="half",
        max_model_len=(args.max_model_len or max_prompt + args.decode_tokens + 16),
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=2.0,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_async_output_proc=True,
        max_num_seqs=1,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens)
    results = [_run_case(llm, case, sampling) for case in cases]
    sparse_stats = None
    if args.mode == "quest":
        worker_stats = llm.collective_rpc("get_sparse_kv_stats")
        sparse_stats = worker_stats[0] if worker_stats else None
    del llm

    payload = {
        "strategy": "quest_resident_quality",
        "task": args.task,
        "model": args.model,
        "mode": args.mode,
        "token_budget": args.token_budget,
        "block_budget": args.block_budget,
        "page_size": args.page_size,
        "samples": len(results),
        "accuracy": sum(int(item["correct"]) for item in results) / len(results),
        "mean_elapsed_ms": statistics.mean(item["elapsed_ms"] for item in results),
        "granulekv_enabled": False,
        "sparse_stats": sparse_stats,
        "results": results,
        "note": "jsonl correctness is answer-substring match; use native LongBench evaluator for reported task metrics",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
