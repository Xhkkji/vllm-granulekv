#!/usr/bin/env python3
"""Resident-only Quest quality evaluation for Passkey and local JSONL data."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--mode",
                        choices=("dense", "quest", "solidattention"),
                        required=True)
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
    parser.add_argument("--all-blocks", action="store_true")
    parser.add_argument("--prefix-fraction", type=float, default=0.0,
                        help="cache this block-aligned prompt fraction before evaluation")
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _configure(args: argparse.Namespace, block_budget: int) -> None:
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_GRANULEKV_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER"] = (
        "1" if args.prefix_fraction else "0")
    os.environ["VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HOST_BACKING_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET"] = str(block_budget)
    os.environ["VLLM_GRANULEKV_SPARSE_VALIDATION_MODE"] = (
        "all_blocks" if args.all_blocks and not args.prefix_fraction else "off")
    if args.mode != "dense":
        os.environ["VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE"] = "1"
        os.environ["VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE"] = "1"
        os.environ["VLLM_GRANULEKV_SPARSE_SELECTED_BLOCKS_ENABLE"] = (
            "1" if args.mode == "solidattention" else "0")
        modules = {
            "quest": (
                "evaluation.paper_reproduction.quest.adapter.policy:QuestPolicy"),
            "solidattention": (
                "evaluation.paper_reproduction.solidattention.adapter:"
                "SolidAttentionPolicy"),
        }
        os.environ["VLLM_GRANULEKV_SPARSE_POLICY_MODULE"] = modules[args.mode]
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


def _prefix_length(prompt_length: int, fraction: float, block_size: int) -> int:
    prefix = int(prompt_length * fraction) // block_size * block_size
    if fraction and not 0 < prefix < prompt_length:
        raise ValueError("cached prefix must contain complete blocks and leave a suffix")
    return prefix


def _run_cached_case(llm: Any, case: dict[str, Any], sampling: Any,
                     args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from vllm import SamplingParams
    from vllm.utils import Device

    if not llm.reset_prefix_cache(Device.GPU):
        raise RuntimeError("quality request still owns cached blocks")
    prompt = case["prompt_token_ids"]
    prefix = _prefix_length(len(prompt), args.prefix_fraction, args.page_size)
    # Populate without sparse validation, so every mode consumes identical KV.
    llm.generate([{"prompt_token_ids": prompt[:prefix]}],
                 SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
    request_id = "quality-" + str(case["id"])
    sparse = args.mode == "solidattention"
    if sparse:
        llm.collective_rpc("get_sparse_kv_stats", kwargs={"reset": True})
        llm.collective_rpc("configure_sparse_validation", kwargs={
            "request_id": request_id,
            "mode": "all_blocks" if args.all_blocks else "record",
            "metadata": {
                "model": args.model,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompt).encode("ascii")).hexdigest(),
                "prefix_blocks": prefix // args.page_size,
                "prefill_attention_steps": max(1, (
                    len(prompt) - prefix + args.max_num_batched_tokens - 1)
                    // args.max_num_batched_tokens),
                "capture_trace": True,
                "capture_layer_data": False,
                "retain_report": True,
            },
        })
    engine = llm.llm_engine
    engine.add_request(request_id, {"prompt_token_ids": prompt}, sampling)
    result = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if str(output.request_id) == request_id and output.finished:
                result = output
    if result is None:
        raise RuntimeError("quality request produced no completed result")
    cached_tokens = int(result.num_cached_tokens or 0)
    if cached_tokens != prefix:
        raise RuntimeError(f"quality prefix coverage differs: {cached_tokens} != {prefix}")
    coverage = {"prefix_tokens": prefix, "cached_tokens": cached_tokens}
    if sparse:
        report = llm.collective_rpc("get_sparse_validation_report", kwargs={
            "request_id": request_id, "reset": True})[0]
        layers = llm.llm_engine.model_config.hf_config.num_hidden_layers
        prefill_steps = report["metadata"]["prefill_attention_steps"]
        records = report["records"]
        prefill = [row for row in records if row["decode_step"] < prefill_steps]
        if len(prefill) != layers * prefill_steps:
            raise RuntimeError("quality run did not cover every sparse prefill layer/chunk")
        coverage.update({
            "prefill_selected_calls": len(prefill),
            "prefill_selected_prefix_layer_blocks": sum(
                len(row["selected_prefix_blocks"]) for row in prefill),
            "prefill_full_prefix_layer_blocks": len(prefill) * (prefix // args.page_size),
            "prefill_selected_history_layer_blocks": sum(
                len(row["selected_history_blocks"]) for row in prefill),
            "prefill_complete_history_candidates": sum(
                row["representative_count"] for row in prefill),
            "prefill_selected_generated_history_blocks": sum(
                sum(block >= prefix // args.page_size
                    for block in row["selected_history_blocks"])
                for row in prefill),
            "sparse_stats": llm.collective_rpc("get_sparse_kv_stats")[0],
        })
    return result, coverage


def _run_case(llm: Any, case: dict[str, Any], sampling: Any,
              args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    coverage = {}
    if args.prefix_fraction:
        result, coverage = _run_cached_case(llm, case, sampling, args)
    else:
        result = llm.generate([{"prompt_token_ids": case["prompt_token_ids"]}],
                              sampling, use_tqdm=False)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    text = result.outputs[0].text
    normalized = _normalize(text)
    answers = [_normalize(str(answer)) for answer in case.get("answers", [])]
    return {
        "id": case["id"],
        "prompt_sha256": hashlib.sha256(json.dumps(
            case["prompt_token_ids"]).encode("ascii")).hexdigest(),
        "prompt_tokens": len(case["prompt_token_ids"]),
        "output_tokens": list(result.outputs[0].token_ids),
        "output_text": text,
        "correct": any(answer and answer in normalized for answer in answers),
        "elapsed_ms": elapsed_ms,
        "context_length": case.get("context_length"),
        "depth": case.get("depth"),
        "answers": case.get("answers", []),
        **coverage,
    }


def main() -> None:
    args = _args()
    if not 0 <= args.prefix_fraction < 1 or args.max_num_batched_tokens <= 0:
        raise ValueError("invalid prefix fraction or chunk size")
    if args.prefix_fraction and args.mode == "quest":
        raise ValueError("cached chunk prefill validation supports Dense/SolidAttention only")
    if args.page_size <= 0 or args.decode_tokens <= 0:
        raise ValueError("page-size and decode-tokens must be positive")
    if args.mode != "dense":
        if args.block_budget <= 0:
            if args.token_budget <= 0:
                raise ValueError("Quest mode requires a positive budget")
            args.block_budget = (args.token_budget + args.page_size - 1) // args.page_size
    else:
        args.block_budget = 1
    _configure(args, args.block_budget)

    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    cases = _load_cases(args, tokenizer)
    if not cases:
        raise ValueError("no quality cases were loaded")
    max_prompt = max(len(case["prompt_token_ids"]) for case in cases)
    max_model_len = args.max_model_len or max_prompt + args.decode_tokens + 16
    config = AutoConfig.from_pretrained(args.model)
    overrides = {}
    if max_model_len > config.max_position_embeddings:
        if config.model_type != "qwen2" or getattr(config, "rope_scaling", None):
            raise ValueError("configure model-specific RoPE before extending context")
        overrides["max_position_embeddings"] = max_model_len
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        device="cuda",
        dtype="half",
        max_model_len=max_model_len,
        hf_overrides=overrides,
        seed=args.seed,
        block_size=args.page_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=2.0,
        enable_prefix_caching=bool(args.prefix_fraction),
        enable_chunked_prefill=bool(args.prefix_fraction),
        max_num_batched_tokens=(args.max_num_batched_tokens
                                if args.prefix_fraction else None),
        enforce_eager=True,
        disable_async_output_proc=True,
        max_num_seqs=1,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens)
    results = []
    for case in cases:
        result = _run_case(llm, case, sampling, args)
        results.append(result)
        print(json.dumps({"case": result["id"], "correct": result["correct"],
                          "prefix_tokens": result.get("prefix_tokens"),
                          "output_text": result["output_text"]}), flush=True)
    sparse_stats = None
    if args.mode != "dense":
        worker_stats = llm.collective_rpc("get_sparse_kv_stats")
        sparse_stats = worker_stats[0] if worker_stats else None
    del llm

    payload = {
        "strategy": f"{args.mode}_resident_quality",
        "task": args.task,
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "manifest": str(args.manifest) if args.manifest else None,
        "max_model_len": max_model_len,
        "token_budget": args.token_budget,
        "block_budget": args.block_budget,
        "all_blocks": args.all_blocks,
        "page_size": args.page_size,
        "prefix_fraction": args.prefix_fraction,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "samples": len(results),
        "accuracy": sum(int(item["correct"]) for item in results) / len(results),
        "mean_elapsed_ms": statistics.mean(item["elapsed_ms"] for item in results),
        "granulekv_enabled": False,
        "sparse_stats": sparse_stats,
        "sparse_stats_scope": "last_case" if args.prefix_fraction else "whole_run",
        "results": results,
        "note": "quality-only elapsed includes prefix preparation and validation; not performance data. JSONL correctness is answer-substring match; use native LongBench evaluator for reported task metrics",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
