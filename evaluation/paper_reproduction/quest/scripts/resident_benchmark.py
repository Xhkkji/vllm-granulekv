#!/usr/bin/env python3
"""Resident-only Quest benchmark for dense/sparse decode comparison.

The benchmark deliberately disables GranuleKV and hierarchical I/O. It keeps
one LLM instance per process, warms it up, and reports vLLM request timing
metrics separately from model initialization.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode",
                        choices=("dense", "quest", "solidattention"),
                        required=True)
    parser.add_argument(
        "--selected-blocks",
        action="store_true",
        help="Use SolidAttention's selected-list CUDA kernel path.")
    parser.add_argument(
        "--selection-reuse",
        choices=("on", "off"),
        default="on",
        help="Reuse SolidAttention selection within one logical block.")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--block-budget", type=int, default=0)
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _configure(args: argparse.Namespace, block_budget: int) -> None:
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_GRANULEKV_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER"] = "0"
    os.environ["VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE"] = (
        "1" if args.mode != "dense" else "0")
    os.environ["VLLM_GRANULEKV_SPARSE_SELECTED_BLOCKS_ENABLE"] = (
        "1" if args.selected_blocks and args.mode == "solidattention" else
        "0")
    os.environ["VLLM_GRANULEKV_SPARSE_SELECTION_REUSE_ENABLE"] = (
        "1" if args.selection_reuse == "on" else "0")
    os.environ["VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET"] = str(block_budget)
    if args.mode != "dense":
        os.environ["VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE"] = "1"
        policy_modules = {
            "quest": "evaluation.paper_reproduction.quest.adapter.policy:QuestPolicy",
            "solidattention": (
                "evaluation.paper_reproduction.solidattention.adapter:"
                "SolidAttentionPolicy"),
        }
        os.environ["VLLM_GRANULEKV_SPARSE_POLICY_MODULE"] = policy_modules[
            args.mode]
    else:
        os.environ["VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE"] = "0"
        os.environ.pop("VLLM_GRANULEKV_SPARSE_POLICY_MODULE", None)


def _prompt(tokenizer: Any, length: int) -> list[int]:
    seed = tokenizer.encode(
        "Quest resident-only attention benchmark. ",
        add_special_tokens=False)
    if not seed:
        raise RuntimeError("tokenizer returned an empty seed sequence")
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def _request_stats(output: Any, elapsed_ms: float,
                   decode_tokens: int) -> dict[str, Optional[float]]:
    metrics = output.metrics
    first_token = getattr(metrics, "first_token_time", None)
    arrival = getattr(metrics, "arrival_time", None)
    finished = getattr(metrics, "finished_time", None)
    ttft_ms = ((first_token - arrival) * 1000.0
               if first_token is not None and arrival is not None else None)
    decode_ms = ((finished - first_token) * 1000.0
                 if finished is not None and first_token is not None else None)
    generated = len(output.outputs[0].token_ids)
    decode_per_token = (decode_ms / max(1, generated - 1)
                        if decode_ms is not None else None)
    return {
        "elapsed_ms": elapsed_ms,
        "prefill_ms": ttft_ms,
        "decode_ms": decode_ms,
        "decode_ms_per_token": decode_per_token,
        "throughput": generated / max(elapsed_ms / 1000.0, 1e-9),
        "generated_tokens": float(generated),
        "requested_decode_tokens": float(decode_tokens),
    }


def main() -> None:
    args = _parse_args()
    if args.context_length <= 0 or args.decode_tokens <= 0:
        raise ValueError("context-length and decode-tokens must be positive")
    if args.selected_blocks and args.mode != "solidattention":
        raise ValueError("--selected-blocks requires solidattention mode")
    if args.page_size <= 0 or args.context_length % args.page_size != 0:
        raise ValueError("context-length must be divisible by positive page-size")
    if args.mode != "dense" and args.block_budget <= 0:
        if args.token_budget <= 0:
            raise ValueError("Quest mode requires token-budget or block-budget")
        args.block_budget = (args.token_budget + args.page_size - 1) // args.page_size
    elif args.block_budget <= 0:
        args.block_budget = args.context_length // args.page_size
    if args.token_budget <= 0:
        args.token_budget = args.block_budget * args.page_size
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")

    _configure(args, args.block_budget)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompt = {"prompt_token_ids": _prompt(tokenizer, args.context_length)}
    sampling = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        device="cuda",
        dtype="half",
        max_model_len=(args.max_model_len or
                       args.context_length + args.decode_tokens + 16),
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=2.0,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_async_output_proc=True,
        max_num_seqs=1,
    )

    for _ in range(args.warmup):
        llm.generate([prompt], sampling, use_tqdm=False)
    if args.mode != "dense":
        llm.collective_rpc("get_sparse_kv_stats", kwargs={"reset": True})

    samples = []
    outputs = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        result = llm.generate([prompt], sampling, use_tqdm=False)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        outputs.append(result.outputs[0].token_ids)
        samples.append(_request_stats(result, elapsed_ms, args.decode_tokens))

    sparse_stats = None
    if args.mode != "dense":
        worker_stats = llm.collective_rpc("get_sparse_kv_stats")
        # Tensor-parallel workers execute the same layer set. Count one
        # worker's logical attention calls, rather than multiplying by TP.
        stats = worker_stats[0] if worker_stats else {}
        sparse_stats = {
            "calls": stats.get("calls", 0),
            "full_blocks": stats.get("full_blocks", 0),
            "selected_blocks": stats.get("selected_blocks", 0),
            "selected_tokens": stats.get("selected_tokens", 0),
            "metadata_build_calls": stats.get("metadata_build_calls", 0),
            "metadata_build_blocks": stats.get("metadata_build_blocks", 0),
            "gpu_selection_calls": stats.get("gpu_selection_calls", 0),
            "attention_selection_calls": stats.get(
                "attention_selection_calls", 0),
            "attention_selection_refreshes": stats.get(
                "attention_selection_refreshes", 0),
            "attention_selection_cache_hits": stats.get(
                "attention_selection_cache_hits", 0),
            "attention_selected_blocks": stats.get(
                "attention_selected_blocks", 0),
            "selected_ratio": stats.get("selected_ratio", 0.0),
            "gpu_memory_allocated": stats.get("gpu_memory_allocated"),
            "gpu_memory_reserved": stats.get("gpu_memory_reserved"),
            "gpu_max_memory_allocated": stats.get("gpu_max_memory_allocated"),
        }
        if sparse_stats["attention_selection_calls"]:
            sparse_stats["selected_blocks"] = sparse_stats[
                "attention_selected_blocks"]
            sparse_stats["selected_ratio"] = (
                sparse_stats["selected_blocks"] /
                max(1, sparse_stats["full_blocks"]))
    del llm
    decode_samples = [item["decode_ms_per_token"] for item in samples
                      if item["decode_ms_per_token"] is not None]
    full_blocks_per_run = args.context_length // args.page_size
    selected_blocks = (sparse_stats["selected_blocks"]
                       if sparse_stats is not None else
                       full_blocks_per_run * max(1, len(decode_samples)))
    full_blocks = (sparse_stats["full_blocks"]
                   if sparse_stats is not None else
                   full_blocks_per_run * max(1, len(decode_samples)))
    decode_p50 = (statistics.median(decode_samples)
                  if decode_samples else None)
    decode_p95 = (sorted(decode_samples)[min(
        len(decode_samples) - 1, int(len(decode_samples) * 0.95))]
                  if decode_samples else None)
    payload = {
        "strategy": f"{args.mode}_resident_attention",
        "model": args.model,
        "backend": os.getenv("VLLM_ATTENTION_BACKEND", "auto"),
        "mode": args.mode,
        "context_length": args.context_length,
        "decode_tokens": args.decode_tokens,
        "page_size": args.page_size,
        "token_budget": args.token_budget,
        "block_budget": args.block_budget,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "granulekv_enabled": False,
        "selected_blocks_kernel": bool(args.selected_blocks),
        "selection_reuse": args.selection_reuse,
        "selected_blocks": selected_blocks,
        "full_blocks": full_blocks,
        "selected_ratio": selected_blocks / max(1, full_blocks),
        "attention_calls": (sparse_stats["calls"] if sparse_stats is not None
                             else max(1, len(decode_samples))),
        "selected_blocks_per_call": selected_blocks / max(
            1, sparse_stats["calls"] if sparse_stats is not None else
            len(decode_samples)),
        "full_blocks_per_call": full_blocks / max(
            1, sparse_stats["calls"] if sparse_stats is not None else
            len(decode_samples)),
        "prefill_ms": statistics.mean(
            item["prefill_ms"] for item in samples
            if item["prefill_ms"] is not None) if any(
                item["prefill_ms"] is not None for item in samples) else None,
        "decode_ms_per_token": statistics.mean(decode_samples)
        if decode_samples else None,
        "decode_p50_ms_per_token": decode_p50,
        "decode_p95_ms_per_token": decode_p95,
        "decode_p50_ms": decode_p50,
        "decode_p95_ms": decode_p95,
        "total_generation_ms": statistics.mean(
            item["elapsed_ms"] for item in samples),
        "throughput": statistics.mean(item["throughput"] for item in samples),
        "output_tokens": [list(item) for item in outputs],
        "request_samples": samples,
        "sparse_stats": sparse_stats,
        "quality": {},
        "note": "prefill_ms is vLLM TTFT; adapter is not official Quest CUDA runtime",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
