#!/usr/bin/env python3
"""Minimal real vLLM/GranuleKV Quest consumer smoke.

The run intentionally validates one concrete boundary: a completed request
populates the GranuleKV prefix, GPU prefix metadata is cleared, and a second
request restores the shared prefix through the hierarchical layer path.  The
decode attention consumer then compacts the existing vLLM block table using the
Quest policy.  It does not claim that the current scheduler has rebuilt SSD
restore mappings from a previous-step query; that control-plane gap is logged
separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--swap-space", type=float, default=8.0)
    parser.add_argument("--block-budget", type=int, default=8)
    parser.add_argument("--dynamic-restore", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _token_digest(values: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


def _build_prompt(tokenizer: Any, token_count: int,
                  offset: int) -> list[int]:
    seed = tokenizer.encode(
        "GranuleKV Quest sparse attention validation sequence. ",
        add_special_tokens=False)
    if not seed:
        raise RuntimeError("tokenizer returned an empty seed sequence")
    result: list[int] = []
    while len(result) < token_count:
        result.extend(seed)
    # Keep the workload deterministic while making the suffix distinguishable.
    result = result[:token_count]
    if token_count:
        result[-1] = (result[-1] + offset) % tokenizer.vocab_size
    return result


def main() -> None:
    args = _args()
    if args.prefix_tokens <= 0 or args.suffix_tokens <= 0:
        raise ValueError("prefix-tokens and suffix-tokens must be positive")
    if args.decode_tokens <= 0 or args.block_budget <= 0:
        raise ValueError("decode-tokens and block-budget must be positive")
    if args.prefix_tokens % 16 != 0:
        raise ValueError(
            "prefix-tokens must be a multiple of vLLM block_size=16")

    # These settings are deliberately explicit in the experiment process so
    # importing this script cannot silently alter another vLLM invocation.
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET",
                          str(args.block_budget))
    os.environ.setdefault(
        "VLLM_GRANULEKV_SPARSE_POLICY_MODULE",
        "evaluation.paper_reproduction.quest.adapter.policy:QuestPolicy")
    if args.dynamic_restore:
        os.environ.setdefault("VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE",
                              "1")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.attention.ops.sparse_kv import sparse_kv_stats
    from vllm.utils import Device

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prefix = _build_prompt(tokenizer, args.prefix_tokens, 11)
    suffix = _build_prompt(tokenizer, args.suffix_tokens, 23)
    suffix_next = _build_prompt(tokenizer, args.suffix_tokens, 31)
    first_prompt = {"prompt_token_ids": prefix}
    second_prompt = {"prompt_token_ids": prefix + suffix}
    third_prompt = {"prompt_token_ids": prefix + suffix_next}
    sampling = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens)

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        device="cuda",
        dtype="half",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        scheduler_cls="vllm.core.async_kv_scheduler.AsyncKVScheduler",
        enforce_eager=True,
        disable_async_output_proc=True,
        max_num_seqs=1,
    )
    first = llm.generate([first_prompt], sampling, use_tqdm=False)
    first_elapsed_ms = (time.perf_counter() - started) * 1000.0

    # Keep the CPU/SSD prefix directory and discard only GPU prefix residency.
    # This forces the second request through the GranuleKV read path.
    if not llm.reset_prefix_cache(Device.GPU):
        raise RuntimeError("failed to clear GPU prefix cache")

    second_started = time.perf_counter()
    second = llm.generate([second_prompt], sampling, use_tqdm=False)
    second_elapsed_ms = (time.perf_counter() - second_started) * 1000.0

    third_elapsed_ms = None
    third_output_tokens = None
    if args.dynamic_restore:
        if not llm.reset_prefix_cache(Device.GPU):
            raise RuntimeError(
                "failed to clear GPU prefix cache before dynamic restore")
        third_started = time.perf_counter()
        third = llm.generate([third_prompt], sampling, use_tqdm=False)
        third_elapsed_ms = (time.perf_counter() - third_started) * 1000.0
        third_output_tokens = len(third[0].outputs[0].token_ids)

    payload = {
        "strategy": "quest_sparse_vllm_granulekv_consumer",
        "validated_scope": (
            "prefix_write_gpu_reset_hierarchical_restore_xformers_sparse_decode"),
        "dynamic_ssd_restore_selection": args.dynamic_restore,
        "model": args.model,
        "backend": os.getenv("VLLM_ATTENTION_BACKEND", "auto"),
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "decode_tokens": args.decode_tokens,
        "block_budget": args.block_budget,
        "prefix_sha256": _token_digest(prefix),
        "first_elapsed_ms": first_elapsed_ms,
        "second_elapsed_ms": second_elapsed_ms,
        "first_output_tokens": len(first[0].outputs[0].token_ids),
        "second_output_tokens": len(second[0].outputs[0].token_ids),
        "third_elapsed_ms": third_elapsed_ms,
        "third_output_tokens": third_output_tokens,
        "sparse_stats": sparse_kv_stats(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

    # LLM owns the worker process; the normal vLLM shutdown path is sufficient.
    del llm


if __name__ == "__main__":
    main()
