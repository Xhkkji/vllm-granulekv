#!/usr/bin/env python3
"""Resident-only Quest sparse attention smoke.

This stage deliberately does not start GranuleKV, MPS, or hierarchical I/O.
The full GPU KV table is resident; the attention backend builds page metadata
from that table and uses the Quest policy for decode steps after warmup.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("dense", "sparse"), required=True)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--block-budget", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.prefix_tokens <= 0 or args.prefix_tokens % 16 != 0:
        raise ValueError("prefix-tokens must be a positive multiple of 16")
    if args.decode_tokens <= 0 or args.block_budget <= 0:
        raise ValueError("decode-tokens and block-budget must be positive")

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_GRANULEKV_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE"] = "0"
    os.environ["VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER"] = "0"
    os.environ["VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE"] = (
        "1" if args.mode == "sparse" else "0")
    os.environ["VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE"] = (
        "1" if args.mode == "sparse" else "0")
    os.environ["VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET"] = str(
        args.block_budget)
    if args.mode == "sparse":
        os.environ["VLLM_GRANULEKV_SPARSE_POLICY_MODULE"] = (
            "evaluation.paper_reproduction.quest.adapter.policy:QuestPolicy")
    else:
        os.environ.pop("VLLM_GRANULEKV_SPARSE_POLICY_MODULE", None)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    seed = tokenizer.encode(
        "Quest resident sparse attention validation sequence. ",
        add_special_tokens=False)
    if not seed:
        raise RuntimeError("tokenizer returned an empty seed sequence")
    prompt_tokens = (seed * ((args.prefix_tokens + len(seed) - 1) //
                             len(seed)))[:args.prefix_tokens]
    prompt = {"prompt_token_ids": prompt_tokens}
    sampling = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens)

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        device="cuda",
        dtype="half",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=2.0,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_async_output_proc=True,
        max_num_seqs=1,
    )
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "stage": "quest_resident_sparse_attention",
        "mode": args.mode,
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "decode_tokens": args.decode_tokens,
        "block_budget": args.block_budget,
        "elapsed_ms": elapsed_ms,
        "output_tokens": list(outputs[0].outputs[0].token_ids),
        "granulekv_enabled": False,
        "hierarchical_prefetch_enabled": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    del llm


if __name__ == "__main__":
    main()
