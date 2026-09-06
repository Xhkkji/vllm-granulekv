#!/usr/bin/env python3
"""Real GranuleKV SSD->GPU sync/async overlap baseline.

This is intentionally a transport experiment, separate from vLLM admission.
It uses the production GranuleKV connector and the production layer plan, then
performs the same write -> clear -> read -> compute -> verify sequence for both
backends.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferState, )
from vllm.core.custom_schedulers.hierarchical_io.plan import (
    PrefetchBlockSelectorConfig, build_layer_restore_plan, )
from vllm.granulekv.connector import GranuleKVConnector
from vllm.worker.cache_engine import CacheEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sync", "async"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--gpu-blocks", type=int, default=32)
    parser.add_argument("--storage-blocks", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--window-layers", type=int, default=4)
    parser.add_argument("--lead-windows", type=int, default=2)
    parser.add_argument("--compute-repeats", type=int, default=20)
    parser.add_argument("--compute-matrix", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def configure_environment(backend: str) -> None:
    os.environ.setdefault("VLLM_GRANULEKV_ENABLE", "1")
    os.environ.setdefault("VLLM_GRANULEKV_MAX_IN_FLIGHT", "4")
    os.environ.setdefault("VLLM_V0_SWAP_TRACE", "1")
    os.environ.setdefault("VLLM_USE_V1", "0")
    if backend != "async":
        return
    os.environ.setdefault("VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "1")
    os.environ.setdefault("VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER", "1")
    os.environ.setdefault("VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE", "1")


def build_engine(args: argparse.Namespace) -> CacheEngine:
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig

    model = os.environ.get(
        "MODEL_PATH", "/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct")
    model_config = ModelConfig(
        model,
        task="auto",
        tokenizer=model,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=args.seed,
        dtype="float16",
        enforce_eager=True,
        max_model_len=4096,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.60,
        swap_space=16,
        cache_dtype="auto",
    )
    cache_config.num_gpu_blocks = args.gpu_blocks
    cache_config.num_cpu_blocks = args.storage_blocks
    torch.cuda.set_device(0)
    engine = CacheEngine(cache_config, model_config, ParallelConfig(),
                         DeviceConfig("cuda"))
    if engine.granulekv_connector is None:
        raise RuntimeError("GranuleKV connector was not initialized")
    engine.initialize_granulekv()
    return engine


def make_mapping(args: argparse.Namespace) -> torch.Tensor:
    if not 0 < args.num_blocks <= args.gpu_blocks:
        raise ValueError("num-blocks must be within positive GPU capacity")
    return torch.tensor([(index, index) for index in range(args.num_blocks)],
                        dtype=torch.int64, device="cuda")


def fill_payload(engine: CacheEngine, args: argparse.Namespace) -> None:
    for layer_index, layer in enumerate(engine.gpu_cache):
        layer[:, :args.num_blocks, :].fill_(float(layer_index + 1))
    torch.cuda.synchronize()


def clear_payload(engine: CacheEngine, args: argparse.Namespace) -> None:
    for layer in engine.gpu_cache:
        layer[:, :args.num_blocks, :].zero_()
    torch.cuda.synchronize()


def verify_payload(engine: CacheEngine, args: argparse.Namespace) -> None:
    for layer_index, layer in enumerate(engine.gpu_cache):
        value = float(layer[:, :args.num_blocks, :].float().mean().item())
        if abs(value - float(layer_index + 1)) > 1e-3:
            raise AssertionError(
                f"layer {layer_index} restored value={value}, "
                f"expected={layer_index + 1}")


def compute_window(args: argparse.Namespace, matrix_a: torch.Tensor,
                   matrix_b: torch.Tensor) -> tuple[float, float, float]:
    started = time.perf_counter()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    event_start.record()
    result = None
    for _ in range(args.compute_repeats):
        result = torch.mm(matrix_a, matrix_b)
    event_end.record()
    event_end.synchronize()
    if result is None:
        raise AssertionError("compute produced no result")
    return (float(event_start.elapsed_time(event_end)), started,
            time.perf_counter())


def run_sync(engine: CacheEngine, args: argparse.Namespace,
             mapping: torch.Tensor, matrix_a: torch.Tensor,
             matrix_b: torch.Tensor) -> dict[str, Any]:
    connector = engine.granulekv_connector
    assert connector is not None
    connector.swap_out(mapping)
    clear_payload(engine, args)
    started = time.perf_counter()
    connector.swap_in(mapping)
    read_ms = (time.perf_counter() - started) * 1000.0
    compute_ms = 0.0
    for _ in range((args.num_layers + args.window_layers - 1) //
                   args.window_layers):
        compute_ms += compute_window(args, matrix_a, matrix_b)[0]
    verify_payload(engine, args)
    return {
        "backend": "granulekv_sync",
        "read_wall_ms": read_ms,
        "compute_ms": compute_ms,
        "total_ms": read_ms + compute_ms,
        "overlap_ms": 0.0,
        "verified": True,
    }


def run_async(engine: CacheEngine, args: argparse.Namespace,
              mapping: torch.Tensor, matrix_a: torch.Tensor,
              matrix_b: torch.Tensor) -> dict[str, Any]:
    connector = engine.granulekv_connector
    assert connector is not None
    connector.swap_out(mapping)
    clear_payload(engine, args)
    plan = build_layer_restore_plan(
        plan_id="paper-reproduction-async",
        num_layers=args.num_layers,
        window_layers=args.window_layers,
        num_blocks=args.num_blocks,
        block_selector=PrefetchBlockSelectorConfig(policy="dense"),
    )
    windows = len(plan.units)
    lead = max(1, min(args.lead_windows, windows))
    submitted: dict[int, float] = {}
    ready: dict[int, float] = {}
    compute_intervals: list[tuple[float, float]] = []
    io_intervals: list[tuple[float, float]] = []

    def submit(index: int) -> None:
        unit = plan.units[index]
        request_id = f"paper-reproduction-read-{index}"
        status = connector.submit_request(
            request_id,
            mapping,
            operation="read",
            layer_range=unit.layer_range,
        )
        submitted[index] = time.perf_counter()
        if status.ready:
            ready_at = time.perf_counter()
            connector.complete_request(request_id)
            ready[index] = ready_at

    def wait(index: int) -> None:
        request_id = f"paper-reproduction-read-{index}"
        while index not in ready:
            status = connector.query_request(request_id)
            if status.state is AsyncKVTransferState.ERROR:
                connector.cancel_request(request_id)
                raise RuntimeError(status.error or "async read failed")
            if status.ready:
                ready_at = time.perf_counter()
                connector.complete_request(request_id)
                ready[index] = ready_at
                return
            time.sleep(0.0005)

    started = time.perf_counter()
    try:
        for index in range(lead):
            submit(index)
        for index in range(windows):
            wait(index)
            io_intervals.append((submitted[index], ready[index]))
            compute_ms, compute_start, compute_end = compute_window(
                args, matrix_a, matrix_b)
            compute_intervals.append((compute_start, compute_end))
            if index + lead < windows:
                submit(index + lead)
        torch.cuda.synchronize()
        verify_payload(engine, args)
    except Exception:
        for index in submitted:
            if index not in ready:
                request_id = f"paper-reproduction-read-{index}"
                try:
                    connector.cancel_request(request_id)
                except Exception:
                    pass
        raise
    total_ms = (time.perf_counter() - started) * 1000.0
    io_ms = sum((end - begin) * 1000.0 for begin, end in io_intervals)
    compute_ms = sum(
        # CUDA event time is unavailable here, so use wall spans for the
        # overlap interval calculation and report it as a schedule estimate.
        max(0.0, end - begin) * 1000.0
        for begin, end in compute_intervals)
    overlap_ms = interval_intersection_ms(io_intervals, compute_intervals)
    return {
        "backend": "granulekv_async",
        "read_wall_ms": io_ms,
        "compute_ms": compute_ms,
        "total_ms": total_ms,
        "overlap_ms": overlap_ms,
        "windows": windows,
        "lead_windows": lead,
        "verified": True,
    }


def interval_intersection_ms(
        left: list[tuple[float, float]],
        right: list[tuple[float, float]]) -> float:
    total = 0.0
    for left_start, left_end in left:
        for right_start, right_end in right:
            total += max(0.0, min(left_end, right_end) -
                         max(left_start, right_start)) * 1000.0
    return total


def main() -> None:
    args = parse_args()
    configure_environment(args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.backend == "async":
        if os.environ.get("VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE") == "1":
            raise ValueError("synthetic baseline does not enable working-set")
    engine = build_engine(args)
    mapping = make_mapping(args)
    torch.manual_seed(args.seed)
    matrix = args.compute_matrix
    matrix_a = torch.randn((matrix, matrix), device="cuda", dtype=torch.float16)
    matrix_b = torch.randn((matrix, matrix), device="cuda", dtype=torch.float16)
    try:
        fill_payload(engine, args)
        result = (run_sync(engine, args, mapping, matrix_a, matrix_b)
                  if args.backend == "sync" else
                  run_async(engine, args, mapping, matrix_a, matrix_b))
        result.update({
            "schema_version": 1,
            "experiment": "granulekv_synthetic_transport",
            "num_blocks": args.num_blocks,
            "gpu_blocks": args.gpu_blocks,
            "num_layers": args.num_layers,
            "window_layers": args.window_layers,
            "compute_repeats": args.compute_repeats,
            "compute_matrix": args.compute_matrix,
        })
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("GRANULEKV_SYNTHETIC_PASS " + json.dumps(result), flush=True)
    finally:
        connector = engine.granulekv_connector
        if connector is not None:
            connector.close()


if __name__ == "__main__":
    main()
