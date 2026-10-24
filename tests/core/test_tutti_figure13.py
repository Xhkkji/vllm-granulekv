# SPDX-License-Identifier: Apache-2.0

"""Tutti Figure 13 workload 和专用 scheduler 约束测试。"""

import pytest

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.custom_schedulers.hierarchical_io.figure13.scheduler import (
    TuttiFigure13Scheduler)
from vllm.core.custom_schedulers.hierarchical_io.figure13.workload import (
    Figure13SweepConfig, build_reuse_prompt_tokens, parse_prefix_tokens)


def test_figure13_sweep_keeps_total_length_and_exact_prefix():
    sweep = Figure13SweepConfig.create(total_tokens=32,
                                       prefix_tokens=(16, 24, 32),
                                       block_size=8)
    assert [(point.prefix_tokens, point.suffix_tokens,
             point.hit_rate) for point in sweep.points] == [
                 (16, 16, 0.5),
                 (24, 8, 0.75),
                 (32, 0, 1.0),
             ]

    base = tuple(range(32))
    replacement = tuple(range(100, 132))
    prompt = build_reuse_prompt_tokens(base, replacement, sweep.points[1])
    assert len(prompt) == 32
    assert prompt[:24] == base[:24]
    assert prompt[24:] == replacement[24:]
    assert build_reuse_prompt_tokens(base, replacement,
                                     sweep.points[2]) == base


def test_figure13_sweep_rejects_unaligned_or_unsorted_points():
    assert parse_prefix_tokens("16, 24,32") == (16, 24, 32)
    with pytest.raises(ValueError, match="strictly increasing"):
        Figure13SweepConfig.create(total_tokens=32,
                                   prefix_tokens=(24, 16),
                                   block_size=8)
    with pytest.raises(ValueError, match="align"):
        Figure13SweepConfig.create(total_tokens=32,
                                   prefix_tokens=(12, ),
                                   block_size=8)


def test_figure13_zero_prefix_and_partial_prompt_tail():
    sweep = Figure13SweepConfig.create(total_tokens=35,
                                       prefix_tokens=(0, 32), block_size=16)
    assert [(point.prefix_tokens, point.suffix_tokens) for point in sweep.points] == [
        (0, 35), (32, 3)]
    base, replacement = tuple(range(35)), tuple(range(100, 135))
    assert build_reuse_prompt_tokens(base, replacement, sweep.points[0]) == replacement


def _scheduler_config(max_num_seqs: int = 1,
                      policy: str = "fcfs") -> SchedulerConfig:
    config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=32,
        max_num_seqs=max_num_seqs,
        max_model_len=64,
        enable_chunked_prefill=True,
    )
    config.policy = policy
    return config


def _cache_config() -> CacheConfig:
    config = CacheConfig(4, 1.0, 1, "auto")
    config.num_cpu_blocks = 32
    config.num_gpu_blocks = 32
    config.enable_prefix_caching = True
    return config


def _enable_hierarchical_figure13(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "4")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "2")
    monkeypatch.setenv("VLLM_GRANULEKV_ASYNC_SCHEDULER_STRATEGY", "native")


def test_figure13_scheduler_is_explicit_fcfs_single_request(monkeypatch):
    _enable_hierarchical_figure13(monkeypatch)
    scheduler = TuttiFigure13Scheduler(_scheduler_config(), _cache_config(),
                                       None)
    assert scheduler.scheduler_strategy == (
        "tutti_figure13:fcfs_single_request:window_2_layers")


@pytest.mark.parametrize(("max_num_seqs", "policy", "message"), [
    (2, "fcfs", "max_num_seqs=1"),
    (1, "priority", "FCFS"),
])
def test_figure13_scheduler_rejects_queueing_confounds(
        monkeypatch, max_num_seqs, policy, message):
    _enable_hierarchical_figure13(monkeypatch)
    with pytest.raises(ValueError, match=message):
        TuttiFigure13Scheduler(
            _scheduler_config(max_num_seqs=max_num_seqs, policy=policy),
            _cache_config(), None)
