# SPDX-License-Identifier: Apache-2.0

"""AsyncKVScheduler 队列迁移和 block mapping 的单元测试。"""

import pytest

import vllm.envs as envs
from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.async_kv_scheduler import AsyncKVScheduler
from vllm.core.block.interfaces import BlockAllocator
from vllm.core.scheduler import Scheduler, SchedulingBudget
from vllm.core.scheduler_policy import (AsyncKVTransferEvent,
                                        AsyncKVTransferOperation,
                                        AsyncKVTransferState)
from vllm.core.custom_schedulers.hierarchical_io import SparseKVPlanFeedback
from vllm.sequence import SequenceStatus
from vllm.utils import Device

from .utils import (append_new_token_seq, append_new_token_seq_group,
                    create_dummy_prompt,
                    schedule_and_update_computed_tokens)


def _create_scheduler(enable_prefix_caching: bool = False) -> AsyncKVScheduler:
    """创建一个同时具备 GPU block 和 storage block 的小型调度器。"""
    block_size = 4
    scheduler_config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=32,
        max_num_seqs=8,
        max_model_len=32,
        enable_chunked_prefill=True,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    cache_config.enable_prefix_caching = enable_prefix_caching
    return AsyncKVScheduler(scheduler_config, cache_config, None)


def test_granulekv_prefix_store_restore_uses_native_computed_semantics(
        monkeypatch):
    """GranuleKV prefix 应完成 populate -> SSD read -> 跳过 prefill 闭环。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    # populate 请求先按普通 prefill 计算。这里显式推进 computed token，模拟
    # 一次真实 model execution 已经完成，随后请求正常结束。
    populate_tokens = list(range(8))
    populate_seq, populate_group = create_dummy_prompt(
        "70",
        prompt_tokens=populate_tokens,
        block_size=4,
    )
    scheduler.add_seq_group(populate_group)
    schedule_and_update_computed_tokens(scheduler)
    populate_seq.status = SequenceStatus.FINISHED_STOPPED
    # 真实 V0 single-step output processor 会先 free_seq，再统一清理 group。
    scheduler.free_seq(populate_seq)
    assert populate_seq.seq_id in scheduler.block_manager.block_tables
    scheduler.free_finished_seq_groups()

    store = scheduler.drain_async_kv_transfers_to_submit()
    assert len(store) == 1
    assert store[0].operation == AsyncKVTransferOperation.WRITE
    populate_page_index_key = store[0].sparse_page_index_key
    assert populate_page_index_key
    assert store[0].request_id in scheduler.prefix_saving
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert populate_seq.seq_id not in scheduler.block_manager.block_tables

    # 正常情况下 HBM 压力会淘汰 GPU prefix。单测直接清空 GPU 层，只保留
    # CPU/storage 索引，从而确定后续命中确实经过 MDS，而不是原生 HBM hit。
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)
    reuse_tokens = populate_tokens + [100, 101, 102, 103]
    reuse_seq, reuse_group = create_dummy_prompt(
        "71",
        prompt_tokens=reuse_tokens,
        block_size=4,
    )
    scheduler.add_seq_group(reuse_group)

    assert scheduler._maybe_start_granulekv_prefix_restore()
    restore = scheduler.drain_async_kv_transfers_to_submit()
    assert len(restore) == 1
    assert restore[0].operation == AsyncKVTransferOperation.READ
    assert restore[0].sparse_page_index_key == populate_page_index_key
    assert len(restore[0].block_mapping) == 2
    assert reuse_seq.status == SequenceStatus.WAITING
    # READY 前只存在 2 个 prefix block；未命中的 suffix 尚未分配。
    assert len(scheduler.block_manager.get_block_table(reuse_seq)) == 2
    assert not scheduler.running

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(restore[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert reuse_seq.status == SequenceStatus.RUNNING
    assert list(scheduler.running) == [reuse_group]
    assert reuse_seq.get_num_computed_tokens() == len(populate_tokens)

    metadata, scheduled = schedule_and_update_computed_tokens(scheduler)
    assert scheduled.scheduled_seq_groups[0].seq_group == reuse_group
    assert len(scheduler.block_manager.get_block_table(reuse_seq)) == 3
    assert metadata[0].computed_block_nums == scheduler.block_manager.get_block_table(
        reuse_seq)[:2]

    # restore 的 SSD source 必须转交为该 sequence 的 clean replica。否则
    # 请求结束后的 prefix store 会把两个已恢复 prefix block 也重新写盘。
    assert scheduler.block_manager.get_num_clean_storage_replicas(
        reuse_seq) == 2
    reuse_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(reuse_seq)
    scheduler.free_finished_seq_groups()
    next_store = scheduler.drain_async_kv_transfers_to_submit()
    assert len(next_store) == 1
    assert next_store[0].operation == AsyncKVTransferOperation.WRITE
    assert [key.logical_index for key in next_store[0].logical_blocks] == [2]


def test_granulekv_prefix_restore_overlaps_running_compute(monkeypatch):
    """pending prefix 不应阻止已有 running 请求，也不能提前成为 hit。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    # 构造一个已写入 CPU/storage prefix cache 的历史请求。
    source_tokens = list(range(8))
    source_seq, source_group = create_dummy_prompt(
        "72", prompt_tokens=source_tokens, block_size=4)
    scheduler.add_seq_group(source_group)
    schedule_and_update_computed_tokens(scheduler)
    source_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(source_seq)
    scheduler.free_finished_seq_groups()
    store = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store.request_id, AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)

    # A 已经在 running；C/D 同时等待两个不同 suffix 的相同历史 prefix。
    active_seq, active_group = create_dummy_prompt(
        "73", prompt_tokens=list(range(40, 48)), block_size=4)
    scheduler.add_seq_group(active_group)
    schedule_and_update_computed_tokens(scheduler)
    append_new_token_seq(active_seq, 999)
    reuse_groups = []
    for request_id, suffix in (("74", [100, 101, 102, 103]),
                               ("75", [200, 201, 202, 203])):
        _, group = create_dummy_prompt(
            request_id,
            prompt_tokens=source_tokens + suffix,
            block_size=4,
        )
        scheduler.add_seq_group(group)
        reuse_groups.append(group)

    assert scheduler._maybe_start_granulekv_prefix_restore() == 2
    restores = scheduler.drain_async_kv_transfers_to_submit()
    assert len(restores) == 2
    assert all(request.operation == AsyncKVTransferOperation.READ
               for request in restores)
    # Pending target 没有注册到 GPU prefix cache，哪怕主动执行全局 mark 也
    # 不能让另一请求把尚未完成 DMA 的物理 block 当成命中。
    for group in reuse_groups:
        seq = group.first_seq
        assert len(scheduler.block_manager.get_block_table(seq)) == 2
        assert scheduler.block_manager.get_granulekv_cached_prefix_blocks(
            seq, Device.GPU) == 0
    scheduler.block_manager.block_allocator.mark_blocks_as_computed([])
    for group in reuse_groups:
        assert scheduler.block_manager.get_granulekv_cached_prefix_blocks(
            group.first_seq, Device.GPU) == 0

    # 即便 prefix read pending，A 仍能计算，且剩余 sequence slot 可以继续
    # admission 一个不命中 SSD 的普通请求；不能因为存在 I/O 冻结 waiting。
    _, ordinary_group = create_dummy_prompt(
        "79", prompt_tokens=list(range(300, 308)), block_size=4)
    scheduler.add_seq_group(ordinary_group)
    _, scheduled = schedule_and_update_computed_tokens(scheduler)
    scheduled_groups = [item.seq_group
                        for item in scheduled.scheduled_seq_groups]
    assert active_group in scheduled_groups
    assert ordinary_group in scheduled_groups

    # 乱序 READY：每个 reservation 独立发布，两个请求都进入 running。
    for request in reversed(restores):
        scheduler.apply_async_kv_event(
            AsyncKVTransferEvent(request.request_id,
                                 AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert all(group in scheduler.running for group in reuse_groups)
    for group in reuse_groups:
        assert len(scheduler.block_manager.get_block_table(
            group.first_seq)) == 2


def test_granulekv_hierarchical_prefix_first_window_admission(monkeypatch):
    """首窗 READY 只准入控制面；全窗 READY 后才能发布并执行。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "4")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "2")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    prefix_tokens = list(range(8))
    source_seq, source_group = create_dummy_prompt(
        "90", prompt_tokens=prefix_tokens, block_size=4)
    scheduler.add_seq_group(source_group)
    schedule_and_update_computed_tokens(scheduler)
    source_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(source_seq)
    scheduler.free_finished_seq_groups()
    store = scheduler.drain_async_kv_transfers_to_submit()[0]
    assert store.layer_range is None
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)

    reuse_seq, reuse_group = create_dummy_prompt(
        "91",
        prompt_tokens=prefix_tokens + [100, 101, 102, 103],
        block_size=4,
    )
    scheduler.add_seq_group(reuse_group)
    assert scheduler._maybe_start_granulekv_prefix_restore() == 1
    windows = scheduler.drain_async_kv_transfers_to_submit()
    assert [request.layer_range for request in windows] == [(0, 2), (2, 4)]
    assert len({request.reservation_id for request in windows}) == 1
    # 通用 PrefetchUnit 接口下，dense layer plan 默认仍为每个层窗读取父
    # reservation 的全部 blocks，不能因为加入可选 block 选择而缩小数据集。
    assert [len(request.block_mapping) for request in windows] == [2, 2]
    assert windows[0].block_mapping == windows[1].block_mapping
    assert windows[0].logical_blocks == windows[1].logical_blocks

    # 第一组层已经在 GPU，但完整 block hash 仍不可发布，也不能进入 running。
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(windows[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.get_hierarchical_admitted_seq_group_ids() == (
        reuse_group.request_id, )
    assert reuse_seq.status == SequenceStatus.WAITING
    assert reuse_group not in scheduler.running
    assert scheduler.block_manager.get_granulekv_cached_prefix_blocks(
        reuse_seq, Device.GPU) == 0

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(windows[1].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert not scheduler.get_hierarchical_admitted_seq_group_ids()
    assert reuse_seq.status == SequenceStatus.RUNNING
    assert reuse_group in scheduler.running
    assert scheduler.block_manager.get_granulekv_cached_prefix_blocks(
        reuse_seq, Device.GPU) == 2


def test_granulekv_sparse_prefetch_selector_is_profiling_only(monkeypatch):
    """部分 block restore 目前只做 I/O profiling，不能发布完整 prefix。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "4")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "2")
    monkeypatch.setenv("VLLM_GRANULEKV_PREFETCH_BLOCK_SELECTOR", "tail_n")
    monkeypatch.setenv("VLLM_GRANULEKV_PREFETCH_BLOCK_COUNT", "1")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    prefix_tokens = list(range(8))
    source_seq, source_group = create_dummy_prompt(
        "92", prompt_tokens=prefix_tokens, block_size=4)
    scheduler.add_seq_group(source_group)
    schedule_and_update_computed_tokens(scheduler)
    source_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(source_seq)
    scheduler.free_finished_seq_groups()
    store = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)

    reuse_seq, reuse_group = create_dummy_prompt(
        "93",
        prompt_tokens=prefix_tokens + [100, 101, 102, 103],
        block_size=4,
    )
    scheduler.add_seq_group(reuse_group)
    assert scheduler._maybe_start_granulekv_prefix_restore() == 1
    windows = scheduler.drain_async_kv_transfers_to_submit()

    # tail_n=1 只恢复父 reservation 的最后一个 block。这个请求能验证 MDS
    # 小粒度 I/O 链路，但由于缺少 sparse attention consumer，不能被当作
    # 完整 prefix hit 发布。
    assert [request.layer_range for request in windows] == [(0, 2), (2, 4)]
    assert [len(request.block_mapping) for request in windows] == [1, 1]
    assert [[key.logical_index for key in request.logical_blocks]
            for request in windows] == [[1], [1]]

    for request in windows:
        scheduler.apply_async_kv_event(
            AsyncKVTransferEvent(request.request_id,
                                 AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()

    assert reuse_seq.status == SequenceStatus.FINISHED_IGNORED
    assert reuse_group not in scheduler.running
    assert scheduler.block_manager.get_granulekv_cached_prefix_blocks(
        reuse_seq, Device.GPU) == 0


def test_granulekv_sparse_consumer_includes_local_suffix_without_reading_it(
        monkeypatch):
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "4")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "2")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_SPARSE_CONSUMER_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_PREFETCH_BLOCK_SELECTOR", "tail_n")
    monkeypatch.setenv("VLLM_GRANULEKV_PREFETCH_BLOCK_COUNT", "1")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    prefix_tokens = list(range(8))
    source_seq, source_group = create_dummy_prompt(
        "94", prompt_tokens=prefix_tokens, block_size=4)
    scheduler.add_seq_group(source_group)
    schedule_and_update_computed_tokens(scheduler)
    source_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(source_seq)
    scheduler.free_finished_seq_groups()
    store = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store.request_id, AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)

    reuse_seq, reuse_group = create_dummy_prompt(
        "95", prompt_tokens=prefix_tokens + [100, 101, 102, 103],
        block_size=4)
    scheduler.add_seq_group(reuse_group)
    assert scheduler._maybe_start_granulekv_prefix_restore() == 1
    windows = scheduler.drain_async_kv_transfers_to_submit()
    assert [len(request.block_mapping) for request in windows] == [1, 1]
    assert all(request.consumer_num_blocks == 3 for request in windows)
    assert all(request.consumer_block_indices == (1, 2)
               for request in windows)
    assert all(request.consumer_blocks_by_layer == ((1, 2), (1, 2))
               for request in windows)


def test_granulekv_layer_barrier_dispatches_after_first_unit(monkeypatch):
    """Step 4 只提前 dispatch，后续 window 完成前仍不可发布 prefix hash。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "4")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "2")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER", "1")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    prefix_tokens = list(range(8))
    source_seq, source_group = create_dummy_prompt(
        "96", prompt_tokens=prefix_tokens, block_size=4)
    scheduler.add_seq_group(source_group)
    schedule_and_update_computed_tokens(scheduler)
    source_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(source_seq)
    scheduler.free_finished_seq_groups()
    store = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store.request_id, AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)

    reuse_seq, reuse_group = create_dummy_prompt(
        "97",
        prompt_tokens=prefix_tokens + [100, 101, 102, 103],
        block_size=4,
    )
    scheduler.add_seq_group(reuse_group)
    assert scheduler._maybe_start_granulekv_prefix_restore() == 1
    windows = scheduler.drain_async_kv_transfers_to_submit()

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(windows[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    # scheduler 可从 suffix prefill 开始；真实的 layer-by-layer 安全性由
    # Worker barrier 在 forward 内保证，因而这些 pending block 尚未可命中。
    assert reuse_seq.status == SequenceStatus.RUNNING
    assert reuse_group in scheduler.running
    assert reuse_seq.get_num_computed_tokens() == len(prefix_tokens)
    assert scheduler.block_manager.get_granulekv_cached_prefix_blocks(
        reuse_seq, Device.GPU) == 0

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(windows[1].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.get_granulekv_cached_prefix_blocks(
        reuse_seq, Device.GPU) == 2
    assert scheduler.running.count(reuse_group) == 1


def test_granulekv_hierarchical_full_hit_recomputes_only_last_token(monkeypatch):
    """完整 block hit 仍须保留最后一个 prompt token 给模型执行。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "4")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "2")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    prompt_tokens = list(range(8))
    source_seq, source_group = create_dummy_prompt(
        "94", prompt_tokens=prompt_tokens, block_size=4)
    scheduler.add_seq_group(source_group)
    schedule_and_update_computed_tokens(scheduler)
    source_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(source_seq)
    scheduler.free_finished_seq_groups()
    store = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)

    reuse_seq, reuse_group = create_dummy_prompt(
        "95", prompt_tokens=prompt_tokens, block_size=4)
    scheduler.add_seq_group(reuse_group)
    assert scheduler._maybe_start_granulekv_prefix_restore() == 1
    windows = scheduler.drain_async_kv_transfers_to_submit()
    for request in windows:
        scheduler.apply_async_kv_event(
            AsyncKVTransferEvent(request.request_id,
                                 AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()

    assert reuse_seq.status == SequenceStatus.RUNNING
    assert reuse_seq.get_num_computed_tokens() == len(prompt_tokens) - 1
    _, scheduled = schedule_and_update_computed_tokens(scheduler)
    assert scheduled.scheduled_seq_groups[0].token_chunk_size == 1


def test_granulekv_hierarchical_abort_waits_for_all_windows(monkeypatch):
    """首窗准入后 abort 也必须等待其余 DMA，不能提前释放 target。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE", "1")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS", "4")
    monkeypatch.setenv("VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", "2")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    prefix_tokens = list(range(8))
    source_seq, source_group = create_dummy_prompt(
        "92", prompt_tokens=prefix_tokens, block_size=4)
    scheduler.add_seq_group(source_group)
    schedule_and_update_computed_tokens(scheduler)
    source_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(source_seq)
    scheduler.free_finished_seq_groups()
    store = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)

    reuse_seq, reuse_group = create_dummy_prompt(
        "93",
        prompt_tokens=prefix_tokens + [100, 101, 102, 103],
        block_size=4,
    )
    free_gpu_before_restore = scheduler.block_manager.get_num_free_gpu_blocks()
    scheduler.add_seq_group(reuse_group)
    assert scheduler._maybe_start_granulekv_prefix_restore() == 1
    windows = scheduler.drain_async_kv_transfers_to_submit()

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(windows[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    scheduler.abort_seq_group(reuse_group.request_id)
    assert reuse_seq.status == SequenceStatus.FINISHED_ABORTED
    assert scheduler.block_manager.get_num_free_gpu_blocks() == (
        free_gpu_before_restore - 2)

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(windows[1].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.get_num_free_gpu_blocks() == (
        free_gpu_before_restore)
    assert not scheduler.hierarchical_prefix_loading
    assert not scheduler.hierarchical_prefix_admitted
    assert not scheduler.block_manager._prefix_restore_reservations


@pytest.mark.parametrize(
    ("abort_first", "terminal_state"),
    [
        (False, AsyncKVTransferState.ERROR),
        (True, AsyncKVTransferState.READY),
        (True, AsyncKVTransferState.ERROR),
    ],
)
def test_granulekv_prefix_restore_abort_releases_pending_transaction(
        monkeypatch, abort_first, terminal_state):
    """ERROR/abort 都不能发布 hash，也不能泄漏 pending target。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    scheduler = _create_scheduler(enable_prefix_caching=True)

    # 先生成只存在于 SSD 层的两个完整 prefix block。这里沿用真实生命周期：
    # 原生 prefill 负责计算，MDS prefix store 建立 storage 索引，再淘汰 HBM。
    prefix_tokens = list(range(8))
    source_seq, source_group = create_dummy_prompt(
        "76", prompt_tokens=prefix_tokens, block_size=4)
    scheduler.add_seq_group(source_group)
    schedule_and_update_computed_tokens(scheduler)
    source_seq.status = SequenceStatus.FINISHED_STOPPED
    scheduler.free_seq(source_seq)
    scheduler.free_finished_seq_groups()
    store = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(store.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert scheduler.block_manager.reset_prefix_cache(Device.GPU)

    # restore reserve 只分配两个 pending prefix target，不为 suffix 分配 block。
    reuse_seq, reuse_group = create_dummy_prompt(
        "77",
        prompt_tokens=prefix_tokens + [100, 101, 102, 103],
        block_size=4,
    )
    free_gpu_before_restore = scheduler.block_manager.get_num_free_gpu_blocks()
    scheduler.add_seq_group(reuse_group)
    assert scheduler._maybe_start_granulekv_prefix_restore() == 1
    restore = scheduler.drain_async_kv_transfers_to_submit()[0]
    assert len(scheduler.block_manager.get_block_table(reuse_seq)) == 2
    assert scheduler.block_manager.get_num_free_gpu_blocks() == (
        free_gpu_before_restore - 2)

    # active DMA 无法安全地立即释放 target。用户 abort 只记录取消意图，等
    # READY/ERROR 终态到达后，再由同一个事务清理路径释放物理 block。
    if abort_first:
        scheduler.abort_seq_group(reuse_group.request_id)
        assert reuse_seq.status == SequenceStatus.FINISHED_ABORTED
        assert restore.request_id in scheduler.prefix_loading

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(restore.request_id, terminal_state,
                             error="injected MDS failure"))
    scheduler.complete_ready_async_kv_transfers()

    # 回收完成后，allocator 空闲数、正式 table 和事务记录必须全部复原；更
    # 关键的是，相同 token 的 GPU lookup 仍为 0，失败 DMA 绝不能发布 hash。
    assert reuse_seq.seq_id not in scheduler.block_manager.block_tables
    assert scheduler.block_manager.get_num_free_gpu_blocks() == (
        free_gpu_before_restore)
    assert scheduler.block_manager.get_granulekv_cached_prefix_blocks(
        reuse_seq, Device.GPU) == 0
    assert not scheduler.block_manager._prefix_restore_reservations
    assert not scheduler.prefix_loading
    assert not scheduler._cancelled_async_kv_requests
    assert not scheduler.has_active_async_kv_transfer()


def test_granulekv_prefix_restore_does_not_bypass_blocked_priority_head(
        monkeypatch):
    """高优先级 restore 暂不可分配时，后续请求不能抢先预留 HBM。"""
    monkeypatch.setenv("VLLM_GRANULEKV_PREFIX_ENABLE", "1")
    scheduler = _create_scheduler(enable_prefix_caching=True)
    scheduler.scheduler_config.policy = "priority"

    _, lower_group = create_dummy_prompt(
        "80", prompt_tokens=list(range(8)), block_size=4)
    _, higher_group = create_dummy_prompt(
        "81", prompt_tokens=list(range(8)), block_size=4)
    lower_group.priority = 10
    higher_group.priority = 0
    scheduler.add_seq_group(lower_group)
    scheduler.add_seq_group(higher_group)

    attempted_request_ids = []

    def fail_reservation(seq_group, num_prefix_blocks,
                         num_gpu_cached_blocks):
        attempted_request_ids.append(seq_group.request_id)
        raise BlockAllocator.NoFreeBlocksError()

    monkeypatch.setattr(scheduler.block_manager,
                        "get_granulekv_cached_prefix_block_counts",
                        lambda seq: (2, 0))
    monkeypatch.setattr(scheduler.block_manager,
                        "reserve_granulekv_prefix_restore", fail_reservation)

    scheduler._sort_waiting_for_prefix_restore()
    assert scheduler._maybe_start_granulekv_prefix_restore() == 0
    assert attempted_request_ids == [higher_group.request_id]
    assert list(scheduler.waiting) == [higher_group, lower_group]
    assert not scheduler.prefix_loading

    # 该请求已经确认有 SSD prefix，只是本轮 target 不足。父类仍可推进已有
    # running，但不能把 waiting 请求直接 admission 成完整 recompute。
    outputs = scheduler._schedule_chunked_prefill_with_reserved_slots()
    assert not outputs.scheduled_seq_groups
    assert list(scheduler.waiting) == [higher_group, lower_group]


def _sync_swap_out(scheduler: AsyncKVScheduler, seq_group) -> None:
    """只用于构造已有 SSD 副本；被测异步路径仍由 AsyncKVScheduler 执行。"""
    Scheduler._swap_out(scheduler, seq_group, [])


def _restore_with_clean_storage_replica(scheduler: AsyncKVScheduler,
                                        request_id: str):
    """构造 GPU 正式表和 SSD clean replica 同时存在的 decode 请求。"""
    seq, seq_group = create_dummy_prompt(
        request_id, prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(seq_group)
    _sync_swap_out(scheduler, seq_group)
    scheduler._add_seq_group_to_swapped(seq_group)
    scheduler._schedule_swapped(
        SchedulingBudget(token_budget=32, max_num_seqs=8), None)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    return seq, seq_group


def test_async_scheduler_requires_chunked_prefill():
    """异步恢复必须依赖有界的 chunk 调度边界。"""
    scheduler_config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=32,
        max_num_seqs=8,
        max_model_len=32,
        enable_chunked_prefill=False,
    )
    cache_config = CacheConfig(4, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    with pytest.raises(ValueError, match="requires chunked prefill"):
        AsyncKVScheduler(scheduler_config, cache_config, None)


def test_swapped_request_waits_for_async_kv_ready():
    """KV 未 READY 时不计算，READY 后才进入下一轮 decode。"""
    scheduler = _create_scheduler()
    seq, seq_group = create_dummy_prompt(
        "1", prompt_length=8, block_size=4)

    # 先构造一个已经完成 prefill、随后被 swap-out 的 decode 请求。
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(8, seq_group, 99)
    _sync_swap_out(scheduler, seq_group)
    scheduler._add_seq_group_to_swapped(seq_group)
    assert seq.status == SequenceStatus.SWAPPED

    # 新策略只预留 GPU block 并生成异步请求，不能把该请求放进当前 batch。
    storage_block_ids = list(scheduler.block_manager.get_block_table(seq))
    free_gpu_before_reserve = scheduler.block_manager.get_num_free_gpu_blocks()
    output = scheduler._schedule_swapped(
        SchedulingBudget(token_budget=32, max_num_seqs=8), None)
    requests = scheduler.drain_async_kv_transfers_to_submit()
    assert not output.decode_seq_groups
    assert not output.prefill_seq_groups
    assert not output.blocks_to_swap_in
    assert len(requests) == 1
    assert requests[0].seq_group_id == seq_group.request_id
    assert requests[0].operation == AsyncKVTransferOperation.READ
    assert requests[0].block_mapping
    assert len(requests[0].logical_blocks) == len(requests[0].block_mapping)
    # reserve 阶段正式表仍指向 storage，目标 GPU block 只是被预留。
    assert scheduler.block_manager.get_block_table(seq) == storage_block_ids
    assert (scheduler.block_manager.get_num_free_gpu_blocks()
            < free_gpu_before_reserve)
    assert seq.status == SequenceStatus.SWAPPED
    assert scheduler.has_active_async_kv_transfer()
    assert scheduler.get_num_unfinished_seq_groups() == 1

    # PENDING 事件不得改变队列或 SequenceStatus。
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(requests[0].request_id,
                             AsyncKVTransferState.PENDING))
    scheduler.complete_ready_async_kv_transfers()
    assert seq.status == SequenceStatus.SWAPPED
    assert not scheduler.running

    # READY 后先完成 swapped -> running，再由父类普通调度执行 decode。
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(requests[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert seq.status == SequenceStatus.RUNNING
    assert scheduler.block_manager.get_block_table(seq) != storage_block_ids
    assert list(scheduler.running) == [seq_group]
    assert not scheduler.loading

    metadata, scheduled = schedule_and_update_computed_tokens(scheduler)
    assert len(metadata) == 1
    assert [item.seq_group
            for item in scheduled.scheduled_seq_groups] == [seq_group]
    assert not scheduled.blocks_to_swap_in

    # READY -> RUNNING 的观测标记只在请求第一次进入执行 batch 前消费，
    # 重复消费不会产生第二个 first_execute 事件。
    markers = scheduler.consume_async_kv_execution_markers(
        [seq_group.request_id])
    assert len(markers) == 1
    assert markers[0].request_id == requests[0].request_id
    assert markers[0].seq_group_id == seq_group.request_id
    assert not scheduler.consume_async_kv_execution_markers(
        [seq_group.request_id])


def test_abort_loading_request_defers_block_free_until_ready():
    """loading 期间 abort 不得提前释放 MDS 正在写入的 GPU block。"""
    scheduler = _create_scheduler()
    seq, seq_group = create_dummy_prompt(
        "2", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(8, seq_group, 100)
    _sync_swap_out(scheduler, seq_group)
    scheduler._add_seq_group_to_swapped(seq_group)

    scheduler._schedule_swapped(
        SchedulingBudget(token_budget=32, max_num_seqs=8), None)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]

    scheduler.abort_seq_group(seq_group.request_id)
    assert seq.status == SequenceStatus.FINISHED_ABORTED
    assert request.request_id in scheduler.loading
    assert scheduler.has_unfinished_seqs()

    # MDS 完成后才真正释放 block 并移除 loading 生命周期。
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert not scheduler.loading
    assert not scheduler.running
    assert not scheduler.has_unfinished_seqs()


def test_ready_decode_runs_with_remaining_prefill_chunk():
    """READY decode 应在下一个边界与剩余 prefill chunk 一起调度。"""
    scheduler = _create_scheduler()

    restored_seq, restored_group = create_dummy_prompt(
        "10", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(restored_group)
    append_new_token_seq_group(8, restored_group, 101)
    _sync_swap_out(scheduler, restored_group)
    scheduler._add_seq_group_to_swapped(restored_group)

    _, prefill_group = create_dummy_prompt(
        "11", prompt_length=40, block_size=4)
    scheduler.add_seq_group(prefill_group)

    # 第一轮提交异步恢复，同时只计算长 prompt 的第一个 32-token chunk。
    first_metadata, first_output = schedule_and_update_computed_tokens(
        scheduler)
    requests = scheduler.drain_async_kv_transfers_to_submit()
    assert len(requests) == 1
    assert [item.seq_group
            for item in first_output.scheduled_seq_groups] == [prefill_group]
    assert first_metadata[0].token_chunk_size == 32
    assert restored_seq.status == SequenceStatus.SWAPPED
    assert prefill_group.is_prefill()

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(requests[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()

    # 第二轮优先恢复 decode，但 32-token budget 仍足以容纳剩余 8-token
    # prefill；SchedulerOutputs 为 attention backend 保持 prefill 在前。
    second_metadata, second_output = schedule_and_update_computed_tokens(
        scheduler)
    scheduled_groups = [
        item.seq_group for item in second_output.scheduled_seq_groups
    ]
    assert scheduled_groups == [prefill_group, restored_group]
    assert second_output.num_prefill_groups == 1
    assert [metadata.token_chunk_size
            for metadata in second_metadata] == [8, 1]
    assert restored_seq.status == SequenceStatus.RUNNING


def test_async_swap_out_keeps_gpu_blocks_until_write_ready():
    """异步 write 完成前 GPU source 不得从正式 block table 释放。"""
    scheduler = _create_scheduler()
    seq, seq_group = create_dummy_prompt(
        "20", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(8, seq_group, 202)

    gpu_block_ids = list(scheduler.block_manager.get_block_table(seq))
    free_gpu_before = scheduler.block_manager.get_num_free_gpu_blocks()
    scheduler._swap_out(seq_group, [])
    scheduler._add_seq_group_to_swapped(seq_group)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]

    assert request.operation == AsyncKVTransferOperation.WRITE
    assert seq.status == SequenceStatus.SWAPPED
    assert scheduler.block_manager.get_block_table(seq) == gpu_block_ids
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free_gpu_before
    assert request.request_id in scheduler.saving

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()

    assert request.request_id not in scheduler.saving
    assert scheduler.block_manager.get_block_table(seq) != gpu_block_ids
    assert scheduler.block_manager.get_num_free_gpu_blocks() > free_gpu_before
    assert list(scheduler.swapped) == [seq_group]


def test_async_writes_use_multiple_granulekv_slots():
    """多笔 write 应在同一轮按 request identity 独立激活和完成。"""
    scheduler = _create_scheduler()
    groups = []
    for index in range(2):
        _, seq_group = create_dummy_prompt(
            str(30 + index), prompt_length=8, block_size=4)
        scheduler._allocate_and_set_running(seq_group)
        scheduler._swap_out(seq_group, [])
        scheduler._add_seq_group_to_swapped(seq_group)
        groups.append(seq_group)

    activated = scheduler.drain_async_kv_transfers_to_submit()
    assert len(activated) == 2
    assert scheduler.drain_async_kv_transfers_to_submit() == ()
    assert not scheduler.async_kv_policy.queued_request_ids

    # completion 不要求遵循提交顺序；每笔 reservation 由 request_id 定位。
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(activated[1].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert activated[1].request_id not in scheduler.saving
    assert activated[0].request_id in scheduler.saving
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(activated[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert not scheduler.saving


def test_abort_active_write_keeps_gpu_source_until_completion():
    """取消 write 时不能让父类提前释放 MDS 正在读取的 GPU 地址。"""
    scheduler = _create_scheduler()
    seq, seq_group = create_dummy_prompt(
        "40", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(seq_group)
    gpu_block_ids = list(scheduler.block_manager.get_block_table(seq))
    scheduler._swap_out(seq_group, [])
    scheduler._add_seq_group_to_swapped(seq_group)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]

    scheduler.abort_seq_group(seq_group.request_id)
    assert seq.status == SequenceStatus.FINISHED_ABORTED
    assert scheduler.block_manager.get_block_table(seq) == gpu_block_ids
    assert request.request_id in scheduler.saving

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert seq.seq_id not in scheduler.block_manager.block_tables
    assert not scheduler.has_unfinished_seqs()


def test_clean_storage_replica_avoids_write_io():
    """GPU 内容未变化时，swap-out 应直接复用 SSD replica。"""
    scheduler = _create_scheduler()
    seq, seq_group = _restore_with_clean_storage_replica(scheduler, "50")
    assert scheduler.block_manager.get_num_clean_storage_replicas(seq) == 2

    scheduler._swap_out(seq_group, [])
    scheduler._add_seq_group_to_swapped(seq_group)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]
    assert request.operation == AsyncKVTransferOperation.WRITE
    assert request.block_mapping == ()
    assert request.logical_blocks == ()

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert seq.status == SequenceStatus.SWAPPED
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 16


def test_only_dirty_suffix_is_written():
    """decode 新增尾块后，只写 dirty suffix，不重写 clean prefix。"""
    scheduler = _create_scheduler()
    seq, seq_group = _restore_with_clean_storage_replica(scheduler, "51")
    append_new_token_seq_group(8, seq_group, 303)
    scheduler.block_manager.append_slots(seq, num_lookahead_slots=0)
    assert len(scheduler.block_manager.get_block_table(seq)) == 3

    scheduler._swap_out(seq_group, [])
    request = scheduler.drain_async_kv_transfers_to_submit()[0]
    assert len(request.block_mapping) == 1
    assert [key.logical_index for key in request.logical_blocks] == [2]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert len(scheduler.block_manager.get_block_table(seq)) == 3


def test_free_gpu_sequence_releases_storage_replicas():
    """sequence 结束时，正式 GPU 表和目录中的 SSD replica 都必须释放。"""
    scheduler = _create_scheduler()
    seq, _ = _restore_with_clean_storage_replica(scheduler, "52")
    assert scheduler.block_manager.get_num_free_cpu_blocks() == 14
    scheduler.free_seq(seq)
    assert scheduler.block_manager.get_num_free_cpu_blocks() == 16
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 16


def test_per_block_residency_and_pin_control():
    """调度策略可以查询、pin 并释放任意逻辑 block。"""
    scheduler = _create_scheduler()
    seq, seq_group = _restore_with_clean_storage_replica(scheduler, "53")

    residency = scheduler.block_manager.get_block_residency(seq)
    assert [item.key.logical_index for item in residency] == [0, 1]
    assert all(item.storage_replica_clean for item in residency)
    assert all(item.storage_block_id is not None for item in residency)

    scheduler.block_manager.pin_blocks(seq, [1])
    residency = scheduler.block_manager.get_block_residency(seq)
    assert [item.pin_count for item in residency] == [0, 1]
    assert not scheduler.block_manager.can_reserve_swap_out(seq_group)

    scheduler.block_manager.unpin_blocks(seq, [1])
    assert scheduler.block_manager.can_reserve_swap_out(seq_group)
    with pytest.raises(RuntimeError, match="not pinned"):
        scheduler.block_manager.unpin_blocks(seq, [1])


def test_queued_read_precedes_deferred_write():
    """active write 完成后，应先提交已排队的 critical read。"""
    scheduler = _create_scheduler()
    _, write_group = create_dummy_prompt(
        "60", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(write_group)
    scheduler._swap_out(write_group, [])
    scheduler._add_seq_group_to_swapped(write_group)
    active_write = scheduler.drain_async_kv_transfers_to_submit()[0]

    _, read_group = create_dummy_prompt(
        "61", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(read_group)
    _sync_swap_out(scheduler, read_group)
    scheduler._add_seq_group_to_swapped(read_group)
    scheduler._schedule_swapped(
        SchedulingBudget(token_budget=32, max_num_seqs=8), None)

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(active_write.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    next_request = scheduler.drain_async_kv_transfers_to_submit()[0]
    assert next_request.operation == AsyncKVTransferOperation.READ
    assert next_request.seq_group_id == read_group.request_id


def test_scheduler_consumes_dynamic_sparse_restore_feedback(monkeypatch):
    scheduler = AsyncKVScheduler.__new__(AsyncKVScheduler)
    scheduler._sparse_restore_plans = {}
    scheduler.hierarchical_io_config = type(
        "Config", (), {
            "consumer_enabled": True,
            "num_layers": 2,
        })()
    monkeypatch.setattr(envs, "VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE",
                        True)
    feedback = SparseKVPlanFeedback("prefix", 2, 4, ((0, 2), (1, 3)))

    scheduler.accept_sparse_kv_plan_feedback((feedback, ))
    plan = scheduler._peek_sparse_restore_plan("prefix", 4)
    assert plan is not None
    assert plan.source == "quest_dynamic"
    assert plan.block_indices_by_layer == ((0, 2), (1, 3))
    assert scheduler._peek_sparse_restore_plan("other", 4) is None
