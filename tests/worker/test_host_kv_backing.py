# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import vllm.worker.worker as module
from vllm.worker.worker import Worker
from vllm.worker.cache_engine import CacheEngine
from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferOperation, AsyncKVTransferPriority,
    AsyncKVTransferRequest, AsyncKVTransferState)
from vllm.core.block_reservation import LogicalBlockKey


class Fence:
    def __init__(self, ready=False, **kwargs):
        self.ready = ready

    def query(self):
        return self.ready

    def record(self, stream):
        pass

    def synchronize(self):
        pytest.fail("backing path must not synchronize on the CPU")


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_HOST_BACKING_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_EVENT_FENCED_LAYER_REUSE_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_GPU_RESIDENCY_REUSE_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_CPU_CACHE_BLOCKS", 2304)
    result = object.__new__(Worker)
    for name in ("_gpu_resident_blocks", "_gpu_destination_owners",
                 "_gpu_region_release_events", "_pending_gpu_to_host",
                 "_backing_forward", "_deferred_gpu_transfers",
                 "_host_async_kv_transfers", "_gpu_transfer_regions",
                 "_gpu_pending_transfers", "_pending_h2d_transfers"):
        setattr(result, name, {})
    result._pending_host_writes = {}
    for name in ("_host_ready_blocks", "_host_staged_blocks",
                 "_host_failed_blocks", "_backing_retired", "_ssd_read_blocks"):
        setattr(result, name, set())
    result._host_stats = defaultdict(int)
    result.cache_config = SimpleNamespace(block_size=16)
    result.model_config = result.parallel_config = None
    result.cache_engine = [SimpleNamespace(
        num_attention_layers=28,
        granulekv_connector=SimpleNamespace(layout=SimpleNamespace(num_gpu_regions=8)))]
    monkeypatch.setattr(CacheEngine, "get_cache_block_size", lambda *args: 28 * 32768)
    monkeypatch.setattr(torch.cuda, "Event", Fence)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(
        wait_event=lambda event: None))
    return result


def test_host_ready_only_after_latest_d2h_completes(worker):
    key = (0, "r", 0, 7)
    old, current = Fence(True), Fence()
    worker._pending_gpu_to_host[key] = old
    worker._pending_gpu_to_host[key] = current
    worker._poll_gpu_to_host_backing(0)
    assert key not in worker._host_ready_blocks
    current.ready = True
    worker._poll_gpu_to_host_backing(0)
    assert key in worker._host_ready_blocks
    assert not worker._pending_gpu_to_host


def test_retired_request_cannot_publish_host_backing(worker):
    key = (0, "r", 0, 7)
    worker._pending_gpu_to_host[key] = Fence(True)
    worker._backing_retired.add((0, "r"))
    worker._poll_gpu_to_host_backing(0)
    assert key not in worker._host_ready_blocks
    assert not worker._pending_gpu_to_host


def test_finish_invalidates_owners_but_keeps_pending_backup_fence(worker):
    key = (0, "r", 0, 7)
    fence = Fence()
    worker._discarded_host_transfers = set()
    worker._host_staging_owner = (0, "r")
    worker._gpu_resident_blocks[key] = 3
    worker._gpu_destination_owners[0, 0, 3] = key
    worker._gpu_region_release_events[0, 0, 3] = ("r", fence)
    worker._pending_gpu_to_host[key] = fence
    worker._host_staged_blocks.add(key)
    worker._forget_host_staging(0, ("r",))
    assert not worker._gpu_resident_blocks
    assert not worker._gpu_destination_owners
    assert key not in worker._host_ready_blocks
    assert worker._gpu_region_release_events[0, 0, 3][1] is fence
    assert worker._host_staging_owner == (0, "r")
    fence.ready = True
    worker._drain_host_staging_owner(0, "r", 1)
    assert worker._host_staging_owner is None
    assert not worker._backing_retired
    assert key not in worker._host_ready_blocks


def test_abort_before_first_forward_has_no_cached_metadata(worker):
    worker._seq_group_metadata_cache = {}
    assert worker._get_cached_seq_group_metadata([], ["not-started"]) == []


def test_back_up_only_gpu_only_and_dirty_tail(worker):
    worker._backing_forward[0, "r"] = ((3, 4, 5), 33, 34, False)
    for block, destination in enumerate((3, 4, 5)):
        key = (0, "r", 0, block)
        worker._gpu_resident_blocks[key] = destination
        worker._gpu_destination_owners[0, 0, destination] = key
    worker._host_ready_blocks.update(((0, "r", 0, 1), (0, "r", 0, 2)))
    copies = []
    ready = Fence()
    worker.cache_engine[0].copy_gpu_blocks_to_host_async = (
        lambda layer, mapping, fences: copies.append(mapping.tolist()) or ready)
    worker._back_up_layer(0, ("r",), 0)
    assert copies == [[[3, 0], [5, 2]]]
    assert worker._pending_gpu_to_host[0, "r", 0, 0] is ready
    assert (0, "r", 0, 2) not in worker._host_ready_blocks
    assert worker._gpu_region_release_events[0, 0, 3][1] is ready
    assert worker._host_stats["gpu_to_host_blocks"] == 2


def test_pending_d2h_is_stream_dependency_not_cpu_wait(worker):
    key = (0, "r", 0, 7)
    worker._pending_gpu_to_host[key] = Fence()
    assert worker.wait_for_host_blocks(0, "r", 0, (7,), 0)


@pytest.mark.parametrize("computed,expected", [(32, ()), (33, (2,))])
def test_sparse_prefill_restores_only_partial_tail_before_query(
        worker, monkeypatch, computed, expected):
    monkeypatch.setattr(module.envs,
                        "VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE", True)
    worker._prefetch_runtime = SimpleNamespace(
        prediction_state=lambda *args: SimpleNamespace(num_prefix_blocks=2))
    worker._backing_forward[0, "r"] = (tuple(range(8)), computed,
                                       computed + 17, True)
    restored = []
    worker._restore_backing_blocks = lambda ve, req, layer, table, blocks: (
        restored.append(tuple(blocks)))
    worker._restore_live_backing(0, ("r",), 0)
    assert restored == [expected]


def test_backing_restore_reuses_owner_and_does_not_repeat_pending_copy(worker):
    copied = []
    worker.wait_for_host_blocks = lambda *args: True
    worker.copy_host_blocks_to_gpu_async = (
        lambda ve, req, layer, mapping, blocks: copied.append(mapping.tolist()))
    table = (3, 4, 5)
    worker._restore_backing_blocks(0, "r", 0, table, (0, 2))
    worker._restore_backing_blocks(0, "r", 0, table, (0, 2))
    assert copied == [[[0, 3], [2, 5]]]
    worker._mark_gpu_blocks_ready(0, "r", 8, (2,), ((2, 5),))
    worker._restore_backing_blocks(0, "r", 0, table, (0, 2))
    assert copied[-1] == [[2, 5]]


def test_sparse_backup_includes_unused_gpu_only_prediction(worker):
    from vllm.core.custom_schedulers.hierarchical_io import activate_sparse_kv_blocks
    worker._backing_forward[0, "r"] = (tuple(range(3, 103)), 33, 34, True)
    worker._prefetch_runtime = SimpleNamespace(prediction_state=lambda *args:
        SimpleNamespace(predicted_prefix_blocks=(0,)))
    for block, destination in enumerate((3, 4, 5)):
        worker._mark_gpu_blocks_ready(0, "r", 0, (block,), ((block, destination),))
    worker._host_ready_blocks.add((0, "r", 0, 1))
    copies = []
    worker.cache_engine[0].copy_gpu_blocks_to_host_async = (
        lambda layer, mapping, fences: copies.append(mapping.tolist()) or Fence())
    with activate_sparse_kv_blocks((1,)):
        worker._back_up_layer(0, ("r",), 0)
    assert copies == [[[3, 0], [5, 2]]]


def test_host_capacity_checked_before_region_allocation(monkeypatch):
    for name in ("VLLM_GRANULEKV_ENABLE", "VLLM_GRANULEKV_HOST_BACKING_ENABLE",
                 "VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE",
                 "VLLM_GRANULEKV_SPARSE_CPU_STAGING_ENABLE",
                 "VLLM_GRANULEKV_SPARSE_GPU_RESIDENCY_REUSE_ENABLE"):
        monkeypatch.setattr(module.envs, name, True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_CPU_CACHE_BLOCKS", 2048)
    with pytest.raises(ValueError, match="capacity must cover 2052 blocks"):
        CacheEngine(SimpleNamespace(block_size=16),
                    SimpleNamespace(max_model_len=32832),
                    SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
                    None)


def test_standard_worker_input_initializes_backing_metadata(worker):
    worker._host_staging_owner = None
    worker._seq_group_metadata_cache = {}
    worker._prefetch_runtime = SimpleNamespace(forget_seq_groups=lambda ids: None)
    # Use the real SequenceGroupMetadata object, as in the V0 driver path.
    from vllm.sequence import SequenceData, SequenceGroupMetadata
    from vllm import SamplingParams
    from array import array
    data = SequenceData(array("l", range(32)))
    data.update_num_computed_tokens(31)
    metadata = SequenceGroupMetadata(
        request_id="r", is_prompt=True, seq_data={0: data},
        sampling_params=SamplingParams(max_tokens=8), block_tables={0: [3, 4]},
        token_chunk_size=1)
    request = SimpleNamespace(virtual_engine=0, finished_requests_ids=[],
                              seq_group_metadata_list=[metadata])
    worker._prepare_host_backing_forward(request)
    assert worker._backing_forward[0, "r"] == ((3, 4), 31, 32, True)
    assert worker._host_staging_owner == (0, "r")


@pytest.mark.parametrize("layer_range", [(8, 9), (8, 12), (0, 28)])
def test_native_read_is_deferred_until_backup_source_is_released(worker, layer_range):
    fence = Fence()
    worker._gpu_region_release_events[0, 0, 3] = ("r", fence)
    request = AsyncKVTransferRequest(
        request_id="io", seq_group_id="r", reservation_id="reservation",
        operation=AsyncKVTransferOperation.READ,
        priority=AsyncKVTransferPriority.CRITICAL_READ,
        logical_blocks=(LogicalBlockKey(0, 0),),
        block_mapping=((10, 3),), layer_range=layer_range)
    mapping = torch.tensor(request.block_mapping)
    engine = worker.cache_engine[0]
    submitted = []
    engine.submit_async_kv_transfer = lambda *args, **kwargs: (
        submitted.append(args) or AsyncKVTransferEvent("io", AsyncKVTransferState.READY))
    assert worker._submit_gpu_transfer(0, engine, request, mapping).state == (
        AsyncKVTransferState.PENDING)
    assert not submitted
    fence.ready = True
    assert worker._poll_gpu_transfer(0, engine, "io").state == AsyncKVTransferState.READY
    assert len(submitted) == 1
    assert worker._host_stats["gpu_direct_read_bytes"] == (
        32768 * (layer_range[1] - layer_range[0]))


def test_host_write_waits_for_d2h_then_uses_existing_native_request(worker):
    key = (0, "r", 0, 7)
    ready = Fence()
    worker._host_staged_blocks.add(key)
    worker._pending_gpu_to_host[key] = ready
    request = AsyncKVTransferRequest(
        request_id="write", seq_group_id="r", reservation_id="reservation",
        operation=AsyncKVTransferOperation.WRITE,
        priority=AsyncKVTransferPriority.DEFERRED_WRITE,
        logical_blocks=(LogicalBlockKey(0, 7),), block_mapping=((7, 90),),
        layer_range=(0, 1), destination="host")
    engine = worker.cache_engine[0]
    calls = []
    engine.submit_async_kv_transfer = lambda *args, **kwargs: (
        calls.append(kwargs) or AsyncKVTransferEvent("write", AsyncKVTransferState.READY))
    mapping = torch.tensor(request.block_mapping)
    assert worker._submit_gpu_transfer(0, engine, request, mapping).state == AsyncKVTransferState.PENDING
    assert not calls
    ready.ready = True
    assert worker._poll_gpu_transfer(0, engine, "write").state == AsyncKVTransferState.READY
    assert calls == [{"layer_range": (0, 1), "destination": "host"}]
    assert worker._host_stats["host_to_ssd_submit_count"] == 1


def test_rewind_remains_explicitly_rejected(worker):
    from array import array
    from vllm.sequence import SequenceData, SequenceGroupMetadata
    from vllm import SamplingParams
    worker._host_staging_owner = (0, "r")
    worker._seq_group_metadata_cache = {}
    worker._prefetch_runtime = SimpleNamespace(forget_seq_groups=lambda ids: None)
    worker._backing_forward[0, "r"] = ((3, 4), 32, 33, False)
    data = SequenceData(array("l", range(32)))
    metadata = SequenceGroupMetadata(
        request_id="r", is_prompt=True, seq_data={0: data},
        sampling_params=SamplingParams(max_tokens=8), block_tables={0: [3, 4]},
        token_chunk_size=1)
    with pytest.raises(RuntimeError, match="request rewind"):
        worker._prepare_host_backing_forward(SimpleNamespace(
            virtual_engine=0, finished_requests_ids=[],
            seq_group_metadata_list=[metadata]))
