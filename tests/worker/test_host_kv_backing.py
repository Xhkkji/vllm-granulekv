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
    monkeypatch.setattr(result, "_uses_bounded_slots", lambda: False)
    for name in ("_gpu_resident_blocks", "_gpu_destination_owners",
                 "_gpu_slot_mappings", "_gpu_required_blocks",
                 "_gpu_region_release_events", "_pending_gpu_to_host",
                 "_backing_forward", "_deferred_gpu_transfers",
                 "_host_async_kv_transfers", "_gpu_transfer_regions",
                 "_gpu_pending_transfers", "_pending_h2d_transfers"):
        setattr(result, name, {})
    result._pending_host_writes = {}
    result._prefetch_runtime = module.RollingPrefetchRuntime(
        module.RollingPrefetchConfig())
    for name in ("_host_ready_blocks", "_host_staged_blocks",
                 "_host_failed_blocks", "_backing_retired", "_ssd_read_blocks"):
        setattr(result, name, set())
    result._host_stats = defaultdict(int)
    result._gpu_working_set_bound = 1
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


def test_bounded_slots_preserve_hits_and_reuse_safe_oldest(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker.cache_engine[0].num_gpu_blocks = 2
    assert worker._reserve_gpu_slots(0, "r", 0, (10, 11)) == (10, 11)
    assert worker._layer_block_table(0, "r", 0, 12)[10:] == (0, 1)
    worker._gpu_required_blocks[0, "r", 0] = {11}
    old = (0, "r", 0, 10)
    worker._host_ready_blocks.add(old)
    worker._gpu_region_release_events[0, 0, 0] = ("r", Fence())
    worker._reserve_gpu_slots(0, "r", 0, (11, 12))
    assert old not in worker._gpu_slot_mappings
    assert worker._gpu_slot_mappings[0, "r", 0, 11] == 1
    assert worker._gpu_slot_mappings[0, "r", 0, 12] == 0
    assert worker._host_stats["gpu_slot_evictions"] == 1
    assert worker._reserve_gpu_slots(0, "r", 1, (10, 11)) == (10, 11)
    assert worker._gpu_slot_mappings[0, "r", 1, 10] == 0


def test_bounded_slots_reject_dirty_overwrite_and_overbudget(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker.cache_engine[0].num_gpu_blocks = 1
    worker._reserve_gpu_slots(0, "r", 0, (7,))
    worker._gpu_required_blocks[0, "r", 0] = set()
    with pytest.raises(RuntimeError, match="safely reusable"):
        worker._reserve_gpu_slots(0, "r", 0, (8,))
    assert worker._gpu_slot_mappings == {(0, "r", 0, 7): 0}
    assert worker._reserve_gpu_slots(0, "r", 0, (9,), prediction=True) == ()
    with pytest.raises(ValueError, match="exceeds GPU slot budget"):
        worker._reserve_gpu_slots(0, "r", 0, (7, 8))


def test_bounded_all_blocks_rejects_before_reserving_any_slot(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker.cache_engine[0].num_gpu_blocks = 1024
    with pytest.raises(ValueError, match="actual KV working set exceeds"):
        worker._reserve_gpu_slots(0, "r", 0, tuple(range(2048)))
    assert not worker._gpu_slot_mappings
    assert not worker._gpu_required_blocks
    assert worker._host_stats["gpu_slot_capacity_rejections"] == 1


def test_bounded_slots_preserve_pending_until_ready(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker.cache_engine[0].num_gpu_blocks = 1
    worker._reserve_gpu_slots(0, "r", 0, (7,))
    worker._gpu_required_blocks[0, "r", 0] = set()
    key = (0, "r", 0, 7)
    event = Fence()
    worker._pending_h2d_transfers[key] = module._PendingHostCopy(0, event, True)
    worker._host_ready_blocks.add(key)
    worker._gpu_region_release_events[0, 0, 0] = ("r", event)
    with pytest.raises(RuntimeError, match="safely reusable"):
        worker._reserve_gpu_slots(0, "r", 0, (8,))
    assert worker._gpu_slot_mappings[key] == 0
    event.ready = True
    worker._reserve_gpu_slots(0, "r", 0, (8,))
    assert worker._gpu_slot_mappings == {(0, "r", 0, 8): 0}
    assert key not in worker._pending_h2d_transfers


def test_bounded_prediction_reserves_actual_working_set_headroom(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker.cache_engine[0].num_gpu_blocks = 4
    worker._gpu_working_set_bound = 3
    assert worker._reserve_gpu_slots(0, "r", 0, (5, 6), prediction=True) == (5,)
    worker._pending_h2d_transfers[0, "r", 0, 5] = module._PendingHostCopy(0, Fence(), True)
    assert worker._reserve_gpu_slots(0, "r", 0, (6,), prediction=True) == ()
    assert worker._reserve_gpu_slots(0, "r", 0, (7, 8, 9)) == (7, 8, 9)


def test_bounded_late_ssd_completion_does_not_publish_reassigned_slot(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker._gpu_slot_mappings[0, "r", 0, 8] = 0
    worker._gpu_pending_transfers[0, "old-read"] = ("r", (0, 1), ((5, 0),), (7,))
    worker._observe_gpu_transfer(0, AsyncKVTransferEvent(
        "old-read", AsyncKVTransferState.READY))
    assert not worker._gpu_resident_blocks
    assert not worker._gpu_destination_owners


def test_bounded_h2d_rejects_unreserved_target(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker._gpu_slot_mappings[0, "r", 0, 7] = 1
    with pytest.raises(RuntimeError, match="unreserved GPU slot"):
        worker.copy_host_blocks_to_gpu_async(0, "r", 0, ((7, 0),))
    assert not worker._pending_h2d_transfers


def test_bounded_owner_publication_rejects_reassigned_slot(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker._gpu_slot_mappings[0, "r", 0, 8] = 0
    with pytest.raises(RuntimeError, match="unreserved slot"):
        worker._mark_gpu_blocks_ready(0, "r", 0, (7,), ((7, 0),))
    assert not worker._gpu_resident_blocks
    assert not worker._gpu_destination_owners


def test_empty_transfer_rpc_cleans_finished_bounded_request(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    discarded = []
    monkeypatch.setattr(module, "get_sparse_kv_policy",
                        lambda: SimpleNamespace(discard=discarded.append))
    worker._discarded_host_transfers = set()
    worker._host_staging_owner = None
    worker._seq_group_metadata_cache = {"r": object()}
    worker._gpu_slot_mappings[0, "r", 0, 7] = 0
    worker._gpu_required_blocks[0, "r", 0] = {7}
    assert worker.submit_async_kv_transfers(0, (), ("r",)) == []
    assert not worker._gpu_slot_mappings
    assert not worker._gpu_required_blocks
    assert not worker._seq_group_metadata_cache
    assert discarded == ["r"]


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


def test_backing_feedback_is_consumed_locally_including_empty_layers(worker):
    updates = []
    worker._prefetch_runtime = SimpleNamespace(residency=SimpleNamespace(
        update_prediction=lambda *args: updates.append(args)))
    feedback = SimpleNamespace(num_prefix_blocks=4, block_indices_by_layer=((), (0, 2)))
    assert worker.update_sparse_restore_prediction("r", feedback)
    assert updates == [("r", 4, ((), (0, 2)))]


def test_layer_release_only_records_backup_fence(worker, monkeypatch):
    order = []
    event = Fence()
    event.record = lambda stream: order.append("release")
    monkeypatch.setattr(torch.cuda, "Event", lambda **kw: event)
    worker._prefetch_next_layer = lambda *args: pytest.fail(
        "layer release must not trigger per-layer prefetch")
    worker._back_up_layer = lambda ve, requests, layer, released: (
        order.append(("backup", layer, released)))
    worker._release_hierarchical_layer(0, ("r",), 3)
    assert order == ["release", ("backup", 3, event)]
    assert worker._host_stats["layer_release_event_record_count"] == 1


def test_prefetch_next_window_covers_four_layers_once(worker, monkeypatch):
    monkeypatch.setattr(module.envs,
                        "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", 4)
    calls = []
    worker._poll_host_transfers = lambda *args: calls.append("poll")
    worker._prefetch_next_layer = lambda ve, requests, layer, **kwargs: calls.append(layer)

    worker._prefetch_next_window(0, ("r",), 4)

    assert calls == ["poll", 4, 5, 6, 7]


def test_wait_triggers_next_window_only_at_window_start(worker, monkeypatch):
    monkeypatch.setattr(module.envs,
                        "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS", 4)
    worker._prefetch_runtime.wait_ready = lambda *args, **kwargs: None
    worker._prefetch_runtime.require_resident_layer = (
        lambda *args, **kwargs: None)
    worker._restore_live_backing = lambda *args, **kwargs: None
    worker._log_prefetch_runtime_traces = lambda: None
    windows = []
    worker._prefetch_next_window = lambda ve, requests, start: windows.append(start)

    for layer in (0, 1, 2, 3, 4, 5):
        worker._wait_for_hierarchical_layer(0, ("r",), layer)

    assert windows == [4, 8]


def test_retired_request_cannot_publish_host_backing(worker):
    key = (0, "r", 0, 7)
    worker._pending_gpu_to_host[key] = Fence(True)
    worker._backing_retired.add((0, "r"))
    worker._poll_gpu_to_host_backing(0)
    assert key not in worker._host_ready_blocks
    assert not worker._pending_gpu_to_host


@pytest.mark.parametrize("dense", [False, True])
def test_finish_invalidates_owners_but_keeps_pending_backup_fence(
        worker, request, dense):
    if dense:
        request.getfixturevalue("dense_writeback")
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


def test_bounded_finish_then_slot_reuse_preserves_old_dma_fence(worker, monkeypatch):
    monkeypatch.setattr(worker, "_uses_bounded_slots", lambda: True)
    worker.cache_engine[0].num_gpu_blocks = 1
    worker._discarded_host_transfers = set()
    worker._host_staging_owner = (0, "old")
    fence = Fence()
    key = (0, "old", 0, 7)
    worker._gpu_slot_mappings[key] = 0
    worker._gpu_region_release_events[0, 0, 0] = ("old", fence)
    worker._pending_gpu_to_host[key] = fence
    worker._forget_host_staging(0, ("old",))
    worker._forget_host_staging(0, ("old",))
    assert not worker._gpu_slot_mappings
    worker._reserve_gpu_slots(0, "new", 0, (9,))
    assert worker._gpu_slot_mappings[0, "new", 0, 9] == 0
    assert worker._take_gpu_region_release_event(0, 0, 0) is fence
    assert key in worker._pending_gpu_to_host


def test_abort_before_first_forward_has_no_cached_metadata(worker):
    worker._seq_group_metadata_cache = {}
    assert worker._get_cached_seq_group_metadata([], ["not-started"]) == []


@pytest.mark.parametrize("consume", [False, True])
def test_prefetch_finish_retains_dma_and_counts_consumption_once(worker, consume):
    worker._discarded_host_transfers = set()
    worker._host_staging_owner = (0, "r")
    event = Fence()
    key = (0, "r", 0, 0)
    worker.cache_engine[0].copy_host_blocks_to_gpu_async = (
        lambda *args: event)
    worker.copy_host_blocks_to_gpu_async(
        0, "r", 0, ((0, 3),), reason="prefetch")
    if consume:
        worker._consume_host_copies(0, "r", 0, (0,))
        worker._consume_host_copies(0, "r", 0, (0,))
    worker._forget_host_staging(0, ("r",))
    worker._forget_host_staging(0, ("r",))
    assert key in worker._pending_h2d_transfers
    assert not worker._gpu_resident_blocks
    assert not worker._gpu_destination_owners
    assert worker._host_stats["host_gpu_prefetch_hits"] == int(consume)
    assert worker._pending_h2d_transfers[key].prefetched == (not consume)
    event.ready = True
    worker._drain_host_staging_owner(0, "r", 1)
    assert not worker._pending_h2d_transfers
    assert worker._host_staging_owner is None
    assert not worker._backing_retired
    assert worker._host_stats["host_gpu_prefetch_blocks"] == 1
    assert worker._host_stats["host_gpu_prefetch_hits"] == int(consume)


def test_discarded_host_read_does_not_release_inflight_prefetch_source(worker):
    worker._host_staging_owner = (0, "r")
    worker._discarded_host_transfers = {(0, "host-read")}
    worker._host_async_kv_transfers[0, "host-read"] = (
        "host-read", "r", (1,), (0, 1))
    worker._pending_h2d_transfers[0, "r", 0, 0] = module._PendingHostCopy(
        3, Fence(), True)
    worker._backing_retired.add((0, "r"))
    worker.cache_engine[0].poll_async_kv_transfer = lambda request: (
        AsyncKVTransferEvent(request, AsyncKVTransferState.READY))
    worker._poll_host_transfers(0)
    assert worker._host_staging_owner == (0, "r")
    assert (0, "r") in worker._backing_retired
    assert not worker._host_ready_blocks


@pytest.fixture
def dense_writeback(worker, monkeypatch):
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_HOST_BACKING_ENABLE", False)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_CPU_STAGING_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE", False)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_CONSUMER_ENABLE", False)
    return worker


def test_dense_back_up_only_dirty_tail(dense_writeback):
    worker = dense_writeback
    worker._backing_forward[0, "r"] = ((3, 4, 5), 33, 34, False)
    copies = []
    ready = Fence()
    worker.cache_engine[0].copy_gpu_blocks_to_host_async = (
        lambda layer, mapping, fences: copies.append(mapping.tolist()) or ready)
    worker._back_up_layer(0, ("r",), 0, Fence())
    assert copies == [[[5, 2]]]
    assert worker._pending_gpu_to_host == {(0, "r", 0, 2): ready}
    assert worker._gpu_region_release_events[0, 0, 5][1] is ready
    assert worker._host_ready_blocks == set()


def test_dense_write_completion_does_not_index_absent_host_prefix(dense_writeback):
    worker = dense_writeback
    worker._pending_host_writes[0, "write"] = SimpleNamespace(
        sparse_page_index_key="prefix", consumer_num_blocks=8,
        logical_blocks=(LogicalBlockKey(0, 7),))
    worker._complete_host_write(0, worker.cache_engine[0],
        AsyncKVTransferEvent("write", AsyncKVTransferState.READY))
    assert not worker._pending_host_writes


def test_dense_window_release_preserves_backup_fence(dense_writeback):
    worker = dense_writeback
    ready = Fence()
    key = (0, "r", 0, 5)
    worker._gpu_destination_owners[0, 0, 5] = key
    worker._pending_gpu_to_host[key] = ready
    worker._gpu_region_release_events[0, 0, 5] = ("r", ready)
    worker._gpu_transfer_regions[0, "io"] = ("r", (0, 4), ((10, 5),))
    worker._record_gpu_region_release_event(0, ("r",), 3)
    assert worker._gpu_region_release_events[0, 0, 5] == ("r", ready)


def test_dense_layer_release_also_advances_window(dense_writeback):
    worker = dense_writeback
    released = []
    worker._back_up_layer = lambda *args: released.append("backup")
    worker._prefetch_runtime = SimpleNamespace(
        release_layer=lambda *args: released.append("release"))
    worker._log_prefetch_runtime_traces = lambda: None
    worker._release_hierarchical_layer(0, ("r",), 3)
    assert released == ["backup", "release"]


def test_dense_compute_waits_for_previous_ring_backup(dense_writeback, monkeypatch):
    worker = dense_writeback
    worker._backing_forward[0, "r"] = ((3, 4, 5), 33, 34, False)
    event = Fence()
    worker._gpu_region_release_events[0, 0, 5] = ("r", event)
    worker._prefetch_runtime.wait_ready = lambda *args, **kw: None
    worker._prefetch_runtime.require_resident_layer = lambda *args: None
    worker._log_prefetch_runtime_traces = lambda: None
    waited = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(
        wait_event=waited.append))
    worker._wait_for_hierarchical_layer(0, ("r",), 8)
    assert waited == [event]
    assert (0, 0, 5) not in worker._gpu_region_release_events


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
    worker._back_up_layer(0, ("r",), 0, Fence())
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
@pytest.mark.parametrize("is_prompt", [False, True])
def test_sparse_prefill_restores_only_partial_tail_before_query(
        worker, monkeypatch, computed, expected, is_prompt):
    monkeypatch.setattr(module.envs,
                        "VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE", True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_SELECTED_BLOCKS_ENABLE", True)
    worker._prefetch_runtime = SimpleNamespace(
        prediction_state=lambda *args: SimpleNamespace(num_prefix_blocks=2))
    worker._backing_forward[0, "r"] = (tuple(range(8)), computed,
                                       computed + (17 if is_prompt else 1), is_prompt)
    restored = []
    worker._restore_backing_blocks = lambda ve, req, layer, table, blocks, **kw: (
        restored.append(tuple(blocks)))
    worker._restore_live_backing(0, ("r",), 0)
    assert restored == [expected]


def test_backing_restore_reuses_owner_and_does_not_repeat_pending_copy(worker):
    copied = []
    worker.wait_for_host_blocks = lambda *args: True
    worker.cache_engine[0].copy_host_blocks_to_gpu_async = (
        lambda layer, mapping, fences: copied.append(mapping.tolist()) or Fence())
    table = (3, 4, 5)
    worker._restore_backing_blocks(0, "r", 0, table, (0, 2))
    worker._restore_backing_blocks(0, "r", 0, table, (0, 2))
    assert copied == [[[0, 3], [2, 5]]]
    worker._pending_h2d_transfers.pop((0, "r", 0, 2))
    worker._mark_gpu_blocks_ready(0, "r", 8, (2,), ((2, 5),))
    worker._restore_backing_blocks(0, "r", 0, table, (0, 2))
    assert copied[-1] == [[2, 5]]


def test_correction_restores_only_selected_history_not_whole_suffix(worker, monkeypatch):
    from vllm.core.custom_schedulers.hierarchical_io.sparse_policy import (
        SparseKVActualSelection)
    from vllm.attention.ops.sparse_kv import SparseKVDeviceSelection

    for name in ("VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE",
                 "VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE",
                 "VLLM_GRANULEKV_SPARSE_SELECTED_BLOCKS_ENABLE"):
        monkeypatch.setattr(module.envs, name, True)
    monkeypatch.setattr(module.envs, "VLLM_GRANULEKV_SPARSE_CPU_STAGING_ENABLE", False)
    monkeypatch.setattr(module, "get_active_layer_sequence_lengths", lambda: {"r": 100})
    state = SimpleNamespace(seq_group_id="r", num_prefix_blocks=2,
                            resident_prefix_blocks=(0, 1),
                            predicted_prefix_blocks=(0, 1),
                            correction_mapping_by_logical=((-1, 8), (-1, 9)))
    worker._prefetch_runtime = SimpleNamespace(prediction_state=lambda *args: state)
    table = tuple(range(8, 15))
    worker._backing_forward[0, "r"] = (table, 81, 100, True)
    selected = SparseKVActualSelection(
        history_blocks=(0, 3, 5),
        attention=SparseKVDeviceSelection(torch.tensor([0, 3, 5, 6]), 4))
    selections, restores = [], []
    monkeypatch.setattr(module, "select_actual_sparse_blocks_device",
                        lambda *args: selections.append(args) or selected)
    worker.gpu_blocks_ready = lambda *args: (0,)
    worker._restore_backing_blocks = lambda *args: restores.append(args)
    worker._prefetch_next_layer = lambda *args: None
    result = worker._correct_hierarchical_layer(0, ("r",), 0, torch.ones(19, 2, 2))
    assert len(selections) == 1
    assert selections[0][3:7] == (2, 81, 100, 16)
    assert restores == [(0, "r", 0, table, (3, 5))]
    assert result.block_indices == (0, 3, 5, 6)
    assert result.device_selection is selected.attention


def test_independent_layers_keep_same_destination_without_reloading(worker):
    worker.cache_engine[0].granulekv_connector.layout.num_gpu_regions = 28
    copied = []
    worker.wait_for_host_blocks = lambda *args: True
    worker.cache_engine[0].copy_host_blocks_to_gpu_async = (
        lambda layer, mapping, fences: copied.append((layer, mapping.tolist())) or Fence())
    for layer in (0, 8, 16, 24):
        worker._restore_backing_blocks(0, "r", layer, (3, 4, 5), (0, 2))
    assert len(copied) == 4
    for layer in (0, 8, 16, 24):
        worker._restore_backing_blocks(0, "r", layer, (3, 4, 5), (0, 2))
        assert worker.gpu_blocks_ready(0, "r", layer, (0, 2)) == (0, 2)
    assert len(copied) == 4


def test_h2d_submit_does_not_wait_and_consumer_waits_only_selected(worker, monkeypatch):
    worker.cache_engine[0].granulekv_connector.layout.num_gpu_regions = 28
    worker._backing_forward[0, "r"] = ((3, 4, 5), 48, 49, False)
    ready = Fence()
    submitted, waits = [], []
    worker.cache_engine[0].copy_host_blocks_to_gpu_async = (
        lambda layer, mapping, fences: submitted.append((layer, mapping.tolist(), fences)) or ready)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(
        wait_event=waits.append))
    release = Fence()
    worker._gpu_region_release_events[0, 1, 3] = ("r", release)
    worker.copy_host_blocks_to_gpu_async(
        0, "r", 1, ((0, 3), (2, 5)), reason="prefetch")
    assert waits == []
    assert submitted == [(1, [[0, 3], [2, 5]], [release])]
    assert worker.gpu_blocks_ready(0, "r", 1, (0, 2)) == ()
    worker._consume_host_copies(0, "r", 1, (2,))
    assert waits == [ready]
    assert worker.gpu_blocks_ready(0, "r", 1, (0, 2)) == (2,)
    worker._consume_host_copies(0, "r", 1, (2,))
    assert waits == [ready]
    assert worker._host_stats["host_gpu_prefetch_hits"] == 1


def test_prefetch_only_ready_missing_complete_prefix_and_pending_dedup(worker):
    worker.cache_engine[0].granulekv_connector.layout.num_gpu_regions = 28
    worker._backing_forward[0, "r"] = (tuple(range(10, 16)), 65, 66, False)
    state = SimpleNamespace(seq_group_id="r", predicted_prefix_blocks=(0, 1, 2, 3, 4))
    worker._prefetch_runtime = SimpleNamespace(prediction_state=lambda *args: state)
    worker._poll_host_transfers = lambda *args: None
    worker._host_ready_blocks.update((0, "r", 1, block) for block in (0, 1, 2, 4))
    worker._mark_gpu_blocks_ready(0, "r", 1, (0,), ((0, 10),))
    worker._gpu_pending_transfers[0, "ssd"] = ("r", (1, 2), ((1, 11),), (1,))
    submitted = []
    worker.cache_engine[0].copy_host_blocks_to_gpu_async = (
        lambda layer, mapping, fences: submitted.append(mapping.tolist()) or Fence())
    worker._prefetch_next_layer(0, ("r",), 1)
    worker._prefetch_next_layer(0, ("r",), 1)
    assert submitted == [[[2, 12]]]
    worker._restore_backing_blocks(0, "r", 1, tuple(range(10, 16)), (0, 2))
    assert submitted == [[[2, 12]]]


def test_h2d_rejects_current_chunk_and_wrong_destination(worker):
    worker._backing_forward[0, "r"] = ((3, 4, 5), 32, 48, True)
    for mapping in (((2, 5),), ((0, 5),), ((-1, 5),)):
        with pytest.raises(RuntimeError, match="outside historical"):
            worker.copy_host_blocks_to_gpu_async(0, "r", 0, mapping)


def test_pending_copy_with_wrong_physical_owner_is_rejected(worker):
    worker._backing_forward[0, "r"] = ((3,), 16, 17, False)
    worker.cache_engine[0].copy_host_blocks_to_gpu_async = lambda *args: Fence()
    worker.copy_host_blocks_to_gpu_async(0, "r", 0, ((0, 3),))
    worker._gpu_destination_owners[0, 0, 3] = (0, "other", 0, 0)
    with pytest.raises(RuntimeError, match="physical owner"):
        worker._consume_host_copies(0, "r", 0, (0,))


def test_host_write_yields_to_pending_read(worker):
    worker._poll_host_transfers = lambda *args: None
    worker._host_ready_blocks.add((0, "r", 0, 0))
    worker._gpu_pending_transfers[0, "read"] = ("r", (1, 2), ((1, 1),), (1,))
    submitted = []
    engine = worker.cache_engine[0]
    engine.submit_async_kv_transfer = lambda *a, **kw: (
        submitted.append(a) or AsyncKVTransferEvent("write", AsyncKVTransferState.PENDING))
    request = AsyncKVTransferRequest(
        request_id="write", seq_group_id="r", reservation_id="res",
        operation=AsyncKVTransferOperation.WRITE,
        priority=AsyncKVTransferPriority.DEFERRED_WRITE,
        block_mapping=((0, 4),), logical_blocks=(LogicalBlockKey(0, 0),),
        layer_range=(0, 1), destination="host")
    mapping = torch.tensor(request.block_mapping)
    worker._submit_gpu_transfer(0, engine, request, mapping)
    assert not submitted
    assert (0, "write") in worker._deferred_gpu_transfers
    worker._gpu_pending_transfers.clear()
    worker._poll_gpu_transfer(0, engine, "write")
    assert len(submitted) == 1


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
        worker._back_up_layer(0, ("r",), 0, Fence())
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
@pytest.mark.parametrize("dense", [False, True])
def test_native_read_is_deferred_until_backup_source_is_released(
        worker, layer_range, request, dense):
    if dense:
        request.getfixturevalue("dense_writeback")
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


@pytest.mark.parametrize("dense", [False, True])
def test_host_write_waits_for_d2h_then_uses_existing_native_request(
        worker, request, dense):
    if dense:
        request.getfixturevalue("dense_writeback")
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
