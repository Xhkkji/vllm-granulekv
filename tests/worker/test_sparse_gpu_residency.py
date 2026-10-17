# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from types import SimpleNamespace

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferOperation, AsyncKVTransferPriority,
    AsyncKVTransferRequest, AsyncKVTransferState)
from vllm.core.block_reservation import LogicalBlockKey
from vllm.worker.worker import Worker
import vllm.worker.worker as worker_module
import torch


class _FakeMapping:

    shape = (1, 2)

    def numel(self):
        return 2


def _worker():
    worker = object.__new__(Worker)
    worker._gpu_resident_blocks = {}
    worker._gpu_destination_owners = {}
    worker._gpu_pending_transfers = {}
    worker._gpu_region_release_events = {}
    worker._gpu_transfer_regions = {}
    worker._ssd_read_blocks = set()
    worker._host_stats = defaultdict(int)
    worker.cache_engine = [SimpleNamespace(granulekv_connector=SimpleNamespace(
        layout=SimpleNamespace(num_gpu_regions=28)))]
    return worker


def _request(request_id="read-1", logical_block=7, destination_block=3):
    return AsyncKVTransferRequest(
        request_id=request_id,
        seq_group_id="request-1",
        reservation_id="reservation-1",
        operation=AsyncKVTransferOperation.READ,
        block_mapping=((11, destination_block),),
        logical_blocks=(LogicalBlockKey(0, logical_block),),
        priority=AsyncKVTransferPriority.CRITICAL_READ,
        layer_range=(2, 3),
    )


def test_gpu_residency_hit_and_destination_overwrite(monkeypatch):
    monkeypatch.setattr(
        worker_module.envs,
        "VLLM_GRANULEKV_SPARSE_GPU_RESIDENCY_REUSE_ENABLE",
        True,
    )
    worker = _worker()
    worker._register_gpu_transfer(0, _request(), _FakeMapping())
    worker._observe_gpu_transfer(
        0, AsyncKVTransferEvent("read-1", AsyncKVTransferState.READY))

    assert worker.gpu_blocks_ready(0, "request-1", 2, (7, )) == (7, )

    worker._invalidate_gpu_destination(0, 2, 3)
    assert worker.gpu_blocks_ready(0, "request-1", 2, (7, )) == ()
    assert worker._host_stats["gpu_resident_released_blocks"] == 1


def test_host_to_gpu_registration_replaces_old_owner(monkeypatch):
    monkeypatch.setattr(
        worker_module.envs,
        "VLLM_GRANULEKV_SPARSE_GPU_RESIDENCY_REUSE_ENABLE",
        True,
    )
    worker = _worker()
    worker._register_gpu_transfer(0, _request(), _FakeMapping())
    worker._observe_gpu_transfer(
        0, AsyncKVTransferEvent("read-1", AsyncKVTransferState.READY))
    worker._mark_gpu_blocks_ready(0, "request-1", 2, (9, ), ((0, 3), ))

    assert worker.gpu_blocks_ready(0, "request-1", 2, (7, 9)) == (9, )


def test_aliased_layer_write_invalidates_old_physical_slot_owner(monkeypatch):
    monkeypatch.setattr(worker_module.envs,
                        "VLLM_GRANULEKV_EVENT_FENCED_LAYER_REUSE_ENABLE", False)
    worker = _worker()
    worker.cache_engine[0].granulekv_connector.layout.num_gpu_regions = 8
    worker._mark_gpu_blocks_ready(0, "request", 0, (7,), ((7, 3),))
    worker._mark_gpu_blocks_ready(0, "request", 24, (7,), ((7, 3),))
    assert worker.gpu_blocks_ready(0, "request", 0, (7,)) == ()
    assert worker.gpu_blocks_ready(0, "request", 24, (7,)) == (7,)
    assert len(worker._gpu_destination_owners) == 1


def test_aliased_layer_takes_same_release_fence_without_cpu_wait():
    worker = _worker()
    worker.cache_engine[0].granulekv_connector.layout.num_gpu_regions = 8
    event = object()
    worker._gpu_region_release_events[worker._gpu_slot_key(0, 0, 3)] = (
        "read", event)
    assert worker._take_gpu_region_release_event(0, 24, 3) is event
    assert not worker._gpu_region_release_events


def test_finish_cleans_owner_and_release_fence_without_waiting():
    worker = _worker()
    worker._host_async_kv_transfers = {}
    worker._host_ready_blocks = set()
    worker._host_failed_blocks = set()
    worker._host_staged_blocks = set()
    worker._pending_h2d_transfers = {}
    worker._host_staging_owner = None
    slot = worker._gpu_slot_key(0, 0, 3)
    owner = (0, "request", 0, 7)
    worker._gpu_resident_blocks[owner] = 3
    worker._gpu_destination_owners[slot] = owner
    worker._gpu_transfer_regions[0, "read"] = ("request", (0, 1), ((7, 3),))
    worker._gpu_region_release_events[slot] = ("read", object())
    worker._forget_host_staging(0, ("request",))
    assert not worker._gpu_resident_blocks
    assert not worker._gpu_destination_owners
    assert not worker._gpu_region_release_events


def test_validation_logits_hook_is_removed_and_off_adds_no_hook(monkeypatch):
    worker = _worker()
    seen = []
    policy = SimpleNamespace(
        configure_validation=lambda *args: None,
        validation_report=lambda *args, **kwargs: {},
        record_validation_logits=lambda request, output: seen.append(request))
    monkeypatch.setattr(worker_module, "get_sparse_kv_policy", lambda: policy)
    processor = torch.nn.Identity()
    worker.model_runner = SimpleNamespace(
        get_model=lambda: SimpleNamespace(logits_processor=processor))
    worker.configure_sparse_validation("request", "record")
    processor(torch.ones(1))
    assert seen == ["request"]
    worker.get_sparse_validation_report("request", reset=True)
    assert not processor._forward_hooks
    worker.configure_sparse_validation("request", "off")
    processor(torch.ones(1))
    assert seen == ["request"]
    assert not processor._forward_hooks


def test_gpu_residency_disabled_keeps_index_empty(monkeypatch):
    monkeypatch.setattr(
        worker_module.envs,
        "VLLM_GRANULEKV_SPARSE_GPU_RESIDENCY_REUSE_ENABLE",
        False,
    )
    worker = _worker()
    worker._register_gpu_transfer(0, _request(), _FakeMapping())
    worker._observe_gpu_transfer(
        0, AsyncKVTransferEvent("read-1", AsyncKVTransferState.READY))

    assert worker.gpu_blocks_ready(0, "request-1", 2, (7, )) == ()


def test_event_fenced_release_waits_before_destination_reuse(monkeypatch):
    class _Event:

        def __init__(self, **_kwargs):
            self.ready = False
            self.recorded = False
            self.synchronized = False

        def record(self, _stream):
            self.recorded = True

        def query(self):
            return self.ready

        def synchronize(self):
            self.synchronized = True
            self.ready = True

    event = _Event()
    monkeypatch.setattr(
        worker_module.envs,
        "VLLM_GRANULEKV_SPARSE_GPU_RESIDENCY_REUSE_ENABLE",
        True,
    )
    monkeypatch.setattr(
        worker_module.envs,
        "VLLM_GRANULEKV_EVENT_FENCED_LAYER_REUSE_ENABLE",
        True,
    )
    monkeypatch.setattr(
        worker_module.envs,
        "VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE",
        True,
    )
    monkeypatch.setattr(worker_module.torch.cuda, "Event", lambda **_: event)
    monkeypatch.setattr(worker_module.torch.cuda, "current_stream",
                        lambda: object())
    worker = _worker()
    worker._register_gpu_transfer(0, _request(), _FakeMapping())
    worker._observe_gpu_transfer(
        0, AsyncKVTransferEvent("read-1", AsyncKVTransferState.READY))

    worker._record_gpu_region_release_event(0, ("request-1", ), 2)
    assert worker._host_stats["layer_release_event_record_count"] == 1
    assert event.recorded

    worker._invalidate_gpu_destination(0, 2, 3)
    assert event.synchronized
    assert worker._host_stats["layer_release_event_wait_count"] == 1


def test_event_fenced_non_window_layer_does_not_record(monkeypatch):
    monkeypatch.setattr(
        worker_module.envs,
        "VLLM_GRANULEKV_EVENT_FENCED_LAYER_REUSE_ENABLE",
        True,
    )
    monkeypatch.setattr(
        worker_module.envs,
        "VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE",
        True,
    )
    worker = _worker()
    worker._register_gpu_transfer(0, _request(), _FakeMapping())
    worker._record_gpu_region_release_event(0, ("request-1", ), 1)
    assert worker._host_stats.get("layer_release_event_record_count", 0) == 0


def test_async_copy_consumes_release_event_without_host_wait():
    class _Event:

        def __init__(self):
            self.synchronized = False

        def synchronize(self):
            self.synchronized = True

    event = _Event()
    worker = _worker()
    worker._gpu_region_release_events[(0, 2, 3)] = ("request-1", event)

    assert worker._take_gpu_region_release_event(0, 2, 3) is event
    assert (0, 2, 3) not in worker._gpu_region_release_events
    assert not event.synchronized
