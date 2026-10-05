# SPDX-License-Identifier: Apache-2.0

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferOperation, AsyncKVTransferPriority,
    AsyncKVTransferRequest, AsyncKVTransferState)
from vllm.core.block_reservation import LogicalBlockKey
from vllm.worker.worker import Worker
import vllm.worker.worker as worker_module


class _FakeMapping:

    shape = (1, 2)

    def numel(self):
        return 2


def _worker():
    worker = object.__new__(Worker)
    worker._gpu_resident_blocks = {}
    worker._gpu_destination_owners = {}
    worker._gpu_pending_transfers = {}
    worker._ssd_read_blocks = set()
    worker._host_stats = {
        "gpu_resident_released_blocks": 0,
        "duplicate_ssd_read_blocks": 0,
    }
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
