from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from vllm.granulekv.connector import (
    GranuleKVConnector,
    GranuleKVRequestSpec,
    GranuleKVTransferState,
)


class _FakeLayout:
    num_layers = 4
    num_gpu_regions = 4


class _FakeClient:

    def __init__(self):
        self.timeout_seconds = 1.0
        self.submitted = []
        self.completed = []
        self.status_state = GranuleKVTransferState.READY

    def submit(self, payload, *, operation, **kwargs):
        handle = object()
        self.submitted.append((payload, operation, kwargs, handle))
        return handle

    def status(self, handle):
        return SimpleNamespace(
            state=SimpleNamespace(value=self.status_state.value),
            io_elapsed_ns=17,
            error_code=0,
        )

    def complete(self, handle):
        self.completed.append(handle)
        return 17

    def close(self):
        pass


def _connector():
    connector = GranuleKVConnector.__new__(GranuleKVConnector)
    connector.layout = _FakeLayout()
    connector.client = _FakeClient()
    connector._pending_transfers = {}
    return connector


def test_request_spec_is_immutable_and_serializes_layer_window():
    spec = GranuleKVRequestSpec(
        operation="read",
        gpu_block_ids=(4, 5),
        storage_block_ids=(14, 15),
        layer_range=(2, 4),
        gpu_region_start=0,
    )
    assert spec.to_payload() == {
        "gpu_block_ids": [4, 5],
        "storage_block_ids": [14, 15],
        "layer_start": 2,
        "layer_end": 4,
        "gpu_region_start": 0,
    }
    with pytest.raises(FrozenInstanceError):
        spec.operation = "write"


def test_swap_in_delegates_to_canonical_request_lifecycle():
    connector = _connector()

    connector.swap_in(torch.tensor([[14, 4], [15, 5]], dtype=torch.int64))

    assert len(connector.client.submitted) == 1
    payload, operation, kwargs, handle = connector.client.submitted[0]
    assert payload == {"gpu_block_ids": [4, 5], "storage_block_ids": [14, 15]}
    assert operation == "read"
    assert kwargs == {}
    assert connector.client.completed == [handle]
    assert not connector._pending_transfers


def test_active_layer_window_uses_normal_request_without_staging():
    connector = _connector()
    mapping = torch.tensor([[20, 2]], dtype=torch.int64)

    connector.submit_request(
        "window-0", mapping, operation="read", layer_range=(1, 2))
    connector.complete_request("window-0")

    payload, operation, kwargs, _ = connector.client.submitted[0]
    assert payload["layer_start"] == 1
    assert payload["layer_end"] == 2
    assert operation == "read"
    assert kwargs == {}


def test_layer_window_sets_working_set_region_without_changing_descriptor_protocol():
    connector = _connector()
    connector.layout.num_gpu_regions = 2
    spec = connector._request_spec(
        torch.tensor([[30, 1]], dtype=torch.int64),
        operation="read",
        layer_range=(3, 4),
    )

    assert spec.layer_range == (3, 4)
    assert spec.gpu_region_start == 1
    assert spec.to_payload()["layer_start"] == 3
    assert spec.to_payload()["layer_end"] == 4


def test_close_rejects_active_requests_before_closing_client():
    connector = _connector()
    connector.submit_request(
        "active", torch.tensor([[1, 2]], dtype=torch.int64), operation="read")

    with pytest.raises(RuntimeError, match="active requests"):
        connector.close()

    assert connector.client.completed == []
