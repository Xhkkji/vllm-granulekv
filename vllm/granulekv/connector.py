# SPDX-License-Identifier: Apache-2.0
"""vLLM control-plane connector for the independent GranuleKV I/O path."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
import time
from typing import Any, Optional, Sequence, Tuple

import torch

import vllm.envs as envs
from vllm.granulekv.kv_layout import GranuleKVLayout
from vllm.logger import init_logger


logger = init_logger(__name__)


@dataclass(frozen=True)
class GranuleKVRequestSpec:
    """CPU-owned logical request shared by ordinary and staged transfers.

    The spec contains logical block addresses and the optional layer window.
    Descriptor construction remains an Executor/native responsibility.
    """

    operation: str
    gpu_block_ids: tuple[int, ...]
    storage_block_ids: tuple[int, ...]
    layer_range: Optional[Tuple[int, int]] = None
    gpu_region_start: Optional[int] = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "gpu_block_ids": list(self.gpu_block_ids),
            "storage_block_ids": list(self.storage_block_ids),
        }
        if self.layer_range is not None:
            payload["layer_start"], payload["layer_end"] = self.layer_range
        if self.gpu_region_start is not None:
            payload["gpu_region_start"] = self.gpu_region_start
        return payload


@dataclass(frozen=True)
class _PendingTransfer:
    handle: object
    spec: GranuleKVRequestSpec
    submitted_at_ns: int
    prefetch_plan_id: Optional[str] = None


class GranuleKVTransferState(str, Enum):
    """Stable vLLM-side state for both ordinary and prefetched transfers."""

    SUBMITTED = "submitted"
    IN_FLIGHT = "in_flight"
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True)
class GranuleKVTransferStatus:
    """Side-effect-free status returned by the canonical transfer API."""

    request_id: str
    state: GranuleKVTransferState
    operation: str
    prefetch_plan_id: Optional[str] = None
    io_elapsed_ns: int = 0
    error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.state is GranuleKVTransferState.READY


class GranuleKVConnector:
    """Translate vLLM block requests into GranuleKV control-plane requests."""

    def __init__(
        self,
        *,
        allocation_shape: Sequence[int],
        stride_order: Sequence[int],
        dtype: torch.dtype,
        device_index: int,
        num_layers: int,
        num_gpu_regions: int,
        num_gpu_blocks: int,
        num_storage_blocks: int,
    ) -> None:
        if not envs.VLLM_GRANULEKV_IOSTACK_ROOT:
            raise ValueError("VLLM_GRANULEKV_IOSTACK_ROOT is required")
        if not envs.VLLM_GRANULEKV_CONTROL_DIR:
            raise ValueError("VLLM_GRANULEKV_CONTROL_DIR is required")

        self.iostack_root = Path(envs.VLLM_GRANULEKV_IOSTACK_ROOT).resolve()
        gids_module = self.iostack_root / "gids_module"
        if str(gids_module) not in sys.path:
            sys.path.insert(0, str(gids_module))
        client_module = importlib.import_module("granulekv.client")
        self.layout = GranuleKVLayout.from_allocation(
            allocation_shape=allocation_shape,
            stride_order=stride_order,
            dtype=dtype,
            num_layers=num_layers,
            num_gpu_regions=num_gpu_regions,
            num_gpu_blocks=num_gpu_blocks,
            num_storage_blocks=num_storage_blocks,
            device_index=device_index,
        )
        self.bridge = self._load_torch_bridge()
        self.client = client_module.GranuleKVClient(
            control_dir=Path(envs.VLLM_GRANULEKV_CONTROL_DIR).resolve(),
            cuda_library=self._resolve_cuda_library(),
            device_index=device_index,
            timeout_seconds=float(envs.VLLM_GRANULEKV_TIMEOUT_SECONDS),
        )
        self.descriptor_pool_capacity = 0
        self.max_blocks_per_pool = 0
        self.max_in_flight = 0
        self.gpu_cache: list[torch.Tensor] = []
        self._pending_transfers: dict[str, _PendingTransfer] = {}
        self._prefetch_templates: dict[
            str, tuple[str, GranuleKVRequestSpec]] = {}
        try:
            self.gpu_cache = self._allocate_and_wrap()
        except Exception:
            self.client.publish_error()
            self.client.close()
            raise

    def _resolve_cuda_library(self) -> Path:
        if envs.VLLM_GRANULEKV_CUDA_IPC_LIBRARY:
            return Path(envs.VLLM_GRANULEKV_CUDA_IPC_LIBRARY).resolve()
        return (self.iostack_root / "gids_module" / "build"
                / "libgranulekv_cuda_ipc.so")

    def _load_torch_bridge(self):
        if envs.VLLM_GRANULEKV_TORCH_BRIDGE_DIR:
            bridge_dir = Path(envs.VLLM_GRANULEKV_TORCH_BRIDGE_DIR).resolve()
        else:
            bridge_dir = Path(__file__).resolve().parent / "build" / "torch_bridge"
        if str(bridge_dir) not in sys.path:
            sys.path.insert(0, str(bridge_dir))
        return importlib.import_module("granulekv_torch_bridge")

    def _allocate_and_wrap(self) -> list[torch.Tensor]:
        result = self.client.allocate(
            self.layout.allocation_payload(client_pid=os.getpid()))
        self.layout.validate_manifest(result.manifest)
        self.descriptor_pool_capacity = int(
            result.manifest["descriptor_pool_capacity"])
        self.max_blocks_per_pool = int(result.manifest["max_blocks_per_pool"])
        self.max_in_flight = int(result.manifest["request_slot_count"])
        if self.max_in_flight != envs.VLLM_GRANULEKV_MAX_IN_FLIGHT:
            raise RuntimeError(
                "GranuleKV request-slot mismatch: daemon="
                f"{self.max_in_flight}, vLLM={envs.VLLM_GRANULEKV_MAX_IN_FLIGHT}")
        physical_regions: list[torch.Tensor] = []
        for region in result.regions:
            tensor = self.bridge.tensor_from_cuda_ptr(
                region.region_ptr,
                list(self.layout.tensor_shape),
                list(self.layout.tensor_strides),
                self.layout.dtype_name,
                self.layout.device_index,
            )
            if (tensor.data_ptr() != region.region_ptr
                    or tuple(tensor.shape) != self.layout.tensor_shape
                    or tuple(tensor.stride()) != self.layout.tensor_strides
                    or region.region_bytes != self.layout.region_bytes):
                raise RuntimeError("imported GranuleKV tensor metadata mismatch")
            physical_regions.append(tensor)
        for tensor in physical_regions:
            tensor.zero_()
        torch.cuda.synchronize(self.layout.device_index)
        tensors = [
            physical_regions[layer % self.layout.num_gpu_regions]
            for layer in range(self.layout.num_layers)
        ]
        logger.info(
            "[GRANULEKV] imported KV cache layers=%d gpu_regions=%d "
            "gpu_blocks=%d storage_blocks=%d fragment_bytes=%d "
            "pool_capacity=%d max_blocks_per_pool=%d max_in_flight=%d",
            self.layout.num_layers, self.layout.num_gpu_regions,
            self.layout.num_gpu_blocks, self.layout.num_storage_blocks,
            self.layout.fragment_bytes, self.descriptor_pool_capacity,
            self.max_blocks_per_pool, self.max_in_flight)
        return tensors

    def start(self) -> None:
        self.client.start()
        logger.info("[GRANULEKV] resident async service enabled after model warmup")

    def close(self) -> None:
        """Close only after all logical requests and plans are retired."""
        if self._pending_transfers:
            raise RuntimeError(
                "cannot close GranuleKV connector with active requests")
        if self._prefetch_templates:
            raise RuntimeError(
                "cannot close GranuleKV connector with staged plans")
        self.client.close()

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        self._run_sync_request(src_to_dst, operation="write")

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        self._run_sync_request(src_to_dst, operation="read")

    def submit_request(
        self,
        request_id: str,
        src_to_dst: torch.Tensor,
        *,
        operation: str,
        layer_range: Optional[Tuple[int, int]] = None,
        prefetch_plan_id: Optional[str] = None,
    ) -> GranuleKVTransferStatus:
        """Submit one ordinary or legacy prefetched transfer.

        This is the canonical asynchronous entry point. The current layerwise
        path submits active windows without ``prefetch_plan_id``; the optional
        staged branch remains only for compatibility with older callers.
        """
        if not request_id:
            raise ValueError("request_id must not be empty")
        if operation not in ("read", "write"):
            raise ValueError(f"unsupported GranuleKV operation: {operation}")
        layer_range = self._validate_layer_range(layer_range)
        if prefetch_plan_id is None:
            if src_to_dst.numel() == 0:
                return GranuleKVTransferStatus(
                    request_id, GranuleKVTransferState.READY, operation)
            spec = self._request_spec(
                src_to_dst, operation=operation, layer_range=layer_range)
        else:
            template = self._prefetch_templates.get(request_id)
            if template is None or template[0] != prefetch_plan_id:
                raise RuntimeError(
                    f"unknown staged prefetch unit: {request_id}")
            spec = template[1]
            if not self._mapping_matches_spec(src_to_dst, spec):
                raise RuntimeError(
                    f"staged prefetch mapping changed after staging: "
                    f"{request_id}")
            if spec.operation != operation or spec.layer_range != layer_range:
                raise RuntimeError(
                    f"staged prefetch unit changed after staging: {request_id}")
            if not spec.gpu_block_ids:
                self.client.cancel_staged_units(
                    prefetch_plan_id, (request_id,))
                del self._prefetch_templates[request_id]
                self._maybe_release_prefetch_plan(prefetch_plan_id)
                return GranuleKVTransferStatus(
                    request_id, GranuleKVTransferState.READY, operation,
                    prefetch_plan_id=prefetch_plan_id)

        pending = self._pending_transfers.get(request_id)
        if pending is not None:
            if (spec != pending.spec
                    or prefetch_plan_id != pending.prefetch_plan_id):
                raise RuntimeError("GranuleKV transfer changed while in flight")
            return self.query_request(request_id)

        payload = spec.to_payload()
        try:
            if prefetch_plan_id is None:
                handle = self.client.submit(payload, operation=operation)
            else:
                handle = self.client.submit(
                    payload,
                    operation=operation,
                    prefetch_plan_id=prefetch_plan_id,
                    prefetch_unit_id=request_id,
                )
        except Exception:
            if prefetch_plan_id is not None:
                self.client.cancel_staged_units(
                    prefetch_plan_id, (request_id,))
                self._prefetch_templates.pop(request_id, None)
                self._maybe_release_prefetch_plan(prefetch_plan_id)
            raise
        self._pending_transfers[request_id] = _PendingTransfer(
            handle=handle, spec=spec, submitted_at_ns=time.monotonic_ns(),
            prefetch_plan_id=prefetch_plan_id)
        return GranuleKVTransferStatus(
            request_id, GranuleKVTransferState.SUBMITTED, operation,
            prefetch_plan_id=prefetch_plan_id)

    def query_request(self, request_id: str) -> GranuleKVTransferStatus:
        """Observe one request without completing or releasing its slot."""
        pending = self._pending_transfers.get(request_id)
        if pending is None:
            raise RuntimeError(
                f"cannot query GranuleKV transfer without a pending request: {request_id}")
        client_status = self.client.status(pending.handle)
        state = GranuleKVTransferState(client_status.state.value)
        error = (None if state is not GranuleKVTransferState.ERROR else
                 f"GranuleKV request failed: error={client_status.error_code}")
        return GranuleKVTransferStatus(
            request_id, state, pending.spec.operation,
            prefetch_plan_id=pending.prefetch_plan_id,
            io_elapsed_ns=client_status.io_elapsed_ns,
            error=error)

    def complete_request(self, request_id: str) -> GranuleKVTransferStatus:
        """Release a READY ordinary or prefetched request."""
        pending = self._pending_transfers.get(request_id)
        if pending is None:
            raise RuntimeError(f"unknown GranuleKV transfer: {request_id}")
        status = self.query_request(request_id)
        if status.state is GranuleKVTransferState.ERROR:
            raise RuntimeError(status.error or "GranuleKV request failed")
        if not status.ready:
            raise RuntimeError("GranuleKV transfer is not complete")
        self.client.complete(pending.handle)
        del self._pending_transfers[request_id]
        if pending.prefetch_plan_id is not None:
            self._prefetch_templates.pop(request_id, None)
            self._maybe_release_prefetch_plan(pending.prefetch_plan_id)
        return status

    def cancel_request(self, request_id: str) -> None:
        """Forget a request after failure; does not fake-cancel device I/O."""
        pending = self._pending_transfers.pop(request_id, None)
        if pending is None:
            raise RuntimeError(f"unknown GranuleKV transfer: {request_id}")
        self.client.cancel(pending.handle)
        if pending.prefetch_plan_id is not None:
            self._prefetch_templates.pop(request_id, None)
            self._maybe_release_prefetch_plan(pending.prefetch_plan_id)

    def stage_plan(
        self,
        plan_id: str,
        units: Sequence[tuple[str, torch.Tensor, str,
                              Optional[Tuple[int, int]]]],
    ) -> None:
        """Register a legacy staged template without starting I/O."""
        staged: dict[str, tuple[dict[str, Any], str]] = {}
        for request_id, mapping_tensor, operation, layer_range in units:
            if operation not in ("read", "write"):
                raise ValueError(f"unsupported GranuleKV operation: {operation}")
            layer_range = self._validate_layer_range(layer_range)
            spec = self._request_spec(
                mapping_tensor, operation=operation, layer_range=layer_range)
            previous = self._prefetch_templates.get(request_id)
            if previous is not None:
                previous_plan, previous_spec = previous
                if previous_plan != plan_id or previous_spec != spec:
                    raise RuntimeError(
                        f"staged prefetch unit changed after staging: {request_id}")
                continue
            staged[request_id] = (spec.to_payload(), operation)
            self._prefetch_templates[request_id] = (plan_id, spec)
        if not staged:
            return
        try:
            self.client.stage_plan(plan_id, staged)
        except Exception:
            for request_id in staged:
                self._prefetch_templates.pop(request_id, None)
            raise

    def cancel_staged_units(self, scheduler_request_ids: Sequence[str]) -> None:
        """Cancel templates created through the legacy staged API."""
        by_plan: dict[str, list[str]] = {}
        for request_id in scheduler_request_ids:
            template = self._prefetch_templates.get(request_id)
            if template is None:
                raise RuntimeError(f"unknown staged prefetch unit: {request_id}")
            by_plan.setdefault(template[0], []).append(request_id)
        for plan_id, request_ids in by_plan.items():
            self.client.cancel_staged_units(plan_id, tuple(request_ids))
            for request_id in request_ids:
                del self._prefetch_templates[request_id]
            self._maybe_release_prefetch_plan(plan_id)

    def _maybe_release_prefetch_plan(self, plan_id: str) -> None:
        if any(template[0] == plan_id
               for template in self._prefetch_templates.values()):
            return
        self.client.release_plan(plan_id)

    def _run_sync_request(self, src_to_dst: torch.Tensor, *, operation: str) -> None:
        """Run the compatibility swap API through the canonical lifecycle."""
        if src_to_dst.numel() == 0:
            return
        request_id = f"sync-{operation}-{time.monotonic_ns()}"
        self.submit_request(request_id, src_to_dst, operation=operation)
        timeout_seconds = float(
            getattr(self.client, "timeout_seconds",
                    envs.VLLM_GRANULEKV_TIMEOUT_SECONDS))
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = self.query_request(request_id)
            if status.state is GranuleKVTransferState.ERROR:
                raise RuntimeError(status.error or "GranuleKV request failed")
            if status.ready:
                self.complete_request(request_id)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for GranuleKV transfer {request_id}")
            time.sleep(0.001)

    def _request_spec(
        self,
        src_to_dst: torch.Tensor,
        *,
        operation: str,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> GranuleKVRequestSpec:
        mappings = tuple(
            (int(source), int(destination))
            for source, destination in src_to_dst.to(
                device="cpu", dtype=torch.int64).tolist())
        source_ids = [mapping[0] for mapping in mappings]
        destination_ids = [mapping[1] for mapping in mappings]
        if operation == "write":
            gpu_ids, storage_ids = source_ids, destination_ids
        elif operation == "read":
            storage_ids, gpu_ids = source_ids, destination_ids
        else:
            raise ValueError(f"unsupported GranuleKV operation: {operation}")
        gpu_region_start: Optional[int] = None
        if layer_range is not None:
            if self.layout.num_gpu_regions < self.layout.num_layers:
                gpu_region_start = (
                    layer_range[0] % self.layout.num_gpu_regions)
        elif self.layout.num_gpu_regions < self.layout.num_layers:
            raise RuntimeError(
                "layer working-set mode only supports layer-ranged GranuleKV I/O")
        return GranuleKVRequestSpec(
            operation=operation,
            gpu_block_ids=tuple(gpu_ids),
            storage_block_ids=tuple(storage_ids),
            layer_range=layer_range,
            gpu_region_start=gpu_region_start,
        )

    @staticmethod
    def _mapping_matches_spec(src_to_dst: torch.Tensor,
                              spec: GranuleKVRequestSpec) -> bool:
        """Check a legacy staged mapping without rebuilding its request spec."""
        mappings = tuple(
            (int(source), int(destination))
            for source, destination in src_to_dst.to(
                device="cpu", dtype=torch.int64).tolist())
        if spec.operation == "read":
            storage_ids = tuple(mapping[0] for mapping in mappings)
            gpu_ids = tuple(mapping[1] for mapping in mappings)
        else:
            gpu_ids = tuple(mapping[0] for mapping in mappings)
            storage_ids = tuple(mapping[1] for mapping in mappings)
        return (gpu_ids == spec.gpu_block_ids
                and storage_ids == spec.storage_block_ids)

    def _validate_layer_range(
        self, layer_range: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if layer_range is None:
            return None
        start_layer, end_layer = (int(layer_range[0]), int(layer_range[1]))
        if not 0 <= start_layer < end_layer <= self.layout.num_layers:
            raise ValueError(
                "GranuleKV layer range is outside local KV cache: "
                f"[{start_layer}, {end_layer}) vs {self.layout.num_layers}")
        return start_layer, end_layer
