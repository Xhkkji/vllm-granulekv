# SPDX-License-Identifier: Apache-2.0
"""CacheEngine class for managing the KV cache."""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import vllm.envs as envs
from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferOperation, AsyncKVTransferState)
from vllm.logger import init_logger
from vllm.core.custom_schedulers.hierarchical_io import (
    get_layer_working_set_regions)
from vllm.granulekv.connector import GranuleKVTransferState
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

logger = init_logger(__name__)


@dataclass
class _AsyncKVTrace:
    """一笔 transfer 的观测时间线；不参与 I/O 正确性判断。"""

    operation: AsyncKVTransferOperation
    num_mappings: int
    started_at: float
    submitted_at_ns: int
    first_poll_at_ns: int | None = None


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.granulekv_enabled = envs.VLLM_GRANULEKV_ENABLE
        if self.granulekv_enabled and (
                parallel_config.tensor_parallel_size != 1
                or parallel_config.pipeline_parallel_size != 1):
            raise ValueError(
                "GranuleKV connector currently requires TP=1 and PP=1")

        self.granulekv_connector = None
        # legacy deferred swap-in 仍是单个 model-execution dependency；新的
        # AsyncKVScheduler 使用下面按 request_id 索引的多槽 trace 表。
        self._granulekv_transfer_started_at: float | None = None
        self._granulekv_async_kv_traces: Dict[str, _AsyncKVTrace] = {}

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=model_config.use_mla)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = (
            [] if self.granulekv_enabled else
            self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
        )
        self.swap_trace_enabled = envs.VLLM_V0_SWAP_TRACE

    def initialize_granulekv(self) -> None:
        """在所有 CUDA warmup 后启用 GranuleKV transport。"""
        if not self.granulekv_enabled:
            return
        assert self.granulekv_connector is not None
        self.granulekv_connector.start()

    def _log_swap_event(self, op_name: str, src_to_dst: torch.Tensor,
                        elapsed_s: float) -> None:
        """记录一次 tensor mapping swap 的核心信息。"""
        self._log_swap_event_count(op_name, int(src_to_dst.shape[0]),
                                   elapsed_s)

    def _log_swap_event_count(self, op_name: str, num_mappings: int,
                              elapsed_s: float) -> None:
        """记录不再持有原始 mapping 的异步 transfer。"""
        if not self.swap_trace_enabled:
            return

        block_bytes = self.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)
        total_bytes = num_mappings * block_bytes
        logger.info(
            "[V0_SWAP_TRACE][CacheEngine] op=%s mappings=%d "
            "block_size=%d block_bytes=%d total_bytes=%d "
            "num_attention_layers=%d elapsed_ms=%.3f",
            op_name,
            num_mappings,
            self.block_size,
            block_bytes,
            total_bytes,
            self.num_attention_layers,
            elapsed_s * 1000,
        )

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device.

        【GranuleKV 直通调用链】新后端开启时，GPU 分配改用 64KB 对齐 owner，
        但返回给 attention 的 tensor shape/stride 与原生分配完全一致。
        """
        kv_cache_generic_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        try:
            kv_cache_stride_order = self.attn_backend.get_kv_cache_stride_order(
            )
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_generic_shape)))

        # The allocation respects the backend-defined stride order to ensure
        # the semantic remains consistent for each backend. We first obtain the
        # generic kv cache shape and then permute it according to the stride
        # order which could result in a non-contiguous tensor.
        kv_cache_allocation_shape = tuple(kv_cache_generic_shape[i]
                                          for i in kv_cache_stride_order)

        if self.granulekv_enabled and device == "cuda":
            from vllm.granulekv.connector import GranuleKVConnector
            working_set_regions = get_layer_working_set_regions()
            self.granulekv_connector = GranuleKVConnector(
                allocation_shape=kv_cache_allocation_shape,
                stride_order=kv_cache_stride_order,
                dtype=self.dtype,
                device_index=self.device_config.device.index or 0,
                num_layers=self.num_attention_layers,
                num_gpu_regions=(working_set_regions
                                 or self.num_attention_layers),
                num_gpu_blocks=num_blocks,
                num_storage_blocks=self.num_cpu_blocks,
            )
            return self.granulekv_connector.gpu_cache

        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            layer_kv_cache = torch.zeros(
                kv_cache_allocation_shape,
                dtype=self.dtype,
                pin_memory=pin_memory,
                device=device).permute(*kv_cache_stride_order)

            # view back to (TOTAL_PAGES, PAGE_SIZE, entry_shape...) for cases
            # when entry_shape is higher than 1D
            kv_cache.append(layer_kv_cache)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        """把 scheduler 的 storage->GPU block mapping 恢复到 KV cache。

        【GranuleKV 直通调用链】新后端下，本函数等待的只是 GPU worker 发布
        request ready；CPU 不读取 NVMe CQ，也不参与 payload 搬运。返回后 Worker
        才会继续发起当前 engine step 的 attention。
        """
        start = time.perf_counter()
        if self.granulekv_connector is not None:
            self.granulekv_connector.swap_in(src_to_dst)
        else:
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.cpu_cache[i],
                                              self.gpu_cache[i], src_to_dst)
        self._log_swap_event("swap_in", src_to_dst, time.perf_counter() - start)

    def swap_in_async(self, src_to_dst: torch.Tensor) -> bool:
        """推进 legacy resident GranuleKV swap-in，未完成时 defer 当前 batch。

        该接口只服务 GranuleKV direct 路径。目标 GPU block 在返回 True 前不能被
        attention 消费；SSD completion 和数据可见性由 daemon 内的常驻 GPU CQ
        service 保证。
        """
        if self.granulekv_connector is None:
            raise RuntimeError("async swap-in is only supported by GranuleKV")
        if self._granulekv_transfer_started_at is None:
            self._granulekv_transfer_started_at = time.perf_counter()
        request_id = "legacy-deferred-swap-in"
        status = self.granulekv_connector.submit_request(
            request_id, src_to_dst, operation="read")
        if status.state is not GranuleKVTransferState.READY:
            status = self.granulekv_connector.query_request(request_id)
        if not status.ready:
            if status.state is GranuleKVTransferState.ERROR:
                self.granulekv_connector.cancel_request(request_id)
                raise RuntimeError(status.error or "GranuleKV swap-in failed")
            return False
        if src_to_dst.numel() != 0:
            self.granulekv_connector.complete_request(request_id)
        elapsed_s = time.perf_counter() - self._granulekv_transfer_started_at
        self._granulekv_transfer_started_at = None
        self._log_swap_event("swap_in", src_to_dst, elapsed_s)
        return True

    def submit_async_kv_transfer(
        self,
        request_id: str,
        operation: AsyncKVTransferOperation,
        src_to_dst: torch.Tensor,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> AsyncKVTransferEvent:
        """把 Scheduler 的异步 read/write 提交给 resident GranuleKV。

        该方法只负责控制面提交，不等待 SSD 数据完成。目标 GPU block 已经
        由 AsyncKVScheduler 预留，因此从提交开始到 READY 之前都禁止
        attention 使用这些 block。

        """
        if self.granulekv_connector is None:
            raise RuntimeError(
                "async KV scheduling requires the resident GranuleKV connector")
        if request_id in self._granulekv_async_kv_traces:
            raise RuntimeError(f"duplicate async KV transfer: {request_id}")
        trace = _AsyncKVTrace(operation=operation,
                              num_mappings=int(src_to_dst.shape[0]),
                              started_at=time.perf_counter(),
                              submitted_at_ns=time.monotonic_ns())
        self._granulekv_async_kv_traces[request_id] = trace
        if self.swap_trace_enabled:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][CacheEngine] phase=submit "
                "operation=%s request_id=%s submit_monotonic_ns=%d "
                "mappings=%d layer_range=%s",
                operation.value,
                request_id,
                trace.submitted_at_ns,
                src_to_dst.shape[0],
                layer_range,
            )
        try:
            # Layerwise runtime owns the plan and mappings. Once a unit is
            # activated it is indistinguishable from an ordinary request.
            status = self.granulekv_connector.submit_request(
                request_id,
                src_to_dst,
                operation=operation.value,
                layer_range=layer_range)
        except Exception as exc:
            del self._granulekv_async_kv_traces[request_id]
            return AsyncKVTransferEvent(request_id,
                                        AsyncKVTransferState.ERROR,
                                        error=str(exc))
        if status.ready:
            self._finish_async_kv_transfer_trace(request_id)
            return AsyncKVTransferEvent(request_id,
                                        AsyncKVTransferState.READY)
        return AsyncKVTransferEvent(request_id, AsyncKVTransferState.PENDING)

    def poll_async_kv_transfer(
            self, request_id: str) -> AsyncKVTransferEvent:
        """非阻塞查询一个已经提交的 GranuleKV read/write。"""
        if self.granulekv_connector is None:
            raise RuntimeError(
                "async KV scheduling requires the resident GranuleKV connector")
        trace = self._granulekv_async_kv_traces.get(request_id)
        if trace is None:
            raise RuntimeError(
                f"CacheEngine has no async KV transfer: {request_id}")

        if trace.first_poll_at_ns is None:
            trace.first_poll_at_ns = time.monotonic_ns()
            if self.swap_trace_enabled:
                logger.info(
                    "[V0_SWAP_TRACE][AsyncKV][CacheEngine] "
                    "phase=first_poll operation=%s request_id=%s "
                    "first_poll_monotonic_ns=%d",
                    trace.operation.value,
                    request_id,
                    trace.first_poll_at_ns,
                )

        try:
            status = self.granulekv_connector.query_request(request_id)
        except Exception as exc:
            del self._granulekv_async_kv_traces[request_id]
            return AsyncKVTransferEvent(request_id,
                                        AsyncKVTransferState.ERROR,
                                        error=str(exc))
        if status.state is GranuleKVTransferState.ERROR:
            try:
                self.granulekv_connector.cancel_request(request_id)
            except Exception:
                logger.exception(
                    "failed to release GranuleKV request after error: %s",
                    request_id)
            del self._granulekv_async_kv_traces[request_id]
            return AsyncKVTransferEvent(request_id,
                                        AsyncKVTransferState.ERROR,
                                        error=status.error)
        if not status.ready:
            return AsyncKVTransferEvent(request_id,
                                        AsyncKVTransferState.PENDING)
        try:
            self.granulekv_connector.complete_request(request_id)
        except Exception as exc:
            del self._granulekv_async_kv_traces[request_id]
            return AsyncKVTransferEvent(request_id,
                                        AsyncKVTransferState.ERROR,
                                        error=str(exc))
        self._finish_async_kv_transfer_trace(request_id)
        return AsyncKVTransferEvent(request_id, AsyncKVTransferState.READY)

    def _finish_async_kv_transfer_trace(self, request_id: str) -> None:
        """在异步 read/write 完成后记录耗时并删除对应 trace。"""
        trace = self._granulekv_async_kv_traces.pop(request_id, None)
        if trace is None:
            raise RuntimeError(f"missing async KV trace: {request_id}")
        ready_ns = time.monotonic_ns()
        elapsed_s = time.perf_counter() - trace.started_at
        if self.swap_trace_enabled:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][CacheEngine] phase=ready "
                "operation=%s request_id=%s submitted_monotonic_ns=%d "
                "first_poll_monotonic_ns=%s ready_monotonic_ns=%d "
                "submit_to_ready_ms=%.3f",
                trace.operation.value,
                request_id,
                trace.submitted_at_ns,
                trace.first_poll_at_ns,
                ready_ns,
                (ready_ns - trace.submitted_at_ns) / 1.0e6,
            )
        self._log_swap_event_count(
            "swap_in" if trace.operation == AsyncKVTransferOperation.READ else
            "swap_out", trace.num_mappings, elapsed_s)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        """把 scheduler 的 GPU->storage block mapping 写出。

        【GranuleKV 直通调用链】新后端直接从 vLLM physical block 发起 SSD
        write，不再先复制到 CPU cache，也不经过中间 payload cache。
        """
        start = time.perf_counter()
        if self.granulekv_connector is not None:
            self.granulekv_connector.swap_out(src_to_dst)
        else:
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.gpu_cache[i],
                                              self.cpu_cache[i], src_to_dst)
        self._log_swap_event("swap_out", src_to_dst,
                             time.perf_counter() - start)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        key_cache_entry = num_heads * head_size

        # For MLA there is no value cache, since the latent vector
        # is joint keys and values.
        value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        total = num_attention_layers * cache_config.block_size * \
            (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
