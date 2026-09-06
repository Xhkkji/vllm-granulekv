# SPDX-License-Identifier: Apache-2.0
"""A GPU worker class."""
import gc
import os
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferOperation, AsyncKVTransferRequest,
    AsyncKVTransferState)
from vllm.core.custom_schedulers.hierarchical_io import (
    RollingPrefetchConfig, RollingPrefetchRuntime, activate_layer_barrier)
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.platforms import current_platform
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta)
from vllm.utils import (GiB_bytes, MemorySnapshot, bind_kv_cache,
                        memory_profiling)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner import GPUModelRunnerBase, ModelRunner
from vllm.worker.model_runner_base import DeferredModelExecution
from vllm.worker.pooling_model_runner import PoolingModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase, WorkerBase,
                                     WorkerInput)

logger = init_logger(__name__)


class Worker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        # 普通异步 transfer 只需要在跨 engine iteration 期间保留 request
        # identity；mapping 已经在 submit 时交给 CacheEngine，poll 不再依赖
        # 或复制它。prefetch unit 则由下面的 Worker-local runtime 持有。
        self._async_kv_transfer_ids: Set[Tuple[int, str]] = set()
        # layer-window 和未来 sparse unit 共用一个 plan runtime。Scheduler
        # 仍是 block/reservation 的唯一所有者；这里仅保存预授权 mapping，
        # 根据 model progress 激活 GranuleKV handle，并缓存 completion event。
        self._prefetch_runtime = RollingPrefetchRuntime(
            RollingPrefetchConfig.from_env())
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.hf_config.model_type ==
                model_config.hf_config.model_type) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ("medusa", "mlp_speculator", "eagle", "deepseek_mtp")) \
                    else {"return_hidden_states": True}

        ModelRunnerClass: Type[GPUModelRunnerBase] = ModelRunner
        if model_config.runner_type == "pooling":
            ModelRunnerClass = PoolingModelRunner
        elif self.model_config.is_encoder_decoder:
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: GPUModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        if model_runner_cls is not None:
            self.model_runner = model_runner_cls(self.model_runner)

        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as pooling models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

        # Buffers saved before sleep
        self._sleep_saved_buffers: Dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def start_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.baseline_snapshot = MemorySnapshot()
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.load_model()

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_runner.save_sharded_state(
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.baseline_snapshot,
                weights_memory=self.model_runner.model_memory_usage) as result:
            self.model_runner.profile_run()

        self._assert_memory_footprint_increased_during_profiling()

        memory_for_current_instance = total_gpu_memory * \
            self.cache_config.gpu_memory_utilization
        available_kv_cache_memory = (memory_for_current_instance -
                                     result.non_kv_cache_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        msg = (f"Memory profiling takes {result.profile_time:.2f} seconds\n"
               "the current vLLM instance can use "
               "total_gpu_memory "
               f"({(total_gpu_memory / GiB_bytes):.2f}GiB)"
               " x gpu_memory_utilization "
               f"({self.cache_config.gpu_memory_utilization:.2f})"
               f" = {(memory_for_current_instance / GiB_bytes):.2f}GiB\n"
               "model weights take "
               f"{(result.weights_memory / GiB_bytes):.2f}GiB;"
               " non_torch_memory takes "
               f"{(result.non_torch_increase / GiB_bytes):.2f}GiB;"
               " PyTorch activation peak memory takes "
               f"{(result.torch_peak_increase / GiB_bytes):.2f}GiB;"
               " the rest of the memory reserved for KV Cache is "
               f"{(available_kv_cache_memory / GiB_bytes):.2f}GiB.")

        logger.info(msg)
        # Final cleanup
        gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def _assert_memory_footprint_increased_during_profiling(self):
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        free_gpu_memory, total = torch.cuda.mem_get_info()
        cuda_memory = total - free_gpu_memory
        assert self.baseline_snapshot.cuda_memory < cuda_memory, (
            "Error in memory profiling. "
            f"Initial used memory {self.baseline_snapshot.cuda_memory}, "
            f"currently used memory {cuda_memory}. "
            f"This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(
            num_gpu_blocks, self.cache_config.block_size,
            self.cache_config.is_attention_free,
            self.model_config.max_model_len,
            self.parallel_config.pipeline_parallel_size)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self._init_cache_engine()
        self._warm_up_model()
        # GranuleKV runtime 必须在模型 warmup、
        # CUDA Graph capture 和 workspace allocation 全部结束后再注册 vLLM KV
        # allocation。persistent CQ worker 仍到首次 submit 才真正启动。
        for cache_engine in self.cache_engine:
            cache_engine.initialize_granulekv()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.gpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]
        bind_kv_cache(self.compilation_config.static_forward_context,
                      self.gpu_cache)

    def _warm_up_model(self) -> None:
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.gpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][Worker.prepare] ve=%d num_seq_groups=%d "
                "num_steps=%d swap_in=%d swap_out=%d copy=%d",
                virtual_engine,
                num_seq_groups,
                num_steps,
                blocks_to_swap_in.shape[0],
                blocks_to_swap_out.shape[0],
                blocks_to_copy.shape[0],
            )

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            if envs.VLLM_V0_SWAP_TRACE:
                logger.info(
                    "[V0_SWAP_TRACE][Worker.execute] ve=%d op=swap_in "
                    "mappings=%d",
                    virtual_engine,
                    worker_input.blocks_to_swap_in.shape[0],
                )
            cache_engine = self.cache_engine[virtual_engine]
            if cache_engine.granulekv_connector is not None:
                if not cache_engine.swap_in_async(
                        worker_input.blocks_to_swap_in):
                    # 当前 scheduler output 已经为 swap-in 分配了目标 GPU
                    # blocks。在 GranuleKV logical request 完成前，必须保留同一批输入
                    # 并禁止 attention 读取尚未恢复完整的 KV。
                    raise DeferredModelExecution(
                        request_ids=[],
                        message=(
                            "resident GranuleKV swap-in is still in flight; retry "
                            f"virtual_engine={virtual_engine}"),
                    )
            else:
                cache_engine.swap_in(worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            if envs.VLLM_V0_SWAP_TRACE:
                logger.info(
                    "[V0_SWAP_TRACE][Worker.execute] ve=%d op=swap_out "
                    "mappings=%d",
                    virtual_engine,
                    worker_input.blocks_to_swap_out.shape[0],
                )
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            if envs.VLLM_V0_SWAP_TRACE:
                logger.info(
                    "[V0_SWAP_TRACE][Worker.execute] ve=%d op=copy "
                    "mappings=%d",
                    virtual_engine,
                    worker_input.blocks_to_copy.shape[0],
                )
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][Worker.execute] ve=%d phase=done "
                "num_seq_groups=%d",
                virtual_engine,
                worker_input.num_seq_groups,
            )

    def submit_async_kv_transfers(
        self,
        virtual_engine: int,
        requests: Sequence[AsyncKVTransferRequest],
    ) -> List[AsyncKVTransferEvent]:
        """批量提交 Scheduler 本轮激活的异步 KV read/write。

        该 RPC 只执行非阻塞 submit。mapping 在 Worker 侧转换一次；普通
        request 只记录 request identity，layerwise unit 则保存到
        ``RollingPrefetchRuntime``，并在真实 model progress 到达时激活。
        后端容量由 scheduler policy 与 GranuleKV request table 共同限制。
        """
        cache_engine = self.cache_engine[virtual_engine]
        events: List[AsyncKVTransferEvent] = []
        prepared: list[tuple[AsyncKVTransferRequest, torch.Tensor]] = []
        seen_ids: Set[Tuple[int, str]] = set()
        for request in requests:
            key = (virtual_engine, request.request_id)
            if key in self._async_kv_transfer_ids or key in seen_ids:
                raise RuntimeError(
                    f"duplicate async KV submit: {request.request_id}")
            seen_ids.add(key)
            mapping = torch.tensor(
                request.block_mapping,
                device="cpu",
                dtype=torch.int64,
            ).view(-1, 2)
            prepared.append((request, mapping))

        for request, mapping in prepared:
            key = (virtual_engine, request.request_id)
            if request.prefetch_plan_id is not None:
                events.extend(self._prefetch_runtime.submit_or_stage(
                    virtual_engine,
                    request,
                    mapping,
                    lambda unit, unit_mapping: cache_engine.
                    submit_async_kv_transfer(
                        unit.request_id,
                        unit.operation,
                        unit_mapping,
                        layer_range=unit.layer_range),
                ))
                continue

            event = cache_engine.submit_async_kv_transfer(
                request.request_id,
                request.operation,
                mapping,
                layer_range=request.layer_range)
            events.append(event)
            if event.state == AsyncKVTransferState.PENDING:
                self._async_kv_transfer_ids.add(key)
        return events

    def poll_async_kv_transfers(
            self, virtual_engine: int) -> List[AsyncKVTransferEvent]:
        """非阻塞轮询当前 virtual engine 的全部 outstanding transfer。"""
        cache_engine = self.cache_engine[virtual_engine]
        events = list(self._prefetch_runtime.poll_units(
            virtual_engine,
            lambda unit: cache_engine.poll_async_kv_transfer(unit.request_id),
        ))
        keys = [
            key for key in self._async_kv_transfer_ids
            if key[0] == virtual_engine
        ]
        for key in keys:
            _, request_id = key
            event = cache_engine.poll_async_kv_transfer(request_id)
            events.append(event)
            if event.state != AsyncKVTransferState.PENDING:
                self._async_kv_transfer_ids.remove(key)
        self._log_prefetch_runtime_traces()
        return events

    def discard_staged_async_kv_transfers(
        self,
        virtual_engine: int,
        request_ids: Sequence[str],
    ) -> None:
        """清理取消 plan 中从未激活的 Worker descriptor templates。"""
        self._prefetch_runtime.discard_units(virtual_engine, request_ids)

    def activate_hierarchical_layer_barrier(
            self, virtual_engine: int,
            request_ids: Tuple[str, ...]):
        """返回覆盖一次 model forward 的 worker-local barrier context。"""
        return activate_layer_barrier(
            self._wait_for_hierarchical_layer,
            virtual_engine=virtual_engine,
            request_ids=request_ids,
            release_callback=self._release_hierarchical_layer,
        )

    def _release_hierarchical_layer(
        self,
        virtual_engine: int,
        request_ids: Tuple[str, ...],
        layer_index: int,
    ) -> None:
        """模型消费完 window 后，把对应环形 KV region 标记为可覆盖。"""
        if envs.VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE:
            # layer.forward 返回只表示 kernel 已入队。GranuleKV DMA 不属于 PyTorch
            # stream，若此时立刻覆盖同一 region，可能与尚未结束的 attention
            # 读发生竞争。验证版在 window 边界显式同步，先保证语义正确；后续
            # 可改成 CUDA event -> MDS activation 的非阻塞依赖协议。
            torch.cuda.synchronize()
        self._prefetch_runtime.release_layer(virtual_engine, request_ids,
                                             layer_index)
        self._log_prefetch_runtime_traces()

    def _wait_for_hierarchical_layer(
        self,
        virtual_engine: int,
        request_ids: Tuple[str, ...],
        layer_index: int,
    ) -> Optional[Tuple[int, ...]]:
        """按 layer progress 激活未来 unit，并等待当前 unit 物理 READY。"""
        cache_engine = self.cache_engine[virtual_engine]
        self._prefetch_runtime.wait_ready(
            virtual_engine,
            request_ids,
            layer_index,
            lambda unit, mapping: cache_engine.submit_async_kv_transfer(
                unit.request_id,
                unit.operation,
                mapping,
                layer_range=unit.layer_range),
            lambda unit: cache_engine.poll_async_kv_transfer(unit.request_id),
            max_active=envs.VLLM_GRANULEKV_MAX_IN_FLIGHT,
        )
            # wait_ready 只保证物理 GranuleKV event 到达 READY；随后再用 logical block
        # 目录验证当前 sparse consumer 的全部 block 已驻留。dense unit 的
        # consumer_block_indices=None，因此此检查不会改变原有 attention。
        sparse_kv_blocks = self._prefetch_runtime.require_resident_layer(
            request_ids, layer_index)
        self._log_prefetch_runtime_traces()
        return sparse_kv_blocks

    def _log_prefetch_runtime_traces(self) -> None:
        """输出 worker 物理 READY 与 barrier wait，避免混入 scheduler 延迟。"""
        traces = self._prefetch_runtime.pop_traces()
        if not envs.VLLM_V0_SWAP_TRACE:
            return
        for trace in traces:
            logger.info(
                "[GRANULEKV_PREFETCH][Worker] phase=%s plan_id=%s "
                "request_id=%s unit=%d monotonic_ns=%d layer=%s "
                "wait_ms=%.3f lead_units=%d",
                trace.phase,
                trace.plan_id,
                trace.request_id,
                trace.unit_index,
                trace.monotonic_ns,
                trace.layer_index,
                trace.wait_ns / 1.0e6,
                trace.lead_units,
            )

    def _get_cached_seq_group_metadata(
            self,
            seq_group_metadata_list: List[Union[SequenceGroupMetadata,
                                                SequenceGroupMetadataDelta]],
            finished_request_ids: List[str]) -> List[SequenceGroupMetadata]:
        """Return a list of cached Sequence Group Metadata after updating its
        state.

        It is used because scheduler only sends delta to workers to reduce
        the data payload size. The function also cleans up cache based on
        a given `finished_request_ids`.
        """
        new_seq_group_metadata_list = []
        for metadata_or_delta in seq_group_metadata_list:
            request_id = metadata_or_delta.request_id
            if request_id not in self._seq_group_metadata_cache:
                # The first prefill.
                assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                self._seq_group_metadata_cache[request_id] = metadata_or_delta
            else:
                # The first prefill is already cached.
                if isinstance(metadata_or_delta, SequenceGroupMetadataDelta):
                    self._seq_group_metadata_cache[request_id].apply_delta(
                        metadata_or_delta)
                else:
                    # If metadata snapshot is sent again, it is
                    # preempted. Reset the cache because we need to start
                    # from scratch.
                    assert isinstance(metadata_or_delta, SequenceGroupMetadata)
                    self._seq_group_metadata_cache[
                        request_id] = metadata_or_delta

            new_seq_group_metadata_list.append(
                self._seq_group_metadata_cache[request_id])

        # Clean up finished ids
        for finished_id in finished_request_ids:
            del self._seq_group_metadata_cache[finished_id]

        return new_seq_group_metadata_list

    def _execute_model_spmd(
        self,
        execute_model_req: ExecuteModelRequest,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Optional[List[SamplerOutput]]:
        if execute_model_req is not None:
            # I/O completion 与请求生命周期不同：长 prompt 在 restore 完成后
            # 仍可能经过多个 chunked-prefill iteration。只在 Engine 明确通知
            # finished/abort 时回收 sparse residency，避免提前丢失消费约束。
            self._prefetch_runtime.forget_seq_groups(
                execute_model_req.finished_requests_ids)
            new_seq_group_metadata_list = self._get_cached_seq_group_metadata(
                execute_model_req.seq_group_metadata_list,
                execute_model_req.finished_requests_ids)

            execute_model_req.seq_group_metadata_list = (
                new_seq_group_metadata_list)
        output = super()._execute_model_spmd(execute_model_req,
                                             intermediate_tensors)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        return self.model_runner.add_prompt_adapter(prompt_adapter_request)

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.remove_lora(prompt_adapter_id)

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        return self.model_runner.pin_prompt_adapter(prompt_adapter_id)

    def list_prompt_adapters(self) -> Set[int]:
        return self.model_runner.list_prompt_adapters()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half.")


def raise_if_cache_size_invalid(num_gpu_blocks, block_size, is_attention_free,
                                max_model_len, pipeline_parallel_size) -> None:
    if is_attention_free and num_gpu_blocks != 0:
        raise ValueError("No memory should be allocated for the cache blocks "
                         f"for an attention-free model, but {num_gpu_blocks} "
                         "blocks are allocated.")
    if not is_attention_free and num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * (num_gpu_blocks // pipeline_parallel_size)
    if not is_attention_free and max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
