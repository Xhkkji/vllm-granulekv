# SPDX-License-Identifier: Apache-2.0
"""A block manager that manages token blocks."""
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

import vllm.envs as envs
from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.custom_schedulers.granulekv_prefix import (
    compute_full_prefix_block_hashes)
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.block_reservation import (BlockPrefixRestoreReservation,
                                         BlockResidency,
                                         BlockSwapReservation, LogicalBlockKey)
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

SeqId = int
EncoderSeqId = str

logger = init_logger(__name__)


@dataclass(frozen=True)
class _ReservedBlockTable:
    """一个 sequence 在 reservation 前后的两份 block table 内容。"""

    seq_id: SeqId
    source_blocks: Tuple[Block, ...]
    target_blocks: Tuple[Block, ...]
    # 只有 newly_allocated_targets 归当前 reservation 临时所有；复用的
    # clean storage prefix 仍由 residency directory 持有，abort 时不能释放。
    newly_allocated_targets: Tuple[Block, ...]


@dataclass(frozen=True)
class _BlockSwapReservationRecord:
    """BlockSpaceManager 私有保存的 reservation 资源所有权。"""

    public: BlockSwapReservation
    tables: Tuple[_ReservedBlockTable, ...]


@dataclass(frozen=True)
class _BlockPrefixRestoreReservationRecord:
    """prefix read 完成前由 block manager 独占的源 block 引用。"""

    public: BlockPrefixRestoreReservation
    seq: Sequence
    source_blocks: Tuple[Block, ...]
    pending_target_blocks: Tuple[Block, ...]


class SelfAttnBlockSpaceManager(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    This class implements the design described in
    https://github.com/vllm-project/vllm/pull/3492.

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator, self.block_size, self.enable_caching)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)

        # reservation 中的目标 block 已经从目标 allocator 分配，但尚未
        # 发布到正式 block table。只要记录仍在这里，源和目标两份物理
        # block 就都不能被复用；commit/abort 必须且只能消费一次记录。
        self._reservation_counter = itertools.count()
        self._block_swap_reservations: Dict[
            str, _BlockSwapReservationRecord] = {}
        self._prefix_restore_reservations: Dict[
            str, _BlockPrefixRestoreReservationRecord] = {}
        # 正式 block table 指向 GPU 时，这里按逻辑位置持有有效 storage
        # 副本。物理 block id 会被 allocator 复用，不能作为跨调度身份；
        # `(seq_id, logical_index)` 才是 residency 和 pin 的稳定键。
        self._storage_replicas: Dict[LogicalBlockKey, Block] = {}
        self._pinned_gpu_blocks: Dict[LogicalBlockKey, int] = {}

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks
                < self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _allocate_sequence(self, seq: Sequence) -> BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        if seq.get_token_ids():
            # NOTE: If there are any factors affecting the block besides
            # token_ids, they should be added as input to extra_hash.
            extra_hash = seq.extra_hash()

            # Add blocks to the block table only if the sequence is non empty.
            block_table.allocate(token_ids=seq.get_token_ids(),
                                 extra_hash=extra_hash)

        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        block_table: BlockTable = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table

        # Track seq
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
            "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            block_table = self._allocate_sequence(encoder_seq)
            self.cross_block_tables[request_id] = block_table

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=seq.data.get_num_computed_tokens(),
            extra_hash=seq.extra_hash(),
        )
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id

        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq.seq_id].physical_block_ids)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # storage replica 不属于正式 block table，sequence 结束时必须单独
        # 回收；若 table 当前就在 storage，directory 中不会保存重复所有权。
        for replica in self._pop_storage_replicas(seq_id):
            self.block_allocator.free(replica)
        for key in tuple(self._pinned_gpu_blocks):
            if key.seq_id == seq_id:
                del self._pinned_gpu_blocks[key]

        # Free table/blocks
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

    def free_granulekv_prefix_store(self, seq: Sequence) -> None:
        """释放 prefix populate 后位于 storage 的正式 block table。

        原生 ``free`` 的 last-access tracker 只支持 GPU id；prefix store
        commit 后 table 已经切到 CPU/storage，所以这里跳过那一步。其余
        sequence tracker、replica 和 block refcount 的释放顺序保持一致，
        完整 immutable storage block 会以 refcount=0 留在 CPU LRU 中。
        """
        seq_id = seq.seq_id
        if seq_id not in self.block_tables:
            return
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)
        for replica in self._pop_storage_replicas(seq_id):
            self.block_allocator.free(replica)
        for key in tuple(self._pinned_gpu_blocks):
            if key.seq_id == seq_id:
                del self._pinned_gpu_blocks[key]
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        return block_ids  # type: ignore

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = self.cross_block_tables[request_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # If prefix caching is enabled, mark immutable blocks as computed
        # right after they have been scheduled (for prefill). This assumes
        # the scheduler is synchronous so blocks are actually computed when
        # scheduling the next batch.
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        computed_seq_block_ids = []
        for seq in seqs:
            all_blocks = self.block_tables[seq.seq_id].physical_block_ids
            num_cached_tokens = (
                self._computed_blocks_tracker.get_num_cached_tokens(seq))
            assert num_cached_tokens % self.block_size == 0
            num_cached_blocks = num_cached_tokens // self.block_size
            computed_block_ids = all_blocks[:num_cached_blocks]
            computed_seq_block_ids.append(computed_block_ids)

        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            computed_seq_block_ids)  # type: ignore

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

        # Track child seq
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.CPU,
                                                         dst_device=Device.GPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id)
                for cpu_block_id, gpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][BlockManager] op=swap_in request_id=%s "
                "seqs=%d mappings=%d",
                seq_group.request_id,
                len(seq_group.get_seqs(status=SequenceStatus.SWAPPED)),
                len(physical_block_id_mapping),
            )

        return physical_block_id_mapping

    def reserve_swap_in(self,
                        seq_group: SequenceGroup) -> BlockSwapReservation:
        """预留 SSD/storage -> GPU 的目标 block，但不发布 block table。"""
        return self._reserve_swap(seq_group, SequenceStatus.SWAPPED,
                                  Device.CPU, Device.GPU)

    def get_granulekv_cached_prefix_blocks(self, seq: Sequence,
                                     device: Device) -> int:
        """返回指定层级中从 token 0 连续命中的完整 block 数。

        这里直接查询 allocator，不写入 ``ComputedBlocksTracker``。restore
        READY 之前缓存长度仍不应对原生 Scheduler 可见。
        """
        storage_blocks, gpu_blocks = self.get_granulekv_cached_prefix_block_counts(
            seq)
        return storage_blocks if device == Device.CPU else gpu_blocks

    def get_granulekv_cached_prefix_block_counts(
            self, seq: Sequence) -> Tuple[int, int]:
        """一次 hash 计算同时返回 ``(storage_blocks, gpu_blocks)``。"""
        if not self.enable_caching:
            return 0, 0
        if envs.VLLM_GRANULEKV_TRUST_PREPOPULATED_PREFIX:
            self._seed_trusted_granulekv_storage_prefix(seq)
        block_hashes = compute_full_prefix_block_hashes(
            seq.get_token_ids(), self.block_size, seq.extra_hash())
        storage_blocks = len(
            self.block_allocator.find_cached_blocks_prefix(
                block_hashes, device=Device.CPU))
        gpu_blocks = len(
            self.block_allocator.find_cached_blocks_prefix(
                block_hashes, device=Device.GPU))
        return storage_blocks, gpu_blocks

    def _seed_trusted_granulekv_storage_prefix(self, seq: Sequence) -> None:
        """为两阶段实验重建已经写入 SSD 的 prefix allocator 元数据。

        此函数不执行任何 I/O，只能在第一阶段使用相同模型、token、block size、
        storage 容量和全新 allocator 顺序写过 SSD 后启用。显式 block 数防止把
        第二阶段人为修改的 suffix 也声明为命中。创建出的 block 立即 free 到
        CPU prefix-cache evictor；后续正常 lookup/reservation 会重新持有引用。
        """
        if not envs.VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE:
            raise RuntimeError(
                "trusted prepopulated prefix is only valid in working-set mode")
        trusted_blocks = envs.VLLM_GRANULEKV_TRUSTED_PREFIX_BLOCKS
        max_full_blocks = len(seq.get_token_ids()) // self.block_size
        if trusted_blocks <= 0 or trusted_blocks > max_full_blocks:
            raise ValueError(
                "VLLM_GRANULEKV_TRUSTED_PREFIX_BLOCKS must describe a positive "
                "complete prefix of the current request")
        token_ids = seq.get_token_ids()[:trusted_blocks * self.block_size]
        hashes = compute_full_prefix_block_hashes(token_ids, self.block_size,
                                                  seq.extra_hash())
        if len(self.block_allocator.find_cached_blocks_prefix(
                hashes, device=Device.CPU)) == trusted_blocks:
            return

        token_blocks = [
            token_ids[index:index + self.block_size]
            for index in range(0, len(token_ids), self.block_size)
        ]
        seeded = self.block_allocator.allocate_immutable_blocks(
            prev_block=None,
            block_token_ids=token_blocks,
            device=Device.CPU,
            extra_hash=seq.extra_hash(),
        )
        block_ids = [block.block_id for block in seeded
                     if block.block_id is not None]
        if len(block_ids) != trusted_blocks:
            raise RuntimeError("failed to seed complete trusted SSD prefix")
        self.block_allocator.mark_blocks_as_computed_on_device(
            block_ids, Device.CPU)
        for block in seeded:
            self.block_allocator.free(block)
        logger.warning(
            "[GRANULEKV_WORKING_SET] trusted prepopulated SSD prefix seeded "
            "seq_id=%s blocks=%d; payload integrity is an experiment premise",
            seq.seq_id, trusted_blocks)

    def can_reserve_granulekv_prefix_restore(
        self,
        num_prefix_blocks: int,
        num_gpu_cached_blocks: int,
    ) -> AllocStatus:
        """判断当前是否能为 SSD 扩展 prefix 预留 GPU target。

        HBM 已命中的前缀只增加 block 引用，不消耗新物理块；真正需要预留
        的数量是 ``SSD prefix - HBM prefix``。完整 suffix 尚不分配，后续
        仍由原生 running prefill 的 ``can_append_slots`` 控制。
        """
        if not 0 <= num_gpu_cached_blocks < num_prefix_blocks:
            return AllocStatus.NEVER
        if (self.num_total_gpu_blocks - num_prefix_blocks
                < self.watermark_blocks):
            return AllocStatus.NEVER
        required = num_prefix_blocks - num_gpu_cached_blocks
        free_blocks = self.get_num_free_gpu_blocks()
        if free_blocks - required >= self.watermark_blocks:
            return AllocStatus.OK
        return AllocStatus.LATER

    def reserve_granulekv_prefix_restore(
        self,
        seq_group: SequenceGroup,
        num_prefix_blocks: int,
        num_gpu_cached_blocks: int,
    ) -> BlockPrefixRestoreReservation:
        """为新请求建立部分 GPU table，并预留 SSD prefix 的直接恢复事务。

        table 只包含连续命中的完整 prefix：已有 HBM hit 直接复用，SSD
        扩展部分使用 allocator 的 pending target。未命中的 suffix 不在这里
        分配，READY 后由原生 running prefill 的 append_slots 补齐。
        """
        if not self.enable_caching:
            raise RuntimeError("GranuleKV prefix restore requires prefix caching")
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        if len(waiting_seqs) != 1:
            raise ValueError("GranuleKV prefix v0 requires one waiting sequence")
        seq = waiting_seqs[0]
        if seq.seq_id in self.block_tables:
            raise RuntimeError("prefix restore target is already allocated")
        if not 0 <= num_gpu_cached_blocks < num_prefix_blocks:
            raise ValueError("SSD prefix must extend the GPU cached prefix")

        source_blocks: Tuple[Block, ...] = ()
        target_blocks: List[Block] = []
        pending_targets: List[Block] = []
        try:
            prefix_token_ids = seq.get_token_ids()[:num_prefix_blocks *
                                                   self.block_size]
            token_blocks = [
                prefix_token_ids[index:index + self.block_size]
                for index in range(0, len(prefix_token_ids), self.block_size)
            ]

            # 先复用已经 computed 的 HBM prefix。lookup 已经确认这部分连续
            # 命中，因此 allocate_immutable_blocks 不会创建新物理 block。
            if num_gpu_cached_blocks:
                target_blocks.extend(
                    self.block_allocator.allocate_immutable_blocks(
                        prev_block=None,
                        block_token_ids=token_blocks[:num_gpu_cached_blocks],
                        device=Device.GPU,
                        extra_hash=seq.extra_hash(),
                    ))
            previous_target = target_blocks[-1] if target_blocks else None

            # SSD 扩展部分分配独立、未注册 hash 的 pending target。即使多个
            # loading 请求共享 token prefix，也不会在 DMA 期间共享物理地址。
            pending_targets = self.block_allocator.allocate_pending_restore_blocks(
                prev_block=previous_target,
                block_token_ids=token_blocks[num_gpu_cached_blocks:],
                device=Device.GPU,
                extra_hash=seq.extra_hash(),
            )
            target_blocks.extend(pending_targets)
            block_table = BlockTable(
                block_size=self.block_size,
                block_allocator=self.block_allocator,
                _blocks=target_blocks,
                max_block_sliding_window=self.max_block_sliding_window,
            )
            self.block_tables[seq.seq_id] = block_table
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

            # 初始化原生 tracker，使 loading 期间 abort/free 仍满足生命周期。
            # 在 continuous batching + priority preempt 下，同一 token prefix
            # 可能已被 allocator 认定为 HBM cached，但这个新的 waiting seq
            # 尚未进入过原生 admission，ComputedBlocksTracker 还没有对应的
            # cached-token frontier。这里以 allocator 查询得到的连续 HBM hit
            # 为准初始化 tracker；SSD 扩展部分仍保持 IO_PENDING，只有 READY
            # 后才会在 commit_granulekv_prefix_restore 中提升到完整 prefix 长度。
            observed_gpu_tokens = self._computed_blocks_tracker.get_num_cached_tokens(
                seq)
            expected_gpu_tokens = num_gpu_cached_blocks * self.block_size
            if observed_gpu_tokens < expected_gpu_tokens:
                logger.info(
                    "[GRANULEKV_PREFIX] phase=tracker_frontier_adjust "
                    "seq_group_id=%s observed_gpu_tokens=%d "
                    "expected_gpu_tokens=%d",
                    seq_group.request_id,
                    observed_gpu_tokens,
                    expected_gpu_tokens,
                )
                self._computed_blocks_tracker.set_num_cached_tokens(
                    seq, expected_gpu_tokens)

            source_blocks = tuple(
                self.block_allocator.allocate_immutable_blocks(
                    prev_block=None,
                    block_token_ids=token_blocks,
                    device=Device.CPU,
                    extra_hash=seq.extra_hash(),
                ))
            mapping = []
            logical_blocks = []
            for logical_index in range(num_gpu_cached_blocks,
                                       num_prefix_blocks):
                source = source_blocks[logical_index]
                target = target_blocks[logical_index]
                assert source.block_id is not None
                assert target.block_id is not None
                mapping.append((
                    self.block_allocator.get_physical_block_id(
                        Device.CPU, source.block_id),
                    self.block_allocator.get_physical_block_id(
                        Device.GPU, target.block_id),
                ))
                logical_blocks.append(
                    LogicalBlockKey(seq.seq_id, logical_index))
        except Exception:
            for source in source_blocks:
                self.block_allocator.free(source)
            if seq.seq_id in self.block_tables:
                self.free(seq)
            else:
                for target in target_blocks:
                    self.block_allocator.free(target)
            raise

        reservation_id = f"block-prefix-restore-{next(self._reservation_counter)}"
        public = BlockPrefixRestoreReservation(
            reservation_id=reservation_id,
            seq_group_id=seq_group.request_id,
            block_mapping=tuple(mapping),
            logical_blocks=tuple(logical_blocks),
            num_prefix_blocks=num_prefix_blocks,
        )
        self._prefix_restore_reservations[reservation_id] = (
            _BlockPrefixRestoreReservationRecord(
                public=public,
                seq=seq,
                source_blocks=source_blocks,
                pending_target_blocks=tuple(pending_targets),
            ))
        return public

    def commit_granulekv_prefix_restore(self, reservation_id: str) -> int:
        """发布 restored prefix，并返回可推进的连续 prefix token 数。"""
        record = self._prefix_restore_reservations.pop(reservation_id)
        self.block_allocator.publish_pending_restore_blocks(
            list(record.pending_target_blocks), Device.GPU)
        self._computed_blocks_tracker.set_num_cached_tokens(
            record.seq,
            record.public.num_prefix_blocks * self.block_size,
        )

        # prefix restore 的 CPU source 是刚完成 DMA 的 SSD clean replica。
        # 它不能在这里直接释放：后续 decode/prefill 未改写这些前缀时，
        # GPU -> SSD swap/store 应复用同一份副本，只写新增的 dirty suffix。
        # 这与普通 swap-in 的所有权转移语义一致；目录在 seq free 或下一次
        # 成功写回时统一回收/替换，避免每个 prefix hit 都重写完整 prompt。
        old_replicas = self._pop_storage_replicas(record.seq.seq_id)
        for replica in old_replicas:
            self.block_allocator.free(replica)
        for logical_index, source_block in enumerate(record.source_blocks):
            key = LogicalBlockKey(record.seq.seq_id, logical_index)
            self._storage_replicas[key] = source_block
        return record.public.num_prefix_blocks * self.block_size

    def finalize_granulekv_prefix_working_set(self, reservation_id: str) -> int:
        """结束一次 one-shot working-set restore，但不发布全局 prefix hash。

        各模型层共用少量环形 HBM regions，forward 结束时早期层已经被覆盖，
        因此这些 block id 绝不能作为“完整 KV 常驻 GPU”的 prefix cache 命中。
        target 保持 pending/hashless 状态，sequence 结束时沿用 allocator 的 pending
        free 路径回收；SSD source 则继续作为干净副本保存在 residency directory。
        """
        record = self._prefix_restore_reservations.pop(reservation_id)
        self._computed_blocks_tracker.set_num_cached_tokens(
            record.seq,
            record.public.num_prefix_blocks * self.block_size,
        )
        old_replicas = self._pop_storage_replicas(record.seq.seq_id)
        for replica in old_replicas:
            self.block_allocator.free(replica)
        for logical_index, source_block in enumerate(record.source_blocks):
            self._storage_replicas[LogicalBlockKey(
                record.seq.seq_id, logical_index)] = source_block
        return record.public.num_prefix_blocks * self.block_size

    def admit_granulekv_prefix_restore_for_layer_barrier(
            self, reservation_id: str) -> int:
        """允许层屏障请求跳过 prefix 重新计算，但暂不发布 hash。

        目标 GPU block 在 reservation 创建时已经进入该 sequence 的 block
        table，且仍被 ``pending_restore`` 集合持有，不会被 prefix lookup 或
        其他请求复用。首个 layer window READY 后，模型可以在每层 barrier
        的保护下使用这些地址；因此这里只推进本请求的 cached-token
        frontier，让原生 chunked prefill 从 suffix 开始。

        特意不调用 ``publish_pending_restore_blocks``：后续 layer 的 DMA
        可能仍在写相同 logical prefix 的其他层，此时提前注册 hash 会把
        未完成 KV 暴露为新的全局 prefix cache 命中。
        """
        record = self._prefix_restore_reservations[reservation_id]
        restored_tokens = record.public.num_prefix_blocks * self.block_size
        self._computed_blocks_tracker.set_num_cached_tokens(
            record.seq, restored_tokens)
        return restored_tokens

    def abort_granulekv_prefix_restore(self, reservation_id: str) -> None:
        """取消 prefix read，并释放临时持有的 SSD source 引用。"""
        record = self._prefix_restore_reservations.pop(reservation_id)
        for source in record.source_blocks:
            self.block_allocator.free(source)

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        """Returns whether we can swap out the given sequence_group 
        with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap out.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            bool: Whether it's possible to swap out current sequence group.
        """
        alloc_status = self._can_swap(seq_group, Device.CPU,
                                      SequenceStatus.RUNNING)
        return alloc_status == AllocStatus.OK

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from GPU to CPU) generated by
        swapping out the given sequence_group with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap out.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from 
                GPU to CPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.GPU,
                                                         dst_device=Device.CPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id)
                for gpu_block_id, cpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][BlockManager] op=swap_out request_id=%s "
                "seqs=%d mappings=%d",
                seq_group.request_id,
                len(seq_group.get_seqs(status=SequenceStatus.RUNNING)),
                len(physical_block_id_mapping),
            )

        return physical_block_id_mapping

    def reserve_swap_out(self,
                         seq_group: SequenceGroup) -> BlockSwapReservation:
        """预留 GPU -> storage，只为 dirty suffix 创建写盘 mapping。"""
        return self._reserve_swap(seq_group, SequenceStatus.RUNNING,
                                  Device.GPU, Device.CPU)

    def reserve_granulekv_prefix_store(
            self, seq_group: SequenceGroup) -> BlockSwapReservation:
        """为正常完成请求建立 GPU -> SSD prefix populate 事务。

        finished sequence 已经不再是 RUNNING，不能复用 ``reserve_swap_out``
        的状态过滤；除此之外，物理 copy、clean replica 复用和 commit 顺序
        与普通 GranuleKV write 完全相同。
        """
        finished = tuple(seq for seq in seq_group.get_seqs()
                         if seq.status in (SequenceStatus.FINISHED_STOPPED,
                                           SequenceStatus.FINISHED_LENGTH_CAPPED))
        if not finished:
            raise ValueError("prefix store requires a normally finished sequence")
        return self._reserve_swap(
            seq_group, None, Device.GPU, Device.CPU,
            complete_only=envs.VLLM_GRANULEKV_HOST_BACKING_ENABLE)

    def can_reserve_granulekv_prefix_store(self,
                                     seq_group: SequenceGroup) -> bool:
        """检查 finished table 是否有足够的 storage 目标 block。"""
        required = 0
        for seq in seq_group.get_seqs():
            source_blocks = tuple(self.block_tables[seq.seq_id].blocks)
            if envs.VLLM_GRANULEKV_HOST_BACKING_ENABLE:
                source_blocks = source_blocks[:
                    seq.data.get_num_computed_tokens() // self.block_size]
            if any(
                    self._pinned_gpu_blocks.get(
                        LogicalBlockKey(seq.seq_id, logical_index), 0)
                    for logical_index in range(len(source_blocks))):
                return False
            required += (len(source_blocks)
                         - self._clean_storage_prefix_length(
                             seq.seq_id, source_blocks))
        return required <= self.get_num_free_cpu_blocks()

    def can_reserve_swap_out(self, seq_group: SequenceGroup) -> bool:
        """判断 storage 空间能否容纳当前 dirty suffix。

        原生 ``can_swap_out`` 按整条 sequence 估算目标 block；有 clean SSD
        replica 后会明显高估空间。异步路径只需要为第一个不匹配 block
        开始的后缀分配新 storage block，因此在这里使用同一口径检查。
        """
        required = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            source_blocks = tuple(self.block_tables[seq.seq_id].blocks)
            if any(
                    self._pinned_gpu_blocks.get(
                        LogicalBlockKey(seq.seq_id, logical_index), 0)
                    for logical_index in range(len(source_blocks))):
                return False
            required += (len(source_blocks)
                         - self._clean_storage_prefix_length(
                             seq.seq_id, source_blocks))
        return required <= self.get_num_free_cpu_blocks()

    def commit_block_swap(self, reservation_id: str) -> None:
        """在后端 I/O 完成后发布目标 block，并更新双副本所有权。

        更新顺序刻意固定为“先让 block table 指向目标对象，再释放源对象”。
        read 的 storage source 会转交给 replica directory 而不是释放；write
        才释放 GPU source。Scheduler 只会在本方法返回后改变状态，因此
        attention 不会观察到未完成的目标 block。
        """
        record = self._get_block_swap_reservation(reservation_id)
        for table_record in record.tables:
            block_table = self.block_tables.get(table_record.seq_id)
            if block_table is None:
                raise RuntimeError(
                    "cannot commit block reservation after table removal: "
                    f"{reservation_id}")
            current_blocks = tuple(block_table.blocks)
            if current_blocks != table_record.source_blocks:
                raise RuntimeError(
                    "source block table changed before reservation commit: "
                    f"{reservation_id}")
        # 所有 source table 都验证通过后才消费记录；校验失败时调用方仍可
        # abort，避免目标 block 因记录提前弹出而泄漏。
        del self._block_swap_reservations[reservation_id]
        is_read = record.public.source_device == Device.CPU
        for table_record in record.tables:
            block_table = self.block_tables[table_record.seq_id]
            block_table.update(list(table_record.target_blocks))
            if is_read:
                # read 后正式表切到 GPU，但 SSD source 逐 block 转交给
                # residency directory。只要 token 内容未变，它就是 clean。
                old_replicas = self._pop_storage_replicas(
                    table_record.seq_id)
                for replica in old_replicas:
                    self.block_allocator.free(replica)
                for logical_index, source_block in enumerate(
                        table_record.source_blocks):
                    key = LogicalBlockKey(table_record.seq_id, logical_index)
                    self._storage_replicas[key] = source_block
            else:
                # write 后 target storage blocks 转交给正式 block table。
                # 复用的 clean block 原来由 directory 持有，删除目录项只
                # 转移所有权；被 dirty block 替换的旧 replica 才真正释放。
                storage_block_ids = [
                    block.block_id for block in table_record.target_blocks
                    if block.block_id is not None
                ]
                # GranuleKV DONE 才意味着 SSD 内容真实可读。只提交本事务的 id，
                # 不能顺带把另一笔仍在飞行的 prefix write 标成 computed。
                self.block_allocator.mark_blocks_as_computed_on_device(
                    storage_block_ids, Device.CPU)
                previous_replicas = self._pop_storage_replicas(
                    table_record.seq_id)
                for source_block in table_record.source_blocks:
                    self.block_allocator.free(source_block)
                retained_ids = {id(block)
                                for block in table_record.target_blocks}
                for replica in previous_replicas:
                    if id(replica) not in retained_ids:
                        self.block_allocator.free(replica)

    def abort_block_swap(self, reservation_id: str) -> None:
        """取消 reservation，只释放未发布的目标 block，源表保持不变。"""
        record = self._pop_block_swap_reservation(reservation_id)
        for table_record in record.tables:
            for target_block in table_record.newly_allocated_targets:
                self.block_allocator.free(target_block)

    def _reserve_swap(
        self,
        seq_group: SequenceGroup,
        sequence_status: Optional[SequenceStatus],
        source_device: Device,
        target_device: Device,
        complete_only: bool = False,
    ) -> BlockSwapReservation:
        """建立一笔双副本 block 事务，并返回后端可直接执行的 mapping。

        目标 block 使用 allocator 的普通公开接口分配，因此 refcount、prefix
        hash 和 block object pool 仍由原生实现维护。发生部分分配失败时，
        已分配目标会在抛出异常前全部回收，源 block table 不受影响。
        """
        table_records: List[_ReservedBlockTable] = []
        physical_mapping: List[Tuple[int, int]] = []
        physical_logical_keys: List[LogicalBlockKey] = []
        reused_blocks = 0
        try:
            sequences = (seq_group.get_seqs(status=sequence_status)
                         if sequence_status is not None else
                         seq_group.get_seqs())
            for seq in sequences:
                source_blocks = tuple(self.block_tables[seq.seq_id].blocks)
                if source_device == Device.GPU:
                    stored_blocks = (source_blocks[:
                        seq.data.get_num_computed_tokens() // self.block_size]
                                     if complete_only else source_blocks)
                    (target_blocks, new_targets,
                     mapping_sources, mapping_targets, logical_keys,
                     reused_for_seq) = self._build_storage_targets(
                         seq.seq_id, stored_blocks, target_device)
                    reused_blocks += reused_for_seq
                else:
                    new_targets = tuple(
                        self._clone_blocks_to_device(source_blocks,
                                                     target_device))
                    target_blocks = new_targets
                    mapping_sources = source_blocks
                    mapping_targets = new_targets
                    logical_keys = [
                        LogicalBlockKey(seq.seq_id, logical_index)
                        for logical_index in range(len(source_blocks))
                    ]
                table_records.append(
                    _ReservedBlockTable(seq_id=seq.seq_id,
                                        source_blocks=source_blocks,
                                        target_blocks=target_blocks,
                                        newly_allocated_targets=new_targets))
                for source_block, target_block in zip(mapping_sources,
                                                      mapping_targets):
                    assert source_block.block_id is not None
                    assert target_block.block_id is not None
                    physical_mapping.append((
                        self.block_allocator.get_physical_block_id(
                            source_device, source_block.block_id),
                        self.block_allocator.get_physical_block_id(
                            target_device, target_block.block_id),
                    ))
                physical_logical_keys.extend(logical_keys)
        except Exception:
            for table_record in table_records:
                for target_block in table_record.newly_allocated_targets:
                    self.block_allocator.free(target_block)
            raise

        reservation_id = f"block-swap-{next(self._reservation_counter)}"
        public = BlockSwapReservation(
            reservation_id=reservation_id,
            seq_group_id=seq_group.request_id,
            source_device=source_device,
            target_device=target_device,
            block_mapping=tuple(physical_mapping),
            num_reused_blocks=reused_blocks,
            logical_blocks=tuple(physical_logical_keys),
        )
        self._block_swap_reservations[reservation_id] = (
            _BlockSwapReservationRecord(public=public,
                                        tables=tuple(table_records)))
        return public

    def _build_storage_targets(
        self,
        seq_id: SeqId,
        source_blocks: Tuple[Block, ...],
        target_device: Device,
    ) -> Tuple[Tuple[Block, ...], Tuple[Block, ...], Tuple[Block, ...],
               Tuple[Block, ...],
               List[LogicalBlockKey], int]:
        """为 GPU->SSD write 复用 clean prefix，只分配 dirty suffix。

        residency 按 block 建索引，但 vLLM Block 的 ``prev_block`` 仍形成链。
        第一个脏块之后重新建立目标链，避免复用一个仍指向旧 predecessor 的
        storage Block。函数内部负责部分失败回收，调用方只接收完整计划。
        """
        targets: List[Block] = []
        new_targets: List[Block] = []
        mapping_sources: List[Block] = []
        mapping_targets: List[Block] = []
        logical_keys: List[LogicalBlockKey] = []
        previous_target: Optional[Block] = None
        reused = 0
        clean_prefix = True
        try:
            for logical_index, source_block in enumerate(source_blocks):
                key = LogicalBlockKey(seq_id, logical_index)
                replica = self._storage_replicas.get(key)
                if (clean_prefix
                        and self._is_clean_storage_replica(key, source_block)):
                    assert replica is not None
                    target_block = replica
                    reused += 1
                else:
                    clean_prefix = False
                    target_block = self._clone_blocks_to_device(
                        (source_block,), target_device,
                        previous_target=previous_target)[0]
                    new_targets.append(target_block)
                    mapping_sources.append(source_block)
                    mapping_targets.append(target_block)
                    logical_keys.append(key)
                targets.append(target_block)
                previous_target = target_block
        except Exception:
            for target in new_targets:
                self.block_allocator.free(target)
            raise
        return (tuple(targets), tuple(new_targets), tuple(mapping_sources),
                tuple(mapping_targets), logical_keys, reused)

    def _clone_blocks_to_device(
        self,
        source_blocks: Tuple[Block, ...],
        target_device: Device,
        previous_target: Optional[Block] = None,
    ) -> List[Block]:
        """按原 token 链在目标设备分配一份尚未发布的 block 对象。"""
        target_blocks: List[Block] = []
        try:
            for source_block in source_blocks:
                extra_hash = getattr(source_block, "extra_hash", None)
                if source_block.is_full:
                    target_block = self.block_allocator.allocate_immutable_block(
                        previous_target,
                        list(source_block.token_ids),
                        target_device,
                        extra_hash=extra_hash,
                    )
                else:
                    target_block = self.block_allocator.allocate_mutable_block(
                        previous_target,
                        target_device,
                        extra_hash=extra_hash,
                    )
                    target_block.append_token_ids(
                        list(source_block.token_ids))
                target_blocks.append(target_block)
                previous_target = target_block
        except Exception:
            for target_block in target_blocks:
                self.block_allocator.free(target_block)
            raise
        return target_blocks

    def _is_clean_storage_replica(self, key: LogicalBlockKey,
                                  source: Block) -> bool:
        """按逻辑位置和内容判断 SSD replica 是否仍与 GPU 一致。"""
        replica = self._storage_replicas.get(key)
        return (replica is not None
                and source.token_ids == replica.token_ids
                and source.extra_hash == replica.extra_hash)

    def _clean_storage_prefix_length(
            self, seq_id: SeqId,
            source_blocks: Tuple[Block, ...]) -> int:
        """在 per-block 目录上计算仍可安全拼接到正式表的 clean prefix。"""
        clean = 0
        for logical_index, source in enumerate(source_blocks):
            if not self._is_clean_storage_replica(
                    LogicalBlockKey(seq_id, logical_index), source):
                break
            clean += 1
        return clean

    def _pop_storage_replicas(self, seq_id: SeqId) -> Tuple[Block, ...]:
        """移除一条 sequence 的目录项，并把 block 所有权返回给调用方。"""
        keys = sorted(key for key in self._storage_replicas
                      if key.seq_id == seq_id)
        return tuple(self._storage_replicas.pop(key) for key in keys)

    def get_num_clean_storage_replicas(self, seq: Sequence) -> int:
        """返回指定 GPU sequence 当前仍有效的 clean SSD block 数。"""
        blocks = tuple(self.block_tables[seq.seq_id].blocks)
        return sum(
            self._is_clean_storage_replica(
                LogicalBlockKey(seq.seq_id, logical_index), block)
            for logical_index, block in enumerate(blocks))

    def get_block_residency(self,
                            seq: Sequence) -> Tuple[BlockResidency, ...]:
        """返回稳定、只读的 per-block residency 快照供调度策略使用。"""
        blocks = tuple(self.block_tables[seq.seq_id].blocks)
        result = []
        for logical_index, block in enumerate(blocks):
            key = LogicalBlockKey(seq.seq_id, logical_index)
            replica = self._storage_replicas.get(key)
            storage_block_id = None
            if replica is not None:
                assert replica.block_id is not None
                storage_block_id = self.block_allocator.get_physical_block_id(
                    Device.CPU, replica.block_id)
            result.append(
                BlockResidency(
                    key=key,
                    storage_block_id=storage_block_id,
                    storage_replica_clean=self._is_clean_storage_replica(
                        key, block),
                    pin_count=self._pinned_gpu_blocks.get(key, 0),
                ))
        return tuple(result)

    def pin_blocks(self, seq: Sequence,
                   logical_indices: GenericSequence[int]) -> None:
        """阻止选中的 GPU block 被异步 swap-out，支持嵌套 pin。"""
        block_count = len(self.block_tables[seq.seq_id].blocks)
        for logical_index in logical_indices:
            if logical_index < 0 or logical_index >= block_count:
                raise IndexError(f"logical block index out of range: "
                                 f"{logical_index}")
            key = LogicalBlockKey(seq.seq_id, logical_index)
            self._pinned_gpu_blocks[key] = (
                self._pinned_gpu_blocks.get(key, 0) + 1)

    def unpin_blocks(self, seq: Sequence,
                     logical_indices: GenericSequence[int]) -> None:
        """释放一次 block pin；未 pin 或重复 unpin 直接报错。"""
        for logical_index in logical_indices:
            key = LogicalBlockKey(seq.seq_id, logical_index)
            count = self._pinned_gpu_blocks.get(key, 0)
            if count <= 0:
                raise RuntimeError(f"logical block is not pinned: {key}")
            if count == 1:
                del self._pinned_gpu_blocks[key]
            else:
                self._pinned_gpu_blocks[key] = count - 1

    def _pop_block_swap_reservation(
            self, reservation_id: str) -> _BlockSwapReservationRecord:
        record = self._block_swap_reservations.pop(reservation_id, None)
        if record is None:
            raise KeyError(f"unknown block reservation: {reservation_id}")
        return record

    def _get_block_swap_reservation(
            self, reservation_id: str) -> _BlockSwapReservationRecord:
        record = self._block_swap_reservations.get(reservation_id)
        if record is None:
            raise KeyError(f"unknown block reservation: {reservation_id}")
        return record

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        return self.block_allocator.reset_prefix_cache(device)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in/out.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        # First determine the number of blocks that will be touched by this
        # swap. Then verify if there are available blocks in the device
        # to perform the swap.
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks is not None:
                # Compute the number blocks to touch for the tokens to be
                # appended. This does NOT include the full blocks that need
                # to be touched for the swap.
                num_blocks_touched += \
                    block_table.get_num_blocks_touched_by_append_slots(
                        block_table.get_unseen_token_ids(seq.get_token_ids()),
                        num_lookahead_slots=num_lookahead_slots)
                blocks.extend(block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)

        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        """Get the number of tokens in blocks that are already computed and
        cached in the block manager for the sequence.
        """
        return self._computed_blocks_tracker.get_num_cached_tokens(seq)
