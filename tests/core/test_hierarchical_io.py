# SPDX-License-Identifier: Apache-2.0

"""层级 I/O plan 和父事务屏障的纯控制面测试。"""

import pytest

from vllm.core.block_reservation import LogicalBlockKey
from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferOperation, AsyncKVTransferPriority,
    AsyncKVTransferRequest, AsyncKVTransferState)
from vllm.core.custom_schedulers.hierarchical_io import (
    HierarchicalIOConfig, HierarchicalLayerBarrierConfig,
    HierarchicalRestoreController, PrefetchBlockSelectorConfig, PrefetchUnit,
    RollingPrefetchConfig, RollingPrefetchRuntime, SparseKVAccessPlan,
    activate_layer_barrier, activate_sparse_kv_blocks,
    build_layer_restore_plan, get_active_layer_sequence_lengths,
    get_active_sparse_kv_blocks,
    get_layer_working_set_regions, release_local_layer,
    select_prefetch_unit_blocks,
    wait_for_local_layer)


def test_layer_restore_plan_is_contiguous_and_default_disabled():
    assert not HierarchicalIOConfig.from_env({}).enabled

    plan = build_layer_restore_plan(plan_id="restore-1",
                                    num_layers=10,
                                    window_layers=4,
                                    created_monotonic_ns=100)
    assert [unit.layer_range for unit in plan.units] == [
        (0, 4), (4, 8), (8, 10)
    ]
    assert [unit.index for unit in plan.units] == [0, 1, 2]
    assert all(unit.block_indices is None for unit in plan.units)


def test_exact_restore_is_opt_in_in_hierarchical_config():
    config = HierarchicalIOConfig.from_env({
        "VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE": "1",
        "VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS": "4",
        "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS": "2",
        "VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE": "1",
    })
    assert config.exact_restore
    assert not HierarchicalIOConfig.from_env({}).exact_restore


def test_prefetch_unit_projects_dense_and_sparse_block_sets():
    """同一个接口应同时表达 layer 全量读取和未来 sparse 子集读取。"""
    mapping = ((10, 20), (11, 21), (12, 22), (13, 23))
    logical_blocks = tuple(LogicalBlockKey(7, index) for index in range(4))

    dense = PrefetchUnit(index=0, start_layer=0, end_layer=2)
    assert select_prefetch_unit_blocks(dense, mapping,
                                       logical_blocks) == (mapping,
                                                           logical_blocks)

    sparse = PrefetchUnit(index=1,
                          start_layer=2,
                          end_layer=4,
                          block_indices=(0, 2, 3))
    selected_mapping, selected_keys = select_prefetch_unit_blocks(
        sparse, mapping, logical_blocks)
    assert selected_mapping == ((10, 20), (12, 22), (13, 23))
    assert [key.logical_index for key in selected_keys] == [0, 2, 3]


def test_prefetch_unit_rejects_ambiguous_or_invalid_block_selection():
    with pytest.raises(ValueError, match="strictly increasing"):
        PrefetchUnit(index=0,
                     start_layer=0,
                     end_layer=2,
                     block_indices=(1, 1))
    unit = PrefetchUnit(index=0,
                        start_layer=0,
                        end_layer=2,
                        block_indices=(2, ))
    # logical block 2 可能已经在 HBM，因此不出现在 SSD mapping 中是合法的；
    # 投影为空只表示当前 unit 无需为它发起物理 read。
    assert select_prefetch_unit_blocks(
        unit, ((10, 20), ), (LogicalBlockKey(7, 0), )) == ((), ())


def test_sparse_prefetch_selector_builds_profiling_only_plan():
    """selector 只改变 block 投影，不改变 layer unit 生命周期。"""
    config = PrefetchBlockSelectorConfig.from_env({
        "VLLM_GRANULEKV_PREFETCH_BLOCK_SELECTOR": "tail_n",
        "VLLM_GRANULEKV_PREFETCH_BLOCK_COUNT": "2",
    })
    plan = build_layer_restore_plan(plan_id="restore-sparse",
                                    num_layers=6,
                                    window_layers=2,
                                    block_selector=config,
                                    num_blocks=5,
                                    created_monotonic_ns=100)
    assert plan.profiling_only
    assert plan.block_selector == "tail_n"
    assert [unit.layer_range for unit in plan.units] == [
        (0, 2), (2, 4), (4, 6)
    ]
    assert [unit.block_indices for unit in plan.units] == [(3, 4)] * 3


def test_stride_prefetch_selector_is_deterministic():
    config = PrefetchBlockSelectorConfig.from_env({
        "VLLM_GRANULEKV_PREFETCH_BLOCK_SELECTOR": "stride",
        "VLLM_GRANULEKV_PREFETCH_BLOCK_STRIDE": "2",
    })
    assert config.select(5) == (0, 2, 4)


def test_sparse_access_plan_keeps_per_layer_selection_and_window_union():
    access = SparseKVAccessPlan(
        num_layers=4,
        num_blocks=8,
        block_indices_by_layer=((0, 2), (1, 2), (4, ), (4, 7)),
        source="attention-mask",
    )
    plan = build_layer_restore_plan(
        plan_id="sparse-layer-plan",
        num_layers=4,
        window_layers=2,
        access_plan=access,
        created_monotonic_ns=100,
    )
    assert plan.units[0].block_indices == (0, 1, 2)
    assert plan.consumer_blocks_for_layer(0) == (0, 2)
    assert plan.consumer_blocks_for_layer(1) == (1, 2)
    assert plan.units[1].block_indices == (4, 7)


def test_exact_restore_groups_only_adjacent_identical_layer_selections():
    access = SparseKVAccessPlan(
        num_layers=4,
        num_blocks=8,
        block_indices_by_layer=((0, 1), (0, 1), (0, 2), (0, 2)),
        source="solidattention_dynamic",
    )
    plan = build_layer_restore_plan(
        plan_id="solidattention-exact",
        num_layers=4,
        window_layers=3,
        access_plan=access,
        exact_grouped=True,
        created_monotonic_ns=100,
    )

    assert plan.exact_grouped
    assert plan.restore_mode == "exact_grouped"
    assert [unit.layer_range for unit in plan.units] == [(0, 2), (2, 4)]
    assert [unit.block_indices for unit in plan.units] == [(0, 1), (0, 2)]
    assert plan.units[0].consumer_blocks_by_layer == ((0, 1), (0, 1))
    assert plan.units[1].consumer_blocks_by_layer == ((0, 2), (0, 2))


def test_exact_restore_respects_window_limit_and_covers_all_layers():
    access = SparseKVAccessPlan(
        num_layers=5,
        num_blocks=4,
        block_indices_by_layer=((1, 2), ) * 5,
        source="solidattention_dynamic",
    )
    plan = build_layer_restore_plan(
        plan_id="solidattention-exact-window",
        num_layers=5,
        window_layers=2,
        access_plan=access,
        exact_grouped=True,
    )

    assert [unit.layer_range for unit in plan.units] == [(0, 2), (2, 4),
                                                          (4, 5)]
    assert plan.units[-1].block_indices == (1, 2)
    assert sum(unit.num_layers for unit in plan.units) == 5


def test_sparse_access_plan_normalizes_dynamic_policy_output():
    access = SparseKVAccessPlan.from_layer_selections(
        num_blocks=6,
        block_indices_by_layer=((4, 1, 1), (2, 5)),
        source="quest_dynamic",
    )
    assert access.block_indices_by_layer == ((1, 4), (2, 5))
    with pytest.raises(ValueError, match="outside"):
        SparseKVAccessPlan.from_layer_selections(
            num_blocks=4,
            block_indices_by_layer=((0, 4), ),
        )


def test_sparse_consumer_mode_is_explicit_and_preserves_profiling_default():
    access = SparseKVAccessPlan(
        num_layers=2,
        num_blocks=4,
        block_indices_by_layer=((0, 3), (1, 3)),
        source="quest_query_aware",
    )
    profiling = build_layer_restore_plan(
        plan_id="sparse-profiling",
        num_layers=2,
        window_layers=1,
        access_plan=access,
    )
    assert profiling.profiling_only
    assert not profiling.consumer_enabled

    consumer = build_layer_restore_plan(
        plan_id="sparse-consumer",
        num_layers=2,
        window_layers=1,
        access_plan=access,
        consumer_enabled=True,
    )
    assert consumer.profiling_only
    assert consumer.consumer_enabled

    dense_consumer = build_layer_restore_plan(
        plan_id="dense-warmup-consumer",
        num_layers=2,
        window_layers=1,
        consumer_enabled=True,
    )
    assert dense_consumer.consumer_enabled
    assert not dense_consumer.profiling_only


def test_sparse_mapping_projects_logical_indices_not_mapping_offsets():
    mapping = ((101, 201), (103, 203))
    logical_blocks = tuple(LogicalBlockKey(7, index) for index in (1, 3))
    unit = PrefetchUnit(index=0,
                        start_layer=0,
                        end_layer=2,
                        block_indices=(0, 3))
    selected_mapping, selected_keys = select_prefetch_unit_blocks(
        unit, mapping, logical_blocks)
    assert selected_mapping == ((103, 203), )
    assert [key.logical_index for key in selected_keys] == [3]


def test_rolling_config_is_explicit_and_validated():
    config = RollingPrefetchConfig.from_env({
        "VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE": "1",
        "VLLM_GRANULEKV_HIERARCHICAL_LEAD_WINDOWS": "2",
        "VLLM_GRANULEKV_HIERARCHICAL_MAX_LEAD_WINDOWS": "3",
        "VLLM_GRANULEKV_HIERARCHICAL_TARGET_SLACK_MS": "0.5",
    })
    assert config.enabled
    assert config.initial_units == 2
    assert config.max_lead_units == 3


def test_layer_working_set_regions_cover_current_and_max_lead_windows():
    values = {
        "VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE": "1",
        "VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE": "1",
        "VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER": "1",
        "VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE": "1",
        "VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS": "28",
        "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS": "2",
        "VLLM_GRANULEKV_HIERARCHICAL_LEAD_WINDOWS": "1",
        "VLLM_GRANULEKV_HIERARCHICAL_MAX_LEAD_WINDOWS": "2",
    }
    assert get_layer_working_set_regions(values) == 6
    assert RollingPrefetchConfig.from_env(values).working_set_enabled


def test_layer_barrier_reports_consumption_after_wait():
    events = []
    with activate_layer_barrier(
            lambda engine, request_ids, layer: events.append(
                ("wait", engine, tuple(request_ids), layer)),
            virtual_engine=0,
            request_ids=("request-0", ),
            release_callback=lambda engine, request_ids, layer: events.append(
                ("release", engine, tuple(request_ids), layer))):
        wait_for_local_layer(3)
        release_local_layer(3)
    assert [event[0] for event in events] == ["wait", "release"]


def _prefetch_request(index: int, *, activate: bool) -> AsyncKVTransferRequest:
    return AsyncKVTransferRequest(
        request_id=f"unit-{index}",
        seq_group_id="seq-0",
        reservation_id="plan-0",
        operation=AsyncKVTransferOperation.READ,
        block_mapping=((10, 20), ),
        logical_blocks=(),
        priority=AsyncKVTransferPriority.CRITICAL_READ,
        layer_range=(index * 2, (index + 1) * 2),
        prefetch_plan_id="plan-0",
        prefetch_unit_index=index,
        activate_on_submit=activate,
    )


def _sparse_prefetch_request() -> AsyncKVTransferRequest:
    return AsyncKVTransferRequest(
        request_id="sparse-unit-0",
        seq_group_id="seq-sparse",
        reservation_id="plan-sparse",
        operation=AsyncKVTransferOperation.READ,
        block_mapping=((10, 20), ),
        logical_blocks=(LogicalBlockKey(9, 3), ),
        priority=AsyncKVTransferPriority.CRITICAL_READ,
        layer_range=(0, 2),
        prefetch_plan_id="plan-sparse",
        prefetch_unit_index=0,
        consumer_block_indices=(0, 3),
        consumer_blocks_by_layer=((0, ), (3, )),
    )


def test_rolling_runtime_activates_future_unit_from_model_progress():
    runtime = RollingPrefetchRuntime(
        RollingPrefetchConfig(enabled=True,
                              lead_units=1,
                              max_lead_units=1))
    activated = []
    ready = set()

    def submit(request, mapping):
        activated.append((request.request_id, mapping))
        return AsyncKVTransferEvent(request.request_id,
                                    AsyncKVTransferState.PENDING)

    def poll(request):
        state = (AsyncKVTransferState.READY
                 if request.request_id in ready else
                 AsyncKVTransferState.PENDING)
        return AsyncKVTransferEvent(request.request_id, state)

    assert runtime.submit_or_stage(0, _prefetch_request(0, activate=True),
                                   "mapping-0", submit)[0].state == (
                                       AsyncKVTransferState.PENDING)
    assert not runtime.submit_or_stage(0,
                                       _prefetch_request(1, activate=False),
                                       "mapping-1", submit)
    assert activated == [("unit-0", "mapping-0")]

    ready.update(("unit-0", "unit-1"))
    runtime.wait_ready(0, ("seq-0", ), 0, submit, poll, max_active=2)
    assert activated == [("unit-0", "mapping-0"),
                         ("unit-1", "mapping-1")]
    events = runtime.poll_units(0, poll)
    assert [(event.request_id, event.state) for event in events] == [
        ("unit-1", AsyncKVTransferState.PENDING),
        ("unit-0", AsyncKVTransferState.READY),
        ("unit-1", AsyncKVTransferState.READY),
    ]
    traces = runtime.pop_traces()
    assert {(trace.phase, trace.unit_index) for trace in traces} >= {
        ("activated", 0),
        ("activated", 1),
        ("physical_ready", 0),
        ("physical_ready", 1),
        ("barrier_ready", 0),
    }


def test_working_set_unit_is_rejected_after_consumption():
    runtime = RollingPrefetchRuntime(
        RollingPrefetchConfig(enabled=True,
                              lead_units=0,
                              max_lead_units=0,
                              working_set_enabled=True))
    request = _prefetch_request(0, activate=True)

    def ready(current, *_args):
        return AsyncKVTransferEvent(current.request_id,
                                    AsyncKVTransferState.READY)

    runtime.submit_or_stage(0, request, "mapping", ready)
    runtime.wait_ready(0, ("seq-0", ), 0, ready, ready, max_active=1)
    runtime.release_layer(0, ("seq-0", ), 1)
    with pytest.raises(RuntimeError, match="already evicted"):
        runtime.wait_ready(0, ("seq-0", ), 0, ready, ready, max_active=1)


def test_sparse_residency_is_checked_before_layer_consumption():
    runtime = RollingPrefetchRuntime(RollingPrefetchConfig())
    request = _sparse_prefetch_request()

    runtime.submit_or_stage(
        0, request, "mapping",
        lambda _request, *_args: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.PENDING))
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.require_resident_layer(("seq-sparse", ), 0)

    runtime.wait_ready(
        0,
        ("seq-sparse", ),
        0,
        lambda _request, *_args: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.PENDING),
        lambda _request: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.READY),
        max_active=1,
    )
    assert runtime.require_resident_layer(("seq-sparse", ), 0) == (0, )
    assert runtime.require_resident_layer(("seq-sparse", ), 1) == (3, )
    runtime.forget_seq_groups(("seq-sparse", ))
    # request 完成后目录被清理；无匹配 plan 时自然退化为 dense/no-op。
    assert runtime.require_resident_layer(("seq-sparse", ), 0) is None


def test_sparse_residency_does_not_expand_selected_unit_to_all_blocks():
    runtime = RollingPrefetchRuntime(RollingPrefetchConfig())
    request = AsyncKVTransferRequest(
        request_id="sparse-subset",
        seq_group_id="seq-subset",
        reservation_id="plan-subset",
        operation=AsyncKVTransferOperation.READ,
        block_mapping=((10, 20), ),
        logical_blocks=(LogicalBlockKey(4, 1), ),
        priority=AsyncKVTransferPriority.CRITICAL_READ,
        layer_range=(0, 1),
        prefetch_plan_id="plan-subset",
        prefetch_unit_index=0,
        consumer_block_indices=(1, ),
        consumer_blocks_by_layer=((1, ), ),
        consumer_num_blocks=4,
    )

    runtime.submit_or_stage(
        0, request, "mapping",
        lambda _request, *_args: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.PENDING))
    runtime.wait_ready(
        0,
        ("seq-subset", ),
        0,
        lambda _request, *_args: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.PENDING),
        lambda _request: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.READY),
        max_active=1,
    )
    assert runtime.require_resident_layer(("seq-subset", ), 0) == (1, )


def test_sparse_residency_miss_is_reported_separately_from_prediction():
    runtime = RollingPrefetchRuntime(RollingPrefetchConfig())
    request = _sparse_prefetch_request()
    runtime.submit_or_stage(
        0, request, "mapping",
        lambda _request, *_args: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.PENDING))
    runtime.wait_ready(
        0,
        ("seq-sparse", ),
        0,
        lambda _request, *_args: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.PENDING),
        lambda _request: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.READY),
        max_active=1,
    )
    runtime.residency.evict_unit(request.request_id)
    with pytest.raises(RuntimeError, match="evicted"):
        runtime.require_resident_layer(("seq-sparse", ), 0)
    assert runtime.residency_stats() == {
        "residency_miss": 1,
        "residency_miss_blocks": 2,
    }


def test_sparse_residency_adds_live_suffix_blocks_from_current_length():
    runtime = RollingPrefetchRuntime(RollingPrefetchConfig())
    prefix_blocks = tuple(range(16))
    request = AsyncKVTransferRequest(
        request_id="sparse-suffix",
        seq_group_id="seq-suffix",
        reservation_id="plan-suffix",
        operation=AsyncKVTransferOperation.READ,
        block_mapping=tuple((index, index + 100) for index in prefix_blocks),
        logical_blocks=tuple(LogicalBlockKey(4, index)
                             for index in prefix_blocks),
        priority=AsyncKVTransferPriority.CRITICAL_READ,
        layer_range=(0, 1),
        prefetch_plan_id="plan-suffix",
        prefetch_unit_index=0,
        consumer_block_indices=prefix_blocks,
        consumer_blocks_by_layer=(prefix_blocks, ),
        consumer_num_blocks=18,
        consumer_local_block_start=16,
    )

    runtime.submit_or_stage(
        0, request, "mapping",
        lambda _request, *_args: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.PENDING))
    runtime.wait_ready(
        0,
        ("seq-suffix", ),
        0,
        lambda _request, *_args: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.PENDING),
        lambda _request: AsyncKVTransferEvent(
            request.request_id, AsyncKVTransferState.READY),
        max_active=1,
    )

    assert runtime.require_resident_layer(
        ("seq-suffix", ), 0, {"seq-suffix": 288}, 16) == tuple(range(18))


def test_first_window_admission_is_separate_from_full_restore():
    plan = build_layer_restore_plan(plan_id="restore-2",
                                    num_layers=8,
                                    window_layers=2,
                                    created_monotonic_ns=100)
    controller = HierarchicalRestoreController()
    controller.register(plan, ["w0", "w1", "w2", "w3"])

    # 后续 window 可以乱序完成，但不能冒充 first-window-ready admission。
    progress = controller.mark_ready("w2", now_monotonic_ns=150)
    assert not progress.first_unit_ready
    assert not progress.all_terminal

    progress = controller.mark_ready("w0", now_monotonic_ns=200)
    assert progress.first_unit_became_ready
    assert progress.first_unit_ready
    assert progress.first_unit_ready_monotonic_ns == 200
    assert not progress.all_terminal

    controller.mark_ready("w1", now_monotonic_ns=250)
    progress = controller.mark_ready("w3", now_monotonic_ns=300)
    assert progress.all_terminal
    assert progress.all_ready
    assert controller.release(plan.plan_id) == plan


def test_window_error_prevents_parent_publish_until_all_terminal():
    plan = build_layer_restore_plan(plan_id="restore-3",
                                    num_layers=4,
                                    window_layers=2)
    controller = HierarchicalRestoreController()
    controller.register(plan, ["w0", "w1"])

    failed = controller.mark_error("w1")
    assert failed.failed
    assert not failed.all_terminal
    ready = controller.mark_ready("w0")
    assert ready.all_terminal
    assert not ready.all_ready
    assert ready.failed
    controller.release(plan.plan_id)


def test_layer_barrier_is_forward_scoped_and_default_disabled():
    assert not HierarchicalLayerBarrierConfig.from_env({}).enabled
    assert HierarchicalLayerBarrierConfig.from_env({
        "VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER": "1"
    }).enabled

    observed = []
    with activate_layer_barrier(
            lambda virtual_engine, request_ids, layer: observed.append(
                (virtual_engine, tuple(request_ids), layer)),
            virtual_engine=3,
            request_ids=("request-a", "request-b"),
            sequence_lengths_by_request={"request-a": 272},
            block_size=16):
        wait_for_local_layer(5)
        assert get_active_layer_sequence_lengths() == {"request-a": 272}

    # context 退出后模型层调用必须自然退化为 no-op，避免状态泄漏到下一 batch。
    wait_for_local_layer(6)
    assert observed == [(3, ("request-a", "request-b"), 5)]


def test_sparse_consumer_blocks_are_forward_scoped():
    assert get_active_sparse_kv_blocks() is None
    with activate_sparse_kv_blocks((1, 4, 7)):
        assert get_active_sparse_kv_blocks() == (1, 4, 7)
    assert get_active_sparse_kv_blocks() is None
