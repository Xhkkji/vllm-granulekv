"""Tests for strategy composition; no native I/O or model execution."""

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferOperation, AsyncKVTransferQueue)
from vllm.core.block_reservation import LogicalBlockKey

from evaluation.paper_reproduction.common.plans import (
    LayerWisePrefetcher, RequestWorkItem, compose_plan,
    enqueue_plan_requests)
from evaluation.paper_reproduction.common.schedulers import (
    DiskHRRNScheduler, FCFSScheduler)
from vllm.granulekv.sparse_policy.selectors import (
    DenseSelector, ExplicitSelector)


def test_composition_reuses_production_layer_plan():
    plan = compose_plan(
        plan_id="test-dense",
        selector=DenseSelector(),
        prefetcher=LayerWisePrefetcher(2),
        scheduler=FCFSScheduler(),
        num_layers=4,
        num_blocks=8,
    )
    assert plan.prefetch_plan.block_selector == "dense"
    assert [unit.layer_range for unit in plan.prefetch_plan.units] == [(0, 2),
                                                                        (2, 4)]
    assert not plan.prefetch_plan.profiling_only


def test_sparse_units_project_into_one_common_request_model():
    selector = ExplicitSelector(((0, 2), (0, 2), (1, 2), (1, 2)), "test")
    plan = compose_plan(
        plan_id="test-sparse",
        selector=selector,
        prefetcher=LayerWisePrefetcher(2),
        scheduler=FCFSScheduler(),
        num_layers=4,
        num_blocks=4,
    )
    queue = AsyncKVTransferQueue(max_in_flight=2)
    requests = enqueue_plan_requests(
        queue,
        plan,
        seq_group_id="seq-1",
        reservation_id="reservation-1",
        block_mapping=((10, 20), (11, 21), (12, 22)),
        logical_blocks=tuple(LogicalBlockKey("seq-1", index)
                             for index in range(3)),
        operation=AsyncKVTransferOperation.READ,
    )
    assert len(requests) == 2
    assert all(request.prefetch_plan_id == "test-sparse"
               for request in requests)
    assert requests[0].layer_range == (0, 2)
    assert requests[0].block_mapping == ((10, 20), (12, 22))
    assert requests[1].block_mapping == ((11, 21), (12, 22))


def test_scheduler_orders_descriptions_without_owning_transfer_state():
    items = (
        RequestWorkItem("slow", useful_bytes=100,
                        estimated_service_us=100, ready_deadline_us=0),
        RequestWorkItem("urgent", useful_bytes=10,
                        estimated_service_us=10, ready_deadline_us=100),
    )
    assert [item.request_id for item in FCFSScheduler().order(items)] == [
        "slow", "urgent"
    ]
    assert DiskHRRNScheduler().order(items)[0].request_id == "urgent"
