from types import SimpleNamespace

import pytest
import torch

from vllm.attention.backends.xformers import XFormersImpl


def _make_impl(*, num_kv_heads: int = 2, head_size: int = 4):
    """构造一个只服务 helper 测试的极简 XFormersImpl。

    这些用例只验证 prefix fallback 的控制面 helper，不需要真的初始化完整
    attention backend，也不需要跑真实 CUDA/xFormers forward。
    因此这里直接用 `__new__` 构造实例，只补齐 helper 会访问到的最小属性，
    让测试更聚焦、更稳定。
    """
    impl = XFormersImpl.__new__(XFormersImpl)
    impl.num_kv_heads = int(num_kv_heads)
    impl.head_size = int(head_size)
    impl.sliding_window = None
    return impl


def _make_layer(layer_name: str = "model.layers.0.self_attn.attn"):
    """构造只包含 `layer_name` 的极简 attention layer 桩。"""
    return SimpleNamespace(layer_name=layer_name)


def _make_prefill_meta(
    *,
    context_lens: list[int],
    query_start_loc: list[int],
    block_tables: list[list[int]],
):
    """构造 prefix fallback helper 需要的最小 metadata。"""
    return SimpleNamespace(
        context_lens_tensor=torch.tensor(context_lens, dtype=torch.int32),
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        num_prefills=len(context_lens),
        block_tables=torch.tensor(block_tables, dtype=torch.int32),
        _cached_prefix_fallback_plan=None,
        _cached_prefix_fallback_plan_key=None,
        _cached_prefix_fallback_workspace=None,
        _cached_prefix_fallback_workspace_key=None,
    )


def test_prefix_fallback_plan_builds_gpu_resident_query_positions():
    """plan 应同时显式给出 prefix/query 在 full KV 中的最终位置。

    这条测试保护的是这次新推进的第二步：

    - prefix 侧已经有 `context_full_positions`
    - query 侧现在也补上 `query_full_positions`

    这样后续无论是 direct scatter，还是更进一步接 persistent kernel /
    service CTA，都不需要再每层按 segment 重新做 Python 切片规划。
    """
    impl = _make_impl()
    prefill_meta = _make_prefill_meta(
        context_lens=[3, 2],
        query_start_loc=[0, 2, 5],
        block_tables=[
            [10, 11],
            [20, 21],
        ],
    )

    plan = impl._get_prefix_fallback_plan(
        prefill_meta=prefill_meta,
        device=torch.device("cpu"),
        block_size=4,
    )

    assert plan.query_lens == (2, 3)
    assert plan.context_lens == (3, 2)
    assert plan.kv_lens == (5, 5)
    assert plan.total_context_tokens == 5
    assert plan.total_query_tokens == 5
    assert plan.total_kv_tokens == 10

    # request 0:
    #   prefix -> full[0,1,2]
    #   query  -> full[3,4]
    # request 1:
    #   prefix -> full[5,6]
    #   query  -> full[7,8,9]
    assert plan.context_full_positions.tolist() == [0, 1, 2, 5, 6]
    assert plan.query_full_positions.tolist() == [3, 4, 7, 8, 9]

    # prefix token 到 physical block / block 内 offset 的映射也应保持正确。
    assert plan.context_block_ids.tolist() == [10, 10, 10, 20, 20]
    assert plan.context_block_offsets.tolist() == [0, 1, 2, 0, 1]


@pytest.mark.parametrize("context,query_tokens,selected", [
    (19, 1, (0, 2, 4)), (19, 9, (0, 2, 4, 5, 6)), (20, 9, (0, 2, 4, 5, 6, 7)),
])
def test_sparse_history_plan_excludes_gaps_and_current_chunk(context, query_tokens, selected):
    impl = _make_impl()
    table = [7, 3, 6, 1, 5, 0, 2, 4]
    meta = _make_prefill_meta(context_lens=[context],
                              query_start_loc=[0, query_tokens], block_tables=[table])
    meta.seq_lens = [context + query_tokens]
    meta.num_prefill_tokens = query_tokens
    plan = impl._get_prefix_fallback_plan(
        prefill_meta=meta, device=torch.device("cpu"), block_size=4,
        selected_blocks=selected, selected_indices=torch.tensor(selected))
    positions = [position for block in selected
                 for position in range(block * 4, min(context, (block + 1) * 4))]
    assert plan.context_block_ids.tolist() == [table[position // 4] for position in positions]
    assert plan.context_block_offsets.tolist() == [position % 4 for position in positions]
    assert plan.total_context_tokens == len(positions)
    assert plan.total_kv_tokens == len(positions) + query_tokens
    assert plan.query_full_positions.tolist() == list(range(len(positions), len(positions) + query_tokens))
    mask = plan.attn_bias.materialize((query_tokens, plan.total_kv_tokens))
    expected = torch.arange(plan.total_kv_tokens)[None, :] <= (
        len(positions) + torch.arange(query_tokens)[:, None])
    assert torch.equal(torch.isfinite(mask), expected)


def test_query_workspace_backend_prefers_direct_scatter(
    monkeypatch: pytest.MonkeyPatch,
):
    """当 direct scatter 条件满足时，应优先走新的 GPU 路径。

    这条测试不跑真实 Triton kernel，只保护新的 backend 选择与分发语义：

    - `_can_use_direct_query_scatter()` 返回 True
    - selector 应明确选出 `direct_scatter`
    - materialize helper 应调用 direct scatter helper
    - 不再继续执行旧的 `segment_copy` 路径
    """
    impl = _make_impl()
    observed = {"scatter_called": 0}

    monkeypatch.setattr(
        impl,
        "_can_use_direct_query_scatter",
        lambda query_key, query_value, full_key, full_value: True,
    )

    def _fake_scatter(**kwargs):
        del kwargs
        observed["scatter_called"] += 1

    monkeypatch.setattr(
        impl,
        "_scatter_query_kv_into_full_buffer_direct",
        _fake_scatter,
    )

    plan = SimpleNamespace(
        segments=(
            SimpleNamespace(
                full_query_start=1,
                full_query_end=3,
                query_start=0,
                query_end=2,
            ),
        ),
        total_query_tokens=2,
        query_full_positions=torch.tensor([1, 2], dtype=torch.int32),
    )
    query_key = torch.arange(2 * 2 * 4, dtype=torch.float32).view(2, 2, 4)
    query_value = torch.arange(2 * 2 * 4, dtype=torch.float32).view(2, 2, 4)
    full_key = torch.zeros(4, 2, 4, dtype=torch.float32)
    full_value = torch.zeros(4, 2, 4, dtype=torch.float32)

    backend_choice = impl._select_prefix_workspace_backends(
        layer=_make_layer(),
        prefill_meta=SimpleNamespace(),
        plan=plan,
        key_cache=full_key,
        value_cache=full_value,
        query_key=query_key,
        query_value=query_value,
        full_key=full_key,
        full_value=full_value,
    )
    assert backend_choice.query_backend == "direct_scatter"

    query_mode, _ = impl._materialize_query_kv_into_workspace(
        query_backend=backend_choice.query_backend,
        query_key=query_key,
        query_value=query_value,
        plan=plan,
        full_key=full_key,
        full_value=full_value,
        profile_enabled=False,
    )

    assert query_mode == "direct_scatter"
    assert observed["scatter_called"] == 1
    assert torch.count_nonzero(full_key).item() == 0
    assert torch.count_nonzero(full_value).item() == 0


def test_dense_prefix_attachment_no_longer_overrides_paged_cache_mainline():
    """存在 dense prefix attachment 时，也不应再覆盖 paged-cache 主线。

    当前主线已经收束为：

    - GPU 后台把 prefix 直接写进最终 paged KV cache
    - xFormers fallback 统一从 paged cache 消费 prefix

    因此 dense attachment 现在只保留为调试/对照数据源，不再参与默认 backend
    选择。这样 attention 数据面就不会再额外依赖一套旁路 dense 解释语义。
    """
    impl = _make_impl()
    plan = SimpleNamespace(
        query_lens=(3, ),
        total_context_tokens=4,
    )
    prefill_meta = SimpleNamespace(
        _granulekv_dense_prefix_chunk_tensors=(
            torch.zeros(2, 1, 4, 8, dtype=torch.float16),
        ),
    )
    backend_choice = impl._select_prefix_workspace_backends(
        layer=_make_layer(),
        prefill_meta=prefill_meta,
        plan=plan,
        key_cache=torch.zeros(1, dtype=torch.float16),
        value_cache=torch.zeros(1, dtype=torch.float16),
        query_key=torch.zeros(1, dtype=torch.float16),
        query_value=torch.zeros(1, dtype=torch.float16),
        full_key=torch.zeros(1, dtype=torch.float16),
        full_value=torch.zeros(1, dtype=torch.float16),
    )
    assert backend_choice.prefix_backend == "gather_then_copy"


def test_dense_prefix_workspace_materialize_writes_current_layer_prefix():
    """dense prefix backend 应按当前层把 chunk tensors 顺序写入 full workspace。"""
    impl = _make_impl()
    layer = _make_layer("model.layers.1.self_attn.attn")
    chunk0 = torch.tensor(
        [
            [
                [[10.0] * 8, [11.0] * 8],
                [[20.0] * 8, [21.0] * 8],
            ],
            [
                [[30.0] * 8, [31.0] * 8],
                [[40.0] * 8, [41.0] * 8],
            ],
        ],
        dtype=torch.float32,
    )
    chunk1 = torch.tensor(
        [
            [
                [[12.0] * 8],
                [[22.0] * 8],
            ],
            [
                [[32.0] * 8],
                [[42.0] * 8],
            ],
        ],
        dtype=torch.float32,
    )
    prefill_meta = SimpleNamespace(
        _granulekv_dense_prefix_chunk_tensors=(chunk0, chunk1),
        _granulekv_dense_prefix_context_tokens=3,
    )
    plan = SimpleNamespace(
        query_lens=(2, ),
        total_context_tokens=3,
    )
    full_key = torch.zeros(5, 2, 4, dtype=torch.float32)
    full_value = torch.zeros(5, 2, 4, dtype=torch.float32)

    profile = impl._fill_prefix_kv_from_runtime_dense_prefix_into_full_buffer(
        layer=layer,
        prefill_meta=prefill_meta,
        plan=plan,
        full_key=full_key,
        full_value=full_value,
    )

    expected_key = torch.tensor(
        [
            [[20.0] * 4, [20.0] * 4],
            [[21.0] * 4, [21.0] * 4],
            [[22.0] * 4, [22.0] * 4],
        ],
        dtype=torch.float32,
    )
    expected_value = torch.tensor(
        [
            [[40.0] * 4, [40.0] * 4],
            [[41.0] * 4, [41.0] * 4],
            [[42.0] * 4, [42.0] * 4],
        ],
        dtype=torch.float32,
    )
    assert profile.mode == "dense_prefix_workspace_consume"
    assert torch.equal(full_key[:3], expected_key)
    assert torch.equal(full_value[:3], expected_value)


def test_can_use_direct_query_scatter_accepts_stable_cuda_like_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    """helper 应在稳定 CUDA/Triton 口径下返回 True。"""
    from vllm.attention.backends import xformers as xformers_mod

    monkeypatch.setattr(xformers_mod, "triton", object())
    fake_tensor = SimpleNamespace(is_cuda=True, dtype=torch.float16)

    assert XFormersImpl._can_use_direct_query_scatter(
        fake_tensor,
        fake_tensor,
        fake_tensor,
        fake_tensor,
    )


def test_single_request_packed_compose_stays_disabled_on_current_mainline(
    monkeypatch: pytest.MonkeyPatch,
):
    """当前主线应显式关闭单请求 compose 快路径。

    这条测试保护当前已经收束好的语义：

    1. 单 request 也不应误开 compose 快路径
    2. 多 request 同样必须保持关闭
    """
    impl = _make_impl()
    monkeypatch.setattr(
        impl,
        "_can_use_packed_prefix_gather",
        lambda key_cache, value_cache: True,
    )
    monkeypatch.setattr(
        impl,
        "_can_use_direct_query_scatter",
        lambda query_key, query_value, full_key, full_value: True,
    )
    fake_tensor = SimpleNamespace(is_cuda=True, dtype=torch.float16)

    single_plan = SimpleNamespace(
        query_lens=(493, ),
        total_context_tokens=1024,
        total_query_tokens=493,
    )
    multi_plan = SimpleNamespace(
        query_lens=(128, 256),
        total_context_tokens=1024,
        total_query_tokens=384,
    )

    assert not impl._can_use_single_request_packed_compose(
        plan=single_plan,
        key_cache=fake_tensor,
        value_cache=fake_tensor,
        query_key=fake_tensor,
        query_value=fake_tensor,
        full_key=fake_tensor,
        full_value=fake_tensor,
    )
    assert not impl._can_use_single_request_packed_compose(
        plan=multi_plan,
        key_cache=fake_tensor,
        value_cache=fake_tensor,
        query_key=fake_tensor,
        query_value=fake_tensor,
        full_key=fake_tensor,
        full_value=fake_tensor,
    )
