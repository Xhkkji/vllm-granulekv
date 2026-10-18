"""GranuleKV physical block pairing, without changing logical K/V views."""

from types import SimpleNamespace

import pytest
import torch

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.granulekv.kv_layout import GranuleKVLayout


@pytest.mark.parametrize("blocks,host_blocks", [(8, 12), (2304, 2304)])
def test_paired_layout_payload_and_views(blocks, host_blocks):
    elements = 16 * 4 * 128
    layout = GranuleKVLayout.from_allocation(
        allocation_shape=(2, blocks, elements), stride_order=(0, 1, 2),
        dtype=torch.float16, num_layers=28, num_gpu_regions=8,
        num_gpu_blocks=blocks, num_storage_blocks=4096, device_index=0,
        num_host_blocks=host_blocks, num_host_regions=28)
    assert layout.tensor_strides == (elements, 2 * elements, 1)
    assert layout.host_tensor_strides == (elements, 2 * elements, 1)
    assert layout.fragment_bytes == 16384
    assert layout.region_bytes == blocks * 32768
    assert layout.host_region_bytes == host_blocks * 32768
    payload = layout.allocation_payload(client_pid=1)
    assert payload["descriptor_bytes"] == 32768
    assert payload["logical_block_bytes"] == 28 * 32768
    assert payload["block_stride_elems"] * 2 == 32768
    assert payload["kv_stride_elems"] * 2 == 16384
    manifest = dict(tensor_shape=layout.tensor_shape,
                    tensor_strides=layout.tensor_strides,
                    dtype=layout.dtype_name, descriptor_bytes=32768,
                    regions=[{}] * 36, gpu_regions=[{}] * 8,
                    host_regions=[{}] * 28)
    layout.validate_manifest(manifest)
    with pytest.raises(RuntimeError, match="descriptor size"):
        layout.validate_manifest(dict(manifest, descriptor_bytes=16384))
    for count, strides in [(blocks, layout.tensor_strides),
                           (host_blocks, layout.host_tensor_strides)]:
        storage = torch.full((count * 2 * elements,), -7, dtype=torch.float16)
        cache = storage.as_strided((2, count, elements), strides)
        k, v = PagedAttention.split_kv_cache(cache, 4, 128)
        assert k.stride(0) == v.stride(0) == 2 * elements
        k[1].fill_(3)
        v[1].fill_(5)
        assert torch.all(storage[2 * elements:3 * elements] == 3)
        assert torch.all(storage[3 * elements:4 * elements] == 5)
        assert torch.all(storage[:2 * elements] == -7)
        assert torch.all(storage[4 * elements:] == -7)
        with pytest.raises(ValueError, match="matching physical KV layouts"):
            PagedAttention.swap_blocks(
                cache, torch.empty(cache.shape, dtype=cache.dtype),
                torch.tensor([[1, 2]], dtype=torch.int64))


@torch.inference_mode()
def test_paired_prefix_prefill_matches_original_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from vllm.attention.backends.xformers import XFormersImpl

    blocks, block_size, heads, kv_heads, dim = 4, 16, 28, 4, 128
    elements = block_size * kv_heads * dim
    storage = torch.randn(2 * blocks * elements, device="cuda",
                          dtype=torch.float16)
    cache = storage.as_strided((2, blocks, elements),
                               (elements, 2 * elements, 1))
    key_cache, value_cache = PagedAttention.split_kv_cache(cache, kv_heads, dim)
    impl = XFormersImpl.__new__(XFormersImpl)
    impl.num_heads = heads
    impl.num_kv_heads = kv_heads
    impl.num_queries_per_kv = heads // kv_heads
    impl.head_size = dim
    impl.scale = dim ** -0.5
    impl.sliding_window = None
    metadata = SimpleNamespace(
        num_prefills=1, num_prefill_tokens=7, seq_lens=[45],
        context_lens_tensor=torch.tensor([38], device="cuda", dtype=torch.int32),
        query_start_loc=torch.tensor([0, 7], device="cuda", dtype=torch.int32),
        block_tables=torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32),
        _cached_prefix_fallback_plan=None, _cached_prefix_fallback_plan_key=None,
        _cached_prefix_fallback_workspace=None,
        _cached_prefix_fallback_workspace_key=None)
    query = torch.randn(7, heads, dim, device="cuda", dtype=torch.float16)
    key = torch.randn(7, kv_heads, dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    kwargs = dict(layer=SimpleNamespace(layer_name="model.layers.0.self_attn.attn"),
                  query=query, key=key, value=value, prefill_meta=metadata)
    paired = impl._run_prefix_attention_fallback(
        key_cache=key_cache, value_cache=value_cache, **kwargs)
    contiguous = impl._run_prefix_attention_fallback(
        key_cache=key_cache.contiguous(), value_cache=value_cache.contiguous(),
        **kwargs)
    torch.testing.assert_close(paired, contiguous, rtol=0, atol=0)


@pytest.mark.parametrize("paired", [False, True])
def test_single_block_swap_layout_detection(monkeypatch, paired):
    from vllm import _custom_ops as ops
    elements = 8192
    source = torch.empty(2, 1, elements)
    target = torch.empty(2, 4, elements)
    if paired:
        source = source.as_strided((2, 1, elements), (elements, 2 * elements, 1))
        target = target.as_strided((2, 4, elements), (elements, 2 * elements, 1))
    calls = []
    monkeypatch.setattr(ops, "swap_blocks", lambda src, dst, _: calls.append(
        (src.shape, dst.shape)))
    PagedAttention.swap_blocks(source, target, torch.tensor([[0, 2]]))
    assert len(calls) == (1 if paired else 2)
    assert calls[0][0] == (1, elements * (2 if paired else 1))
