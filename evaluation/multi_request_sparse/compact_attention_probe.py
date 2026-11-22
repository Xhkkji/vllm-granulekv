"""V100 compact backend integration probe; no model or storage service required."""
import argparse
import json, math
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.attention.backends.xformers import XFormersImpl
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.ops.paged_attn import PagedAttention as P
from vllm.core.custom_schedulers.hierarchical_io.barrier import (
    SparseKVLayerSelection, SparseKVBatchLayerSelection,
    activate_layer_barrier, activate_sparse_kv_blocks)
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(31)
impl=object.__new__(XFormersImpl)
impl.num_heads=28;impl.num_kv_heads=4;impl.head_size=128
impl.scale=1/math.sqrt(128);impl.alibi_slopes=None
impl.attn_type=AttentionType.DECODER;impl.kv_cache_dtype='auto'
scale=torch.tensor(1.,device='cuda');layer=SimpleNamespace(_k_scale=scale,_v_scale=scale)
results=[]
original_decode = P.forward_decode
original_write = P.write_to_paged_cache
calls = {'attention': 0, 'write': 0}
def counted_decode(*args, **kwargs):
    calls['attention'] += 1
    return original_decode(*args, **kwargs)
def counted_write(*args, **kwargs):
    calls['write'] += 1
    return original_write(*args, **kwargs)
P.forward_decode = counted_decode
P.write_to_paged_cache = counted_write
for tokens in (33,512,513,8192,8193):
    count=(tokens+15)//16;total=count+5;block=16;kh=4;dim=128
    storage=torch.randn(2*total*block*kh*dim,device='cuda',dtype=torch.float16)*.3
    cache=storage.as_strided((2,total,block*kh*dim),(block*kh*dim,2*block*kh*dim,1))
    kc,vc=P.split_kv_cache(cache,kh,dim)
    blocks=[tuple(range(count-1))+(1000,), tuple(range(max(0,count-2)))+(1000,)]
    rows=[tuple(range(count-1))+(total-2,),tuple(range(max(0,count-2)))+(total-1,)]
    lengths=[tokens,(len(rows[1])-1)*16+7]
    q=torch.randn(2,28*128,device='cuda',dtype=torch.float16)*.3
    k=torch.randn(2,4*128,device='cuda',dtype=torch.float16)*.3
    v=torch.randn_like(k)
    slots=[rows[i][-1]*16+(lengths[i]-1)%16 for i in range(2)]
    for order in ((0,1),(1,0),(0,)):
        ids=tuple('ab'[i] for i in order)
        selections={rid:SparseKVLayerSelection(blocks[i],None,True,(rows[i],),(slots[i],),lengths[i]) for rid,i in zip(ids,order)}
        metadata=SimpleNamespace(_sparse_request_metadata=tuple(SimpleNamespace(query_len=1) for i in order),use_cuda_graph=False)
        calls.update(attention=0, write=0)
        with activate_layer_barrier(lambda *a:None,virtual_engine=0,request_ids=ids), activate_sparse_kv_blocks(SparseKVBatchLayerSelection(selections)):
            out=impl.forward(layer,q[list(order)],k[list(order)],v[list(order)],cache,metadata).view(-1,28,128)
        assert calls == {"attention": 1, "write": 1}
        for j,i in enumerate(order):
            keys=kc[list(rows[i])].permute(0,3,1,2,4).reshape(-1,kh,dim)[:lengths[i]].float().repeat_interleave(7,1).transpose(0,1)
            vals=vc[list(rows[i])].permute(0,3,1,2).reshape(-1,kh,dim)[:lengths[i]].float().repeat_interleave(7,1).transpose(0,1)
            ref=((q[i].view(28,128).float().unsqueeze(1)@keys.transpose(1,2)*impl.scale).softmax(-1)@vals).squeeze(1)
            torch.testing.assert_close(out[j].float(),ref,rtol=.02,atol=.02)
        results.append({'tokens':tokens,'order':order,'passed':True, 'calls':dict(calls)})
# Exercise the epilogue across a real logical block boundary, including BA.
from vllm.granulekv.sparse_policy.solidattention import SolidAttentionPolicy
from vllm.core.custom_schedulers.hierarchical_io.sparse_policy import configure_sparse_kv_policy
layer.layer_name = "model.layers.0.self_attn.attn"
for order in ((0, 1), (1, 0), (0,)):
    policy = SolidAttentionPolicy(1, 1)
    configure_sparse_kv_policy(policy)
    storage = torch.randn(2 * 68 * 16 * 4 * 128, device='cuda', dtype=torch.float16)
    cache = storage.as_strided((2, 68, 16 * 4 * 128), (16 * 4 * 128, 2 * 16 * 4 * 128, 1))
    for i in order:
        policy._representatives['ab'[i]][0] = (63, torch.randn(63, 2, 4, 128, device='cuda', dtype=torch.float16))
    for end in (1023, 1024, 1025):
        ids = tuple('ab'[i] for i in order)
        logical = (0, 63) if end <= 1024 else (0, 63, 64)
        rows = [(0, 63 + i) if end <= 1024 else (0, 63 + i, 65 + i) for i in order]
        selections = {rid: SparseKVLayerSelection(logical, None, True, (row,),
            (row[-1] * 16 + (end - 1) % 16,), (len(row) - 1) * 16 + (end - 1) % 16 + 1)
            for rid, row in zip(ids, rows)}
        metadata = SimpleNamespace(use_cuda_graph=False, _sparse_request_metadata=tuple(
            SimpleNamespace(query_len=1, seq_len=end) for _ in order))
        calls.update(attention=0, write=0)
        with activate_layer_barrier(lambda *a: None, virtual_engine=0, request_ids=ids), activate_sparse_kv_blocks(SparseKVBatchLayerSelection(selections)):
            out = impl.forward(layer, q[list(order)], k[list(order)], v[list(order)], cache, metadata)
        assert calls == {'attention': 1, 'write': 1}
        kc, _ = P.split_kv_cache(cache, 4, 128)
        from vllm.attention.ops.sparse_kv import build_page_representatives_from_paged_key_cache
        for i, rid in zip(order, ids):
            count, reps = policy._representatives[rid][0]
            assert count == end // 16
            if end >= 1024:
                expected = build_page_representatives_from_paged_key_cache(kc, torch.tensor([63+i], device='cuda'), output_device=kc.device)
                torch.testing.assert_close(reps[-1:], expected, rtol=0, atol=0)
            policy.select_actual_blocks_device(rid, 0, q[i].view(28, 128), 63, end, end+1, 16, 32)
        results.append({'lifecycle_end': end, 'order': order, 'passed': True, 'calls': dict(calls)})
configure_sparse_kv_policy(None)
print(json.dumps(results))
args.output.write_text(json.dumps(results,indent=2))
