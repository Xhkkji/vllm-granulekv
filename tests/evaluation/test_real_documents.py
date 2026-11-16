import json
import pytest
from evaluation.multi_request_sparse.real_documents import (
    split_prompts, token_hash, load_scenario, decode_segments)


def test_real_prompt_split_preserves_tokens_and_question_remainder(tmp_path):
    shared = list(range(611))
    prompts = [shared + [800,801], shared + [900]]
    prefix, suffix = split_prompts(prompts)
    assert len(prefix) == 608
    assert [prefix+s for s in suffix] == prompts
    path = tmp_path/'manifest.json'
    scenario = {'name':'natural', 'prompt_token_ids':prompts,
                'token_sha256':list(map(token_hash,prompts)),
                'common_prefix_tokens':608}
    path.write_text(json.dumps({'scenarios':[scenario]}))
    assert load_scenario(path,'natural')[2:] == (prefix,suffix)
    scenario['prompt_token_ids'][1][-1] += 1
    path.write_text(json.dumps({'scenarios':[scenario]}))
    with pytest.raises(ValueError,match='hash'):
        load_scenario(path,'natural')


def test_fixed512_and_natural_early_eos_segments():
    rows = decode_segments([[10*i,i] for i in range(1,513)])
    assert [r['outputs_observed'] for r in rows] == [64,64,128,256]
    assert all(r['itl_p95_ms'] == 10 for r in rows)
    rows = decode_segments([[10,1],[20,2],[20,2],[30,3]])
    assert rows[0]['outputs_observed'] == 3
    assert rows[-1]['last_ms'] is None
    assert rows[-1]['itl_p50_ms'] is None


def test_grouped_outputs_are_not_invented_token_latencies():
    rows = decode_segments([[10,1],[30,3]])
    assert rows[0]['multi_token_deliveries'] == 1
    assert rows[0]['itl_p50_ms'] is None


def test_diagnostic_copy_classification_and_source_use_balance(monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace
    from evaluation.multi_request_sparse.long_decode_checks import LongDecodeChecks
    from vllm.core.custom_schedulers.hierarchical_io.residency import SharedHostKVDirectory
    from vllm.worker.cache_engine import CacheEngine
    import vllm.core.custom_schedulers.hierarchical_io as io
    monkeypatch.setattr(io, 'get_sparse_kv_policy', lambda: SimpleNamespace(init_blocks=1,local_blocks=1))
    monkeypatch.setattr(CacheEngine, 'get_cache_block_size', lambda *args: 28*32768)
    from vllm.attention.ops.paged_attn import PagedAttention
    monkeypatch.setattr(PagedAttention, 'forward_decode', staticmethod(lambda x:x))
    class Event:
        def __init__(self, **kwargs): pass
        def record(self, *args): pass
        def synchronize(self): pass
        def elapsed_time(self, other): return 1.5
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda.Stream,'wait_event',lambda stream,event:None)
    directory = SharedHostKVDirectory(4)
    directory.acquire(('p',0),(0,'a'))
    calls = []
    engine = SimpleNamespace(num_attention_layers=28,
        copy_gpu_blocks_to_host_async=lambda *a: calls.append(('d2h',a)),
        copy_host_blocks_to_gpu_async=lambda *a: calls.append(('h2d',a)),
        submit_async_kv_transfer=lambda *a: calls.append(('ssd',a)))
    model = torch.nn.Module()
    model.logits_processor = torch.nn.Identity()
    worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda:model),
        cache_engine=[engine],cache_config=None,model_config=None,parallel_config=None,
        _host_kv_directory=directory,_take_gpu_region_release_event=lambda *args:Event(),_backing_forward={(0,'a'):(None,16,17,False)},
        _gpu_slot_mappings={(0,'a',0):{0:2,1:3}})
    observer = LongDecodeChecks(worker,{},[],tmp_path)
    try:
        observer.frame = {'ids':['a'], 'steps':{'a':16}}
        assert PagedAttention.forward_decode('result') == 'result'
        fence = worker._take_gpu_region_release_event(0,0,0)
        torch.cuda.Stream.wait_event(SimpleNamespace(cuda_stream=7),fence)
        observer.frame['steps']['a'] = 18
        assert PagedAttention.forward_decode('result') == 'result'
        engine.copy_gpu_blocks_to_host_async(0,torch.tensor([[2,0],[3,1]]),())
        engine.copy_host_blocks_to_gpu_async(0,torch.tensor([[0,2]]),())
        directory.begin_use(('p',0)); directory.begin_use(('p',0))
        directory.finish_use(('p',0)); directory.finish_use(('p',0))
        with pytest.raises(ValueError,match='without matching'):
            directory.finish_use(('p',0))
        assert directory._entries['p',0].in_flight == 0
    finally:
        observer.close()
    report = json.loads((tmp_path/'long_diagnostic.json').read_text())
    assert report['dirty_d2h_bytes'] == report['immutable_d2h_bytes'] == 32768
    assert report['new_effective_kv_bytes'] == 2048
    assert report['counts']['source_acquires'] == report['counts']['source_releases'] == 2
    assert report['outstanding_source_uses'] == {}
    assert report['counts']['paged_attention_calls'] == 2
    assert report['checkpoint_attention_gpu_event_ms'] == [1.5]
    assert report['checkpoint_reuse_fence_gpu_event_ms'] == [{'stream':7,'milliseconds':1.5}]
    assert [c[0] for c in calls] == ['d2h','h2d']


def test_private_suffix_capture_uses_request_mapping_not_sparse_read_holes():
    from evaluation.multi_request_sparse.real_documents import live_physical_ids
    mapping = {2204:347,2205:348,2206:349,2207:350}
    assert live_physical_ids(mapping,2204,2208,2135) == [347,348,349,350]
    with pytest.raises(KeyError):
        live_physical_ids({2204:347},2204,2208,2135)
    with pytest.raises(ValueError):
        live_physical_ids({0:-1},0,1,2135)
    with pytest.raises(ValueError):
        live_physical_ids({0:7,1:7},0,2,2135)
