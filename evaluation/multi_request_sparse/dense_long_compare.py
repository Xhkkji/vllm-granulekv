"""Test-only dense full-layer baseline: independent reversed logical KV replay plus local FP32 gate."""
import argparse,json,os,time,traceback,hashlib,signal
from pathlib import Path
from types import MethodType
from dataclasses import replace
import torch
from transformers import AutoConfig,AutoTokenizer
from vllm import LLM,SamplingParams
from vllm.utils import Device
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.core.custom_schedulers.hierarchical_io.residency import SharedHostKVDirectory
from evaluation.multi_request_sparse.validation import BatchReference,update_logical,check_mapping,check_cache_contents,logical_attention,numerical_error

class DenseReference(BatchReference):
 def __init__(self,*args):
  self.active=True;self.batch_logits=[]
  super().__init__(*args)
 def _capture_inputs(self,layer,inputs):
  if self.active:super()._capture_inputs(layer,inputs)
 def _observed_backend(self,original):
  def forward(layer,q,k,v,cache,metadata,*args,**kw):
   frame=self.current
   if not self.active or self.reference_active or not frame or not all(r.startswith('batch-') for r in frame['ids']):
    return original(layer,q,k,v,cache,metadata,*args,**kw)
   index=layer.impl._layer_index_from_name(layer.layer_name)
   for row,rid in enumerate(frame['ids']):
    interval=frame['intervals'][rid];positions=frame['positions'][interval];n=len(positions)
    if n!=1:raise RuntimeError('dense restore gate requires one query per request')
    logical=self.actual_live.setdefault((rid,index),tuple(x.clone() for x in self._source(rid,index)))
    self.actual_live[rid,index]=update_logical(logical,k[interval].reshape(n,layer.num_kv_heads,layer.head_size),v[interval].reshape(n,layer.num_kv_heads,layer.head_size),int(positions[0]))
   out=original(layer,q,k,v,cache,metadata,*args,**kw)
   kc,vc=PagedAttention.split_kv_cache(cache,layer.num_kv_heads,layer.head_size)
   for row,rid in enumerate(frame['ids']):
    interval=frame['intervals'][rid];positions=frame['positions'][interval];end=int(positions[-1])+1
    blocks=list(range((end+15)//16));table=metadata.block_tables[row:row+1];slots=metadata.slot_mapping[interval]
    physical=check_mapping(table,slots,blocks,torch.arange(len(blocks)),len(blocks),positions,cache.shape[1])
    self.report['mapping_checks']+=1
    logical=self.actual_live[rid,index]
    check_cache_contents(kc,vc,table,blocks,end,logical);self.report['logical_kv_checks']+=1
    expected=logical_attention(q[interval].reshape(-1,layer.num_heads,layer.head_size),logical,blocks,positions,layer.impl.scale)
    err=numerical_error(out[interval].reshape_as(expected),expected)
    self.report['fp32_attention_checks']+=1
    self.report['fp32_attention_max_abs_error']=max(self.report['fp32_attention_max_abs_error'],err['max_abs_error'])
    if not err['passed']:raise RuntimeError(f'dense local FP32 mismatch {rid}/{index}: {err}')
   return out
  return forward
 def replay_attention(self,layer,q,k,v,output_shape=None):
  frame=self.reference_frame;index=layer.impl._layer_index_from_name(layer.layer_name);out=torch.empty_like(q)
  for row,rid in enumerate(frame['ids']):
   interval=frame['intervals'][rid];pos=frame['positions'][interval];n=len(pos);start=int(pos[0]);end=int(pos[-1])+1
   logical=self.reference_live.setdefault((rid,index),tuple(x.clone() for x in self._source(rid,index)))
   logical=update_logical(logical,k[interval].reshape(n,layer.num_kv_heads,layer.head_size),v[interval].reshape(n,layer.num_kv_heads,layer.head_size),start)
   self.reference_live[rid,index]=logical
   total=(end+15)//16
   table=torch.arange(total-1,-1,-1,dtype=torch.int32,device=q.device)[None,:]
   cache=torch.zeros(PagedAttention.get_kv_cache_shape(total,16,layer.num_kv_heads,layer.head_size),dtype=k.dtype,device=q.device)
   kc,vc=PagedAttention.split_kv_cache(cache,layer.num_kv_heads,layer.head_size)
   # Explicit independent reversed physical placement from logical post-RoPE KV.
   lk=torch.zeros((total*16,layer.num_kv_heads,layer.head_size),dtype=k.dtype);lv=torch.zeros_like(lk)
   lk[:end]=logical[0][:end];lv[:end]=logical[1][:end]
   kc.copy_(lk.reshape(total,16,layer.num_kv_heads,layer.head_size//8,8).flip(0).permute(0,2,3,1,4))
   vc.copy_(lv.reshape(total,16,layer.num_kv_heads,layer.head_size).flip(0).permute(0,2,3,1))
   slots=table[0,pos.to(q.device)//16].long()*16+pos.to(q.device)%16
   tensor=lambda x:torch.tensor(x,dtype=torch.int32,device=q.device)
   prefill=row<frame['metadata'].num_prefills
   meta=replace(frame['metadata'],num_prefills=int(prefill),num_prefill_tokens=n if prefill else 0,num_decode_tokens=0 if prefill else n,
    slot_mapping=slots,block_tables=table,seq_lens=[end],seq_lens_tensor=tensor([end]),query_start_loc=tensor([0,n]),seq_start_loc=tensor([0,end]),
    context_lens_tensor=tensor([start]),max_query_len=n,max_decode_query_len=1,max_prefill_seq_len=end if prefill else 0,max_decode_seq_len=0 if prefill else end,
    use_cuda_graph=False,_cached_prefill_metadata=None,_cached_decode_metadata=None)
   original=next(f for impl,f in self.original_backends if impl is layer.impl)
   out[interval]=original(layer,q[interval],k[interval],v[interval],cache,meta)
  return out
 def replay(self):
  from vllm.forward_context import set_forward_context
  originals=[(l,l.forward) for l in self.layers];self.reference_active=True;reports=[];tokens={}
  try:
   for l,_ in originals:l.forward=MethodType(self.replay_attention,l)
   with torch.inference_mode():
    for i,frame in enumerate(self.frames):
     self.reference_frame=frame
     with set_forward_context(frame['metadata'],self.worker.model_runner.vllm_config,0):
      hidden=self.model(input_ids=frame['input_tokens'],positions=frame['positions'].to(frame['input_tokens'].device))
      logits=self.model.logits_processor(self.model.lm_head,hidden.index_select(0,frame['sampling_indices'].long()),None)
     err=numerical_error(self.batch_logits[i],logits.cpu());reports.append(err)
     if not err['passed']:raise RuntimeError(f'dense independent model logits mismatch: {err}')
     for row,rid in enumerate(frame['ids']):tokens.setdefault(rid,[]).append(int(logits[row].argmax()))
   return dict(logits=reports,tokens=tokens)
  finally:
   for l,f in originals:l.forward=f
   self.reference_active=False

from evaluation.multi_request_sparse.real_documents import load_scenario, decode_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-manifest', type=Path, required=True)
    parser.add_argument('--scenario', choices=('fixed32k', 'fixed128k'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--service-timeout', type=float, default=1800)
    parser.add_argument('--no-progress-timeout', type=float, default=120)
    args = parser.parse_args()
    manifest, scenario, prompt, suffixes = load_scenario(args.real_manifest, args.scenario)
    if any(suffixes) or scenario['max_output_tokens'] != 512 or not scenario['ignore_eos']:
        raise ValueError('requires identical exact-length fixed512 workload')
    capacity = int(os.environ['VLLM_GRANULEKV_SPARSE_CPU_CACHE_BLOCKS'])
    if capacity != scenario['cpu_slots']:
        raise ValueError('CPU capacity differs from manifest')
    result = dict(status='started', performance_valid=False, runs=[], gates=[], performance=[],
                  config=dict(model=manifest['model'], prefix_tokens=len(prompt), output_tokens=512,
                              cpu_slots=capacity, dtype='half', seed=1234,
                              service_timeout=args.service_timeout, no_progress_timeout=args.no_progress_timeout),
                  scenario={k:v for k,v in scenario.items() if k != 'prompt_token_ids'},
                  manifest_sha256=hashlib.sha256(args.real_manifest.read_bytes()).hexdigest(),
                  environment={k:v for k,v in os.environ.items() if k.startswith('VLLM_')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    ref = None
    try:
        cfg = AutoConfig.from_pretrained(manifest['model'])
        maxlen = len(prompt) + 512 + 16
        requested_slots = 2 * ((maxlen+15)//16) + 128
        result['config'].update(max_model_len=maxlen, requested_gpu_slots=requested_slots,
                               hf_overrides={'max_position_embeddings':max(maxlen,cfg.max_position_embeddings)},
                               rope_theta=cfg.rope_theta, rope_scaling=getattr(cfg,'rope_scaling',None))
        llm = LLM(model=manifest['model'], dtype='half', max_model_len=maxlen,
                  hf_overrides=result['config']['hf_overrides'], gpu_memory_utilization=.85,
                  swap_space=32, enable_prefix_caching=True, enable_chunked_prefill=True,
                  scheduler_cls='vllm.core.async_kv_scheduler.AsyncKVScheduler',
                  enforce_eager=True, disable_async_output_proc=True, max_num_seqs=4,
                  max_num_batched_tokens=2048, num_gpu_blocks_override=requested_slots, seed=1234)
        engine = llm.llm_engine
        worker = engine.model_executor.driver_worker.worker
        cache = worker.cache_engine[0]
        result['config'].update(actual_gpu_slots=cache.num_gpu_blocks,
                                regions=cache.granulekv_connector.layout.num_gpu_regions)
        ref = DenseReference(worker, [prompt,prompt], args.output.parent)
        result['independent_checks'] = ref.report
        forwards, shapes, native = [], [], []
        original_execute = worker.model_runner.execute_model
        def execute(*pos, **kw):
            inp = kw.get('model_input', pos[0] if pos else None)
            ids = list(inp.request_ids_to_seq_ids or {})
            forwards.append(ids)
            shapes.append(dict(ids=ids,query_lens=list(inp.query_lens),seq_lens=list(inp.seq_lens)))
            if ref.active:
                ref.begin_forward(inp)
            return original_execute(*pos, **kw)
        worker.model_runner.execute_model = execute
        def logits_hook(module, inputs, out):
            if ref.active and not ref.reference_active and ref.current and all(
                    rid.startswith('batch-') for rid in ref.current['ids']):
                ref.batch_logits.append(out.detach().cpu())
        hook = worker.model_runner.get_model().logits_processor.register_forward_hook(logits_hook)
        original_submit = cache.submit_async_kv_transfer
        def submit(request_id, operation, mapping, layer_range=None, destination='gpu', host_block_mapping=()):
            native.append(dict(id=request_id, operation=operation.value,
                               bytes=(len(mapping)+len(host_block_mapping))*32768*(28 if layer_range is None else layer_range[1]-layer_range[0]),
                               destination=destination))
            return original_submit(request_id,operation,mapping,layer_range,destination,host_block_mapping)
        cache.submit_async_kv_transfer = submit
        params = SamplingParams(temperature=0,max_tokens=2,ignore_eos=True)
        def drained():
            pending = {name:len(getattr(worker,name)) for name in (
                '_host_async_kv_transfers','_pending_gpu_to_host','_pending_host_writes',
                '_deferred_gpu_transfers','_gpu_transfer_children','_gpu_shared_waits','_inflight_h2d_sources')
                if getattr(worker,name)}
            if cache.granulekv_connector._pending_transfers:
                pending['native'] = len(cache.granulekv_connector._pending_transfers)
            if any(e.in_flight or e.loading_layers or e.owners for e in worker._host_kv_directory._entries.values()):
                pending['host_references'] = True
            if pending:
                result['undrained_boundary'] = pending
                raise RuntimeError('undrained dense boundary')
        def run(label, count):
            forwards.clear(); shapes.clear(); native.clear()
            worker.get_sparse_kv_stats(reset=True)
            record = dict(label=label,request_ids=[f'{label}-{i}' for i in range(count)],
                          ttft_ms={},token_ids={},output_finished_ms={},token_timestamps_ms={},finish_reasons={})
            result['runs'].append(record)
            (args.output.parent/'progress.json').write_text(json.dumps(dict(label=label,phase='running',started_unix=time.time())))
            start = time.perf_counter()
            last_output = last_progress = start
            def watchdog(signum, frame):
                now = time.perf_counter()
                if now-start >= args.service_timeout or now-last_progress >= args.no_progress_timeout:
                    raise TimeoutError(f'{label}: service or output progress watchdog expired')
                signal.setitimer(signal.ITIMER_REAL,min(args.service_timeout-(now-start),args.no_progress_timeout-(now-last_progress)))
            old_alarm = signal.signal(signal.SIGALRM,watchdog)
            signal.setitimer(signal.ITIMER_REAL,min(args.service_timeout,args.no_progress_timeout))
            try:
                for rid in record['request_ids']:
                    engine.add_request(rid,{'prompt_token_ids':prompt},params)
                while engine.has_unfinished_requests() or any(s._finished_requests_ids for s in engine.scheduler):
                    for output in engine.step():
                        if output.outputs and output.outputs[0].token_ids:
                            now = time.perf_counter()
                            tokens = list(output.outputs[0].token_ids)
                            rid = output.request_id
                            record['ttft_ms'].setdefault(rid,(now-start)*1000)
                            if len(tokens)>len(record['token_ids'].get(rid,[])):
                                last_progress=now
                                record['token_timestamps_ms'].setdefault(rid,[]).append([(now-start)*1000,len(tokens)])
                            record['token_ids'][rid]=tokens
                            record['finish_reasons'][rid]=output.outputs[0].finish_reason
                            if output.finished:
                                last_output=now
                                record['output_finished_ms'][rid]=(now-start)*1000
                for copy_event in tuple(worker._pending_h2d_transfers.values()):
                    copy_event.event.synchronize()
                worker._poll_host_transfers(0)
                drained()
                end=time.perf_counter()
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                signal.signal(signal.SIGALRM,old_alarm)
            record.update(elapsed_with_drain_ms=(end-start)*1000,after_last_output_ms=(end-last_output)*1000,
                          forward_batches=list(forwards),forward_shapes=list(shapes),max_forward_batch=max(map(len,forwards),default=0),
                          native_transfers=list(native),ssd_read_requests=sum(x['operation']=='read' for x in native),
                          ssd_read_bytes=sum(x['bytes'] for x in native if x['operation']=='read'),
                          ssd_write_bytes=sum(x['bytes'] for x in native if x['operation']=='write'),
                          stats=worker.get_sparse_kv_stats(),
                          decode_segments={rid:decode_segments(ts) for rid,ts in record['token_timestamps_ms'].items()})
            if any(record['stats'].get(k,0) for k in ('host_transfer_errors','host_coverage_missing_blocks','gpu_slot_capacity_rejections','host_write_backlog')):
                raise RuntimeError('dense transfer/capacity/backing failure')
            if len(record['token_ids'])!=count or any(len(t)!=params.max_tokens for t in record['token_ids'].values()):
                raise RuntimeError('incomplete output')
            (args.output.parent/'progress.json').write_text(json.dumps(dict(label=label,phase='completed',elapsed_ms=record['elapsed_with_drain_ms'])))
            print(label,round(record['elapsed_with_drain_ms'],1),'max_forward',record['max_forward_batch'],flush=True)
            save()
            return record
        def snapshot():
            entries=worker._host_kv_directory._entries
            return dict(slots=len(entries),ready_layer_blocks=sum(len(e.ready_layers) for e in entries.values()))
        def reset(cold=False):
            worker._poll_host_transfers(0);drained()
            if not llm.reset_prefix_cache(Device.GPU):
                raise RuntimeError('GPU prefix reset failed')
            if cold:
                worker._host_kv_directory=SharedHostKVDirectory(capacity)
                worker._host_ready_blocks.clear();worker._host_staged_blocks.clear();worker._host_failed_blocks.clear()
        pop=run('populate-0-0',1)
        ref.seal_populate(pop['request_ids'][0],0)
        reset(cold=True)
        for state in ('cold','hot'):
            if state=='hot':reset()
            ref.actual_live.clear();ref.reference_live.clear();ref.frames.clear();ref.batch_logits.clear()
            before={f:ref.report[f] for f in ('mapping_checks','logical_kv_checks','fp32_attention_checks')}
            batch=run('batch',2)
            counts={f:ref.report[f]-before[f] for f in before}
            if any(v!=112 for v in counts.values()):
                raise RuntimeError(f'incomplete dense gate: {counts}')
            replay=ref.replay()
            if replay['tokens']!=batch['token_ids']:
                raise RuntimeError('dense independent reference token mismatch')
            result['gates'].append(dict(cache=state,reference=replay,checks=counts,
                                       forward_sizes=[len(f) for f in batch['forward_batches']]))
            save()
        result['status']='correctness_passed';save()
        expected_prefix=list(batch['token_ids'].values())
        ref.active=False;ref.close();hook.remove()
        ref.actual_live.clear();ref.reference_live.clear();ref.sources.clear();ref.frames.clear();ref.current=None;ref.batch_logits.clear()
        params=SamplingParams(temperature=0,max_tokens=512,ignore_eos=True)
        result['performance_method']=dict(numeric_recording=False,profiler=False,repeats=1,
             boundary='submission through outputs, writeback and drain',
             cold='empty CPU directory at sample start',hot='full512 same-count warmup; retain CPU; clear GPU prefix',
             batch='simultaneous submission; report actual forward sizes; no admission change',
             comparison='system path, not equal GPU capacity or pure attention comparison')
        for state in ('cold','hot'):
            for count in (1,2):
                prep=time.perf_counter();reset(cold=True)
                warm=None
                if state=='hot':
                    warm=run(f'warmup-{state}-b{count}',count);reset()
                preparation_ms=(time.perf_counter()-prep)*1000
                initial=snapshot()
                if (state=='cold' and initial['slots']) or (state=='hot' and not initial['ready_layer_blocks']):
                    raise RuntimeError('cache state mismatch')
                sample=run(f'perf-{state}-b{count}',count)
                actual=[sample['token_ids'][rid] for rid in sample['request_ids']]
                if any(t[:2]!=expected_prefix[i] for i,t in enumerate(actual)):
                    raise RuntimeError('long output prefix differs from validated dense gate')
                if warm and actual!=[warm['token_ids'][rid] for rid in warm['request_ids']]:
                    raise RuntimeError('warmup/formal full512 token mismatch')
                result['performance'].append(dict(cache=state,batch_size=count,mode='batch',
                    batch_with_drain_ms=sample['elapsed_with_drain_ms'],preparation_ms=preparation_ms,
                    initial_cpu=initial,final_cpu=snapshot(),actual_output_tokens=list(map(len,actual)),
                    output_tokens_per_second=1000*sum(map(len,actual))/sample['elapsed_with_drain_ms'],samples=[sample['label']]))
                save()
        result['status']='performance_passed';result['performance_valid']=True;save()
    except BaseException:
        result.update(status='failed',performance_valid=False,error=traceback.format_exc())
        if ref is not None and ref.active:
            ref.report['passed'] = False
        if 'worker' in locals():
            result['failure_stats']=worker.get_sparse_kv_stats()
        save()
        raise
    finally:
        if ref:ref.close()


if __name__=='__main__':
    main()
