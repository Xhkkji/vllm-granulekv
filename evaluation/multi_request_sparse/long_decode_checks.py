"""Test-only checkpoint oracle. Captures model inputs, never transfer outputs.

Checks selection, mapping, exact logical KV and local FP32 Attention. The initial
2-token gate separately checks independent whole-model logits. This observer
is not a 512-step whole-model numerical proof; checkpoint Q/K/V are teacher-forced
from the actual forward, with all intervening post-RoPE KV inputs retained.
"""
from collections import Counter
import json
from pathlib import Path
import torch
from evaluation.multi_request_sparse.validation import (
    update_logical, request_slices, check_mapping, check_cache_contents,
    logical_attention, numerical_error)

CHECKPOINTS = (1, 2, 16, 17, 64, 128, 256, 512)


class LongDecodeChecks:
    def __init__(self, worker, sources, prompts, directory, checkpoints=CHECKPOINTS):
        from vllm.attention.layer import Attention
        from vllm.attention.ops.paged_attn import PagedAttention
        from evaluation.paper_reproduction.solidattention.adapter.solidattention import SolidAttentionPolicy
        from vllm.core.custom_schedulers.hierarchical_io import get_sparse_kv_policy
        self.worker, self.sources, self.prompts = worker, sources, prompts
        self.checkpoints = tuple(checkpoints)
        self.max_steps = Counter()
        self.immutable, self.shared_slots = {}, {}
        self.directory = Path(directory)
        policy = get_sparse_kv_policy()
        self.policy = SolidAttentionPolicy(policy.init_blocks, policy.local_blocks)
        self.logical, self.frame, self.checked = {}, None, []
        self.counts, self.peak, self.evictions = Counter(), 0, 0
        self.source_uses = Counter()
        self.attention_events = []
        self.reuse_events, self.fence_timings = {}, []
        self.attention_class = PagedAttention
        self.originals, self.handles = [], []
        self.originals.append((PagedAttention, 'forward_decode', PagedAttention.forward_decode))
        original_attention = PagedAttention.forward_decode
        def counted_attention(*args, **kwargs):
            self.counts['paged_attention_calls'] += 1
            sampled = self.frame is not None and any(step in self.checkpoints for step in self.frame['steps'].values())
            if sampled:
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
            result = original_attention(*args, **kwargs)
            if sampled:
                end.record()
                self.attention_events.append((start,end))
            return result
        PagedAttention.forward_decode = staticmethod(counted_attention)
        self.handles.append(worker.model_runner.get_model().logits_processor.register_forward_hook(self.capture_logits))
        for layer in worker.model_runner.get_model().modules():
            if isinstance(layer, Attention):
                self.handles.append(layer.register_forward_pre_hook(self.capture))
                original = layer.impl.forward
                self.originals.append((layer.impl, 'forward', original))
                layer.impl.forward = self.observe(original)
        original_take = worker._take_gpu_region_release_event
        self.originals.append((worker,'_take_gpu_region_release_event',original_take))
        def take_fence(*args, **kwargs):
            event = original_take(*args, **kwargs)
            if event is not None and self.frame is not None and any(
                    step in self.checkpoints for step in self.frame['steps'].values()):
                self.reuse_events[id(event)] = event
            return event
        worker._take_gpu_region_release_event = take_fence
        original_wait = torch.cuda.Stream.wait_event
        self.originals.append((torch.cuda.Stream,'wait_event',original_wait))
        def timed_wait(stream,event):
            if (id(event) not in self.reuse_events or self.frame is None
                    or not any(step in self.checkpoints for step in self.frame['steps'].values())):
                return original_wait(stream,event)
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(stream)
            result = original_wait(stream,event)
            end.record(stream)
            self.fence_timings.append((int(stream.cuda_stream),start,end))
            return result
        torch.cuda.Stream.wait_event = timed_wait
        engine = worker.cache_engine[0]
        original_d2h = engine.copy_gpu_blocks_to_host_async
        self.originals.append((engine, 'copy_gpu_blocks_to_host_async', original_d2h))
        def counted_d2h(layer, mapping, dependencies):
            dirty_slots = set()
            for rid in self.frame['ids'] if self.frame else ():
                state = worker._backing_forward.get((0, rid))
                if state is None:
                    continue
                _, start, end, _ = state
                table = worker._gpu_slot_mappings.get((0,rid,layer),{})
                dirty_slots.update(table[b] for b in range(start//16,(end+15)//16) if b in table)
                self.counts['new_effective_kv_tokens_layers'] += end-start
            pairs = mapping.tolist()
            dirty = sum(source in dirty_slots for source,_ in pairs)
            self.counts['dirty_d2h_blocks_layers'] += dirty
            self.counts['immutable_d2h_blocks_layers'] += len(pairs)-dirty
            self.counts['d2h_submissions'] += 1
            return original_d2h(layer,mapping,dependencies)
        engine.copy_gpu_blocks_to_host_async = counted_d2h
        from vllm.worker.cache_engine import CacheEngine
        self.block_bytes = CacheEngine.get_cache_block_size(worker.cache_config,
            worker.model_config,worker.parallel_config)//engine.num_attention_layers
        original_h2d = engine.copy_host_blocks_to_gpu_async
        self.originals.append((engine, 'copy_host_blocks_to_gpu_async', original_h2d))
        def counted_h2d(layer, mapping, dependencies=()):
            self.counts['h2d_blocks_layers'] += len(mapping)
            self.counts['h2d_submissions'] += 1
            self.counts['h2d_release_dependencies'] += len(dependencies)
            return original_h2d(layer,mapping,dependencies)
        engine.copy_host_blocks_to_gpu_async = counted_h2d
        original_submit = engine.submit_async_kv_transfer
        self.originals.append((engine,'submit_async_kv_transfer',original_submit))
        def counted_submit(request_id,operation,mapping,layer_range=None,
                           destination='gpu',host_block_mapping=()):
            count = (len(mapping)+len(host_block_mapping))*(28 if layer_range is None else layer_range[1]-layer_range[0])
            self.counts[f'ssd_{operation.value}_bytes'] += count*self.block_bytes
            self.counts[f'ssd_{operation.value}_submissions'] += 1
            return original_submit(request_id,operation,mapping,layer_range,destination,host_block_mapping)
        engine.submit_async_kv_transfer = counted_submit
        directory = worker._host_kv_directory
        begin,finish = directory.begin_use,directory.finish_use
        self.originals.extend([(directory,'begin_use',begin),(directory,'finish_use',finish)])
        def begin_use(key):
            result = begin(key)
            self.source_uses[key] += 1
            self.counts['source_acquires'] += 1
            return result
        def finish_use(key):
            if self.source_uses[key] <= 0:
                raise ValueError('source released without matching actual use')
            result = finish(key)
            self.source_uses[key] -= 1
            self.counts['source_releases'] += 1
            return result
        directory.begin_use, directory.finish_use = begin_use,finish_use

    def begin(self, model_input):
        ids = list(model_input.request_ids_to_seq_ids)
        self.counts['model_forwards'] += 1
        self.counts['single_query_forwards'] += int(all(n == 1 for n in model_input.query_lens))
        positions = model_input.input_positions.detach().cpu()
        meta = model_input.attn_metadata
        intervals = request_slices(ids,model_input.query_lens,model_input.seq_lens,
            positions,meta.num_prefills,meta.num_prefill_tokens,meta.num_decode_tokens)
        self.shared_slots = {}
        for rid in ids:
            self.immutable.setdefault(rid,int(positions[intervals[rid]][0])//16)
        self.sampling_ids = [rid for rid,group in zip(ids,model_input.sampling_metadata.seq_groups) if group.do_sample]
        self.frame = {'ids':ids,'positions':positions,'intervals':intervals,
                      'steps':{rid:end-len(self.prompts[int(rid.split('-')[-1])])+1
                               for rid,end in zip(ids,model_input.seq_lens)}}
        for rid,step in self.frame['steps'].items():
            self.max_steps[rid] = max(self.max_steps[rid],step)
        # Retain all replay inputs; this file is explicitly diagnostic, not formal timing.
        with (self.directory/'long_inputs.jsonl').open('a') as stream:
            stream.write(json.dumps({'ids':ids,'positions':positions.tolist(),
                'input_tokens':model_input.input_tokens.cpu().tolist(),'steps':self.frame['steps'],
                'query_lens':list(model_input.query_lens),'seq_lens':list(model_input.seq_lens)})+'\n')

    def capture_logits(self, module, inputs, output):
        if self.frame is None:
            return
        if output.shape[0] != len(self.sampling_ids):
            raise ValueError('diagnostic logit rows differ from sampling metadata')
        for row,rid in enumerate(self.sampling_ids):
            step = self.frame['steps'][rid]
            if step not in self.checkpoints:
                continue
            logits = output[row].detach().float().cpu()
            scores,tokens = logits.topk(5)
            record = {'rid':rid,'step':step,'top5_tokens':tokens.tolist(),
                      'top5_logits':scores.tolist(),'top2_margin':float(scores[0]-scores[1])}
            with (self.directory/'long_logits.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            torch.save(logits,self.directory/f'logits_{rid}_{step}.pt')

    def capture(self, layer, inputs):
        if self.frame is None:
            return
        _, key, value = inputs[:3]
        index = layer.impl._layer_index_from_name(layer.layer_name)
        for rid in self.frame['ids']:
            row = self.frame['intervals'][rid]
            positions = self.frame['positions'][row]
            if (rid,index) not in self.logical:
                self.logical[rid,index] = tuple(x.clone() for x in self.sources[int(rid.split('-')[-1]),index])
            logical = self.logical[rid,index]
            self.logical[rid,index] = update_logical(logical,
                key[row].reshape(-1,layer.num_kv_heads,layer.head_size),
                value[row].reshape(-1,layer.num_kv_heads,layer.head_size), int(positions[0]))

    def observe(self, original):
        def forward(layer, query, key, value, cache, metadata, *args, **kwargs):
            from vllm.core.custom_schedulers.hierarchical_io.barrier import (
                get_active_layer_request_ids, get_active_sparse_kv_selections,
                get_active_sparse_kv_mapping, get_active_sparse_kv_blocks,
                get_active_sparse_kv_device_selection)
            from vllm.attention.ops.paged_attn import PagedAttention
            ids = get_active_layer_request_ids()
            selections = get_active_sparse_kv_selections()
            # Noncompact batched backend recursively invokes its single-request path.
            compact = selections is not None and all(
                s.compact_kv_length is not None for s in selections.values())
            output = original(layer,query,key,value,cache,metadata,*args,**kwargs)
            if self.frame is None or (len(ids)>1 and not compact):
                return output
            index = layer.impl._layer_index_from_name(layer.layer_name)
            evictions = self.worker._host_stats['gpu_slot_evictions']
            first_after_reclaim = evictions>0 and self.evictions==0
            self.evictions = evictions
            self.peak = max(self.peak, len(self.worker._gpu_reserved_slots.get((0,index),())))
            if not first_after_reclaim and not any(self.frame['steps'][rid] in self.checkpoints for rid in ids):
                return output
            for row_index,rid in enumerate(ids):
                step = self.frame['steps'][rid]
                interval = self.frame['intervals'][rid]
                positions = self.frame['positions'][interval]
                start, end = int(positions[0]), int(positions[-1])+1
                if compact:
                    selected = selections[rid]
                    blocks = selected.block_indices
                    full = [-1]*((end+15)//16)
                    for b,slot in zip(blocks,selected.block_table[0]):
                        full[b] = slot
                    table = torch.tensor([full],device=query.device,dtype=torch.int32)
                    slots = torch.tensor(selected.slot_mapping,device=query.device,dtype=torch.int64)
                    q, out = query[row_index:row_index+1], output[row_index:row_index+1]
                    device_selection = selected.device_selection
                else:
                    table,slots = get_active_sparse_kv_mapping()
                    blocks = tuple(get_active_sparse_kv_blocks())
                    device_selection = get_active_sparse_kv_device_selection()
                    q,out = query,output
                logical = self.logical[rid,index]
                physical = check_mapping(table,slots,blocks,device_selection.logical_block_indices,
                    device_selection.count,positions,cache.shape[1])
                for block,slot in zip(blocks,physical):
                    identity = ('common_prefix',block) if block<self.immutable[rid] else (rid,block)
                    previous = self.shared_slots.setdefault((index,slot),identity)
                    if previous != identity:
                        raise ValueError('long checkpoint private/incompatible physical alias')
                if step not in self.checkpoints and not first_after_reclaim:
                    continue
                kc,vc = PagedAttention.split_kv_cache(cache,layer.num_kv_heads,layer.head_size)
                check_cache_contents(kc,vc,table,blocks,end,logical)
                # Build representatives from independent logical inputs, never the
                # tested physical mapping or selected-block list.
                complete = start//16
                keys = logical[0][:complete*16].to(query.device).reshape(
                    complete,16,layer.num_kv_heads,layer.head_size//8,8).permute(0,2,3,1,4).contiguous()
                self.policy._cached_representatives(rid,index,keys,
                    torch.arange(complete,device=query.device),complete,query.device)
                expected = self.policy.select_actual_blocks_device(rid,index,
                    q.reshape(-1,layer.num_heads,layer.head_size),
                    self.sources[int(rid.split('-')[-1]),index][0].shape[0]//16,
                    start,end,16,32)
                expected_blocks = tuple(sorted(set(expected.history_blocks).union(range(start//16,(end+15)//16))))
                if tuple(blocks) != expected_blocks:
                    raise ValueError(f'long checkpoint independent selection mismatch: {rid} step={step} layer={index}')
                fp32 = logical_attention(q.reshape(-1,layer.num_heads,layer.head_size),
                    logical,expected_blocks,positions,layer.impl.scale)
                error = numerical_error(out.reshape_as(fp32),fp32)
                record = {'request_id':rid,'step':step,'layer':index,'blocks':blocks,
                    'physical':physical,'first_after_reclaim':first_after_reclaim,'fp32':error,
                    'selected_blocks':len(blocks),
                    'request_resident_blocks':len(self.worker._gpu_slot_mappings.get((0,rid,index),{})),
                    'layer_occupied_slots':len(self.worker._gpu_reserved_slots.get((0,index),())),
                    'pending_h2d_records':len(self.worker._pending_h2d_transfers),
                    'request_ready_records':len(self.worker._host_ready_blocks)}
                with (self.directory/'long_checks.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                self.checked.append((rid,step,index))
                if not error['passed']:
                    torch.save({'record':record,'query':q.cpu(),'logical_kv':logical,
                        'actual':out.cpu(),'expected':fp32.cpu()},self.directory/'first_long_failure.pt')
                    raise ValueError('long checkpoint FP32 mismatch')
            return output
        return forward

    def close(self):
        for handle in self.handles:
            handle.remove()
        for obj,name,original in reversed(self.originals):
            setattr(obj,name,staticmethod(original) if obj is self.attention_class else original)
        attention_gpu_ms = []
        for start,end in self.attention_events:
            end.synchronize()
            attention_gpu_ms.append(start.elapsed_time(end))
        fence_gpu_ms = []
        for stream,start,end in self.fence_timings:
            end.synchronize()
            fence_gpu_ms.append({'stream':stream,'milliseconds':start.elapsed_time(end)})
        report = {'performance_valid':False,'counts':dict(self.counts),
            'gpu_slots_peak_observed_at_attention':self.peak,'evictions':self.evictions,
            'checked_layer_steps':self.checked,'requested_output_steps':self.checkpoints,
            'unreached_checkpoints':{rid:[s for s in self.checkpoints if s>last] for rid,last in self.max_steps.items()},
            'scope':'independent logical KV/selection/local FP32 checkpoints; intervening QKV captured; not all-step independent logits',
            'outstanding_source_uses':{str(k):v for k,v in self.source_uses.items() if v},
            'bytes_per_layer_block':self.block_bytes,
            'checkpoint_attention_gpu_event_ms':attention_gpu_ms,
            'gpu_timing_scope':'sampled Attention API intervals on existing compute stream; excludes independent FP32 checks; not whole-run GPU time',
            'dirty_d2h_bytes':self.counts['dirty_d2h_blocks_layers']*self.block_bytes,
            'new_effective_kv_bytes':self.counts['new_effective_kv_tokens_layers']*self.block_bytes//16,
            'h2d_bytes':self.counts['h2d_blocks_layers']*self.block_bytes,
            'immutable_d2h_bytes':self.counts['immutable_d2h_blocks_layers']*self.block_bytes,
            'effective_kv_note':'computed valid KV, including /0 prompt-tail recomputation and natural-task suffix',
            'expected_paged_attention_calls_for_all_single_query':28*self.counts['single_query_forwards'] if self.counts['model_forwards']==self.counts['single_query_forwards'] else None,
            'checkpoint_reuse_fence_gpu_event_ms':fence_gpu_ms,
            'fence_note':'sampled existing slot-reuse dependency intervals, including event/enqueue overhead; streams can overlap, do not add to service time'}
        (self.directory/'long_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
