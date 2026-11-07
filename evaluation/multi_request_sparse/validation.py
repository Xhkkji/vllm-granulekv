"""Test-only logical KV oracle and same-batch reference for the 1K gate.

No scheduler/transfer callbacks are used by the reference. Its initial KV comes
from populate's post-RoPE K/V inputs, before any cache write or SSD store.
"""
from copy import copy
from dataclasses import replace
from pathlib import Path
from types import MethodType
import json

import torch

BLOCK_SIZE = 16


def request_slices(ids, query_lens, seq_lens, positions, num_prefills,
                   num_prefill_tokens, num_decode_tokens):
    if (len(ids) != len(set(ids)) or len(ids) != len(query_lens)
            or len(ids) != len(seq_lens) or not 0 <= num_prefills <= len(ids)):
        raise ValueError('request/sequence metadata mismatch')
    if (sum(query_lens[:num_prefills]) != num_prefill_tokens
            or sum(query_lens[num_prefills:]) != num_decode_tokens
            or sum(query_lens) != len(positions)):
        raise ValueError('query intervals do not cover forward tokens')
    result, offset = {}, 0
    for i, (rid, length, end) in enumerate(zip(ids, query_lens, seq_lens)):
        if length <= 0 or end < length or (i >= num_prefills and length != 1):
            raise ValueError('invalid prefill/decode query length')
        stop = offset + length
        if positions[offset:stop].tolist() != list(range(end - length, end)):
            raise ValueError('query positions do not match request boundary')
        result[rid] = slice(offset, stop)
        offset = stop
    return result


def check_mapping(table, slots, blocks, indices, count, positions, cache_blocks):
    table, slots = table.flatten().cpu().long(), slots.flatten().cpu().long()
    indices = indices.flatten().cpu().long()
    if (not blocks or list(blocks) != sorted(set(blocks))
            or count != len(blocks) or count > indices.numel()
            or indices[:count].tolist() != list(blocks)):
        raise ValueError('selection valid length/content mismatch')
    if min(blocks) < 0 or max(blocks) >= table.numel():
        raise ValueError('selected logical block outside table')
    physical = table[list(blocks)]
    if (physical.min() < 0 or physical.max() >= cache_blocks
            or len(set(physical.tolist())) != len(blocks)):
        raise ValueError('invalid or aliased selected physical blocks')
    current = torch.div(positions.cpu(), BLOCK_SIZE, rounding_mode='floor')
    if not set(current.tolist()).issubset(blocks):
        raise ValueError('current KV blocks missing from selection')
    expected = table[current] * BLOCK_SIZE + positions.cpu() % BLOCK_SIZE
    if not torch.equal(slots, expected):
        raise ValueError('write slots disagree with logical token positions')
    return physical.tolist()


def cache_tokens(key_cache, value_cache, table, blocks, sequence_length):
    """Read packed storage without using the backend's gather implementation."""
    physical = table.flatten()[list(blocks)].long()
    k = key_cache.index_select(0, physical)
    v = value_cache.index_select(0, physical)
    k = k.permute(0, 3, 1, 2, 4).reshape(-1, k.shape[1], k.shape[2] * k.shape[4])
    v = v.permute(0, 3, 1, 2).reshape(-1, v.shape[1], v.shape[2])
    logical = torch.tensor([b * BLOCK_SIZE + t for b in blocks
                            for t in range(BLOCK_SIZE)], device=k.device)
    valid = logical < sequence_length
    return k[valid], v[valid], logical[valid]


def check_cache_contents(key_cache, value_cache, table, blocks, length, logical_kv):
    k, v, positions = cache_tokens(key_cache, value_cache, table, blocks, length)
    expected_k, expected_v = (x[positions.cpu()] for x in logical_kv)
    if not torch.equal(k.cpu(), expected_k) or not torch.equal(v.cpu(), expected_v):
        raise ValueError('mapped KV bytes differ from independent logical source')


def numerical_error(actual, expected):
    if actual.shape != expected.shape or not actual.numel():
        raise ValueError('numerical comparison shape/coverage mismatch')
    a, b = actual.float(), expected.float().to(actual.device)
    finite = torch.isfinite(a).all() and torch.isfinite(b).all()
    close = torch.isclose(a, b, rtol=.02, atol=.02)
    return {'passed': bool(finite and close.all()),
            'max_abs_error': float((a - b).abs().max()),
            'outside_tolerance': int((~close).sum())}


def logical_attention(query, logical_kv, blocks, positions, scale):
    """FP32 GQA reference with a mask in original logical token coordinates."""
    length = int(positions[-1]) + 1
    selected = torch.tensor([b * BLOCK_SIZE + t for b in blocks
                             for t in range(BLOCK_SIZE)
                             if b * BLOCK_SIZE + t < length], device=query.device)
    k, v = (x[selected.cpu()].to(query.device).float() for x in logical_kv)
    groups = query.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1).transpose(0, 1)
    v = v.repeat_interleave(groups, dim=1).transpose(0, 1)
    scores = query.float().transpose(0, 1) @ k.transpose(1, 2) * scale
    visible = selected[None, :] <= positions.to(query.device)[:, None]
    return (scores.masked_fill(~visible, float('-inf')).softmax(-1) @ v).transpose(0, 1)


def update_logical(kv, key, value, start):
    key, value = key.detach().cpu(), value.detach().cpu()
    if not 0 <= start <= kv[0].shape[0]:
        raise ValueError('logical KV has an uninitialized history gap')
    stop = start + key.shape[0]
    result = []
    for previous, new in zip(kv, (key, value)):
        if stop > previous.shape[0]:
            previous = torch.cat((previous, torch.zeros_like(new[:stop - previous.shape[0]])))
        previous[start:stop] = new
        result.append(previous)
    return tuple(result)


def live_payload(policy, key_cache, value_cache, table, start, length):
    blocks = list(range(start, (length + BLOCK_SIZE - 1) // BLOCK_SIZE))
    k, v, _ = cache_tokens(key_cache, value_cache, table, blocks, length)
    # Preserve the existing report's packed key dimension convention.
    return {'live_key': policy._tensor_payload(k.reshape(k.shape[0], k.shape[1], -1, 8)),
            'live_value': policy._tensor_payload(v), 'immutable_blocks': start}


class BatchReference:
    def __init__(self, worker, prefixes, directory):
        from vllm.attention.layer import Attention
        self.worker = worker
        self.model = worker.model_runner.get_model()
        self.prefixes = prefixes
        self.directory = Path(directory)
        self.populate_sources = {}
        self.sources = {}
        self.actual_live = {}
        self.reference_live = {}
        self.frames = []
        self.current = None
        self.reference_active = False
        self.report = {'source': 'populate post-RoPE inputs before KV/SSD write',
                       'mapping_checks': 0, 'logical_kv_checks': 0,
                       'fp32_attention_checks': 0, 'fp32_attention_max_abs_error': 0.,
                       'passed': True, 'failures': []}
        self.shared_slots = {}
        self.immutable = {}
        self.handles, self.original_backends = [], []
        self.layers = [m for m in self.model.modules() if isinstance(m, Attention)]
        for layer in self.layers:
            self.handles.append(layer.register_forward_pre_hook(self._capture_inputs))
            original = layer.impl.forward
            self.original_backends.append((layer.impl, original))
            layer.impl.forward = self._observed_backend(original)

    def begin_forward(self, model_input):
        ids = list(model_input.request_ids_to_seq_ids or {})
        if any(len(v) != 1 for v in model_input.request_ids_to_seq_ids.values()):
            raise ValueError('gate requires one sequence per request')
        meta = model_input.attn_metadata
        positions = model_input.input_positions.detach().cpu().clone()
        intervals = request_slices(ids, model_input.query_lens, model_input.seq_lens,
                                  positions, meta.num_prefills, meta.num_prefill_tokens,
                                  meta.num_decode_tokens)
        if len(model_input.input_tokens) != len(positions):
            raise ValueError('input tokens do not cover query positions')
        sampling_indices = model_input.sampling_metadata.selected_token_indices.detach().clone()
        if sampling_indices.cpu().tolist() != [intervals[r].stop - 1 for r in ids]:
            raise ValueError('sampling rows do not match request query ends')
        frame = {'ids': ids, 'query_lens': list(model_input.query_lens),
                 'seq_lens': list(model_input.seq_lens), 'positions': positions,
                 'input_tokens': model_input.input_tokens.detach().clone(),
                 'intervals': intervals, 'metadata': copy(meta),
                 'sampling_indices': sampling_indices}
        self.current = frame
        self.shared_slots = {}
        if ids and all(r.startswith('batch-') for r in ids):
            frame['index'] = len(self.frames)
            self.frames.append(frame)
            for i, rid in enumerate(ids):
                self.immutable.setdefault(rid, (frame['seq_lens'][i] - frame['query_lens'][i]) // BLOCK_SIZE)
        return {'request_ids': ids, 'query_lens': frame['query_lens'],
                'seq_lens': frame['seq_lens'], 'positions': positions.tolist(),
                'input_tokens': frame['input_tokens'].cpu().tolist(),
                'sampling_indices': frame['sampling_indices'].cpu().tolist(),
                'num_prefills': meta.num_prefills,
                'num_prefill_tokens': meta.num_prefill_tokens,
                'num_decode_tokens': meta.num_decode_tokens}

    def _source(self, rid, layer):
        return self.sources[int(rid.split('-')[-1]), layer]

    def _capture_inputs(self, layer, inputs):
        if self.reference_active or self.current is None:
            return
        frame = self.current
        query, key, value = inputs[:3]
        if any(x.shape[0] != len(frame['positions']) for x in (query, key, value)):
            raise ValueError('Q/K/V row count differs from model input')
        index = layer.impl._layer_index_from_name(layer.layer_name)
        for rid in frame['ids']:
            if not rid.startswith('populate-'):
                continue
            interval = frame['intervals'][rid]
            prefix = int(rid.split('-')[2])
            positions = frame['positions'][interval]
            valid = positions < len(self.prefixes[prefix])
            if not valid.any():
                continue
            shape = (len(self.prefixes[prefix]), layer.num_kv_heads, layer.head_size)
            source = self.populate_sources.setdefault((rid, index), (
                torch.zeros(shape, dtype=key.dtype), torch.zeros(shape, dtype=value.dtype),
                torch.zeros(shape[0], dtype=torch.bool)))
            pos = positions[valid]
            source[0][pos] = key[interval].reshape(-1, *shape[1:]).detach().cpu()[valid]
            source[1][pos] = value[interval].reshape(-1, *shape[1:]).detach().cpu()[valid]
            source[2][pos] = True

    def seal_populate(self, rid, prefix):
        for layer in range(len(self.layers)):
            k, v, covered = self.populate_sources.pop((rid, layer))
            if not covered.all():
                raise ValueError('populate source has missing token positions')
            self.sources[prefix, layer] = (k, v)
            if self.prefixes[0] == self.prefixes[1]:
                self.sources[1, layer] = (k, v)
        torch.save(self.sources, self.directory / 'populate_kv.pt')

    def _observed_backend(self, original):
        def forward(layer, query, key, value, cache, metadata, *args, **kwargs):
            from vllm.core.custom_schedulers.hierarchical_io.barrier import (
                get_active_layer_request_ids, get_active_sparse_kv_selections,
                get_active_sparse_kv_mapping, get_active_sparse_kv_blocks,
                get_active_sparse_kv_device_selection)
            from vllm.attention.ops.paged_attn import PagedAttention
            ids = get_active_layer_request_ids()
            observe = (not self.reference_active and len(ids) == 1
                       and ids[0].startswith('batch-')
                       and get_active_sparse_kv_selections() is None)
            if not observe:
                return original(layer, query, key, value, cache, metadata, *args, **kwargs)
            rid = ids[0]
            frame = self.current
            index = layer.impl._layer_index_from_name(layer.layer_name)
            interval = frame['intervals'][rid]
            positions = frame['positions'][interval]
            n, end = len(positions), int(positions[-1]) + 1
            mapping = get_active_sparse_kv_mapping()
            if mapping is None:
                raise ValueError('bounded batch has no per-request physical mapping')
            table, slots = mapping
            selected = get_active_sparse_kv_device_selection()
            blocks = tuple(get_active_sparse_kv_blocks())
            physical = check_mapping(table, slots, blocks, selected.logical_block_indices,
                                     selected.count, positions, cache.shape[1])
            self.report['mapping_checks'] += 1
            for logical, slot in zip(blocks, physical):
                identity = ((tuple(self.prefixes[int(rid.split('-')[-1])]), logical)
                            if logical < self.immutable[rid] else (rid, logical))
                prior = self.shared_slots.setdefault((index, slot), identity)
                if prior != identity:
                    raise ValueError('physical slot shared by incompatible requests')
            logical = self.actual_live.setdefault((rid, index), tuple(x.clone() for x in self._source(rid, index)))
            logical = update_logical(logical, key.reshape(n, layer.num_kv_heads, layer.head_size),
                                     value.reshape(n, layer.num_kv_heads, layer.head_size), int(positions[0]))
            self.actual_live[rid, index] = logical
            output = original(layer, query, key, value, cache, metadata, *args, **kwargs)
            kc, vc = PagedAttention.split_kv_cache(cache, layer.num_kv_heads, layer.head_size)
            check_cache_contents(kc, vc, table, blocks, end, logical)
            self.report['logical_kv_checks'] += 1
            expected = logical_attention(query.reshape(n, layer.num_heads, layer.head_size),
                                         logical, blocks, positions, layer.impl.scale)
            error = numerical_error(output.reshape_as(expected), expected)
            self.report['fp32_attention_checks'] += 1
            self.report['fp32_attention_max_abs_error'] = max(self.report['fp32_attention_max_abs_error'], error['max_abs_error'])
            record = {'forward': frame['index'], 'request_id': rid, 'layer': index,
                      'logical_blocks': blocks, 'physical_blocks': physical,
                      'positions': positions.tolist(), 'fp32_attention': error}
            with (self.directory / 'mapping_checks.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if not error['passed']:
                self.report['passed'] = False
                self.report['failures'].append(record)
                torch.save({'query': query.cpu(), 'logical_kv': logical,
                            'actual': output.cpu(), 'expected': expected.cpu()},
                           self.directory / 'first_attention_failure.pt')
                raise ValueError('independent FP32 attention check failed')
            return output
        return forward

    def run(self):
        from evaluation.paper_reproduction.solidattention.adapter.solidattention import SolidAttentionPolicy
        from vllm.core.custom_schedulers.hierarchical_io import configure_sparse_kv_policy, get_sparse_kv_policy
        from vllm.core.custom_schedulers.hierarchical_io.barrier import activate_layer_barrier
        from vllm.forward_context import set_forward_context
        old_policy = get_sparse_kv_policy()
        policy = SolidAttentionPolicy(old_policy.init_blocks, old_policy.local_blocks)
        rids = list(dict.fromkeys(r for frame in self.frames for r in frame['ids']))
        for rid in rids:
            policy.configure_validation(rid, 'record', metadata={
                'capture_trace': True, 'capture_layer_data': True,
                'capture_kv_fingerprints': True, 'prefill_attention_steps': 1, 'num_layers': 28})
        record_attention = policy.record_validation_attention
        def capture(rid, layer, kc, vc, table, output, dense=None, *, sequence_length=None):
            record_attention(rid, layer, kc, vc, table, output, dense, sequence_length=sequence_length)
            state = policy._validation_states[rid]
            record = state.records_by_key[state.current_layer_keys[layer]]
            record.update(live_payload(policy, kc, vc, table, self.immutable[rid], sequence_length))
        policy.record_validation_attention = capture
        originals = [(layer, layer.forward) for layer in self.layers]
        hook = getattr(self.worker, '_sparse_validation_logits_hook', None)
        if hook is not None:
            hook.remove()
            self.worker._sparse_validation_logits_hook = None
        tokens = {rid: [] for rid in rids}
        self.reference_active = True
        try:
            configure_sparse_kv_policy(policy)
            for layer, _ in originals:
                layer.forward = MethodType(self._reference_attention, layer)
            with torch.inference_mode():
                for frame in self.frames:
                    self.reference_frame = frame
                    ids = frame['ids']
                    with activate_layer_barrier(lambda *a: None, virtual_engine=0,
                            request_ids=ids, sequence_lengths_by_request=dict(zip(ids, frame['seq_lens'])),
                            query_lengths_by_request=dict(zip(ids, frame['query_lens'])), block_size=BLOCK_SIZE):
                        with set_forward_context(frame['metadata'], self.worker.model_runner.vllm_config, 0):
                            hidden = self.model(input_ids=frame['input_tokens'],
                                                positions=frame['positions'].to(frame['input_tokens'].device))
                            logits = self.model.logits_processor(self.model.lm_head,
                                        hidden.index_select(0, frame['sampling_indices'].long()), None)
                        if logits.shape[0] != len(ids):
                            raise ValueError('reference logits/request coverage differs')
                        for row, rid in enumerate(ids):
                            policy.record_validation_logits(rid, logits[row:row + 1])
                            tokens[rid].append(int(logits[row].argmax()))
            reports = {rid: policy.validation_report(rid) for rid in rids}
            return {'validation_reports': reports, 'token_ids': tokens,
                    'forward_request_ids': [f['ids'] for f in self.frames]}
        finally:
            for layer, original in originals:
                layer.forward = original
            configure_sparse_kv_policy(old_policy)
            self.reference_active = False

    def _reference_attention(self, layer, query, key, value, output_shape=None):
        from vllm.attention.ops.paged_attn import PagedAttention
        from vllm.core.custom_schedulers.hierarchical_io import get_sparse_kv_policy
        from vllm.core.custom_schedulers.hierarchical_io.barrier import activate_sparse_kv_blocks, SparseKVLayerSelection
        policy = get_sparse_kv_policy()
        frame = self.reference_frame
        index = layer.impl._layer_index_from_name(layer.layer_name)
        output = torch.empty_like(query)
        for row, rid in enumerate(frame['ids']):
            interval = frame['intervals'][rid]
            positions = frame['positions'][interval]
            start, end, n = int(positions[0]), int(positions[-1]) + 1, len(positions)
            logical = self.reference_live.setdefault((rid, index), tuple(x.clone() for x in self._source(rid, index)))
            total = (end + BLOCK_SIZE - 1) // BLOCK_SIZE
            table = torch.arange(total - 1, -1, -1, dtype=torch.int32, device=query.device)[None, :]
            cache = torch.zeros(PagedAttention.get_kv_cache_shape(total, BLOCK_SIZE, layer.num_kv_heads, layer.head_size), dtype=key.dtype, device=query.device)
            kc, vc = PagedAttention.split_kv_cache(cache, layer.num_kv_heads, layer.head_size)
            # Explicit logical-to-physical copy; no production gather/transfer code.
            for b in range(total):
                valid = min(BLOCK_SIZE, logical[0].shape[0] - b * BLOCK_SIZE)
                if valid <= 0:
                    continue
                physical = total - 1 - b
                k, v = (x[b * BLOCK_SIZE:b * BLOCK_SIZE + valid].to(query.device) for x in logical)
                kc[physical, :, :, :valid, :] = k.reshape(valid, layer.num_kv_heads, -1, 8).permute(1, 2, 0, 3)
                vc[physical, :, :, :valid] = v.permute(1, 2, 0)
            policy._cached_representatives(rid, index, kc, table[0], start // BLOCK_SIZE, query.device)
            actual = policy.select_actual_blocks_device(rid, index,
                query[interval].view(n, layer.num_heads, layer.head_size),
                len(self.prefixes[int(rid.split('-')[-1])]) // BLOCK_SIZE,
                start, end, BLOCK_SIZE, 32)
            blocks = tuple(sorted(set(actual.history_blocks).union(range(start // BLOCK_SIZE, total))))
            slots = table[0, positions.to(query.device) // BLOCK_SIZE].long() * BLOCK_SIZE + positions.to(query.device) % BLOCK_SIZE
            tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=query.device)
            prefill = row < frame['metadata'].num_prefills
            meta = replace(frame['metadata'], num_prefills=int(prefill),
                num_prefill_tokens=n if prefill else 0, num_decode_tokens=0 if prefill else n,
                slot_mapping=slots, block_tables=table, seq_lens=[end], seq_lens_tensor=tensor([end]),
                query_start_loc=tensor([0, n]), seq_start_loc=tensor([0, end]),
                context_lens_tensor=tensor([start]), max_query_len=n, max_decode_query_len=1,
                max_prefill_seq_len=end if prefill else 0, max_decode_seq_len=0 if prefill else end,
                use_cuda_graph=False, _cached_prefill_metadata=None, _cached_decode_metadata=None)
            selection = SparseKVLayerSelection(blocks, actual.attention, True, table, slots)
            with activate_sparse_kv_blocks(selection, request_id=rid):
                output[interval] = layer.impl.forward(layer, query[interval], key[interval], value[interval], cache,
                                                      meta.single_token_sparse_prefill())
            self.reference_live[rid, index] = update_logical(logical,
                key[interval].reshape(n, layer.num_kv_heads, layer.head_size),
                value[interval].reshape(n, layer.num_kv_heads, layer.head_size), start)
        return output

    def close(self):
        for handle in self.handles:
            handle.remove()
        for impl, original in self.original_backends:
            impl.forward = original
