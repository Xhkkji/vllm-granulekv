"""Correctness gate and timings for real AsyncKVScheduler request batches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def compare_reports(expected, actual, output_tokens):
    """Require complete coverage before comparing recorded numerical evidence.

    KV hashes are exact; attention/logits use the existing FP16 validation
    tolerance (rtol=atol=0.02). Missing records never count as a pass.
    """
    import torch
    from evaluation.paper_reproduction.solidattention.adapter.solidattention import (
        SolidAttentionPolicy)

    checks = {'rtol': 0.02, 'atol': 0.02, 'kv_comparison': 'immutable exact SHA256; recomputed/new KV FP16 tolerance',
              'layers_compared': 0, 'logits_compared': 0,
              'attention_max_abs_error': 0.0, 'logits_max_abs_error': 0.0,
              'passed': True, 'failures': []}
    for report in (expected, actual):
        if (report.get('attention_checks') != 28 * output_tokens
                or report.get('logit_checks') != output_tokens
                or len(report.get('records', [])) != 28 * output_tokens):
            raise RuntimeError('incomplete KV/attention/logit validation coverage')
        if any(value for key, value in report.items()
               if key.endswith('_mismatches') and isinstance(value, (int, float))):
            raise RuntimeError('validation report contains mismatches')
    expected_records = {(r['decode_step'], r['layer_index']): r
                        for r in expected['records']}
    actual_records = {(r['decode_step'], r['layer_index']): r
                      for r in actual['records']}
    required_keys = {(step, layer) for step in range(output_tokens) for layer in range(28)}
    if expected_records.keys() != required_keys or actual_records.keys() != required_keys:
        raise RuntimeError('layer/step coverage differs or contains duplicates')
    for records in (expected_records, actual_records):
        for key, record in records.items():
            blocks = record.get('attention_blocks')
            if (not blocks or blocks != sorted(set(blocks)) or min(blocks) < 0
                    or max(blocks) >= (record['sequence_length'] + record['block_size'] - 1) // record['block_size']):
                raise RuntimeError(f'invalid selection at {key}')
            fingerprints = record.get('kv_fingerprints', {})
            if any(str(block) not in fingerprints for block in blocks):
                raise RuntimeError(f'selected block fingerprint missing at {key}')
            if ('logits' in record) != (key[1] == 27):
                raise RuntimeError(f'logits coverage differs at {key}')
    checks['kv_max_abs_error'] = 0.0
    checks['selection_differences'] = []
    immutable_blocks = min(r['history_tokens'] // r['block_size'] for r in expected_records.values())
    for key, ref in expected_records.items():
        got = actual_records[key]
        for field in ('history_tokens', 'block_size', 'sequence_length'):
            if ref[field] != got[field]:
                raise RuntimeError(f'{field} mismatch at step/layer {key}')
        if ref['attention_blocks'] != got['attention_blocks']:
            checks['passed'] = False
            checks['failures'].append(f'attention_blocks mismatch at step/layer {key}')
            checks['selection_differences'].append({
                'step_layer': key,
                'expected_only': sorted(set(ref['attention_blocks']) - set(got['attention_blocks'])),
                'actual_only': sorted(set(got['attention_blocks']) - set(ref['attention_blocks']))})
        for block in set(ref['attention_blocks']).intersection(got['attention_blocks']):
            if block < immutable_blocks and ref['kv_fingerprints'][str(block)] != got['kv_fingerprints'][str(block)]:
                checks['passed'] = False
                checks['failures'].append(f'immutable KV mismatch at {key}, block {block}')
        for field, metric in (('live_key', 'kv'), ('live_value', 'kv'),
                              ('attention_output', 'attention'), ('logits', 'logits')):
            if field == 'logits' and field not in ref and field not in got:
                continue
            if field not in ref or field not in got:
                raise RuntimeError(f'missing {field} at step/layer {key}')
            a = SolidAttentionPolicy._tensor_from_payload(ref[field]).float()
            b = SolidAttentionPolicy._tensor_from_payload(got[field]).float()
            if a.shape != b.shape or not a.numel():
                raise RuntimeError(f'{field} shape mismatch at step/layer {key}')
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise RuntimeError(f'non-finite {field} at step/layer {key}')
            if not torch.allclose(a, b, rtol=0.02, atol=0.02):
                checks['passed'] = False
                checks['failures'].append(f'{field} numerical mismatch at step/layer {key}')
            checks[f'{field}_outside_tolerance'] = checks.get(f'{field}_outside_tolerance', 0) + int((~torch.isclose(a, b, rtol=0.02, atol=0.02)).sum())
            checks[f'{metric}_max_abs_error'] = max(
                checks[f'{metric}_max_abs_error'], (a - b).abs().max().item())
            if field == 'logits':
                checks['logits_compared'] += 1
        checks['layers_compared'] += 1
    if checks['logits_compared'] != output_tokens:
        raise RuntimeError('recorded logits do not cover every output step')
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--prefix-tokens', type=int, default=1024)
    parser.add_argument('--suffix-tokens', type=int, default=0)
    parser.add_argument('--output-tokens', type=int, default=2)
    parser.add_argument('--case', choices=['same', 'different'], default='same')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--performance', action='store_true')
    args = parser.parse_args()
    if (args.prefix_tokens % 16 or args.prefix_tokens <= 36 * 16
            or args.suffix_tokens < 0 or args.suffix_tokens > 1024 or args.output_tokens != 2):
        parser.error('requires aligned prefix >576, suffix in [0,1024], output=2')
    result = {'config': vars(args) | {'output': str(args.output)},
              'environment': {k: v for k, v in os.environ.items()
                              if k.startswith('VLLM_')}, 'runs': [],
              'status': 'started'}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    save()
    reference = None
    try:
        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.utils import Device
        from evaluation.paper_reproduction.quest.scripts.vllm_sparse_e2e import (
            _seed_static_sparse_plan)
        from vllm.core.custom_schedulers.hierarchical_io import (
            get_active_layer_request_ids, get_sparse_kv_policy)

        performance_active = False
        forward_batches = []
        forward_shapes = []
        native_reads = []
        live_kv = {}
        live_starts = {}
        policy = get_sparse_kv_policy()
        record_attention = policy.record_validation_attention
        def capture_attention(rid, layer, key_cache, value_cache, table, output,
                              dense=None, *, sequence_length=None):
            record_attention(rid, layer, key_cache, value_cache, table, output,
                             dense, sequence_length=sequence_length)
            state = policy._validation_states.get(rid)
            if state is None or state.mode == 'off':
                return
            record = state.records_by_key[state.current_layer_keys[layer]]
            begin = live_starts.setdefault(rid, record['history_tokens'] // record['block_size'])
            end = (sequence_length + 15) // 16
            physical = table[0, begin:end].long()
            live_kv[rid, record['decode_step'], layer] = (
                key_cache.index_select(0, physical).detach().cpu(),
                value_cache.index_select(0, physical).detach().cpu())
        policy.record_validation_attention = capture_attention
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        def prompt(text, count):
            seed = tokenizer.encode(text, add_special_tokens=False)
            return (seed * ((count + len(seed) - 1) // len(seed)))[:count]

        prefixes = [prompt('The experiment studies memory and inference. ',
                           args.prefix_tokens)]
        prefixes.append(prefixes[0] if args.case == 'same' else prompt(
            'A separate document describes oceans and geography. ', args.prefix_tokens))
        suffixes = [prompt(text, args.suffix_tokens) for text in (
            ' Summarize the preceding document. ', ' Explain the main subject of this text. ')]
        max_len = args.prefix_tokens + args.suffix_tokens + args.output_tokens + 16
        model_config = AutoConfig.from_pretrained(args.model)
        # max_model_len alone does not grow Qwen2's RoPE lookup table. Include
        # decode positions without changing the base frequencies/scaling.
        rope_positions = max(max_len, model_config.max_position_embeddings)
        hf_overrides = {'max_position_embeddings': rope_positions}
        result['engine_config'] = {
            'hf_overrides': hf_overrides,
            'rope_theta': model_config.rope_theta,
            'rope_scaling': getattr(model_config, 'rope_scaling', None),
            'max_model_len': max_len, 'max_num_seqs': 2,
            'max_num_batched_tokens': 2048, 'num_gpu_blocks_override': 2135,
            'swap_space_gib': 32, 'gpu_memory_utilization': 0.85,
            'seed': 1234, 'dtype': 'half',
            'scheduler': 'vllm.core.async_kv_scheduler.AsyncKVScheduler'}
        save()
        llm = LLM(model=args.model, dtype='half', max_model_len=max_len,
                  hf_overrides=hf_overrides,
                  gpu_memory_utilization=0.85, swap_space=32,
                  enable_prefix_caching=True, enable_chunked_prefill=True,
                  scheduler_cls='vllm.core.async_kv_scheduler.AsyncKVScheduler',
                  enforce_eager=True, disable_async_output_proc=True,
                  max_num_seqs=2, max_num_batched_tokens=2048,
                  num_gpu_blocks_override=2135,
                  seed=1234)
        engine = llm.llm_engine
        worker = engine.model_executor.driver_worker.worker
        from validation import BatchReference
        reference = BatchReference(worker, prefixes, args.output.parent)
        result['independent_checks'] = reference.report
        execute = worker.model_runner.execute_model

        def traced_execute(*pos, **kw):
            model_input = kw.get('model_input', pos[0] if pos else None)
            ids = list(model_input.request_ids_to_seq_ids or {})
            forward_batches.append(ids)
            if performance_active:
                forward_shapes.append({'ids': ids,
                                       'query_lens': list(model_input.query_lens),
                                       'seq_lens': list(model_input.seq_lens)})
            if not performance_active:
                frame = reference.begin_forward(model_input)
                with (args.output.parent / 'forwards.jsonl').open('a') as trace:
                    trace.write(json.dumps(frame) + '\n')
            return execute(*pos, **kw)

        worker.model_runner.execute_model = traced_execute
        sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                  ignore_eos=True)

        def run(label, prompts, validate=False):
            forward_batches.clear()
            forward_shapes.clear()
            native_reads.clear()
            llm.collective_rpc('get_sparse_kv_stats', kwargs={'reset': True})
            record = {'label': label, 'request_ids': [], 'ttft_ms': {},
                      'token_ids': {}, 'forward_batches': forward_batches}
            result['runs'].append(record)
            if not performance_active:
                save()
            start = time.perf_counter()
            for index, tokens in enumerate(prompts):
                rid = f'{label}-{index}'
                record['request_ids'].append(rid)
                if validate:
                    worker.configure_sparse_validation(rid, 'record', metadata={
                        'capture_trace': True, 'capture_layer_data': True,
                        'retain_report': True, 'capture_kv_fingerprints': True,
                        'prefill_attention_steps': 1, 'num_layers': 28,
                    })
                engine.add_request(rid, {'prompt_token_ids': tokens}, sampling)
            if validate:
                # The production validation hook captures only the last configured
                # request. Split logits by actual forward IDs in this experiment.
                worker._sparse_validation_logits_hook.remove()

                def record_logits(module, inputs, output):
                    ids = get_active_layer_request_ids()
                    if output.shape[0] != len(ids):
                        raise RuntimeError('logit rows do not match forward request IDs')
                    for row, rid in enumerate(ids):
                        get_sparse_kv_policy().record_validation_logits(
                            rid, output[row:row + 1])

                worker._sparse_validation_logits_hook = (
                    worker.model_runner.get_model().logits_processor
                    .register_forward_hook(record_logits))
            last_output = start
            finished_at = {}
            while (engine.has_unfinished_requests() or (performance_active and any(
                    scheduler._finished_requests_ids for scheduler in engine.scheduler))):
                for output in engine.step():
                    if output.outputs and output.outputs[0].token_ids:
                        now = time.perf_counter()
                        record['ttft_ms'].setdefault(output.request_id, (now-start)*1000)
                        record['token_ids'][output.request_id] = list(output.outputs[0].token_ids)
                        if output.finished:
                            last_output = now
                            finished_at[output.request_id] = now
                if time.perf_counter() - start > 300:
                    raise TimeoutError(f'{label} exceeded 300 seconds')
            if performance_active:
                # Include outstanding copies in service time, then require all
                # native reads and writes to have completed before taking time.
                for copy in tuple(worker._pending_h2d_transfers.values()):
                    copy.event.synchronize()
                worker._poll_host_transfers(0)
                assert_drained()
            end = time.perf_counter()
            record['elapsed_with_drain_ms'] = (end-start)*1000
            record['after_last_output_ms'] = (end-last_output)*1000
            record['after_first_token_ms'] = {rid: record['elapsed_with_drain_ms'] - ttft for rid, ttft in record['ttft_ms'].items()}
            record['output_finished_ms'] = {rid: (t-start)*1000 for rid, t in finished_at.items()}
            record['output_tokens_per_second'] = sum(map(len, record['token_ids'].values())) / (end-start)
            record['max_forward_batch'] = max(map(len, forward_batches), default=0)
            record['forward_batches'] = list(forward_batches)
            record['stats'] = llm.collective_rpc('get_sparse_kv_stats')[0]
            if performance_active:
                record['forward_shapes'] = list(forward_shapes)
                record['native_reads'] = list(native_reads)
                record['ssd_read_requests'] = len(native_reads)
                record['ssd_read_bytes'] = sum(r['bytes'] for r in native_reads)
                stats = record['stats']
                if record['ssd_read_bytes'] != (stats['gpu_direct_read_bytes']
                                                 + stats['cpu_direct_read_bytes']):
                    raise RuntimeError('native and worker read byte counts differ')
                if any(stats[field] for field in ('host_transfer_errors',
                       'host_coverage_missing_blocks', 'gpu_slot_capacity_rejections',
                       'host_write_backlog')):
                    raise RuntimeError('invalid performance sample: transfer/capacity/backing failure')
            if validate:
                record['validation_reports'] = {
                    rid: worker.get_sparse_validation_report(rid, reset=True)
                    for rid in record['request_ids']}
                for rid, report in record['validation_reports'].items():
                    for layer_record in report['records']:
                        k, v = live_kv.pop((rid, layer_record['decode_step'], layer_record['layer_index']))
                        layer_record['immutable_blocks'] = live_starts[rid]
                        valid = layer_record['sequence_length'] - live_starts[rid] * 16
                        # Compare only valid positions; trailing block padding is not KV data.
                        key_tokens = k.permute(0, 3, 1, 2, 4).reshape(-1, k.shape[1], k.shape[2], k.shape[4])[:valid]
                        value_tokens = v.permute(0, 3, 1, 2).reshape(-1, v.shape[1], v.shape[2])[:valid]
                        layer_record['live_key'] = policy._tensor_payload(key_tokens)
                        layer_record['live_value'] = policy._tensor_payload(value_tokens)
            if not performance_active:
                save()
                print(json.dumps({k: v for k, v in record.items()
                                  if k not in ('stats', 'forward_batches',
                                               'validation_reports')}), flush=True)
            return record

        populate_generation = 0
        def populate_initial_prefixes():
            nonlocal populate_generation
            for index, prefix in enumerate(prefixes[:1] if args.case == 'same' else prefixes):
                populated = run(f'populate-{populate_generation}-{index}', [prefix])
                if not performance_active:
                    reference.seal_populate(populated['request_ids'][0], index)
                key = hashlib.sha256(repr((tuple(prefix), None, 16,
                                          len(prefix) // 16)).encode()).hexdigest()
                seeded = worker.seed_sparse_page_metadata(key, key, len(prefix) // 16)
                result.setdefault('metadata_layers_seeded', []).append(seeded)
                if seeded != 28:
                    raise RuntimeError(f'populate metadata incomplete: {seeded}/28 layers')
            populate_generation += 1

        def reset_suffix_storage():
            # References write their suffix to SSD. Recreate the intended initial
            # prefix so later samples do not silently restore a longer prompt.
            if args.suffix_tokens:
                if not llm.reset_prefix_cache():
                    raise RuntimeError('prefix cache reset failed')
                populate_initial_prefixes()

        populate_initial_prefixes()
        refs = []
        for index, prefix in enumerate(prefixes):
            if index:
                reset_suffix_storage()
            if not llm.reset_prefix_cache(Device.GPU):
                raise RuntimeError('GPU prefix reset failed')
            _seed_static_sparse_plan(llm, prefix, 16, 28, 32)
            refs.append(run(f'reference-{index}', [prefix + suffixes[index]], validate=True))
        reset_suffix_storage()
        if not llm.reset_prefix_cache(Device.GPU):
            raise RuntimeError('GPU prefix reset failed')
        if args.performance:
            # Exercise cold SSD->CPU completion under the numerical gate too.
            from vllm.core.custom_schedulers.hierarchical_io.residency import SharedHostKVDirectory
            while any(s._finished_requests_ids for s in engine.scheduler):
                engine.step()
            worker._poll_host_transfers(0)
            if (worker._host_async_kv_transfers or worker._pending_gpu_to_host
                    or any(e.in_flight or e.loading_layers or e.owners
                           for e in worker._host_kv_directory._entries.values())):
                raise RuntimeError('cold correctness gate requires drained transfers')
            worker._host_kv_directory = SharedHostKVDirectory(
                int(os.environ['VLLM_GRANULEKV_SPARSE_CPU_CACHE_BLOCKS']))
            worker._host_ready_blocks.clear()
            worker._host_staged_blocks.clear()
            worker._host_failed_blocks.clear()
        for prefix in prefixes:
            _seed_static_sparse_plan(llm, prefix, 16, 28, 32)
        batch = run('batch', [p + suffix for p, suffix in zip(prefixes, suffixes)], validate=True)
        if batch['max_forward_batch'] != 2:
            raise RuntimeError('requests did not execute in a shared forward')
        result['sequential_diagnostics'] = []
        for index, ref in enumerate(refs):
            comparison = compare_reports(
                ref['validation_reports'][f'reference-{index}-0'],
                batch['validation_reports'][f'batch-{index}'], args.output_tokens)
            comparison['tokens_match'] = (batch['token_ids'].get(f'batch-{index}') ==
                                          ref['token_ids'][f'reference-{index}-0'])
            result['sequential_diagnostics'].append(comparison)
        save()
        matched = reference.run()
        result['batch_reference'] = matched
        result['comparisons'] = []
        for rid in batch['request_ids']:
            comparison = compare_reports(matched['validation_reports'][rid],
                                         batch['validation_reports'][rid], args.output_tokens)
            comparison['tokens_match'] = matched['token_ids'][rid] == batch['token_ids'][rid]
            if not comparison['tokens_match']:
                comparison['passed'] = False
                comparison['failures'].append('generated token mismatch')
            result['comparisons'].append(comparison)
        expected_checks = 28 * args.output_tokens * len(batch['request_ids'])
        if any(reference.report[field] != expected_checks for field in (
                'mapping_checks', 'logical_kv_checks', 'fp32_attention_checks')):
            raise RuntimeError('incomplete independent check coverage')
        save()
        if not reference.report['passed'] or not all(check['passed'] for check in result['comparisons']):
            raise RuntimeError('same-batch correctness comparison failed; see comparisons')
        reference.close()
        result['status'] = 'numerical_check_passed'
        result['validation_scope'] = f'{args.prefix_tokens}-token same-batch independent logical KV reference; per-request mapping; FP32 local attention; selection/KV/attention/logits/tokens'
        result['reference_kind'] = 'same_batch_independent_cache'
        result['performance_valid'] = False
        result['performance_exclusion'] = 'numerical recording and forward tracing enabled'
        result['cache_state'] = ('cold CPU for numerical batch gate' if args.performance
                                 else 'CPU cache warmed by populate and sequential references')
        save()
        if args.performance:
            from vllm.core.custom_schedulers.hierarchical_io.residency import SharedHostKVDirectory
            from vllm.worker.cache_engine import CacheEngine
            import logging
            policy.record_validation_attention = record_attention
            logging.getLogger('vllm').setLevel(logging.WARNING)
            cache_engine = worker.cache_engine[0]
            block_bytes = CacheEngine.get_cache_block_size(
                worker.cache_config, worker.model_config, worker.parallel_config)
            submit = cache_engine.submit_async_kv_transfer

            def counted_submit(request_id, operation, mapping, layer_range=None,
                               destination='gpu', host_block_mapping=()):
                if operation.value == 'read':
                    layers = 28 if layer_range is None else layer_range[1] - layer_range[0]
                    native_reads.append({'id': request_id, 'destination': destination,
                                         'bytes': (len(mapping) + len(host_block_mapping))
                                         * block_bytes * layers // 28})
                return submit(request_id, operation, mapping, layer_range,
                              destination, host_block_mapping)

            cache_engine.submit_async_kv_transfer = counted_submit

            def assert_drained():
                pending = {name: list(map(str, getattr(worker, name))) for name in (
                    '_host_async_kv_transfers', '_pending_gpu_to_host',
                    '_pending_host_writes', '_deferred_gpu_transfers',
                    '_gpu_transfer_children', '_gpu_shared_waits')}
                pending['native'] = list(cache_engine.granulekv_connector._pending_transfers)
                pending['host_references'] = {str(k): {
                    'owners': sorted(e.owners), 'in_flight': e.in_flight,
                    'loading_layers': sorted(e.loading_layers)}
                    for k, e in worker._host_kv_directory._entries.items()
                    if e.in_flight or e.loading_layers or e.owners}
                if any(pending.values()):
                    result['undrained_boundary'] = pending
                    raise RuntimeError('performance boundary has outstanding transfer/owner; see undrained_boundary')

            def cache_snapshot():
                entries = worker._host_kv_directory._entries
                return {'slots': len(entries), 'ready_layer_blocks': sum(
                    len(e.ready_layers) for e in entries.values()),
                    'prefix_ready_layer_blocks': {
                        key: sum(len(e.ready_layers) for (prefix, _), e in entries.items()
                                 if prefix == key) for key in prefix_keys}}

            prefix_keys = {hashlib.sha256(repr((tuple(p), None, 16,
                                              len(p) // 16)).encode()).hexdigest()
                           for p in prefixes}
            capacity = int(os.environ['VLLM_GRANULEKV_SPARSE_CPU_CACHE_BLOCKS'])
            result['performance_method'] = {
                'repeats': 3, 'numeric_recording': False,
                'cpu_slots': capacity, 'cpu_bytes': capacity * block_bytes,
                'logical_backing_blocks_two_requests': 2 * (
                    (args.prefix_tokens + args.suffix_tokens + args.output_tokens + 15) // 16),
                'gpu_slots_per_layer': 2135,
                'boundary': 'request submission through outputs and all writeback/copy completion',
                'sequential_total': 'sum of two drained service intervals; GPU eviction and stats collection excluded',
                'cold': 'empty CPU directory at pair start; second sequential request may reuse first CPU copy',
                'hot': 'same workload warmup, rebuild initial SSD prefix for suffix case, retain CPU directory',
                'gpu': 'evict GPU prefix before batch and each sequential request',
                'ssd_read_requests': 'native submissions; mixed GPU/CPU arms counted once',
                'ssd_read_bytes': 'descriptor payload, not device-sector amplification'}
            performance_active = True
            workload = [p + suffix for p, suffix in zip(prefixes, suffixes)]

            def reset_restore(cold=False, rebuild=False):
                # vLLM delivers the final finished-ID cleanup on the next step.
                # A completed validation run may leave that notification queued.
                while any(s._finished_requests_ids for s in engine.scheduler):
                    engine.step()
                if rebuild:
                    reset_suffix_storage()
                worker._poll_host_transfers(0)
                assert_drained()
                if not llm.reset_prefix_cache(Device.GPU):
                    raise RuntimeError('GPU prefix reset failed')
                if cold:
                    worker._host_kv_directory = SharedHostKVDirectory(capacity)
                    worker._host_ready_blocks.clear()
                    worker._host_staged_blocks.clear()
                    worker._host_failed_blocks.clear()
                for prefix in prefixes:
                    _seed_static_sparse_plan(llm, prefix, 16, 28, 32)
            result['performance'] = []
            for cache in ('cold', 'hot'):
                for repeat in range(3):
                    for mode in ('sequential', 'batch'):
                        reset_restore(cold=True, rebuild=True)
                        if cache == 'hot':
                            run(f'warmup-{cache}-{repeat}-{mode}', workload)
                            reset_restore(rebuild=True)
                        initial_cpu = cache_snapshot()
                        if cache == 'cold' and initial_cpu['slots']:
                            raise RuntimeError('cold sample has populated CPU directory')
                        if cache == 'hot' and not all(initial_cpu['prefix_ready_layer_blocks'].values()):
                            raise RuntimeError('hot sample lacks ready prefix copies')
                        started = time.perf_counter()
                        if mode == 'batch':
                            samples = [run(f'perf-{cache}-{repeat}-{mode}', workload)]
                            if samples[0]['max_forward_batch'] != 2:
                                raise RuntimeError('performance sample did not execute batch=2')
                        else:
                            samples = []
                            for index, tokens in enumerate(workload):
                                if index:
                                    reset_restore()
                                samples.append(run(f'perf-{cache}-{repeat}-{mode}-{index}', [tokens]))
                        wall_ms = (time.perf_counter() - started) * 1000
                        elapsed = sum(s['elapsed_with_drain_ms'] for s in samples)
                        actual_tokens = [sample['token_ids'][rid] for sample in samples
                                         for rid in sample['request_ids']]
                        expected_tokens = ([batch['token_ids'][f'batch-{i}'] for i in range(2)]
                                           if mode == 'batch' else [refs[i]['token_ids'][f'reference-{i}-0'] for i in range(2)])
                        if actual_tokens != expected_tokens:
                            raise RuntimeError('performance output differs from validated workload')
                        result['performance'].append({
                            'cache': cache, 'repeat': repeat, 'mode': mode,
                            'batch_with_drain_ms': elapsed,
                            'wall_including_harness_ms': wall_ms,
                            'initial_cpu': initial_cpu, 'final_cpu': cache_snapshot(),
                            'output_tokens_per_second': 2 * args.output_tokens * 1000 / elapsed,
                            'samples': [sample['label'] for sample in samples]})
                        save()
            result['performance_valid'] = True
            result['status'] = 'performance_passed'
            result.pop('performance_exclusion', None)
            save()
    except BaseException:
        result['status'] = 'failed'
        result['performance_valid'] = False
        result['performance_exclusion'] = ('performance gate failed' if
            'performance_method' in result else 'correctness gate failed')
        result['error'] = traceback.format_exc()
        if reference is not None and not result.get('comparisons'):
            reference.report['passed'] = False
        save()
        raise
    finally:
        if reference is not None:
            reference.close()


if __name__ == '__main__':
    main()
