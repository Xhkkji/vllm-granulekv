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

    checks = {'rtol': 0.02, 'atol': 0.02, 'kv_comparison': 'exact_sha256',
              'layers_compared': 0, 'logits_compared': 0,
              'attention_max_abs_error': 0.0, 'logits_max_abs_error': 0.0}
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
    if (expected_records.keys() != actual_records.keys()
            or len(expected_records) != 28 * output_tokens):
        raise RuntimeError('layer/step coverage differs or contains duplicates')
    for key, ref in expected_records.items():
        got = actual_records[key]
        for field in ('attention_blocks', 'kv_fingerprints'):
            if not ref.get(field) or ref[field] != got.get(field):
                raise RuntimeError(f'{field} mismatch at step/layer {key}')
        for field, metric in (('attention_output', 'attention'), ('logits', 'logits')):
            if field == 'logits' and field not in ref and field not in got:
                continue
            if field not in ref or field not in got:
                raise RuntimeError(f'missing {field} at step/layer {key}')
            a = SolidAttentionPolicy._tensor_from_payload(ref[field]).float()
            b = SolidAttentionPolicy._tensor_from_payload(got[field]).float()
            if (a.shape != b.shape or not a.numel()
                    or not torch.allclose(a, b, rtol=0.02, atol=0.02)):
                raise RuntimeError(f'{field} numerical mismatch at step/layer {key}')
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
    args = parser.parse_args()
    if (args.prefix_tokens % 16 or args.prefix_tokens <= 36 * 16
            or args.suffix_tokens != 0 or args.output_tokens != 2):
        parser.error('initial gate requires aligned prefix >576, suffix=0, output=2')
    result = {'config': vars(args) | {'output': str(args.output)},
              'environment': {k: v for k, v in os.environ.items()
                              if k.startswith('VLLM_')}, 'runs': [],
              'status': 'started'}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    save()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.utils import Device
        from evaluation.paper_reproduction.quest.scripts.vllm_sparse_e2e import (
            _seed_static_sparse_plan)
        from vllm.core.custom_schedulers.hierarchical_io import (
            get_active_layer_request_ids, get_sparse_kv_policy)

        forward_batches = []
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        def prompt(text, count):
            seed = tokenizer.encode(text, add_special_tokens=False)
            return (seed * ((count + len(seed) - 1) // len(seed)))[:count]

        prefixes = [prompt('The experiment studies memory and inference. ',
                           args.prefix_tokens)]
        prefixes.append(prefixes[0] if args.case == 'same' else prompt(
            'A separate document describes oceans and geography. ', args.prefix_tokens))
        suffix = prompt(' Summarize the preceding document. ', args.suffix_tokens)
        max_len = args.prefix_tokens + args.suffix_tokens + args.output_tokens + 16
        result['engine_config'] = {
            'max_model_len': max_len, 'max_num_seqs': 2,
            'max_num_batched_tokens': 1024, 'num_gpu_blocks_override': 2135,
            'swap_space_gib': 32, 'gpu_memory_utilization': 0.85,
            'seed': 1234, 'dtype': 'half',
            'scheduler': 'vllm.core.async_kv_scheduler.AsyncKVScheduler'}
        save()
        llm = LLM(model=args.model, dtype='half', max_model_len=max_len,
                  gpu_memory_utilization=0.85, swap_space=32,
                  enable_prefix_caching=True, enable_chunked_prefill=True,
                  scheduler_cls='vllm.core.async_kv_scheduler.AsyncKVScheduler',
                  enforce_eager=True, disable_async_output_proc=True,
                  max_num_seqs=2, max_num_batched_tokens=1024,
                  num_gpu_blocks_override=2135,
                  seed=1234)
        engine = llm.llm_engine
        worker = engine.model_executor.driver_worker.worker
        execute = worker.model_runner.execute_model

        def traced_execute(*pos, **kw):
            model_input = kw.get('model_input', pos[0] if pos else None)
            ids = list(model_input.request_ids_to_seq_ids or {})
            forward_batches.append(ids)
            with (args.output.parent / 'forwards.jsonl').open('a') as trace:
                trace.write(json.dumps({
                    'request_ids': ids, 'seq_lens': model_input.seq_lens,
                    'num_prefills': model_input.attn_metadata.num_prefills,
                    'num_prefill_tokens': model_input.attn_metadata.num_prefill_tokens,
                    'num_decode_tokens': model_input.attn_metadata.num_decode_tokens,
                }) + '\n')
            return execute(*pos, **kw)

        worker.model_runner.execute_model = traced_execute
        sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                  ignore_eos=True)

        def run(label, prompts, validate=False):
            forward_batches.clear()
            llm.collective_rpc('get_sparse_kv_stats', kwargs={'reset': True})
            record = {'label': label, 'request_ids': [], 'ttft_ms': {},
                      'token_ids': {}, 'forward_batches': forward_batches}
            result['runs'].append(record)
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
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.outputs and output.outputs[0].token_ids:
                        now = time.perf_counter()
                        record['ttft_ms'].setdefault(output.request_id, (now-start)*1000)
                        record['token_ids'][output.request_id] = list(output.outputs[0].token_ids)
                        if output.finished:
                            last_output = now
                if time.perf_counter() - start > 300:
                    raise TimeoutError(f'{label} exceeded 300 seconds')
            end = time.perf_counter()
            record['elapsed_with_drain_ms'] = (end-start)*1000
            record['after_last_output_ms'] = (end-last_output)*1000
            record['max_forward_batch'] = max(map(len, forward_batches), default=0)
            record['forward_batches'] = list(forward_batches)
            record['stats'] = llm.collective_rpc('get_sparse_kv_stats')[0]
            if validate:
                record['validation_reports'] = {
                    rid: worker.get_sparse_validation_report(rid, reset=True)
                    for rid in record['request_ids']}
            save()
            print(json.dumps({k: v for k, v in record.items()
                              if k not in ('stats', 'forward_batches',
                                           'validation_reports')}), flush=True)
            return record

        for index, prefix in enumerate(prefixes[:1] if args.case == 'same' else prefixes):
            run(f'populate-{index}', [prefix])
            key = hashlib.sha256(repr((tuple(prefix), None, 16,
                                      len(prefix) // 16)).encode()).hexdigest()
            seeded = worker.seed_sparse_page_metadata(key, key, len(prefix) // 16)
            result.setdefault('metadata_layers_seeded', []).append(seeded)
            if seeded != 28:
                raise RuntimeError(f'populate metadata incomplete: {seeded}/28 layers')
        refs = []
        for index, prefix in enumerate(prefixes):
            if not llm.reset_prefix_cache(Device.GPU):
                raise RuntimeError('GPU prefix reset failed')
            _seed_static_sparse_plan(llm, prefix, 16, 28, 32)
            refs.append(run(f'reference-{index}', [prefix + suffix], validate=True))
        if not llm.reset_prefix_cache(Device.GPU):
            raise RuntimeError('GPU prefix reset failed')
        for prefix in prefixes:
            _seed_static_sparse_plan(llm, prefix, 16, 28, 32)
        batch = run('batch', [p + suffix for p in prefixes], validate=True)
        if batch['max_forward_batch'] != 2:
            raise RuntimeError('requests did not execute in a shared forward')
        for index, ref in enumerate(refs):
            expected = ref['token_ids'][f'reference-{index}-0']
            if batch['token_ids'].get(f'batch-{index}') != expected:
                raise RuntimeError(f'output mismatch for batch-{index}')
            result.setdefault('comparisons', []).append(compare_reports(
                ref['validation_reports'][f'reference-{index}-0'],
                batch['validation_reports'][f'batch-{index}'], args.output_tokens))
        result['status'] = 'numerical_check_passed'
        result['validation_scope'] = 'same-forward IDs, selected blocks, exact KV hashes, attention, logits, output tokens'
        result['performance_valid'] = False
        result['performance_exclusion'] = 'numerical recording and forward tracing enabled'
        result['cache_state'] = 'CPU cache warmed by populate and sequential references'
        save()
    except BaseException:
        result['status'] = 'failed'
        result['performance_valid'] = False
        result['performance_exclusion'] = 'correctness gate failed'
        result['error'] = traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
