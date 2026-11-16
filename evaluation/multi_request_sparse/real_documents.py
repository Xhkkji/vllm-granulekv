"""Reproducible real-document inputs and metrics; no serving state changes."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = '/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct'
SOURCE = '/home/xhk/llm-inference/datasets/tutti/organized/qwen25/full/all.jsonl'
VERSIONS = {'leval': 'cd34b050269148aed75acbbe4a599873ad0f37e9',
            'loogle': '6734382215bea3a63f055ffd7873b2967b6a2477'}


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def split_prompts(prompts):
    if len(prompts) != 2 or not all(prompts):
        raise ValueError('requires two nonempty prompts')
    common = 0
    for a, b in zip(*prompts):
        if a != b:
            break
        common += 1
    common = common // 16 * 16
    if common <= 576:
        raise ValueError('common prefix too short for the existing initial prediction')
    return prompts[0][:common], [p[common:] for p in prompts]


def load_scenario(path, name):
    manifest = json.loads(Path(path).read_text())
    scenario = next(s for s in manifest['scenarios'] if s['name'] == name)
    prompts = scenario['prompt_token_ids']
    if list(map(token_hash, prompts)) != scenario['token_sha256']:
        raise ValueError('prompt token hash mismatch')
    prefix, suffixes = split_prompts(prompts)
    if len(prefix) != scenario['common_prefix_tokens']:
        raise ValueError('common prefix length mismatch')
    return manifest, scenario, prefix, suffixes


def live_physical_ids(mapping, begin, end, capacity):
    ids = [mapping[b] for b in range(begin,end)]
    if len(set(ids)) != len(ids) or any(slot < 0 or slot >= capacity for slot in ids):
        raise ValueError('invalid private KV mapping in short numerical gate')
    return ids


def decode_segments(timestamps):
    # No interpolation for multi-token output deliveries: report their count.
    rows = []
    previous_count, previous_time = 0, None
    points = []
    for now, count in timestamps:
        if count <= previous_count:
            continue
        points.append((count, now, None if previous_time is None else now-previous_time,
                       count-previous_count))
        previous_count, previous_time = count, now
    for lo, hi in ((1,64), (65,128), (129,256), (257,512)):
        group = [p for p in points if lo <= p[0] <= hi]
        latencies = sorted(p[2] for p in group if p[2] is not None and p[3] == 1)
        def percentile(q):
            if not latencies:
                return None
            pos = (len(latencies)-1)*q
            a = int(pos)
            return latencies[a] + (latencies[min(a+1,len(latencies)-1)]-latencies[a])*(pos-a)
        rows.append({'range': [lo,hi], 'outputs_observed': sum(p[3] for p in group),
                     'first_ms': group[0][1] if group else None,
                     'last_ms': group[-1][1] if group else None,
                     'itl_p50_ms': percentile(.5), 'itl_p95_ms': percentile(.95),
                     'multi_token_deliveries': sum(p[3] != 1 for p in group)})
    return rows


def build(output, model=MODEL, source=SOURCE):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    wanted = {('leval','natural_question',11), ('leval','narrative_qa',13),
              ('leval','narrative_qa',19), ('loogle','shortdep_cloze',1)}
    groups = {key: [] for key in wanted}
    with open(source) as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['source_dataset'],row['source_task'],row['source_record_idx'])
            if key in groups:
                groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r['source_question_idx'])
        if not rows:
            raise ValueError('source record missing')
    scenarios = []
    specs = [('fixed32k',('leval','natural_question',11),32768),
             ('fixed128k',('leval','narrative_qa',13),131072),
             ('natural32k',('leval','natural_question',11),None),
             ('natural35k',('loogle','shortdep_cloze',1),None),
             ('natural76k',('leval','narrative_qa',19),None)]
    for name, key, count in specs:
        rows = groups[key]
        first = rows[0]
        second = next(r for r in rows[1:] if r['input'] != first['input'])
        chosen = [first,second]
        if count:
            tokens = tokenizer.encode(first['context'],add_special_tokens=False)
            if len(tokens) < count:
                raise ValueError('document is shorter than requested slice')
            prompts = [tokens[:count],tokens[:count]]
        else:
            prompts = [tokenizer.encode(r['prompt'],add_special_tokens=False) for r in chosen]
        prefix, suffixes = split_prompts(prompts)
        maximum = max(map(len,prompts))
        capacity = 4*((maximum+33)//16)
        needed = sum((len(p)+512+15)//16 for p in prompts)
        if capacity < needed:
            raise ValueError('existing CPU capacity does not cover two growing requests')
        scenarios.append({'name':name, 'kind':'fixed' if count else 'natural',
            'source_dataset':key[0], 'source_task':key[1], 'source_record_idx':key[2],
            'source_version':VERSIONS[key[0]], 'source_ids':[r['_id'] for r in chosen],
            'source_question_indices':[r['source_question_idx'] for r in chosen],
            'questions':None if count else [r['input'] for r in chosen],
            'answers':None if count else [r['answers'] for r in chosen],
            'slice_rule':f'first {count} context tokens; no padding or repetition' if count else 'complete stored Passage/Question/Answer prompt; tokenize before splitting',
            'prompt_token_ids':prompts, 'token_sha256':list(map(token_hash,prompts)),
            'input_lengths':list(map(len,prompts)), 'common_prefix_tokens':len(prefix),
            'suffix_lengths':list(map(len,suffixes)), 'max_output_tokens':512,
            'ignore_eos':bool(count), 'cpu_slots':capacity,
            'two_request_growth_blocks_upper_bound':needed,
            'max_model_len':maximum+512+16})
    manifest = {'model':model,'tokenizer_files_sha256':{
        f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(model).glob('*token*') if f.is_file()},
        'source_manifest':source,'source_manifest_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'paper_alignment':'local workload inspired by SolidAttention output512 and Tutti real documents; not a full reproduction',
        'scenarios':scenarios}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(manifest,ensure_ascii=False)+'\n')
    print(json.dumps([{k:v for k,v in s.items() if k not in ('prompt_token_ids','questions','answers')} for s in scenarios],indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    build(args.output)
