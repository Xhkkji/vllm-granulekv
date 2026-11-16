"""Summarize completed samples without inventing missing/failed comparisons."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(root):
    summaries = []
    for path in sorted(root.glob('*/summary.json')):
        result = json.loads(path.read_text())
        summaries.append((path, result))
    lines = ['# 真实文档长 decode 实验记录', '',
        '检查点：vLLM `43778dd04`、GranuleKV `e3975913`，已核对远端。旧可运行版为同一分支 `origin/xhk/bam-GranuleKV-GPU_CPU` 的历史提交 `245a49dfb`（生产代码等同 `401f4416b`）。实验改动仅在测试侧，尚未再次提交推送。', '',
        '单次样本仅作趋势；顺序为两个请求各自含排空时间之和。准备、warmup和独立诊断不计入正式服务时间。', '',
        '| 实验配置：Qwen2.5-7B-Instruct FP16／V100S 32GB；两请求；block16/window4/lead1/每头预算32；2135 GPU slots/层；CPU容量见各行；正式计时含输出、写回、排空，关闭数值记录和profiler | 输入token A/B | 输出设置 | CPU状态／slots | 模式 | 实际输出 A/B | 服务秒 | 顺序÷同时提交 |',
        '| --- | --- | --- | --- | --- | --- | ---: | ---: |']
    samples, gates, failures = [], [], []
    for path,r in summaries:
        name = path.parent.name
        sc = r.get('scenario',{})
        gates.append({'run':name,'status':r['status'],'checks':r.get('independent_checks'),
                      'comparisons':r.get('comparisons'), 'gate_cache':r['config'].get('gate_cache')})
        if r.get('error'):
            failures.append({'run':name,'error':r['error'],'evidence':str(path)})
        if r['config'].get('long_diagnostic'):
            continue
        bylabel = {s['label']:s for s in r['runs']}
        for p in r.get('performance',[]):
            observed = [bylabel[label] for label in p['samples']]
            entry = {'run':name,'scenario':sc.get('name'), 'cache':p['cache'],
                'mode':p['mode'], 'service_seconds':p['batch_with_drain_ms']/1000,
                'preparation_ms':p['preparation_ms'], 'sample_status':r['status'],
                'tokens':p['actual_output_tokens'], 'samples':observed}
            samples.append(entry)
            matches = {q['mode']:q for q in r['performance'] if q['cache']==p['cache']}
            speedup = (matches['sequential']['batch_with_drain_ms']/matches['batch']['batch_with_drain_ms']
                       if 'sequential' in matches and 'batch' in matches else None)
            ratio = f'{speedup:.3f}×' if p['mode']=='batch' and speedup else '—'
            if speedup and matches['sequential']['actual_output_tokens'] != matches['batch']['actual_output_tokens']:
                ratio += '（输出数不同）' if p['mode']=='batch' else ''
            lengths = '/'.join(map(str,sc.get('input_lengths',[])))
            outputs = '固定512' if sc.get('ignore_eos') else 'EOS，最多512'
            tokens = '/'.join(map(str,p['actual_output_tokens']))
            lines.append(f"| {name} | {lengths} | {outputs} | {p['cache']}／{sc.get('cpu_slots')} | {p['mode']} | {tokens} | {entry['service_seconds']:.3f} | {ratio} |")
    lines += ['', '自然任务若顺序与batch输出数量不同，服务时间比含EOS/生成轨迹差异，不代表相同工作量加速；实际token吞吐另存samples/summary。所有表格均为单次样本。', '', '## TTFT、decode 与排空', '',
        '| 配置同上；单位秒 | 请求 | TTFT | 首token至末token | 末输出后排空（服务级） | forward大小频数 | 回收次数 |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: |']
    for entry in samples:
        for sample in entry['samples']:
            for rid in sample['request_ids']:
                ttft = sample['ttft_ms'][rid]/1000
                decode = sample['output_finished_ms'][rid]/1000-ttft
                counts = dict(Counter(map(len,sample['forward_batches'])))
                lines.append(f"| {entry['run']}／{sample['label']} | {rid} | {ttft:.3f} | {decode:.3f} | {sample['after_last_output_ms']/1000:.3f} | {counts} | {sample['stats']['gpu_slot_evictions']} |")
    lines += ['', '## 分段 ITL', '',
        '| 配置同上；单位毫秒 | 请求 | 输出段 | 已观察输出数 | ITL p50 | ITL p95 |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for entry in samples:
        for sample in entry['samples']:
            for rid,segments in sample.get('decode_segments',{}).items():
                for segment in segments:
                    if not segment['outputs_observed']:
                        continue
                    val = lambda k: '—' if segment[k] is None else f'{segment[k]:.2f}'
                    lines.append(f"| {entry['run']} | {rid} | {segment['range']} | {segment['outputs_observed']} | {val('itl_p50_ms')} | {val('itl_p95_ms')} |")
    lines += ['', '## 准备、吞吐与传输', '',
        '配置同首表。字节为现有统计的提交量，非独立测得的PCIe/SSD实际总线流量；顺序行汇总两个请求。准备不计入服务时间，populate/warmup另见原始runs。', '',
        '| 样本／模式 | 准备秒 | 实际输出token/s | H2D提交／字节 | D2H提交／字节 | SSD读字节 | SSD写提交／字节 |',
        '| --- | ---: | ---: | --- | --- | ---: | --- |']
    for entry in samples:
        total = lambda key: sum(s['stats'].get(key, 0) for s in entry['samples'])
        throughput = sum(entry['tokens']) / entry['service_seconds']
        lines.append(f"| {entry['run']}／{entry['mode']} | {entry['preparation_ms']/1000:.3f} | {throughput:.4f} | "
                     f"{total('host_gpu_submit_count')}／{total('host_gpu_copy_bytes')} | "
                     f"{total('gpu_to_host_submit_count')}／{total('gpu_to_host_bytes')} | "
                     f"{sum(s['ssd_read_bytes'] for s in entry['samples'])} | "
                     f"{total('host_to_ssd_submit_count')}／{total('host_to_ssd_bytes')} |")
    lines += ['', '## 正确性与覆盖', '']
    for gate in gates:
        check = gate.get('checks') or {}
        verified = gate.get('comparisons') and all(c['passed'] and c['tokens_match'] for c in gate['comparisons']) and check.get('passed')
        lines.append(f"- {gate['run']}：两token门禁{'通过' if verified else '尚未通过/未完成'}；逻辑KV {check.get('logical_kv_checks',0)}、局部FP32 {check.get('fp32_attention_checks',0)}；整体状态 `{gate['status']}`。")
    lines += ['', '两token门禁不代表512步全量数值验证。长输出检查点另见诊断结果；没有诊断结果的步骤不记为通过。回收次数为0的样本未覆盖容量压力。自然任务答案与参考保存在summary与manifest，不把独立计算正确性等同任务质量。', '', '## 失败现场', '']
    for failure in failures:
        lines += [f"- {failure['run']}：`{failure['evidence']}`", '```text', failure['error'], '```']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'samples.json').write_text(json.dumps(samples,ensure_ascii=False)+'\n')
    (root/'gates.json').write_text(json.dumps(gates,ensure_ascii=False,indent=2)+'\n')
    (root/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    summarize(args.root)
