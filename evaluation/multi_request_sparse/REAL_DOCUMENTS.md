# 真实文档长 decode 实验

生产代码检查点为 `43778dd0424651a2804a8a8ff7c25ade41a7a041`，已推送到
`origin/xhk/bam-GranuleKV-GPU_CPU`。旧可运行版 `245a49dfb` 是同一分支的历史提交，
生产代码等同于 `401f4416b`。GranuleKV 检查点 `e3975913` 无新改动，未创建空提交。
本目录后续实验改动保持本地，生产 Worker、scheduler、Attention 未改。

本轮结果、旧版128K停止现场、跨版本差异及未完成覆盖见
[真实文档长decode结果](20261115真实文档长decode结果.md)。旧版失败后不要直接重启全矩阵或用失败耗时计算加速比。

实验借鉴 SolidAttention 的最大输出512尺度和 Tutti 的真实文档多问题复用，
不构成论文完整复现。硬件/模型保持 Qwen2.5-7B-Instruct FP16、V100S 32GB、
每层2135 GPU slots、block16/window4/lead1/每头预算32。

## 重跑

在 vllm-granulekv 根目录执行，使用现有 pytorch-vllm 环境：

```bash
python evaluation/multi_request_sparse/real_documents.py --output /绝对路径/manifest.json
python evaluation/multi_request_sparse/run_real_documents.py \
  --manifest /绝对路径/manifest.json --scenario fixed32k --cache cold \
  --output-dir /绝对路径/新的结果目录
```

场景名为 `fixed32k`、`fixed128k`、`natural32k`、`natural35k`、`natural76k`；
缓存为 `cold` 或 `hot`。每次调用先跑两token独立门禁，再跑对应缓存的顺序、同时提交样本，
各一次。门禁失败直接停止，不进入性能采样。默认保留生产 I/O 超时120秒，
外层服务时间1800秒、无输出进展120秒；不在失败后放宽。新目录不会覆盖已有证据。

固定输入取真实文档正文前32768/131072 token，不循环、不拼接。自然输入使用完整
Passage/Question/Answer，先完整分词，再计算公共前缀并向下对齐16，保留原剩余token。
manifest 保存数据版本、样本/问题编号、tokenizer文件与token序列哈希。
自然场景名仅为便于调用的近似标签，报告必须使用 manifest 的实际长度。

固定场景忽略EOS，生成512 token；自然场景正常EOS，最多512。两token门禁强制输出2，
保持既有56层步/请求覆盖。该门禁不等于512步全量数值验证。
每次正式样本及顺序B之前清GPU prefix，CPU在请求对开始时按冷/热设置；
执行期间不清缓存、不凑batch。初始预测仍由原测试入口注入，实际选集不固定。

结果的 `performance` 只包含正式样本；`runs` 同时保留populate、参考、门禁、warmup。
正式时间包含输出、写回及排空；准备另计。保存全部输出token、EOS结束原因、
forward形状和分段ITL，分段为1–64/65–128/129–256/257–512。
同次交付多个token时不伪造逐token延迟。自然任务吞吐按实际输出token数计算。

本轮原始证据目录：
`GranuleKV/vllm_evaluation/tutti_layerwise/result/20261115_real_documents/`。
原始manifest、逻辑KV、日志与大体积结果不纳入git。

## 停止规则

数值、引用、owner/fence、旧generation、安全回收、传输死锁或无进展失败时，
保存首次现场，停止相关实验；不修改生产逻辑使测试通过。测试入口自身错误可以最小修复，
保留原失败记录。未完成样本不能计算跨版本加速比，不能用诊断耗时填正式表。

## 长输出诊断入口

通过 `run.sh` 传入 `--long-diagnostic --modes batch --cache-states cold`（或hot）
可在同一短输出门禁之后单独运行检查点诊断。它持续从模型post-RoPE输入保存逻辑KV，
在1/2/16/17/64/128/256/512及首次回收后的消费点，独立生成代表向量与选集、检查实际物理映射、
精确KV和局部FP32 Attention。它不使用被测选集强制参考一致。

诊断Q/K/V使用当前forward输入，不能证明非检查点的全模型误差传播或512步独立logits；
两token独立全模型门禁与长输出检查点覆盖分别报告。未到达的检查点不记为通过。
诊断包装会同步/拷贝数据、计数真实PagedAttention与D2H分类，不进入正式性能表。

## 旧版隔离对照

旧版源码目录：`/tmp/real_documents_baseline_245a49dfb`（detached worktree，HEAD `245a49dfb`）。
只复制测试侧benchmark、validation和真实数据入口；Python生产目录不移植新版修复。
复用未修改的编译算子和GranuleKV bridge，保留源码SHA与native库SHA证据。
通过 `REAL_SOURCE_ROOT=/tmp/real_documents_baseline_245a49dfb` 调用同一launcher。
若测试入口有后续修正，运行前应同步测试文件并记录diff；不能修改旧版生产代码。
旧版512-token失败时保留无完整基线，不使用64-token耗时外推512-token速度。
