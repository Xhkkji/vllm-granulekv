# SolidAttention-Style Adapter

This boundary exposes init/local/dynamic choices as an access plan. A future
runner may attach prior-iteration history and correction statistics, but no
second transfer state machine or replacement attention runtime is introduced.

The current adapter provides a resident-only `SolidAttentionPolicy`. It keeps
initial blocks and recent local blocks, scores the remaining immutable prefix
on the GPU, and returns logical block indices through the shared
`select_blocks_device()` bridge. It also has an opt-in per-head CUDA runtime
under `evaluation/paper_reproduction/quest/runtime`; that runtime reads the
native vLLM key/value cache directly and supports Qwen-style 28-query-head /
4-KV-head GQA. It is a SolidAttention-style block consumer, not the paper's
attention-inner CUDA implementation.

The runtime is enabled only by
`VLLM_GRANULEKV_SPARSE_QUEST_RUNTIME_ENABLE=1` together with resident-only
XFormers execution. The ordinary bridge and dense paths are unchanged. The
runtime is correctness-complete for the supported FP16 layout, but the current
single-warp-per-head consumer is retained as an experimental comparison path;
on the V100S/Qwen2.5-7B 8K smoke it is slower than the existing compact bridge.

## Prediction Miss and Correction

SolidAttention-style speculative access can be expressed with the existing
logical block plan:

1. Use the previous iteration's information to predict the next set of KV
   blocks and prefetch those blocks.
2. When the current query arrives, run the actual sparse selection.
3. Compute `missing = actual_selection - resident_blocks`.
4. Read the missing blocks through the existing restore path before attention
   consumes them.
5. Record speculative hits, speculative misses, correction reads, and wasted
   prefetched blocks.

The first implementation boundary is layer/block level. It does not add an
attention-inner microtask DAG, a replacement attention runtime, or a second
transfer state machine. Prediction, correction, residency checks, and
statistics remain policy-independent; SolidAttention only supplies the
prediction and actual block selections. GranuleKV native protocol, CUDA IPC,
descriptor handling, mapping, and completion are unchanged.

This correction path is intentionally separate from the first Quest attention
experiment. Quest is currently evaluated with all KV blocks resident on GPU;
dynamic SSD restore and prediction/correction are a later experiment.
