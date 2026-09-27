# SolidAttention-Style Adapter

This boundary exposes init/local/dynamic choices as an access plan. A future
runner may attach prior-iteration history and correction statistics, but no
second transfer state machine or replacement attention runtime is introduced.

The current adapter provides a resident-only `SolidAttentionPolicy`. It keeps
initial blocks and recent local blocks, scores the remaining immutable prefix
on the GPU, and returns logical block indices through the shared
`select_blocks_device()` bridge. The current implementation uses the existing
attention consumer and is a block-level approximation, not the paper's
attention-inner CUDA runtime.

## Prediction Miss and Correction

SolidAttention-style speculative access can be expressed with the existing
logical block plan:

1. Use the previous iteration's information to predict the next set of KV
   blocks and prefetch those blocks.
2. Restore either exact adjacent groups or a four-layer union through the
   existing layer-range request.
3. Let the worker residency directory verify the planned layer selection after
   the corresponding unit reaches `READY`.
4. Record prediction gaps separately from physical residency misses.

The first implementation boundary is layer/block level. It does not add an
attention-inner microtask DAG, a replacement attention runtime, or a second
transfer state machine. Prediction, correction, residency checks, and
statistics remain policy-independent; SolidAttention only supplies the
prediction and actual block selections. GranuleKV native protocol, CUDA IPC,
descriptor handling, mapping, and completion are unchanged.

The dynamic implementation computes the current-query actual prefix before
decode attention. With correction disabled, a prediction gap remains
fail-fast. The optional first correction path projects missing blocks from the
read-only admission catalog and submits a single-layer GranuleKV request
through the existing submit/poll/complete lifecycle:

```bash
export VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE=1
export VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE=1
export VLLM_GRANULEKV_SPARSE_CORRECTION_ENABLE=1
export VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE=1
export VLLM_GRANULEKV_SPARSE_SELECTED_BLOCKS_ENABLE=1
export VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS=0
```

The selected-list mode passes the current-query GPU selection directly from
correction to `forward_decode_selected`; the CPU tuple from the same selection
is used only for the prediction-gap and SSD correction decision. Setting exact
restore to `0` unions predictions inside each configured layer window while
keeping the per-layer attention sets exact. Prediction guard blocks expand
only the previous-query prefetch and never change the actual attention budget.
Correction remains unsupported with layer working-set ring overwrite mode.

Run the complete 8K exact/union/guard ablation with:

```bash
bash evaluation/paper_reproduction/solidattention/scripts/run_dynamic_ablation.sh
```

The runner writes per-case summaries plus `ablation_summary.json` and
`ablation_summary.csv`, and rejects output mismatches, residency errors, or a
selected-list case that falls back to compact attention.

## Offline Prediction Oracle

The prediction oracle keeps the complete KV table available and compares the
previous-query prediction with the next query's actual prefix selection. It
does not start GranuleKV or perform SSD I/O:

```bash
PYTHONPATH=. /home/xhk/miniconda3/envs/pytorch-vllm/bin/python \
  evaluation/paper_reproduction/solidattention/scripts/prediction_oracle.py \
  --device cuda --prefix-blocks 512 --block-budget 32 --steps 32 \
  --output /tmp/solidattention-prediction.json
```

The result reports predicted, actual, hit, miss, and wasted prefix blocks.
Current experiments report prediction recall, prediction restore requests and
bytes, guard size, correction blocks/bytes, submit time, wait time, and total
correction blocking separately.
