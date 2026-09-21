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
2. Group adjacent layers only when their predicted prefix block sets are
   identical, and submit each group through the existing layer-range request.
3. Let the worker residency directory verify the planned layer selection after
   the corresponding unit reaches `READY`.
4. Record prediction gaps separately from physical residency misses.

The first implementation boundary is layer/block level. It does not add an
attention-inner microtask DAG, a replacement attention runtime, or a second
transfer state machine. Prediction, correction, residency checks, and
statistics remain policy-independent; SolidAttention only supplies the
prediction and actual block selections. GranuleKV native protocol, CUDA IPC,
descriptor handling, mapping, and completion are unchanged.

The first dynamic implementation is fail-fast: it does not issue an on-demand
correction read and does not silently expand the request to dense attention.
Enable exact grouped restore only for a SolidAttention dynamic experiment:

```bash
export VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE=1
export VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE=1
```

The exact mode reuses the existing rolling layer barrier and GranuleKV request
lifecycle. The selected prefix mapping excludes the live suffix; the suffix is
kept only in the per-layer attention working set.

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
Dynamic restore must be treated as a fail-fast experiment when the current
query selects a prefix block that is not resident; it does not silently
expand to dense attention or issue an on-demand correction read.
