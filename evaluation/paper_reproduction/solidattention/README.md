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
