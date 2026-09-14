# Quest Adapter

This directory isolates the first Quest selector experiment. It implements the
page-level query-aware score in pure PyTorch and emits the existing
`SparseKVAccessPlan` and layer-window `PrefetchPlan`. It does not copy a Quest
runtime or change GranuleKV.

The production integration boundary is now deliberately small:

- `QuestPolicy` and `QuestPageIndex` stay in this directory and are loaded as
  an optional policy plugin.
- `vllm/attention/ops/sparse_kv.py` is the user-owned generic logical-to-
  physical block-table bridge.
- `TorchSDPABackendImpl` has a user-owned batch-one decode branch that consumes
  the active logical block set exposed by the layer barrier.
- `consumer_enabled` is opt-in. Without it, non-dense plans remain the old
  profiling-only path and cannot enter normal generation.
- A consumer-enabled restore exposes the selected immutable prefix blocks plus
  the live request suffix. The suffix is allocated and filled by the normal
  vLLM scheduler; it is never added to the GranuleKV SSD mapping.

The current vLLM scheduler still obtains its restore plan from the configured
plan source. The generic attention hook observes live decode queries, but
supplying page representatives and rebuilding the next scheduler restore plan
remain an external producer concern. The adapter does not fabricate those
tensors or silently fall back to dense attention when the sparse context is
missing.

The reproduction is now being advanced in three isolated stages. The first
stage uses the full vLLM GPU KV table and validates Quest sparse attention
without GranuleKV, MPS, layer barriers, or hierarchical prefetch. The later
GranuleKV full-restore and layer-wise stages reuse the same policy and
consumer, so storage failures cannot be confused with sparse-attention
failures.

1. Resident-only: all vLLM KV blocks remain on GPU and Quest selection is
   validated without GranuleKV.
2. Full restore: GranuleKV restores the complete prefix before Quest decode.
3. Layer-wise restore: GranuleKV restores layer windows behind the shared
   Worker barrier.

This is a real vLLM sparse-consumer validation, but it is not yet a claim that
previous-step queries reduce the SSD restore bytes. That requires a scheduler
control-plane producer that rebuilds a future restore request before the next
admission decision; the missing capability is tracked in
`../common/open_issues.md`.

The dynamic entry point is:

```python
selector.select_from_query(
    query, page_representatives, block_budget, layer_index
)
```

The optional dynamic SSD restore smoke is enabled explicitly. It performs a
dense warmup, uses the resulting decode history to build the next prefix plan,
and restores a third request through the selected block mapping:

```bash
VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE=1 \
bash evaluation/paper_reproduction/quest/scripts/run_vllm_sparse_e2e.sh
```

The scheduler still reserves the complete prefix target table in this first
version, but the GranuleKV read mapping is projected to the predicted prefix
blocks. Check `console.log` for `dynamic_restore=true` and the per-window
`selected_blocks` counts before claiming an SSD read reduction.

For multi-head input, each head selects its top-k pages and the adapter uses
their sorted union. This is an explicit integration approximation because the
shared `SparseKVAccessPlan` stores one block set per layer. The result is a
logical page/block selection only; page size is a selector parameter and is
not silently treated as a GranuleKV physical block size.

`QuestSelector.select(num_layers, num_blocks)` remains available for existing
plan smoke and static experiments. Its tail selection is a proxy and must not
be reported as an official Quest result.

## Selector-only commands

From the repository root:

```bash
bash evaluation/paper_reproduction/quest/scripts/run_selector_smoke.sh
bash evaluation/paper_reproduction/quest/scripts/run_selector_eval.sh
```

The evaluation writes a JSON summary below `quest/results/`, which is ignored
as a runtime artifact. It reports exact page-selection recall and relative
attention output error on a synthetic PyTorch workload. These numbers are
mechanism checks, not the paper's performance or accuracy results.

Run the Quest-style selector sweep with page size 16, token budgets 512/1024/
2048/4096, context lengths 8K/16K/32K, and three deterministic seeds:

```bash
bash evaluation/paper_reproduction/quest/scripts/run_selector_sweep.sh
```

The script converts token budget to the block budget used by the adapter. The
JSON includes both values and the actual union size across query heads.

## Resident attention benchmark

The formal first-stage benchmark keeps the complete KV cache resident on GPU
and disables GranuleKV, MPS, layer barriers, and hierarchical prefetch:

```bash
bash evaluation/paper_reproduction/quest/scripts/run_attention_benchmark.sh
```

Each dense or Quest process initializes the model once, performs one warmup,
then runs five generations of 256 tokens. The result reports TTFT as the
`prefill_ms` proxy, decode latency per token, P50/P95 across measured runs,
throughput, GPU memory, output tokens, and the actual sparse block ratio when
the bridge statistics are available. It reports both cumulative block counts
and per-attention-call counts. Model initialization and tokenizer time are not
included in these measurements.

The benchmark uses the current vLLM/XFormers adapter rather than Quest's
official specialized CUDA kernel. A lower selected-block ratio is therefore a
selection result; an end-to-end speedup depends on the existing paged-
attention kernel and its fixed overhead.

## Resident quality checks

Run the paper-inspired Passkey sweep after the attention smoke has passed:

```bash
bash evaluation/paper_reproduction/quest/scripts/run_passkey.sh
```

For a small local LongBench TriviaQA check, use the prepared Qwen2.5 manifest:

```bash
bash evaluation/paper_reproduction/quest/scripts/run_longbench_triviaqa.sh
```

Both commands compare dense and Quest with all KV blocks resident on GPU. The
Passkey result reports retrieval accuracy. The JSONL runner reports an answer
substring screening score and keeps each generated answer; use the official
LongBench evaluator for a publishable TriviaQA score. PG-19 is deliberately
left as a later token-level loss experiment because this vLLM runner exposes
generation, not per-token cross-entropy.

## Decode consumer smoke

The restricted consumer closes the selector-to-consumer loop for batch-size-one
decode without modifying vLLM attention:

```bash
bash evaluation/paper_reproduction/quest/scripts/run_consumer_smoke.sh
```

It reads the current layer's block set from `SparseKVAccessPlan`, verifies that
the set is included in the active layer context, and computes selected-page
reference attention. Missing active blocks are rejected. This validates plan
consumption and residency checking only; it is not a production vLLM attention
backend and does not measure GranuleKV SSD or GPU performance.

The plan-only smoke remains available:

Run the plan-only smoke check from the repository root:

```bash
python -m evaluation.paper_reproduction.common.runner.plan_smoke \
  --strategy quest --output evaluation/paper_reproduction/quest/results/smoke/plan.json
```

Use `--prefetcher on_demand` for Quest synchronous/on-demand comparison and
`--prefetcher layer_wise --window N` for asynchronous layer prefetch.
