# Paper Reproduction Experiments

This tree contains isolated strategy adapters on top of the existing vLLM and
GranuleKV integration. Every adapter has three responsibilities kept separate:

* selector: which logical KV blocks are needed;
* prefetcher: which layer windows are staged and activated;
* scheduler: the order of request descriptions.

Adapters emit the production `SparseKVAccessPlan`, `PrefetchPlan`, and
`AsyncKVTransferRequest` descriptions. Actual submission and completion remain
with the production scheduler and GranuleKV connector. No paper runtime is
copied here, and no native GranuleKV code is modified by these experiments.

## Execution policy

Run one smoke check and at most one formal comparison per strategy. A smoke
check validates plan construction only; a formal run must use the local vLLM
launch path and record real SSD markers, request lifecycle, overlap, and
correctness metrics. Missing measurements must remain unset in the result.

The first Quest/HiSparse/SolidAttention selectors are semantic boundaries. A
faithful paper implementation must provide explicit per-layer block choices;
the built-in proxy is only useful for checking the adapter and metrics wiring.

Formal wrappers require `VLLM_PAPER_REPRODUCTION_COMMAND` and execute the
existing local benchmark command verbatim. They do not invent a launch command
or write synthetic metrics.

## Backend boundary

The adapters use the existing `submit_request`, `query_request`,
`complete_request`, and `cancel_request` integration from the production path.
The old `stage_plan` entry remains only as a compatibility surface; current
layerwise experiments keep plans in the scheduler/worker and submit a window
only when it becomes active. Adapters must not add a native protocol, MPS
mechanism, completion path, or parallel transfer state machine. Blockers belong
in `common/open_issues.md` and the affected strategy is skipped.
