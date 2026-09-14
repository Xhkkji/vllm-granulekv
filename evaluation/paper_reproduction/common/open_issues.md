# Open Issues

This file records blockers found during reproduction. A strategy must be
skipped rather than changing the GranuleKV native protocol when it requires a
new descriptor format, MPS/completion semantics, a second transfer state
machine, or a different SSD-to-GPU path.

| Strategy | Missing capability | Trigger | GranuleKV backend issue | Minimal follow-up |
| --- | --- | --- | --- | --- |
| Quest | Scheduler-side producer for page representatives and previous-step selection | The attention hook observes queries, but the current prefix restore is planned before Worker feedback; no dynamic plan rebuild RPC exists. The validated consumer path now keeps the live suffix separate from the SSD prefix mapping. | No | Add one vLLM control-plane hook that consumes a CPU `SparseKVAccessPlan` before the next restore; keep GranuleKV request API unchanged |
| HiSparse | Faithful SGLang selection/residency semantics are not yet wired into vLLM | Local SGLang code exists separately, but the current adapter only emits a proxy plan | No | Port only the policy output into `SparseKVAccessPlan`; do not copy the SGLang runtime |
| SolidAttention | Attention-inner speculative microtask and correction semantics | Existing API has layer-window requests, not an attention-kernel microtask stream | Not established | Skip until the policy can be expressed as ordinary staged requests; do not modify the native protocol |
| Quest stage 1 resident smoke | GPU execution context unavailable to the test process | vLLM failed in `torch.cuda.mem_get_info()` before model execution: V100 is in `Exclusive_Process` mode and an existing `nvidia-cuda-mps-server` owns the device; the resident script was not attached to that MPS pipe | No | Run the resident script either after the existing MPS service is released, or explicitly as an MPS client using the service's configured pipe directory; do not change Quest or GranuleKV code for this environment condition |
