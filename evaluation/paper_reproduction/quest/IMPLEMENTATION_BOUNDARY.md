# Quest Implementation Boundary

This document separates upstream vLLM code, existing GranuleKV integration,
and code added for the Quest experiment.

| Area | Ownership |
| --- | --- |
| `vllm/attention/backends/*` dense paths | Upstream vLLM |
| `vllm/attention/ops/paged_attn.py` | Upstream vLLM |
| `vllm/core/custom_schedulers/hierarchical_io/*` existing layer lifecycle | GranuleKV/vLLM integration |
| `evaluation/paper_reproduction/quest/*` | User Quest adapter |
| `vllm/attention/ops/sparse_kv.py` | User generic sparse bridge |
| sparse branch in `torch_sdpa.py` | User minimal backend extension |

The Quest adapter owns page scoring, query history, CPU page metadata, and
selection metrics. It does not own GranuleKV requests, physical allocation,
layer barriers, MPS, NVMe descriptors, or attention kernels.

The generic policy hook is intentionally paper-independent. vLLM core loads a
plugin using `module:Class`; it must not import Quest directly. Later policies
can implement the same `select_blocks` and `observe_query` methods.

The sparse attention bridge maps logical sequence block indices to the physical
ids in the existing vLLM block table. It creates temporary metadata only: it
does not mutate allocator state, the original table, or the GranuleKV native
protocol.

For a prefix restore, logical blocks before the restore frontier are immutable
SSD-backed prefix blocks. Logical blocks after that frontier are the live local
suffix. Consumer residency includes both sets, while the GranuleKV mapping
contains only the SSD-backed set. This distinction prevents a local suffix from
being treated as an SSD read or being omitted from decode attention.

The current implementation is a first integration boundary, not an official
Quest CUDA runtime reproduction. Live query/page-index production and the
full vLLM/GranuleKV E2E launcher remain separate validation work.
