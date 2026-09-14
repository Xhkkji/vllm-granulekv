# Source Manifest

The adapter is a local plan boundary, not a copy of Quest source. The external
checkout at `/home/xhk/llm-inference/Quest` is reference material only and is
not a vLLM runtime dependency.

Current source reference:

- Paper: <https://arxiv.org/abs/2406.10774>
- Repository (SSH): `git@github.com:mit-han-lab/Quest.git`
- Repository URL: <https://github.com/mit-han-lab/Quest>
- Fixed commit: `01c1623bf9395009520874e989e29f683203b357` (checked 2026-09-06)
- License: MIT (`Quest/LICENSE`)
- Relevant upstream reference: `quest/models/QuestAttention.py` and
  `evaluation/quest_attention.py`

Local adaptation boundary:

- Implements pure PyTorch page scoring, top-k selection, synthetic attention
  comparison, CPU page metadata, and a minimal policy plugin.
- Uses the existing `SparseKVAccessPlan` for logical per-layer blocks.
- Does not copy upstream CUDA extensions, KV-cache manager, or model runtime.
- The vLLM sparse bridge and TORCH_SDPA branch are local integration code, not
  the official Quest runtime.
- Multi-head union is an adapter approximation and is not claimed to be the
  official Quest runtime behavior.
- `IMPLEMENTATION_BOUNDARY.md` records which files are upstream vLLM,
  existing GranuleKV integration, and user-owned Quest/sparse bridge code.
