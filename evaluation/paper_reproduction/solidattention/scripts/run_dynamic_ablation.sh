#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
RUNNER="${VLLM_ROOT}/evaluation/paper_reproduction/quest/scripts/run_vllm_sparse_e2e.sh"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RUN_DIR:-${VLLM_ROOT}/evaluation/paper_reproduction/solidattention/results/${RUN_ID}_dynamic_ablation}"
MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
POLICY="evaluation.paper_reproduction.solidattention.adapter.solidattention:SolidAttentionPolicy"
COMMON_ARGS="--prefix-tokens 8192 --suffix-tokens 16 --decode-tokens 64 --max-model-len 8448 --gpu-memory-utilization 0.85 --block-budget 32 --warmup-iterations 2 --iterations 5"

if [[ -e "${RESULT_ROOT}" ]]; then
  echo "result directory already exists: ${RESULT_ROOT}" >&2
  exit 2
fi
mkdir -p "${RESULT_ROOT}"

run_case() {
  local name="$1"
  local restore_mode="$2"
  local dynamic="$3"
  local exact="$4"
  local selected="$5"
  local guard="$6"
  local extra_args="$7"
  env \
    RUN_ID="${RUN_ID}_${name}" \
    RUN_DIR="${RESULT_ROOT}/${name}" \
    MODEL_PATH="${MODEL_PATH}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    VLLM_GRANULEKV_RESTORE_MODE="${restore_mode}" \
    VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE="${dynamic}" \
    VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE="${exact}" \
    VLLM_GRANULEKV_SPARSE_CORRECTION_ENABLE="${dynamic}" \
    VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE="${selected}" \
    VLLM_GRANULEKV_SPARSE_SELECTED_BLOCKS_ENABLE="${selected}" \
    VLLM_GRANULEKV_SPARSE_PREDICTION_GUARD_BLOCKS="${guard}" \
    VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET=32 \
    VLLM_GRANULEKV_SPARSE_POLICY_MODULE="${POLICY}" \
    VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS=4 \
    VLLM_GRANULEKV_MAX_IN_FLIGHT=4 \
    VLLM_GRANULEKV_E2E_ARGS="${COMMON_ARGS} ${extra_args}" \
    bash "${RUNNER}"
}

run_case A_dense_full full 0 0 0 0 "--attention-mode dense --benchmark-third"
run_case B_exact_compact layerwise 1 1 0 0 ""
run_case C_exact_selected layerwise 1 1 1 0 ""
run_case D_union_selected layerwise 1 0 1 0 ""
run_case E1_guard_1 layerwise 1 0 1 1 ""
run_case E2_guard_2 layerwise 1 0 1 2 ""
run_case E3_guard_4 layerwise 1 0 1 4 ""

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_dynamic_ablation.py" "${RESULT_ROOT}"
echo "SolidAttention dynamic ablation: ${RESULT_ROOT}"
