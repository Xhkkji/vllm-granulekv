#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
TASK="${TASK:-passkey}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/evaluation/paper_reproduction/solidattention/results/quality_$(date +%Y%m%d_%H%M%S)}"
QUALITY_RUNNER="${ROOT}/evaluation/paper_reproduction/quest/scripts/quality_eval.py"
SUMMARY_RUNNER="${ROOT}/evaluation/paper_reproduction/solidattention/scripts/summarize_quality.py"

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

common=(--model "${MODEL_PATH}" --task "${TASK}" --decode-tokens 8)
if [[ "${TASK}" == "passkey" ]]; then
  common+=(--contexts 8192,16384,32768 --depths 0,25,50,75,100 \
           --samples "${SAMPLES:-20}")
elif [[ "${TASK}" == "jsonl" ]]; then
  : "${MANIFEST:?MANIFEST is required for TASK=jsonl}"
  common+=(--manifest "${MANIFEST}" --limit "${LIMIT:-20}")
else
  echo "unsupported TASK=${TASK}" >&2
  exit 2
fi

"${PYTHON_BIN}" "${QUALITY_RUNNER}" "${common[@]}" \
  --mode dense --block-budget 1 --output "${RESULT_DIR}/dense.json"

sparse_outputs=()
for budget in 16 32 64 128; do
  output="${RESULT_DIR}/solidattention_budget${budget}.json"
  "${PYTHON_BIN}" "${QUALITY_RUNNER}" "${common[@]}" \
    --mode solidattention --block-budget "${budget}" --output "${output}"
  sparse_outputs+=("${output}")
done

all_blocks="${RESULT_DIR}/solidattention_all_blocks.json"
"${PYTHON_BIN}" "${QUALITY_RUNNER}" "${common[@]}" \
  --mode solidattention --block-budget 32 --all-blocks --output "${all_blocks}"
sparse_outputs+=("${all_blocks}")

"${PYTHON_BIN}" "${SUMMARY_RUNNER}" \
  --dense "${RESULT_DIR}/dense.json" \
  --sparse "${sparse_outputs[@]}" \
  --output "${RESULT_DIR}/selection_quality.csv"

echo "SolidAttention quality validation: ${RESULT_DIR}"
