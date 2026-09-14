#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/evaluation/paper_reproduction/quest/results/passkey}"
PAGE_SIZE="${PAGE_SIZE:-16}"
SAMPLES="${SAMPLES:-20}"

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for context in 8192 16384 32768; do
  for token_budget in 512 1024 2048 4096; do
    for mode in dense quest; do
      output="${RESULT_DIR}/${mode}_ctx${context}_budget${token_budget}.json"
      "${PYTHON_BIN}" "${ROOT}/evaluation/paper_reproduction/quest/scripts/quality_eval.py" \
        --model "${MODEL_PATH}" --mode "${mode}" --task passkey \
        --contexts "${context}" --samples "${SAMPLES}" \
        --token-budget "${token_budget}" --page-size "${PAGE_SIZE}" \
        --output "${output}" 2>&1 | tee "${output%.json}.log"
    done
  done
done
