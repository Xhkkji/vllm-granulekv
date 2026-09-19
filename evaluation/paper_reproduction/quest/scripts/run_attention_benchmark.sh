#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/evaluation/paper_reproduction/quest/results/attention_benchmark}"
PAGE_SIZE="${PAGE_SIZE:-16}"
DECODE_TOKENS="${DECODE_TOKENS:-256}"
ITERATIONS="${ITERATIONS:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-0}"
MODES="${MODES:-dense quest solidattention}"

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for context in 8192 16384 32768; do
  for token_budget in 512 1024 2048 4096; do
    for mode in ${MODES}; do
      output="${RESULT_DIR}/${mode}_ctx${context}_budget${token_budget}.json"
      "${PYTHON_BIN}" "${ROOT}/evaluation/paper_reproduction/quest/scripts/resident_benchmark.py" \
        --model "${MODEL_PATH}" --mode "${mode}" \
        --context-length "${context}" --page-size "${PAGE_SIZE}" \
        --token-budget "${token_budget}" --decode-tokens "${DECODE_TOKENS}" \
        --iterations "${ITERATIONS}" --max-model-len "${MAX_MODEL_LEN}" \
        --output "${output}" 2>&1 | tee "${output%.json}.log"
    done
  done
done
