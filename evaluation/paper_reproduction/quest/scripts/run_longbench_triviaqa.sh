#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
MANIFEST="${MANIFEST:-/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/all.jsonl}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/evaluation/paper_reproduction/quest/results/longbench_triviaqa}"
LIMIT="${LIMIT:-20}"
TOKEN_BUDGET="${TOKEN_BUDGET:-2048}"

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for mode in dense quest; do
  output="${RESULT_DIR}/${mode}.json"
  "${PYTHON_BIN}" "${ROOT}/evaluation/paper_reproduction/quest/scripts/quality_eval.py" \
    --model "${MODEL_PATH}" --mode "${mode}" --task jsonl \
    --manifest "${MANIFEST}" --limit "${LIMIT}" \
    --token-budget "${TOKEN_BUDGET}" --output "${output}" \
    2>&1 | tee "${output%.json}.log"
done
