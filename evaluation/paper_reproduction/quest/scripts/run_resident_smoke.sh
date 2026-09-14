#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/evaluation/paper_reproduction/quest/results/resident_smoke}"

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for mode in dense sparse; do
  "${PYTHON_BIN}" "${ROOT}/evaluation/paper_reproduction/quest/scripts/resident_sparse_e2e.py" \
    --model "${MODEL_PATH}" \
    --mode "${mode}" \
    --output "${RESULT_DIR}/${mode}.json" \
    2>&1 | tee "${RESULT_DIR}/${mode}.log"
done

echo "resident-only Quest smoke completed: ${RESULT_DIR}"
