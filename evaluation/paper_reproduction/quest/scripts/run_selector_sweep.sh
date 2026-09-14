#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/evaluation/paper_reproduction/quest/results/selector_sweep}"
PAGE_SIZE="${PAGE_SIZE:-16}"
mkdir -p "${RESULT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for tokens in 8192 16384 32768; do
  for token_budget in 512 1024 2048 4096; do
    for seed in 7 17 27; do
      output="${RESULT_DIR}/tokens${tokens}_budget${token_budget}_seed${seed}.json"
      "${PYTHON_BIN}" -m evaluation.paper_reproduction.quest.scripts.selector_eval \
        --tokens "${tokens}" --page-size "${PAGE_SIZE}" \
        --token-budget "${token_budget}" --seed "${seed}" \
        --output "${output}"
    done
  done
done
