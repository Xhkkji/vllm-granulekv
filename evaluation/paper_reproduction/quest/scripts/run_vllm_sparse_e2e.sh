#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
GRANULEKV_ROOT="$(cd -- "${VLLM_ROOT}/../GranuleKV" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RUN_DIR:-${VLLM_ROOT}/evaluation/paper_reproduction/quest/results/${RUN_ID}}"
CONTROL_DIR="${VLLM_GRANULEKV_CONTROL_DIR:-${RESULT_DIR}/control}"
CUDA_IPC_LIBRARY="${CUDA_IPC_LIBRARY:-${GRANULEKV_ROOT}/gids_module/build/libgranulekv_cuda_ipc.so}"
TORCH_BRIDGE_DIR="${TORCH_BRIDGE_DIR:-${VLLM_ROOT}/vllm/granulekv/build/torch_bridge}"
MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/granulekv-mps-pipe}"
MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/granulekv-mps-log}"
RESTORE_MODE="${VLLM_GRANULEKV_RESTORE_MODE:-layerwise}"
DYNAMIC_RESTORE="${VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE:-0}"
EXACT_RESTORE="${VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE:-0}"

case "${RESTORE_MODE}" in
  full|layerwise) ;;
  *)
    echo "VLLM_GRANULEKV_RESTORE_MODE must be full or layerwise" >&2
    exit 2
    ;;
esac

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -n env \
    "RUN_ID=${RUN_ID}" "RUN_DIR=${RUN_DIR:-}" \
    "MODEL_PATH=${MODEL_PATH:-}" "PYTHON_BIN=${PYTHON_BIN}" \
    "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}" \
    "VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET=${VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET:-8}" \
    "VLLM_GRANULEKV_SPARSE_POLICY_MODULE=${VLLM_GRANULEKV_SPARSE_POLICY_MODULE:-evaluation.paper_reproduction.quest.adapter.policy:QuestPolicy}" \
    "VLLM_GRANULEKV_MAX_IN_FLIGHT=${VLLM_GRANULEKV_MAX_IN_FLIGHT:-4}" \
    "VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS=${VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS:-4}" \
    "VLLM_GRANULEKV_RESTORE_MODE=${RESTORE_MODE}" \
    "VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE=${DYNAMIC_RESTORE}" \
    "VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE=${EXACT_RESTORE}" \
    "CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_DIRECTORY}" \
    "CUDA_MPS_LOG_DIRECTORY=${MPS_LOG_DIRECTORY}" \
    /usr/bin/bash "$0" "$@"
fi

if [[ -n "${VLLM_GRANULEKV_E2E_COMMAND:-}" ]]; then
  E2E_COMMAND="${VLLM_GRANULEKV_E2E_COMMAND}"
else
  DYNAMIC_ARGS=""
  if [[ "${DYNAMIC_RESTORE}" == "1" ]]; then
    DYNAMIC_ARGS="--dynamic-restore"
  fi
  E2E_COMMAND="${PYTHON_BIN} ${SCRIPT_DIR}/vllm_sparse_e2e.py \
    --model ${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct} \
    ${DYNAMIC_ARGS} --output ${RESULT_DIR}/summary.json"
fi

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}" "${CONTROL_DIR}"

export PYTHONPATH="${VLLM_ROOT}:${GRANULEKV_ROOT}/gids_module:${GRANULEKV_ROOT}/gids_module/build${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${GRANULEKV_ROOT}/gids_module/build:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIRECTORY}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIRECTORY}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_GRANULEKV_ENABLE=1
export VLLM_GRANULEKV_IOSTACK_ROOT="${GRANULEKV_ROOT}"
export VLLM_GRANULEKV_CONTROL_DIR="${CONTROL_DIR}"
export VLLM_GRANULEKV_CUDA_IPC_LIBRARY="${CUDA_IPC_LIBRARY}"
export VLLM_GRANULEKV_TORCH_BRIDGE_DIR="${TORCH_BRIDGE_DIR}"
export VLLM_GRANULEKV_SERVICE_LIFETIME=resident
export VLLM_GRANULEKV_TIMEOUT_SECONDS="${VLLM_GRANULEKV_TIMEOUT_SECONDS:-300}"
export VLLM_GRANULEKV_MAX_IN_FLIGHT="${VLLM_GRANULEKV_MAX_IN_FLIGHT:-4}"
export VLLM_GRANULEKV_PREFIX_ENABLE=1
export VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE=1
export VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS="${VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS:-28}"
export VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS="${VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS:-4}"
if [[ "${RESTORE_MODE}" == "layerwise" ]]; then
  export VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER=1
  export VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE=1
  export VLLM_GRANULEKV_SPARSE_CONSUMER_ENABLE=1
  export VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE=0
else
  export VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE=0
  export VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER=0
  export VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE=0
  export VLLM_GRANULEKV_SPARSE_CONSUMER_ENABLE=0
  export VLLM_GRANULEKV_SPARSE_RESIDENT_ENABLE=1
fi
export VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET="${VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET:-8}"
export VLLM_GRANULEKV_SPARSE_POLICY_MODULE="${VLLM_GRANULEKV_SPARSE_POLICY_MODULE:-evaluation.paper_reproduction.quest.adapter.policy:QuestPolicy}"
export VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE="${DYNAMIC_RESTORE}"
export VLLM_GRANULEKV_SPARSE_EXACT_RESTORE_ENABLE="${EXACT_RESTORE}"

if [[ ! -f "${CUDA_IPC_LIBRARY}" ]]; then
  echo "missing GranuleKV CUDA IPC library: ${CUDA_IPC_LIBRARY}" >&2
  exit 1
fi
if ! compgen -G "${TORCH_BRIDGE_DIR}/granulekv_torch_bridge*.so" >/dev/null; then
  echo "missing GranuleKV torch bridge: ${TORCH_BRIDGE_DIR}" >&2
  exit 1
fi

bash "${GRANULEKV_ROOT}/gids_module/start_granulekv_mps.sh"
bash "${GRANULEKV_ROOT}/gids_module/check_granulekv_mps.sh" >"${RESULT_DIR}/mps.txt"

DAEMON_PID=""
cleanup() {
  if [[ -n "${DAEMON_PID}" ]] && kill -0 "${DAEMON_PID}" 2>/dev/null; then
    kill -TERM "${DAEMON_PID}" || true
    wait "${DAEMON_PID}" || true
  fi
}
trap cleanup EXIT

"${PYTHON_BIN}" -m granulekv.daemon \
  --control-dir "${CONTROL_DIR}" \
  --cuda-ipc-library "${CUDA_IPC_LIBRARY}" \
  --ssd-index "${VLLM_GRANULEKV_SSD_INDEX:-0}" \
  --max-in-flight "${VLLM_GRANULEKV_MAX_IN_FLIGHT}" \
  >"${RESULT_DIR}/daemon.log" 2>&1 &
DAEMON_PID=$!
for _ in $(seq 1 300); do
  [[ -f "${CONTROL_DIR}/control.slot" ]] && break
  if ! kill -0 "${DAEMON_PID}" 2>/dev/null; then
    echo "GranuleKV daemon exited during startup" >&2
    exit 1
  fi
  sleep 0.1
done
[[ -f "${CONTROL_DIR}/control.slot" ]] || {
  echo "timed out waiting for GranuleKV daemon" >&2
  exit 1
}

echo "[Quest E2E] backend=${VLLM_ATTENTION_BACKEND} result=${RESULT_DIR}"
bash -lc "${E2E_COMMAND}" 2>&1 | tee "${RESULT_DIR}/console.log"
