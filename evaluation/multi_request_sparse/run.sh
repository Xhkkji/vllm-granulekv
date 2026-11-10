#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/xhk/llm-inference
VLLM_ROOT=${ROOT}/vllm-granulekv
GKV_ROOT=${ROOT}/GranuleKV
PYTHON=/home/xhk/miniconda3/envs/pytorch-vllm/bin/python
PREFIX=${1:-1024}
SUFFIX=${2:-0}
CASE=${3:-same}
RUN_DIR=${4:?provide a new absolute result directory}
if [[ ${EUID} -ne 0 ]]; then
    exec sudo -n /usr/bin/bash "$0" "$@"
fi
[[ ! -e ${RUN_DIR} ]] || { echo "Result directory already exists" >&2; exit 2; }
mkdir -p "${RUN_DIR}/control"
export PYTHONPATH=${VLLM_ROOT}:${GKV_ROOT}/gids_module:${GKV_ROOT}/gids_module/build
export LD_LIBRARY_PATH=${GKV_ROOT}/gids_module/build:${GKV_ROOT}/bam/build/lib
export CUDA_MPS_PIPE_DIRECTORY=/tmp/granulekv-mps-pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/granulekv-mps-log
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_GRANULEKV_ENABLE=1 VLLM_GRANULEKV_PREFIX_ENABLE=1
export VLLM_GRANULEKV_IOSTACK_ROOT=${GKV_ROOT}
export VLLM_GRANULEKV_CONTROL_DIR=${RUN_DIR}/control
export VLLM_GRANULEKV_CUDA_IPC_LIBRARY=${GKV_ROOT}/gids_module/build/libgranulekv_cuda_ipc.so
export VLLM_GRANULEKV_TORCH_BRIDGE_DIR=${VLLM_ROOT}/vllm/granulekv/build/torch_bridge
export VLLM_GRANULEKV_SERVICE_LIFETIME=resident VLLM_GRANULEKV_MAX_IN_FLIGHT=4
export VLLM_GRANULEKV_TIMEOUT_SECONDS=120
export VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE=1 VLLM_GRANULEKV_HIERARCHICAL_NUM_LAYERS=28
export VLLM_GRANULEKV_HIERARCHICAL_WINDOW_LAYERS=4 VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER=1
export VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE=1 VLLM_GRANULEKV_SPARSE_CONSUMER_ENABLE=1
export VLLM_GRANULEKV_SPARSE_BLOCK_BUDGET=32
export VLLM_GRANULEKV_SPARSE_POLICY_MODULE=evaluation.paper_reproduction.solidattention.adapter.solidattention:SolidAttentionPolicy
export VLLM_GRANULEKV_SPARSE_DYNAMIC_RESTORE_ENABLE=1 VLLM_GRANULEKV_SPARSE_CORRECTION_ENABLE=0
export VLLM_GRANULEKV_SPARSE_GPU_SELECT_ENABLE=1 VLLM_GRANULEKV_SPARSE_SELECTED_BLOCKS_ENABLE=1
export VLLM_GRANULEKV_SPARSE_CPU_STAGING_ENABLE=1 VLLM_GRANULEKV_SPARSE_CPU_PREFETCH_ENABLE=1
export VLLM_GRANULEKV_HOST_BACKING_ENABLE=1 VLLM_GRANULEKV_HOST_TO_GPU_ASYNC_PREFETCH_ENABLE=1
export VLLM_GRANULEKV_SPARSE_GPU_RESIDENCY_REUSE_ENABLE=1
export VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE=1 VLLM_GRANULEKV_EVENT_FENCED_LAYER_REUSE_ENABLE=1
export VLLM_GRANULEKV_SPARSE_CPU_CACHE_BLOCKS=$((4 * ((PREFIX + SUFFIX + 33) / 16)))
bash "${GKV_ROOT}/gids_module/start_granulekv_mps.sh"
DAEMON_PID=
cleanup() {
    if [[ -n ${DAEMON_PID} ]] && kill -0 "${DAEMON_PID}" 2>/dev/null; then
        kill -TERM "${DAEMON_PID}"
        wait "${DAEMON_PID}" || true
    fi
}
trap cleanup EXIT
"${PYTHON}" -m granulekv.daemon --control-dir "${RUN_DIR}/control" \
    --cuda-ipc-library "${VLLM_GRANULEKV_CUDA_IPC_LIBRARY}" --ssd-index 0 --max-in-flight 4 \
    >"${RUN_DIR}/daemon.log" 2>&1 &
DAEMON_PID=$!
for ((i=0; i<300; i++)); do
    [[ -f ${RUN_DIR}/control/control.slot ]] && break
    kill -0 "${DAEMON_PID}" 2>/dev/null || { echo 'Daemon failed'; exit 1; }
    sleep 0.1
done
[[ -f ${RUN_DIR}/control/control.slot ]] || { echo 'Daemon startup timed out'; exit 1; }
cd "${VLLM_ROOT}"
cp evaluation/multi_request_sparse/benchmark.py "${RUN_DIR}/benchmark.py"
cp evaluation/multi_request_sparse/validation.py "${RUN_DIR}/validation.py"
git diff --binary > "${RUN_DIR}/source.diff"
git rev-parse HEAD > "${RUN_DIR}/vllm_commit.txt"
git -C "${GKV_ROOT}" rev-parse HEAD > "${RUN_DIR}/granulekv_commit.txt"
git -C "${GKV_ROOT}" diff --binary > "${RUN_DIR}/granulekv_source.diff"
sha256sum "${GKV_ROOT}/gids_module/build/granulekv_native/granulekv_native.so" > "${RUN_DIR}/granulekv_native.sha256"
"${PYTHON}" "${RUN_DIR}/benchmark.py" \
    --prefix-tokens "${PREFIX}" --suffix-tokens "${SUFFIX}" --case "${CASE}" \
    --output "${RUN_DIR}/summary.json" "${@:5}" 2>&1 | tee "${RUN_DIR}/console.log"
