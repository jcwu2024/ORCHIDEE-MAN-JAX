#!/bin/bash
# Diagnostic only: CPU shared test node or one free gln01 GPU. No Slurm submission.
set -euo pipefail

PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
BACKEND=cpu
CODE_DIR=
OUTPUT_DIR=
WINDOW_DIR=
ASSET_ROOT="$PROJECT/runtime/assets/teacher_669_a403cd3"
ENTRY=benchmark
CACHE_MODE=off
CACHE_ROOT=
GPU_INDEX=0
BUDGET_FILE="$PROJECT/runtime/outputs/research/global_throughput/gpu-round2-budget.json"
CPU_LIST=0-3
TIMEOUT_SECONDS=1200
DRY_RUN=0
BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND=$2; shift 2 ;;
    --code-dir) CODE_DIR=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --window-dir) WINDOW_DIR=$2; shift 2 ;;
    --entry) ENTRY=$2; shift 2 ;;
    --cache-mode) CACHE_MODE=$2; shift 2 ;;
    --cache-root) CACHE_ROOT=$2; shift 2 ;;
    --gpu-index) GPU_INDEX=$2; shift 2 ;;
    --budget-file) BUDGET_FILE=$2; shift 2 ;;
    --asset-root) ASSET_ROOT=$2; shift 2 ;;
    --cpu-list) CPU_LIST=$2; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; BENCH_ARGS=("$@"); break ;;
    *) echo "unknown launcher argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$BACKEND" == cpu || "$BACKEND" == gpu ]]
[[ "$ENTRY" == benchmark || "$ENTRY" == cache-probe ]]
[[ "$CACHE_MODE" == off || "$CACHE_MODE" == persistent ]]
[[ "$GPU_INDEX" == 0 || "$GPU_INDEX" == 1 ]]
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ && "$TIMEOUT_SECONDS" -ge 10 && "$TIMEOUT_SECONDS" -le 1800 ]]
[[ -n "$CODE_DIR" && -n "$OUTPUT_DIR" ]]
WINDOW_DIR=${WINDOW_DIR:-"$OUTPUT_DIR/windows/$(date -u +%Y%m%dT%H%M%SZ)"}
CACHE_ROOT=${CACHE_ROOT:-"$OUTPUT_DIR/xla_cache"}
for TARGET in "$CODE_DIR" "$OUTPUT_DIR" "$WINDOW_DIR" "$ASSET_ROOT" "$CACHE_ROOT" "$BUDGET_FILE"; do
  RESOLVED=$(realpath -m "$TARGET")
  case "$RESOLVED" in /WORK/liwei_work/jcwu/*) ;; *) echo "path outside jcwu: $RESOLVED" >&2; exit 2 ;; esac
done
CODE_DIR=$(realpath "$CODE_DIR")
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
ASSET_ROOT=$(realpath "$ASSET_ROOT")
WINDOW_DIR=$(realpath -m "$WINDOW_DIR")
CACHE_ROOT=$(realpath -m "$CACHE_ROOT")
BUDGET_FILE=$(realpath -m "$BUDGET_FILE")
test -f "$CODE_DIR/scripts/dev/benchmark_global_batch.py"
test -z "$(cd "$CODE_DIR" && git status --porcelain --untracked-files=all)"
PYTHON="$PROJECT/.venvs/orcjax_${BACKEND}/bin/python"
test -x "$PYTHON"
HOST=$(hostname -s)
if [[ "$DRY_RUN" -eq 0 ]]; then
  case "$BACKEND:$HOST" in
    cpu:ibtest0[1-4]|cpu:test0[1-4]|gpu:ibc13b02n04|gpu:gln01) ;;
    *) echo "unsupported free compute node: $BACKEND:$HOST" >&2; exit 2 ;;
  esac
fi
export PYTHONDONTWRITEBYTECODE=1
export ORCHIDEE_REPO_ROOT="$CODE_DIR"
export ORCHIDEE_DATA_ROOT="$PROJECT/runtime/data"
export ORCHIDEE_REFERENCE_ROOT="$PROJECT/runtime/assets"
export ORCHIDEE_OUTPUT_ROOT="$OUTPUT_DIR"
export XDG_CACHE_HOME="$OUTPUT_DIR/xdg-cache"
export CUDA_CACHE_PATH="$OUTPUT_DIR/cuda-cache"
export ORCHJAX_DISABLE_COMPILATION_CACHE=1
[[ "$CACHE_MODE" == off ]] || export ORCHJAX_DISABLE_COMPILATION_CACHE=0
export JAX_ENABLE_COMPILATION_CACHE=false
unset JAX_COMPILATION_CACHE_DIR
if [[ "$CACHE_MODE" == persistent ]]; then
  export JAX_ENABLE_COMPILATION_CACHE=true
  export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT"
fi
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
if [[ "$BACKEND" == cpu ]]; then
  export JAX_PLATFORMS=cpu
  SCRIPT=benchmark_global_batch.py
  [[ "$ENTRY" == benchmark ]] || SCRIPT=probe_global_batch_cache.py
  COMMAND=("$PYTHON" "$CODE_DIR/scripts/dev/$SCRIPT")
else
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  test -f "$CUDA_DRIVER"
  test -f "$NVML_DRIVER"
  BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  export SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  export ORCHJAX_PHYSICAL_GPU_UUID
  export ORCHJAX_GPU_DRIVER_VERSION
  ORCHJAX_PHYSICAL_GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
  ORCHJAX_GPU_DRIVER_VERSION=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=driver_version --format=csv,noheader)
  export SINGULARITYENV_ORCHJAX_PHYSICAL_GPU_UUID="$ORCHJAX_PHYSICAL_GPU_UUID"
  export SINGULARITYENV_ORCHJAX_GPU_DRIVER_VERSION="$ORCHJAX_GPU_DRIVER_VERSION"
  export SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_batch.py
  [[ "$ENTRY" == benchmark ]] || export SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY=probe_global_batch_cache.py
  export SINGULARITYENV_JAX_ENABLE_X64=true
  export SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_DIR"
  export SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ORCHIDEE_DATA_ROOT"
  export SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ORCHIDEE_REFERENCE_ROOT"
  export SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$OUTPUT_DIR"
  export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
  export SINGULARITYENV_CUDA_CACHE_PATH="$CUDA_CACHE_PATH"
  export SINGULARITYENV_ORCHJAX_DISABLE_COMPILATION_CACHE="$ORCHJAX_DISABLE_COMPILATION_CACHE"
  export SINGULARITYENV_JAX_ENABLE_COMPILATION_CACHE="$JAX_ENABLE_COMPILATION_CACHE"
  unset SINGULARITYENV_JAX_COMPILATION_CACHE_DIR
  [[ "$CACHE_MODE" == off ]] || export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT"
  export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1
  export SINGULARITYENV_MKL_NUM_THREADS=1 SINGULARITYENV_NUMEXPR_NUM_THREADS=1
  COMMAND=(singularity exec --nv --bind "$BINDS" --pwd "$CODE_DIR"
    /apps/soft/sif/foundationpose "$PYTHON" "$CODE_DIR/scripts/hpc/global_batch_gpu_entry.py")
fi
[[ "$ENTRY" != benchmark ]] || COMMAND+=(--asset-root "$ASSET_ROOT")
COMMAND+=(--output-dir "$OUTPUT_DIR" --cache-mode "$CACHE_MODE" --cache-root "$CACHE_ROOT" "${BENCH_ARGS[@]}")
SUPERVISOR=("$PROJECT/.venvs/orcjax_cpu/bin/python" "$CODE_DIR/scripts/hpc/global_batch_free_window.py"
  --timeout-seconds "$TIMEOUT_SECONDS" --output-dir "$WINDOW_DIR" --report-dir "$OUTPUT_DIR")
[[ "$BACKEND" != gpu ]] || SUPERVISOR+=(--budget-file "$BUDGET_FILE")
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'cwd=%q\n' "$CODE_DIR"
  printf 'xdg_cache_home=%q cuda_cache_path=%q jax_cache_mode=%q cache_root=%q budget_file=%q\n' "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$CACHE_MODE" "$CACHE_ROOT" "$BUDGET_FILE"
  printf 'command='
  printf '%q ' "${SUPERVISOR[@]}" -- taskset -c "$CPU_LIST" /usr/bin/time -v "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
test ! -e "$WINDOW_DIR/launch-status.json"
mkdir -p "$OUTPUT_DIR" "$WINDOW_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
cd "$CODE_DIR"
{
  date -u +%FT%TZ
  hostname
  uptime
  git rev-parse HEAD
  taskset -pc $$
  printf 'cpu_list=%s backend=%s timeout_seconds=%s\n' "$CPU_LIST" "$BACKEND" "$TIMEOUT_SECONDS"
  printf 'xdg_cache_home=%s cuda_cache_path=%s jax_cache_mode=%s cache_root=%s\n' "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$CACHE_MODE" "$CACHE_ROOT"
  if [[ "$BACKEND" == gpu ]]; then
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  fi
} | tee "$WINDOW_DIR/launch-context.txt"
if [[ "$BACKEND" == gpu ]]; then
  test -z "$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader)"
  nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=1 \
    > "$WINDOW_DIR/gpu-memory-samples.csv" &
  MONITOR_PID=$!
  trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
fi
"${SUPERVISOR[@]}" -- taskset -c "$CPU_LIST" /usr/bin/time -v "${COMMAND[@]}"
