#!/bin/bash
# Diagnostic only: CPU shared test node or one free gln01 GPU. No Slurm submission.
set -euo pipefail

PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
BACKEND=cpu
CODE_DIR=
OUTPUT_DIR=
ASSET_ROOT="$PROJECT/runtime/assets/teacher_669_a403cd3"
CPU_LIST=0-3
TIMEOUT_SECONDS=1200
DRY_RUN=0
BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND=$2; shift 2 ;;
    --code-dir) CODE_DIR=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --asset-root) ASSET_ROOT=$2; shift 2 ;;
    --cpu-list) CPU_LIST=$2; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; BENCH_ARGS=("$@"); break ;;
    *) echo "unknown launcher argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$BACKEND" == cpu || "$BACKEND" == gpu ]]
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ && "$TIMEOUT_SECONDS" -gt 0 && "$TIMEOUT_SECONDS" -le 1800 ]]
[[ -n "$CODE_DIR" && -n "$OUTPUT_DIR" ]]
for TARGET in "$CODE_DIR" "$OUTPUT_DIR" "$ASSET_ROOT"; do
  RESOLVED=$(realpath -m "$TARGET")
  case "$RESOLVED" in /WORK/liwei_work/jcwu/*) ;; *) echo "path outside jcwu: $RESOLVED" >&2; exit 2 ;; esac
done
CODE_DIR=$(realpath "$CODE_DIR")
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
ASSET_ROOT=$(realpath "$ASSET_ROOT")
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
export ORCHJAX_DISABLE_COMPILATION_CACHE=1
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
if [[ "$BACKEND" == cpu ]]; then
  export JAX_PLATFORMS=cpu
  COMMAND=("$PYTHON" "$CODE_DIR/scripts/dev/benchmark_global_batch.py")
else
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  test -f "$CUDA_DRIVER"
  test -f "$NVML_DRIVER"
  BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  export SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
  export SINGULARITYENV_JAX_ENABLE_X64=true
  export SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_DIR"
  export SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ORCHIDEE_DATA_ROOT"
  export SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ORCHIDEE_REFERENCE_ROOT"
  export SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$OUTPUT_DIR"
  export SINGULARITYENV_ORCHJAX_DISABLE_COMPILATION_CACHE=1
  export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1
  export SINGULARITYENV_MKL_NUM_THREADS=1 SINGULARITYENV_NUMEXPR_NUM_THREADS=1
  COMMAND=(singularity exec --nv --bind "$BINDS" --pwd "$CODE_DIR"
    /apps/soft/sif/foundationpose "$PYTHON" "$CODE_DIR/scripts/hpc/global_batch_gpu_entry.py")
fi
COMMAND+=(--asset-root "$ASSET_ROOT" --output-dir "$OUTPUT_DIR" "${BENCH_ARGS[@]}")
if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'cwd=%q\n' "$CODE_DIR"
  printf 'command='
  printf '%q ' timeout "$TIMEOUT_SECONDS" taskset -c "$CPU_LIST" /usr/bin/time -v "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
cd "$CODE_DIR"
{
  date -u +%FT%TZ
  hostname
  uptime
  git rev-parse HEAD
  taskset -pc $$
  printf 'cpu_list=%s backend=%s timeout_seconds=%s\n' "$CPU_LIST" "$BACKEND" "$TIMEOUT_SECONDS"
  if [[ "$BACKEND" == gpu ]]; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  fi
} | tee "$OUTPUT_DIR/launch-context.txt"
if [[ "$BACKEND" == gpu ]]; then
  nvidia-smi -i 0 --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv --loop=1 \
    > "$OUTPUT_DIR/gpu-memory-samples.csv" &
  MONITOR_PID=$!
  trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
fi
timeout "$TIMEOUT_SECONDS" taskset -c "$CPU_LIST" /usr/bin/time -v "${COMMAND[@]}" \
  > "$OUTPUT_DIR/stdout.log" 2> "$OUTPUT_DIR/stderr.log"
