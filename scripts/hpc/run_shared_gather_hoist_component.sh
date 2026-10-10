#!/bin/bash
# Free GPU0 original-FP32 gather-only component; no scheduler submission.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-gather-hoist-20261001"
DIAG="$ROOT/runtime/outputs/diagnostics"
STEM=shared-gather-hoist-20261001
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ ${1:-} =~ ^(preflight|component)$ && $# == 1 ]] || exit 2
STAGE=$1
OUTPUT="$DIAG/$STEM-$STAGE.json"
LOG="$DIAG/$STEM-$STAGE.log"
START=$(date +%s.%N)
GPU_STARTED=false
PHASE=validating
finish_record() {
  STATUS=$?
  [[ $DRY_RUN == 0 ]] || return 0
  FINISH=$(date +%s.%N)
  {
    printf 'variant=original_FP32_raw_gather_only_hoist\nstage=%s\nphase=%s\ngpu_started=%s\npaid_jobs=0\nexit_status=%s\n' "$STAGE" "$PHASE" "$GPU_STARTED" "$STATUS"
    awk -v a="$START" -v b="$FINISH" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\ngpu_count=%d\nlauncher_card_hours=%.12f\n",b-a,(started=="true"?1:0),(started=="true"?(b-a)/3600:0)}'
  } > "$OUTPUT.launcher.txt"
}
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ ! -e $OUTPUT && ! -e $OUTPUT.launcher.txt && ! -e $LOG ]] || exit 2
if [[ $STAGE == component ]]; then
  for ARM in default gather_hoist; do
    for EXTRA in result.pkl forward.executable.pkl native_vjp.executable.pkl forward.hlo.txt native_vjp.hlo.txt repeat{0..6}.full_vjp.pkl; do
      [[ ! -e $DIAG/$STEM-$STAGE.$ARM.$EXTRA ]] || exit 2
    done
  done
fi
trap finish_record EXIT
if [[ $DRY_RUN == 0 ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
  LINE=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false
if [[ $STAGE == component ]]; then FLAGS="$FLAGS --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL,CUDNN"; fi
ENV=(env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_PYTHONFAULTHANDLER=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  "SINGULARITYENV_XLA_FLAGS=$FLAGS")
CONTAINER=(singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
if [[ $STAGE == preflight ]]; then
  COMMAND=(timeout --signal=TERM --kill-after=10s 300s "${ENV[@]}" "${CONTAINER[@]}"
    "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu --output "$OUTPUT")
else
  COMMAND=(timeout --signal=TERM --kill-after=10s 900s "${ENV[@]}" "${CONTAINER[@]}"
    "$OVERLAY/probe_shared_gather_hoist_component.py"
    --payload "$DIAG/shared-normalized-keys-payload-20261001.pkl"
    --default-report "$DIAG/shared-command-buffer-loops-threshold-component-default-20261001.json"
    --runtime-preflight "$DIAG/$STEM-preflight.json" --output "$OUTPUT")
fi
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
PHASE=running
GPU_STARTED=true
{
  date --iso-8601=seconds
  hostname
  uptime
  grep Cpus_allowed_list /proc/self/status
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  "${COMMAND[@]}"
} 2>&1 | tee "$LOG"
