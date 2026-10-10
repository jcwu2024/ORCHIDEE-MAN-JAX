#!/bin/bash
# Free one-GPU original JVP reference; no paid submission or candidate run.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-original-state-jvp-20261002"
DIAG="$ROOT/runtime/outputs/diagnostics/shared-original-state-jvp-20261002"
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ $# == 1 && ${1:-} =~ ^(preflight|model)$ ]] || exit 2
STAGE=$1
OUTPUT="$DIAG/$STAGE.json"
START=$(date +%s.%N)
PHASE=validating
GPU_STARTED=false
finish() {
  STATUS=$?
  [[ $DRY_RUN == 0 ]] || return 0
  {
    printf 'exit_status=%s\nphase=%s\ngpu_started=%s\ngpu_count=1\npaid_jobs=0\n' "$STATUS" "$PHASE" "$GPU_STARTED"
    awk -v a="$START" -v b="$(date +%s.%N)" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\nlauncher_card_hours=%.12f\n",b-a,(started=="true"?(b-a)/3600:0)}'
  } > "$OUTPUT.actual.exit"
}
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ ! -e $OUTPUT && ! -e $OUTPUT.actual.exit && ! -e $OUTPUT.source.py ]] || exit 2
trap finish EXIT
if [[ $DRY_RUN == 0 ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
  LINE=$(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
ENV=(env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES=1
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY:$ROOT/runtime/transfer/shared-forward-reverse-20260930:$ROOT/runtime/transfer/shared-command-buffer-loops-20261001"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_PYTHONFAULTHANDLER=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false)
CONTAINER=(singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
if [[ $STAGE == preflight ]]; then
  ENTRY="$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py"
  COMMAND=(timeout --signal=TERM --kill-after=10s 300s "${ENV[@]}" "${CONTAINER[@]}" "$ENTRY"
    --backend gpu --output "$OUTPUT")
else
  ENTRY="$OVERLAY/probe_shared_original_state_jvp.py"
  COMMAND=(timeout --signal=TERM --kill-after=10s 1500s "${ENV[@]}" "${CONTAINER[@]}" "$ENTRY"
    --runtime-preflight "$DIAG/preflight.json" --output "$OUTPUT")
fi
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
cp "$ENTRY" "$OUTPUT.source.py"
cp "$OVERLAY/run_shared_original_state_jvp.sh" "$OUTPUT.launcher.sh"
PHASE=executing
GPU_STARTED=true
{
  date --iso-8601=seconds
  hostname
  uptime
  grep Cpus_allowed_list /proc/self/status
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  "${COMMAND[@]}"
} 2>&1 | tee "$OUTPUT.log"
