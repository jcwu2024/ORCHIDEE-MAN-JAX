#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-recurrent-checkpoint-policy-20260930-v3"
FORWARD_OVERLAY="$ROOT/runtime/transfer/shared-forward-reverse-20260930"
DIAG="$ROOT/runtime/outputs/diagnostics"
STAGE=${1:-}
HORIZON=${2:-}
DRY_RUN=${3:-}
case "$STAGE" in reference|candidate|screen) ;; *) exit 2 ;; esac
case "$HORIZON" in 7|30) ;; *) exit 2 ;; esac
case "$DRY_RUN" in ''|--dry-run) ;; *) exit 2 ;; esac
GPU_INDEX=${GPU_INDEX:-0}
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
BASE="$DIAG/shared-recurrent-checkpoint-policy-20260930-v3-T$HORIZON"
OUTPUT="$BASE-$STAGE.json"
REFERENCE_BASE="$BASE"
if [[ "$STAGE" != reference ]]; then
  REFERENCE_BASE="$DIAG/shared-recurrent-checkpoint-policy-20260930-v2-T$HORIZON"
fi
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ -f "$OVERLAY/probe_shared_recurrent_checkpoint_policy.py" && -f "$FORWARD_OVERLAY/probe_shared_forward_reverse.py" ]] || exit 2
if [[ "$DRY_RUN" != --dry-run ]]; then
  [[ -z ${SLURM_JOB_ID:-} && $(hostname -s) == ibc13b02n04 ]] || exit 2
  GPU_LINE=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r GPU_MEMORY GPU_UTIL <<<"${GPU_LINE//,/ }"
  [[ $GPU_MEMORY =~ ^[0-9]+$ && $GPU_UTIL =~ ^[0-9]+$ && $GPU_MEMORY -le 2000 && $GPU_UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
RUNTIME=(env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu
  SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-checkpoint-policy-20260930-v2"
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1
  SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind
  "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose
  "$ROOT/.venvs/orcjax_gpu/bin/python")
PREFLIGHT=(timeout --signal=TERM --kill-after=10s 180s "${RUNTIME[@]}"
  "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py"
  --backend gpu --output "$BASE-preflight.json")
# A prior matching T7 graph took 222s to lower and 2549s to compile; older
# recurrent compiles took up to 5657s. Keep room for first execution and gates.
RUN_SECONDS=7200
COMMAND=(timeout --signal=TERM --kill-after=10s "$RUN_SECONDS" "${RUNTIME[@]}"
  "$OVERLAY/probe_shared_recurrent_checkpoint_policy.py"
  --stage "$STAGE" --horizon "$HORIZON"
  --source-profile "$DIAG/shared-split-profile-c599236.json"
  --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
  --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
  --daily-admission-report "$DIAG/shared-combined-real-daily-20260925.json"
  --seven-admission-report "$DIAG/shared-combined-recurrent-T7-20260926.json"
  --thirty-admission-report "$DIAG/shared-combined-recurrent-T30-20260926.json"
  --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py"
  --support-owner-file "$ROOT/runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py"
  --candidate-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
  --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --forward-probe "$FORWARD_OVERLAY/probe_shared_forward_reverse.py"
  --reference-result "$REFERENCE_BASE.reference.pkl"
  --reference-executable "$REFERENCE_BASE.reference.executable.pkl"
  --reference-report "$REFERENCE_BASE-reference.json"
  --output "$OUTPUT" --repeats 3)
if [[ "$DRY_RUN" == --dry-run ]]; then
  if [[ "$STAGE" == reference ]]; then printf '%q ' "${PREFLIGHT[@]}"; printf '\n'; fi
  printf '%q ' "${COMMAND[@]}"; printf '\n'
  exit 0
fi
exec > >(tee "$BASE-$STAGE.log") 2>&1
PHASE=launcher
STARTED_AT=$(date --iso-8601=seconds)
printf 'started_at=%s\nphase=launcher\n' "$STARTED_AT" > "$BASE-$STAGE.exit"
record_exit() {
  local status=$?
  trap - EXIT
  {
    printf 'started_at=%s\nended_at=%s\nphase=%s\nlauncher_exit_status=%s\n' \
      "$STARTED_AT" "$(date --iso-8601=seconds)" "$PHASE" "$status"
    if [[ -n ${PREFLIGHT_STATUS:-} ]]; then printf 'preflight_process_exit_status=%s\n' "$PREFLIGHT_STATUS"; fi
    if [[ -n ${MODEL_STATUS:-} ]]; then printf 'model_process_exit_status=%s\n' "$MODEL_STATUS"; fi
  } > "$BASE-$STAGE.exit.tmp"
  mv -f "$BASE-$STAGE.exit.tmp" "$BASE-$STAGE.exit"
  printf 'launcher_exit_status=%s phase=%s\n' "$status" "$PHASE"
}
trap record_exit EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
if [[ "$STAGE" == reference ]]; then
  PHASE=preflight
  set +e
  "${PREFLIGHT[@]}"
  PREFLIGHT_STATUS=$?
  set -e
  if [[ "$PREFLIGHT_STATUS" -ne 0 ]]; then exit "$PREFLIGHT_STATUS"; fi
fi
PHASE=model
set +e
"${COMMAND[@]}"
MODEL_STATUS=$?
set -e
date --iso-8601=seconds
printf 'model_process_exit_status=%s\n' "$MODEL_STATUS"
exit "$MODEL_STATUS"
