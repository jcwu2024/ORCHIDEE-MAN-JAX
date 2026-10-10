#!/bin/bash
# Independent free one-card corrected T7 paired efficiency diagnostic.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
DIAG="$ROOT/runtime/outputs/diagnostics"
TOOLCHAIN="$ROOT/runtime/toolchains/cuda-nvlink-12.6.85"
LEDGER="$DIAG/latent-reader-T7-20261005.gpu-budget.json"
BUDGET_SCOPE_ARGS=()
if [[ ${T7_FREE_MIGRATION_7200:-0} == 1 ]]; then
  LEDGER="$DIAG/latent-reader-T7-20261006-free-migration.gpu-budget.json"
  BUDGET_SCOPE_ARGS=(--migration-7200)
elif [[ ${T7_FREE_MIGRATION_7200:-0} != 0 ]]; then
  exit 2
fi
STAGE=${1:?runtime-preflight, compile-original, compile-candidate or matched required}
DRY_RUN=${2:-}
T7_GPU_INDEX=${T7_GPU_INDEX:-1}
case "$STAGE" in
  runtime-preflight) SESSION_SECONDS=180 ;;
  compile-original) SESSION_SECONDS=${T7_SESSION_SECONDS:-4500}; BACKEND_SECONDS=${T7_BACKEND_SECONDS:-3600} ;;
  compile-candidate) SESSION_SECONDS=${T7_SESSION_SECONDS:-7200}; BACKEND_SECONDS=${T7_BACKEND_SECONDS:-6300} ;;
  matched|saved-recovery) SESSION_SECONDS=${T7_SESSION_SECONDS:-900}; BACKEND_SECONDS=1 ;;
  *) exit 2 ;;
esac
[[ $T7_GPU_INDEX == 0 || $T7_GPU_INDEX == 1 ]] || exit 2
[[ $SESSION_SECONDS =~ ^[0-9]+$ && $SESSION_SECONDS -ge 20 && $SESSION_SECONDS -le 7200 ]] || exit 2
[[ -z $DRY_RUN || $DRY_RUN == --dry-run ]] || exit 2
[[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
[[ $(cd "$CODE_ROOT" && git rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
RUN_SUFFIX=${T7_RUN_SUFFIX:-}
[[ -z $RUN_SUFFIX || $RUN_SUFFIX =~ ^[A-Za-z0-9_-]{1,64}$ ]] || exit 2
OUTPUT="$DIAG/latent-reader-T7-20261005-$STAGE${RUN_SUFFIX:+-$RUN_SUFFIX}.json"
CANDIDATE_EXECUTABLE=${T7_CANDIDATE_EXECUTABLE:-$DIAG/latent-reader-T7-20261005-compile-candidate.executable.pkl}
[[ $CANDIDATE_EXECUTABLE == "$DIAG"/latent-reader-T7-20261005-compile-candidate*.executable.pkl ||
   $CANDIDATE_EXECUTABLE == "$DIAG"/latent-reader-T7-20261005-paid*-compile-candidate*.executable.pkl ]] || exit 2
if [[ $DRY_RUN != --dry-run ]]; then
  [[ ! -e $OUTPUT ]] || exit 2
  if [[ $STAGE == compile-candidate ]]; then
    [[ -z $(compgen -G "$DIAG/latent-reader-T7-20261005-compile-candidate*.executable.pkl") ]] || exit 2
    [[ -z $(compgen -G "$DIAG/latent-reader-T7-20261005-paid*-compile-candidate*.executable.pkl") ]] || exit 2
  fi
  [[ -z $(nvidia-smi -i "$T7_GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
  LINE=$(nvidia-smi -i "$T7_GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 100 && $UTIL -eq 0 ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
COMMAND=(taskset -c 8-15 env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES="$T7_GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/latent-reader-T7-20261005"
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  SINGULARITYENV_PREPEND_PATH="$OVERLAY/nvlink-bin:$TOOLCHAIN/bin"
  SINGULARITYENV_ORCHIDEE_NVLINK_LOG="$OUTPUT.nvlink.log" SINGULARITYENV_ORCHIDEE_NVLINK_STAGE="latent-T7-$STAGE"
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$TOOLCHAIN:$TOOLCHAIN:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python" -B)
if [[ $STAGE == runtime-preflight ]]; then
  COMMAND+=("$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu --output "$OUTPUT")
else
  PROBE_STAGE="$STAGE"
  [[ $STAGE != saved-recovery ]] || PROBE_STAGE=matched
  COMMAND+=("$OVERLAY/probe_shared_latent_reader_T7_pair.py" --stage "$PROBE_STAGE"
    --manifest "$OVERLAY/latent-reader-T7-20261005.manifest.json"
    --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
    --parameters "$DIAG/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl"
    --field-adapter "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
    --backbone-adapter "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
    --transformer-module "$OVERLAY/shared_transformer.py" --latent-module "$OVERLAY/shared_latent_reader.py"
    --turnover-overlay "$OVERLAY/turnover_leaf_age_fall.guarded-source.py"
    --runtime-preflight "$DIAG/latent-reader-T7-20261005-runtime-preflight.json"
    --wall-seconds "$((SESSION_SECONDS-30))" --lower-seconds 600 --backend-seconds "$BACKEND_SECONDS"
    --call-seconds 180 --pairs 3 --output "$OUTPUT")
  if [[ $STAGE == matched || $STAGE == saved-recovery ]]; then
    COMMAND+=(--original-executable "$DIAG/latent-reader-T7-20261005-compile-original.executable.pkl"
      --candidate-executable "$CANDIDATE_EXECUTABLE")
  fi
  COMMAND=(bash "$OVERLAY/shared_latent_T7_memory.sh" "$T7_GPU_INDEX" "$OUTPUT.memory.csv" -- "${COMMAND[@]}")
fi
WRAPPED=(/apps/soft/anaconda3/bin/python "$OVERLAY/shared_latent_T7_gpu_budget.py"
  --ledger "$LEDGER" --stage "$STAGE" --seconds "$SESSION_SECONDS" "${BUDGET_SCOPE_ARGS[@]}" -- "${COMMAND[@]}")
if [[ $DRY_RUN == --dry-run ]]; then printf '%q ' "${WRAPPED[@]}"; printf '\n'; exit 0; fi
date -Is
uptime
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
exec "${WRAPPED[@]}"
