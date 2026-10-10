#!/bin/bash
# Single idle free V100, isolated7200 GPU-second grant, no optimizer updates.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
DIAG="$ROOT/runtime/outputs/diagnostics"
LEDGER="$DIAG/leaf-full-state-cuda-20261005.gpu-budget.json"
TOOLCHAIN="$ROOT/runtime/toolchains/cuda-nvlink-12.6.85"
STAGE=${1:?runtime-preflight, full-state, saved-recovery or saved-original-control required}
DRY_RUN=${2:-}
LEAF_GPU_INDEX=${LEAF_GPU_INDEX:-1}
SESSION_SECONDS=${LEAF_SESSION_SECONDS:-6000}
case "$STAGE" in
  runtime-preflight) SESSION_SECONDS=180 ;;
  full-state) ;;
  saved-recovery) SESSION_SECONDS=${LEAF_SESSION_SECONDS:-900} ;;
  saved-original-control) SESSION_SECONDS=${LEAF_SESSION_SECONDS:-600} ;;
  *) exit 2 ;;
esac
[[ $LEAF_GPU_INDEX == 0 || $LEAF_GPU_INDEX == 1 ]] || exit 2
[[ $SESSION_SECONDS =~ ^[0-9]+$ && $SESSION_SECONDS -ge 20 && $SESSION_SECONDS -le 7200 ]] || exit 2
[[ -z $DRY_RUN || $DRY_RUN == --dry-run ]] || exit 2
[[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
[[ $(cd "$CODE_ROOT" && git rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
OUTPUT="$DIAG/leaf-full-state-cuda-20261005-$STAGE.json"
if [[ $DRY_RUN != --dry-run ]]; then
  [[ ! -e $OUTPUT ]] || exit 2
  [[ -z $(nvidia-smi -i "$LEAF_GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
  LINE=$(nvidia-smi -i "$LEAF_GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 100 && $UTIL -eq 0 ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
COMMAND=(taskset -c 8-15 env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES="$LEAF_GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/leaf-full-state-cuda-20261005"
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  SINGULARITYENV_PREPEND_PATH="$OVERLAY/nvlink-bin:$TOOLCHAIN/bin"
  SINGULARITYENV_ORCHIDEE_NVLINK_LOG="$OUTPUT.nvlink.log" SINGULARITYENV_ORCHIDEE_NVLINK_STAGE="leaf-$STAGE"
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$TOOLCHAIN:$TOOLCHAIN:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python" -B)
if [[ $STAGE == runtime-preflight ]]; then
  COMMAND+=("$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu --output "$OUTPUT")
else
  PROBE="$OVERLAY/probe_shared_leaf_full_state_cuda.py"
  [[ $STAGE != saved-original-control ]] || PROBE="$OVERLAY/probe_shared_leaf_original_control.py"
  COMMAND+=("$PROBE"
    --manifest "$OVERLAY/original-full-state-schedule-resume.manifest.json"
    --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
    --old-snapshot "$DIAG/shared-original-full-state-schedule-20261005.old-diagnostic.pkl"
    --new-snapshot "$DIAG/shared-original-full-state-schedule-20261005.new-diagnostic.pkl"
    --field-adapter "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
    --backbone-adapter "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
    --turnover-overlay "$OVERLAY/turnover_leaf_age_fall.guarded-source.py"
    --turnover-base-sha256 bb8674f3d2ceadb28df907346d18f6300dd87f46d09ff6010c795f72dbee6485
    --turnover-sha256 f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c
    --runtime-preflight "$DIAG/leaf-full-state-cuda-20261005-runtime-preflight.json"
    --wall-seconds "$((SESSION_SECONDS-30))" --output "$OUTPUT")
  if [[ $STAGE == saved-original-control ]]; then
    COMMAND+=(--corrected-output "$DIAG/leaf-full-state-cuda-20261005-full-state.json" --call-seconds 60)
  else
    COMMAND+=(--lower-seconds 900 --backend-seconds 4200)
  fi
  if [[ $STAGE == saved-recovery ]]; then
    COMMAND+=(--saved-executable "$DIAG/leaf-full-state-cuda-20261005-full-state.executable.pkl")
  fi
fi
if [[ $STAGE != runtime-preflight ]]; then
  COMMAND=(bash "$OVERLAY/monitor_leaf_cuda_memory.sh" "$LEAF_GPU_INDEX" "$OUTPUT.memory.csv" -- "${COMMAND[@]}")
fi
WRAPPED=(/apps/soft/anaconda3/bin/python "$OVERLAY/shared_leaf_gpu_budget.py"
  --ledger "$LEDGER" --stage "$STAGE" --seconds "$SESSION_SECONDS" -- "${COMMAND[@]}")
if [[ $DRY_RUN == --dry-run ]]; then printf '%q ' "${WRAPPED[@]}"; printf '\n'; exit 0; fi
date -Is
uptime
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
exec "${WRAPPED[@]}"
