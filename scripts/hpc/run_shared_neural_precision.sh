#!/bin/bash
# Free-node one-day native parameter/state AD screen, not training.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-neural-precision-20260927"
HELPERS="$ROOT/runtime/transfer/shared-ffn-precision-20260926"
COMBINED="$ROOT/runtime/transfer/shared-combined-recurrent-20260926"
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
RUN_SUFFIX=${RUN_SUFFIX:-20260927-cache-fixed}
GPU_INDEX=${GPU_INDEX:-0}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
MODE=${1:?reference or candidate required}
shift
case "$MODE" in reference|candidate) ;; *) exit 2 ;; esac
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
if [[ $DRY_RUN != 1 ]]; then
  case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1 || true)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1 || true)
if [[ $DRY_RUN == 1 ]]; then
  CUDA_DRIVER=${CUDA_DRIVER:-/usr/lib64/libcuda.so.1}
  NVML_DRIVER=${NVML_DRIVER:-/usr/lib64/libnvidia-ml.so.1}
else
  [[ -n "$CUDA_DRIVER" && -n "$NVML_DRIVER" ]] || exit 2
fi
RUNTIME=(env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$DIAGNOSTICS/shared-neural-reference-20260927-staged.stages-cache"
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
PREFLIGHT=(timeout --signal=TERM --kill-after=10s 3m "${RUNTIME[@]}"
  "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu
  --output "$DIAGNOSTICS/shared-neural-runtime-20260927.json")
COMMAND=(timeout --signal=TERM --kill-after=10s 20m "${RUNTIME[@]}"
  "$OVERLAY/probe_shared_neural_precision.py" --mode "$MODE" --require-gpu
  --execution staged --neural-remat
  --process-seconds 1200 --final-neural-compile-seconds 600
  --stage-cache "$DIAGNOSTICS/shared-neural-reference-20260927-staged.stages-cache"
  --source-profile "$DIAGNOSTICS/shared-split-profile-c599236.json"
  --prepared-context "$DIAGNOSTICS/shared-expert-real-daily-context-c599236.pkl"
  --real-daily-probe-file "$HELPERS/probe_shared_expert_real_daily.py"
  --candidate-module-file "$COMBINED/shared_decoder_expert_execution.py"
  --backbone-module-file "$COMBINED/shared_backbone_execution.py"
  --output "$DIAGNOSTICS/shared-neural-$MODE-$RUN_SUFFIX.json"
  --result "$DIAGNOSTICS/shared-neural-$MODE-$RUN_SUFFIX.pkl")
if [[ $MODE == candidate ]]; then
  COMMAND+=(--reference-result "$DIAGNOSTICS/shared-neural-reference-$RUN_SUFFIX.pkl")
  COMMAND+=(--reuse-stage-report "$DIAGNOSTICS/shared-neural-reference-$RUN_SUFFIX.json")
fi
COMMAND+=("$@")
if [[ $DRY_RUN == 1 ]]; then
  if [[ $MODE == reference ]]; then printf '%q ' "${PREFLIGHT[@]}"; printf '\n'; fi
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
[[ -f "$OVERLAY/probe_shared_neural_precision.py" ]] || exit 2
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
if [[ $MODE == reference ]]; then
  if [[ -f "$DIAGNOSTICS/shared-neural-runtime-20260927.json" ]]; then
    echo 'reusing completed unchanged runtime preflight shared-neural-runtime-20260927.json'
  else
    "${PREFLIGHT[@]}"
  fi
fi
"${COMMAND[@]}"
