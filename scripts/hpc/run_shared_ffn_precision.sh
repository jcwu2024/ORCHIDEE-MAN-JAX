#!/bin/bash
# Free GPU only. Capture and fragment are separate bounded processes; no training.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-ffn-precision-20260926"}
COMBINED_OVERLAY=${COMBINED_OVERLAY:-"$ROOT/runtime/transfer/shared-combined-recurrent-20260926"}
GPU_INDEX=${GPU_INDEX:-0}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
MODE=${1:-capture}
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in capture|fragment) ;; *) exit 2 ;; esac
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
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
COMMAND=(timeout --signal=TERM --kill-after=10s 10m "${RUNTIME[@]}"
  "$OVERLAY/probe_shared_ffn_precision.py" --mode "$MODE"
  --real-daily-probe-file "$OVERLAY/probe_shared_expert_real_daily.py"
  --output "$DIAGNOSTICS/shared-ffn-$MODE-20260926.json"
  --payload "$DIAGNOSTICS/shared-ffn-payload-20260926.pkl" --require-gpu)
if [[ $MODE == capture ]]; then
  COMMAND+=(--source-profile "$DIAGNOSTICS/shared-split-profile-c599236.json"
    --prepared-context "$DIAGNOSTICS/shared-expert-real-daily-context-c599236.pkl"
    --candidate-module-file "$COMBINED_OVERLAY/shared_decoder_expert_execution.py"
    --backbone-module-file "$COMBINED_OVERLAY/shared_backbone_execution.py")
fi
COMMAND+=("$@")
PREFLIGHT=(timeout --signal=TERM --kill-after=10s 3m "${RUNTIME[@]}"
  "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu
  --output "$DIAGNOSTICS/shared-ffn-runtime-20260926.json")
if [[ $DRY_RUN == 1 ]]; then
  if [[ $MODE == capture ]]; then printf '%q ' "${PREFLIGHT[@]}"; printf '\n'; fi
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
[[ -f "$OVERLAY/probe_shared_ffn_precision.py" && -f "$OVERLAY/probe_shared_expert_real_daily.py" ]] || exit 2
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
if [[ $MODE == capture ]]; then "${PREFLIGHT[@]}"; fi
"${COMMAND[@]}"
