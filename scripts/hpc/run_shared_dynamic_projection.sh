#!/bin/bash
# Free-node FP32 dynamic K/V component only; no full neural gradient or training.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-dynamic-projection-20260927"
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
MODE=${1:?inspect, capture or compare required}
shift
case "$MODE" in inspect|capture|compare) ;; *) exit 2 ;; esac
GPU_INDEX=${GPU_INDEX:-0}
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
PLATFORMS=cuda,cpu
TIMING_BOUNDARY=${TIMING_BOUNDARY:-banks}
case "$TIMING_BOUNDARY" in banks|scalar) ;; *) exit 2 ;; esac
OUTPUT_MODE="$MODE"
if [[ $MODE == compare && $TIMING_BOUNDARY == scalar ]]; then OUTPUT_MODE=compare-scalar; fi
SINGULARITY_COMMAND=(singularity exec)
if [[ $MODE == inspect ]]; then PLATFORMS=cpu; else SINGULARITY_COMMAND+=(--nv); fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1 || true)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1 || true)
if [[ $DRY_RUN == 1 ]]; then
  CUDA_DRIVER=${CUDA_DRIVER:-/usr/lib64/libcuda.so.1}
  NVML_DRIVER=${NVML_DRIVER:-/usr/lib64/libnvidia-ml.so.1}
else
  case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
  [[ -n "$CUDA_DRIVER" && -n "$NVML_DRIVER" ]] || exit 2
fi
COMMAND=(timeout --signal=TERM --kill-after=10s 10m env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS="$PLATFORMS" SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$DIAGNOSTICS/shared-dynamic-projection-20260927.cache"
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  "${SINGULARITY_COMMAND[@]}"
  --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/probe_shared_dynamic_projection.py" --mode "$MODE"
  --timing-boundary "$TIMING_BOUNDARY"
  --process-seconds 590 --stage-cache "$DIAGNOSTICS/shared-dynamic-projection-20260927.cache"
  --source-profile "$DIAGNOSTICS/shared-split-profile-c599236.json"
  --prepared-context "$DIAGNOSTICS/shared-expert-real-daily-context-c599236.pkl"
  --real-daily-probe-file "$ROOT/runtime/transfer/shared-ffn-precision-20260926/probe_shared_expert_real_daily.py"
  --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --payload "$DIAGNOSTICS/shared-dynamic-projection-20260927.payload.pkl"
  --output "$DIAGNOSTICS/shared-dynamic-projection-20260927.$OUTPUT_MODE.json")
COMMAND+=("$@")
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
