#!/bin/bash
# Free-node cache-only replay. Never submit paid work from this launcher.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-union-replay-20260927"
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
GPU_INDEX=${GPU_INDEX:-0}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1 || true)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1 || true)
if [[ $DRY_RUN == 1 ]]; then
  CUDA_DRIVER=${CUDA_DRIVER:-/usr/lib64/libcuda.so.1}
  NVML_DRIVER=${NVML_DRIVER:-/usr/lib64/libnvidia-ml.so.1}
else
  case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
  [[ -n "$CUDA_DRIVER" && -n "$NVML_DRIVER" ]] || exit 2
fi
COMMAND=(timeout --signal=TERM --kill-after=10s 500s env -u LD_LIBRARY_PATH
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
  singularity exec --nv
  --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/probe_shared_union_replay.py" --process-seconds 480 --call-seconds 100 --cache-seconds 120
  --stage-cache "$DIAGNOSTICS/shared-neural-reference-20260927-staged.stages-cache"
  --source-profile "$DIAGNOSTICS/shared-split-profile-c599236.json"
  --prepared-context "$DIAGNOSTICS/shared-expert-real-daily-context-c599236.pkl"
  --saved-result "$DIAGNOSTICS/shared-union-integrated-20260927.pkl"
  --real-daily-probe-file "$ROOT/runtime/transfer/shared-ffn-precision-20260926/probe_shared_expert_real_daily.py"
  --neural-probe-file "$ROOT/runtime/transfer/shared-neural-precision-20260927/probe_shared_neural_precision.py"
  --candidate-module-file "$OVERLAY/shared_decoder_expert_execution.py"
  --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --output "$DIAGNOSTICS/shared-union-replay-20260927.json")
COMMAND+=("$@")
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
