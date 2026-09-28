#!/bin/bash
# Free-node actual daily union native AD, saved field binary replay; no training.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-union-integrated-20260927"
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
DRY_RUN=0
SLURM_MODE=0
while [[ ${1:-} == --dry-run || ${1:-} == --slurm ]]; do
  case "$1" in --dry-run) DRY_RUN=1 ;; --slurm) SLURM_MODE=1 ;; esac
  shift
done
if [[ $SLURM_MODE == 1 ]]; then
  if [[ $DRY_RUN != 1 ]]; then
    [[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 ]] || exit 2
    [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || exit 2
  fi
  GPU_INDEX=${CUDA_VISIBLE_DEVICES:-0}
  [[ "$GPU_INDEX" != *,* ]] || exit 2
  case "$GPU_INDEX" in [0-7]|GPU-*) ;; *) exit 2 ;; esac
else
  GPU_INDEX=${GPU_INDEX:-0}
  case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1 || true)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1 || true)
if [[ $DRY_RUN == 1 ]]; then
  CUDA_DRIVER=${CUDA_DRIVER:-/usr/lib64/libcuda.so.1}
  NVML_DRIVER=${NVML_DRIVER:-/usr/lib64/libnvidia-ml.so.1}
else
  if [[ $SLURM_MODE != 1 ]]; then
    case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
  fi
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
  singularity exec --nv
  --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
PREFLIGHT=(timeout --signal=TERM --kill-after=10s 180s "${RUNTIME[@]}"
  "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu
  --output "$DIAGNOSTICS/shared-union-integrated-20260927.runtime.json")
COMMAND=(timeout --signal=TERM --kill-after=10s 1020s "${RUNTIME[@]}"
  "$OVERLAY/probe_shared_union_integrated.py"
  --process-seconds 1000 --final-neural-compile-seconds 600 --memory-limit-gib 24 --repeats 3
  --stage-cache "$DIAGNOSTICS/shared-neural-reference-20260927-staged.stages-cache"
  --source-profile "$DIAGNOSTICS/shared-split-profile-c599236.json"
  --prepared-context "$DIAGNOSTICS/shared-expert-real-daily-context-c599236.pkl"
  --real-daily-probe-file "$ROOT/runtime/transfer/shared-ffn-precision-20260926/probe_shared_expert_real_daily.py"
  --neural-probe-file "$ROOT/runtime/transfer/shared-neural-precision-20260927/probe_shared_neural_precision.py"
  --candidate-module-file "$OVERLAY/shared_decoder_expert_execution.py"
  --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --reuse-stage-report "$DIAGNOSTICS/shared-neural-reference-20260927-cache-fixed.json"
  --output "$DIAGNOSTICS/shared-union-integrated-20260927.json"
  --result "$DIAGNOSTICS/shared-union-integrated-20260927.pkl")
COMMAND+=("$@")
if [[ $DRY_RUN == 1 ]]; then
  printf '%q ' "${PREFLIGHT[@]}"; printf '\n'
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
printf 'execution_mode_slurm=%s job_id=%s allocated_cpus=%s visible_gpu=%s\n' "$SLURM_MODE" "${SLURM_JOB_ID:-none}" "${SLURM_CPUS_PER_TASK:-unallocated}" "$GPU_INDEX"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${PREFLIGHT[@]}"
"${COMMAND[@]}"
