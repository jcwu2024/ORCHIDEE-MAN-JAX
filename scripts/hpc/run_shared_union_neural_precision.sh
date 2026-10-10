#!/bin/bash
# One candidate-only full-state screen; separate micro4 requires its passing report.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY=${OVERLAY:-""}
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
DRY_RUN=0
SLURM_MODE=0
MICRO4=0
CACHE_COMPATIBILITY_ONLY=0
SCREEN_REPORT=""
OUTPUT=""
FORWARDED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --slurm) SLURM_MODE=1; shift ;;
    --micro4) MICRO4=1; shift ;;
    --cache-compatibility-only) CACHE_COMPATIBILITY_ONLY=1; shift ;;
    --screen-report) [[ $# -ge 2 ]] || exit 2; SCREEN_REPORT=$2; shift 2 ;;
    --output) [[ $# -ge 2 ]] || exit 2; OUTPUT=$2; shift 2 ;;
    --output=*) OUTPUT=${1#--output=}; shift ;;
    *) FORWARDED+=("$1"); shift ;;
  esac
done
if [[ $CACHE_COMPATIBILITY_ONLY == 1 ]]; then
  [[ $MICRO4 == 0 ]] || { echo 'Cache admission does not authorize a precision micro4 screen.' >&2; exit 2; }
  OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-union-neural-precision-20260929-v3"}
  PROCESS_SECONDS=520
  OUTER_SECONDS=540
  DIAGNOSTIC_NAME=shared-union-cache-compatibility
  PROBE_MODE_ARGS=(--cache-compatibility-only)
else
  OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-union-neural-precision-20260929-v2"}
  PROCESS_SECONDS=1200
  OUTER_SECONDS=1240
  DIAGNOSTIC_NAME=shared-union-neural-precision
  PROBE_MODE_ARGS=()
fi
PREFLIGHT_REPORT="$DIAGNOSTICS/shared-union-micro4-paid-20260929-15461046.runtime.json"
if [[ $MICRO4 == 1 ]]; then
  [[ -n "$SCREEN_REPORT" ]] || { echo 'Micro4 requires the passing full parameter/state screen.' >&2; exit 2; }
  MODE_ARGS=()
  [[ $DRY_RUN == 1 ]] && MODE_ARGS+=(--dry-run)
  [[ $SLURM_MODE == 1 ]] && MODE_ARGS+=(--slurm)
  OUTPUT=${OUTPUT:-"$DIAGNOSTICS/shared-union-neural-micro4-20260929-${SLURM_JOB_ID:-free}.json"}
  if [[ $DRY_RUN != 1 && $SLURM_MODE != 1 ]]; then
    [[ -z $(nvidia-smi -i "${GPU_INDEX:-0}" --query-compute-apps=pid --format=csv,noheader) ]] || exit 2
    [[ $(nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=utilization.gpu --format=csv,noheader,nounits) == 0 ]] || exit 2
  fi
  exec env OVERLAY="$OVERLAY" CACHE_DIR="${OUTPUT%.json}.jax-cache" bash "$OVERLAY/run_shared_union_micro4.sh" ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} \
    --reuse-runtime-preflight "$PREFLIGHT_REPORT" --output "$OUTPUT" \
    --neural-probe-file "$OVERLAY/probe_shared_neural_precision.py" \
    --required-precision-screen "$SCREEN_REPORT" ${FORWARDED[@]+"${FORWARDED[@]}"}
fi
if [[ $SLURM_MODE == 1 ]]; then
  if [[ $DRY_RUN != 1 ]]; then
    [[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 ]] || exit 2
    [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || exit 2
  fi
  GPU_BINDING=${CUDA_VISIBLE_DEVICES:-0}
  case "$GPU_BINDING" in [0-7]|GPU-?*) ;; *) exit 2 ;; esac
  [[ "$GPU_BINDING" != *,* && "$GPU_BINDING" != *[[:space:]]* ]] || exit 2
  for GPU_COUNT in "${SLURM_GPUS:-}" "${SLURM_GPUS_ON_NODE:-}" "${SLURM_GPUS_PER_TASK:-}"; do
    case "$GPU_COUNT" in ''|1|*:1) ;; *) exit 2 ;; esac
  done
  for GPU_SET in "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}"; do
    case "$GPU_SET" in *,*|[0-9]-[0-9]) exit 2 ;; esac
  done
  OUTPUT=${OUTPUT:-"$DIAGNOSTICS/$DIAGNOSTIC_NAME-paid-20260929-${SLURM_JOB_ID:-dry-run}.json"}
else
  GPU_BINDING=${GPU_INDEX:-0}
  case "$GPU_BINDING" in 0|1) ;; *) exit 2 ;; esac
  [[ -z ${SLURM_JOB_ID:-} ]] || exit 2
  OUTPUT=${OUTPUT:-"$DIAGNOSTICS/$DIAGNOSTIC_NAME-20260929.json"}
fi
STAGE_CACHE="${OUTPUT%.json}.stages-cache"
case $(readlink -m "$OUTPUT") in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
if [[ $DRY_RUN != 1 ]]; then
  if [[ $SLURM_MODE != 1 ]]; then
    case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
    # Resident external processes disqualify a free GPU even at zero utilization.
    [[ -z $(nvidia-smi -i "$GPU_BINDING" --query-compute-apps=pid --format=csv,noheader) ]] || exit 2
    [[ $(nvidia-smi -i "$GPU_BINDING" --query-gpu=utilization.gpu --format=csv,noheader,nounits) == 0 ]] || exit 2
  fi
  [[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
  [[ -f "$PREFLIGHT_REPORT" && -f "$OVERLAY/probe_shared_union_neural_precision.py" ]] || exit 2
  [[ ! -e "$OUTPUT" && ! -e "${OUTPUT%.json}.pkl" && ! -e "$STAGE_CACHE" ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1 || true)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1 || true)
if [[ $DRY_RUN == 1 ]]; then
  CUDA_DRIVER=${CUDA_DRIVER:-/usr/lib64/libcuda.so.1}
  NVML_DRIVER=${NVML_DRIVER:-/usr/lib64/libnvidia-ml.so.1}
else
  [[ -n "$CUDA_DRIVER" && -n "$NVML_DRIVER" ]] || exit 2
fi
COMMAND=(timeout --signal=TERM --kill-after=10s "${OUTER_SECONDS}s" env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_BINDING"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$STAGE_CACHE"
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv
  --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/probe_shared_union_neural_precision.py"
  --source-profile "$DIAGNOSTICS/shared-split-profile-c599236.json"
  --prepared-context "$DIAGNOSTICS/shared-expert-real-daily-context-c599236.pkl"
  --reference-result "$DIAGNOSTICS/shared-union-integrated-20260927.pkl"
  --reuse-stage-report "$DIAGNOSTICS/shared-union-integrated-20260927.json"
  --reference-stage-cache "$DIAGNOSTICS/shared-neural-reference-20260927-staged.stages-cache"
  --stage-cache "$STAGE_CACHE"
  --neural-probe-file "$OVERLAY/probe_shared_neural_precision.py"
  --real-daily-probe-file "$ROOT/runtime/transfer/shared-ffn-precision-20260926/probe_shared_expert_real_daily.py"
  --candidate-module-file "$ROOT/runtime/transfer/shared-union-integrated-20260927/shared_decoder_expert_execution.py"
  --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --runtime-preflight-report "$PREFLIGHT_REPORT"
  --process-seconds "$PROCESS_SECONDS" --neural-forward-compile-seconds 300 --final-neural-compile-seconds 600
  --memory-limit-gib 24 --output "$OUTPUT" --result "${OUTPUT%.json}.pkl")
COMMAND+=(${PROBE_MODE_ARGS[@]+"${PROBE_MODE_ARGS[@]}"})
COMMAND+=(${FORWARDED[@]+"${FORWARDED[@]}"})
if [[ $DRY_RUN == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
printf 'execution_mode_slurm=%s job_id=%s allocated_cpus=%s visible_gpu=%s\n' "$SLURM_MODE" "${SLURM_JOB_ID:-none}" "${SLURM_CPUS_PER_TASK:-unallocated}" "$GPU_BINDING"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
