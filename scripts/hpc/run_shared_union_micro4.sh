#!/bin/bash
# Prepared diagnostic; paid --slurm execution requires envelope approval.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-union-micro4-20260929"}
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
UNION_OVERLAY="$ROOT/runtime/transfer/shared-union-integrated-20260927"
OUTPUT=${OUTPUT:-}
DRY_RUN=0
SLURM_MODE=0
RUN_PREFLIGHT=1
PREFLIGHT_REPORT=""
FORWARDED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --slurm) SLURM_MODE=1; shift ;;
    --reuse-runtime-preflight)
      [[ $# -ge 2 ]] || exit 2
      RUN_PREFLIGHT=0; PREFLIGHT_REPORT=$2; shift 2 ;;
    --output) [[ $# -ge 2 ]] || exit 2; OUTPUT=$2; shift 2 ;;
    --output=*) OUTPUT=${1#--output=}; shift ;;
    *) FORWARDED+=("$1"); shift ;;
  esac
done
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
  OUTPUT=${OUTPUT:-"$DIAGNOSTICS/shared-union-micro4-paid-20260929-${SLURM_JOB_ID:-dry-run}.json"}
  CACHE_DIR=${CACHE_DIR:-"$ROOT/runtime/cache/jax/orcjax_gpu/shared-union-micro4-paid-20260929-${SLURM_JOB_ID:-dry-run}"}
else
  GPU_BINDING=${GPU_INDEX:-0}
  case "$GPU_BINDING" in 0|1) ;; *) exit 2 ;; esac
  [[ -z ${SLURM_JOB_ID:-} ]] || { echo 'Free mode rejects Slurm execution; select --slurm.' >&2; exit 2; }
  OUTPUT=${OUTPUT:-"$DIAGNOSTICS/shared-union-micro4-20260929.json"}
  CACHE_DIR=${CACHE_DIR:-"$ROOT/runtime/cache/jax/orcjax_gpu/shared-union-micro4-20260929"}
fi
case $(readlink -m "$OUTPUT") in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
if [[ $DRY_RUN != 1 ]]; then
  if [[ $SLURM_MODE != 1 ]]; then
    case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
  fi
  [[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
  [[ -f "$OVERLAY/probe_shared_union_micro4.py" ]] || exit 2
  [[ ! -e "$OUTPUT" ]] || { echo 'Preserve prior output; choose a new stem.' >&2; exit 2; }
  if [[ $RUN_PREFLIGHT == 0 ]]; then [[ -f "$PREFLIGHT_REPORT" ]] || exit 2; fi
  if [[ $RUN_PREFLIGHT == 1 ]]; then [[ ! -e "${OUTPUT%.json}.runtime.json" ]] || exit 2; fi
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
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_BINDING"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_DIR"
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
  --output "${OUTPUT%.json}.runtime.json")
if [[ $RUN_PREFLIGHT == 1 ]]; then PREFLIGHT_REPORT="${OUTPUT%.json}.runtime.json"; fi
COMMAND=(timeout --signal=TERM --kill-after=10s 1520s "${RUNTIME[@]}"
  "$OVERLAY/probe_shared_union_micro4.py"
  --lower-seconds 180 --backend-seconds 900 --process-seconds 1500
  --first-execution-seconds 100 --memory-limit-gib 24
  --source-profile "$DIAGNOSTICS/shared-split-profile-c599236.json"
  --input-bundle "$DIAGNOSTICS/shared-split-dual-inputs-c599236.pkl"
  --prepared-context "$DIAGNOSTICS/shared-expert-real-daily-context-c599236.pkl"
  --baseline-admission-report "$DIAGNOSTICS/shared-combined-real-daily-20260925.json"
  --reference-result "$DIAGNOSTICS/shared-combined-real-daily-20260925.candidate.pkl"
  --canonical-reference-result "$DIAGNOSTICS/shared-expert-real-daily-20260925.reference.pkl"
  --real-daily-probe-file "$ROOT/runtime/transfer/shared-ffn-precision-20260926/probe_shared_expert_real_daily.py"
  --candidate-module-file "$UNION_OVERLAY/shared_decoder_expert_execution.py"
  --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --runtime-preflight-report "$PREFLIGHT_REPORT"
  --output "$OUTPUT")
# This expansion also works for an empty array with Bash4.2 nounset.
COMMAND+=(${FORWARDED[@]+"${FORWARDED[@]}"})
if [[ $DRY_RUN == 1 ]]; then
  if [[ $RUN_PREFLIGHT == 1 ]]; then printf '%q ' "${PREFLIGHT[@]}"; printf '\n';
  else printf 'reuse_runtime_preflight=%q\n' "$PREFLIGHT_REPORT"; fi
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
printf 'execution_mode_slurm=%s job_id=%s allocated_cpus=%s visible_gpu=%s\n' "$SLURM_MODE" "${SLURM_JOB_ID:-none}" "${SLURM_CPUS_PER_TASK:-unallocated}" "$GPU_BINDING"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
if [[ $RUN_PREFLIGHT == 1 ]]; then "${PREFLIGHT[@]}";
else printf 'reuse_runtime_preflight=%s\n' "$PREFLIGHT_REPORT"; fi
"${COMMAND[@]}"
