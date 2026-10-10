#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-update-tree-slice-20260929"}
DIAGNOSTICS="$ROOT/runtime/outputs/diagnostics"
OUTPUT=${OUTPUT:-}
SOURCE_PROFILE=${SOURCE_PROFILE:-"$DIAGNOSTICS/shared-split-profile-c599236.json"}
INPUT_BUNDLE=${INPUT_BUNDLE:-"$DIAGNOSTICS/shared-split-dual-inputs-c599236.pkl"}
SUPPORT_OWNER=${SUPPORT_OWNER:-"$ROOT/runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py"}
DAILY_ADMISSION=${DAILY_ADMISSION:-"$DIAGNOSTICS/shared-combined-real-daily-20260925.json"}
T7_ADMISSION=${T7_ADMISSION:-"$DIAGNOSTICS/shared-combined-recurrent-T7-20260926.json"}
T30_ADMISSION=${T30_ADMISSION:-"$DIAGNOSTICS/shared-combined-recurrent-T30-20260926.json"}
DRY_RUN=0
SLURM_MODE=0
INSPECT_ONLY=0
INCLUDE_DAILY64=0
PROFILE_TRACE_DIR=${PROFILE_TRACE_DIR:-}
FORWARDED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --slurm) SLURM_MODE=1; shift ;;
    --inspect-only) INSPECT_ONLY=1; shift ;;
    --include-daily64) INCLUDE_DAILY64=1; shift ;;
    --output) [[ $# -ge 2 ]] || exit 2; OUTPUT=$2; shift 2 ;;
    --output=*) OUTPUT=${1#--output=}; shift ;;
    --profile-trace-dir) [[ $# -ge 2 ]] || exit 2; PROFILE_TRACE_DIR=$2; shift 2 ;;
    --profile-trace-dir=*) PROFILE_TRACE_DIR=${1#--profile-trace-dir=}; shift ;;
    *) FORWARDED+=("$1"); shift ;;
  esac
done
if [[ -z $OUTPUT ]]; then OUTPUT="$DIAGNOSTICS/shared-update-tree-slice-${SLURM_JOB_ID:-manual}.json"; fi
if [[ $INSPECT_ONLY == 1 ]]; then
  case $(readlink -m "$OUTPUT") in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
  CPU_COMMAND=(env ORCHIDEE_REPO_ROOT="$CODE_ROOT" PYTHONPATH="$CODE_ROOT" \
    "$ROOT/.venvs/orcjax_cpu/bin/python" "$OVERLAY/probe_shared_update_tree_slice.py" \
    --source-profile "$SOURCE_PROFILE" --input-bundle "$INPUT_BUNDLE" \
    --slice-helper-file "$OVERLAY/shared_daily_microbatch.py" \
    --support-owner-file "$SUPPORT_OWNER" \
    --daily-admission-report "$DAILY_ADMISSION" \
    --seven-admission-report "$T7_ADMISSION" --thirty-admission-report "$T30_ADMISSION" \
    --output "$OUTPUT" --inspect-only)
  CPU_COMMAND+=(${FORWARDED[@]+"${FORWARDED[@]}"})
  if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${CPU_COMMAND[@]}"; printf '\n'; exit 0; fi
  [[ -z ${SLURM_JOB_ID:-} ]] || exit 2
  case $(hostname -s) in test0[1-4]|ibtest0[1-4]) ;; *) exit 2 ;; esac
  [[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
  "${CPU_COMMAND[@]}"
  exit $?
fi
if [[ $SLURM_MODE == 1 ]]; then
  if [[ $DRY_RUN != 1 ]]; then [[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 && -n ${CUDA_VISIBLE_DEVICES:-} ]] || exit 2; fi
  GPU_BINDING=${CUDA_VISIBLE_DEVICES:-0}
  case "$GPU_BINDING" in [0-7]|GPU-?*) ;; *) exit 2 ;; esac
  [[ "$GPU_BINDING" != *,* ]] || exit 2
  for GPU_COUNT in "${SLURM_GPUS:-}" "${SLURM_GPUS_ON_NODE:-}" "${SLURM_GPUS_PER_TASK:-}"; do case "$GPU_COUNT" in ''|1|*:1) ;; *) exit 2 ;; esac; done
  for GPU_SET in "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}"; do case "$GPU_SET" in *,*|[0-9]-[0-9]) exit 2 ;; esac; done
  CACHE_DIR=${CACHE_DIR:-"$ROOT/runtime/cache/jax/orcjax_gpu/shared-update-bottleneck-20260930-${SLURM_JOB_ID:-dry-run}"}
else
  GPU_BINDING=${GPU_INDEX:-0}
  case "$GPU_BINDING" in 0|1) ;; *) exit 2 ;; esac
  [[ -z ${SLURM_JOB_ID:-} ]] || exit 2
  CACHE_DIR=${CACHE_DIR:-"$ROOT/runtime/cache/jax/orcjax_gpu/shared-update-bottleneck-20260930"}
fi
case $(readlink -m "$OUTPUT") in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
if [[ -n $PROFILE_TRACE_DIR ]]; then
  case $(readlink -m "$PROFILE_TRACE_DIR") in "$ROOT/runtime/outputs/diagnostics/"*) ;; *) exit 2 ;; esac
fi
if [[ $DRY_RUN != 1 ]]; then
  if [[ $SLURM_MODE != 1 ]]; then
    case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
    GPU_LINE=$(nvidia-smi -i "$GPU_BINDING" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    read -r GPU_MEMORY GPU_UTIL <<<"${GPU_LINE//,/ }"
    [[ $GPU_MEMORY =~ ^[0-9]+$ && $GPU_UTIL =~ ^[0-9]+$ && $GPU_MEMORY -le 2000 && $GPU_UTIL -le 5 ]] || { echo "GPU is occupied: $GPU_LINE" >&2; exit 2; }
    GPU_PIDS=$(nvidia-smi -i "$GPU_BINDING" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z "$GPU_PIDS" ]] || { echo "GPU has resident compute processes: $GPU_PIDS" >&2; exit 2; }
  fi
  [[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
  [[ -f "$OVERLAY/probe_shared_update_tree_slice.py" ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1 || true)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1 || true)
if [[ $DRY_RUN == 1 ]]; then CUDA_DRIVER=${CUDA_DRIVER:-/usr/lib64/libcuda.so.1}; NVML_DRIVER=${NVML_DRIVER:-/usr/lib64/libnvidia-ml.so.1}; fi
RUNTIME=(env -u LD_LIBRARY_PATH SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_BINDING"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_DIR" SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
COMMAND=(timeout --signal=TERM --kill-after=10s 1080 "${RUNTIME[@]}" "$OVERLAY/probe_shared_update_tree_slice.py"
  --source-profile "$SOURCE_PROFILE" --input-bundle "$INPUT_BUNDLE"
  --slice-helper-file "$OVERLAY/shared_daily_microbatch.py"
  --support-owner-file "$SUPPORT_OWNER"
  --daily-admission-report "$DAILY_ADMISSION"
  --seven-admission-report "$T7_ADMISSION" --thirty-admission-report "$T30_ADMISSION"
  --output "$OUTPUT" --repeats 3)
if [[ $INSPECT_ONLY == 1 ]]; then COMMAND+=(--inspect-only); fi
if [[ $INCLUDE_DAILY64 == 1 ]]; then COMMAND+=(--include-daily64); fi
if [[ -n $PROFILE_TRACE_DIR ]]; then COMMAND+=(--profile-trace-dir "$PROFILE_TRACE_DIR"); fi
COMMAND+=(${FORWARDED[@]+"${FORWARDED[@]}"})
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
printf 'execution_mode_slurm=%s job_id=%s allocated_cpus=%s visible_gpu=%s\n' "$SLURM_MODE" "${SLURM_JOB_ID:-none}" "${SLURM_CPUS_PER_TASK:-unallocated}" "$GPU_BINDING"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
