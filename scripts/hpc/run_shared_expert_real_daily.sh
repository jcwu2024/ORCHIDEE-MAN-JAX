#!/bin/bash
# Bounded real daily objective admission on free resources; no optimizer retention.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-expert-real-daily-20260925"}
PREPARE_ONLY=${PREPARE_ONLY:-0}
GPU_INDEX=${GPU_INDEX:-0}
PREPARED_CONTEXT=${PREPARED_CONTEXT:-"$ROOT/runtime/outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl"}
OUTPUT=${OUTPUT:-"$ROOT/runtime/outputs/diagnostics/shared-expert-real-daily-20260925.json"}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
MODE_ARGS=()
if [[ $PREPARE_ONLY == 1 ]]; then
  case $(hostname -s) in test0[1-4]|ibtest0[1-4]) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
  BACKEND_ENV=(SINGULARITYENV_JAX_PLATFORMS=cpu SINGULARITYENV_CUDA_VISIBLE_DEVICES=)
  CONTAINER_OPTIONS=(--bind /WORK:/WORK)
  MODE_ARGS+=(--prepare-only)
  RUN_TIMEOUT=15m
elif [[ $PREPARE_ONLY == 0 ]]; then
  case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
  case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  BACKEND_ENV=(SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX")
  CONTAINER_OPTIONS=(--nv --bind "/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
  MODE_ARGS+=(--require-gpu --candidate-module-file "$OVERLAY/shared_decoder_expert_execution.py")
  RUN_TIMEOUT=30m
else
  exit 2
fi
RUNTIME=(env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  "${BACKEND_ENV[@]}"
  SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-transformer-v1"
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data"
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec "${CONTAINER_OPTIONS[@]}"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
COMMAND=(timeout --signal=TERM --kill-after=30s "$RUN_TIMEOUT" "${RUNTIME[@]}"
  "$OVERLAY/probe_shared_expert_real_daily.py"
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json"
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json"
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json"
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json"
  --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json"
  --validation-manifest "$ROOT/runtime/outputs/training/fresh-method-validation-20260921/prepared_manifest.json"
  --joint-admission-report "$ROOT/runtime/outputs/diagnostics/shared-split-caed647-recurrent-resume1.json"
  --protocol "$CODE_ROOT/manifests/coarse_graining/shared_split_learning_v1.json"
  --source-profile "$ROOT/runtime/outputs/diagnostics/shared-split-profile-c599236.json"
  --input-bundle "$ROOT/runtime/outputs/diagnostics/shared-split-dual-inputs-c599236.pkl"
  --prepared-context "$PREPARED_CONTEXT" --output "$OUTPUT"
  "${MODE_ARGS[@]}" "$@")
PREFLIGHT=(timeout --signal=TERM --kill-after=30s 3m "${RUNTIME[@]}"
  "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py"
  --backend gpu --output "${OUTPUT%.json}.runtime.json")
if [[ $DRY_RUN == 1 ]]; then
  if [[ $PREPARE_ONLY == 0 ]]; then printf '%q ' "${PREFLIGHT[@]}"; printf '\n'; fi
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
cd "$CODE_ROOT"
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
echo "host=$(hostname -s) code=$(git rev-parse --short HEAD) prepare=$PREPARE_ONLY gpu=$GPU_INDEX"
uptime
grep Cpus_allowed_list /proc/self/status
if [[ $PREPARE_ONLY == 0 ]]; then
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  "${PREFLIGHT[@]}"
fi
"${COMMAND[@]}"
