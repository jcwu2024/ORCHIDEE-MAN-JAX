#!/bin/bash
# Bounded CPU input preparation or GPU replay; no model gradient compilation.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
ENTRY_FILE=${ENTRY_FILE:-"$ROOT/runtime/transfer/probe_shared_serialized_dual.py"}
OUTPUT=${OUTPUT:-"$ROOT/runtime/outputs/diagnostics/shared-split-serialized-dual-20260925.json"}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
GPU_DEVICE=${GPU_DEVICE:-0}
WORKER_GPU_DEVICE=${WORKER_GPU_DEVICE:-1}
RUN_TIMEOUT=${RUN_TIMEOUT:-30m}
PREPARE_INPUT_BUNDLE=${PREPARE_INPUT_BUNDLE:-0}
INSPECT_ONLY=${INSPECT_ONLY:-0}
EXTRA_ARGS=()
if [[ -n ${INPUT_BUNDLE:-} ]]; then EXTRA_ARGS+=(--input-bundle "$INPUT_BUNDLE"); fi
if [[ $PREPARE_INPUT_BUNDLE == 1 || $INSPECT_ONLY == 1 ]]; then
  [[ -n ${INPUT_BUNDLE:-} ]] || exit 2
  case $(hostname -s) in test0[1-4]|ibtest0[1-4]) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
  BACKEND_ENV=(SINGULARITYENV_JAX_PLATFORMS=cpu SINGULARITYENV_CUDA_VISIBLE_DEVICES=)
  CONTAINER_OPTIONS=(--bind /WORK:/WORK)
  if [[ $INSPECT_ONLY == 1 ]]; then
    EXTRA_ARGS+=(--inspect-only)
  else
    EXTRA_ARGS+=(--prepare-input-bundle)
  fi
elif [[ $PREPARE_INPUT_BUNDLE == 0 ]]; then
  case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) [[ $DRY_RUN == 1 || -n ${SLURM_JOB_ID:-} ]] || exit 2 ;; esac
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  BACKEND_ENV=(SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE")
  CONTAINER_OPTIONS=(--nv --bind "/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
  EXTRA_ARGS+=(--worker-cuda-device "$WORKER_GPU_DEVICE")
else
  exit 2
fi
COMMAND=(timeout --signal=TERM --kill-after=30s "$RUN_TIMEOUT" env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  "${BACKEND_ENV[@]}"
  SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-transformer-v1"
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data"
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec "${CONTAINER_OPTIONS[@]}"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$ENTRY_FILE"
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
  --output "$OUTPUT" --technical-updates 2 --repeats 3
  "${EXTRA_ARGS[@]}" "$@")
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
cd "$CODE_ROOT"
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
echo "host=$(hostname -s) code=$(git rev-parse --short HEAD) prepare=$PREPARE_INPUT_BUNDLE inspect=$INSPECT_ONLY gpu=$GPU_DEVICE worker_gpu=$WORKER_GPU_DEVICE"
uptime
grep Cpus_allowed_list /proc/self/status
if [[ $PREPARE_INPUT_BUNDLE == 0 && $INSPECT_ONLY == 0 ]]; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
fi
"${COMMAND[@]}"
