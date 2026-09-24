#!/bin/bash
# Fixed100-update pilot on free nodes. Paid execution has no path here.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?set immutable checkout}
OUTPUT=${OUTPUT:?set report JSON under runtime/outputs}
MODE=${MODE:-gpu}
ENTRY_MODULE=${ENTRY_MODULE:-scripts.hpc.run_shared_learning}
case "$ENTRY_MODULE" in scripts.hpc.run_shared_learning|scripts.hpc.probe_shared_execution|scripts.hpc.probe_shared_projection_reuse|scripts.hpc.probe_shared_split_timing_cache|scripts.hpc.probe_shared_split_profile) ;; *) exit 2 ;; esac
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT"/runtime/outputs/*.json) ;; *) exit 2 ;; esac
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
HOST=$(hostname -s)
COMMON_ARGS=(
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json"
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json"
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json"
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json"
  --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json"
  --joint-admission-report "$ROOT/runtime/outputs/diagnostics/shared-joint-c2498e4-gpu.json"
  --validation-manifest "$ROOT/runtime/outputs/training/fresh-method-validation-20260921/prepared_manifest.json"
  --protocol "$CODE_ROOT/manifests/coarse_graining/shared_learning_v1.json"
  --output "$OUTPUT"
)
if [[ $MODE == cpu ]]; then
  case "$HOST" in test0[1-4]|ibtest0[1-4]) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
  COMMAND=(env
    ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
    ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
    ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" PYTHONPATH="$CODE_ROOT"
    JAX_ENABLE_X64=true JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
    timeout --signal=TERM "${RUN_TIMEOUT:-5m}" "$ROOT/.venvs/orcjax_cpu/bin/python"
    -m "$ENTRY_MODULE" "${COMMON_ARGS[@]}" "$@" --prepare-only)
elif [[ $MODE == gpu ]]; then
  case "$HOST" in gln01|ibgln01|ibc13b02n04) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
  GPU_DEVICE=${GPU_DEVICE:?select idle free GPU}
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  CACHE_ROOT="$ROOT/runtime/cache/jax/orcjax_gpu/shared-transformer-v1"
  CONTAINER=(env -u LD_LIBRARY_PATH
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
    SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_JAX_PLATFORMS=cuda,cpu
    SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
    SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT"
    SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
    SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
    SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data"
    SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
    SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
    SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
    SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
    singularity exec --nv --bind "$GPU_BINDS" --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose
    "$ROOT/.venvs/orcjax_gpu/bin/python")
  COMMAND=(timeout --signal=TERM "${RUN_TIMEOUT:-240m}" "${CONTAINER[@]}"
    -m "$ENTRY_MODULE" "${COMMON_ARGS[@]}" "$@")
  if [[ $ENTRY_MODULE == scripts.hpc.probe_shared_execution || ${CHECK_RUNTIME:-0} == 1 ]]; then
    RUNTIME_CHECK=(timeout --signal=TERM 5m "${CONTAINER[@]}"
      -m scripts.hpc.check_daily_training_runtime --backend gpu --output "${OUTPUT%.json}-runtime.json")
  fi
else
  exit 2
fi
if [[ $DRY_RUN == 1 ]]; then
  if [[ ${RUNTIME_CHECK+x} ]]; then printf '%q ' "${RUNTIME_CHECK[@]}"; printf '\n'; fi
  printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
mkdir -p "$(dirname "$OUTPUT")"
cd "$CODE_ROOT"
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
echo "host=$HOST code=$(git rev-parse --short HEAD) mode=$MODE output=$OUTPUT"
# The original learning runner reuses7359. New vmapped gradient execution runs
# one small deployed runtime check before its real-model comparison.
if [[ $MODE == gpu ]]; then mkdir -p "$CACHE_ROOT"; fi
if [[ ${RUNTIME_CHECK+x} ]]; then "${RUNTIME_CHECK[@]}"; fi
"${COMMAND[@]}"
