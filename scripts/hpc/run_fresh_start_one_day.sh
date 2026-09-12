#!/bin/bash
# Bounded free/approved-Slurm GPU run; no checkpoint chain.
set -euo pipefail
trap 'RUN_STATUS=$?; echo "run_exit_status=$RUN_STATUS"' EXIT
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:-$ROOT}
case "$CODE_ROOT" in "$ROOT"|"$ROOT"/runtime/checkouts/*) ;; *) exit 2;; esac
GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/fresh-start-one-day-v1.json}
case "$(hostname -s)" in
  gln01|ibgln01|ibc13b02n04) ;;
  *) [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_PARTITION:-}" == gnall ]] || exit 2 ;;
esac
case "$OUTPUT" in "$ROOT/runtime/outputs/"*.json) ;; *) exit 2 ;; esac
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
CACHE_ROOT=$ROOT/runtime/cache/jax/orcjax_gpu/fresh-start-one-day-v1
PYTHON_MODULE=${PYTHON_MODULE:-scripts.hpc.run_fresh_start_one_day}
RUN_TIMEOUT=${RUN_TIMEOUT:-60m}
mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT"
cd "$CODE_ROOT"
echo "host=$(hostname -s) code=$(git rev-parse --short HEAD) gpu=$GPU_DEVICE output=$OUTPUT"
env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_JAX_PLATFORMS=cuda \
  SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT" \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT" \
  timeout --signal=TERM "$RUN_TIMEOUT" singularity exec --nv --bind "$GPU_BINDS" --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose \
    "$ROOT/.venvs/orcjax_gpu/bin/python" -m "$PYTHON_MODULE" \
    --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
    --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
    --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
    --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
    --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
    --prepared-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json" \
    --output "$OUTPUT" "$@"
