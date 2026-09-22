#!/bin/bash
# Daily-only compiler/cache diagnostic; no parameter updates or paid submission.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?set existing immutable shared-execution checkout}
DIAGNOSTIC_SCRIPT=${DIAGNOSTIC_SCRIPT:-$CODE_ROOT/scripts/hpc/probe_shared_compile_cache.py}
OUTPUT=${OUTPUT:?set diagnostic report path}
DIAGNOSTIC_MODE=${DIAGNOSTIC_MODE:-key-only}
MODE=${MODE:-gpu}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2 ;; esac
case "$DIAGNOSTIC_SCRIPT" in "$ROOT"/runtime/tools/*|"$CODE_ROOT"/scripts/hpc/*) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT"/runtime/outputs/*.json) ;; *) exit 2 ;; esac
case "$DIAGNOSTIC_MODE" in key-only|cached-only|compile) ;; *) exit 2 ;; esac
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
  --benchmark-protocol "$CODE_ROOT/manifests/coarse_graining/shared_execution_benchmark_v1.json"
  --resume-checkpoint "$ROOT/runtime/outputs/diagnostics/shared-learning-4408daf-gpu.update-000100.checkpoint.pkl"
  --output "$OUTPUT" --diagnostic-mode "$DIAGNOSTIC_MODE"
)
if [[ $MODE == cpu ]]; then
  case "$HOST" in test0[1-4]|ibtest0[1-4]) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
  COMMAND=(timeout --signal=TERM 3m env -u LD_LIBRARY_PATH
    SINGULARITYENV_PYTHONPATH="$CODE_ROOT" SINGULARITYENV_JAX_PLATFORMS=cpu
    SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_OPENBLAS_NUM_THREADS=1
    SINGULARITYENV_OMP_NUM_THREADS=2
    singularity exec --bind /WORK:/WORK --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose
    "$ROOT/.venvs/orcjax_gpu/bin/python"
    "$DIAGNOSTIC_SCRIPT" --self-test --output "$OUTPUT" --diagnostic-mode "$DIAGNOSTIC_MODE" "$@")
elif [[ $MODE == gpu ]]; then
  if [[ ${SLURM_JOB_PARTITION:-} == gnall && -n ${SLURM_JOB_ID:-} ]]; then
    GPU_DEVICE=${CUDA_VISIBLE_DEVICES:?Slurm must select the allocated GPU}
  else
    case "$HOST" in gln01|ibgln01|ibc13b02n04) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
    GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
    # Cold compilation belongs in an approved paid allocation for this probe.
    [[ $DIAGNOSTIC_MODE != compile || $DRY_RUN == 1 ]] || exit 2
  fi
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  COMMAND=(timeout --signal=TERM "${RUN_TIMEOUT:-15m}" env -u LD_LIBRARY_PATH
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
    SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_JAX_PLATFORMS=cuda,cpu
    SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
    SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-transformer-v1"
    SINGULARITYENV_JAX_EXPLAIN_CACHE_MISSES=true
    SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
    SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
    SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data"
    SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
    SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
    SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
    SINGULARITYENV_OPENBLAS_NUM_THREADS=1
    SINGULARITYENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    singularity exec --nv --bind "$GPU_BINDS" --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose
    "$ROOT/.venvs/orcjax_gpu/bin/python" "$DIAGNOSTIC_SCRIPT" "${COMMON_ARGS[@]}" "$@")
else
  exit 2
fi
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
mkdir -p "$(dirname "$OUTPUT")"
cd "$CODE_ROOT"
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date '+%F %T %Z'
echo "host=$HOST code=$(git rev-parse --short HEAD) mode=$MODE diagnostic=$DIAGNOSTIC_MODE"
uptime
free -m
if [[ $MODE == gpu ]]; then
  nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
fi
"${COMMAND[@]}"
