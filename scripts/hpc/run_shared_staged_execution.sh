#!/bin/bash
# Persistent reference first, then one selected microbatch; no sbatch here.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?set immutable staged benchmark checkout}
OUTPUT=${OUTPUT:?set report JSON}
STAGE=${STAGE:-reference}
MODE=${MODE:-gpu}
STATE_DIR=${STATE_DIR:-$ROOT/runtime/outputs/diagnostics/shared-staged-source100-v1}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
case "$CODE_ROOT" in "$ROOT"/runtime/checkouts/*) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT"/runtime/outputs/*.json) ;; *) exit 2 ;; esac
case "$STATE_DIR" in "$ROOT"/runtime/outputs/diagnostics/*) ;; *) exit 2 ;; esac
case "$STAGE" in reference|micro2|micro4) ;; *) exit 2 ;; esac
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
  --output "$OUTPUT" --stage "$STAGE" --state-dir "$STATE_DIR"
)
if [[ $MODE == cpu ]]; then
  case "$HOST" in test0[1-4]|ibtest0[1-4]) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
  COMMAND=(env PYTHONPATH="$CODE_ROOT" JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
    ORCHIDEE_REPO_ROOT="$CODE_ROOT" ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
    ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
    ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
    timeout --signal=TERM 5m "$ROOT/.venvs/orcjax_cpu/bin/python"
    -m scripts.hpc.probe_shared_execution "${COMMON_ARGS[@]}" --prepare-only "$@")
elif [[ $MODE == gpu ]]; then
  if [[ ${SLURM_JOB_PARTITION:-} == gnall && -n ${SLURM_JOB_ID:-} ]]; then
    GPU_DEVICE=${CUDA_VISIBLE_DEVICES:?Slurm must select the allocated GPU}
  else
    case "$HOST" in gln01|ibgln01|ibc13b02n04) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
    GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
    [[ $STAGE == reference || $DRY_RUN == 1 ]] || exit 2
  fi
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  COMMAND=(timeout --signal=TERM "${RUN_TIMEOUT:-40m}" env -u LD_LIBRARY_PATH
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
    SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_JAX_PLATFORMS=cuda,cpu
    SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
    SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-transformer-v1"
    SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
    SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
    SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data"
    SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets"
    SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs"
    SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
    SINGULARITYENV_OPENBLAS_NUM_THREADS=1
    SINGULARITYENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    singularity exec --nv --bind "$GPU_BINDS" --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose
    "$ROOT/.venvs/orcjax_gpu/bin/python" -m scripts.hpc.probe_shared_execution
    "${COMMON_ARGS[@]}" "$@")
else
  exit 2
fi
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
mkdir -p "$(dirname "$OUTPUT")"
cd "$CODE_ROOT"
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date '+%F %T %Z'
echo "host=$HOST code=$(git rev-parse --short HEAD) stage=$STAGE"
uptime
free -m
if [[ $MODE == gpu ]]; then
  nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
fi
"${COMMAND[@]}"
