#!/bin/bash
# Candidate-first FULL64 validation and measurement on one allocated GPU.
# The launcher is a preparation artifact; submission still requires approval.
#SBATCH -J orchjax-srv-b64
#SBATCH -p gnall
#SBATCH --nodelist=ibc13b03n03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o runtime/logs/global_server_gpu_b64_%j.txt
#SBATCH -e runtime/logs/global_server_gpu_b64_%j.txt
#SBATCH --no-requeue

set -euo pipefail

PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
CODE_DIR=${CODE_DIR:-$PROJECT/runtime/worktrees/global-throughput-v1}
CODE_DIR=$(realpath -m "$CODE_DIR")
case "$CODE_DIR" in
  "$PROJECT"/runtime/worktrees/*) ;;
  *) echo "CODE_DIR must stay under the project runtime worktrees" >&2; exit 2 ;;
esac
PYTHON=$PROJECT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose
PREPARED_INPUT=${PREPARED_INPUT:-$PROJECT/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl}
FULL_REFERENCE_DIR=${FULL_REFERENCE_DIR:-$PROJECT/runtime/outputs/research/global_throughput/gpu-round4/model}
CONFIG=${CONFIG:-$CODE_DIR/configs/orchidee_man_250919.yaml}
ASSET_ROOT=${ASSET_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2}
RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-b64-${SLURM_JOB_ID:-dry-run}}
RUN_ROOT=$(realpath -m "$RUN_ROOT")
case "$RUN_ROOT" in
  "$PROJECT"/runtime/outputs/research/global_throughput/server-benchmark/*) ;;
  *) echo "RUN_ROOT must stay under the project server-benchmark output directory" >&2; exit 2 ;;
esac
OUTPUT_DIR=$RUN_ROOT/candidate-first
ENTRY=$CODE_DIR/scripts/hpc/global_batch_gpu_entry.py
MEASURE_B4_BASELINE=${MEASURE_B4_BASELINE:-0}
GPU_SELECTOR=${SLURM_JOB_GPUS:-}
GPU_SELECTOR=${GPU_SELECTOR//[[:space:]]/}
DRY_RUN=0

if [[ ${1:-} == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ -n $GPU_SELECTOR && $GPU_SELECTOR == *,* ]]; then
  echo "requires exactly one global GPU ID in SLURM_JOB_GPUS" >&2
  exit 2
fi
[[ $MEASURE_B4_BASELINE == 0 || $MEASURE_B4_BASELINE == 1 ]] || {
  echo "MEASURE_B4_BASELINE must be 0 or 1" >&2; exit 2;
}
[[ ${SLURM_CPUS_PER_TASK:-4} == 4 ]] || { echo "requires exactly four auxiliary Slurm CPUs" >&2; exit 2; }

if [[ $DRY_RUN -eq 0 ]]; then
  [[ -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-1} == 1 ]] || { echo "requires one active Slurm task" >&2; exit 2; }
  [[ ${SLURM_JOB_PARTITION:-} == gnall ]] || { echo "requires gnall" >&2; exit 2; }
  [[ -n ${SLURM_JOB_GPUS:-} && -n ${CUDA_VISIBLE_DEVICES:-} ]] || {
    echo "Slurm GPU allocation variables are required" >&2; exit 2;
  }
  [[ -n $GPU_SELECTOR && $GPU_SELECTOR != *,* ]] || {
    echo "requires exactly one global GPU ID in SLURM_JOB_GPUS" >&2; exit 2;
  }
  [[ $CUDA_VISIBLE_DEVICES != *,* ]] || { echo "requires exactly one visible allocated GPU" >&2; exit 2; }
  for path in "$PYTHON" "$ENTRY" "$PREPARED_INPUT" "$FULL_REFERENCE_DIR" "$CONFIG" "$ASSET_ROOT" "$IMAGE"; do
    [[ -e $path ]] || { echo "required path missing: $path" >&2; exit 2; }
  done
  test -z "$(cd "$CODE_DIR" && git status --porcelain --untracked-files=all)" || {
    echo "benchmark checkout must be clean and immutable" >&2; exit 2;
  }
  [[ ! -e $RUN_ROOT ]] || { echo "RUN_ROOT must be new: $RUN_ROOT" >&2; exit 2; }
fi

export PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export ORCHIDEE_REPO_ROOT=$CODE_DIR
export ORCHIDEE_DATA_ROOT=$PROJECT/runtime/data
export ORCHIDEE_OUTPUT_ROOT=$OUTPUT_DIR

args=(--prepared-input "$PREPARED_INPUT" --full-reference-dir "$FULL_REFERENCE_DIR"
  --output-dir "$OUTPUT_DIR" --config "$CONFIG" --asset-root "$ASSET_ROOT"
  --sizes 64 16 --warm-repeats 5 --cache-mode off --execution-order candidate-first)
if [[ $MEASURE_B4_BASELINE == 1 ]]; then
  args+=(--measure-b4-baseline)
fi

if [[ $DRY_RUN -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'<slurm-assigned-device>'}
  printf 'partition=gnall node=ibc13b03n03 nodes=1 tasks=1 cpus=4 gpus=1 wall=01:00:00 output=%q full_reference=%q measure_b4=%s\n' \
    "$OUTPUT_DIR" "$FULL_REFERENCE_DIR" "$MEASURE_B4_BASELINE"
  printf 'host_gpu_identity_source=SLURM_JOB_GPUS global_gpu_selector=%q preserved_cuda_visible_devices=%q alloc_tres=%q\n' \
    "${GPU_SELECTOR:-<from-slurm>}" "$CUDA_VISIBLE_DEVICES" "${ORCHJAX_SLURM_ALLOC_TRES:-<from-scontrol>}"
  CUDA_DRIVER=$(readlink -m /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -m /usr/lib64/libnvidia-ml.so.1)
  BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  printf 'command='; printf '%q ' "$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
    /usr/bin/time -v singularity exec --nv --bind "$BINDS" --pwd "$CODE_DIR" "$IMAGE" "$PYTHON" "$ENTRY" "${args[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$RUN_ROOT" "$PROJECT/runtime/logs"
ALLOC_TRES_RECORD=$("$SLURM_BIN/scontrol" show job -o "$SLURM_JOB_ID" | awk '{
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^AllocTRES=/) allocated = substr($i, 11)
    else if ($i ~ /^TRES=/) observed = substr($i, 6)
  }
} END {
  if (allocated != "") printf "AllocTRES\t%s\n", allocated
  else if (observed != "") printf "TRES\t%s\n", observed
}')
[[ -n $ALLOC_TRES_RECORD ]] || { echo "Slurm did not report allocated TRES" >&2; exit 2; }
IFS=$'\t' read -r ALLOC_TRES_SOURCE ALLOC_TRES <<< "$ALLOC_TRES_RECORD"
export ORCHJAX_SLURM_ALLOC_TRES=$ALLOC_TRES
export ORCHJAX_SLURM_ALLOC_TRES_SOURCE=$ALLOC_TRES_SOURCE
GPU_INFO=$(nvidia-smi -i "$GPU_SELECTOR" --query-gpu=uuid,name,driver_version --format=csv,noheader)
GPU_UUID=${GPU_INFO%%,*}
GPU_REMAINDER=${GPU_INFO#*, }
GPU_NAME=${GPU_REMAINDER%%,*}
GPU_DRIVER=${GPU_REMAINDER##*, }
[[ -n $GPU_UUID && -n $GPU_NAME && -n $GPU_DRIVER ]] || { echo "cannot resolve allocated GPU identity" >&2; exit 2; }
export ORCHJAX_PHYSICAL_GPU_UUID=$GPU_UUID
export ORCHJAX_GPU_DRIVER_VERSION=$GPU_DRIVER
export ORCHJAX_SLURM_GLOBAL_GPU_ID=$GPU_SELECTOR

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
export SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export SINGULARITYENV_JAX_ENABLE_X64=true
export SINGULARITYENV_ORCHIDEE_REPO_ROOT=$ORCHIDEE_REPO_ROOT
export SINGULARITYENV_ORCHIDEE_DATA_ROOT=$ORCHIDEE_DATA_ROOT
export SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT=$ORCHIDEE_OUTPUT_ROOT
export SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_scale.py
export SINGULARITYENV_ORCHJAX_PHYSICAL_GPU_UUID=$ORCHJAX_PHYSICAL_GPU_UUID
export SINGULARITYENV_ORCHJAX_GPU_DRIVER_VERSION=$ORCHJAX_GPU_DRIVER_VERSION
export SINGULARITYENV_ORCHJAX_SLURM_GLOBAL_GPU_ID=$ORCHJAX_SLURM_GLOBAL_GPU_ID
export SINGULARITYENV_ORCHJAX_SLURM_ALLOC_TRES=$ORCHJAX_SLURM_ALLOC_TRES
export SINGULARITYENV_ORCHJAX_SLURM_ALLOC_TRES_SOURCE=$ORCHJAX_SLURM_ALLOC_TRES_SOURCE
export SINGULARITYENV_SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export SINGULARITYENV_SLURM_JOB_ID=$SLURM_JOB_ID
export SINGULARITYENV_SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION
export SINGULARITYENV_SLURM_JOB_GPUS=$SLURM_JOB_GPUS
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_NUMEXPR_NUM_THREADS=1

{
  date -u +%FT%TZ
  hostname
  uptime
  "$SLURM_BIN/scontrol" show job --details "$SLURM_JOB_ID"
  taskset -pc $$
  lscpu -e=CPU,CORE,SOCKET,ONLINE
  (cd "$CODE_DIR" && git rev-parse HEAD)
  printf 'slurm_cpus_per_task=%s slurm_job_cpus_per_node=%s slurm_job_gpus=%s alloc_tres=%s alloc_tres_source=%s\n' \
    "$SLURM_CPUS_PER_TASK" "${SLURM_JOB_CPUS_PER_NODE:-unknown}" "$SLURM_JOB_GPUS" \
    "$ORCHJAX_SLURM_ALLOC_TRES" "$ORCHJAX_SLURM_ALLOC_TRES_SOURCE"
  printf 'allocated_gpu_global_id=%s uuid=%s name=%s driver=%s cuda_visible_devices=%s\n' \
    "$ORCHJAX_SLURM_GLOBAL_GPU_ID" "$ORCHJAX_PHYSICAL_GPU_UUID" "$GPU_NAME" "$ORCHJAX_GPU_DRIVER_VERSION" "$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
} | tee "$RUN_ROOT/launch-context.txt"

mkdir -p "$OUTPUT_DIR"
cd "$CODE_DIR"
"$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
  /usr/bin/time -v singularity exec --nv --bind "$BINDS" --pwd "$CODE_DIR" \
  "$IMAGE" "$PYTHON" "$ENTRY" "${args[@]}"
