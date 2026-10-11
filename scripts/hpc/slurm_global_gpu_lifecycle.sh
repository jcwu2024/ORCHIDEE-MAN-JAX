#!/usr/bin/env bash
# Two-arm B1024 lifecycle comparison. Submission requires separate approval.
#SBATCH -J orchjax-gpu-life
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_%j.txt

set -Eeuo pipefail
STEP=argument_admission
trap 'code=$?; printf "GPU lifecycle launcher failed: step=%s line=%s exit=%s command=%q\n" "$STEP" "$LINENO" "$code" "$BASH_COMMAND" >&2' ERR

PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
IMAGE=/apps/soft/sif/foundationpose
PYTHON=$PROJECT/.venvs/orcjax_gpu/bin/python
METADATA_PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
CODE_DIR=${CODE_DIR:-$PROJECT/runtime/worktrees/global-throughput-gpu-lifecycle-v1}
PREPARED_INPUT=${PREPARED_INPUT:-$PROJECT/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl}
FULL_REFERENCE_DIR=${FULL_REFERENCE_DIR:-$PROJECT/runtime/outputs/research/global_throughput/gpu-round4/model}
MEMORY_BASELINE_DIR=${MEMORY_BASELINE_DIR:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first}
B1024_EVIDENCE_DIR=${B1024_EVIDENCE_DIR:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1/capacity}
CONFIG=${CONFIG:-$CODE_DIR/configs/orchidee_man_250919.yaml}
ASSET_ROOT=${ASSET_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2}
RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-${SLURM_JOB_ID:-dry-run}}
EXPECTED_CODE_SHA=${EXPECTED_CODE_SHA:-}
DRY_RUN=0

if [[ ${1:-} == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
[[ $EXPECTED_CODE_SHA =~ ^[0-9a-fA-F]{40}$ ]] || { echo "EXPECTED_CODE_SHA must be a full 40-character Git SHA" >&2; exit 2; }
[[ -n ${XLA_PYTHON_CLIENT_ALLOCATOR:-} ]] && {
  echo "explicit XLA_PYTHON_CLIENT_ALLOCATOR overrides are forbidden" >&2; exit 2;
}
[[ -n ${SINGULARITYENV_XLA_PYTHON_CLIENT_ALLOCATOR:-} ]] && {
  echo "SINGULARITYENV_XLA_PYTHON_CLIENT_ALLOCATOR overrides are forbidden" >&2; exit 2;
}

CODE_DIR=$(realpath -m "$CODE_DIR")
RUN_ROOT=$(realpath -m "$RUN_ROOT")
case "$CODE_DIR" in
  "$PROJECT"/runtime/worktrees/*) ;;
  *) echo "CODE_DIR must stay under project runtime/worktrees" >&2; exit 2 ;;
esac
case "$RUN_ROOT" in
  "$PROJECT"/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-*) ;;
  *) echo "RUN_ROOT must be a unique gpu-lifecycle-* server-benchmark directory" >&2; exit 2 ;;
esac
ACTUAL_CODE_SHA=$(cd "$CODE_DIR" && git rev-parse HEAD)
[[ ${ACTUAL_CODE_SHA,,} == ${EXPECTED_CODE_SHA,,} ]] || {
  echo "CODE_DIR HEAD does not match EXPECTED_CODE_SHA" >&2; exit 2;
}
CODE_DIR_STATUS=$(cd "$CODE_DIR" && git status --porcelain --untracked-files=all)

ENTRY=$CODE_DIR/scripts/hpc/global_batch_gpu_entry.py
RUNNER=$CODE_DIR/scripts/dev/benchmark_global_gpu_lifecycle.py
COORDINATOR=$CODE_DIR/scripts/hpc/global_gpu_lifecycle_coordinator.py
IDENTITY_HELPER=$CODE_DIR/scripts/hpc/global_gpu_identity.py
export ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_lifecycle.py
export SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_lifecycle.py
if [[ $DRY_RUN -eq 0 ]]; then
  STEP=slurm_allocation_admission
  [[ -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-} == 1 && ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}} == 1 && ${SLURM_JOB_PARTITION:-} == gnall ]] || {
    echo "requires one active gnall Slurm task" >&2; exit 2;
  }
  [[ ${SLURM_CPUS_PER_TASK:-} == 4 && -n ${SLURM_JOB_GPUS:-} && -n ${CUDA_VISIBLE_DEVICES:-} ]] || {
    echo "requires exactly four Slurm CPUs and one allocated visible GPU" >&2; exit 2;
  }
  [[ ${SLURM_JOB_GPUS} != *,* && ${CUDA_VISIBLE_DEVICES} != *,* ]] || {
    echo "requires exactly one global and visible GPU" >&2; exit 2;
  }
  for path in "$PYTHON" "$METADATA_PYTHON" "$IDENTITY_HELPER" "$ENTRY" "$RUNNER" \
      "$COORDINATOR" "$PREPARED_INPUT" "$FULL_REFERENCE_DIR" "$MEMORY_BASELINE_DIR" \
      "$B1024_EVIDENCE_DIR" "$CONFIG" "$ASSET_ROOT" "$IMAGE"; do
    [[ -e $path ]] || { echo "required path missing: $path" >&2; exit 2; }
  done
  [[ -z $CODE_DIR_STATUS ]] || {
    echo "benchmark checkout must be clean and immutable" >&2; exit 2;
  }
  [[ ! -e $RUN_ROOT ]] || { echo "RUN_ROOT must be new: $RUN_ROOT" >&2; exit 2; }
fi

args=(--prepared-input "$PREPARED_INPUT" --full-reference-dir "$FULL_REFERENCE_DIR"
  --memory-baseline-dir "$MEMORY_BASELINE_DIR" --b1024-evidence-dir "$B1024_EVIDENCE_DIR"
  --config "$CONFIG" --asset-root "$ASSET_ROOT" --output-dir __ARM_OUTPUT_DIR__
  --warm-repeats 5 --budget-seconds 7200 --stage-shape 1024
  --measurement-strategy __MEASUREMENT_STRATEGY__ --expected-code-sha "$EXPECTED_CODE_SHA")
if [[ $DRY_RUN -eq 1 ]]; then
  child=("$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores
    /usr/bin/time -v singularity exec --nv --bind '<canonical-cuda-nvml-driver-binds>' --pwd "$CODE_DIR"
    "$IMAGE" "$PYTHON" "$ENTRY" "${args[@]}")
  printf 'proposal=GPU-B1024-phased-memory-lifecycle-new-science-v1 partition=gnall nodes=1 tasks=1 cpus=4 gpus=1 wall=02:00:00 no_requeue=true exclusive=false expected_code_sha=%s\n' "$EXPECTED_CODE_SHA"
  printf 'planned_environment=ORCHJAX_GLOBAL_BATCH_ENTRY=%q SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY=%q allocator=default preallocate=false x64=true\n' \
    "$ORCHJAX_GLOBAL_BATCH_ENTRY" "$SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY"
  printf 'code_dir=%q run_root=%q output_layout=%q/%q/capacity/B1024\n' \
    "$CODE_DIR" "$RUN_ROOT" "$RUN_ROOT" '<strategy>'
  printf 'cost_primary_cny=4.80 cost_conservative_cny=4.96 selected_node=requires_fresh_scheduler_review\n'
  printf 'command='
  printf '%q ' "$METADATA_PYTHON" "$COORDINATOR" --output-dir "$RUN_ROOT" \
    --budget-seconds 7200 --expected-code-sha "$EXPECTED_CODE_SHA" --full-reference-dir "$FULL_REFERENCE_DIR" \
    --code-dir "$CODE_DIR" --dry-run -- "${child[@]}"
  printf '\n'
  exit 0
fi

STEP=runtime_preparation
mkdir -p "$RUN_ROOT" "$PROJECT/runtime/logs"
exec > >(tee -a "$RUN_ROOT/launch-context.txt") 2>&1
export ORCHJAX_GPU_WINDOW_START_EPOCH=$(date +%s)
export PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export ORCHIDEE_REPO_ROOT=$CODE_DIR ORCHIDEE_DATA_ROOT=$PROJECT/runtime/data ORCHIDEE_OUTPUT_ROOT=$RUN_ROOT

date -u +%FT%TZ
hostname
uptime
taskset -pc $$
STEP=allocation_metadata
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
export ORCHJAX_SLURM_ALLOC_TRES=$ALLOC_TRES ORCHJAX_SLURM_ALLOC_TRES_SOURCE=$ALLOC_TRES_SOURCE
export ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_lifecycle.py
export ORCHJAX_PHYSICAL_GPU_UUID='' ORCHJAX_GPU_DRIVER_VERSION='' ORCHJAX_SLURM_GLOBAL_GPU_ID=$SLURM_JOB_GPUS
GPU_INFO=$("$METADATA_PYTHON" "$IDENTITY_HELPER" --slurm-global-id "$SLURM_JOB_GPUS" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" --output "$RUN_ROOT/gpu-identity.json")
IFS=$'\t' read -r ORCHJAX_PHYSICAL_GPU_UUID GPU_NAME ORCHJAX_GPU_DRIVER_VERSION <<< "$GPU_INFO"
[[ -n $ORCHJAX_PHYSICAL_GPU_UUID && -n $GPU_NAME && -n $ORCHJAX_GPU_DRIVER_VERSION ]] || {
  echo "cannot resolve allocated GPU identity" >&2; exit 2;
}
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
[[ -f $CUDA_DRIVER && -f $NVML_DRIVER ]] || { echo "canonical CUDA/NVML host drivers missing" >&2; exit 2; }
BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
child=("$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores
  /usr/bin/time -v singularity exec --nv --bind "$BINDS" --pwd "$CODE_DIR"
  "$IMAGE" "$PYTHON" "$ENTRY" "${args[@]}")
export SINGULARITYENV_ORCHJAX_GPU_WINDOW_START_EPOCH=$ORCHJAX_GPU_WINDOW_START_EPOCH
export SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export SINGULARITYENV_JAX_ENABLE_X64=true
export SINGULARITYENV_ORCHIDEE_REPO_ROOT=$CODE_DIR SINGULARITYENV_ORCHIDEE_DATA_ROOT=$ORCHIDEE_DATA_ROOT
export SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT=$RUN_ROOT
export SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_lifecycle.py
export SINGULARITYENV_ORCHJAX_PHYSICAL_GPU_UUID=$ORCHJAX_PHYSICAL_GPU_UUID
export SINGULARITYENV_ORCHJAX_GPU_DRIVER_VERSION=$ORCHJAX_GPU_DRIVER_VERSION
export SINGULARITYENV_ORCHJAX_SLURM_GLOBAL_GPU_ID=$ORCHJAX_SLURM_GLOBAL_GPU_ID
export SINGULARITYENV_SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK SINGULARITYENV_SLURM_JOB_ID=$SLURM_JOB_ID
export SINGULARITYENV_SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION SINGULARITYENV_SLURM_JOB_GPUS=$SLURM_JOB_GPUS
export SINGULARITYENV_ORCHJAX_SLURM_ALLOC_TRES=$ORCHJAX_SLURM_ALLOC_TRES
export SINGULARITYENV_ORCHJAX_SLURM_ALLOC_TRES_SOURCE=$ORCHJAX_SLURM_ALLOC_TRES_SOURCE
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
export SINGULARITYENV_OMP_NUM_THREADS=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1 SINGULARITYENV_NUMEXPR_NUM_THREADS=1

{
  echo "allocation_window_start_epoch=$ORCHJAX_GPU_WINDOW_START_EPOCH"
  echo "job=$SLURM_JOB_ID node=$(hostname) cpus=$SLURM_CPUS_PER_TASK gpu_global_id=$SLURM_JOB_GPUS uuid=$ORCHJAX_PHYSICAL_GPU_UUID name=$GPU_NAME driver=$ORCHJAX_GPU_DRIVER_VERSION visible=$CUDA_VISIBLE_DEVICES"
  echo "expected_code_sha=$EXPECTED_CODE_SHA actual_code_sha=$ACTUAL_CODE_SHA alloc_tres_source=$ALLOC_TRES_SOURCE alloc_tres=$ALLOC_TRES"
  "$SLURM_BIN/scontrol" show job --details "$SLURM_JOB_ID"
  lscpu -e=CPU,CORE,SOCKET,ONLINE
  nvidia-smi -i "$ORCHJAX_PHYSICAL_GPU_UUID" --query-gpu=uuid,name,driver_version,memory.total,memory.free,memory.used --format=csv
  nvidia-smi -i "$ORCHJAX_PHYSICAL_GPU_UUID" --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
} > "$RUN_ROOT/allocation-identity.txt" 2>&1
[[ $GPU_NAME == *V100* ]] || { echo "approved resource contract requires a V100 32GB GPU" >&2; exit 2; }
GPU_TOTAL_MIB=$(nvidia-smi -i "$ORCHJAX_PHYSICAL_GPU_UUID" --query-gpu=memory.total --format=csv,noheader,nounits)
[[ $GPU_TOTAL_MIB =~ ^[0-9]+$ && $GPU_TOTAL_MIB -ge 30720 && $GPU_TOTAL_MIB -le 34816 ]] || {
  echo "selected card total memory is outside the V100 32GB contract" >&2; exit 2;
}

STEP=two_arm_coordinator
cd "$CODE_DIR"
"$METADATA_PYTHON" -B "$COORDINATOR" --output-dir "$RUN_ROOT" --budget-seconds 7200 \
  --expected-code-sha "$EXPECTED_CODE_SHA" --full-reference-dir "$FULL_REFERENCE_DIR" --code-dir "$CODE_DIR" \
  --gpu-identity "$RUN_ROOT/gpu-identity.json" -- "${child[@]}"
