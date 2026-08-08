#!/bin/bash
# Submit only after reviewing the exact array indices, resources, wall time,
# paths, command, and worst-case cost. Start with --array=0.
#SBATCH --job-name=orcjax-process-v1
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/first_process_v1_%A_%a.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/first_process_v1_%A_%a.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$ROOT/runtime
WORKTREE=${WORKTREE:-$ROOT}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
DATASET=$RUNTIME_ROOT/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$RUNTIME_ROOT/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$RUNTIME_ROOT/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$RUNTIME_ROOT/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
PROTOCOL=$WORKTREE/manifests/coarse_graining/first_neural_process_training_v1.json
: "${FIRST_PROCESS_EXPECTED_GIT_HEAD:?set FIRST_PROCESS_EXPECTED_GIT_HEAD to the approved commit}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as an array with an approved index range}"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose an allocated GPU}"

case "$WORKTREE" in
  "$ROOT"|"$RUNTIME_ROOT/worktrees/"*) ;;
  *) echo "WORKTREE must be the repository root or stay under $RUNTIME_ROOT/worktrees" >&2; exit 2 ;;
esac
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 5 )); then
  echo "first-process array index must be 0 through 5" >&2
  exit 2
fi

ARMS=(
  axial_process_operator_v1
  factorized_process_control_v1
  axial_process_operator_v1
  factorized_process_control_v1
  axial_process_operator_v1
  factorized_process_control_v1
)
SEEDS=(20260807 20260807 20260808 20260808 20260809 20260809)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" "$PROCESS_STATISTICS" "$PROTOCOL"; do
  test -f "$path"
done
test -x "$PYTHON"
test -x "$SLURM_BIN/srun"
cd "$WORKTREE"
test -z "$(git status --porcelain --untracked-files=all)"
ACTUAL_GIT_HEAD=$(git rev-parse HEAD)
if [[ "$ACTUAL_GIT_HEAD" != "$FIRST_PROCESS_EXPECTED_GIT_HEAD" ]]; then
  echo "first-process worktree HEAD mismatch: $ACTUAL_GIT_HEAD" >&2
  exit 2
fi

HEAD_SHORT=$(git rev-parse --short=7 HEAD)
OUTPUT_ROOT=$RUNTIME_ROOT/outputs/training/first-neural-process-head-v1-formal-$HEAD_SHORT
OUTPUT_DIR=$OUTPUT_ROOT/$ARM/seed-$SEED
CACHE_DIR=$RUNTIME_ROOT/cache/jax/first-neural-process-v1/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p "$RUNTIME_ROOT/logs" "$OUTPUT_ROOT" "$CACHE_DIR"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "git_head=$ACTUAL_GIT_HEAD job_id=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
echo "arm=$ARM seed=$SEED output_dir=$OUTPUT_DIR cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
sha256sum "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" "$PROCESS_STATISTICS" "$PROTOCOL"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

GPU_ENV=(
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=$CACHE_DIR
  SINGULARITYENV_ORCHIDEE_REPO_ROOT=$WORKTREE
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT=$RUNTIME_ROOT
  SINGULARITYENV_ORCHIDEE_DATA_ROOT=$RUNTIME_ROOT/data
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT=$RUNTIME_ROOT/assets
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT=$RUNTIME_ROOT/outputs
)
COMMAND=(
  env -u LD_LIBRARY_PATH "${GPU_ENV[@]}"
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE"
  "$PYTHON" -m scripts.hpc.run_first_neural_process_gpu_train
  --dataset-manifest "$DATASET"
  --acceptance-report "$ACCEPTANCE"
  --base-statistics "$BASE_STATISTICS"
  --process-statistics "$PROCESS_STATISTICS"
  --protocol "$PROTOCOL"
  --arm "$ARM"
  --seed "$SEED"
  --source-git-head "$ACTUAL_GIT_HEAD"
  --output-dir "$OUTPUT_DIR"
)

"$SLURM_BIN/srun" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v "${COMMAND[@]}"

test -f "$OUTPUT_DIR/checkpoint.pkl"
test -f "$OUTPUT_DIR/best_checkpoint.pkl"
test -f "$OUTPUT_DIR/training_report.json"
sha256sum "$OUTPUT_DIR/checkpoint.pkl" "$OUTPUT_DIR/best_checkpoint.pkl" "$OUTPUT_DIR/training_report.json"
