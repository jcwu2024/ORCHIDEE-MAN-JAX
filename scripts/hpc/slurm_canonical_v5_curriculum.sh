#!/bin/bash
# One-step v5 initialization followed by bounded 1/3/7-day retained-tail training.
#SBATCH -J orcjax-v5-curr
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_v5_curriculum_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_v5_curriculum_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved code snapshot}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose
DATASET=${DATASET:?set DATASET to the accepted v5 dataset manifest}
STATISTICS=${STATISTICS:?set STATISTICS to the matching v5 statistics JSON}
ACCEPTANCE=${ACCEPTANCE:?set ACCEPTANCE to the matching v5 acceptance report}
PLAN=${PLAN:?set PLAN to the hash-bound Teacher generation plan}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new experiment directory}
ONE_STEP_EPOCHS=${ONE_STEP_EPOCHS:-10}
ONE_STEP_BATCH_SIZE=${ONE_STEP_BATCH_SIZE:-256}
ONE_STEP_LEARNING_RATE=${ONE_STEP_LEARNING_RATE:-0.001}
CURRICULUM=${CURRICULUM:-1:64,3:64,7:64}
MULTISTEP_BATCH_SIZE=${MULTISTEP_BATCH_SIZE:-4}
MULTISTEP_LEARNING_RATE=${MULTISTEP_LEARNING_RATE:-0.0001}
SEED=${SEED:-20260724}

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
for path in "$DATASET" "$STATISTICS" "$ACCEPTANCE" "$PLAN"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "input assets must stay under $ROOT/runtime" >&2; exit 2 ;;
  esac
  test -f "$path"
done
case "$OUTPUT_ROOT" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_ROOT must stay under $ROOT/runtime" >&2; exit 2 ;;
esac

test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$ROOT/runtime/logs" "$ROOT/runtime/cache/jax/orcjax_gpu"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
GPU_ENV=(
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
  SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=$ROOT/runtime/cache/jax/orcjax_gpu
  SINGULARITYENV_ORCHIDEE_REPO_ROOT=$ROOT
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
  SINGULARITYENV_ORCHIDEE_DATA_ROOT=$ROOT/runtime/data
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs
)

env "${GPU_ENV[@]}" singularity exec \
  --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
  "$PYTHON" -m scripts.hpc.verify_orcjax_gpu

env "${GPU_ENV[@]}" singularity exec \
  --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
  "$PYTHON" -m scripts.hpc.run_canonical_gpu_train \
    train \
    --dataset "$DATASET" \
    --statistics "$STATISTICS" \
    --acceptance "$ACCEPTANCE" \
    --output-dir "$OUTPUT_ROOT/one_step" \
    --epochs "$ONE_STEP_EPOCHS" \
    --batch-size "$ONE_STEP_BATCH_SIZE" \
    --learning-rate "$ONE_STEP_LEARNING_RATE" \
    --seed "$SEED"

env "${GPU_ENV[@]}" singularity exec \
  --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
  "$PYTHON" -m scripts.hpc.run_canonical_multistep_gpu_train \
    --dataset "$DATASET" \
    --statistics "$STATISTICS" \
    --acceptance "$ACCEPTANCE" \
    --plan "$PLAN" \
    --output-dir "$OUTPUT_ROOT/multistep" \
    --initialize-checkpoint "$OUTPUT_ROOT/one_step/best_checkpoint.pkl" \
    --curriculum "$CURRICULUM" \
    --batch-size "$MULTISTEP_BATCH_SIZE" \
    --learning-rate "$MULTISTEP_LEARNING_RATE" \
    --seed "$SEED"

test -f "$OUTPUT_ROOT/one_step/training_report.json"
test -f "$OUTPUT_ROOT/multistep/training_report.json"
test -f "$OUTPUT_ROOT/multistep/best_checkpoint.pkl"
