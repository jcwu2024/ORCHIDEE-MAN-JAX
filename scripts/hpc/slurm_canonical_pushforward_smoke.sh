#!/bin/bash
# One detached-prefix matched-Teacher update before a bounded objective A/B.
#SBATCH -J orcjax-pf-smoke
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_pushforward_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_pushforward_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved snapshot}
DATASET=${DATASET:?set DATASET to the accepted v5 dataset manifest}
STATISTICS=${STATISTICS:?set STATISTICS to the matching statistics JSON}
ACCEPTANCE=${ACCEPTANCE:?set ACCEPTANCE to the matching acceptance report}
PLAN=${PLAN:?set PLAN to the hash-bound Teacher generation plan}
INITIALIZE_CHECKPOINT=${INITIALIZE_CHECKPOINT:?set the accepted initialization checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new smoke directory}
PREFIX_CURRICULUM=${PREFIX_CURRICULUM:-3:1}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
for path in "$DATASET" "$STATISTICS" "$ACCEPTANCE" "$PLAN" "$INITIALIZE_CHECKPOINT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime inputs must stay under $ROOT/runtime" >&2; exit 2 ;;
  esac
  test -f "$path"
done
case "$OUTPUT_DIR" in
  "$ROOT/runtime/outputs/"*) ;;
  *) echo "OUTPUT_DIR must stay under $ROOT/runtime/outputs" >&2; exit 2 ;;
esac

test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test ! -e "$OUTPUT_DIR"
mkdir -p "$ROOT/runtime/logs" "$ROOT/runtime/cache/jax/orcjax_gpu"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

env \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=$ROOT/runtime/cache/jax/orcjax_gpu \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT=$ROOT \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT=$ROOT/runtime/data \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs \
  singularity exec \
    --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.run_canonical_pushforward_gpu_train \
      --dataset "$DATASET" \
      --statistics "$STATISTICS" \
      --acceptance "$ACCEPTANCE" \
      --plan "$PLAN" \
      --initialize-checkpoint "$INITIALIZE_CHECKPOINT" \
      --output-dir "$OUTPUT_DIR" \
      --prefix-curriculum "$PREFIX_CURRICULUM" \
      --batch-size 1 \
      --learning-rate 0.0001 \
      --seed 20260725

test -f "$OUTPUT_DIR/training_report.json"
test -f "$OUTPUT_DIR/best_checkpoint.pkl"
