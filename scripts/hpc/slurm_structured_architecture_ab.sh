#!/bin/bash
# Matched continuation A/B for flat and process-conditioned daily operators.
#SBATCH -J orcjax-arch-ab
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/structured_architecture_ab_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/structured_architecture_ab_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved snapshot}
DATASET=${DATASET:?set DATASET to the accepted v5 dataset manifest}
STATISTICS=${STATISTICS:?set STATISTICS to the matching statistics JSON}
ACCEPTANCE=${ACCEPTANCE:?set ACCEPTANCE to the matching acceptance report}
PLAN=${PLAN:?set PLAN to the hash-bound Teacher generation plan}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:?set BASELINE_CHECKPOINT to the accepted flat checkpoint}
MATRIX=${MATRIX:?set MATRIX to the frozen four-way rollout matrix}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new architecture A/B directory}
CURRICULUM=${CURRICULUM:-1:64,3:64,7:64}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
SEED=${SEED:-20260724}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
for path in "$DATASET" "$STATISTICS" "$ACCEPTANCE" "$PLAN" "$BASELINE_CHECKPOINT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime inputs must stay under $ROOT/runtime" >&2; exit 2 ;;
  esac
  test -f "$path"
done
case "$MATRIX" in
  "$WORKTREE/manifests/"*) ;;
  *) echo "MATRIX must stay under the pinned worktree manifests" >&2; exit 2 ;;
esac
case "$OUTPUT_ROOT" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_ROOT must stay under $ROOT/runtime" >&2; exit 2 ;;
esac

test -f "$MATRIX"
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

run_gpu() {
  env "${GPU_ENV[@]}" singularity exec \
    --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" "$@"
}

run_gpu -m scripts.hpc.verify_orcjax_gpu

for arm in flat structured; do
  if [[ "$arm" = flat ]]; then
    architecture=canonical_flat_v1
  else
    architecture=structured_process_film_v1
  fi
  mkdir -p "$OUTPUT_ROOT/$arm"
  run_gpu -m scripts.hpc.run_canonical_multistep_gpu_train \
    --dataset "$DATASET" \
    --statistics "$STATISTICS" \
    --acceptance "$ACCEPTANCE" \
    --plan "$PLAN" \
    --output-dir "$OUTPUT_ROOT/$arm/training" \
    --initialize-checkpoint "$BASELINE_CHECKPOINT" \
    --curriculum "$CURRICULUM" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --objective canonical_multistep_v1 \
    --model-architecture "$architecture" \
    --seed "$SEED"

  checkpoint="$OUTPUT_ROOT/$arm/training/best_checkpoint.pkl"
  test -f "$checkpoint"
  run_gpu -m scripts.hpc.run_canonical_gpu_rollout_matrix \
    --matrix "$MATRIX" \
    --dataset "$DATASET" \
    --statistics "$STATISTICS" \
    --acceptance "$ACCEPTANCE" \
    --plan "$PLAN" \
    --checkpoint "$checkpoint" \
    --output-dir "$OUTPUT_ROOT/$arm/four_way"

  run_gpu -m scripts.hpc.run_spatial_conditioning_gpu \
    --dataset "$DATASET" \
    --statistics "$STATISTICS" \
    --acceptance "$ACCEPTANCE" \
    --checkpoint "$checkpoint" \
    --output "$OUTPUT_ROOT/$arm/spatial_conditioning.json"
done

test -f "$OUTPUT_ROOT/flat/training/training_report.json"
test -f "$OUTPUT_ROOT/structured/training/training_report.json"
test -f "$OUTPUT_ROOT/flat/four_way/matrix_report.json"
test -f "$OUTPUT_ROOT/structured/four_way/matrix_report.json"
test -f "$OUTPUT_ROOT/flat/spatial_conditioning.json"
test -f "$OUTPUT_ROOT/structured/spatial_conditioning.json"
