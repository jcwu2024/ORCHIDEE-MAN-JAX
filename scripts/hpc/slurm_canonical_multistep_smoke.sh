#!/bin/bash
# Real one-window GPU gate before the bounded v5 curriculum experiment.
#SBATCH -J orcjax-ms-smoke
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_multistep_smoke_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_multistep_smoke_%j.txt
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
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new smoke directory}
INITIALIZE_CHECKPOINT=${INITIALIZE_CHECKPOINT:-}
MULTISTEP_OBJECTIVE=${MULTISTEP_OBJECTIVE:-canonical_multistep_v1}
MODEL_ARCHITECTURE=${MODEL_ARCHITECTURE:-canonical_flat_v1}
STATE_INCREMENT_LOSS_WEIGHT=${STATE_INCREMENT_LOSS_WEIGHT:-1.0}
STATE_DELTA_FLOOR_RATIO=${STATE_DELTA_FLOOR_RATIO:-0.001}

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
if [[ -n "$INITIALIZE_CHECKPOINT" ]]; then
  case "$INITIALIZE_CHECKPOINT" in
    "$ROOT/runtime/"*) ;;
    *) echo "INITIALIZE_CHECKPOINT must stay under $ROOT/runtime" >&2; exit 2 ;;
  esac
  test -f "$INITIALIZE_CHECKPOINT"
fi
case "$OUTPUT_DIR" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_DIR must stay under $ROOT/runtime" >&2; exit 2 ;;
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
TRAIN_ARGS=(
  --dataset "$DATASET"
  --statistics "$STATISTICS"
  --acceptance "$ACCEPTANCE"
  --plan "$PLAN"
  --output-dir "$OUTPUT_DIR"
  --curriculum 1:1
  --batch-size 1
  --learning-rate 0.0001
  --objective "$MULTISTEP_OBJECTIVE"
  --model-architecture "$MODEL_ARCHITECTURE"
  --state-increment-loss-weight "$STATE_INCREMENT_LOSS_WEIGHT"
  --state-delta-floor-ratio "$STATE_DELTA_FLOOR_RATIO"
  --seed 20260724
)
if [[ -n "$INITIALIZE_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--initialize-checkpoint "$INITIALIZE_CHECKPOINT")
fi

env \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=$ROOT/runtime/cache/jax/orcjax_gpu \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT=$ROOT \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT=$ROOT/runtime/data \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs \
  singularity exec \
    --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.run_canonical_multistep_gpu_train \
      "${TRAIN_ARGS[@]}"

test -f "$OUTPUT_DIR/training_report.json"
test -f "$OUTPUT_DIR/best_checkpoint.pkl"
