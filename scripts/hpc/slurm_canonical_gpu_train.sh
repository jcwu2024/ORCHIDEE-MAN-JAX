#!/bin/bash
#SBATCH -J orcjax-train
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o runtime/logs/canonical_train_%j.txt
#SBATCH -e runtime/logs/canonical_train_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose
DATASET=${DATASET:?set DATASET to the accepted dataset_manifest.json}
STATISTICS=${STATISTICS:?set STATISTICS to the accepted training statistics JSON}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to an isolated experiment directory}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-256}
LEARNING_RATE=${LEARNING_RATE:-0.001}
SEED=${SEED:-20260723}

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" runtime/logs runtime/cache/jax/orcjax_gpu

env \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=$ROOT/runtime/cache/jax/orcjax_gpu \
  singularity exec \
    --nv \
    --bind /WORK:/WORK \
    --pwd "$ROOT" \
    "$IMAGE" \
    "$PYTHON" scripts/hpc/verify_orcjax_gpu.py

env \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=$ROOT/runtime/cache/jax/orcjax_gpu \
  singularity exec \
    --nv \
    --bind /WORK:/WORK \
    --pwd "$ROOT" \
    "$IMAGE" \
    "$PYTHON" -m research.daily_coarse_graining.canonical_training_run \
      train \
      --dataset "$DATASET" \
      --statistics "$STATISTICS" \
      --output-dir "$OUTPUT_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --seed "$SEED"
