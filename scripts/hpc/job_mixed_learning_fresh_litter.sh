#!/bin/bash
#SBATCH -J fresh-litter
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/fresh-litter-v1/slurm-%j.log
#SBATCH --no-requeue
# Proposal only: explicit approval required; (8*.07 + 2*2.20)*5 = CNY24.80.
# Pre-create Slurm output directory before submission.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT=$ROOT/runtime/checkouts/fresh-litter-v1
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:?Slurm must declare allocated GPUs}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUN_TIMEOUT=295m
export OUTPUT=$ROOT/runtime/outputs/training/fresh-litter-v1/report.json
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_fresh_litter.sh" train \
  --run-until-update 1000 "$@"
