#!/bin/bash
#SBATCH -J mixed-learn200
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/mixed-learning-v1/slurm-%j.log
#SBATCH --no-requeue
# Proposed envelope: CPU 8*6*.07=3.36 + GPU 1*6*2.20=13.20 = CNY16.56.
# Submit only after approval of this concrete envelope.
set -euo pipefail
export CODE_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-v1
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning.sh" train "$@"
