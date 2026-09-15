#!/bin/bash
#SBATCH -J mixed-stages-pilot
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/mixed-learning-stages-pilot-v1/slurm-%j.log
#SBATCH --no-requeue
# Prepared only. Freeze joint count after timing and obtain paid approval first.
# CPU8*8*.07=4.48 + GPU1*8*2.20=17.60 = CNY22.08 maximum.
set -euo pipefail
export CODE_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-parallel-routes-v1
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CODE_ROOT"
bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_stages.sh" joint "$@"
bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_stages.sh" staged "$@"
