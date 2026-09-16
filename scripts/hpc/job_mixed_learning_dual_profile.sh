#!/bin/bash
#SBATCH -J mixed-dual-profile
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-dual-paid-%j.log
#SBATCH --no-requeue
# Max additive charge: 8*1.5*.07 + 2*1.5*2.20 = CNY7.44.
# Full same-update parity/timing, NOT a new scientific learning run.
set -euo pipefail
export CODE_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-dual-paid-v1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUN_TIMEOUT=85m
export OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-dual-paid-${SLURM_JOB_ID:-dry-run}.json
export SINGULARITYENV_JAX_DEBUG_LOG_MODULES=jax._src.compiler
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_dual_device.sh" profile "$@"
