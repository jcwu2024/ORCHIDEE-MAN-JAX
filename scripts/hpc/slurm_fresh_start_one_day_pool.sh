#!/bin/bash
# Proposed one-epoch run. Submit only after the concrete paid budget is approved.
#SBATCH -J daily_pool_v1
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/one-day-pool-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/one-day-pool-%j.log
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/one-day-pool-%j.log
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT="$ROOT/runtime/checkouts/one-day-pool-v1"
export GPU_DEVICE=0 PYTHON_MODULE=scripts.hpc.run_fresh_start_one_day_pool RUN_TIMEOUT=235m
export OUTPUT="$ROOT/runtime/outputs/training/fresh-one-day-pool-${SLURM_JOB_ID}/report.json"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "$CODE_ROOT"
bash scripts/hpc/run_fresh_start_one_day.sh \
  --catalogue-report "$ROOT/runtime/outputs/diagnostics/one-day-catalogue-v1.json" "$@"
