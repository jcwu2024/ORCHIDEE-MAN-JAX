#!/bin/bash
# Proposed maximum CPU charge: 4 nodes * 8 CPUs * 2 hours * 0.07 = CNY4.48.
#SBATCH --job-name=daily_data_expand
#SBATCH --partition=cnall
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/one-day-expansion-v1
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/one-day-expansion-%j.log
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/one-day-expansion-%j.log
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT="$ROOT/runtime/checkouts/one-day-expansion-v1"
RESULT_ROOT="$ROOT/runtime/outputs/training/one-day-expansion-${SLURM_JOB_ID:-dry-run}"
export OUTPUT_DIR="$RESULT_ROOT/chunks/worker-WORKER_ID"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
cd "$CODE_ROOT"
PREPARE=(/rmprog/slurm/v22.05.7/bin/srun --ntasks=32 --ntasks-per-node=8 --cpus-per-task=1
  --cpu-bind=cores --kill-on-bad-exit=1 --output="$RESULT_ROOT/worker-%t.log"
  bash "$CODE_ROOT/scripts/hpc/run_expanded_one_day_worker.sh")
MERGE=(timeout --signal=TERM 8m "$ROOT/.venvs/orcjax_cpu/bin/python"
  -m scripts.hpc.merge_expanded_one_day
  --plan "$ROOT/runtime/outputs/diagnostics/one-day-expansion-plan-v1.json"
  --output-dir "$RESULT_ROOT")
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' "${PREPARE[@]}"
  printf '\n'
  printf '%q ' "${MERGE[@]}"
  printf '\n'
else
  mkdir -p "$RESULT_ROOT"
  "${PREPARE[@]}"
  "${MERGE[@]}"
fi
