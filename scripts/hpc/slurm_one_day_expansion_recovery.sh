#!/bin/bash
# Proposed recovery cap: 2 nodes * 4 CPUs * 1 hour * 0.07 = CNY0.56.
#SBATCH --job-name=daily_data_recover
#SBATCH --partition=cnall
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/one-day-expansion-recovery-v1
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/one-day-recovery-%j.log
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/one-day-recovery-%j.log
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT="$ROOT/runtime/checkouts/one-day-expansion-recovery-v1"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime"
cd "$CODE_ROOT"
PREPARE=(/rmprog/slurm/v22.05.7/bin/srun --ntasks=8 --ntasks-per-node=4 --cpus-per-task=1
  --cpu-bind=cores --wait=0 --kill-on-bad-exit=1
  timeout --signal=TERM 50m "$ROOT/.venvs/orcjax_cpu/bin/python"
  -m scripts.hpc.run_expansion_recovery_worker
  --plan "$ROOT/runtime/outputs/diagnostics/one-day-expansion-recovery-v1.json")
MERGE=(timeout --signal=TERM 8m "$ROOT/.venvs/orcjax_cpu/bin/python"
  -m scripts.hpc.merge_expanded_one_day
  --plan "$ROOT/runtime/outputs/diagnostics/one-day-expansion-plan-v1.json"
  --output-dir "$ROOT/runtime/outputs/training/one-day-expansion-15061225")
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' "${PREPARE[@]}"; printf '\n'
  printf '%q ' "${MERGE[@]}"; printf '\n'
else
  "${PREPARE[@]}"
  "${MERGE[@]}"
fi
