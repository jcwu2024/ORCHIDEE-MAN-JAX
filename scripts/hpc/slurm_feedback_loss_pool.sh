#!/bin/bash
# Proposed CNY16.56 maximum: 1GPU + 8CPU for6h. Submit after explicit approval.
#SBATCH -J daily_feedback_v1
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/feedback-pool-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/feedback-pool-%j.log
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/feedback-pool-%j.log
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT="$ROOT/runtime/checkouts/feedback-pool-v1"
export GPU_DEVICE=0 PYTHON_MODULE=scripts.hpc.run_fresh_start_one_day_pool RUN_TIMEOUT=350m
export OUTPUT="$ROOT/runtime/outputs/training/fresh-feedback-pool-${SLURM_JOB_ID:-dry-run}/report.json"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$CODE_ROOT"
COMMAND=(bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh"
  --pool-protocol "$CODE_ROOT/manifests/coarse_graining/fresh_start_feedback_pool_v1.json"
  --catalogue-report "$ROOT/runtime/outputs/diagnostics/one-day-catalogue-v1.json")
if [[ "${1:-}" == --dry-run ]]; then
  printf 'PYTHON_MODULE=%s RUN_TIMEOUT=%s OUTPUT=%s\n' "$PYTHON_MODULE" "$RUN_TIMEOUT" "$OUTPUT"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
else
  "${COMMAND[@]}" "$@"
fi
