#!/bin/bash
#SBATCH -J mixed-dual-integrate
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/mixed-learning-dual-integration-%j.log
#SBATCH --no-requeue
# Four bounded runner invocations, nine updates total; no scientific training.
# Additive ceiling: (8*.07 + 2*2.20)*2 = CNY9.92.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT=$ROOT/runtime/checkouts/mixed-learning-dual-integration-v1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES:?Slurm must declare allocated GPUs}
export GPU_MEMORY_FRACTION=0.85 RUN_TIMEOUT=115m
export OUTPUT=$ROOT/runtime/outputs/diagnostics/mixed-learning-dual-integration-${SLURM_JOB_ID:-dry-run}.json
export PYTHON_MODULE=scripts.hpc.run_dual_training_integration
ARGS=(--protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_model_v1.json"
  --mixed-protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_v1.json"
  --catalogue-report "$ROOT/runtime/outputs/training/one-day-expansion-15061225/merge_report.json"
  --recurrent-schedule "$ROOT/runtime/outputs/diagnostics/mixed-learning-200-recurrent-schedule-v1.json"
  --selection "$ROOT/runtime/outputs/diagnostics/trained-forward-selection-v1.json"
  --execution-policy "$CODE_ROOT/manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json")
if [[ "${1:-}" == --dry-run ]]; then
  printf '%q ' env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" OUTPUT="$OUTPUT" \
    PYTHON_MODULE="$PYTHON_MODULE" RUN_TIMEOUT="$RUN_TIMEOUT" GPU_MEMORY_FRACTION="$GPU_MEMORY_FRACTION" \
    bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" "${ARGS[@]}" --dry-run
  printf '\n'
  exit 0
fi
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" "${ARGS[@]}" "$@"
