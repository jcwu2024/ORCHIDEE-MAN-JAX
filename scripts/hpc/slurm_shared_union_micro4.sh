#!/bin/bash
# Prepared paid fallback; no submission before explicit envelope approval.
# Worst-case additive charge: (8 * CNY0.07 + 1 * CNY2.20) * 0.5h = CNY1.38.
#SBATCH -J orcjax-union-m4
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-union-micro4-paid-%j.log
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-union-micro4-paid-%j.log
#SBATCH --no-requeue

set -euo pipefail
OVERLAY=${OVERLAY:-/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-union-micro4-20260929}
if [[ ${1:-} == --dry-run ]]; then
  shift
  printf '/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores bash %q --slurm\n' "$OVERLAY/run_shared_union_micro4.sh"
  bash "$OVERLAY/run_shared_union_micro4.sh" --slurm --dry-run "$@"
  exit 0
fi
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 ]] || exit 2
/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
  bash "$OVERLAY/run_shared_union_micro4.sh" --slurm "$@"
