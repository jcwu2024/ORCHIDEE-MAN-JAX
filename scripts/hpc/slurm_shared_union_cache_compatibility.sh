#!/bin/bash
# Preparation only: every model stage must reuse its completed FP32 cache.
# New explicit approval required; no precision-screen or micro4 continuation.
# Worst-case additive charge: (8*CNY0.07 + 1*CNY2.20)*(10/60)h = CNY0.46.
#SBATCH -J orcjax-union-cache
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodelist=ibc13b03n04
#SBATCH --time=00:10:00
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-union-cache-compatibility-paid-%j.log
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-union-cache-compatibility-paid-%j.log
#SBATCH --no-requeue

set -euo pipefail
OVERLAY=${OVERLAY:-/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-union-neural-precision-20260929-v3}
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || {
  echo 'This fixed cache-only envelope accepts only --dry-run.' >&2; exit 2;
}
if [[ ${1:-} == --dry-run ]]; then
  printf '/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores bash %q --slurm --cache-compatibility-only\n' "$OVERLAY/run_shared_union_neural_precision.sh"
  bash "$OVERLAY/run_shared_union_neural_precision.sh" --slurm --cache-compatibility-only --dry-run
  exit 0
fi
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 ]] || exit 2
/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
  bash "$OVERLAY/run_shared_union_neural_precision.sh" --slurm --cache-compatibility-only
