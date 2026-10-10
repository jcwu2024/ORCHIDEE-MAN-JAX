#!/bin/bash
# Approval covers this first stage only, not automatic micro4/training continuation.
# Repaired retry proposal; new approval required after the failed first attempt.
# Worst-case additive charge: (8 * CNY0.07 + CNY2.20) * (25/60)h = CNY1.15.
#SBATCH -J orcjax-union-f16
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-union-neural-precision-paid-%j.log
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-union-neural-precision-paid-%j.log
#SBATCH --no-requeue

set -euo pipefail
OVERLAY=${OVERLAY:-/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-union-neural-precision-20260929-v2}
for ARGUMENT in "$@"; do
  [[ "$ARGUMENT" != --micro4 ]] || { echo 'This allocation is first-stage only.' >&2; exit 2; }
done
if [[ ${1:-} == --dry-run ]]; then
  shift
  printf '/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores bash %q --slurm\n' "$OVERLAY/run_shared_union_neural_precision.sh"
  bash "$OVERLAY/run_shared_union_neural_precision.sh" --slurm --dry-run "$@"
  exit 0
fi
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 ]] || exit 2
/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
  bash "$OVERLAY/run_shared_union_neural_precision.sh" --slurm "$@"
