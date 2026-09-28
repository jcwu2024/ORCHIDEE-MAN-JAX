#!/bin/bash
# Prepared paid fallback; submission requires the existing explicit approval.
#SBATCH -J orcjax-union
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-union-integrated-paid-%j.log
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-union-integrated-paid-%j.log
#SBATCH --no-requeue

set -euo pipefail
OVERLAY=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-union-integrated-20260927
if [[ ${1:-} == --dry-run ]]; then
  printf 'srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores bash %q --slurm\n' "$OVERLAY/run_shared_union_integrated.sh"
  bash "$OVERLAY/run_shared_union_integrated.sh" --slurm --dry-run
  exit 0
fi
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} == gnall ]] || exit 2
/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
  bash "$OVERLAY/run_shared_union_integrated.sh" --slurm
