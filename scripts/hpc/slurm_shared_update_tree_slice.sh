#!/bin/bash
# Prepared 20-minute nonexclusive paid envelope; do not submit without explicit approval.
# Worst-case additive estimate: (8 * CNY0.07 + 1 * CNY2.20) * 1/3h = CNY0.92.
# The optional daily64 workload is excluded from this admission.
#SBATCH -J orcjax-update-slice
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --nodelist=ibc13b03n04
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-update-tree-slice-%j.log
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-update-tree-slice-%j.log
#SBATCH --no-requeue

set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-update-tree-slice-20260929"}
if [[ ${1:-} == --dry-run ]]; then
  shift
  printf '/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores bash %q --slurm\n' "$OVERLAY/run_shared_update_tree_slice.sh"
  bash "$OVERLAY/run_shared_update_tree_slice.sh" --slurm --dry-run "$@"
  exit 0
fi
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 && -n ${CUDA_VISIBLE_DEVICES:-} ]] || exit 2
/rmprog/slurm/v22.05.7/bin/srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
  bash "$OVERLAY/run_shared_update_tree_slice.sh" --slurm "$@"
