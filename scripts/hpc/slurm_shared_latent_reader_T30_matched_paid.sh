#!/bin/bash
# Prepared only; an exact fresh paid approval is required by the helper.
#SBATCH -J orcjax-T30-matched-paid
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T30-20261006-paid-matched-%j.out
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T30-20261006-paid-matched-%j.err
set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
HOST_PYTHON=/apps/soft/anaconda3/bin/python
HELPER="$OVERLAY/shared_latent_T30_matched_paid_budget.py"
APPROVAL=${T30_MATCHED_PAID_APPROVAL:-"$OVERLAY/latent-reader-T30-20261006-paid-matched-approval.json"}

if [[ ${1:-} == --dry-run ]]; then
  exec "$HOST_PYTHON" "$HELPER" --approval "$APPROVAL" --dry-run
fi
if [[ ${1:-} == --validate-assets ]]; then
  exec "$HOST_PYTHON" "$HELPER" --validate-assets
fi
[[ $# == 0 && -n ${SLURM_JOB_ID:-} ]] || exit 2
[[ ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 && ${SLURM_NTASKS:-} == 1 ]] || exit 2
[[ ${SLURM_GPUS_ON_NODE:-} == 1 && ${SLURM_MEM_PER_NODE:-} == 65536 ]] || exit 2
[[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}} == 1 ]] || exit 2
[[ -f "$APPROVAL" ]] || exit 2
exec "$HOST_PYTHON" "$HELPER" --approval "$APPROVAL"
