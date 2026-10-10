#!/bin/bash
# Prepared only. The receipt and a separate paid ledger are mandatory.
#SBATCH -J orcjax-T7-B4-paid
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@360
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-batch-20261009-paid-%j.out
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-batch-20261009-paid-%j.err
set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
HOST_PYTHON=/apps/soft/anaconda3/bin/python
APPROVAL=${T7_B4_PAID_APPROVAL:-"$OVERLAY/latent-reader-T7-batch-20261009-paid.approval.json"}

if [[ ${1:-} == --dry-run ]]; then
  exec "$HOST_PYTHON" -B "$OVERLAY/run_shared_latent_T7_batch_paid.py" --approval "$APPROVAL" --dry-run
fi

[[ $# == 0 && -n ${SLURM_JOB_ID:-} ]] || exit 2
[[ ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 && ${SLURM_NTASKS:-} == 1 ]] || exit 2
[[ ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}} == 1 && ${SLURM_MEM_PER_NODE:-} == 65536 ]] || exit 2
[[ -f "$APPROVAL" ]] || exit 2
exec "$HOST_PYTHON" -B "$OVERLAY/run_shared_latent_T7_batch_paid.py" --approval "$APPROVAL" --allow-new-compile
