#!/bin/bash
# Prepared only. Explicit paid approval receipt and selected node are required.
#SBATCH -J orcjax-T7-paid
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-%j.out
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-%j.err
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
ROUTE_ARGS=()
if [[ ${T7_PAID_REPLACEMENT_OF:-} == 15580715 ]]; then
  PREFIX=latent-reader-T7-20261005-paid-replacement
  APPROVAL="$OVERLAY/$PREFIX.approval.json"
  ROUTE_ARGS=(--replacement-of-job 15580715 --prefix "$PREFIX"
    --ledger "$ROOT/runtime/outputs/diagnostics/$PREFIX.gpu-budget.json"
    --aggregate-budget "$ROOT/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid.aggregate-budget.json")
elif [[ -n ${T7_PAID_REPLACEMENT_OF:-} ]]; then
  exit 2
else
  APPROVAL=${T7_PAID_APPROVAL:-"$OVERLAY/latent-reader-T7-20261005-paid.approval.json"}
fi
PREFLIGHT=${T7_RUNTIME_PREFLIGHT:-"$ROOT/runtime/outputs/diagnostics/latent-reader-T7-20261005-runtime-preflight.json"}
HOST_PYTHON=/apps/soft/anaconda3/bin/python
if [[ ${1:-} == --dry-run ]]; then
  exec "$HOST_PYTHON" "$OVERLAY/shared_latent_T7_paid_budget.py" --approval "$APPROVAL" --runtime-preflight "$PREFLIGHT" "${ROUTE_ARGS[@]}" --dry-run
fi
[[ $# == 0 && -n ${SLURM_JOB_ID:-} ]] || exit 2
[[ ${SLURM_JOB_PARTITION:-} == gnall && ${SLURM_CPUS_PER_TASK:-} == 8 && ${SLURM_NTASKS:-} == 1 ]] || exit 2
[[ -f "$APPROVAL" ]] || exit 2
exec "$HOST_PYTHON" "$OVERLAY/shared_latent_T7_paid_budget.py" --approval "$APPROVAL" --runtime-preflight "$PREFLIGHT" "${ROUTE_ARGS[@]}"
