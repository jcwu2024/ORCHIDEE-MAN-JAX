#!/usr/bin/env bash
#SBATCH --job-name=T7-affine-paid
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ibc13b03n01
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-20261011-v2
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-gpu-20261011-v2/slurm-%j.stdout.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-gpu-20261011-v2/slurm-%j.stderr.txt
# Nonexclusive, bounded affine diagnostic; no training or paid submission here.
set -euo pipefail
CODE=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-20261011-v2
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
ID=T7-summary-affine-paid-gpu-20261011-v2
DIAG=$ROOT/outputs/diagnostics/$ID
INPUT=$ROOT/transfer/T7-summary-recovery-20261011-v1
SCOPE=$DIAG/scope.v1.json
if [[ ${1:-} != --dry-run ]]; then
  [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { printf '%s\n' 'Scheduler attempt state unknown; no automatic retry; operator must retain pre-submission receipt' >&2; exit 2; }
  # This is the first filesystem operation of a live launch, even when the
  # frozen checkout or prepared scope is missing. Never create the output dir.
  ATTEMPT=$DIAG/slurm-attempt.v1.json
  if ! (set -o noclobber; printf '{"scope_id":"%s","job_id":"%s","status":"scheduler_started","retry_allowed":false}\n' \
      "$ID" "$SLURM_JOB_ID" > "$ATTEMPT"); then
    printf '%s\n' 'Scheduler attempt consumed or state unknown; no automatic retry; operator must retain pre-submission receipt' >&2
    exit 2
  fi
fi
[[ $# -le 1 && ( $# == 0 || $1 == --dry-run ) ]] || exit 2
# Deployment must prepare these directories/scope; a job never invents approval.
[[ -d $DIAG && -d $CODE && -f $SCOPE && -f $CODE/scripts/hpc/run_shared_latent_T7_batch_numerics_paid.py ]] || exit 2
COMMAND=(bash "$CODE/scripts/hpc/run_shared_latent_T7_batch_numerics_paid.sh"
  --checkout-root "$CODE"
  --operands "$INPUT/summary-recovery.v1.operands.npz"
  --capture-receipt "$INPUT/summary-recovery.v1.capture.json"
  --capture-source "$CODE/scripts/hpc/capture_shared_latent_T7_summary_operands.py"
  --output "$DIAG/result.v1.json"
  --scope-json "$SCOPE"
  --ledger "$DIAG/$ID.ledger.json")
if [[ ${1:-} == --dry-run ]]; then
  printf '%q ' sbatch --partition gnall --nodes 1 --ntasks 1 --cpus-per-task 4 --gres gpu:1 --time 00:20:00 --no-requeue --nodelist ibc13b03n01 \
    --chdir "$CODE" \
    --output "$DIAG/slurm-%j.stdout.txt" --error "$DIAG/slurm-%j.stderr.txt" "$CODE/scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh"
  printf '\n'
  printf '%q ' "${COMMAND[@]}" --dry-run
  printf '\n'
  exit 0
fi
cd -- "$CODE"
exec "${COMMAND[@]}"
