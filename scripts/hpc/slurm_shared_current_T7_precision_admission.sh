#!/usr/bin/env bash
#SBATCH --job-name=T7-current-precision
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-20261014-v2
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-admission-20261014-v2/slurm-%j.stdout.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-admission-20261014-v2/slurm-%j.stderr.txt
# Preparation only. This script neither submits a job nor creates an approval.
set -euo pipefail
CODE=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-20261014-v2
RUNTIME=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
OLD=$RUNTIME/outputs/diagnostics/shared-latent-reader-20261004
OUT=$RUNTIME/outputs/diagnostics/T7-current-model-precision-admission-20261014-v2
COMMAND=(bash "$CODE/scripts/hpc/run_shared_current_T7_precision_admission.sh"
  --stage execute
  --parameters "$RUNTIME/outputs/diagnostics/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl"
  --prepared-context "$RUNTIME/outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl"
  --training-prepared-manifest "$RUNTIME/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json"
  --dataset-manifest "$RUNTIME/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json"
  --saved-identity-receipt "$OLD/T7-raw-neural-jvp-20261010.result.b1.optimized-hlo.txt.gz.identity.json"
  --state-input-metadata "$OLD/T7-state-inputs-20261009.json"
  --turnover-overlay "$RUNTIME/transfer/shared-latent-reader-20261004/turnover_leaf_age_fall.guarded-source.py"
  --fp32-reference "$OLD/T7-B1-state-jvp-completion-20261009.independent-analysis.json"
  --source-cache "$CODE/outputs/T7-current-model-precision-admission-20261014-v2/source-cache"
  --approval-report "$OUT/approval.v2.json"
  --runtime-preflight "$OUT/runtime-preflight.v2.json"
  --output "$OUT/result.v2.json")
[[ $# -le 1 && ( $# == 0 || $1 == --dry-run ) ]] || exit 2
if [[ ${1:-} == --dry-run ]]; then
  printf '%q ' sbatch --partition gnall --nodes 1 --ntasks 1 --cpus-per-task 4 --gres gpu:1 \
    --time 01:30:00 --no-requeue --chdir "$CODE" \
    --output "$OUT/slurm-%j.stdout.txt" --error "$OUT/slurm-%j.stderr.txt" \
    "$CODE/scripts/hpc/slurm_shared_current_T7_precision_admission.sh"
  printf '\n'
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_CPUS_PER_TASK:-} == 4 ]] || exit 2
[[ -d $CODE && -d $OUT && -f $OUT/approval.v2.json ]] || exit 2
cd -- "$CODE"
exec "${COMMAND[@]}"
