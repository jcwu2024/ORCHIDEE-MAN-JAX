#!/bin/bash
#SBATCH --job-name=shared-dense
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-dense-stages-%j.log
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-dense-stages-%j.log
#SBATCH --no-requeue
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_ROOT=$ROOT/runtime/checkouts/shared-dynamic-local-v2
TOOLS=$ROOT/runtime/tools/shared-dense-stages-20260923
export PROBE_SCRIPT=$TOOLS/probe_shared_dense_stages.py
export DENSE_MODULE_FILE=$TOOLS/shared_decoder_dense.py
export RUN_TIMEOUT=12m
cd "$CODE_ROOT"
# Micro4 runs only after the micro1 numerical/resource gate passes.
for MICRO in 1 4; do
  bash "$TOOLS/run_shared_dense_stages.sh" "$@" --microbatch "$MICRO" \
    --width 768 --blocks 2 --fields 4 --queries 64 --dense-chunks 0 128 \
    --output "$ROOT/runtime/outputs/diagnostics/shared-dense-stages-${SLURM_JOB_ID}-micro${MICRO}.json"
done
