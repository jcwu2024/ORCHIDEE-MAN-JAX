#!/bin/bash
#SBATCH --job-name=shared-micro
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-staged-%j.log
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-staged-%j.log
#SBATCH --no-requeue

set -euo pipefail
# Select an idle node with sbatch --nodelist after approval and occupancy check.
export CODE_ROOT=${CODE_ROOT:?set approved immutable checkout}
export MODE=gpu RUN_TIMEOUT=110m
export STAGE=${STAGE:?set micro2, micro4, micro8 or micro16 for this independent job}
case "$STAGE" in micro2|micro4|micro8|micro16) ;; *) exit 2 ;; esac
export OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-staged-${SLURM_JOB_ID}.json
exec bash "$CODE_ROOT/scripts/hpc/run_shared_staged_execution.sh" "$@"
