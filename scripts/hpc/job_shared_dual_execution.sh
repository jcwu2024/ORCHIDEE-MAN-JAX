#!/bin/bash
#SBATCH --job-name=shared-dual
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-dual-%j.log
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-dual-%j.log
#SBATCH --no-requeue

set -euo pipefail
# Slurm copies this script: never infer the checkout from its spooled filename.
export CODE_ROOT=${CODE_ROOT:?export the approved immutable checkout}
export MODE=gpu
export OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-dual-${SLURM_JOB_ID}.json
exec bash "$CODE_ROOT/scripts/hpc/run_shared_dual_execution.sh" "$@"
