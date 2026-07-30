#!/bin/bash
# Formal one-V100 Experiment B post-training screening.
#SBATCH -J orcjax-rollout-screen
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=07:30:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/rollout_stability_screening_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/rollout_stability_screening_%j.txt
#SBATCH --no-requeue

set -euo pipefail

WORKTREE=${WORKTREE:?set WORKTREE}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:?set EXPERIMENT_ROOT}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT}

export WORKTREE EXPECTED_GIT_HEAD EXPERIMENT_ROOT OUTPUT_ROOT
exec "$WORKTREE/scripts/hpc/run_rollout_stability_screening.sh"
