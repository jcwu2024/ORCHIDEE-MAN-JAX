#!/bin/bash
# Persistent scale-selected coupled-GPP training.
# Ceiling: 8 CPU cores * 16 h * CNY 0.07 + 1 GPU * 16 h * CNY 2.20
#          = CNY 8.96 + CNY 35.20 = CNY 44.16 worst case.
# Submission requires explicit approval of this exact envelope.
# Set OUTPUT_DIR to an existing run directory to resume its recovery.pkl.
#SBATCH --job-name=orcjax-gpp-persist
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_persist_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gpp_persist_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/target-oriented-coupled-gpp-persistent-${SLURM_JOB_ID}}
OUTPUT=${OUTPUT:-$OUTPUT_DIR/report.json}
RUN_TIMEOUT=950m
ADMISSION_ONLY=0

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT_DIR OUTPUT RUN_TIMEOUT ADMISSION_ONLY
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE output_dir=$OUTPUT_DIR"
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_persistent_scale_training.sh"
