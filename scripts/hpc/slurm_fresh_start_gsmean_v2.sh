#!/bin/bash
# Queue only with session authorization. Ceiling: (2.20 + 8*0.07)*1 = CNY 2.76.
#SBATCH --job-name=orcjax-gsmean-v2
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gsmean-v2_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/gsmean-v2_%j.txt
#SBATCH --no-requeue
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
cd "$ROOT"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
export GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
export RUN_TIMEOUT=55m
export OUTPUT=$ROOT/runtime/outputs/training/fresh-start-gsmean-v2-${SLURM_JOB_ID}/report.json
exec bash scripts/hpc/run_fresh_start_one_day.sh \
  --protocol manifests/coarse_graining/fresh_start_one_day_probe_v2.json
