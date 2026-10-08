#!/bin/bash
# Proposed command only; explicit new paid approval required before execution.
set -euo pipefail
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v2 \
RUN_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v2-continue256512 \
B128_BASELINE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity \
/rmprog/slurm/v22.05.7/bin/sbatch --export=ALL --time=01:30:00 --nodelist=ibc13b03n03 \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v2/scripts/hpc/slurm_global_gpu_capacity_continue.sh
