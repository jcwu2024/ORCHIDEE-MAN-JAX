#!/bin/bash
# Corrected proposal only; new explicit paid approval required before execution.
set -euo pipefail
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3 \
RUN_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512 \
B128_BASELINE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity \
/rmprog/slurm/v22.05.7/bin/sbatch --export=ALL --time=01:30:00 --nodelist=ibc13b03n02 \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3/scripts/hpc/slurm_global_gpu_capacity_continue.sh
