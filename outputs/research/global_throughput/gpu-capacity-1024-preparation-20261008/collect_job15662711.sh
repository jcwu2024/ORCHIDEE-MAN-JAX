#!/bin/bash
# Saved-array collection only, executed on test02 after terminal raw9 capture.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-gpu-fixed1024-v1
RUN_ROOT=$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
date -u +%Y-%m-%dT%H:%M:%SZ
hostname
taskset -pc $$
/usr/bin/time -v taskset -c 0-3 "$PROJECT/.venvs/orcjax_cpu/bin/python" -B "$CODE_DIR/outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/collect_fixed1024_result.py" \
  --code-dir "$CODE_DIR" --expected-code-sha bd9e4c04af68d77e79b984bfc8aa064977f893d3 \
  --stage-dir "$RUN_ROOT/capacity" --sacct-file "$RUN_ROOT/sacct-raw9.txt" \
  --admission "$PROJECT/runtime/outputs/research/global_throughput/gpu-sweetspot-preparation-20261006/target-input-admission.json" \
  --b512-evidence-dir "$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-15657822/capacity" \
  --job-id 15662711 --job-name orchjax-gpu-cap1024 --node ibc13b03n05 \
  --gpu-uuid GPU-51897aa5-dac4-9d20-523e-830c362cf7a6 --slurm-global-gpu-id 0 --driver-version 525.85.12 \
  --approved-nodes 1 --approved-tasks 1 --approved-cpus 4 --approved-gpus 1 --approved-wall-seconds 5400 \
  --output "$RUN_ROOT/collected-fixed1024.json"
