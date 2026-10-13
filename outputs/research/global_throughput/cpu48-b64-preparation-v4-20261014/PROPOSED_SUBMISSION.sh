#!/usr/bin/env bash
# Reviewable proposal only. Not executed during v4 preparation.
# Deploy and verify the exact artifacts only after supervisor approval.
(
  set -euo pipefail
  for cpu48_prefix in SBATCH_ ORCHJAX_; do
    while IFS= read -r cpu48_variable; do
      unset "$cpu48_variable"
    done < <(compgen -v "$cpu48_prefix")
  done
  unset SLURM_BIN CODE_DIR PYTHON ENTRY COORDINATOR PREPARED_INPUT GPU_REFERENCE_DIR CONFIG
  unset CPU_REFERENCE_DIR CPU_REFERENCE_ADMISSION_RECORD EXPECTED_CPU_REFERENCE_ADMISSION_SHA256
  unset RUN_ROOT EXPECTED_CODE_COMMIT EVIDENCE_DIR ORCHIDEE_REPO_ROOT ORCHIDEE_DATA_ROOT
  cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX

  export CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4
  export PYTHON=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python
  export EXPECTED_CODE_COMMIT=60dccbddde10be4295aab7e7fe55fa7864d3c3f3
  export CPU_REFERENCE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar
  export CPU_REFERENCE_ADMISSION_RECORD=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json
  export EXPECTED_CPU_REFERENCE_ADMISSION_SHA256=f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d

  /rmprog/slurm/v22.05.7/bin/sbatch \
    --job-name=orchjax-cpu48-b64-v4 \
    --partition=cnall --nodes=1 --ntasks=1 --cpus-per-task=48 \
    --time=01:00:00 --no-requeue \
    --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4 \
    --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_cpu48_b64_v4_%j.txt \
    --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_cpu48_b64_v4_%j.txt \
    --export=ALL \
    /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4/scripts/hpc/slurm_global_server_cpu_48core.sh
)
