#!/usr/bin/env bash
# Reviewable proposal only. No paid submission is authorized by preparation.
(
  set -euo pipefail
  for cpu48_prefix in SBATCH_ ORCHJAX_; do
    while IFS= read -r cpu48_variable; do unset "$cpu48_variable"; done < <(compgen -v "$cpu48_prefix")
  done
  unset SLURM_BIN CODE_DIR PYTHON ENTRY COORDINATOR PREPARED_INPUT GPU_REFERENCE_DIR CONFIG
  unset CPU_REFERENCE_DIR CPU_REFERENCE_ADMISSION_RECORD EXPECTED_CPU_REFERENCE_ADMISSION_SHA256
  unset RUN_ROOT EXPECTED_CODE_COMMIT EVIDENCE_DIR ORCHIDEE_REPO_ROOT ORCHIDEE_DATA_ROOT
  cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
  export CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v5
  export PYTHON=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python
  export EXPECTED_CODE_COMMIT=0100c8b09be5137b822173e0a0be66cb64939727
  export CPU_REFERENCE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar
  export CPU_REFERENCE_ADMISSION_RECORD=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/scalar-admission.json
  export EXPECTED_CPU_REFERENCE_ADMISSION_SHA256=4d2d3d18d92c0d6c6bc737059bc2b2555cf36f77b4a3c4406844071ab6a3e7f5
  if [[ ${1:-} == --dry-run ]]; then
    shift
    [[ $# -eq 0 ]] || exit 2
    exec bash "$CODE_DIR/scripts/hpc/slurm_global_server_cpu_48core.sh" --dry-run
  fi
  [[ $# -eq 0 ]] || exit 2
  /rmprog/slurm/v22.05.7/bin/sbatch \
    --job-name=orchjax-cpu48-b64-v5 \
    --partition=cnall --nodes=1 --ntasks=1 --cpus-per-task=48 \
    --time=01:00:00 --no-requeue \
    --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v5 \
    --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_cpu48_b64_v5_%j.txt \
    --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_cpu48_b64_v5_%j.txt \
    --export=ALL \
    /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v5/scripts/hpc/slurm_global_server_cpu_48core.sh
)
