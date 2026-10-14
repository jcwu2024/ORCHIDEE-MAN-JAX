#!/bin/bash
# Proposed one-off pilot submission. Not approved or executed by preparation.
# Review the exact pin/resources/cost first. --dry-run only prints the command.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-cpu-scalar-pool-v1
EXPECTED_CODE_COMMIT=50864f065363f6cd0de685c18d9c257723bf2838
export CPU_REFERENCE_DIR=$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar
export PREPARED_INPUT=$PROJECT/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl
export GPU_REFERENCE_DIR=$PROJECT/runtime/outputs/research/global_throughput/gpu-round4/model
export CPU_REFERENCE_ADMISSION_RECORD=$PROJECT/runtime/outputs/research/global_throughput/cpu-scalar-pool-preparation-20261014/scalar-admission.json
export EXPECTED_CPU_REFERENCE_ADMISSION_SHA256=fafd685bb54c7bb366f0d10d38cba31957ba754b9a5bcd8801e75abfd7907883
export EXPECTED_GPU_REFERENCE_SHA256=a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6
export CONFIG=$CODE_DIR/configs/orchidee_man_250919.yaml
export CPU4_METHODS=pool BENCHMARK_SIZES=64 POOL_WORKERS=4 POOL_START_CONCURRENCY=1
export POOL_START_TIMEOUT_SECONDS=1200 POOL_COMMAND_TIMEOUT_SECONDS=300
export POOL_MEMORY_LIMIT_GIB=64 POOL_MEMORY_GUARD_GIB=64
export CACHE_MODE=persistent VALIDATION_MODE=exploratory CPU_REFERENCE_ROLE=cpu-internal
export ORCHJAX_PARENT_CPU_COUNT=4
# The candidate launcher creates cpu4-${SLURM_JOB_ID} and its own worker-cache.
unset RUN_ROOT CACHE_ROOT POOL_WORKER_ADDRESS_SPACE_GIB
# Prevent inherited sbatch options from silently changing the reviewed request.
while IFS= read -r option; do unset "$option"; done < <(compgen -A variable SBATCH_)
submission=(/rmprog/slurm/v22.05.7/bin/sbatch --parsable --partition=cnall
  --nodes=1 --ntasks=1 --cpus-per-task=4 --time=01:30:00 --no-requeue
  --chdir="$CODE_DIR" --export=ALL
  --output="$PROJECT/runtime/logs/global_scalar_pool_%j.txt"
  --error="$PROJECT/runtime/logs/global_scalar_pool_%j.txt"
  "$CODE_DIR/scripts/hpc/slurm_global_server_cpu_4core.sh")
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
  printf 'not_submitted: '; printf '%q ' "${submission[@]}"; printf '\n'
  exit 0
fi
[[ $# == 0 ]] || { echo "usage: $0 [--dry-run]" >&2; exit 2; }
[[ -d $PROJECT/runtime/logs ]] || { echo 'missing logs directory' >&2; exit 2; }
[[ $(cd "$CODE_DIR" && git rev-parse HEAD) == "$EXPECTED_CODE_COMMIT" ]] || {
  echo 'candidate execution pin mismatch' >&2; exit 2;
}
[[ -z $(cd "$CODE_DIR" && git status --porcelain --untracked-files=all) ]] || {
  echo 'candidate must be immutable and clean' >&2; exit 2;
}
actual_admission=$(sha256sum "$CPU_REFERENCE_ADMISSION_RECORD")
[[ ${actual_admission%% *} == "$EXPECTED_CPU_REFERENCE_ADMISSION_SHA256" ]] || {
  echo 'new scalar admission checksum mismatch' >&2; exit 2;
}
cd "$CODE_DIR"
exec "${submission[@]}"
