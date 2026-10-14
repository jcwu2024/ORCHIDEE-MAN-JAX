#!/bin/bash
# Four allocated physical cores: persistent independent B1 workers, pool only.
# No model execution occurs outside this Slurm allocation.
#SBATCH -J orchjax-srv-cpu4
#SBATCH -p cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH -o runtime/logs/global_server_cpu4_%j.txt
#SBATCH -e runtime/logs/global_server_cpu4_%j.txt
#SBATCH --no-requeue

set -euo pipefail

PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
CODE_DIR=${CODE_DIR:?set CODE_DIR to the reviewed immutable pool candidate checkout}
CODE_DIR=$(realpath -m "$CODE_DIR")
case "$CODE_DIR" in
  "$PROJECT"/runtime/worktrees/*) ;;
  *) echo "CODE_DIR must stay under the project runtime worktrees" >&2; exit 2 ;;
esac
PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
PREPARED_INPUT=${PREPARED_INPUT:-$PROJECT/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl}
GPU_REFERENCE_DIR=${GPU_REFERENCE_DIR:-$PROJECT/runtime/outputs/research/global_throughput/gpu-round4/model}
CPU_REFERENCE_DIR=${CPU_REFERENCE_DIR:?set CPU_REFERENCE_DIR to a complete CPU scalar snapshot with the explicitly selected role}
CONFIG=${CONFIG:-$CODE_DIR/configs/orchidee_man_250919.yaml}
RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/cpu4-${SLURM_JOB_ID:-dry-run}}
RUN_ROOT=$(realpath -m "$RUN_ROOT")
case "$RUN_ROOT" in
  "$PROJECT"/runtime/outputs/research/global_throughput/server-benchmark/*) ;;
  *) echo "RUN_ROOT must stay under the project server-benchmark output directory" >&2; exit 2 ;;
esac
ENTRY=$CODE_DIR/scripts/dev/benchmark_global_server_cpu.py
VALIDATION_MODE=${VALIDATION_MODE:-exploratory}
CPU_REFERENCE_ROLE=${CPU_REFERENCE_ROLE:-cpu-internal}
BENCHMARK_SIZES=${BENCHMARK_SIZES:-64}
CPU4_METHODS=${CPU4_METHODS:-pool}
POOL_WORKERS=${POOL_WORKERS:-4}
POOL_START_CONCURRENCY=${POOL_START_CONCURRENCY:-1}
POOL_START_TIMEOUT_SECONDS=${POOL_START_TIMEOUT_SECONDS:-1800}
POOL_COMMAND_TIMEOUT_SECONDS=${POOL_COMMAND_TIMEOUT_SECONDS:-1800}
POOL_MEMORY_LIMIT_GIB=${POOL_MEMORY_LIMIT_GIB:-64}
POOL_MEMORY_GUARD_GIB=${POOL_MEMORY_GUARD_GIB:-64}
POOL_WORKER_ADDRESS_SPACE_GIB=${POOL_WORKER_ADDRESS_SPACE_GIB:-}
CACHE_MODE=${CACHE_MODE:-persistent}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/scalar-worker-cache}
CPU_REFERENCE_ADMISSION_RECORD=${CPU_REFERENCE_ADMISSION_RECORD:-}
EXPECTED_CPU_REFERENCE_ADMISSION_SHA256=${EXPECTED_CPU_REFERENCE_ADMISSION_SHA256:-}
EXPECTED_GPU_REFERENCE_SHA256=${EXPECTED_GPU_REFERENCE_SHA256:-}
ENV_ENTRY=$CODE_DIR/scripts/dev/global_cpu_child_environment.py
read -r -a sizes <<< "$BENCHMARK_SIZES"
case "$VALIDATION_MODE:$CPU_REFERENCE_ROLE" in
  strict:accepted|exploratory:accepted|exploratory:cpu-internal) ;;
  *) echo "invalid validation mode/reference role" >&2; exit 2 ;;
esac
[[ ${#sizes[@]} -gt 0 ]] || { echo "BENCHMARK_SIZES must contain 16 and/or 64" >&2; exit 2; }
declare -A seen_sizes=()
previous_size=0
for size in "${sizes[@]}"; do
  [[ $size == 16 || $size == 64 ]] && [[ -z ${seen_sizes[$size]:-} ]] && [[ $size -gt $previous_size ]] || {
    echo "sizes must be unique ascending 16/64 values" >&2; exit 2;
  }
  seen_sizes[$size]=1
  previous_size=$size
done
[[ $CPU4_METHODS == pool || $CPU4_METHODS == batch ]] || {
  echo "CPU4_METHODS must be pool (current route) or explicitly batch (historical only)" >&2; exit 2;
}
if [[ $CPU4_METHODS == pool ]]; then
  [[ $POOL_WORKERS =~ ^[1-4]$ && $POOL_START_CONCURRENCY =~ ^[1-4]$ && $POOL_START_CONCURRENCY -le $POOL_WORKERS ]] || {
    echo "pool workers/start concurrency must fit four allocated physical cores" >&2; exit 2;
  }
  if [[ $CPU_REFERENCE_ROLE == cpu-internal ]]; then
    [[ $BENCHMARK_SIZES == 64 && -n $CPU_REFERENCE_ADMISSION_RECORD && $EXPECTED_CPU_REFERENCE_ADMISSION_SHA256 =~ ^[0-9a-f]{64}$ && $EXPECTED_GPU_REFERENCE_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
      echo "pool CPU-internal requires FULL64 plus exact new scalar admission and GPU SHA256" >&2; exit 2;
    }
  fi
fi
[[ $CACHE_MODE == off || $CACHE_MODE == persistent ]] || { echo "invalid CACHE_MODE" >&2; exit 2; }
DRY_RUN=0

if [[ ${1:-} == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

[[ ${SLURM_CPUS_PER_TASK:-4} == 4 ]] || { echo "requires exactly four allocated CPUs" >&2; exit 2; }
if [[ $DRY_RUN -eq 0 ]]; then
  [[ -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-1} == 1 ]] || { echo "requires one active Slurm task" >&2; exit 2; }
  [[ ${SLURM_JOB_PARTITION:-} == cnall ]] || { echo "requires cnall" >&2; exit 2; }
  for path in "$PYTHON" "$ENTRY" "$ENV_ENTRY" "$PREPARED_INPUT" "$GPU_REFERENCE_DIR" "$CPU_REFERENCE_DIR" "$CONFIG"; do
    [[ -e $path ]] || { echo "required path missing: $path" >&2; exit 2; }
  done
  if [[ -n $CPU_REFERENCE_ADMISSION_RECORD ]]; then
    [[ -f $CPU_REFERENCE_ADMISSION_RECORD ]] || { echo "scalar admission record missing" >&2; exit 2; }
  fi
  test -z "$(cd "$CODE_DIR" && git status --porcelain --untracked-files=all)" || {
    echo "benchmark checkout must be clean and immutable" >&2; exit 2;
  }
  [[ ! -e $RUN_ROOT ]] || { echo "RUN_ROOT must be new: $RUN_ROOT" >&2; exit 2; }
fi

export PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export ORCHIDEE_REPO_ROOT=$CODE_DIR
export ORCHIDEE_DATA_ROOT=$PROJECT/runtime/data

common=(--cpu-budget 4 --sizes "${sizes[@]}" --prepared-input "$PREPARED_INPUT"
  --gpu-reference-dir "$GPU_REFERENCE_DIR" --cpu-reference-dir "$CPU_REFERENCE_DIR"
  --config "$CONFIG" --warm-repeats 5 --validation-mode "$VALIDATION_MODE"
  --cpu-reference-role "$CPU_REFERENCE_ROLE")
pool=(--mode pool "${common[@]}" --output-dir "$RUN_ROOT/pool4")
pool+=(--memory-guard-gib "$POOL_MEMORY_GUARD_GIB" --pool-memory-limit-gib "$POOL_MEMORY_LIMIT_GIB"
  --pool-workers "$POOL_WORKERS" --pool-start-concurrency "$POOL_START_CONCURRENCY"
  --pool-start-timeout-seconds "$POOL_START_TIMEOUT_SECONDS"
  --pool-command-timeout-seconds "$POOL_COMMAND_TIMEOUT_SECONDS" --cache-mode "$CACHE_MODE"
  --require-child-environment)
if [[ $CACHE_MODE == persistent ]]; then pool+=(--cache-root "$CACHE_ROOT"); fi
if [[ -n $POOL_WORKER_ADDRESS_SPACE_GIB ]]; then pool+=(--pool-worker-address-space-gib "$POOL_WORKER_ADDRESS_SPACE_GIB"); fi
if [[ -n $CPU_REFERENCE_ADMISSION_RECORD ]]; then
  pool+=(--cpu-reference-admission-record "$CPU_REFERENCE_ADMISSION_RECORD"
    --expected-cpu-reference-admission-sha256 "$EXPECTED_CPU_REFERENCE_ADMISSION_SHA256")
fi
if [[ -n $EXPECTED_GPU_REFERENCE_SHA256 ]]; then pool+=(--expected-gpu-reference-sha256 "$EXPECTED_GPU_REFERENCE_SHA256"); fi
batch=(--mode batch "${common[@]}" --memory-guard-gib 24 --output-dir "$RUN_ROOT/batch4")

if [[ $DRY_RUN -eq 1 ]]; then
  printf 'script_defaults: partition=cnall node=not_pinned cpus=4 wall=01:30:00 memory_reservation=none pool_admission_guard_gib=%q pool_sampled_RSS_abort_gib=%q output=%q cpu_reference=%q\n' "$POOL_MEMORY_GUARD_GIB" "$POOL_MEMORY_LIMIT_GIB" "$RUN_ROOT" "$CPU_REFERENCE_DIR"
  printf 'validation_mode=%q cpu_reference_role=%q sizes=%q methods=%q\n' "$VALIDATION_MODE" "$CPU_REFERENCE_ROLE" "$BENCHMARK_SIZES" "$CPU4_METHODS"
  if [[ $CPU4_METHODS == pool ]]; then
    printf 'command='; printf '%q ' "$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores /usr/bin/time -v "$PYTHON" "$ENV_ENTRY" --thread-budget 1 -- "$PYTHON" "$ENTRY" "${pool[@]}"; printf '\n'
  fi
  if [[ $CPU4_METHODS == batch ]]; then
    printf 'command='; printf '%q ' "$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores /usr/bin/time -v "$PYTHON" "$ENTRY" "${batch[@]}"; printf '\n'
  fi
  exit 0
fi

mkdir -p "$RUN_ROOT" "$PROJECT/runtime/logs"
ALLOC_TRES_RECORD=$("$SLURM_BIN/scontrol" show job -o "$SLURM_JOB_ID" | awk '{
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^AllocTRES=/) allocated = substr($i, 11)
    else if ($i ~ /^TRES=/) observed = substr($i, 6)
  }
} END {
  if (allocated != "") printf "AllocTRES\t%s\n", allocated
  else if (observed != "") printf "TRES\t%s\n", observed
}')
[[ -n $ALLOC_TRES_RECORD ]] || { echo "Slurm did not report allocated TRES" >&2; exit 2; }
IFS=$'\t' read -r ALLOC_TRES_SOURCE ALLOC_TRES <<< "$ALLOC_TRES_RECORD"
export ORCHJAX_SLURM_ALLOC_TRES=$ALLOC_TRES
export ORCHJAX_SLURM_ALLOC_TRES_SOURCE=$ALLOC_TRES_SOURCE
{
  date -u +%FT%TZ
  hostname
  uptime
  "$SLURM_BIN/scontrol" show job --details "$SLURM_JOB_ID"
  taskset -pc $$
  lscpu -e=CPU,CORE,SOCKET,ONLINE
  (cd "$CODE_DIR" && git rev-parse HEAD)
  printf 'slurm_cpus_per_task=%s slurm_job_cpus_per_node=%s\n' "$SLURM_CPUS_PER_TASK" "${SLURM_JOB_CPUS_PER_NODE:-unknown}"
  printf 'slurm_alloc_tres=%s slurm_alloc_tres_source=%s\n' \
    "$ORCHJAX_SLURM_ALLOC_TRES" "$ORCHJAX_SLURM_ALLOC_TRES_SOURCE"
  printf 'validation_mode=%q cpu_reference_role=%q sizes=%q methods=%q memory_reservation=none pool_admission_guard_gib=%q pool_sampled_RSS_abort_gib=%q workers=%q startup_concurrency=%q cache_mode=%q cache_root=%q\n' "$VALIDATION_MODE" "$CPU_REFERENCE_ROLE" "$BENCHMARK_SIZES" "$CPU4_METHODS" "$POOL_MEMORY_GUARD_GIB" "$POOL_MEMORY_LIMIT_GIB" "$POOL_WORKERS" "$POOL_START_CONCURRENCY" "$CACHE_MODE" "$CACHE_ROOT"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
} | tee "$RUN_ROOT/launch-context.txt"

run_case() {
  local mode=$1
  shift
  printf 'starting mode=%s utc=%s\n' "$mode" "$(date -u +%FT%TZ)" | tee -a "$RUN_ROOT/launch-context.txt"
  if [[ $mode == pool ]]; then
    "$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
      /usr/bin/time -v "$PYTHON" "$ENV_ENTRY" --thread-budget 1 -- "$PYTHON" "$ENTRY" "$@"
  else
    "$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
      /usr/bin/time -v "$PYTHON" "$ENTRY" "$@"
  fi
}

if [[ $CPU4_METHODS == pool ]]; then
  run_case pool "${pool[@]}"
fi
if [[ $CPU4_METHODS == batch ]]; then
  run_case batch "${batch[@]}"
fi
