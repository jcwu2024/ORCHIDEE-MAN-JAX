#!/bin/bash
# Four allocated physical cores: four persistent one-core workers and JAX batch.
# No model execution occurs outside this Slurm allocation.
#SBATCH -J orchjax-srv-cpu4
#SBATCH -p cnall
#SBATCH --nodelist=ibc11b02n25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o runtime/logs/global_server_cpu4_%j.txt
#SBATCH -e runtime/logs/global_server_cpu4_%j.txt
#SBATCH --no-requeue

set -euo pipefail

PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
CODE_DIR=${CODE_DIR:-$PROJECT/runtime/worktrees/global-throughput-v1}
CODE_DIR=$(realpath -m "$CODE_DIR")
case "$CODE_DIR" in
  "$PROJECT"/runtime/worktrees/*) ;;
  *) echo "CODE_DIR must stay under the project runtime worktrees" >&2; exit 2 ;;
esac
PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
PREPARED_INPUT=${PREPARED_INPUT:-$PROJECT/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl}
GPU_REFERENCE_DIR=${GPU_REFERENCE_DIR:-$PROJECT/runtime/outputs/research/global_throughput/gpu-round4/model}
CPU_REFERENCE_DIR=${CPU_REFERENCE_DIR:?set CPU_REFERENCE_DIR to the passed cpu1 scalar reference output}
CONFIG=${CONFIG:-$CODE_DIR/configs/orchidee_man_250919.yaml}
RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/cpu4-${SLURM_JOB_ID:-dry-run}}
RUN_ROOT=$(realpath -m "$RUN_ROOT")
case "$RUN_ROOT" in
  "$PROJECT"/runtime/outputs/research/global_throughput/server-benchmark/*) ;;
  *) echo "RUN_ROOT must stay under the project server-benchmark output directory" >&2; exit 2 ;;
esac
ENTRY=$CODE_DIR/scripts/dev/benchmark_global_server_cpu.py
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
  for path in "$PYTHON" "$ENTRY" "$PREPARED_INPUT" "$GPU_REFERENCE_DIR" "$CPU_REFERENCE_DIR" "$CONFIG"; do
    [[ -e $path ]] || { echo "required path missing: $path" >&2; exit 2; }
  done
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

common=(--cpu-budget 4 --sizes 16 64 --prepared-input "$PREPARED_INPUT"
  --gpu-reference-dir "$GPU_REFERENCE_DIR" --cpu-reference-dir "$CPU_REFERENCE_DIR"
  --config "$CONFIG" --warm-repeats 5)
pool=(--mode pool "${common[@]}" --output-dir "$RUN_ROOT/pool4")
pool+=(--memory-guard-gib 60)
batch=(--mode batch "${common[@]}" --memory-guard-gib 20 --output-dir "$RUN_ROOT/batch4")

if [[ $DRY_RUN -eq 1 ]]; then
  printf 'partition=cnall node=ibc11b02n25 cpus=4 wall=01:00:00 output=%q cpu_reference=%q\n' "$RUN_ROOT" "$CPU_REFERENCE_DIR"
  printf 'command='; printf '%q ' "$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores /usr/bin/time -v "$PYTHON" "$ENTRY" "${pool[@]}"; printf '\n'
  printf 'command='; printf '%q ' "$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores /usr/bin/time -v "$PYTHON" "$ENTRY" "${batch[@]}"; printf '\n'
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
} | tee "$RUN_ROOT/launch-context.txt"

run_case() {
  local mode=$1
  shift
  printf 'starting mode=%s utc=%s\n' "$mode" "$(date -u +%FT%TZ)" | tee -a "$RUN_ROOT/launch-context.txt"
  "$SLURM_BIN/srun" --ntasks=1 --cpus-per-task=4 --cpu-bind=cores \
    /usr/bin/time -v "$PYTHON" "$ENTRY" "$@"
}

run_case pool "${pool[@]}"
run_case batch "${batch[@]}"
