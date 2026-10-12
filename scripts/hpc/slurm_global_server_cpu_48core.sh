#!/usr/bin/env bash
# One native FULL64 B64 workload in one 48-CPU parent allocation.
# Preparation script only; paid submission requires separate explicit approval.
#SBATCH -J orchjax-cpu48-b64-v2
#SBATCH -p cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:30:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_cpu48_b64_v2_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_cpu48_b64_v2_%j.txt
#SBATCH --no-requeue

set -euo pipefail

PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
SLURM_BIN=${SLURM_BIN:-/rmprog/slurm/v22.05.7/bin}
CODE_DIR=${CODE_DIR:-$PROJECT/runtime/worktrees/global-throughput-cpu48-b64-v2}
CODE_DIR=$(realpath -m "$CODE_DIR")
case "$CODE_DIR" in
  "$PROJECT"/runtime/worktrees/*) ;;
  *) echo "CODE_DIR must stay under project runtime/worktrees" >&2; exit 2 ;;
esac
PYTHON=${PYTHON:-$PROJECT/.venvs/orcjax_cpu/bin/python}
ENTRY=${ENTRY:-$CODE_DIR/scripts/dev/benchmark_global_server_cpu.py}
COORDINATOR=${COORDINATOR:-$CODE_DIR/scripts/hpc/global_cpu48_b64_coordinator.py}
PREPARED_INPUT=${PREPARED_INPUT:-$PROJECT/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl}
GPU_REFERENCE_DIR=${GPU_REFERENCE_DIR:-$PROJECT/runtime/outputs/research/global_throughput/gpu-round4/model}
CONFIG=${CONFIG:-$CODE_DIR/configs/orchidee_man_250919.yaml}
RUN_ROOT=${RUN_ROOT:-$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012/${SLURM_JOB_ID:-dry-run}}
RUN_ROOT=$(realpath -m "$RUN_ROOT")
EXPECTED_CODE_COMMIT=${EXPECTED_CODE_COMMIT:?set EXPECTED_CODE_COMMIT to the reviewed execution-only commit}

case "$RUN_ROOT" in
  "$PROJECT"/runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012/*) ;;
  *) echo "RUN_ROOT must stay under project CPU48/B64 v2 outputs" >&2; exit 2 ;;
esac
[[ ${SLURM_CPUS_PER_TASK:-48} == 48 ]] || { echo "requires one 48-CPU parent allocation" >&2; exit 2; }

DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--dry-run]" >&2; exit 2; }

if [[ $DRY_RUN -eq 0 ]]; then
  [[ -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-1} == 1 ]] || { echo "requires one active Slurm task" >&2; exit 2; }
  [[ ${SLURM_JOB_PARTITION:-} == cnall ]] || { echo "requires cnall" >&2; exit 2; }
  for path in "$PYTHON" "$ENTRY" "$COORDINATOR" "$PREPARED_INPUT" "$GPU_REFERENCE_DIR" "$CONFIG"; do
    [[ -e $path ]] || { echo "required path missing: $path" >&2; exit 2; }
  done
  [[ "$PYTHON" == "$PROJECT/.venvs/orcjax_cpu/bin/python" \
     && "$ENTRY" == "$CODE_DIR/scripts/dev/benchmark_global_server_cpu.py" \
     && "$COORDINATOR" == "$CODE_DIR/scripts/hpc/global_cpu48_b64_coordinator.py" ]] || {
    echo "actual execution must use the canonical CPU environment and CODE_DIR entrypoints" >&2; exit 2;
  }
  [[ $(cd "$CODE_DIR" && git rev-parse HEAD) == "$EXPECTED_CODE_COMMIT" ]] || {
    echo "execution checkout HEAD differs from EXPECTED_CODE_COMMIT" >&2; exit 2;
  }
  test -z "$(cd "$CODE_DIR" && git status --porcelain --untracked-files=all)" || {
    echo "execution checkout must be clean and immutable" >&2; exit 2;
  }
  [[ ! -e $RUN_ROOT ]] || { echo "RUN_ROOT must be new: $RUN_ROOT" >&2; exit 2; }
fi

export PYTHONDONTWRITEBYTECODE=1
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export ORCHIDEE_REPO_ROOT=$CODE_DIR
export ORCHIDEE_DATA_ROOT=$PROJECT/runtime/data
export ORCHJAX_PARENT_CPU_COUNT=48
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset XLA_FLAGS

args=(--output-dir "$RUN_ROOT" --project-root "$PROJECT" --code-dir "$CODE_DIR"
  --python "$PYTHON" --entry "$ENTRY" --expected-code-commit "$EXPECTED_CODE_COMMIT"
  --expected-gpu-reference-sha256 a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6
  --prepared-input "$PREPARED_INPUT" --gpu-reference-dir "$GPU_REFERENCE_DIR" --config "$CONFIG"
  --partition cnall --slurm-bin "$SLURM_BIN" --budget-seconds 5400 --reference-cap-seconds 2700
  --minimum-candidate-seconds 1800)
if [[ $DRY_RUN -eq 1 ]]; then
  args+=(--dry-run)
  printf 'allocation_contract: partition=cnall nodes=1 tasks=1 parent_cpus=48 wall=01:30:00 memory_reservation=none primary_max_cny=3.60 conservative_max_cny=5.04 planned_chdir=%q\n' "$CODE_DIR"
  printf 'coordinator='; printf '%q ' "$PYTHON" "$COORDINATOR" "${args[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$PROJECT/runtime/logs"
EVIDENCE_DIR=${EVIDENCE_DIR:-$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012}
EVIDENCE_DIR=$(realpath -m "$EVIDENCE_DIR")
case "$EVIDENCE_DIR" in
  "$PROJECT"/runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012) ;;
  *) echo "EVIDENCE_DIR must stay in the CPU48/B64 v2 output namespace" >&2; exit 2 ;;
esac
mkdir -p "$EVIDENCE_DIR"
export ORCHJAX_SLURM_JOB_RECORD_STDOUT=$EVIDENCE_DIR/scontrol-job-${SLURM_JOB_ID}.stdout
export ORCHJAX_SLURM_JOB_RECORD_STDERR=$EVIDENCE_DIR/scontrol-job-${SLURM_JOB_ID}.stderr
export ORCHJAX_SLURM_JOB_RECORD_EXITCODE=$EVIDENCE_DIR/scontrol-job-${SLURM_JOB_ID}.exitcode
for path in "$ORCHJAX_SLURM_JOB_RECORD_STDOUT" "$ORCHJAX_SLURM_JOB_RECORD_STDERR" "$ORCHJAX_SLURM_JOB_RECORD_EXITCODE"; do
  [[ ! -e $path ]] || { echo "scontrol evidence path already exists: $path" >&2; exit 2; }
done
set +e
"$SLURM_BIN/scontrol" show job -o "$SLURM_JOB_ID" >"$ORCHJAX_SLURM_JOB_RECORD_STDOUT" 2>"$ORCHJAX_SLURM_JOB_RECORD_STDERR"
SCONTROL_EXITCODE=$?
set -e
printf '%s\n' "$SCONTROL_EXITCODE" >"$ORCHJAX_SLURM_JOB_RECORD_EXITCODE"
VALIDATED_ALLOCATION=$("$PYTHON" "$COORDINATOR" --validate-allocation) || {
  echo "saved scontrol allocation evidence failed the CPU48 contract" >&2
  exit 2
}
export ORCHJAX_SLURM_ALLOC_TRES=$(printf '%s' "$VALIDATED_ALLOCATION" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["allocated_tres"])')
export ORCHJAX_SLURM_ALLOC_TRES_SOURCE=$(printf '%s' "$VALIDATED_ALLOCATION" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["allocated_tres_source"])')
cd "$CODE_DIR"
exec "$PYTHON" "$COORDINATOR" "${args[@]}"
