#!/bin/bash
# Default resources are the two-node admission topology. Override --nodes,
# --ntasks, --ntasks-per-node, and --time at sbatch for full production.
#SBATCH --job-name=orcjax-teacher-mn
#SBATCH --partition=cnall
#SBATCH --nodes=2
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_multinode_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_multinode_%j.txt
#SBATCH --no-requeue

set -euo pipefail

REPO=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$REPO/runtime
LAUNCHER_ROOT=${LAUNCHER_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}
TASK_SCRIPT=$LAUNCHER_ROOT/scripts/hpc/run_teacher_production_worker_task.sh

: "${WORKTREE:?WORKTREE must identify the admitted Teacher worktree}"
: "${TEACHER_PLAN:?TEACHER_PLAN must be an absolute generation-plan path}"
: "${TEACHER_WORKER_COUNT:?TEACHER_WORKER_COUNT must be set}"
: "${TEACHER_EXPECTED_GIT_HEAD:?TEACHER_EXPECTED_GIT_HEAD must be set}"
: "${SLURM_NTASKS:?submit this script through sbatch}"
: "${SLURM_JOB_ID:?submit this script through sbatch}"

TEACHER_WORKER_OFFSET=${TEACHER_WORKER_OFFSET:-0}
if ! [[ "$TEACHER_WORKER_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "TEACHER_WORKER_OFFSET must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "${TEACHER_WORKER_INDICES:-}" ]]; then
  if ! [[ "$TEACHER_WORKER_INDICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "TEACHER_WORKER_INDICES must be a comma-separated integer list" >&2
    exit 2
  fi
  IFS=',' read -r -a WORKER_INDICES <<< "$TEACHER_WORKER_INDICES"
  if (( ${#WORKER_INDICES[@]} != SLURM_NTASKS )); then
    echo "TEACHER_WORKER_INDICES count must equal SLURM_NTASKS" >&2
    exit 2
  fi
  declare -A SEEN_WORKER_INDICES=()
  for index in "${WORKER_INDICES[@]}"; do
    if (( index >= TEACHER_WORKER_COUNT )); then
      echo "TEACHER_WORKER_INDICES contains an out-of-range worker" >&2
      exit 2
    fi
    if [[ -n "${SEEN_WORKER_INDICES[$index]:-}" ]]; then
      echo "TEACHER_WORKER_INDICES contains a duplicate worker" >&2
      exit 2
    fi
    SEEN_WORKER_INDICES[$index]=1
  done
elif (( TEACHER_WORKER_OFFSET + SLURM_NTASKS > TEACHER_WORKER_COUNT )); then
  echo "allocated tasks exceed TEACHER_WORKER_COUNT" >&2
  exit 2
fi

test -f "$TASK_SCRIPT"
mkdir -p "$RUNTIME_ROOT/logs"

echo "launcher_git_head=$(cd "$LAUNCHER_ROOT" && git rev-parse HEAD)"
echo "teacher_worktree=$WORKTREE expected_teacher_git_head=$TEACHER_EXPECTED_GIT_HEAD"
echo "job_id=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES tasks=$SLURM_NTASKS worker_offset=$TEACHER_WORKER_OFFSET worker_indices=${TEACHER_WORKER_INDICES:-contiguous}"
echo "plan=$TEACHER_PLAN workers_per_node=${SLURM_NTASKS_PER_NODE:-unknown}"

/rmprog/slurm/v22.05.7/bin/srun \
  --kill-on-bad-exit=0 \
  --output="$RUNTIME_ROOT/logs/teacher_${SLURM_JOB_ID}_worker_%t.txt" \
  --error="$RUNTIME_ROOT/logs/teacher_${SLURM_JOB_ID}_worker_%t.txt" \
  bash "$TASK_SCRIPT"
