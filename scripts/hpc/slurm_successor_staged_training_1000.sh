#!/bin/bash
# Resume both staged-v3 arms from update 500 through update 1000.
# Ceiling: 4 CPU cores * 2 h * CNY 0.07 + 1 GPU * 2 h * CNY 2.20
#          = CNY 0.56 + CNY 4.40 = CNY 4.96.
# Submission requires explicit approval of this exact resource envelope.
#SBATCH --job-name=orcjax-staged-fit1000
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/staged_fit1000_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/staged_fit1000_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set the independently admitted 501-1000 manifest}
VALIDATION_PREPARED_MANIFEST=${VALIDATION_PREPARED_MANIFEST:?set fixed validation prepared manifest}
PARENT_RECOVERY=${PARENT_RECOVERY:?set the accepted update-500 recovery}
PARENT_RECOVERY_SHA256=${PARENT_RECOVERY_SHA256:?set the accepted recovery SHA256}
MILESTONE=1000
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/runtime/outputs/training/successor-staged-fit-1000-$HEAD_SHORT}
OUTPUT=${OUTPUT:-$OUTPUT_ROOT/stage_report.json}
RECOVERY_PATH=${RECOVERY_PATH:-$OUTPUT_ROOT/recovery_state.pkl}
BEST_CHECKPOINT=${BEST_CHECKPOINT:-$OUTPUT_ROOT/best_checkpoint.pkl}
SRUN=/rmprog/slurm/v22.05.7/bin/srun
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}

for path in "$PREPARED_MANIFEST" "$VALIDATION_PREPARED_MANIFEST" \
  "$PARENT_RECOVERY"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime input escaped the project root: $path" >&2; exit 2 ;;
  esac
  test -f "$path"
done
case "$OUTPUT_ROOT" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_ROOT escaped the project runtime root" >&2; exit 2 ;;
esac
test -x "$SRUN"
test "${SLURM_CPUS_PER_TASK:?missing CPU allocation}" = 4
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
printf '%s  %s\n' "$PARENT_RECOVERY_SHA256" "$PARENT_RECOVERY" | sha256sum -c -
mkdir -p "$OUTPUT_ROOT" "$ROOT/runtime/logs"
if [ ! -e "$RECOVERY_PATH" ]; then
  cp "$PARENT_RECOVERY" "$RECOVERY_PATH"
fi

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE MILESTONE PREPARED_MANIFEST
export VALIDATION_PREPARED_MANIFEST OUTPUT RECOVERY_PATH BEST_CHECKPOINT
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD"
echo "gpu_device=$GPU_DEVICE milestone=$MILESTONE output=$OUTPUT"
echo "parent_recovery=$PARENT_RECOVERY recovery_copy=$RECOVERY_PATH"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

exec "$SRUN" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v bash "$WORKTREE/scripts/hpc/run_successor_staged_training.sh"
