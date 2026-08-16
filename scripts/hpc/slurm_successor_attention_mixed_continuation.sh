#!/bin/bash
# Continue only the Attention arm for 250 mixed point-year updates.
# Ceiling: 4 CPU cores * 2 h * CNY 0.07 + 1 GPU * 2 h * CNY 2.20
#          = CNY 0.56 + CNY 4.40 = CNY 4.96.
# Submission requires explicit approval of this exact resource envelope.
#SBATCH --job-name=orcjax-attn-mixed
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/attention_mixed_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/attention_mixed_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set the accepted 1001-2000 manifest}
PARENT_RECOVERY=${PARENT_RECOVERY:?set the accepted update-1000 recovery}
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/runtime/outputs/training/successor-attention-mixed-annealed-$HEAD_SHORT}
SRUN=/rmprog/slurm/v22.05.7/bin/srun
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}

for path in "$WORKTREE" "$PREPARED_MANIFEST" "$PARENT_RECOVERY" "$OUTPUT_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "path escaped the project runtime root: $path" >&2; exit 2 ;;
  esac
done
test -d "$WORKTREE"
test -f "$PREPARED_MANIFEST"
test -f "$PARENT_RECOVERY"
test -x "$SRUN"
test "${SLURM_JOB_PARTITION:?missing partition allocation}" = gnall
test "${SLURM_CPUS_PER_TASK:?missing CPU allocation}" = 4
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
mkdir -p "$OUTPUT_ROOT"

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE PREPARED_MANIFEST
export PARENT_RECOVERY OUTPUT_ROOT
echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD"
echo "gpu_device=$GPU_DEVICE output_root=$OUTPUT_ROOT"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

exec "$SRUN" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v bash \
    "$WORKTREE/scripts/hpc/run_successor_attention_mixed_continuation.sh"
