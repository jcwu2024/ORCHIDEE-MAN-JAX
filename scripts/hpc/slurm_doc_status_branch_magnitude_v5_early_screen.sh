#!/bin/bash
# V5 no-checkpoint 10-epoch DOC early-stop screen.
# Ceiling: 4 CPU cores * 2 h * CNY 0.07 + 1 GPU * 2 h * CNY 2.20
#          = CNY 0.56 + CNY 4.40 = CNY 4.96.
# Submission requires explicit approval of this exact resource envelope.
# The two-hour ceiling includes cold compilation; the matched v4 cold smoke
# took 37:28 and the complete v4 100-epoch run took 05:30:23.
#SBATCH --job-name=orcjax-doc-v5-10ep
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/doc_status_branch_magnitude_v5_early_screen_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/doc_status_branch_magnitude_v5_early_screen_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research checkout}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/doc-status-branch-magnitude-attention-v5-litter-10-epoch-$HEAD_SHORT/report.json}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/doc-status-branch-magnitude-attention-v5-litter-$HEAD_SHORT}
SRUN=/rmprog/slurm/v22.05.7/bin/srun

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must be an isolated project runtime checkout" >&2; exit 2 ;;
esac
case "$OUTPUT" in
  "$ROOT/runtime/outputs/"*) ;;
  *) echo "OUTPUT escaped the project runtime outputs" >&2; exit 2 ;;
esac
case "$CACHE_ROOT" in
  "$ROOT/runtime/cache/"*) ;;
  *) echo "CACHE_ROOT escaped the project runtime cache" >&2; exit 2 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPU}"
GPU_DEVICE=${CUDA_VISIBLE_DEVICES%%,*}
test -x "$SRUN"
test "${SLURM_JOB_PARTITION:?missing partition allocation}" = gnall
test "${SLURM_CPUS_PER_TASK:?missing CPU allocation}" = 4
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
mkdir -p "$ROOT/runtime/logs"

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT CACHE_ROOT
export MECHANISTIC_OUTPUT=1
export HIERARCHICAL_COORDINATE_ATTENTION=0
export COORDINATE_COUPLED_DOC_TRANSITION=0
export DOC_STATUS_BRANCH_MAGNITUDE=1
export ISOLATED_EXPERT=litter_poc_doc
export TRAINING_SMOKE=0 MAXIMUM_EPOCHS=10 EVALUATION_INTERVAL=5

echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD"
echo "gpu_device=$GPU_DEVICE output=$OUTPUT cache_root=$CACHE_ROOT"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

exec "$SRUN" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v bash \
    "$WORKTREE/scripts/hpc/run_strong_baseline_one_point_qualification.sh"
