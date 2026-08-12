#!/bin/bash
# Prepare staged-v3 updates 501-1000 with four resumable CPU workers.
# Ceiling: 4 CPU cores * 4 h * CNY 0.07 = CNY 1.12.
# Submission requires explicit approval of this exact resource envelope.
#SBATCH --job-name=orcjax-staged-prep1000
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/staged_prep1000_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/staged_prep1000_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
MILESTONE=1000
WORKER_COUNT=4
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/successor-staged-prepared-0501-1000-$HEAD_SHORT}
SRUN=/rmprog/slurm/v22.05.7/bin/srun

case "$OUTPUT_DIR" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_DIR escaped the project runtime root" >&2; exit 2 ;;
esac
test -x "$SRUN"
test "${SLURM_NTASKS:?missing Slurm task allocation}" = "$WORKER_COUNT"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR/prepared_manifest.json"
mkdir -p "$OUTPUT_DIR/worker-logs" "$ROOT/runtime/logs"

echo "job_id=$SLURM_JOB_ID host=$(hostname) git_head=$EXPECTED_GIT_HEAD"
echo "milestone=$MILESTONE workers=$WORKER_COUNT output_dir=$OUTPUT_DIR"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

pids=()
for worker_index in $(seq 0 $((WORKER_COUNT - 1))); do
  "$SRUN" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
    env WORKTREE="$WORKTREE" EXPECTED_GIT_HEAD="$EXPECTED_GIT_HEAD" \
      MILESTONE="$MILESTONE" OUTPUT_DIR="$OUTPUT_DIR" \
      WORKER_INDEX="$worker_index" WORKER_COUNT="$WORKER_COUNT" \
      bash "$WORKTREE/scripts/hpc/prepare_successor_staged_batches.sh" \
      > "$OUTPUT_DIR/worker-logs/worker-$worker_index.log" 2>&1 &
  pids+=("$!")
done

worker_failure=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    worker_failure=1
  fi
done
if [ "$worker_failure" != 0 ]; then
  echo "one or more staged preparation workers failed" >&2
  exit 1
fi

"$SRUN" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
  env WORKTREE="$WORKTREE" EXPECTED_GIT_HEAD="$EXPECTED_GIT_HEAD" \
    MILESTONE="$MILESTONE" OUTPUT_DIR="$OUTPUT_DIR" FINALIZE=true \
    WORKER_COUNT="$WORKER_COUNT" \
    bash "$WORKTREE/scripts/hpc/prepare_successor_staged_batches.sh"

test -f "$OUTPUT_DIR/prepared_manifest.json"
sha256sum "$OUTPUT_DIR/prepared_manifest.json"
