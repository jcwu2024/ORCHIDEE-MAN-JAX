#!/bin/bash
# Proposal only: scan accepted v6 arrays for the 79 added value-head statistics.
# Requested ceiling: 8 CPU cores * 2 h * CNY 0.07 = CNY 1.12 worst case.
# No Teacher generation, historical v5 read, model training, or GPU allocation.
# Submission requires a fresh explicit approval block.
#SBATCH -J orcjax-faststats
#SBATCH -p cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/fast_owner_statistics_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/fast_owner_statistics_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$ROOT/runtime
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
DATASET=${DATASET:?set DATASET to the accepted v6 dataset_manifest.json}
ACCEPTANCE=${ACCEPTANCE:?set ACCEPTANCE to the accepted v6 acceptance_report.json}
BASE_STATISTICS=${BASE_STATISTICS:?set BASE_STATISTICS to accepted training_statistics.json}
PROCESS_STATISTICS=${PROCESS_STATISTICS:?set PROCESS_STATISTICS to accepted process_training_statistics.json}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new runtime output directory}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python
SRUN=/rmprog/slurm/v22.05.7/bin/srun
PARTITIONS=8

case "$WORKTREE" in
  "$ROOT") ;;
  "$RUNTIME_ROOT/worktrees/"*) ;;
  *) echo "WORKTREE must be the repository root or stay under runtime/worktrees" >&2; exit 2 ;;
esac
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" "$PROCESS_STATISTICS" "$OUTPUT_DIR"; do
  case "$path" in
    "$RUNTIME_ROOT/"*) ;;
    *) echo "data, statistics, and output must stay under runtime" >&2; exit 2 ;;
  esac
done

test -x "$PYTHON"
test -x "$SRUN"
test -f "$DATASET"
test -f "$ACCEPTANCE"
test -f "$BASE_STATISTICS"
test -f "$PROCESS_STATISTICS"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/partials" "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/cache/xdg"

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export XDG_CACHE_HOME=$RUNTIME_ROOT/cache/xdg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$WORKTREE"

pids=()
for partition in $(seq 0 $((PARTITIONS - 1))); do
  "$SRUN" --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
    "$PYTHON" -m scripts.hpc.fit_successor_fast_owner_statistics fit \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --partition-index "$partition" \
      --partition-count "$PARTITIONS" \
      --chunk-rows 64 \
      --output "$OUTPUT_DIR/partials/part-$partition.json" \
      > "$OUTPUT_DIR/partials/part-$partition.log" 2>&1 &
  pids+=("$!")
done
worker_failure=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    worker_failure=1
  fi
done
if [[ "$worker_failure" != 0 ]]; then
  echo "one or more fast-owner statistics partitions failed" >&2
  exit 1
fi

"$PYTHON" -m scripts.hpc.fit_successor_fast_owner_statistics merge \
  --dataset-manifest "$DATASET" \
  --acceptance-report "$ACCEPTANCE" \
  --base-statistics "$BASE_STATISTICS" \
  --process-statistics "$PROCESS_STATISTICS" \
  --partials-dir "$OUTPUT_DIR/partials" \
  --output "$OUTPUT_DIR/successor_fast_owner_statistics.json" \
  > "$OUTPUT_DIR/merge.log"

test -f "$OUTPUT_DIR/successor_fast_owner_statistics.json"
test -f "$OUTPUT_DIR/successor_fast_owner_statistics.npz"
