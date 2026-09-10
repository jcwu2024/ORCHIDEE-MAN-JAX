#!/bin/bash
# Ten serial CPU workers; ceiling 10 * 1 * 12 * 0.07 = CNY 8.40.
#SBATCH --job-name=orcjax-data-expand
#SBATCH --partition=cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9%10
#SBATCH --time=12:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/data-expand_%A_%a.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/data-expand_%A_%a.txt
#SBATCH --no-requeue
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set the admitted preparation checkout}
TASK=${SLURM_ARRAY_TASK_ID:?array index is required}
case "$TASK" in 0|1|2|3|4|5|6|7|8|9) ;; *) exit 2 ;; esac
START_UPDATE=$((1001 + 200 * TASK))
END_UPDATE=$((1200 + 200 * TASK))
OUTPUT_DIR=$ROOT/runtime/outputs/training/conditioned-coverage-expanded-${SLURM_ARRAY_JOB_ID}/chunks/${START_UPDATE}-${END_UPDATE}
cd "$WORKTREE"
test "$(git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export ORCHIDEE_REPO_ROOT=$WORKTREE ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs PYTHONPATH=$WORKTREE
timeout --signal=TERM 710m "$ROOT/.venvs/orcjax_cpu/bin/python" \
  -m scripts.hpc.prepare_target_oriented_training_batches \
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
  --training-contract "$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json" \
  --updates 3000 --start-update "$START_UPDATE" --end-update "$END_UPDATE" \
  --reuse-retained-inputs-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-250-9b82b5a/prepared_manifest.json" \
  --output-dir "$OUTPUT_DIR"
