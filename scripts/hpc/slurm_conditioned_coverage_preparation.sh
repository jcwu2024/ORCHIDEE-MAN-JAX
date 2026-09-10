#!/bin/bash
# One allocation, 4 nodes x 10 CPU workers; ceiling 40 * 4 * 0.07 = CNY 11.20.
#SBATCH --job-name=orcjax-data-expand
#SBATCH --partition=cnall
#SBATCH --nodes=4
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/data-expand_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/data-expand_%j.txt
#SBATCH --no-requeue
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set the admitted preparation checkout}
WORKER_COUNT=40
WORKERS_PER_NODE=10
SRUN=/rmprog/slurm/v22.05.7/bin/srun
SCONTROL=/rmprog/slurm/v22.05.7/bin/scontrol
OUTPUT_ROOT=$ROOT/runtime/outputs/training/conditioned-coverage-expanded-${SLURM_JOB_ID:?job allocation required}
cd "$WORKTREE"
if [ "$(git rev-parse HEAD)" != "$EXPECTED_GIT_HEAD" ]; then
  echo "Preparation checkout differs from EXPECTED_GIT_HEAD=$EXPECTED_GIT_HEAD" >&2
  exit 2
fi
if [ "${SLURM_NTASKS:?task allocation required}" != "$WORKER_COUNT" ]; then
  echo "This launcher needs 40 tasks, 10 per node, each with one CPU" >&2
  exit 2
fi
mapfile -t NODES < <("$SCONTROL" show hostnames "${SLURM_JOB_NODELIST:?node allocation required}")
if [ "${#NODES[@]}" != 4 ]; then
  echo "This launcher needs four allocated nodes" >&2
  exit 2
fi
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export ORCHIDEE_REPO_ROOT=$WORKTREE ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs PYTHONPATH=$WORKTREE
mkdir -p "$OUTPUT_ROOT/worker-logs"
echo "job=$SLURM_JOB_ID workers=$WORKER_COUNT packages=1001-3000 output=$OUTPUT_ROOT"
pids=()
for ((worker=0; worker<WORKER_COUNT; worker++)); do
  START_UPDATE=$((1001 + 50 * worker))
  END_UPDATE=$((START_UPDATE + 49))
  NODE=${NODES[$((worker / WORKERS_PER_NODE))]}
  OUTPUT_DIR=$OUTPUT_ROOT/chunks/${START_UPDATE}-${END_UPDATE}
  echo "worker=$worker node=$NODE packages=$START_UPDATE-$END_UPDATE"
  "$SRUN" --exclusive --nodes=1 --nodelist="$NODE" --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
  timeout --signal=TERM 230m "$ROOT/.venvs/orcjax_cpu/bin/python" \
  -m scripts.hpc.prepare_target_oriented_training_batches \
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
  --training-contract "$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json" \
  --updates 3000 --start-update "$START_UPDATE" --end-update "$END_UPDATE" \
  --reuse-retained-inputs-manifest "$ROOT/runtime/outputs/training/target-oriented-prepared-250-9b82b5a/prepared_manifest.json" \
  --output-dir "$OUTPUT_DIR" \
  > "$OUTPUT_ROOT/worker-logs/worker-$worker.log" 2>&1 &
  pids+=("$!")
done

worker_failure=0
for ((worker=0; worker<WORKER_COUNT; worker++)); do
  if wait "${pids[$worker]}"; then
    echo "worker=$worker completed"
  else
    code=$?
    echo "worker=$worker failed exit=$code; see worker log" >&2
    worker_failure=1
  fi
done
if [ "$worker_failure" != 0 ]; then
  exit 1
fi
echo "All 2000 packages prepared; chunk manifests are under $OUTPUT_ROOT/chunks"
