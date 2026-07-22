#!/bin/bash
#SBATCH --job-name=orcjax-cpu-benchmark
#SBATCH --partition=cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/out_orcjax_cpu_benchmark_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/out_orcjax_cpu_benchmark_%j.txt
#SBATCH --no-requeue

set -euo pipefail

JCWU_ROOT=/WORK/liwei_work/jcwu
REPO=$JCWU_ROOT/ORCHIDEE-MAN-JAX
ASSETS=$JCWU_ROOT/orchidee_man_jax_assets
PYTHON=$JCWU_ROOT/.venvs/orcjax_cpu/bin/python
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
OUTPUT_ROOT=$JCWU_ROOT/orchidee_man_jax_outputs
RESULT_DIR=$OUTPUT_ROOT/performance/daily_coarse_graining/cpu_slurm
RESULT=$RESULT_DIR/compiled_training_capture_cpu_job_${SLURM_JOB_ID}.json
CACHE=$OUTPUT_ROOT/xla_cache/teacher_cpu_benchmark/${SLURM_JOB_ID}

export ORCHIDEE_REPO_ROOT=$REPO
export ORCHIDEE_DATA_ROOT=$JCWU_ROOT/orchidee_man_jax_data
export ORCHIDEE_REFERENCE_ROOT=$ASSETS
export ORCHIDEE_OUTPUT_ROOT=$OUTPUT_ROOT
export XDG_CACHE_HOME=$JCWU_ROOT/.cache
export JAX_COMPILATION_CACHE_DIR=$CACHE
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

test -x "$PYTHON"
test -x "$SLURM_BIN/srun"
test -f "$ASSETS/checkpoints/paper_driver_1961_year_end_state.pkl"
test -f "$ASSETS/configs/teacher_compatibility_used_run.def"
test -d "$ASSETS/reference_case_001_071"
mkdir -p "$RESULT_DIR" "$CACHE"

cd "$REPO"
test -z "$(git status --porcelain --untracked-files=all)"

echo "git_head=$(git rev-parse HEAD)"
echo "job_id=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK host=$(hostname)"
echo "result=$RESULT"

/usr/bin/time -v "$SLURM_BIN/srun" --cpu-bind=cores "$PYTHON" \
  scripts/dev/benchmark_compiled_training_capture.py \
  --state-cache "$ASSETS/checkpoints/paper_driver_1961_year_end_state.pkl" \
  --run-def "$ASSETS/configs/teacher_compatibility_used_run.def" \
  --reference-run-dir "$ASSETS/reference_case_001_071" \
  --year 1962 \
  --days 29 \
  --block-size 28 \
  --hot-repeats 3 \
  --output "$RESULT"

test -s "$RESULT"
echo "completed_result=$RESULT"
