#!/bin/bash
#SBATCH --job-name=orcjax-teacher-2y
#SBATCH --partition=cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=/WORK/liwei_work/jcwu/out_orcjax_teacher_2y_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/out_orcjax_teacher_2y_%j.txt
#SBATCH --no-requeue

set -euo pipefail

JCWU_ROOT=/WORK/liwei_work/jcwu
REPO=$JCWU_ROOT/ORCHIDEE-MAN-JAX
ASSETS=$JCWU_ROOT/orchidee_man_jax_assets
TEACHER_CONFIG=$REPO/configs/orchidee_man_250919.yaml
PYTHON=$JCWU_ROOT/.venvs/orcjax_cpu/bin/python
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
OUTPUT_ROOT=$JCWU_ROOT/orchidee_man_jax_outputs
DATASET_ID=teacher-two-year-worker-${SLURM_JOB_ID}
DATASET_ROOT=$OUTPUT_ROOT/training/$DATASET_ID
PLAN=$DATASET_ROOT/generation_plan.json
CACHE=$OUTPUT_ROOT/xla_cache/teacher_two_year_worker/${SLURM_JOB_ID}

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
test -f "$TEACHER_CONFIG"
test -f "$ASSETS/checkpoints/paper_driver_1961_year_end_state.pkl"
test -f "$ASSETS/configs/teacher_compatibility_used_run.def"
test -d "$ASSETS/reference_case_001_071"
mkdir -p "$DATASET_ROOT" "$CACHE"

cd "$REPO"
test -z "$(git status --porcelain --untracked-files=all)"

"$PYTHON" - "$PLAN" "$DATASET_ID" "$DATASET_ROOT" "$ASSETS" "$TEACHER_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
dataset_id = sys.argv[2]
output_root = Path(sys.argv[3])
assets = Path(sys.argv[4])
teacher_config = Path(sys.argv[5])
run_def = assets / "configs" / "teacher_compatibility_used_run.def"
reference = assets / "reference_case_001_071"
state_cache = assets / "checkpoints" / "paper_driver_1961_year_end_state.pkl"
entries = []
for year in (1962, 1963):
    entry = {
        "landpoint_id": "001.0-071.0",
        "year": year,
        "days": 365,
        "spatial_split": "train",
        "temporal_split": "train",
        "run_def": str(run_def),
        "reference_run_dir": str(reference),
    }
    if year == 1962:
        entry["state_cache"] = str(state_cache)
    entries.append(entry)
payload = {
    "schema_version": "daily_teacher_generation_plan_v1",
    "dataset_id": dataset_id,
    "teacher_config": str(teacher_config),
    "block_size": 28,
    "output_root": str(output_root),
    "entries": entries,
}
plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "git_head=$(git rev-parse HEAD)"
echo "job_id=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK host=$(hostname)"
echo "plan=$PLAN dataset_root=$DATASET_ROOT cache=$CACHE"

"$PYTHON" -m research.daily_coarse_graining.teacher_shards validate \
  --plan "$PLAN" --worker-count 1

"$SLURM_BIN/srun" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v "$PYTHON" \
  -m research.daily_coarse_graining.teacher_shards generate \
  --plan "$PLAN" --output-root "$DATASET_ROOT" \
  --worker-index 0 --worker-count 1

"$PYTHON" -m research.daily_coarse_graining.teacher_shards aggregate \
  --plan "$PLAN" --output-root "$DATASET_ROOT" --worker-count 1

test -s "$DATASET_ROOT/dataset_manifest.json"
echo "completed_manifest=$DATASET_ROOT/dataset_manifest.json"
