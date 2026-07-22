#!/bin/bash
#SBATCH --job-name=orcjax-teacher-1961
#SBATCH --partition=cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=ibc11b04n04
#SBATCH --time=01:15:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/orcjax_teacher_1961_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/orcjax_teacher_1961_%j.txt
#SBATCH --no-requeue

set -euo pipefail

JCWU_ROOT=/WORK/liwei_work/jcwu
REPO=$JCWU_ROOT/ORCHIDEE-MAN-JAX
RUNTIME_ROOT=$REPO/runtime
ASSETS=$RUNTIME_ROOT/assets
PILOT_ASSETS=$ASSETS/daily_teacher_pilot_v2
LANDPOINT_ID=001.0-071.0
LANDPOINT_ASSETS=$PILOT_ASSETS/landpoints/$LANDPOINT_ID
TEACHER_CONFIG=$REPO/configs/orchidee_man_250919.yaml
PYTHON=$REPO/.venvs/orcjax_cpu/bin/python
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
OUTPUT_ROOT=$RUNTIME_ROOT/outputs
DATASET_ID=teacher-1961-cold-start-promotion-${SLURM_JOB_ID}
DATASET_ROOT=$OUTPUT_ROOT/training/$DATASET_ID
PLAN=$DATASET_ROOT/generation_plan.json
CACHE=${TEACHER_PROMOTION_CACHE:-$OUTPUT_ROOT/xla_cache/teacher_1961_cold_start/${SLURM_JOB_ID}}

export ORCHIDEE_REPO_ROOT=$REPO
export ORCHIDEE_RUNTIME_ROOT=$RUNTIME_ROOT
export ORCHIDEE_DATA_ROOT=$RUNTIME_ROOT/data
export ORCHIDEE_REFERENCE_ROOT=$ASSETS
export ORCHIDEE_OUTPUT_ROOT=$OUTPUT_ROOT
export XDG_CACHE_HOME=$RUNTIME_ROOT/cache/xdg
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
test -f "$ASSETS/checkpoints/paper_driver_1961_year_end_state_current.pkl"
test -f "$LANDPOINT_ASSETS/used_run.def"
test -d "$LANDPOINT_ASSETS/reference"
test -f "$ORCHIDEE_DATA_ROOT/forcing/cruncep_twodeg_1961.nc"
test -f "$ORCHIDEE_DATA_ROOT/forcing/cruncep_twodeg_1962.nc"
mkdir -p "$DATASET_ROOT" "$CACHE" "$RUNTIME_ROOT/logs"

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
landpoint_assets = assets / "daily_teacher_pilot_v2" / "landpoints" / "001.0-071.0"
run_def = landpoint_assets / "used_run.def"
reference = landpoint_assets / "reference"
accepted_1961 = assets / "checkpoints" / "paper_driver_1961_year_end_state_current.pkl"
payload = {
    "schema_version": "daily_teacher_generation_plan_v2",
    "dataset_id": dataset_id,
    "teacher_config": str(teacher_config),
    "block_size": 7,
    "output_root": str(output_root),
    "entries": [
        {
            "landpoint_id": "001.0-071.0",
            "year": 1961,
            "days": 365,
            "initialization_mode": "cold_start_bootstrap",
            "spatial_split": "train",
            "temporal_split": "train",
            "run_def": str(run_def),
            "reference_run_dir": str(reference),
            "acceptance_checkpoint": str(accepted_1961),
        },
        {
            "landpoint_id": "001.0-071.0",
            "year": 1962,
            "days": 365,
            "initialization_mode": "year_start_checkpoint",
            "spatial_split": "train",
            "temporal_split": "train",
            "run_def": str(run_def),
            "reference_run_dir": str(reference),
        },
    ],
}
plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "git_head=$(git rev-parse HEAD)"
echo "job_id=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK host=$(hostname)"
echo "plan=$PLAN dataset_root=$DATASET_ROOT cache=$CACHE"
echo "loadavg=$(cat /proc/loadavg)"
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status

"$PYTHON" -m research.daily_coarse_graining.teacher_shards validate \
  --plan "$PLAN" --worker-count 1

"$SLURM_BIN/srun" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v "$PYTHON" \
  -m research.daily_coarse_graining.teacher_shards generate \
  --plan "$PLAN" --output-root "$DATASET_ROOT" \
  --worker-index 0 --worker-count 1

"$PYTHON" -m research.daily_coarse_graining.teacher_shards aggregate \
  --plan "$PLAN" --output-root "$DATASET_ROOT" --worker-count 1

"$PYTHON" - "$DATASET_ROOT/dataset_manifest.json" "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

manifest_path = Path(sys.argv[1])
dataset_root = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
shards = {int(item["year"]): item for item in manifest["shards"]}
assert set(shards) == {1961, 1962}
cold = shards[1961]
restart = shards[1962]
assert cold["initialization_mode"] == "cold_start_bootstrap"
assert cold["bootstrap_day"] == 1
assert cold["transition_start_day"] == 2
assert cold["transition_count"] == 364
assert cold["year_end_acceptance"]["status"] in {"exact", "numeric_close"}
assert restart["initialization_mode"] == "year_start_checkpoint"
assert restart["bootstrap_day"] is None
assert restart["transition_start_day"] == 1
assert restart["transition_count"] == 365
assert restart["preceding_checkpoint_sha256"] == cold["checkpoint_sha256"]
with np.load(dataset_root / cold["shard"], allow_pickle=False) as arrays:
    day_index = arrays["day_index"]
    assert day_index.shape == (364,)
    assert int(day_index[0]) == 2
    assert int(day_index[-1]) == 365
    assert arrays["state_trajectory"].shape[0] == 365
summary = {
    "status": "passed",
    "teacher_git_head": manifest["teacher_git_head"],
    "cold_start": {
        "transition_count": cold["transition_count"],
        "year_end_acceptance": cold["year_end_acceptance"],
        "timing_seconds": cold["timing_seconds"],
    },
    "restart_1962": {
        "transition_count": restart["transition_count"],
        "timing_seconds": restart["timing_seconds"],
    },
}
print("cold_start_promotion_summary=" + json.dumps(summary, sort_keys=True))
PY

echo "completed_manifest=$DATASET_ROOT/dataset_manifest.json"
