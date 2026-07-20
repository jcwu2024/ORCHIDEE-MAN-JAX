#!/usr/bin/env bash
# Generic Slurm template. Adapt account/partition/time/memory to the cluster.
#SBATCH --job-name=orchidee-jax-pft14
#SBATCH --array=0-668
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
manifest="${repo_root}/manifests/landpoints_669.json"

LANDPOINT_ID="$(${UV_PYTHON:-python} -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_landpoint_ids"][int(sys.argv[2])])' \
  "${manifest}" "${SLURM_ARRAY_TASK_ID}")"
export LANDPOINT_ID

exec "${repo_root}/scripts/hpc/run_landpoint.sh"
