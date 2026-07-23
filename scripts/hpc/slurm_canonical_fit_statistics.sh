#!/bin/bash
#SBATCH -J orcjax-stats
#SBATCH -p cnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH -o runtime/logs/canonical_statistics_%j.txt
#SBATCH -e runtime/logs/canonical_statistics_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python
DATASET=${DATASET:?set DATASET to the accepted dataset_manifest.json}
OUTPUT=${OUTPUT:?set OUTPUT to the training_statistics.json destination}

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" runtime/logs
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg

"$PYTHON" -m research.daily_coarse_graining.canonical_training_run \
  fit-statistics \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --chunk-rows 64
