#!/bin/bash
#SBATCH --job-name=shared-compile
#SBATCH --partition=gnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-compile-%j.log
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-compile-%j.log
#SBATCH --no-requeue

set -euo pipefail
# Submit with --nodelist=<currently idle node>; this does not reserve the whole node.
export CODE_ROOT=${CODE_ROOT:?set existing immutable shared-execution checkout}
DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:?set deployed diagnostic tools directory}
export DIAGNOSTIC_SCRIPT="$DIAGNOSTIC_ROOT/probe_shared_compile_cache.py"
export OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-compile-${SLURM_JOB_ID}.json
export MODE=gpu DIAGNOSTIC_MODE=compile RUN_TIMEOUT=40m
exec bash "$DIAGNOSTIC_ROOT/run_shared_compile_cache.sh" "$@"
