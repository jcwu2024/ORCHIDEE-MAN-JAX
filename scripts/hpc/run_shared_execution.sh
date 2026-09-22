#!/bin/bash
# Execution-only replay on free nodes; never submits a paid job.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?set immutable checkout}
export ENTRY_MODULE=scripts.hpc.probe_shared_execution
export RUN_TIMEOUT=${RUN_TIMEOUT:-180m}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-$ROOT/runtime/outputs/diagnostics/shared-learning-4408daf-gpu.update-000100.checkpoint.pkl}
bash "$CODE_ROOT/scripts/hpc/run_shared_learning.sh" "$@" \
  --benchmark-protocol "$CODE_ROOT/manifests/coarse_graining/shared_execution_benchmark_v1.json" \
  --resume-checkpoint "$SOURCE_CHECKPOINT"
