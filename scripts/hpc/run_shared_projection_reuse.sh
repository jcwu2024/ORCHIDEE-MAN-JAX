#!/bin/bash
# Free-node component admission; no Slurm submission or retained updates.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?set immutable candidate checkout}
export ENTRY_MODULE=scripts.hpc.probe_shared_projection_reuse
export RUN_TIMEOUT=${RUN_TIMEOUT:-90m}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-$ROOT/runtime/outputs/diagnostics/shared-learning-4408daf-gpu.update-000100.checkpoint.pkl}
REFERENCE_DIR=${REFERENCE_DIR:-$ROOT/runtime/outputs/diagnostics/shared-staged-source100-v1}
CANDIDATE_DIR=${CANDIDATE_DIR:?set separate candidate component directory}
case "$CANDIDATE_DIR" in "$ROOT"/runtime/outputs/diagnostics/*) ;; *) exit 2 ;; esac
case "$REFERENCE_DIR" in "$ROOT"/runtime/outputs/diagnostics/*) ;; *) exit 2 ;; esac
bash "$CODE_ROOT/scripts/hpc/run_shared_learning.sh" "$@" \
  --benchmark-protocol "$CODE_ROOT/manifests/coarse_graining/shared_execution_benchmark_v1.json" \
  --resume-checkpoint "$SOURCE_CHECKPOINT" --state-dir "$REFERENCE_DIR" \
  --candidate-dir "$CANDIDATE_DIR" --microbatch-size 4
