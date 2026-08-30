#!/bin/bash
# Run exactly two chained broad-training updates on free gln01.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking gln01 availability}
OUTPUT_DIR=${OUTPUT_DIR:?set a new diagnostics output directory}
RUN_TIMEOUT=${RUN_TIMEOUT:-90m}
MAXIMUM_UPDATES=2

export WORKTREE EXPECTED_GIT_HEAD GPU_DEVICE OUTPUT_DIR RUN_TIMEOUT MAXIMUM_UPDATES
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_structured_gpp_broad_training.sh"
