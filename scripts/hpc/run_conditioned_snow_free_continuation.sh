#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=${WORKTREE:-$ROOT}
export PYTHON_MODULE=scripts.hpc.run_conditioned_snow_free_continuation
export RUN_TIMEOUT=${RUN_TIMEOUT:-45m}
exec bash "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.sh"
