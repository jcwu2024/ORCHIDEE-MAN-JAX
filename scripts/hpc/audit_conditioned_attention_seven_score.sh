#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export PYTHON_MODULE=scripts.hpc.audit_conditioned_attention_seven_score
export RUN_TIMEOUT=${RUN_TIMEOUT:-45m}
exec bash "${WORKTREE:-$ROOT}/scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.sh"
