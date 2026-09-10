#!/bin/bash
# Free-node diagnostic; use existing ce7cd79 runtime without changing its checkout.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=$ROOT
export EXPECTED_GIT_HEAD=ce7cd7985e28fb5b53ae2d578d11f6b1d44761db
export GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/conditioned-score-discrepancy-14991510.json}
export RUN_TIMEOUT=45m
export SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
# Reuse the admitted argument/environment setup; change only the invoked module.
export PYTHON_MODULE=scripts.hpc.run_conditioned_snow_free_continuation
source <(sed '/^exec bash /d' "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.sh")
DIAGNOSTIC_MODE=${DIAGNOSTIC_MODE:-}
case "$DIAGNOSTIC_MODE" in ''|--trace-localization) ;; *) exit 2 ;; esac
bash <(sed "s|\"\$PYTHON\" -m \"\$PYTHON_MODULE\"|\"\$PYTHON\" -m runtime.transfer.audit_conditioned_score_discrepancy $DIAGNOSTIC_MODE|" \
  "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh")
