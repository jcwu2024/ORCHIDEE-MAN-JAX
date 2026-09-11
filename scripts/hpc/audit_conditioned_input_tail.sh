#!/bin/bash
# Fixed-weight admission on one idle free GPU, never sbatch or training.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=$ROOT
export EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set the transferred code revision}
export GPU_DEVICE=${GPU_DEVICE:?select an idle free GPU}
export OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/diagnostics/conditioned-input-tail-14991510.json}
export RUN_TIMEOUT=45m
export SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export PYTHON_MODULE=scripts.hpc.run_conditioned_snow_free_continuation
source <(sed '/^exec bash /d' "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.sh")
TAIL_SCOPE=${TAIL_SCOPE:-all}
EXTRA_ARGS=
case "$TAIL_SCOPE" in
  all) ;;
  doc) EXTRA_ARGS="--tail-scope doc --reference-report $ROOT/runtime/outputs/diagnostics/conditioned-input-tail-14991510.json"
       export OUTPUT=$ROOT/runtime/outputs/diagnostics/conditioned-doc-tail-14991510.json ;;
  *) exit 2 ;;
esac
bash <(sed "s|\"\$PYTHON\" -m \"\$PYTHON_MODULE\"|\"\$PYTHON\" -m scripts.hpc.audit_conditioned_input_tail $EXTRA_ARGS|" \
  "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh")
