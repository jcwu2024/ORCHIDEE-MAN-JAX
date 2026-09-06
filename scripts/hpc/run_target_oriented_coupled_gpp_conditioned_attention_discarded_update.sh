#!/bin/bash
# Run one discarded conditioned-attention update preflight on free gln01.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=${WORKTREE:-$ROOT}
export PYTHON_MODULE=scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update
export ARTIFACT_AWARE_PERSISTENT_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.json
PERSISTENT_DIR=$ROOT/runtime/outputs/training/target-oriented-coupled-gpp-artifact-aware-persistent-714aa0d
export PERSISTENT_TRAINING_REPORT=$PERSISTENT_DIR/report.json
export PERSISTENT_ARTIFACT_AWARE_RECOVERY=$PERSISTENT_DIR/recovery.pkl
export CONDITIONING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_v1.json
export CONDITIONED_ADMISSION_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-conditioned-attention-812e851.json
export CONDITIONED_DISCARDED_UPDATE_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_discarded_update_v1.json
export RUN_TIMEOUT=${RUN_TIMEOUT:-180m}

exec bash "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh"
