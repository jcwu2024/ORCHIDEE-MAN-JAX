#!/bin/bash
# Localize block-0 attention heads and source tokens on free gln01 without training.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=${WORKTREE:-$ROOT}
export PYTHON_MODULE=scripts.hpc.audit_target_oriented_coupled_gpp_attention_head_token_attribution
export ARTIFACT_AWARE_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.json
export MULTI_GROUP_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-multi-group-direction-c5c0a56.json
export CONTINUITY_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.json
export ARTIFACT_AWARE_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-artifact-aware-direction-77d5a45.json
export ARTIFACT_AWARE_PERSISTENT_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.json
export CONTINUITY_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-artifact-aware-continuity-c345e64.json
PERSISTENT_DIR=$ROOT/runtime/outputs/training/target-oriented-coupled-gpp-artifact-aware-persistent-714aa0d
export PERSISTENT_TRAINING_REPORT=$PERSISTENT_DIR/report.json
export PERSISTENT_REJECTED_PROPOSAL=$PERSISTENT_DIR/rejected-proposal.json
export PERSISTENT_ARTIFACT_AWARE_RECOVERY=$PERSISTENT_DIR/recovery.pkl
export LOCALIZED_JUMP_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-localized-jump-e436b96.json
export SNOWHEAT_CAUSAL_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-snowheat-causal-67fae75.json
export CROSSED_ATTRIBUTION_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-day5-crossed-1c7dc9b.json
export STRUCTURED_PRECISION_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-structured-precision-587c3fd.json
export POTENTIAL_INTERNAL_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-potential-internal-56ec8f2.json
export RUN_TIMEOUT=${RUN_TIMEOUT:-45m}

exec bash "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh"
