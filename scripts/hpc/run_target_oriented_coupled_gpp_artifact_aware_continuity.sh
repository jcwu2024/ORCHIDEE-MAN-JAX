#!/bin/bash
# Run the no-checkpoint three-update artifact-aware continuity smoke on free gln01.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
export WORKTREE=${WORKTREE:-$ROOT}
export PYTHON_MODULE=scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_continuity
export ARTIFACT_AWARE_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.json
export MULTI_GROUP_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-multi-group-direction-c5c0a56.json
export CONTINUITY_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.json
export ARTIFACT_AWARE_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-artifact-aware-direction-77d5a45.json
export RUN_TIMEOUT=${RUN_TIMEOUT:-240m}

exec bash "$WORKTREE/scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh"
