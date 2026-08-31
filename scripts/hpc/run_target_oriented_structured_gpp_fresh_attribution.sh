#!/bin/bash
# Run the frozen structured-GPP fresh-panel attribution on free gln01.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking gln01 availability}
MODE=${MODE:?set MODE to admission or formal}
FRESH_VALIDATION=${FRESH_VALIDATION:?set fresh prepared-manifest path}
FORMAL_REPORT=${FORMAL_REPORT:?set accepted fresh formal report path}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR inside the project runtime output tree}
RUN_TIMEOUT=${RUN_TIMEOUT:-120m}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/target-oriented-structured-gpp-joint-calibration}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$MODE" in admission|formal) ;; *) exit 2 ;; esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$FRESH_VALIDATION" in "$ROOT/runtime/outputs/"*/prepared_manifest.json) ;; *) exit 2 ;; esac
case "$FORMAL_REPORT" in "$ROOT/runtime/outputs/diagnostics/"*/report.json) ;; *) exit 2 ;; esac
case "$OUTPUT_DIR" in "$ROOT/runtime/outputs/diagnostics/"*) ;; *) exit 2 ;; esac
case "$CACHE_ROOT" in "$ROOT/runtime/cache/"*) ;; *) exit 2 ;; esac
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR/report.json"

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
TRAINING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json
FLUX_BASIS_AMENDMENT=$WORKTREE/manifests/coarse_graining/target_oriented_flux_basis_amendment_v1.json
PREPARED=$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json
OLD_VALIDATION=$ROOT/runtime/outputs/training/target-oriented-validation-9b82b5a/prepared_manifest.json
CONSUMED_PANEL=$ROOT/runtime/outputs/training/target-oriented-validation-9b82b5a/validation_panel.json
STAGE_CHECKPOINT=$ROOT/runtime/outputs/training/target-oriented-training-formal-u1000-433d286/stage-update-1000.pkl
STAGE_CHECKPOINT_SHA256=a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7
DISTRIBUTION_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_distribution_aware_maintenance_v1.json
DISTRIBUTION_DIR=$ROOT/runtime/outputs/training/target-oriented-distribution-aware-u250-14856147
DISTRIBUTION_REPORT=$DISTRIBUTION_DIR/report.json
DISTRIBUTION_CHECKPOINT=$DISTRIBUTION_DIR/stage-update-0250.pkl
DISTRIBUTION_CHECKPOINT_SHA256=baae0c13aa7f22f04e56bec43b915e7e2dab8c2ca775e91d260f26136d55c7da
STRUCTURED_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_v1.json
BROAD_TRAINING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_broad_training_v1.json
BROAD_DIR=$ROOT/runtime/outputs/training/target-oriented-structured-gpp-broad-14883081
BROAD_REPORT=$BROAD_DIR/report.json
SELECTED_CHECKPOINT=$BROAD_DIR/stage-update-1000.pkl
SCREEN_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_fresh_screen_v1.json
ATTRIBUTION_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_fresh_attribution_v1.json

for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" "$PROCESS_STATISTICS" \
  "$FAST_STATISTICS" "$TRAINING_CONTRACT" "$FLUX_BASIS_AMENDMENT" "$PREPARED" \
  "$OLD_VALIDATION" "$FRESH_VALIDATION" "$CONSUMED_PANEL" "$STAGE_CHECKPOINT" \
  "$DISTRIBUTION_CONTRACT" "$DISTRIBUTION_REPORT" "$DISTRIBUTION_CHECKPOINT" \
  "$STRUCTURED_CONTRACT" "$BROAD_TRAINING_CONTRACT" "$BROAD_REPORT" \
  "$SELECTED_CHECKPOINT" "$SCREEN_CONTRACT" "$ATTRIBUTION_CONTRACT" \
  "$FORMAL_REPORT"; do
  test -f "$path"
done
mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "host=$(hostname) mode=$MODE git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE output=$OUTPUT_DIR"
nvidia-smi -i "$GPU_DEVICE" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT" \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  timeout --signal=TERM "$RUN_TIMEOUT" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.audit_target_oriented_structured_gpp_fresh_attribution \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --training-contract "$TRAINING_CONTRACT" \
      --flux-basis-amendment "$FLUX_BASIS_AMENDMENT" \
      --prepared-manifest "$PREPARED" \
      --prepared-updates 1000 \
      --validation-prepared-manifest "$OLD_VALIDATION" \
      --fresh-validation-prepared-manifest "$FRESH_VALIDATION" \
      --consumed-validation-panel "$CONSUMED_PANEL" \
      --stage-checkpoint "$STAGE_CHECKPOINT" \
      --stage-checkpoint-sha256 "$STAGE_CHECKPOINT_SHA256" \
      --distribution-contract "$DISTRIBUTION_CONTRACT" \
      --distribution-training-report "$DISTRIBUTION_REPORT" \
      --distribution-checkpoint "$DISTRIBUTION_CHECKPOINT" \
      --distribution-checkpoint-sha256 "$DISTRIBUTION_CHECKPOINT_SHA256" \
      --structured-contract "$STRUCTURED_CONTRACT" \
      --broad-training-contract "$BROAD_TRAINING_CONTRACT" \
      --broad-training-report "$BROAD_REPORT" \
      --selected-checkpoint "$SELECTED_CHECKPOINT" \
      --screen-contract "$SCREEN_CONTRACT" \
      --attribution-contract "$ATTRIBUTION_CONTRACT" \
      --fresh-formal-report "$FORMAL_REPORT" \
      --mode "$MODE" \
      --output-dir "$OUTPUT_DIR"

test -f "$OUTPUT_DIR/report.json"
sha256sum "$OUTPUT_DIR/report.json"
