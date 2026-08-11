#!/bin/bash
# Run the contract-bound 500-update successor fit with atomic checkpoints.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
UPDATES=${UPDATES:?set UPDATES to the approved production budget}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set PREPARED_MANIFEST to the accepted CPU-prepared manifest}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
RUNNER_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_small_fit_runner_v1.json
CHECKPOINT_HANDOFF=$WORKTREE/manifests/coarse_graining/successor_small_fit_checkpoint_handoff_v1.json
COEFFICIENT_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_small_fit_frozen_coefficients_v1.json
COEFFICIENT_SOURCE_REPORT=$ROOT/runtime/outputs/training/successor-small-fit-hashseed0-cold-2u-1c63494/runner_report.json
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/successor-small-fit-${UPDATES}u-$HEAD_SHORT/runner_report.json}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$(dirname "$OUTPUT")/checkpoints}
RECOVERY_PATH=${RECOVERY_PATH:-$(dirname "$OUTPUT")/recovery_state.pkl}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/successor-small-fit-runner-$HEAD_SHORT}
REPLAY_ONLY_UPDATE=${REPLAY_ONLY_UPDATE:-}
DIAGNOSE_UPDATE=${DIAGNOSE_UPDATE:-}
RECOVERY_IDENTITY_GIT_HEAD=${RECOVERY_IDENTITY_GIT_HEAD:-}
DIAGNOSE_HORIZON=${DIAGNOSE_HORIZON:-}
DIAGNOSE_WINDOW=${DIAGNOSE_WINDOW:-}

case "$WORKTREE" in
  "$ROOT"|"$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE escaped the project-owned root" >&2; exit 2 ;;
esac
case "$UPDATES" in
  ''|*[!0-9]*) echo "UPDATES must be an integer" >&2; exit 2 ;;
esac
if [ "$UPDATES" -ne 500 ]; then
  echo "the checkpoint-producing successor fit requires exactly 500 updates" >&2
  exit 2
fi
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$PREPARED_MANIFEST" \
  "$COEFFICIENT_SOURCE_REPORT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "input escaped the project runtime root: $path" >&2; exit 2 ;;
  esac
  test -f "$path"
done
for path in "$OUTPUT" "$CHECKPOINT_DIR" "$RECOVERY_PATH" "$CACHE_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime output escaped the project root: $path" >&2; exit 2 ;;
  esac
done
test -f "$RUNNER_CONTRACT"
test -f "$CHECKPOINT_HANDOFF"
test -f "$COEFFICIENT_CONTRACT"
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
if [ -e "$OUTPUT" ] && [ ! -e "$RECOVERY_PATH" ]; then
  echo "output exists without a resumable recovery state: $OUTPUT" >&2
  exit 2
fi
test ! -e "$CHECKPOINT_DIR"
command -v nvidia-smi >/dev/null
nvidia-smi -L >/dev/null
mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT"

EXTRA_ARGS=
if [ -n "$REPLAY_ONLY_UPDATE" ] && [ -n "$DIAGNOSE_UPDATE" ]; then
  echo "choose either replay-only or gradient-diagnostic mode" >&2
  exit 2
fi
if [ -n "$REPLAY_ONLY_UPDATE" ]; then
  case "$REPLAY_ONLY_UPDATE" in
    ''|*[!0-9]*) echo "REPLAY_ONLY_UPDATE must be an integer" >&2; exit 2 ;;
  esac
  EXTRA_ARGS="--replay-only-update $REPLAY_ONLY_UPDATE"
fi
if [ -n "$DIAGNOSE_UPDATE" ]; then
  case "$DIAGNOSE_UPDATE" in
    ''|*[!0-9]*) echo "DIAGNOSE_UPDATE must be an integer" >&2; exit 2 ;;
  esac
  test -n "$RECOVERY_IDENTITY_GIT_HEAD"
  EXTRA_ARGS="--diagnose-update $DIAGNOSE_UPDATE --recovery-identity-git-head $RECOVERY_IDENTITY_GIT_HEAD"
  if [ -n "$DIAGNOSE_HORIZON" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --diagnose-horizon $DIAGNOSE_HORIZON"
  fi
  if [ -n "$DIAGNOSE_WINDOW" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --diagnose-window $DIAGNOSE_WINDOW"
  fi
fi

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE updates=$UPDATES output=$OUTPUT"
nvidia-smi -i "$GPU_DEVICE" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT" \
  SINGULARITYENV_PYTHONHASHSEED=0 \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.run_successor_small_fit \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --runner-contract "$RUNNER_CONTRACT" \
      --checkpoint-handoff "$CHECKPOINT_HANDOFF" \
      --prepared-manifest "$PREPARED_MANIFEST" \
      --coefficient-contract "$COEFFICIENT_CONTRACT" \
      --coefficient-source-report "$COEFFICIENT_SOURCE_REPORT" \
      --updates "$UPDATES" \
      --output "$OUTPUT" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --recovery-path "$RECOVERY_PATH" \
      --recovery-interval 25 \
      $EXTRA_ARGS

test -f "$OUTPUT"
if [ -n "$REPLAY_ONLY_UPDATE" ] || [ -n "$DIAGNOSE_UPDATE" ]; then
  sha256sum "$OUTPUT"
  exit 0
fi
test -f "$CHECKPOINT_DIR/checkpoint_manifest.json"
sha256sum "$OUTPUT"
sha256sum "$CHECKPOINT_DIR/checkpoint_manifest.json"
