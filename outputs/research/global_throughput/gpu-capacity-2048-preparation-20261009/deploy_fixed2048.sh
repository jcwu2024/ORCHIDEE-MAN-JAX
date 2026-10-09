#!/bin/bash
# Independent Git checkout only; no model/GPU work or scheduler submission.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
PREP=$PROJECT/runtime/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009
SOURCE=$PROJECT/runtime/worktrees/global-throughput-gpu-fixed1024-v1
TARGET=$PROJECT/runtime/worktrees/global-throughput-gpu-fixed2048-v1
EXPECTED=${1:?exact execution commit required}
BUNDLE_NAME=${2:?bundle basename required}
BUNDLE_SHA=${3:?bundle SHA256 required}
BUNDLE_BYTES=${4:?bundle bytes required}
[[ $EXPECTED =~ ^[0-9a-f]{40}$ && $BUNDLE_SHA =~ ^[0-9a-f]{64}$ && $BUNDLE_BYTES =~ ^[0-9]+$ ]]
[[ $BUNDLE_NAME == fixed2048-*.bundle && $BUNDLE_NAME != */* ]]
BUNDLE=$PREP/$BUNDLE_NAME
test ! -e "$TARGET"
test ! -e "$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1"
test "$(sha256sum "$BUNDLE" | cut -d' ' -f1)" = "$BUNDLE_SHA"
test "$(wc -c < "$BUNDLE")" = "$BUNDLE_BYTES"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = bd9e4c04af68d77e79b984bfc8aa064977f893d3
test -z "$(git status --porcelain --untracked-files=all)"
git bundle verify "$BUNDLE"
git clone --no-hardlinks --no-checkout "$SOURCE" "$TARGET"
cd "$TARGET"
git fetch "$BUNDLE" refs/heads/research/global-throughput
git checkout --detach "$EXPECTED"
test "$(git rev-parse HEAD)" = "$EXPECTED"
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e .git/objects/info/alternates
printf 'target_commit=%s\ntarget_clean=true\nindependent_objects=true\nno_alternates=true\n' "$(git rev-parse HEAD)"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = bd9e4c04af68d77e79b984bfc8aa064977f893d3
test -z "$(git status --porcelain --untracked-files=all)"
printf 'source_commit=%s\nsource_clean_after=true\n' "$(git rev-parse HEAD)"
date -u +%FT%TZ
