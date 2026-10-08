#!/bin/bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
SOURCE=$PROJECT/runtime/worktrees/global-throughput-gpu-metadata-v1
TARGET=$PROJECT/runtime/worktrees/global-throughput-gpu-fixed1024-v1
BUNDLE=$PROJECT/runtime/outputs/research/global_throughput/gpu-capacity-1024-preparation-20261008/fixed1024-bd9e4c0.bundle
EXPECTED=bd9e4c04af68d77e79b984bfc8aa064977f893d3
test ! -e "$TARGET"
test ! -e "$PROJECT/runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed1024-v1"
test "$(sha256sum "$BUNDLE" | cut -d' ' -f1)" = a84da4a91faab7ca2925e7aaabe6815654f1eca9feb4467dc6898bf94b36113d
test "$(wc -c < "$BUNDLE")" = 273676
cd "$SOURCE"
test "$(git rev-parse HEAD)" = ad6c0e1faa7e22b590627844e9cb0ad492e9deb3
test -z "$(git status --porcelain --untracked-files=all)"
git bundle verify "$BUNDLE"
git clone --no-hardlinks --no-checkout "$SOURCE" "$TARGET"
cd "$TARGET"
git fetch "$BUNDLE" refs/heads/research/global-throughput
git checkout --detach "$EXPECTED"
test "$(git rev-parse HEAD)" = "$EXPECTED"
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e .git/objects/info/alternates
printf 'target_commit=%s\n' "$(git rev-parse HEAD)"
printf 'target_git_dir=%s\n' "$(git rev-parse --git-dir)"
printf 'target_clean=true\nindependent_objects=true\nno_alternates=true\n'
cd "$SOURCE"
test "$(git rev-parse HEAD)" = ad6c0e1faa7e22b590627844e9cb0ad492e9deb3
test -z "$(git status --porcelain --untracked-files=all)"
printf 'source_commit=%s\nsource_clean_after=true\n' "$(git rev-parse HEAD)"
date -u +%FT%TZ
