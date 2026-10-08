#!/bin/bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
SOURCE=$PROJECT/runtime/worktrees/global-throughput-gpu-sweetspot-v2
TARGET=$PROJECT/runtime/worktrees/global-throughput-gpu-sweetspot-v3
BUNDLE=$PROJECT/runtime/transfer/continuation-spool-12b495b.bundle
for path in "$SOURCE" "$TARGET" "$BUNDLE"; do
  case "$(realpath -m "$path")" in /WORK/liwei_work/jcwu/*) ;; *) exit 2 ;; esac
done
test ! -e "$TARGET"
test -f "$BUNDLE"
cd "$SOURCE"
test "$(git rev-parse HEAD)" = 9d19f0760b93e1186eaf796db4903ddfd16a806f
test -z "$(git status --porcelain --untracked-files=all)"
git clone --no-hardlinks "$SOURCE" "$TARGET"
cd "$TARGET"
git bundle verify "$BUNDLE"
git fetch "$BUNDLE" research/global-throughput
git merge --ff-only FETCH_HEAD
test "$(git rev-parse HEAD)" = 12b495b9425de3ad1dc16efa3a5a0e69d7e89222
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e .git/objects/info/alternates
test -z "$(git diff a33b82d -- jax_orchidee)"
bash -n scripts/hpc/slurm_global_gpu_capacity_continue.sh
bash -n scripts/hpc/slurm_global_gpu_capacity.sh
git rev-parse HEAD
printf 'new_checkout_clean=true shared_object_alternates=false physics_unchanged=true\n'
cd "$SOURCE"
test "$(git rev-parse HEAD)" = 9d19f0760b93e1186eaf796db4903ddfd16a806f
test -z "$(git status --porcelain --untracked-files=all)"
printf 'failed_v2_checkout_unchanged=true\n'
