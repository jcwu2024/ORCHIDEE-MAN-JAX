#!/bin/bash
# Recipe only. Root supplies the accepted commit after committing; no paid job.
set -Eeuo pipefail
CODE_SHA=${1:?accepted full code SHA required}
BUNDLE=${2:?incremental bundle absolute path required}
SOURCE_REPO=${3:?immutable v3 repository path required}
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-gpu-metadata-v1
[[ $CODE_SHA =~ ^[0-9a-f]{40}$ ]] || exit 2
for path in "$BUNDLE" "$SOURCE_REPO" "$CODE_DIR"; do
  case "$(realpath -m "$path")" in /WORK/liwei_work/jcwu/*) ;; *) exit 2 ;; esac
done
[[ $(realpath -m "$SOURCE_REPO") == "$PROJECT/runtime/worktrees/global-throughput-gpu-sweetspot-v3" ]] || exit 2
[[ -d $SOURCE_REPO/.git ]] || { echo 'immutable v3 standalone Git directory required' >&2; exit 2; }
[[ ! -e $CODE_DIR ]] || { echo 'new immutable CODE_DIR already exists' >&2; exit 2; }
(
  cd "$SOURCE_REPO"
  [[ $(git rev-parse HEAD) == 12b495b9425de3ad1dc16efa3a5a0e69d7e89222 ]] || exit 2
)
# Confirmed remote Git1.8.3.1: independent clone, no hardlinks or shared-object
# dependency. Every fetch/checkout writes only the NEW repository.
git clone --no-hardlinks --no-checkout "$SOURCE_REPO" "$CODE_DIR"
(
  cd "$CODE_DIR"
  git bundle verify "$BUNDLE"
  git fetch "$BUNDLE" HEAD
  [[ $(git rev-parse FETCH_HEAD) == "$CODE_SHA" ]] || exit 2
  git checkout --detach "$CODE_SHA"
)
(
  cd "$CODE_DIR"
  [[ $(git rev-parse HEAD) == "$CODE_SHA" ]] || exit 2
  [[ -z $(git status --porcelain --untracked-files=all) ]] || exit 2
)
printf 'deployed_new_immutable_checkout=%s code_sha=%s\n' "$CODE_DIR" "$CODE_SHA"
