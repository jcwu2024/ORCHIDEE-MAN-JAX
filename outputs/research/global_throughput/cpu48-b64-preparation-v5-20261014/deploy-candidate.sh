set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OUT=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014
T=$P/runtime/transfer/global-cpu48-b64-v5-0100c8b09be5
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v5
test ! -e "$CODE"
test "$(sha256sum "$T/cpu48-b64-0100c8b09be5.bundle" | cut -d' ' -f1)" = 0668675de188c67c23ebacaf5f52a722458badbc2d9c2117bf18bca288ba91c1
test "$(wc -c < "$T/cpu48-b64-0100c8b09be5.bundle")" -eq 6638
test "$(sha256sum "$OUT/scalar-admission.json" | cut -d' ' -f1)" = 4d2d3d18d92c0d6c6bc737059bc2b2555cf36f77b4a3c4406844071ab6a3e7f5
cd "$P/runtime/deployment/repository.git"
git bundle verify "$T/cpu48-b64-0100c8b09be5.bundle"
test "$(git bundle list-heads "$T/cpu48-b64-0100c8b09be5.bundle")" = '0100c8b09be5137b822173e0a0be66cb64939727 refs/heads/codex/global-cpu48-b64-v5'
git fetch "$T/cpu48-b64-0100c8b09be5.bundle" refs/heads/codex/global-cpu48-b64-v5:refs/heads/codex/global-cpu48-b64-v5
git clone --no-hardlinks --no-checkout "$P/runtime/deployment/repository.git" "$CODE"
cd "$CODE"
git checkout --detach 0100c8b09be5137b822173e0a0be66cb64939727
test "$(git rev-parse HEAD)" = 0100c8b09be5137b822173e0a0be66cb64939727
test "$(git rev-parse HEAD^ )" = 60dccbddde10be4295aab7e7fe55fa7864d3c3f3
test "$(git rev-parse HEAD:jax_orchidee)" = 75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e .git/objects/info/alternates
git diff --name-only HEAD^ HEAD
git rev-parse HEAD HEAD:jax_orchidee
sha256sum "$OUT/execution-code-pin.json" "$OUT/scalar-admission.json" "$OUT/PROPOSED_SUBMISSION.sh"
cd "$P/runtime/worktrees/global-throughput-cpu48-b64-v4"
test -z "$(git status --porcelain --untracked-files=all)"
git rev-parse HEAD HEAD:jax_orchidee
printf 'DEPLOYMENT=clean_new_immutable_checkout_no_alternates\n'
