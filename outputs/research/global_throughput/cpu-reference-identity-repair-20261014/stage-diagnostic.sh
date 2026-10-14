set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$P/runtime/diagnostics/global-cpu-reference-identity-20261014
OUT=$P/runtime/outputs/research/global_throughput/cpu-reference-identity-repair-20261014
test ! -e "$CODE"
test ! -e "$OUT"
mkdir -p "$P/runtime/diagnostics" "$OUT/before-overlay" "$OUT/after-overlay"
git clone --shared --no-checkout "$P/runtime/deployment/repository.git" "$CODE"
cd "$CODE"
git checkout --detach 60dccbddde10be4295aab7e7fe55fa7864d3c3f3
git rev-parse HEAD HEAD:jax_orchidee
test -z "$(git status --porcelain --untracked-files=all)"
cat .git/objects/info/alternates
cd "$P/runtime/worktrees/global-throughput-cpu48-b64-v4"
test -z "$(git status --porcelain --untracked-files=all)"
git rev-parse HEAD HEAD:jax_orchidee
