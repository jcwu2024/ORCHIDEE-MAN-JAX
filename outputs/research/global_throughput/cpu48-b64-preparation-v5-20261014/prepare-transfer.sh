set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OUT=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014
T=$P/runtime/transfer/global-cpu48-b64-v5-0100c8b09be5
test ! -e "$OUT"
test ! -e "$T"
test ! -e "$P/runtime/worktrees/global-throughput-cpu48-b64-v5"
mkdir "$OUT" "$T"
cd "$P/runtime/deployment/repository.git"
git cat-file -e 60dccbddde10be4295aab7e7fe55fa7864d3c3f3^{commit}
test -z "$(git show-ref --verify --quiet refs/heads/codex/global-cpu48-b64-v5 && printf exists || true)"
sha256sum "$P/runtime/deployment/registry.json"
cat "$P/runtime/deployment/registry.json"
/rmprog/slurm/v22.05.7/bin/scontrol show partition cntest
date -Is
