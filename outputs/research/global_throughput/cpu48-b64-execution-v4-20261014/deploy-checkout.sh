set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
T=$P/runtime/transfer/global-cpu48-b64-v4-60dccbddde10
BARE=$P/runtime/deployment/repository.git
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v4
ADMISSION=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json
date -Is
hostname
sha256sum -c <<HASHES
970fc5be9bf3f59061b27ad6647d457e824392ce14336a69ff0308e3c91b928e  $T/cpu48-b64-60dccbddde10.bundle
e9e301aeab7237d59a0e204ffc93fc3eee96d28ab997d89be31383d735ec2769  $T/execution-code-pin.json
b86068c62c96464319175135e12e5d84ce2e494a00875aceaa01ee0b1a48c160  $T/PROPOSED_SUBMISSION.sh
f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d  $ADMISSION
HASHES
wc -c "$T/cpu48-b64-60dccbddde10.bundle" "$T/execution-code-pin.json" "$T/PROPOSED_SUBMISSION.sh" "$ADMISSION"
git --git-dir="$BARE" bundle verify "$T/cpu48-b64-60dccbddde10.bundle"
git --git-dir="$BARE" bundle list-heads "$T/cpu48-b64-60dccbddde10.bundle"
test "$(git --git-dir="$BARE" bundle list-heads "$T/cpu48-b64-60dccbddde10.bundle")" = '60dccbddde10be4295aab7e7fe55fa7864d3c3f3 refs/heads/codex/global-cpu48-b64-v4'
git --git-dir="$BARE" cat-file -t 03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8
git --git-dir="$BARE" fetch "$T/cpu48-b64-60dccbddde10.bundle" refs/heads/codex/global-cpu48-b64-v4:refs/heads/codex/global-cpu48-b64-v4
test "$(git --git-dir="$BARE" rev-parse refs/heads/codex/global-cpu48-b64-v4)" = 60dccbddde10be4295aab7e7fe55fa7864d3c3f3
test ! -e "$CODE"
git clone --no-hardlinks --no-checkout "$BARE" "$CODE"
cd "$CODE"
git checkout --detach 60dccbddde10be4295aab7e7fe55fa7864d3c3f3
test "$(git rev-parse HEAD)" = 60dccbddde10be4295aab7e7fe55fa7864d3c3f3
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git rev-parse HEAD:jax_orchidee)" = 75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba
test "$(git rev-parse 03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8:jax_orchidee)" = 75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba
git show-ref
git rev-parse HEAD HEAD:jax_orchidee
git status --porcelain --untracked-files=all
if test -e .git/objects/info/alternates; then cat .git/objects/info/alternates; else echo ALTERNATES=none; fi
printf 'DEPLOYMENT=clean_exact_checkout\n'
