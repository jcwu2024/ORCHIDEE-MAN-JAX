set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014
D=$P/runtime/deployment
test "$(sha256sum "$D/registry.json" | cut -d' ' -f1)" = 0cb5e63c2a518d6adc2749b6bcf85c797032439286909dc11930369be827ff77
test "$(sha256sum "$E/registry-terminal.json" | cut -d' ' -f1)" = a9ec87f475effcb5cc1a2afe9489566ce32bb310c27491565ed442fb15535b62
test "$(sha256sum "$E/CURRENT_RESULT.md" | cut -d' ' -f1)" = caf713cba625876d50298fb16cf11cd89c68607379c50b1cb9e9dc6e5e4c3ae7
test "$(sha256sum "$E/runtime-evidence-manifest.json" | cut -d' ' -f1)" = 5603c0d6b9cb8046c0966dfe0cf0cb08870a502488f73b12a750f4ee646f9515
test "$(sha256sum "$E/identity-and-child-receipts.json" | cut -d' ' -f1)" = 287b18364a89e23582cb8c89e30e8bc65580ab503f24ed8806b1b265ad22e6f1
test ! -e "$D/registry.before-cpu48-v4-terminal.json"
cp "$D/registry.json" "$D/registry.before-cpu48-v4-terminal.json"
cp "$E/registry-terminal.json" "$D/registry.json.next-cpu48-v4-terminal"
mv "$D/registry.json.next-cpu48-v4-terminal" "$D/registry.json"
cat >> "$D/README.md" <<'README'

CPU48 v4 terminal update: the one authorized paid submission was job 15853683,
cnall/48CPU/0GPU, FAILED exit 1:0 after50s on ibc11b09n36. Actual child CPU48
resource admission passed; the real runner then stopped on scalar-admission
candidate provenance/current-identity mismatch before any model measurement.
Pinned failure collector ran on free one-CPU job15853855, preserved failure,
and validated whole-parent cost scenarios CNY0.033333333/0.046666667 separately
from old scalar job15820467 (2197s). Billing remains unverified. No resubmit,
repair or throughput claim. See runtime evidence CURRENT_RESULT.md and manifest.
README
/rmprog/slurm/v22.05.7/bin/sacct -j 15853600,15853855 --parsable2 --format=JobIDRaw,Partition,State,ExitCode,Elapsed,AllocCPUS,NodeList > "$E/terminal/free-stage-terminal.sacct.raw"
cd "$P/runtime/worktrees/global-throughput-cpu48-b64-v4"
git rev-parse HEAD HEAD:jax_orchidee
test -z "$(git status --porcelain --untracked-files=all)"
sha256sum "$D/registry.json" "$D/registry.before-cpu48-v4-terminal.json" "$D/README.md" "$E/CURRENT_RESULT.md" "$E/runtime-evidence-manifest.json"
cat "$E/terminal/free-stage-terminal.sacct.raw"
date -Is
printf 'TERMINAL_METADATA_REGISTERED=passed\n'
