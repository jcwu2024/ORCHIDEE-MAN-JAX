set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OUT=$P/runtime/outputs/research/global_throughput/cpu-reference-identity-repair-20261014
/rmprog/slurm/v22.05.7/bin/sacct -j 15855041,15855062 --parsable2 --format=JobIDRaw,Partition,State,ExitCode,Elapsed,Timelimit,AllocCPUS,AllocTRES,NodeList > "$OUT/replay-terminal.sacct.raw"
cat "$OUT/replay-terminal.sacct.raw"
cd "$P/runtime/worktrees/global-throughput-cpu48-b64-v4"
git rev-parse HEAD HEAD:jax_orchidee
test -z "$(git status --porcelain --untracked-files=all)"
cd "$P/runtime/diagnostics/global-cpu-reference-identity-20261014"
git diff --name-only -- jax_orchidee
test -z "$(git diff --name-only -- jax_orchidee)"
sha256sum "$OUT/before/report.json" "$OUT/after/report.json" "$OUT/original-admission.json" "$OUT/diagnostic-admission.json"
