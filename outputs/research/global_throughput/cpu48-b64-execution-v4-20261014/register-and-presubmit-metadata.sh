set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
T=$P/runtime/transfer/global-cpu48-b64-v4-60dccbddde10
D=$P/runtime/deployment
test "$(sha256sum "$D/registry.json" | cut -d' ' -f1)" = f453142af6835d71a77bc56e05ba2a1c9a4ab06585c7558086d1e409c82373ae
test "$(sha256sum "$T/registry-deployed.json" | cut -d' ' -f1)" = 0cb5e63c2a518d6adc2749b6bcf85c797032439286909dc11930369be827ff77
test ! -e "$D/registry.before-cpu48-v4-deployment.json"
cp "$D/registry.json" "$D/registry.before-cpu48-v4-deployment.json"
cp "$T/registry-deployed.json" "$D/registry.json.next-cpu48-v4"
mv "$D/registry.json.next-cpu48-v4" "$D/registry.json"
cat >>"$D/README.md" <<'README'

## CPU48 v4 approved immutable deployment, 2026-10-14

The separately approved one-attempt batch-only CPU48 v4 checkout is
`runtime/worktrees/global-throughput-cpu48-b64-v4` at
`60dccbddde10be4295aab7e7fe55fa7864d3c3f3`, science tree
`75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba` unchanged from v3.
The prerequisite bundle was imported without force into the bare store;
the new clone is detached/clean with no Git alternates. Existing checkouts,
scientific assets, main/daily/research refs and old execution records remain.
Real production-path identity/reference and exact launcher dry-run passed in
one-CPU free job 15853600, with no JAX/model execution. This is not CPU48
physical admission; the sole cnall 48CPU/0GPU/3600s job must pass its real gates.
The exact original proposal and persistent scalar admission are registered;
failure/timeout closes this attempt without retry. Runtime evidence is under
`runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014`.
README
sha256sum "$D/registry.json" "$D/README.md"
date -Is
hostname
/rmprog/slurm/v22.05.7/bin/sacct -j 15853600 --noheader --parsable2 --format=JobID,Partition,State,ExitCode,Elapsed,AllocCPUS,NodeList
/rmprog/slurm/v22.05.7/bin/scontrol show partition cnall
/rmprog/slurm/v22.05.7/bin/sinfo -N -p cnall -o '%N %P %t %c %C'
/rmprog/slurm/v22.05.7/bin/squeue -p cnall -o '%.18i %.9P %.12j %.8u %.2t %.10M %.6D %R'
cd "$P/runtime/worktrees/global-throughput-cpu48-b64-v4"
git rev-parse HEAD HEAD:jax_orchidee
test -z "$(git status --porcelain --untracked-files=all)"
printf 'PRE_SUBMIT_METADATA=passed\n'
