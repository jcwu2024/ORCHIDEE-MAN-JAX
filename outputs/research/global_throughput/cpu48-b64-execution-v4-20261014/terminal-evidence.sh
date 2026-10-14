set -euo pipefail
S=/rmprog/slurm/v22.05.7/bin
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014
R=$E/15853683
T=$E/terminal
test ! -e "$T"
mkdir "$T"
date -Is > "$T/captured-at.txt"
"$S/sacct" -j 15853683 --noheader --parsable2 --format=JobIDRaw,State,Elapsed,AllocTRES > "$T/all-parent-format.sacct.raw"
awk -F'|' '$1=="15853683"' "$T/all-parent-format.sacct.raw" > "$T/terminal-parent.sacct.raw"
test "$(wc -l < "$T/terminal-parent.sacct.raw")" -eq 1
"$S/sacct" -j 15853683 --parsable2 --units=K --format=JobIDRaw,JobName,Partition,State,ExitCode,Submit,Start,End,Elapsed,ElapsedRaw,Timelimit,AllocCPUS,AllocTRES,ReqTRES,NodeList,TotalCPU,UserCPU,SystemCPU,CPUTimeRAW,MaxRSS > "$T/terminal-full.sacct.raw"
cp "$E/submission-invoked.txt" "$E/submission.stdout.raw" "$E/submission.stderr.raw" "$E/submission.exitcode.raw" "$E/presubmit-node-facts.txt" "$E/presubmit-queue.txt" "$T/"
cp "$P/runtime/logs/global_cpu48_b64_v4_15853683.txt" "$T/slurm-parent.log"
cd "$P/runtime/worktrees/global-throughput-cpu48-b64-v4"
git rev-parse HEAD HEAD:jax_orchidee > "$T/postrun-checkout-identity.txt"
git status --porcelain --untracked-files=all > "$T/postrun-checkout-status.txt"
test ! -s "$T/postrun-checkout-status.txt"
sha256sum scripts/dev/collect_global_cpu48_b64.py > "$T/collector-source.sha256"
test "$(cut -d' ' -f1 "$T/collector-source.sha256")" = efcb32efa754bbdbb75936f358dc6cb5a286e6742948096bda3e3ba22d27a9e3
sha256sum "$R/orchestration.json" "$R/batch.log" "$R/batch/report.json" "$T/terminal-parent.sacct.raw" "$T/terminal-full.sacct.raw"
cat "$T/terminal-parent.sacct.raw" "$T/postrun-checkout-identity.txt"
