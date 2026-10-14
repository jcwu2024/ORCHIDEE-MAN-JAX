set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014
R=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/15857765
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v5
S=/rmprog/slurm/v22.05.7/bin
T=$E/terminal
test ! -e "$T"
mkdir "$T"
date -Is > "$T/captured-at.txt"
"$S/sacct" -j 15857765 --noheader --parsable2 --format=JobIDRaw,State,Elapsed,AllocTRES > "$T/all-parent-format.sacct.raw"
awk -F'|' '$1=="15857765"' "$T/all-parent-format.sacct.raw" > "$T/terminal-parent.sacct.raw"
test "$(wc -l < "$T/terminal-parent.sacct.raw")" -eq 1
"$S/sacct" -j 15857765 --parsable2 --units=K --format=JobIDRaw,JobName,Partition,State,ExitCode,Submit,Start,End,Elapsed,ElapsedRaw,Timelimit,AllocCPUS,AllocTRES,ReqTRES,NodeList,TotalCPU,UserCPU,SystemCPU,CPUTimeRAW,MaxRSS > "$T/terminal-full.sacct.raw"
"$S/scontrol" show job -dd 15857765 > "$T/terminal.scontrol.stdout" 2> "$T/terminal.scontrol.stderr" || printf '%s\n' "$?" > "$T/terminal.scontrol.exitcode"
cp "$P/runtime/logs/global_cpu48_b64_v5_15857765.txt" "$T/slurm-parent.log"
cd "$CODE"
git rev-parse HEAD HEAD:jax_orchidee > "$T/postrun-checkout-identity.txt"
git status --porcelain --untracked-files=all > "$T/postrun-checkout-status.txt"
test ! -s "$T/postrun-checkout-status.txt"
test "$(git rev-parse HEAD)" = 0100c8b09be5137b822173e0a0be66cb64939727
sha256sum scripts/dev/collect_global_cpu48_b64.py > "$T/collector-source.sha256"
test "$(cut -d' ' -f1 "$T/collector-source.sha256")" = efcb32efa754bbdbb75936f358dc6cb5a286e6742948096bda3e3ba22d27a9e3
cat "$T/terminal-full.sacct.raw"
tail -c 6000 "$T/slurm-parent.log"
tail -c 8000 "$R/batch.log"
sha256sum "$R/orchestration.json" "$R/batch.log" "$R/batch/report.json" "$T/terminal-parent.sacct.raw" "$T/terminal-full.sacct.raw"
