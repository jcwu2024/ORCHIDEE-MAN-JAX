set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OUT=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014
test ! -e "$OUT/free-check/15857559/sacct-terminal.txt"
/rmprog/slurm/v22.05.7/bin/sacct -j 15857559 --parsable2 --units=K --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Timelimit,AllocCPUS,NodeList > "$OUT/free-check/15857559/sacct-terminal.txt"
cat "$OUT/free-check/15857559/sacct-terminal.txt"
ls -l "$P/runtime/deployment"
if [[ -f "$P/runtime/deployment/README.md" ]]; then
  sha256sum "$P/runtime/deployment/README.md"
  cat "$P/runtime/deployment/README.md"
fi
sha256sum "$P/runtime/deployment/registry.json"
