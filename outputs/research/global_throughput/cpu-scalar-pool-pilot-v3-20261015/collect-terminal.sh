#!/bin/bash
set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015
R=$P/runtime/outputs/research/global_throughput/server-benchmark/cpu4-15869035
C=$P/runtime/worktrees/global-throughput-cpu-scalar-pool-v3
S=/rmprog/slurm/v22.05.7/bin
[[ $(realpath -m "$E") == "$P/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015" ]]
date -u +%FT%TZ > "$E/collection-utc.txt"
"$S/scontrol" show job --details 15869035 > "$E/terminal-scontrol.txt"
"$S/scontrol" show node ibc11b02n24 > "$E/terminal-node-snapshot.txt"
"$S/sacct" -j 15869035 --noheader --parsable2 --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,TotalCPU,UserCPU,SystemCPU,CPUTimeRAW,MaxRSS,AveRSS,Start,End,Submit,ReqMem,ReqCPUS,NTasks,NNodes > "$E/terminal-sacct-all.psv"
awk -F'|' '$1 == "15869035"' "$E/terminal-sacct-all.psv" > "$E/terminal-sacct-parent.psv"
cp "$P/runtime/logs/global_scalar_pool_15869035.txt" "$E/job-15869035.raw.log"
cp "$R/launch-context.txt" "$E/launch-context.raw.txt"
find "$R" -printf '%y %P %s bytes\n' > "$E/run-root-inventory.txt"
cd "$C"
git rev-parse HEAD > "$E/terminal-code-head.txt"
git status --porcelain --untracked-files=all > "$E/terminal-code-status.txt"
"$P/.venvs/orcjax_cpu/bin/python" -B - "$E" "$R" "$P" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
e, run_root, project = map(Path, sys.argv[1:])
paths = ['pool4', 'scalar-worker-cache', 'pool4/workers', 'pool4/report.json',
         'pool4/stage_manifest.json', 'pool4/raw_B64_pool_before_parity.pkl']
receipt = {'schema': 'global_scalar_pool_v3_terminal_inventory_v1',
    'utc': datetime.datetime.utcnow().isoformat() + 'Z', 'job_id': '15869035',
    'original_run_root': str(run_root),
    'original_job_log': str(project / 'runtime/logs/global_scalar_pool_15869035.txt'),
    'root_entries': sorted(p.name for p in run_root.iterdir()),
    'expected_artifact_existence': {name: (run_root / name).exists() for name in paths},
    'scientific_arrays_opened': False, 'physical_model_executed': False,
    'files': []}
for path in sorted(e.iterdir()):
    if path.is_file() and path.name != 'terminal-inventory.json':
        receipt['files'].append({'name': path.name, 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
(e / 'terminal-inventory.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt, sort_keys=True))
PY
cat "$E/terminal-sacct-all.psv"
cat "$E/terminal-node-snapshot.txt"
