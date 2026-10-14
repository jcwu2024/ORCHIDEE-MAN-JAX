set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014
R=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/15857765
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v5
PY=$P/.venvs/orcjax_cpu/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
unset XLA_FLAGS
test "$(sha256sum "$CODE/scripts/dev/collect_global_cpu48_b64.py" | cut -d' ' -f1)" = efcb32efa754bbdbb75936f358dc6cb5a286e6742948096bda3e3ba22d27a9e3
cd "$CODE"
"$PY" -B - "$E" "$R" <<'PY'
import hashlib,json,os,socket,sys
from pathlib import Path
e,run=map(Path,sys.argv[1:]); raw=(e/'terminal/terminal-parent.sacct.raw').read_bytes()
job,state,elapsed,tres=raw.decode().rstrip('\n').split('|')
assert job=='15857765' and state=='FAILED' and elapsed=='00:13:39'
assert os.environ['SLURM_JOB_PARTITION']=='cntest' and os.environ['SLURM_CPUS_PER_TASK']=='1'
assert os.environ['SLURM_NTASKS']=='1' and len(os.sched_getaffinity(0))==1
accounting={'source':'sacct','scope':'whole_slurm_parent_job','job_id':job,
 'terminal_status':state,'Elapsed':elapsed,'AllocTRES':tres,'wall_seconds':819,
 'allocated_CPU_count':48,'GPU_count':0,'directories':[str(run/'batch')],
 'raw_record':raw.decode(),'raw_record_sha256':hashlib.sha256(raw).hexdigest()}
with (e/'terminal/parent-accounting.json').open('x') as f: json.dump(accounting,f,indent=2); f.write('\n')
receipt={'schema':'global_cpu48_terminal_collector_execution_v1','job_id':os.environ['SLURM_JOB_ID'],
 'node':socket.gethostname(),'partition':os.environ['SLURM_JOB_PARTITION'],
 'cpus_per_task':os.environ['SLURM_CPUS_PER_TASK'],'affinity':sorted(os.sched_getaffinity(0)),
 'canonical_python':sys.executable,'code_dir':str(Path.cwd()),'model_execution':False,
 'collector_sha256':hashlib.sha256((Path.cwd()/'scripts/dev/collect_global_cpu48_b64.py').read_bytes()).hexdigest()}
with (e/'terminal/collector-execution-receipt.json').open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
print(json.dumps(receipt))
PY
set +e
"$PY" -B "$CODE/scripts/dev/collect_global_cpu48_b64.py" --run-root "$R" --output "$E/collector-result.json" --parent-accounting "$E/terminal/parent-accounting.json" > "$E/terminal/collector.stdout.raw" 2> "$E/terminal/collector.stderr.raw"
collector_exit=$?
set -e
printf '%s\n' "$collector_exit" > "$E/terminal/collector.exitcode.raw"
cat "$E/terminal/collector.stdout.raw"
cat "$E/terminal/collector.stderr.raw" >&2
test "$collector_exit" = 0
sha256sum "$E/collector-result.json" "$E/terminal/parent-accounting.json"
test -z "$(git status --porcelain --untracked-files=all)"
date -Is > "$E/terminal/collector-complete.txt"
