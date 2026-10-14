set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
T=$P/runtime/transfer/global-cpu48-b64-v4-60dccbddde10
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v4
OUT=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/preflight/$SLURM_JOB_ID
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1
hostname
date -Is
/rmprog/slurm/v22.05.7/bin/scontrol show job -o "$SLURM_JOB_ID" >"$OUT/scontrol-free-job.txt"
cd "$CODE"
"$P/.venvs/orcjax_cpu/bin/python" -B "$T/verify-production.py" >"$OUT/production-preflight.stdout.txt" 2>"$OUT/production-preflight.stderr.txt"
bash -n "$T/PROPOSED_SUBMISSION.sh"
bash -n "$CODE/scripts/hpc/slurm_global_server_cpu_48core.sh"
for cpu48_prefix in SBATCH_ ORCHJAX_; do
  while IFS= read -r cpu48_variable; do unset "$cpu48_variable"; done < <(compgen -v "$cpu48_prefix")
done
unset SLURM_BIN CODE_DIR PYTHON ENTRY COORDINATOR PREPARED_INPUT GPU_REFERENCE_DIR CONFIG
unset CPU_REFERENCE_DIR CPU_REFERENCE_ADMISSION_RECORD EXPECTED_CPU_REFERENCE_ADMISSION_SHA256
unset RUN_ROOT EXPECTED_CODE_COMMIT EVIDENCE_DIR ORCHIDEE_REPO_ROOT ORCHIDEE_DATA_ROOT
test "$SLURM_CPUS_PER_TASK" = 1
unset SLURM_CPUS_PER_TASK
export CODE_DIR=$CODE
export PYTHON=$P/.venvs/orcjax_cpu/bin/python
export EXPECTED_CODE_COMMIT=60dccbddde10be4295aab7e7fe55fa7864d3c3f3
export CPU_REFERENCE_DIR=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar
export CPU_REFERENCE_ADMISSION_RECORD=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json
export EXPECTED_CPU_REFERENCE_ADMISSION_SHA256=f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d
bash "$CODE/scripts/hpc/slurm_global_server_cpu_48core.sh" --dry-run >"$OUT/launcher-dry-run.stdout.txt" 2>"$OUT/launcher-dry-run.stderr.txt"
"$PYTHON" -B - "$OUT" <<'PY'
import json,shlex,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1])
line=next(x for x in (out/'launcher-dry-run.stdout.txt').read_text().splitlines() if x.startswith('coordinator='))
argv=shlex.split(line[len('coordinator='):])
(out/'coordinator-actual-argv.json').write_text(json.dumps(argv,indent=2)+'\n')
with (out/'coordinator-dry-run.stdout.txt').open('w') as stdout,(out/'coordinator-dry-run.stderr.txt').open('w') as stderr:
    result=subprocess.run(argv,stdout=stdout,stderr=stderr,check=False)
if result.returncode: raise SystemExit(result.returncode)
receipt=json.loads((out/'coordinator-dry-run.stdout.txt').read_text())
assert receipt['expected_execution_commit']=='60dccbddde10be4295aab7e7fe55fa7864d3c3f3'
assert receipt['execution_route']=='batch-only' and receipt['allocation']['wall_seconds']==3600
assert receipt['JAX_imported'] is False and receipt['model_execution'] is False
assert receipt['runner_cli_dry_runs']['batch']['runner_result']['actual_child_prejax_receipt']['affinity']==[0] or len(receipt['runner_cli_dry_runs']['batch']['runner_result']['actual_child_prejax_receipt']['affinity'])==1
assert 'jax' not in sys.modules
(out/'preflight-complete.txt').write_text('real production paths and computed identity/seal passed; no JAX/model execution\n')
print(json.dumps({'free_job_id':out.name,'production_preflight':'passed','no_path_substitution':True}))
PY
test -z "$(git status --porcelain --untracked-files=all)"
date -Is
