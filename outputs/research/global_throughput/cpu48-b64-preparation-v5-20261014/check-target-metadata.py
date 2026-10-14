"""Small immutable-package and actual dry-run checks on the allocated test node."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
OUT=Path(sys.argv[1]); CODE=Path(sys.argv[2])
pin=json.loads((OUT/'execution-code-pin.json').read_bytes())
def git(*args): return subprocess.check_output(['git',*args],cwd=CODE)
assert git('rev-parse','HEAD').decode().strip()==pin['execution_code_commit']
assert not git('status','--porcelain','--untracked-files=all').strip()
assert not (CODE/'.git/objects/info/alternates').exists()
hashes={}
for path,expected in pin['harness_blob_sha256'].items():
    raw=(CODE/path).read_bytes(); assert raw==git('show','HEAD:'+path)
    actual=hashlib.sha256(raw).hexdigest(); assert actual==expected; hashes[path]=actual
admission=json.loads((OUT/'scalar-admission.json').read_bytes())
assert admission['candidate_identity']==pin['scalar_admission']['candidate_identity']
assert hashlib.sha256((OUT/'scalar-admission.json').read_bytes()).hexdigest()==pin['scalar_admission']['sha256']
assert admission['candidate_identity']['execution_code_dir']==str(CODE)
assert admission['candidate_identity']['execution_commit']==pin['execution_code_commit']
run=OUT/'free-check'/os.environ['SLURM_JOB_ID']
line=next(s for s in (run/'launcher-dry-run.stdout').read_text().splitlines() if s.startswith('coordinator='))
import shlex
argv=shlex.split(line[len('coordinator='):])
(run/'coordinator-actual-argv.json').write_text(json.dumps(argv,indent=2)+'\n')
with (run/'coordinator-dry-run.stdout').open('w') as stdout,(run/'coordinator-dry-run.stderr').open('w') as stderr:
    result=subprocess.run(argv,stdout=stdout,stderr=stderr,check=False)
assert result.returncode==0
dry=json.loads((run/'coordinator-dry-run.stdout').read_bytes())
assert dry['execution_route']=='batch-only' and dry['expected_execution_commit']==pin['execution_code_commit']
assert dry['commands']['batch']['argv'][0]==str(CODE/'scripts/dev/benchmark_global_server_cpu.py') or 'benchmark_global_server_cpu.py' in ' '.join(dry['commands']['batch']['argv'])
assert dry['model_execution'] is False and dry['JAX_imported'] is False
assert dry['runner_cli_dry_runs']['batch']['runner_result']['model_execution'] is False
assert dry['runner_cli_dry_runs']['batch']['runner_result']['CPU_budget']==48
summary={'schema':'cpu48_v5_package_and_argument_check_v1','harness_blob_sha256':hashes,
    'actual_free_cpus':int(os.environ['SLURM_CPUS_PER_TASK']),'actual_free_affinity':sorted(os.sched_getaffinity(0)),
    'execution_commit':pin['execution_code_commit'],'code_dir':str(CODE),'checkout_clean':True,
    'proposal_sha256':hashlib.sha256((OUT/'PROPOSED_SUBMISSION.sh').read_bytes()).hexdigest(),
    'normal_admission_binding_passed':True,'launcher_coordinator_runner_argument_expansion_passed':True,
    'dry_run_scope':'planned48 arguments only; no actual48 resource admission or model execution',
    'planned_run_root':argv[argv.index('--output-dir')+1],'paid_submission':False,'JAX_imported':False}
expected=pin['run_root_pattern'].replace('${SLURM_JOB_ID}',os.environ['SLURM_JOB_ID'])
assert summary['planned_run_root']==expected
assert summary['actual_free_cpus']==1 and len(summary['actual_free_affinity'])==1
(run/'package-and-arguments.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
