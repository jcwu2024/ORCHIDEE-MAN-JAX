import base64, json
from pathlib import Path
root=Path(__file__).resolve().parents[4]
files=['scripts/dev/global_source_compatibility.py','scripts/dev/benchmark_global_gpu_capacity.py',
       'scripts/hpc/slurm_global_gpu_capacity.sh','scripts/hpc/slurm_global_gpu_capacity_512.sh']
payload={p:base64.b64encode((root/p).read_bytes()).decode() for p in files}
runner='''import base64,json,sys,types,hashlib,subprocess,os,time
from pathlib import Path
started=time.perf_counter()
payload=json.loads(base64.b64decode(PAYLOAD))
project=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
old=project/'runtime/worktrees/global-throughput-gpu-sweetspot-v3'
diag=project/'runtime/worktrees/global-throughput-metadata-command-diagnostic-20261008'
out=project/'runtime/outputs/research/global_throughput/gpu-metadata-repair-20261008/canonical-command-probe'
sys.path.insert(0,str(old))
print(json.dumps({'phase':'command_probe','hashes':{p:hashlib.sha256(base64.b64decode(v)).hexdigest() for p,v in payload.items()},'old_ROOT_for_stdlib_source_admission':str(old),'JAX_imported':'jax' in sys.modules}),flush=True)
import scripts.dev
name='scripts.dev.global_source_compatibility'
compat=types.ModuleType(name); compat.__file__=str(old/'scripts/dev/global_source_compatibility.py')
exec(compile(base64.b64decode(payload['scripts/dev/global_source_compatibility.py']),compat.__file__,'exec'),compat.__dict__)
sys.modules[name]=compat
capacity=types.ModuleType('capacity_probe'); capacity.__file__=str(old/'scripts/dev/benchmark_global_gpu_capacity.py')
exec(compile(base64.b64decode(payload['scripts/dev/benchmark_global_gpu_capacity.py']),capacity.__file__,'exec'),capacity.__dict__)
for args in [['--help'], ['--dry-run','--continue-512','--budget-seconds','2700','--prepared-input','declared-prepared.pkl','--full-reference-dir','declared-reference','--memory-baseline-dir','declared-B64','--output-dir','declared-new-runroot','--b128-baseline-dir','sealed-B128','--b256-baseline-dir','sealed-B256']]:
    sys.argv=['canonical_capacity_snapshot']+args
    try: capacity.main()
    except SystemExit as e: assert e.code==0
for path in ['scripts/hpc/slurm_global_gpu_capacity.sh','scripts/hpc/slurm_global_gpu_capacity_512.sh']:
    target=diag/path; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(base64.b64decode(payload[path]).replace(bytes([13,10]),bytes([10])))
out.mkdir(parents=True,exist_ok=True)
spool=out/'slurm_script'
spool.write_bytes((diag/'scripts/hpc/slurm_global_gpu_capacity_512.sh').read_bytes())
env={**os.environ,'CODE_DIR':str(diag),'RUN_ROOT':str(project/'runtime/outputs/research/global_throughput/server-benchmark/gpu-metadata-command-dryrun-only')}
for label,path in [('direct',diag/'scripts/hpc/slurm_global_gpu_capacity_512.sh'),('spool',spool)]:
    result=subprocess.run(['bash',str(path),'--dry-run'],env=env,capture_output=True,text=True,timeout=15)
    print(json.dumps({'phase':'bash_dry_run','label':label,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr}),flush=True)
    assert result.returncode==0
    assert '--continue-512' in result.stdout and '--b256-baseline-dir' in result.stdout and 'wall=00:45:00' in result.stdout
    assert '--continue-256-512' not in result.stdout
print(json.dumps({'phase':'complete','seconds':time.perf_counter()-started,'JAX_imported':'jax' in sys.modules,'model_execution':False,'limitations':'streamed current argument/source-certificate code uses original immutable v3 sourceROOT for stdlib admission; launcher-only diagnostic directory is not deployment or real data/prepared-input admission'}),flush=True)
'''
encoded=base64.b64encode(json.dumps(payload).encode()).decode()
Path(__file__).with_name('canonical_command_probe.py').write_text('PAYLOAD='+repr(encoded)+'\n'+runner)
