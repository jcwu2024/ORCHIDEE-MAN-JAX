"""One-off exact six-file package; does not modify the locked production builder."""
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT))
from scripts.dev.prepare_global_cpu48_execution import git,science_identity
from scripts.dev.global_server_reference import bind_cpu_reference_admission

BASE='60dccbddde10be4295aab7e7fe55fa7864d3c3f3'
SOURCE='978946bbf90593b5ea1c93d414dc6410a82b3f59'
REF='codex/global-cpu48-b64-v5'
P='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX'
TARGET=P+'/runtime/worktrees/global-throughput-cpu48-b64-v5'
REMOTE=P+'/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014'
PATHS=sorted(['scripts/dev/benchmark_global_batch.py','scripts/dev/benchmark_global_server_cpu.py',
    'scripts/dev/diagnose_global_cpu_references.py','tests/unit/test_global_cpu_reference_preparation.py',
    'tests/unit/test_global_server_benchmark.py','tests/unit/test_global_server_exploratory_cost.py'])

def rev(spec): return git(ROOT,'rev-parse',spec).decode().strip()
def sha(data): return hashlib.sha256(data).hexdigest()
assert rev('HEAD')==SOURCE
assert git(ROOT,'branch','--show-current').decode().strip()=='research/global-throughput'
assert not git(ROOT,'status','--porcelain','--untracked-files=all').strip()
assert not subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+REF],cwd=ROOT).returncode==0
science=science_identity(ROOT,BASE)
assert science=={'science_sha256':'11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5',
    'actual_science_source_sha256':'b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8',
    'science_file_count':104,'science_tree_oid':'75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba'}
patch=git(ROOT,'diff','--binary','--no-ext-diff',BASE,SOURCE,'--',*PATHS)
with tempfile.TemporaryDirectory(prefix='orchjax-fixed-cpu48-index-') as temporary:
    env={**os.environ,'GIT_INDEX_FILE':str(Path(temporary)/'index')}
    git(ROOT,'read-tree',BASE,env=env)
    git(ROOT,'apply','--cached','--whitespace=error',env=env,data=patch)
    tree=git(ROOT,'write-tree',env=env).decode().strip()
changed=git(ROOT,'diff','--name-only',BASE,tree).decode().splitlines()
assert changed==PATHS
assert rev(tree+':jax_orchidee')==science['science_tree_oid']
assert not git(ROOT,'diff','--raw',BASE,tree,'--','jax_orchidee').strip()
for path in PATHS:
    assert git(ROOT,'show',tree+':'+path)==git(ROOT,'show',SOURCE+':'+path)
message=f'Package accepted CPU reference identity projection repair\n\nSource: {SOURCE}\nBase: {BASE}\nSix harness/test files only; all104 physical blobs unchanged.\nNo paid execution authorization.\n'
execution=git(ROOT,'commit-tree',tree,'-p',BASE,data=message.encode()).decode().strip()
git(ROOT,'update-ref','refs/heads/'+REF,execution,'0'*40)
assert rev(execution+'^')==BASE
receipt=OUT.parent/'cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-audit-receipt-v3.json'
receipt_sha='cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431'
candidate={'job_id':None,'execution_commit':execution,'execution_code_dir':TARGET,
    'prepared_input_path':P+'/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl',
    'prepared_input_sha256':'be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7',
    'gpu_reference_path':P+'/runtime/outputs/research/global_throughput/gpu-round4/model/stage_B4_group_references.pkl',
    'gpu_reference_sha256':'a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6',
    'certified_science_sha256':science['science_sha256'],
    'actual_science_source_sha256':science['actual_science_source_sha256'],
    'physical_science_tree_oid':science['science_tree_oid'],'baseline_ref':'133e35aec7b83b9d79888c46092f64d518e97299'}
binding=bind_cpu_reference_admission(receipt,OUT/'scalar-admission.json',candidate,expected_receipt_sha256=receipt_sha)
admission=json.loads((OUT/'scalar-admission.json').read_bytes())
old=json.loads((OUT.parent/'cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json').read_bytes())
assert admission['scalar_source']==old['scalar_source'] and admission['audit']==old['audit']
assert admission['candidate_identity']==candidate
bundle=OUT/f'cpu48-b64-{execution[:12]}.bundle'
git(ROOT,'bundle','create',str(bundle),'refs/heads/'+REF,'^'+BASE)
git(ROOT,'bundle','verify',str(bundle))
heads=git(ROOT,'bundle','list-heads',str(bundle)).decode().strip()
assert heads==execution+' refs/heads/'+REF
(OUT/'execution-harness.patch').write_bytes(patch)
pin={'schema':'cpu48-b64-fixed-identity-candidate-v1','status':'prepared_not_paid_authorized',
    'source_research_commit':SOURCE,'base_execution_commit':BASE,'execution_code_commit':execution,
    'execution_ref':REF,'target_checkout':TARGET,'scientific_source':science,
    'research_science_tree_oid':rev(SOURCE+':jax_orchidee'),
    'changed_paths':changed,'every_physical_blob_identical_to_base':True,
    'harness_blob_sha256':{p:sha(git(ROOT,'show',execution+':'+p)) for p in PATHS},
    'bundle':{'name':bundle.name,'sha256':sha(bundle.read_bytes()),'bytes':bundle.stat().st_size,
              'prerequisite':BASE,'verify':'passed'},
    'patch':{'name':'execution-harness.patch','sha256':sha(patch),'bytes':len(patch)},
    'scalar_admission':{'name':'scalar-admission.json','remote_path':REMOTE+'/scalar-admission.json',
        'sha256':binding['sha256'],'bytes':(OUT/'scalar-admission.json').stat().st_size,'candidate_identity':candidate,
        'source_job_id':'15820467','source_commit':'03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8',
        'checkpoint_sha256':'1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa',
        'audit_receipt_path':str(receipt),'audit_receipt_sha256':receipt_sha,'new_audit':False,
        'scalar_source_and_audit_equal_original_admission':True},
    'run_root_pattern':P+'/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/${SLURM_JOB_ID}',
    'paid_submitted':False,'scientific_acceptance':False}
assert pin['research_science_tree_oid']=='62e90547ee8eadab387713397bb111c399c81b6c'
(OUT/'execution-code-pin.json').write_text(json.dumps(pin,indent=2)+'\n',encoding='utf-8',newline='\n')
assert rev('HEAD')==SOURCE and git(ROOT,'branch','--show-current').decode().strip()=='research/global-throughput'
print(json.dumps({'execution_commit':execution,'ref':REF,'bundle':pin['bundle'],'admission_sha256':binding['sha256']}))
