"""Extract compact evidence from the sole completed check; does not run a model."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name): return json.loads((ROOT/name).read_bytes())
def identity(name):
    raw=(ROOT/name).read_bytes()
    return {'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
pin=read('execution-code-pin.json'); report=read('free-check/15857559/report.json')
retention=read('registry-retention-receipt.json')
assert report['status']=='reference_preparation_passed'
assert report['output_leaf_counts']=={'GPU':760,'CPU_scalar':760}
assert report['actual_stage_runtime_git_head']==pin['execution_code_commit']
assert all(pair['candidate']==pair['stage'] for pair in report['reference_preparation_identity'].values())
assert not report['physical_runner_created'] and not report['physical_lower_compile_run']
summary={'schema':'global_cpu48_v5_preparation_result_v1','status':'prepared_deployed_free_checked_not_paid_approved',
 'execution_commit':pin['execution_code_commit'],'execution_ref':pin['execution_ref'],
 'parent_execution_commit':pin['base_execution_commit'],'source_research_commit':pin['source_research_commit'],
 'target_checkout':pin['target_checkout'],'scientific_source':pin['scientific_source'],
 'changed_paths':pin['changed_paths'],'scalar_admission':{
    k:pin['scalar_admission'][k] for k in ('sha256','source_job_id','source_commit','checkpoint_sha256','audit_receipt_sha256','new_audit','scalar_source_and_audit_equal_original_admission')},
 'free_check':{'job_id':'15857559','partition':'cntest','node':'ibtest02','actual_cpus':1,
    'affinity':report['diagnostic_allocation']['affinity'],'limit_seconds':300,
    'sacct_state':'COMPLETED','sacct_exit_code':'0:0','sacct_parent_elapsed_seconds':65,
    'status':report['status'],'workload_points':report['workload_points'],
    'output_leaf_counts':report['output_leaf_counts'],'reference_preparation_identity':report['reference_preparation_identity'],
    'actual_stage_runtime_git_head':report['actual_stage_runtime_git_head'],
    'environment':{k:report['environment'][k] for k in ('jax','jaxlib','x64_enabled','gpu_count','cpu_affinity')},
    'candidate_provenance':report['candidate_provenance'],
    'physical_runner_created':False,'physical_lower_compile_run':False,
    'exit_boundary':report['exit_boundary'],'scientific_acceptance':False},
 'target_shell_dry_run':read('free-check/15857559/package-and-arguments.json'),
 'retention':{'registry_before_sha256':retention['registry_before_sha256'],
    'registry_after_sha256':retention['registry_after_sha256'],
    'README_before_sha256':retention['README_before_sha256'],'README_after_sha256':retention['README_after_sha256'],
    'old_records_preserved':True,'old_README_bytes_preserved':True,'archive_or_delete':False},
 'paid_proposal':{'partition':'cnall','nodes':1,'tasks':1,'allocated_cpus':48,'gpus':0,
    'time_limit_seconds':3600,'nonexclusive':True,'requeue':False,'added_memory_or_placement_restriction':False,
    'unverified_worst_case_CNY':[2.40,3.36],'run_root_pattern':pin['run_root_pattern'],
    'submitted':False,'approved':False,'next_dispatch':'supervisor approval'},
 'limitations':['No actual48 admission, physical compilation/run, numerical candidate gates or warm throughput measured.',
    'Scalar61/64 GPU passes; three finite failing points/four comparisons persist; scientific acceptance remains false.',
    'Reset remains sealed proof only; old paid grants remain closed; no automatic retry.'],
 'artifacts':[identity(n) for n in ('execution-code-pin.json','execution-harness.patch',pin['bundle']['name'],
    'scalar-admission.json','PROPOSED_SUBMISSION.sh','PAID_PROPOSAL.md','registry-retention-receipt.json',
    'free-check/15857559/report.json','free-check/15857559/actual-argv.txt',
    'free-check/15857559/actual-free-allocation.txt','free-check/15857559/sacct-terminal.txt',
    'free-check/15857559/exitcode.raw','free-check/15857559/launcher-dry-run.stdout',
    'free-check/15857559/coordinator-dry-run.stdout','free-check/15857559/package-and-arguments.json')]}
(ROOT/'preparation-result.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'status':summary['status'],'report':summary['artifacts'][7],'summary':identity('preparation-result.json')}))
