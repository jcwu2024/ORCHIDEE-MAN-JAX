"""Assemble existing terminal evidence; no physical calls or new scientific audit."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX'
E=P+'/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014'
RUN=P+'/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/15857765'
def load(name): return json.loads((ROOT/name).read_bytes())
def digest(name): return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
def write(name,data): (ROOT/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
r=load('model-report.json'); o=load('orchestration.json'); c=load('collector-result.json')
resources=r['server_resources']; topology=resources['topology']
physical={(v['package_id'],v['physical_core_id']) for v in topology}
assert len(physical)==48 and len(resources['selected_logical_cpus'])==48
assert r['status']=='failed' and o['status']=='batch_failed'
assert c['status']=='incomplete_run_failure_preserved'
assert c['whole_parent_allocation_cost']['wall_seconds']==819
assert c['reused_scalar_source_parent_cost']['wall_seconds']==2197
assert c['whole_parent_allocation_cost']['raw_parent_record_sha256']==digest('terminal/terminal-parent.sacct.raw')
cross=next(g for g in r['gates'] if g['name']=='B64_batch_cross_mode_server_scalar')
gpu=[g for g in r['gates'] if g['name'].endswith('_GPU_reference')]
isolation=[g for g in r['gates'] if g['name'] in ('B64_server_batch_cross_group_permutation','B64_server_batch_replacement_isolation')]
assert len(gpu)==64 and sum(g['passed'] for g in gpu)==61 and len(isolation)==2 and all(g['passed'] for g in isolation)
assert not any('warm_frozen_state_core' in k or 'preprepared_wrapper' in k for k in r['timing'])
assert r['environment']['git_head']=='0100c8b09be5137b822173e0a0be66cb64939727'
parent=c['whole_parent_allocation_cost']; old=c['reused_scalar_source_parent_cost']
comparison={'schema':'cpu48_v5_gpu_same_workload_comparison_v1','point_days':896,
 'GPU_authority':'../cpu48-b64-execution-v4-20261014/GPU_COMPARATOR_EVIDENCE.md',
 'GPU_B64_job':'15579869','GPU_core_median_seconds':5.632416466251016,
 'GPU_wrapper_median_seconds':6.222332641482353,
 'GPU_core_point_days_per_second':159.07914575720022,
 'GPU_wrapper_point_days_per_second':143.99744462818447,
 'CPU_B64_job':'15857765','CPU_core_warm_samples':[],'CPU_wrapper_warm_samples':[],
 'CPU_core_warm_median_seconds':None,'CPU_wrapper_warm_median_seconds':None,
 'CPU_warm_point_days_per_second':None,'CPU_GPU_warm_speed_ratio':None,
 'CPU_GPU_warm_unit_cost_ratio':None,
 'GPU_warm_cost_scenarios':{label:{'allocated_CPUs':4,'allocated_GPUs':1,'CNY_per_hour':rate,
    'core_CNY_per_896_point_days':rate*5.632416466251016/3600,
    'wrapper_CNY_per_896_point_days':rate*6.222332641482353/3600}
    for label,rate in [('primary_unverified',2.40),('conservative_unverified',2.48)]},
 'GPU_cost_boundary':'rate assumptions only; actual GPU billing unverified; authority retains original allocation/source limitations',
 'CPU_warm_unit_cost':None,'reason':'strict CPU internal scalar gate failed before five core+five wrapper warm repetitions',
 'new_CPU_allocation_scope':'batch-only candidate job15857765; old scalar15820467 is a separately completed prior asset, not generated or charged inside this new allocation',
 'accounting_boundary':'raw sacct819s parent is authority; unchanged collector generic scalar-step wording does not imply scalar work in this batch-only run',
 'cold_CPU_timings_seconds':r['timing'],'new_CPU_whole_parent_cost':parent,
 'separate_prior_scalar_asset_cost':old,'B1024_scope':'different workload background only; not used as same-workload comparison',
 'scientific_acceptance':False,'claim':'No validated CPU48 warm throughput, speedup or warm-unit-cost advantage is available.'}
write('CPU_GPU_COMPARISON.json',comparison)
summary={'schema':'global_cpu48_v5_terminal_summary_v1','job_id':'15857765','status':'FAILED',
 'exit_code':'1:0','parent_elapsed_seconds':819,'start_CST':'2026-10-14T14:34:22+08:00',
 'end_CST':'2026-10-14T14:48:01+08:00','node':'ibc12b03n02','paid_attempt_closed':True,
 'execution_commit':r['environment']['git_head'],'execution_code_dir':P+'/runtime/worktrees/global-throughput-cpu48-b64-v5',
 'run_root':RUN,'workload':{'unique_points':64,'blocks':[[2,8],[9,15]],'point_days':896,'native_output_leaves':760},
 'runtime':{k:r['environment'][k] for k in ('jax','jaxlib','x64_enabled')},
 'actual_node_start_snapshot':o['host_start_snapshot'],
 'resource_admission':{'allocated_CPUs':48,'distinct_physical_cores':len(physical),
    'selected_logical_cpus':resources['selected_logical_cpus'],'parent_billing_CPU_count':resources['billing_CPU_request_count'],
    'thread_environment_before_JAX':resources['thread_environment_before_JAX'],
    'host_memory_admission':resources['host_memory_admission'],
    'post_compile_observed_thread_count':r['thread_affinity_after_compile']['observed_thread_count'],
    'all_post_compile_thread_masks_within_declared_cores':r['thread_affinity_after_compile']['all_observed_thread_masks_within_declared_cores'],
    'scope':'allocation/affinity is not demonstrated48-core utilization'},
 'first_substantive_failure':{'gate':cross['name'],'type':'CPU internal strict numerical tolerance gate; not startup/provenance failure',
    'compared_leaves':cross['compared_leaves'],'failed_leaf_comparisons':cross['failures'],
    'max_bound_ratio':cross['max_bound_ratio'],'whole_760_leaf_tree_max_abs':cross['max_abs'],
    'whole_tree_max_abs_is_not_failed_coordinate_absdiff':True,'witness_file':'saved-failure-values.json'},
 'separate_GPU_gate_results':{'passed_points':61,'total_points':64,
    'failing_points':[{k:g[k] for k in ('name','failures','max_abs','max_bound_ratio')} for g in gpu if not g['passed']],
    'failure_comparisons':sum(len(g['failures']) for g in gpu if not g['passed']),
    'causality_relative_to_CPU_internal_failure':'not established'},
 'recorded_isolation_gates':[{'name':g['name'],'passed':g['passed'],'compared_leaves':g['compared_leaves']} for g in isolation],
 'gate_boundary':'permutation/replacement calls and recorded gates completed; no independent sealed-array isolation/reset recomputation; reset/warm/validation-array seal not reached',
 'phases':r['phases'],'timing_seconds':r['timing'],
 'compiled_input_signature':r['compilations']['server_batch_B64']['compiled_input_signature'],
 'peak_RSS_bytes_sampled':r['peak_rss_bytes'],'linux_ru_maxrss_peak_bytes':r['linux_ru_maxrss_peak_bytes'],
 'model_process_CPU_user_system_seconds':r['process_CPU_user_system_seconds'],
 'model_benchmark_wall_seconds':r['benchmark_wall_seconds'],
 'no_complete_batch_stage':True,'warm_core_samples':[],'warm_wrapper_samples':[],
 'new_parent_accounting':parent,'prior_scalar_asset_accounting':old,
 'accounting_boundary':'raw sacct819s parent is authority; unchanged collector generic scalar-step scope string does not mean scalar was regenerated or double charged',
 'collector':{'job_id':'15864638','source_SHA256':'efcb32efa754bbdbb75936f358dc6cb5a286e6742948096bda3e3ba22d27a9e3',
    'result_path':'collector-result.json','result_SHA256':digest('collector-result.json'),
    'status':c['status'],'model_execution':False},
 'original_error':r['message'],'traceback':r['traceback'],'scientific_acceptance':False,
 'limits':['No completed batch stage, sealed validation arrays, reset gate or five+five warm samples.',
    'No CPU48 warm throughput or CPU/GPU warm speed/cost ratio.',
    'Separate GPU three finite failing points/four comparisons, scalar61/64 and sealed-reset-proof limitation persist.',
    'No repair, retry, extension, cancellation, environment/source change or scientific acceptance.']}
write('execution-summary.json',summary)
manifest=load('saved-runtime-artifacts.json')
manifest['arrays_inspected']=True
manifest['array_inspection_scope']='only the three reported internal failing leaf arrays were extracted with NumPy from existing candidate/scalar files; see saved-failure-values.json; no full scientific re-audit or model calls'
mapping={'batch.log':'batch.log','orchestration.json':'orchestration.json','batch/report.json':'model-report.json',
    'batch/stage_manifest.json':'stage_manifest.json','batch/raw_B64_batch_before_parity.pkl':'raw_B64_batch_before_parity.pkl'}
for item in manifest['files']:
    suffix=item['path'][len(RUN)+1:]
    local=mapping.get(suffix)
    item['local_mirror']=local
    item['tracked']=local in ('batch.log','stage_manifest.json')
    if local:
        assert digest(local)==item['sha256']
manifest['scalar_source_local_mirror']={'path':'source-scalar.pkl','sha256':digest('source-scalar.pkl'),'tracked':False,
 'source_job':'15820467','new_generation':False}
manifest['compact_products']=[{'path':n,'sha256':digest(n),'bytes':(ROOT/n).stat().st_size} for n in
    ('AUTHORIZATION.md','execution-summary.json','CPU_GPU_COMPARISON.json','collector-result.json','saved-failure-values.json',
    'stage_manifest.json','terminal/parent-accounting.json','terminal/terminal-full.sacct.raw','terminal/collector-execution-receipt.json')]
if (ROOT/'terminal-registry-receipt.json').exists():
    manifest['compact_products'].append({'path':'terminal-registry-receipt.json',
        'sha256':digest('terminal-registry-receipt.json'),'bytes':(ROOT/'terminal-registry-receipt.json').stat().st_size})
manifest['launch_scontrol']=[{'path':'terminal/launch-scontrol.'+s,'sha256':digest('terminal/launch-scontrol.'+s),
    'remote_path':P+'/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/scontrol-job-15857765.'+s}
    for s in ('stdout','stderr','exitcode')]
write('runtime-evidence-manifest.json',manifest)
entry={'job_id':'15857765','terminal_state':'FAILED','exit_code':'1:0','parent_seconds':819,'allocated_CPUs':48,'GPUs':0,
    'execution_commit':summary['execution_commit'],'execution_checkout':summary['execution_code_dir'],'run_root':RUN,
    'evidence_dir':E,'failure_gate':cross['name'],'physical_compile_and_first_call_completed':True,
    'warm_samples':0,'whole_parent_CNY_scenarios':[0.546,0.7644],'prior_scalar_job_id':'15820467',
    'prior_scalar_seconds':2197,'prior_scalar_recharged':False,'scientific_acceptance':False,'attempt_closed':True,
    'collector_job_id':'15864638','collector_result_sha256':digest('collector-result.json'),
    'raw_outputs_sha256':'a135f9436afa868176bf8cc76f28134218e510c65b75909075ba24a8964810ce',
    'stage_manifest_sha256':digest('stage_manifest.json'),'next_action':'wait for new supervisor dispatch; no retry',
    'preserve_all_prior_records_and_assets':True}
write('terminal-registration-entry.json',entry)
print(json.dumps({'status':summary['status'],'summary_sha256':digest('execution-summary.json'),
    'manifest_sha256':digest('runtime-evidence-manifest.json'),'comparison_sha256':digest('CPU_GPU_COMPARISON.json')}))
