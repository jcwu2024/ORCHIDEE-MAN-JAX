"""Build a versioned registry addition locally from preserved actual receipts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
before=ROOT/'registry-before-deploy.json'
registry=json.loads(before.read_bytes())
assert 'latest_global_cpu48_b64_v4_deployment' not in registry
pin=json.loads((ROOT.parent/'cpu48-b64-preparation-v4-20261014/execution-code-pin.json').read_bytes())
preflight=json.loads((ROOT/'preflight/15853600/production-preflight.json').read_bytes())
registry['latest_global_cpu48_b64_v4_deployment']={
    'schema':'global_cpu48_b64_v4_deployment_v1','utc':datetime.now(timezone.utc).isoformat(),
    'execution_sha':pin['execution_code_commit'],'execution_ref':pin['execution_ref'],
    'base_sha':pin['base_execution_commit'],'code_dir':pin['target_checkout'],
    'checkout_state':'detached_clean','alternate_dependencies':[],
    'deployment_method':'verified prerequisite bundle non-forced bare fetch; fresh clone --no-hardlinks',
    'science_tree_oid':pin['scientific_source']['science_tree_oid'],
    'harness_blob_sha256':preflight['harness_blob_sha256'],
    'transfer_directory':'/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/global-cpu48-b64-v4-60dccbddde10',
    'bundle_sha256':pin['bundle']['sha256'],'pin_sha256':'e9e301aeab7237d59a0e204ffc93fc3eee96d28ab997d89be31383d735ec2769',
    'proposal_sha256':'b86068c62c96464319175135e12e5d84ce2e494a00875aceaa01ee0b1a48c160',
    'actual_preflight_job':'15853600','actual_preflight_node':'ibtest01',
    'actual_preflight_candidate_identity':preflight['actual_candidate_identity'],
    'canonical_python':preflight['canonical_python'],'package_versions':preflight['package_versions'],
    'persistent_scalar_admission':preflight['scalar_admission'],
    'runtime_evidence':'/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014',
    'allocation':{'partition':'cnall','nodes':1,'tasks':1,'cpus':48,'gpus':0,'wall_seconds':3600,
        'no_requeue':True,'exclusive':False,'memory_reservation':None,'placement_restriction':None,
        'primary_worst_case_cny':2.40,'conservative_worst_case_cny':3.36,'rates_verified':False},
    'execution_route':'batch-only','submitted':False,'sbatch_invocation_count':0,
    'registry_previous_sha256':hashlib.sha256(before.read_bytes()).hexdigest(),
    'scientific_acceptance':False}
out=ROOT/'registry-deployed.json'
out.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'previous_sha256':hashlib.sha256(before.read_bytes()).hexdigest(),
                 'next_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
