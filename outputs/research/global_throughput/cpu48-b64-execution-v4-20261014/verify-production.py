"""One-free-CPU real-checkout metadata/seal preflight; no model imports."""
from __future__ import annotations
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

P = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
CODE = P / 'runtime/worktrees/global-throughput-cpu48-b64-v4'
TRANSFER = P / 'runtime/transfer/global-cpu48-b64-v4-60dccbddde10'
OUT = P / 'runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014'
ADMISSION = P / 'runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json'
SCALAR = P / 'runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar'

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()

def git(*args):
    return subprocess.check_output(['git',*args], cwd=CODE, text=True).strip()

def main():
    assert os.environ['SLURM_JOB_PARTITION']=='cntest'
    assert os.environ['SLURM_CPUS_PER_TASK']=='1'
    assert len(os.sched_getaffinity(0))==1
    pin_path=TRANSFER/'execution-code-pin.json'
    assert sha(pin_path)=='e9e301aeab7237d59a0e204ffc93fc3eee96d28ab997d89be31383d735ec2769'
    pin=json.loads(pin_path.read_text())
    assert git('rev-parse','HEAD')==pin['execution_code_commit']=='60dccbddde10be4295aab7e7fe55fa7864d3c3f3'
    assert not git('status','--porcelain','--untracked-files=all')
    assert git('rev-parse','HEAD:jax_orchidee')==pin['scientific_source']['science_tree_oid']
    harness={}
    for path,expected in pin['harness_blob_sha256'].items():
        actual=sha(CODE/path)
        blob=subprocess.check_output(['git','show',f"{pin['execution_code_commit']}:{path}"],cwd=CODE)
        assert actual==expected==hashlib.sha256(blob).hexdigest(),path
        harness[path]=actual
    sys.path.insert(0,str(CODE))
    from scripts.dev.global_cpu_candidate_identity import candidate_identity
    from scripts.dev.global_server_reference import validate_cpu_reference_admission
    args=SimpleNamespace(entry=CODE/'scripts/dev/benchmark_global_server_cpu.py',
        prepared_input=P/'runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl',
        gpu_reference_dir=P/'runtime/outputs/research/global_throughput/gpu-round4/model')
    actual_identity=candidate_identity(args,root=CODE)
    admission=validate_cpu_reference_admission(ADMISSION,
        'f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d',
        SCALAR,expected_source=actual_identity)
    assert actual_identity['job_id']==os.environ['SLURM_JOB_ID']
    assert actual_identity['prepared_input_sha256']=='be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7'
    assert actual_identity['gpu_reference_sha256']=='a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6'
    versions={name:importlib.metadata.version(name) for name in ('jax','jaxlib')}
    assert versions=={'jax':'0.4.38','jaxlib':'0.4.38'}
    module=importlib.util.find_spec('jax_orchidee').origin
    assert module==str(CODE/'jax_orchidee/__init__.py')
    assert sys.executable==str(P/'.venvs/orcjax_cpu/bin/python')
    config=CODE/'configs/orchidee_man_250919.yaml'
    assert config.is_file() and os.access(config,os.R_OK)
    probe=P/'runtime/logs'/f"cpu48-v4-write-probe-{os.environ['SLURM_JOB_ID']}"
    with probe.open('xb') as stream: stream.write(b'production preflight log write probe\n')
    probe.unlink()
    assert 'jax' not in sys.modules
    result={'schema':'cpu48-v4-production-preflight-v1','status':'actual_production_identity_and_seal_passed',
        'hostname':socket.gethostname(),'partition':os.environ['SLURM_JOB_PARTITION'],
        'job_id':os.environ['SLURM_JOB_ID'],'allocated_cpus':1,'affinity':sorted(os.sched_getaffinity(0)),
        'canonical_python':sys.executable,'python_version':sys.version,'package_versions':versions,
        'module_import_origin_without_import':module,'actual_candidate_identity':actual_identity,
        'scalar_admission':{'path':admission['path'],'sha256':admission['sha256'],
                            'stage':admission['stage'],'source_job_id':admission['source_job_id']},
        'harness_blob_sha256':harness,'config_path':str(config),'config_sha256':sha(config),
        'shared_logs_write_probe':'passed_removed_own_probe','JAX_imported':False,'model_execution':False,
        'loadavg':os.getloadavg(),'script_sha256':sha(__file__),'checkout_clean':True}
    output=OUT/'preflight'/os.environ['SLURM_JOB_ID']
    output.mkdir(parents=True,exist_ok=True)
    (output/'production-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
