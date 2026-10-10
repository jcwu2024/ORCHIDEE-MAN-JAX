"""Finite one-backend free-node launcher; dry-run initializes no GPU."""
import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time

ROOT=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
CHECKOUT=ROOT/'runtime/checkouts/shared-split-width-v1'
OVERLAY=ROOT/'runtime/transfer/shared-bank-adjoint-accumulation-20261002'
FAMILY=ROOT/'runtime/outputs/diagnostics/shared-bank-adjoint-accumulation-T7-20261002'
GLOBAL_CAP=3600
GRACE=10

def effective_limit(started,phase_cap):
    return max(0,int(min(phase_cap,GLOBAL_CAP-(time.perf_counter()-started)-GRACE)))

def commands(gpu,model_seconds=3590):
    old_forward=ROOT/'runtime/transfer/shared-forward-reverse-20260930'
    old_loops=ROOT/'runtime/transfer/shared-command-buffer-loops-20261001'
    env=['env','-u','LD_LIBRARY_PATH','-u','LD_PRELOAD',
        'SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs',f'SINGULARITYENV_CUDA_VISIBLE_DEVICES={gpu}',
        'SINGULARITYENV_JAX_PLATFORMS=cuda,cpu','SINGULARITYENV_JAX_ENABLE_X64=true',
        'SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false',f'SINGULARITYENV_ORCHIDEE_REPO_ROOT={CHECKOUT}',
        f'SINGULARITYENV_PYTHONPATH={CHECKOUT}:{OVERLAY}:{old_forward}:{old_loops}',
        'SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1','SINGULARITYENV_OPENBLAS_NUM_THREADS=1',
        'SINGULARITYENV_OMP_NUM_THREADS=4','SINGULARITYENV_XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false']
    container=['singularity','exec','--nv','--bind',
        f'/WORK:/WORK,{ROOT}/runtime/data:{ROOT}/data:ro,/usr/lib64/libcuda.so.525.85.12:/usr/lib/x86_64-linux-gnu/libcuda.so.1,/usr/lib64/libnvidia-ml.so.525.85.12:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1',
        '--pwd',str(CHECKOUT),'/apps/soft/sif/foundationpose',str(ROOT/'.venvs/orcjax_gpu/bin/python'),'-B']
    prefix=env+container
    return {
        'preflight':prefix+[str(CHECKOUT/'scripts/hpc/check_daily_training_runtime.py'),'--backend','gpu','--output',str(FAMILY/'preflight.json')],
        'model':prefix+[str(OVERLAY/'probe_shared_bank_adjoint_T7.py'),'--root',str(ROOT),
            '--runtime-preflight',str(FAMILY/'preflight.json'),'--output-dir',str(FAMILY/'model'),
            '--lower-seconds','300','--backend-seconds','2800','--process-seconds',str(model_seconds)]}

def run_phase(command,label,started,cap,destination,records):
    limit=effective_limit(started,cap)
    if limit<=0:
        records.append({'phase':label,'exit_status':124,'main_timeout_seconds':0,'started':False,'reason':'shared3600s exhausted before phase'})
        return 124
    actual=['timeout','--signal=TERM','--kill-after=10s',f'{limit}s',*command]
    began=time.perf_counter()
    record={'phase':label,'start_UTC_seconds':time.time(),'start_family_elapsed_seconds':began-started,
        'main_timeout_seconds':limit,'kill_grace_seconds':GRACE,'command':actual,'started':True}
    records.append(record)
    try:
        with (destination/(label+'.log')).open('x') as output:
            result=subprocess.run(actual,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        record['exit_status']=result.returncode
        return result.returncode
    except BaseException as error:
        record.update(exit_status=124 if isinstance(error,subprocess.TimeoutExpired) else 1,exception=repr(error))
        raise
    finally:
        record.update(end_UTC_seconds=time.time(),elapsed_seconds=time.perf_counter()-began)

def setup_command(argv,started,records,*,cwd=None):
    limit=effective_limit(started,20)
    began=time.perf_counter()
    record={'phase':'setup','command':argv,'main_timeout_seconds':limit,'start_UTC_seconds':time.time(),
        'start_family_elapsed_seconds':began-started}
    records.append(record)
    try:
        if limit<=0:raise subprocess.TimeoutExpired(argv,0)
        result=subprocess.run(argv,text=True,capture_output=True,cwd=cwd,timeout=limit)
        record['exit_status']=result.returncode
        if result.returncode:raise RuntimeError(f'context command failed: {argv}: {result.stderr}')
        return result.stdout
    except BaseException as error:
        record.update(exit_status=124 if isinstance(error,subprocess.TimeoutExpired) else 1,exception=repr(error))
        raise
    finally:record.update(end_UTC_seconds=time.time(),elapsed_seconds=time.perf_counter()-began)

def read_context(gpu,started,records):
    def command(argv):
        return setup_command(argv,started,records)
    occupancy=command(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']).strip()
    contexts=command(['nvidia-smi','-i',gpu,'--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])
    if occupancy!='0, 0' or contexts.strip():raise ValueError('selected free GPU not empty')
    return {'hostname':command(['hostname']),'host_load':command(['uptime']),
        'CPU_affinity':command(['grep','Cpus_allowed_list','/proc/self/status']),
        'occupancy_before':occupancy,'visible_GPU_processes':command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',choices=('0','1'),default='1')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        expanded=commands(args.gpu)
        print(json.dumps({'kind':'unapproved_preparation_only','GPU_calls':0,'real_model_input_reads':0,
            'global_cap_including_grace_seconds':3600,'lower_cap':300,'sole_backend_cap':2800,
            'preflight_cap':300,'max_backend_attempts':1,'paid_jobs':0,'CNY':0,
            'checkout':str(CHECKOUT),'overlay':str(OVERLAY),'output':str(FAMILY)},indent=2))
        for label,command in expanded.items():print(label+': '+shlex.join(command))
        subprocess.run([str(ROOT/'.venvs/orcjax_cpu/bin/python'),'-B',str(OVERLAY/'probe_shared_bank_adjoint_T7.py'),
            '--runtime-preflight',str(FAMILY/'preflight.json'),'--output-dir',str(FAMILY/'model'),
            '--lower-seconds','300','--backend-seconds','2800','--process-seconds','3590','--cli-check-only'],check=True)
        return 0
    started=time.perf_counter()
    if socket.gethostname().split('.')[0]!='ibc13b02n04' or __import__('os').environ.get('SLURM_JOB_ID'):
        raise ValueError('requires the approved free gln01 node, outside paid Slurm')
    if FAMILY.exists():raise FileExistsError('preserve existing family; no automatic retries')
    FAMILY.mkdir(parents=True)
    records=[];status=1;context={};family_exception=None
    try:
        head=setup_command(['git','rev-parse','HEAD'],started,records,cwd=CHECKOUT).strip()
        if head!='c599236bcf57b745e9bafed678298010f4c3b58f':raise ValueError('immutable checkout identity differs')
        context=read_context(args.gpu,started,records)
        status=run_phase(commands(args.gpu)['preflight'],'preflight',started,300,FAMILY,records)
        if status:return status
        remaining=effective_limit(started,3590)
        if remaining<=0:status=124;return status
        status=run_phase(commands(args.gpu,remaining)['model'],'model',started,3590,FAMILY,records)
        return status
    except BaseException as error:
        status=124 if isinstance(error,subprocess.TimeoutExpired) else 1
        family_exception=repr(error)
        raise
    finally:
        elapsed=time.perf_counter()-started
        model=FAMILY/'model/model.json'
        candidate_attempts=json.loads(model.read_text()).get('candidate_backend_attempts',0) if model.exists() else 0
        ledger={'exit_status':status,'family_seconds':elapsed,'card_hours':elapsed/3600,
            'global_cap_including_grace_seconds':GLOBAL_CAP,'kill_grace_seconds':GRACE,
            'candidate_backend_attempts':candidate_attempts,'max_backend_attempts':1,
            'lower_cap_seconds':300,'backend_cap_seconds':2800,'records':records,
            'family_exception':family_exception,
            'physical_GPU':int(args.gpu),'GPU_count':1,'paid_jobs':0,'CNY':0,'execution_context':context,
            'scientific_reference_adopted':False,'training_updates':0}
        (FAMILY/'actual-exit.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(ledger,indent=2))

if __name__=='__main__':raise SystemExit(main())
