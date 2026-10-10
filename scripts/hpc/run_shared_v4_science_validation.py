"""Persistent CPU300s/GPU1500s windows for the approved held-out forward panel."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

ROOT=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
CHECKOUT=ROOT/'runtime/checkouts/shared-split-width-v1'
OVERLAY=ROOT/'runtime/transfer/shared-v4-heldout-science-20261004'
FAMILY=ROOT/'runtime/outputs/diagnostics/shared-v4-heldout-science-20261004'
SOURCES=('probe_shared_v4_science_validation.py','run_shared_v4_science_validation.py','run_shared_v4_science_validation.sh','fresh_method_validation_pool.py','shared_learning_monitor.py')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def shared_launcher():
    path=ROOT/'runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_adjoint_T7.py'
    spec=importlib.util.spec_from_file_location('science_shared_launcher',path);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner);return owner

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('cpu','gpu'));parser.add_argument('--gpu',choices=('0','1'),default='1');parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    cap=300 if args.mode=='cpu' else 1500;grace=5 if args.mode=='cpu' else 10
    shared=shared_launcher();shared.GLOBAL_CAP=cap;shared.GRACE=grace
    old=shared.commands(args.gpu)['model'];prefix=old[:old.index(str(shared.OVERLAY/'probe_shared_bank_adjoint_T7.py'))]
    prefix=[value.replace('SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':','SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':'+str(OVERLAY)+':') for value in prefix]
    if args.mode=='cpu':
        command=['env','JAX_PLATFORMS=cpu','JAX_ENABLE_X64=true','PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=2',f'ORCHIDEE_REPO_ROOT={CHECKOUT}',f'PYTHONPATH={CHECKOUT}:{OVERLAY}',str(ROOT/'.venvs/orcjax_cpu/bin/python'),'-B',str(OVERLAY/SOURCES[0]),'--prepare-cpu']
    else:command=prefix+[str(OVERLAY/SOURCES[0]),'--execute-approved']
    if args.dry_run:
        print(json.dumps(dict(mode=args.mode,cumulative_cap_seconds=cap,grace_seconds=grace,model_calls=0,execution=False,max_model_forward_compiles=2),indent=2));print(shlex.join(command));return 0
    FAMILY.mkdir(parents=True,exist_ok=True);origin_path=FAMILY/(args.mode+'-origin.json')
    if origin_path.exists():origin=json.loads(origin_path.read_text())['original_start_UTC_seconds']
    else:origin=time.time();write(origin_path,dict(original_start_UTC_seconds=origin,deadline_UTC_seconds=origin+cap,total_cap_seconds=cap))
    elapsed=time.time()-origin;started=time.perf_counter()-elapsed
    history_path=FAMILY/(args.mode+'-attempt-history.json');history=json.loads(history_path.read_text()) if history_path.exists() else []
    attempt=len(history);records=[];status=1;context={};error=None
    try:
        if args.mode=='cpu' and not socket.gethostname().split('.')[0].endswith('test02'):raise ValueError('CPU saved-array preparation requires test02')
        if args.mode=='gpu' and (socket.gethostname().split('.')[0]!='ibc13b02n04' or os.environ.get('SLURM_JOB_ID')):raise ValueError('Approved free gln01 only; no paid fallback')
        if shared.effective_limit(started,cap)<=0:raise TimeoutError('Original cumulative allowance exhausted')
        snapshot=FAMILY/f'{args.mode}-source-attempt-{attempt}';snapshot.mkdir()
        for name in SOURCES:(snapshot/name).write_bytes((OVERLAY/name).read_bytes())
        if args.mode=='gpu':
            if not (FAMILY/'CPU-preparation.json').is_file():raise ValueError('CPU preparation missing')
            if json.loads((FAMILY/'training-overlap.json').read_text())['status']!='passed_no_exact_day_or_window_overlap':raise ValueError('Actual training overlap admission missing')
            if (FAMILY/'model/model.json').exists() and json.loads((FAMILY/'model/model.json').read_text()).get('status')=='completed_not_adopted':raise ValueError('Completed result blocks another GPU execution')
            context=shared.read_context(args.gpu,started,records)
            head=shared.setup_command(['git','rev-parse','HEAD'],started,records,cwd=CHECKOUT).strip()
            if head!='c599236bcf57b745e9bafed678298010f4c3b58f':raise ValueError('Immutable checkout differs')
            if not (FAMILY/'preflight.json').exists():
                preflight=prefix+[str(CHECKOUT/'scripts/hpc/check_daily_training_runtime.py'),'--backend','gpu','--output',str(FAMILY/'preflight.json')]
                status=shared.run_phase(preflight,f'preflight-{attempt}',started,180,FAMILY,records)
                if status:return status
        remaining=shared.effective_limit(started,cap)
        environment=os.environ.copy();environment['SCIENCE_PROCESS_REMAINING_SECONDS']=str(remaining)
        if args.mode=='gpu':command[0:0]=['env',f'SINGULARITYENV_SCIENCE_PROCESS_REMAINING_SECONDS={remaining}']
        os.environ['SCIENCE_PROCESS_REMAINING_SECONDS']=str(remaining)
        status=shared.run_phase(command,f'{args.mode}-{attempt}',started,cap,FAMILY,records)
        return status
    except BaseException as exception:
        status=124 if isinstance(exception,(TimeoutError,subprocess.TimeoutExpired)) else 1;error=dict(type=type(exception).__name__,message=str(exception));raise
    finally:
        wall=time.time()-origin;model_path=FAMILY/'model/model.json';model=json.loads(model_path.read_text()) if model_path.exists() else {}
        record=dict(mode=args.mode,attempt=attempt,exit_status=status,original_start_UTC_seconds=origin,cumulative_family_seconds=wall,
            total_cap_seconds=cap,records=records,execution_context=context,exception=error,
            GPU_count=1 if args.mode=='gpu' else 0,card_hours=wall/3600 if args.mode=='gpu' else 0,
            physical_GPU=int(args.gpu) if args.mode=='gpu' else None,paid_jobs=0,CNY=0,model_status=model.get('status'),
            model_calls=model.get('model_calls',0),model_calls_started=model.get('model_calls_started',0),training_updates=0,optimizer_executions=0)
        history.append(record);write(history_path,history);write(FAMILY/('actual-'+args.mode+'-exit.json'),record);print(json.dumps(record,indent=2))

if __name__=='__main__':raise SystemExit(main())
