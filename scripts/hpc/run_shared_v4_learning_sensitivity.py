"""Prepared launcher; execution needs a new scientific approval.

Dry-run expands the real command without GPU IO, input reads or updates.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time

ROOT=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
CHECKOUT=ROOT/'runtime/checkouts/shared-split-width-v1'
OVERLAY=ROOT/'runtime/transfer/shared-v4-learning-sensitivity-20261004'
BASE=ROOT/'runtime/transfer/shared-v4-same-input-update-20261003'
FAMILY=ROOT/'runtime/outputs/diagnostics/shared-v4-learning-sensitivity-20261004'
TOTAL,GRACE=1800,10
SOURCES=('probe_shared_v4_learning_sensitivity.py','run_shared_v4_learning_sensitivity.py',
    'run_shared_v4_learning_sensitivity.sh')


def shared_launcher():
    path=ROOT/'runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_adjoint_T7.py'
    spec=importlib.util.spec_from_file_location('learning_sensitivity_shared_launcher',path)
    shared=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    shared.GLOBAL_CAP=TOTAL
    return shared


def command(shared,gpu,remaining=1790):
    old=shared.commands(gpu)['model']
    prefix=old[:old.index(str(shared.OVERLAY/'probe_shared_bank_adjoint_T7.py'))]
    prefix=[value.replace('SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':',
        'SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':'+str(OVERLAY)+':'+str(BASE)+':') for value in prefix]
    return prefix+[str(OVERLAY/SOURCES[0]),'--execute-approved-diagnostic',
        '--output-dir',str(FAMILY/'model'),'--process-seconds',str(remaining)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',choices=('0','1'),default='1')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--execute-approved-diagnostic',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not args.dry_run and not args.execute_approved_diagnostic:
        parser.error('learning execution is unapproved; use --dry-run')
    shared=shared_launcher()
    if args.dry_run:
        print(json.dumps(dict(status='prepared_unapproved_learning_diagnostic',GPU_calls=0,
            model_calls=0,optimizer_updates=0,Teacher_reads=0,total_seconds=TOTAL,
            kill_grace_seconds=GRACE,max_card_hours=.5,paid_jobs=0,CNY=0,
            steps_per_arm=16,planned_optimizer_updates=48,planned_model_calls=580,
            model_AD_compiles=0,checkout=str(CHECKOUT),output=str(FAMILY),
            required_free_disk_GiB=20,approval_required=True,resume=args.resume,
            resume_only_before_any_model_call=True),indent=2))
        print(shlex.join(command(shared,args.gpu)))
        return 0
    origin=float(os.environ['V4_LEARNING_SENSITIVITY_ORIGIN_UTC_SECONDS'])
    prior=[]
    if args.resume:
        old=json.loads((FAMILY/'actual-exit.json').read_text())
        origin=old['original_start_UTC_seconds']
        path=FAMILY/'model/model.json'
        previous=json.loads(path.read_text()) if path.exists() else {}
        if previous.get('model_calls_started',0) or previous.get('completed_optimizer_updates',0):
            raise ValueError('execution already started; no trajectory reset or duplicate learning')
        prior=old['records']
    elapsed=time.time()-origin
    started=time.perf_counter()-elapsed
    if elapsed<0 or elapsed>=TOTAL-GRACE:
        raise TimeoutError('original allowance exhausted; no origin reset')
    if args.resume:
        preserved=FAMILY/('preserved-pre-model-attempt-'+str(len(prior)))
        preserved.mkdir()
        for path in tuple(FAMILY.iterdir()):
            if path!=preserved and not path.name.startswith('preserved-pre-model-attempt-'):
                path.rename(preserved/path.name)
    else:FAMILY.mkdir(parents=True,exist_ok=False)
    records=[];status=1;context={};error_record=None
    try:
        if socket.gethostname().split('.')[0]!='ibc13b02n04' or os.environ.get('SLURM_JOB_ID'):
            raise ValueError('free gln01 outside paid Slurm required; no paid fallback')
        (FAMILY/'origin.json').write_text(json.dumps(dict(original_start_UTC_seconds=origin,
            deadline_UTC_seconds=origin+TOTAL)))
        snapshot=FAMILY/('source-attempt-'+str(len(prior)));snapshot.mkdir()
        for name in SOURCES:(snapshot/name).write_bytes((OVERLAY/name).read_bytes())
        if shutil.disk_usage(FAMILY).free<20*2**30:
            raise OSError('requires20GiB free; preserve old scientific data')
        head=shared.setup_command(['git','rev-parse','HEAD'],started,records,cwd=CHECKOUT).strip()
        if head!='c599236bcf57b745e9bafed678298010f4c3b58f':
            raise ValueError('immutable checkout differs')
        context=shared.read_context(args.gpu,started,records)
        remaining=shared.effective_limit(started,TOTAL-GRACE)
        status=shared.run_phase(command(shared,args.gpu,remaining),'diagnostic',started,
            TOTAL-GRACE,FAMILY,records)
        return status
    except BaseException as error:
        status=124 if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else 1
        error_record=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        path=FAMILY/'model/model.json'
        try:model=json.loads(path.read_text()) if path.exists() else {}
        except (ValueError,OSError) as error:
            model=dict(ledger_read_error=str(error))
        elapsed=time.perf_counter()-started
        ledger=dict(exit_status=status,original_start_UTC_seconds=origin,
            deadline_UTC_seconds=origin+TOTAL,total_cap_seconds=TOTAL,kill_grace_seconds=GRACE,
            family_seconds=elapsed,card_hours=elapsed/3600,GPU_count=1,
            physical_GPU=int(args.gpu),paid_jobs=0,CNY=0,records=prior+records,
            execution_context=context,exception=error_record,model_status=model.get('status'),
            model_calls=model.get('model_calls',0),model_calls_started=model.get('model_calls_started',0),
            completed_optimizer_updates=model.get('completed_optimizer_updates',0),
            diagnostic_only=True,formal_training=False,production_adoption=False)
        temporary=FAMILY/'actual-exit.json.tmp'
        temporary.write_text(json.dumps(ledger,indent=2)+'\n')
        temporary.replace(FAMILY/'actual-exit.json')


if __name__=='__main__':raise SystemExit(main())
