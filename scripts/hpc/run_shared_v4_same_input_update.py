"""Bounded free-node saved-binary update diagnostic; dry-run performs no GPU IO."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import time
try:import run_shared_bank_adjoint_T7 as shared
except ModuleNotFoundError:
    spec=importlib.util.spec_from_file_location('v4_shared_launcher',Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_adjoint_T7.py'))
    shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
ROOT,CHECKOUT=shared.ROOT,shared.CHECKOUT
OVERLAY=ROOT/'runtime/transfer/shared-v4-same-input-update-20261003'
FAMILY=ROOT/'runtime/outputs/diagnostics/shared-v4-same-input-update-20261003'
TOTAL,GRACE=1800,10
shared.GLOBAL_CAP=TOTAL
SOURCES=('probe_shared_v4_same_input_update.py','run_shared_v4_same_input_update.py',
    'run_shared_v4_same_input_update.sh','analyze_shared_v4_ab_offline.py')

def command(gpu,remaining=1790):
    old=shared.commands(gpu)['model'];prefix=old[:old.index(str(shared.OVERLAY/'probe_shared_bank_adjoint_T7.py'))]
    prefix=[value.replace('SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':',
        'SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':'+str(OVERLAY)+':') for value in prefix]
    return prefix+[str(OVERLAY/SOURCES[0]),'--output-dir',str(FAMILY/'model'),'--process-seconds',str(remaining)]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',choices=('0','1'),default='1')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(status='runnable_ordinary_free_diagnostic',GPU_calls=0,Teacher_reads=0,
            total_seconds=TOTAL,kill_grace_seconds=GRACE,exact_proposals=6,exact_model_calls=60,
            pairs=[['reference','candidate'],['candidate','reference'],['reference','candidate']],
            same_parameters_and_optimizer_each_pair=True,advance_reference_only=True,
            daily32_eight_distinct_micro4=True,shared_original_T30=True,model_AD_compiles=0,
            extra_bootstrap_or_warm_calls=0,strict_A_B_failure_does_not_stop_if_hard_pass=True,
            first_pair_cold_separate=True,last_two_pairs_for_warm_times=True,
            whole_origin_before_Python=True,resume=args.resume,paid_jobs=0,CNY=0,max_card_hours=.5,
            output=str(FAMILY),new_sources=list(SOURCES),estimated_new_raw_bytes=6*4*64884481*4,
            retained_training_updates=0,production_adoption=False),indent=2))
        print(shlex.join(command(args.gpu)));return 0
    if socket.gethostname().split('.')[0]!='ibc13b02n04' or os.environ.get('SLURM_JOB_ID'):raise ValueError('free gln01 node required; no paid fallback')
    origin=float(os.environ['V4_SAME_INPUT_ORIGIN_UTC_SECONDS']);prior=[]
    if args.resume:
        old=json.loads((FAMILY/'actual-exit.json').read_text());origin=old['original_start_UTC_seconds']
        model_path=FAMILY/'model/model.json';model=json.loads(model_path.read_text()) if model_path.exists() else {}
        if model.get('model_calls_started',0) or model.get('model_calls',0) or any((FAMILY/'model').glob('*.full-proposal.pkl')):raise ValueError('model calls/proposal terminal; no repeat')
        prior=old['records'];preserved=FAMILY/('preserved-pre-model-attempt-'+str(len(prior)));preserved.mkdir()
        for name in ('model','diagnostic.log','actual-exit.json'):
            path=FAMILY/name
            if path.exists():path.rename(preserved/name)
    else:
        FAMILY.mkdir(parents=True,exist_ok=False)
        (FAMILY/'origin.json').write_text(json.dumps(dict(original_start_UTC_seconds=origin,deadline_UTC_seconds=origin+TOTAL)))
    elapsed=time.time()-origin
    if elapsed<0 or elapsed>=TOTAL-GRACE:raise TimeoutError('original allowance exhausted; no origin reset')
    started=time.perf_counter()-elapsed;records=[];status=1;context={}
    snapshot=FAMILY/('source-attempt-'+str(len(prior)));snapshot.mkdir()
    for name in SOURCES:(snapshot/name).write_bytes((OVERLAY/name).read_bytes())
    try:
        if shutil.disk_usage(FAMILY).free<7*2**30:raise OSError('requires7GiB free for six full proposals; do not delete old science')
        head=shared.setup_command(['git','rev-parse','HEAD'],started,records,cwd=CHECKOUT).strip()
        if head!='c599236bcf57b745e9bafed678298010f4c3b58f':raise ValueError('immutable checkout differs')
        context=shared.read_context(args.gpu,started,records)
        remaining=shared.effective_limit(started,TOTAL-GRACE)
        status=shared.run_phase(command(args.gpu,remaining),'diagnostic',started,TOTAL-GRACE,FAMILY,records)
        return status
    finally:
        path=FAMILY/'model/model.json';model=json.loads(path.read_text()) if path.exists() else {}
        elapsed=time.perf_counter()-started
        (FAMILY/'actual-exit.json').write_text(json.dumps(dict(exit_status=status,original_start_UTC_seconds=origin,
            deadline_UTC_seconds=origin+TOTAL,family_seconds=elapsed,total_cap_seconds=TOTAL,kill_grace_seconds=GRACE,
            card_hours=elapsed/3600,GPU_count=1,physical_GPU=int(args.gpu),paid_jobs=0,CNY=0,records=prior+records,
            execution_context=context,completed_pairs=model.get('completed_pairs',0),model_calls=model.get('model_calls',0),
            model_calls_started=model.get('model_calls_started',0),model_status=model.get('status'),retained_training_updates=0,production_adoption=False),indent=2)+'\n')

if __name__=='__main__':raise SystemExit(main())
