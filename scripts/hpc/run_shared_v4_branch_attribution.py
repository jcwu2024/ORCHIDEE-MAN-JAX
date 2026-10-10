"""One900s ordinary freeGPU window, exactly38 saved-binary branch calls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import time
spec=importlib.util.spec_from_file_location('v4_base_launcher',Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-v4-same-input-update-20261003/run_shared_v4_same_input_update.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
shared=base.shared;ROOT=base.ROOT;CHECKOUT=base.CHECKOUT
OVERLAY=ROOT/'runtime/transfer/shared-v4-branch-attribution-20261003'
FAMILY=ROOT/'runtime/outputs/diagnostics/shared-v4-branch-attribution-20261003'
TOTAL,GRACE=900,10
shared.GLOBAL_CAP=TOTAL
SOURCES=('probe_shared_v4_branch_attribution.py','run_shared_v4_branch_attribution.py','run_shared_v4_branch_attribution.sh')

def command(gpu,remaining=890):
    old=base.command(gpu,remaining)
    index=old.index(str(base.OVERLAY/'probe_shared_v4_same_input_update.py'))
    prefix=[value.replace('SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':',
        'SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':'+str(OVERLAY)+':') for value in old[:index]]
    return prefix+[str(OVERLAY/SOURCES[0]),'--output-dir',str(FAMILY/'model'),'--process-seconds',str(remaining)]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',choices=('0','1'),default='1')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(status='authorized_ordinary_free_branch_diagnostic',total_seconds=TOTAL,grace=GRACE,
            model_calls=38,daily_micro4_calls=32,T7_calls=4,shared_T30_calls=2,repeats=2,orders=[['reference','candidate'],['candidate','reference']],
            fixed_input=str(base.FAMILY/'model/pair-1-reference.full-proposal.pkl'),daily_slice_execution='leaf',
            optimizer_updates=0,new_model_AD_compiles=0,extra_warm_calls=0,whole_origin_before_Python=True,
            output=str(FAMILY),GPU_count=1,max_card_hours=.25,paid_jobs=0,CNY=0,resume=args.resume,
            raw_saved_before_classification=True,retained_training_updates=0,production_adoption=False),indent=2))
        print(shlex.join(command(args.gpu)));return 0
    if socket.gethostname().split('.')[0]!='ibc13b02n04' or os.environ.get('SLURM_JOB_ID'):raise ValueError('ordinary free gln01 only')
    origin=float(os.environ['V4_BRANCH_ORIGIN_UTC_SECONDS']);prior=[]
    if args.resume:
        previous=json.loads((FAMILY/'actual-exit.json').read_text());origin=previous['original_start_UTC_seconds'];prior=previous['records']
        path=FAMILY/'model/model.json';model=json.loads(path.read_text()) if path.exists() else {}
        if model.get('model_calls_started',0) or any((FAMILY/'model').glob('call-*.pkl')):raise ValueError('begun model call terminal; no retry')
        preserve=FAMILY/('preserved-pre-model-attempt-'+str(len(prior)));preserve.mkdir()
        for name in ('model','diagnostic.log','actual-exit.json'):
            path=FAMILY/name
            if path.exists():path.rename(preserve/name)
    else:
        FAMILY.mkdir(exist_ok=False,parents=True)
        (FAMILY/'origin.json').write_text(json.dumps(dict(original_start_UTC_seconds=origin,deadline_UTC_seconds=origin+TOTAL)))
    elapsed=time.time()-origin
    if not 0<=elapsed<TOTAL-GRACE:raise TimeoutError('original allowance exhausted')
    started=time.perf_counter()-elapsed;records=[];status=1;context={}
    snapshot=FAMILY/('source-attempt-'+str(len(prior)));snapshot.mkdir()
    for name in SOURCES:(snapshot/name).write_bytes((OVERLAY/name).read_bytes())
    for name in ('probe_shared_v4_same_input_update.py','analyze_shared_v4_ab_offline.py'):
        (snapshot/name).write_bytes((base.OVERLAY/name).read_bytes())
    try:
        if shutil.disk_usage(FAMILY).free<16*2**30:raise OSError('requires16GiB free for preserved raw branch results')
        head=shared.setup_command(['git','rev-parse','HEAD'],started,records,cwd=CHECKOUT).strip()
        if head!='c599236bcf57b745e9bafed678298010f4c3b58f':raise ValueError('immutable checkout differs')
        context=shared.read_context(args.gpu,started,records)
        status=shared.run_phase(command(args.gpu,shared.effective_limit(started,TOTAL-GRACE)),
            'diagnostic',started,TOTAL-GRACE,FAMILY,records);return status
    finally:
        path=FAMILY/'model/model.json';model=json.loads(path.read_text()) if path.exists() else {};elapsed=time.perf_counter()-started
        (FAMILY/'actual-exit.json').write_text(json.dumps(dict(exit_status=status,original_start_UTC_seconds=origin,
            deadline_UTC_seconds=origin+TOTAL,family_seconds=elapsed,total_cap_seconds=TOTAL,kill_grace_seconds=GRACE,
            card_hours=elapsed/3600,GPU_count=1,physical_GPU=int(args.gpu),paid_jobs=0,CNY=0,records=prior+records,
            execution_context=context,model_calls=model.get('model_calls',0),model_calls_started=model.get('model_calls_started',0),
            model_status=model.get('status'),optimizer_updates=0,retained_training_updates=0,production_adoption=False),indent=2)+'\n')

if __name__=='__main__':raise SystemExit(main())
