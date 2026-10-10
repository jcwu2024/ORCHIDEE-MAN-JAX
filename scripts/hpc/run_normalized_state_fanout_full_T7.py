"""Preparation-only launcher until user authorizes a new finite free window."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
try:
    import run_shared_bank_adjoint_T7 as shared
except ModuleNotFoundError as error:
    if error.name!='run_shared_bank_adjoint_T7': raise
    helper=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_adjoint_T7.py')
    spec=importlib.util.spec_from_file_location('fanout_shared_launcher',helper)
    shared=importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)

ROOT,CHECKOUT=shared.ROOT,shared.CHECKOUT
OVERLAY=ROOT/'runtime/transfer/normalized-state-fanout-full-T7-20261003'
FAMILY=ROOT/'runtime/outputs/diagnostics/normalized-state-fanout-full-T7-20261003'
COMPONENT=ROOT/'runtime/transfer/normalized-state-fanout-component-20261003'
TOTAL_CAP,MAIN_CAP,GRACE=3600,3590,10
HEAD='c599236bcf57b745e9bafed678298010f4c3b58f'
SOURCES=('normalized_state_fanout_full_T7.py','probe_normalized_state_fanout_full_T7.py',
    'run_normalized_state_fanout_full_T7.py','run_normalized_state_fanout_full_T7.sh',
    'normalized_state_fanout_proto.py','normalized_state_fanout_component.py')
shared.GLOBAL_CAP=TOTAL_CAP


def commands(gpu,seconds=MAIN_CAP):
    model=shared.commands(gpu)['model']
    prefix=model[:model.index(str(shared.OVERLAY/'probe_shared_bank_adjoint_T7.py'))]
    prefix=[x.replace('SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':',
        'SINGULARITYENV_PYTHONPATH='+str(CHECKOUT)+':'+str(OVERLAY)+':'+str(COMPONENT)+':') for x in prefix]
    return dict(preflight=prefix+[str(CHECKOUT/'scripts/hpc/check_daily_training_runtime.py'),'--backend','gpu',
        '--output',str(FAMILY/'preflight.json')],
        candidate=prefix+[str(OVERLAY/'probe_normalized_state_fanout_full_T7.py'),'--root',str(ROOT),
            '--runtime-preflight',str(FAMILY/'preflight.json'),'--output-dir',str(FAMILY/'model'),
            '--process-seconds',str(seconds)])


def reusable_preflight(prior):
    path=FAMILY/'preflight.json'
    if not path.is_file() or not any(r.get('phase') in ('preflight','preflight_reused') and r.get('exit_status')==0
        for r in prior.get('records',[])): return False
    report=json.loads(path.read_text())
    if report.get('status')!='passed' or not report.get('passed') or report.get('actual_backend')!='gpu' or report.get('jax_version')!='0.4.38': return False
    return all((FAMILY/('preflight-source-'+name)).read_bytes()==(CHECKOUT/'scripts/hpc'/name).read_bytes()
        for name in ('check_daily_training_runtime.py','orcjax_gpu_runtime.py'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',choices=('1',),default='1')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true',help='precompile interface repair within exact first origin only')
    args=parser.parse_args(argv)
    if args.dry_run:
        if 'jax' in sys.modules: raise RuntimeError('dry-run imported JAX')
        print(json.dumps(dict(status='PREPARATION_ONLY_new_user_GPU_authorization_required',resume=args.resume,
            total_seconds=TOTAL_CAP,main_seconds=MAIN_CAP,kill_grace_seconds=GRACE,preflight_seconds=90,
            setup_assets_original_replay_seconds=300,combined_lower_backend_seconds=3100,
            candidate_first_raw_binary_proto_seconds=90,interleaved_pairs_seconds=180,
            maximum_candidate_compilations=1,reference_compilations=0,matched_pairs=3,
            pair_orders=[['reference','candidate'],['candidate','reference'],['reference','candidate']],
            pair_stop_rule='ABI/finite/physical/mask/memory/resource/budget terminal; strict gradient discrepancy does not suppress pairs',
            speed_pass_threshold=None,full_optimizer_update_speed_claim=False,
            all_caps_clipped_to_original_deadline=True,full_HLO_text_export=False,native_proto_export=True,
            proto_certificate='source/marker/SSA/shape scoped offline review; fail closed for unknown paths',
            first_origin='Bash timestamp before Python imports; resumes retain exact first origin',
            checkout=str(CHECKOUT),overlay=str(OVERLAY),output=str(FAMILY),
            full_parameter_coordinates=64884481,full_state_coordinates=3854,delta_inputs=False,custom_VJP=False,
            physical_precision='original FP64',forward_precision='original FP32',expert_chunk=128,
            GPU_calls=0,Teacher_reads=0,training_updates=0,reference_adoption=False,paid_jobs=0,CNY=0),indent=2))
        for label,command in commands(args.gpu).items(): print(label+': '+shlex.join(command))
        return 0
    if socket.gethostname().split('.')[0]!='ibc13b02n04' or os.environ.get('SLURM_JOB_ID'):
        raise ValueError('requires authorized free gln01 physicalGPU1 outside paid Slurm')
    if not os.environ.get('NORMALIZED_FANOUT_ORIGIN_UTC_SECONDS'):
        raise ValueError('use Bash entry to anchor first origin before Python imports')
    origin=float(os.environ['NORMALIZED_FANOUT_ORIGIN_UTC_SECONDS'])
    prior_records=[]; reuse=False
    if args.resume:
        prior=json.loads((FAMILY/'actual-exit.json').read_text()); origin=prior['original_start_UTC_seconds']
        if prior.get('candidate_compile_attempts',0) or (FAMILY/'model/candidate-first.pkl').exists():
            raise ValueError('candidate graph attempted or raw result saved; terminal/no retry')
        for declared,current in (('declared-policy-source.py',SOURCES[0]),('declared-component-source.py',SOURCES[-1])):
            if (FAMILY/declared).read_bytes()!=(OVERLAY/current).read_bytes():
                raise ValueError('ordinary interface repair cannot change declared native mechanism')
        if prior.get('candidate_status') in ('hard_admission_failed','strict_numerical_failed',
            'strict_numerical_passed_mechanism_review_pending','paired_measurement_incomplete'):
            raise ValueError('hard/scientific/resource terminal cannot resume')
        prior_records=prior['records']; reuse=reusable_preflight(prior)
    elapsed=time.time()-origin
    if elapsed<0 or elapsed>=MAIN_CAP: raise TimeoutError('first dispatch allowance exhausted/no deadline reset')
    started=time.perf_counter()-elapsed
    if args.resume:
        preserved=FAMILY/('preserved-precompile-attempt-'+str(len(prior_records))); preserved.mkdir(exist_ok=False)
        for name in ('actual-exit.json','candidate.log','model')+(() if reuse else ('preflight.log','preflight.json')):
            source=FAMILY/name
            if source.exists(): source.rename(preserved/name)
    else:
        FAMILY.mkdir(parents=True,exist_ok=False)
        (FAMILY/'origin.json').write_text(json.dumps(dict(original_start_UTC_seconds=origin,hard_deadline_UTC_seconds=origin+TOTAL_CAP)))
        (FAMILY/'declared-policy-source.py').write_bytes((OVERLAY/SOURCES[0]).read_bytes())
        (FAMILY/'declared-component-source.py').write_bytes((OVERLAY/SOURCES[-1]).read_bytes())
    snapshot=FAMILY/('source-attempt-'+str(len(prior_records))); snapshot.mkdir(exist_ok=False)
    for name in SOURCES: (snapshot/name).write_bytes((OVERLAY/name).read_bytes())
    records=[]; context={}; exception=None; status=1
    try:
        head=shared.setup_command(['git','rev-parse','HEAD'],started,records,cwd=CHECKOUT).strip()
        if head!=HEAD: raise ValueError('immutable source checkout differs')
        context=shared.read_context(args.gpu,started,records)
        if reuse:
            records.append(dict(phase='preflight_reused',exit_status=0,new_GPU_calls=0,same_original_window=True))
        else:
            for name in ('check_daily_training_runtime.py','orcjax_gpu_runtime.py'):
                (FAMILY/('preflight-source-'+name)).write_bytes((CHECKOUT/'scripts/hpc'/name).read_bytes())
            status=shared.run_phase(commands(args.gpu)['preflight'],'preflight',started,90,FAMILY,records)
            if status: return status
        remaining=shared.effective_limit(started,MAIN_CAP)
        status=shared.run_phase(commands(args.gpu,remaining)['candidate'],'candidate',started,MAIN_CAP,FAMILY,records)
        return status
    except BaseException as error:
        exception=repr(error); status=124 if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)) else 1
        raise
    finally:
        path=FAMILY/'model/model.json'; model=json.loads(path.read_text()) if path.exists() else {}
        elapsed=time.perf_counter()-started
        ledger=dict(exit_status=status,original_start_UTC_seconds=origin,hard_deadline_UTC_seconds=origin+TOTAL_CAP,
            family_seconds=elapsed,card_hours=elapsed/3600,total_cap_seconds=TOTAL_CAP,kill_grace_seconds=GRACE,
            candidate_compile_attempts=model.get('candidate_compile_attempts',0),candidate_status=model.get('status'),
            candidate_backend_attempts=model.get('backend_attempts',{}).get('candidate',0),
            completed_pairs=model.get('completed_pairs',0),records=prior_records+records,execution_context=context,
            family_exception=exception,runtime_preflight_reused=reuse,physical_GPU=1,GPU_count=1,
            paid_jobs=0,CNY=0,training_updates=0,reference_adoption=False,production_adoption=False)
        (FAMILY/'actual-exit.json').write_text(json.dumps(ledger,indent=2)+'\n'); print(json.dumps(ledger,indent=2))


if __name__=='__main__':
    raise SystemExit(main())
