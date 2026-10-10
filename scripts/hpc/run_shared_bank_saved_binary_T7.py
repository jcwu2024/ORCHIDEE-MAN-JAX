"""Pending added authorization: one480s free-GPU saved-binary acceptance."""
import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time
import run_shared_bank_adjoint_T7 as previous

ROOT=previous.ROOT
OVERLAY=ROOT/'runtime/transfer/shared-bank-adjoint-accumulation-20261002'
FAMILY=ROOT/'runtime/outputs/diagnostics/shared-bank-adjoint-saved-binary-T7-20261003'
previous.GLOBAL_CAP=480
TOTAL_CAP=480
MAIN_CAP=470
REQUIRE_SAVED_PROOF=False

def command(gpu,digest,seconds=470):
    prior=previous.commands(gpu)['model']
    entry=str(OVERLAY/'probe_shared_bank_adjoint_T7.py')
    prefix=prior[:prior.index(entry)]
    return prefix+[str(OVERLAY/'probe_shared_bank_saved_binary_T7.py'),'--root',str(ROOT),
        '--output-dir',str(FAMILY/'model'),'--candidate-sha256',digest,'--process-seconds',str(seconds)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',choices=('0','1'),default='1')
    parser.add_argument('--candidate-sha256',required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true',help='ordinary repair only; preserve original480s deadline')
    parser.add_argument('--artifact-only-repair',action='store_true',help='resume only; preserve compact HLO without repeated original loading')
    parser.add_argument('--saved-IR-proof-sha256')
    args=parser.parse_args()
    if len(args.candidate_sha256)!=64 or any(char not in '0123456789abcdef' for char in args.candidate_sha256):
        parser.error('requires exact64character lower-case SHA256')
    if args.artifact_only_repair and not args.resume:parser.error('artifact-only repair requires same-deadline --resume')
    if args.artifact_only_repair and args.saved_IR_proof_sha256:parser.error('reviewed-proof numerical route does not use artifact-only repair')
    if REQUIRE_SAVED_PROOF and args.saved_IR_proof_sha256 is None:parser.error('this route requires reviewed actual-IR proof SHA256')
    if args.saved_IR_proof_sha256 is not None and (len(args.saved_IR_proof_sha256)!=64 or any(char not in '0123456789abcdef' for char in args.saved_IR_proof_sha256)):
        parser.error('requires exact64character lower-case certificate SHA256')
    if args.dry_run:
        dry_command=command(args.gpu,args.candidate_sha256,MAIN_CAP)
        if args.artifact_only_repair:dry_command.append('--artifact-only-repair')
        if args.saved_IR_proof_sha256:dry_command.extend(['--saved-IR-proof-sha256',args.saved_IR_proof_sha256])
        print(json.dumps({'status':'explicit_additional_budget_required','total_cap_including_grace':TOTAL_CAP,'grace':10,
            'model_compilations':0,'new_runtime_preflights':0,'GPU_calls':0,'Teacher_reads':0,
            'source_family':'shared-bank-adjoint-accumulation-T7-20261002','output':str(FAMILY),
            'saved_IR_proof_sha256':args.saved_IR_proof_sha256,
            'compact_capture_cap':0 if args.saved_IR_proof_sha256 else 75,
            'resident_CPU_proof_cap':0 if args.saved_IR_proof_sha256 else 220,
            'estimated_memory_limit_GiB':24,'strict_rtol':1e-5,'strict_atol':1e-6},indent=2))
        print(shlex.join(dry_command))
        cli_command=[str(ROOT/'.venvs/orcjax_cpu/bin/python'),'-B',str(OVERLAY/'probe_shared_bank_saved_binary_T7.py'),
            '--cli-check-only','--candidate-sha256',args.candidate_sha256,'--process-seconds',str(MAIN_CAP)]
        if args.saved_IR_proof_sha256:cli_command.extend(['--saved-IR-proof-sha256',args.saved_IR_proof_sha256])
        subprocess.run(cli_command,check=True)
        return 0
    started=time.perf_counter();original_start=time.time();prior_records=[];prior_exits=[]
    if socket.gethostname().split('.')[0]!='ibc13b02n04' or __import__('os').environ.get('SLURM_JOB_ID'):
        raise ValueError('requires free gln01 outside paid Slurm')
    if args.resume:
        prior=json.loads((FAMILY/'actual-exit.json').read_text())
        if prior['candidate_sha256']!=args.candidate_sha256:raise ValueError('resume cannot change candidate')
        if prior.get('saved_IR_proof_sha256')!=args.saved_IR_proof_sha256:
            raise ValueError('resume cannot change reviewed actual-IR certificate')
        original_start=prior['original_start_UTC_seconds']
        elapsed=time.time()-original_start
        if elapsed<0 or elapsed>=MAIN_CAP:raise TimeoutError('original family allowance is exhausted; no reset')
        started=time.perf_counter()-elapsed;prior_records=prior['records'];prior_exits=prior.get('prior_exits',[])+[prior['exit_status']]
        model_path=FAMILY/'model/model.json'
        if model_path.exists() and json.loads(model_path.read_text())['status'] in ('hard_admission_failed','screening_failed','diagnostic_passed_no_adoption'):
            raise ValueError('scientific/hard result is terminal; resume only ordinary runtime/proof repair')
        preserved=FAMILY/f'preserved-attempt-{len(prior_exits)}';preserved.mkdir(exist_ok=False)
        for filename in ('model','model.log','actual-exit.json'):
            source=FAMILY/filename
            if source.exists():source.rename(preserved/filename)
    else:FAMILY.mkdir(parents=True,exist_ok=False)
    records=[];status=1;context={};exception=None
    try:
        head=previous.setup_command(['git','rev-parse','HEAD'],started,records,cwd=previous.CHECKOUT).strip()
        if head!='c599236bcf57b745e9bafed678298010f4c3b58f':raise ValueError('immutable checkout differs')
        context=previous.read_context(args.gpu,started,records)
        remaining=previous.effective_limit(started,MAIN_CAP)
        model_command=command(args.gpu,args.candidate_sha256,remaining)
        if args.artifact_only_repair:model_command.append('--artifact-only-repair')
        if args.saved_IR_proof_sha256:model_command.extend(['--saved-IR-proof-sha256',args.saved_IR_proof_sha256])
        status=previous.run_phase(model_command,'model',started,MAIN_CAP,FAMILY,records)
        if list((FAMILY/'model').glob('*.timeout.json')):status=124
        return status
    except BaseException as error:
        status=1;exception=repr(error);raise
    finally:
        seconds=time.perf_counter()-started
        ledger={'exit_status':status,'family_seconds':seconds,'card_hours':seconds/3600,'GPU_count':1,
            'physical_GPU':int(args.gpu),'paid_jobs':0,'CNY':0,'total_cap_including_grace':TOTAL_CAP,'kill_grace_seconds':10,
            'new_model_backend_attempts':0,'original_candidate_backend_attempts':1,'runtime_preflight_reused':True,
            'candidate_sha256':args.candidate_sha256,'saved_IR_proof_sha256':args.saved_IR_proof_sha256,
            'records':records,'execution_context':context,
            'original_start_UTC_seconds':original_start,'hard_deadline_UTC_seconds':original_start+TOTAL_CAP,
            'prior_exits':prior_exits,
            'family_exception':exception,'training_updates':0,'reference_adoption':False}
        ledger['records']=prior_records+records
        (FAMILY/'actual-exit.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(ledger,indent=2))

if __name__=='__main__':raise SystemExit(main())
