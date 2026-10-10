"""Resume the approved route with two source-backed representative shapes."""
import argparse
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time

ROOT=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
FAMILY=ROOT/'runtime/outputs/diagnostics/shared-parameter-gradient-precision-20261002'
KNOWN='UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_F16_F16_F32'
BASE_SECONDS=121.74309029775593

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--gpu',choices=('0','1'),default='1')
    parser.add_argument('--public-diagnostic-continue',action='store_true')
    parser.add_argument('--resume-known-repair',action='store_true')
    args=parser.parse_args()
    lines=(FAMILY/f'preparation-final-v2/dry-run-gpu{args.gpu}.txt').read_text().splitlines()
    interface=shlex.split(next(line for line in lines if line.startswith('interface cap600:')).split(': ',1)[1])
    daily=shlex.split(next(line for line in lines if line.startswith('conditional daily cap2400:')).split(': ',1)[1])
    if args.resume_known_repair and not args.public_diagnostic_continue:
        parser.error('known repair resume belongs to explicit public diagnostic continuation')
    previous_public=json.loads((FAMILY/'public-F4S/actual-exit.json').read_text()) if args.resume_known_repair else None
    if previous_public is not None:
        assert previous_public['compatibility_repairs']==1 and previous_public['public_backend_attempts']==1
        assert KNOWN in (FAMILY/'public-F4S/daily.log').read_text()
    destination=FAMILY/('public-F4S-repaired' if args.resume_known_repair else 'public-F4S' if args.public_diagnostic_continue else 'corrected-actual-f4s')
    prior_seconds=(json.loads((FAMILY/'corrected-actual-f4s/actual-exit.json').read_text())['family_total_seconds']
        if args.public_diagnostic_continue else BASE_SECONDS)
    initial_report=FAMILY/'corrected-actual-f4s/interface/interface.json' if args.public_diagnostic_continue else destination/'interface/interface.json'
    if args.resume_known_repair:
        prior_seconds=previous_public['family_total_seconds']
        initial_report=FAMILY/'public-F4S/interface-operand-upcaster/interface.json'
        public_remaining=2400-previous_public['corrected_stage_wall_seconds']
        flag_index=next(i for i,value in enumerate(daily) if value.startswith('SINGULARITYENV_XLA_FLAGS='))
        daily[flag_index]+=' --xla_disable_hlo_passes=operand_upcaster'
    cpu_report=FAMILY/'corrected-cpu-interface/interface.json'
    interface[interface.index('--output-dir')+1]=str(destination/'interface')
    interface+=['--actual-f4s-shapes','--cpu-interface-report',str(cpu_report)]
    def daily_command(report,stem):
        command=daily.copy()
        replacements={'--interface-report':str(report),'--output':str(destination/(stem+'.json')),
            '--result-output':str(destination/(stem+'.candidate.pkl')),
            '--executable-output':str(destination/(stem+'.candidate.executable.pkl'))}
        for flag,value in replacements.items():
            command[command.index(flag)+1]=value
        if args.public_diagnostic_continue:
            command+=['--component-speed-screen-failed-diagnostic-continue']
        return command
    if args.dry_run:
        print(json.dumps({'interface':interface if not args.public_diagnostic_continue else 'not_repeated',
            'daily':daily_command(initial_report,'daily'),
            'corrected_interface_including_grace_cap':550,'public_including_grace_cap':public_remaining if args.resume_known_repair else 2400,
            'family_base_seconds':prior_seconds,'family_cap':3500},indent=2))
        script=next(i for i,value in enumerate(daily) if value.endswith('/probe_shared_parameter_gradient_precision_daily.py'))
        subprocess.run([__import__('sys').executable,'-B',*daily_command(initial_report,'daily')[script:],'--cli-check-only'],check=True)
        return
    assert socket.gethostname().split('.')[0]=='ibc13b02n04'
    assert not destination.exists(),'preserve earlier corrected-stage assets'
    occupancy=subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    assert occupancy=='0, 0',occupancy
    assert not subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    destination.mkdir()
    started=time.perf_counter()
    phase_started=started
    repair=1 if args.resume_known_repair else 0
    public_attempts=1 if args.resume_known_repair else 0
    phase='interface'
    outcome='incomplete'
    phase_cap=550
    records=[]
    context={label:subprocess.run(command,text=True,capture_output=True).stdout for label,command in (
        ('hostname',['hostname']),('host_load',['uptime']),
        ('CPU_affinity',['grep','Cpus_allowed_list','/proc/self/status']),
        ('visible_GPU_processes',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']))}
    def run(command,label):
        remaining=min(phase_cap-(time.perf_counter()-phase_started),3500-prior_seconds-(time.perf_counter()-started))-10
        if remaining<1:
            return 124
        main_limit=int(remaining)
        with (destination/(label+'.log')).open('x') as output:
            began=time.perf_counter()
            result=subprocess.run(['timeout','--signal=TERM','--kill-after=10s',str(main_limit)+'s',*command],stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        records.append({'phase':label,'exit_status':result.returncode,'main_timeout_seconds':main_limit,
            'elapsed_seconds':time.perf_counter()-began,'command':command})
        return result.returncode
    def repaired(command):
        adjusted=command.copy()
        index=next(i for i,value in enumerate(adjusted) if value.startswith('SINGULARITYENV_XLA_FLAGS='))
        adjusted[index]+=' --xla_disable_hlo_passes=operand_upcaster'
        return adjusted
    selected=interface
    report=initial_report
    status=1
    try:
        status=0 if args.public_diagnostic_continue else run(selected,'interface')
        if not args.public_diagnostic_continue and status and KNOWN in (destination/'interface.log').read_text():
            repair=1
            selected=repaired(interface)
            selected[selected.index('--output-dir')+1]=str(destination/'interface-operand-upcaster')
            report=destination/'interface-operand-upcaster/interface.json'
            status=run(selected,'interface-operand-upcaster')
        if status:
            outcome='corrected_component_gate_failed_or_incomplete'
            return status
        phase='public_F4S'
        phase_started=time.perf_counter()
        phase_cap=public_remaining if args.resume_known_repair else 2400
        candidate=daily_command(report,'daily')
        if repair and not args.resume_known_repair:
            candidate=repaired(candidate)
        public_attempts=2 if args.resume_known_repair else 1
        status=run(candidate,'daily')
        if status and repair==0 and KNOWN in (destination/'daily.log').read_text():
            repair=1
            selected=repaired(interface)
            selected[selected.index('--output-dir')+1]=str(destination/'interface-operand-upcaster')
            report=destination/'interface-operand-upcaster/interface.json'
            status=run(selected,'public-compatibility-interface')
            if status==0:
                public_attempts=2
                status=run(repaired(daily_command(report,'daily-operand-upcaster')),'daily-operand-upcaster')
        outcome='public_F4S_screen_passed_no_adoption' if status==0 else 'public_F4S_screen_failed_or_incomplete'
        return status
    finally:
        elapsed=time.perf_counter()-started
        ledger={'exit_status':status,'phase':phase,'outcome':outcome,'corrected_stage_wall_seconds':elapsed,
            'prior_family_seconds':prior_seconds,'family_total_seconds':prior_seconds+elapsed,
            'family_card_hours':(prior_seconds+elapsed)/3600,'family_cap_seconds':3500,
            'component_speed_screen_failed_diagnostic_continue':args.public_diagnostic_continue,
            'resume_known_repair_without_repeating_interface':args.resume_known_repair,
            'current_public_phase_cap_including_grace_seconds':phase_cap,
            'original_interface_consumed_upper_bound_seconds':49.0265195665035,
            'corrected_interface_cap_including_grace_seconds':550,'public_phase_cap_including_grace_seconds':2400,
            'physical_GPU':int(args.gpu),'occupancy_before':occupancy,'compatibility_repairs':repair,
            'public_backend_attempts':public_attempts,'paid_jobs':0,'CNY':0,'records':records,'execution_context':context}
        (destination/'actual-exit.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(ledger,indent=2))

if __name__=='__main__':
    raise SystemExit(main())
