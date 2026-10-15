"""One-shot 90-minute supervisor for the current B1/T7 precision probe.

This module owns scheduler admission, GPU discovery, bounded child phases,
accounting and release verification. It never submits a job.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

_ENTRY_ROOT = Path(__file__).resolve().parents[2]
if str(_ENTRY_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENTRY_ROOT))

from scripts.hpc import probe_shared_current_T7_precision_admission as probe
from scripts.hpc import run_shared_latent_T7_batch_numerics_paid as paid
from scripts.hpc import run_shared_latent_T7_batch_numerics as free
from scripts.hpc.shared_affine_gpu_admission import AffineScopeLock, SCOPE_LOCK

ROOT = probe.ROOT
TOTAL_SECONDS = 5400
CAPS = dict(load=300, preflight=240, lower=300, compile=3000, calls=1380, reserve=180)
WORK_SECONDS = TOTAL_SECONDS - CAPS['reserve']
MAX_COST_CNY = 3.72
GPU_PYTHON = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python'
SLURM = paid.SLURM


def _allocated_tres(job, *, expected_node):
    values, sources = paid.allocated_tres(job, expected_node=expected_node)
    if values.get('cpu') != '4' or values.get('gres/gpu') != '1' or values.get('node') != '1':
        raise ValueError('actual allocated TRES CPU/GPU/node count differs')
    typed = [value for key, value in values.items() if key.startswith('gres/gpu:')]
    if len(typed) > 1 or any(value != '1' for value in typed):
        raise ValueError('typed GPU allocation differs')
    return values, sources


def validate_allocation(environment, job, hostname, *, now=None):
    """Validate live 1xV100/4-CPU/90-minute nonexclusive scheduler allocation."""
    node_list = job.get('NodeList', '')
    node_field = job.get('Nodes')
    allocated_node = node_list.split(',')[0].split('.')[0]
    required = dict(JobId=environment.get('SLURM_JOB_ID'), JobState='RUNNING',
        Partition='gnall', NumNodes='1', NumCPUs='4', NumTasks='1',
        Requeue='0', TimeLimit='01:30:00')
    if (not str(required['JobId'] or '').isdigit()
            or not node_list or ',' in node_list or '[' in node_list or ']' in node_list
            or not re.fullmatch(r'[A-Za-z0-9_-]+', allocated_node)
            or (node_field is not None and node_field != allocated_node)
            or hostname.split('.')[0] != allocated_node
            or any(job.get(key) != value for key, value in required.items())
            or environment.get('SLURM_CPUS_PER_TASK') != '4'
            or environment.get('SLURM_NTASKS') != '1'
            or environment.get('SLURM_JOB_NUM_NODES') != '1'
            or environment.get('SLURM_JOB_PARTITION') != 'gnall'
            or environment.get('SLURM_GPUS_ON_NODE', '1') != '1'
            or job.get('OverSubscribe') != 'OK'):
        raise ValueError('actual Slurm job/node/resources differ from 90-minute scope')
    allocation, sources = _allocated_tres(job, expected_node=allocated_node)
    details = [job[key] for key in ('GresDetail', 'GRES') if key in job]
    physical_gpu = environment.get('SLURM_JOB_GPUS')
    if (not physical_gpu or not details or len(set(details)) != 1
            or any(not re.fullmatch(r'gpu(?::[A-Za-z0-9_.-]+)?:1\(IDX:[0-9]+\)', detail)
                   or re.findall(r'IDX:([0-9]+)', detail) != [physical_gpu]
                   for detail in details)):
        raise ValueError('allocated GPU detail differs from scheduler environment')
    starts = paid.datetime.datetime.fromisoformat(job['StartTime']).timestamp()
    ends = paid.datetime.datetime.fromisoformat(job['EndTime']).timestamp()
    current = time.time() if now is None else now
    runtime = paid.seconds(job.get('RunTime', ''))
    if (ends - starts != TOTAL_SECONDS or starts > current or current >= ends
            or runtime >= WORK_SECONDS or abs(current - starts - runtime) > 15):
        raise ValueError('scheduler allocation clock differs or release reserve is exhausted')
    for key, epoch in (('SLURM_JOB_START_TIME', starts), ('SLURM_JOB_END_TIME', ends)):
        if key in environment and (not str(environment[key]).isdigit() or int(environment[key]) != int(epoch)):
            raise ValueError('original scheduler epoch differs from live allocation')
    return dict(job=job, allocated_node=allocated_node, start_epoch=starts, end_epoch=ends, allocated_tres=allocation,
        tres_fields=sources, physical_gpu=physical_gpu, nonexclusive=True)


def _diagnostic_text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def _save_allocation_diagnostic(path, observation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    free.atomic_json(path, observation)


def read_live_allocation(environment=None, *, diagnostic_output=None):
    environment = os.environ if environment is None else environment
    job_id = environment.get('SLURM_JOB_ID', '')
    if not job_id.isdigit():
        raise ValueError('numeric live Slurm job ID required')
    command = [SLURM, 'show', 'job', '--details', '-o', job_id]
    hostname = socket.gethostname()
    observation = dict(schema_version='current_T7_live_allocation_diagnostic_v1',
        status='scheduler_query_started', command=command, hostname=hostname, job_id=job_id)
    path = None if diagnostic_output is None else Path(diagnostic_output)
    if path is not None and path.exists():
        raise FileExistsError('preserve existing live allocation diagnostic')
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=4, check=False)
    except Exception as error:
        observation.update(status='scheduler_query_failed',
            query_error_type=type(error).__name__, query_error=str(error),
            stdout=_diagnostic_text(getattr(error, 'stdout', getattr(error, 'output', None))),
            stderr=_diagnostic_text(getattr(error, 'stderr', None)))
        if path is not None:
            _save_allocation_diagnostic(path, observation)
        raise
    observation.update(status='scheduler_response_saved_before_validation',
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if path is not None:
        _save_allocation_diagnostic(path, observation)
    if result.returncode:
        observation.update(status='scheduler_command_failed',
            query_error_type='CalledProcessError', query_error=f'scontrol exited {result.returncode}')
        if path is not None:
            _save_allocation_diagnostic(path, observation)
        raise subprocess.CalledProcessError(result.returncode, command,
            output=result.stdout, stderr=result.stderr)
    try:
        job = paid.parse_job(result.stdout)
        observation['parsed_job'] = job
        allocation = validate_allocation(environment, job, hostname)
    except Exception as error:
        observation.update(status='allocation_validation_failed',
            validation_error_type=type(error).__name__, validation_error=str(error))
        if path is not None:
            _save_allocation_diagnostic(path, observation)
        raise
    observation.update(status='allocation_validated',
        allocated_node=allocation['allocated_node'],
        allocated_tres=allocation['allocated_tres'],
        allocated_tres_fields=allocation['tres_fields'],
        physical_gpu=allocation['physical_gpu'])
    if path is not None:
        _save_allocation_diagnostic(path, observation)
    return allocation


def quote_cost(allocation_seconds):
    seconds = max(float(allocation_seconds), 0.0)
    cpu = 4 * seconds / 3600 * 0.07
    gpu = seconds / 3600 * 2.20
    return dict(allocation_seconds=seconds, cpu_hours=4 * seconds / 3600,
        gpu_hours=seconds / 3600, cpu_cny=cpu, gpu_cny=gpu,
        allocation_cost_estimate_cny=cpu + gpu,
        approved_cost_ceiling_cny=MAX_COST_CNY,
        ceiling_exposure_cny=min(cpu + gpu, MAX_COST_CNY),
        actual_bill_known=False)


def build_plan(args, argv):
    output = args.output.resolve()
    ledger = output.with_suffix('.ledger.json')
    supervisor_output = output.with_suffix('.supervisor.json')
    preflight_output = (args.runtime_preflight or output.with_suffix('.runtime-preflight.json')).resolve()
    discovery_output = output.with_suffix('.discovery.json')
    allocation_diagnostic_output = output.with_suffix('.allocation-diagnostic.json')
    shell = ROOT / 'scripts/hpc/run_shared_current_T7_precision_admission.sh'
    child = lambda entry, child_args: ['bash', str(shell), '--runtime-child', GPU_PYTHON,
        str(ROOT / 'scripts/hpc' / entry), *child_args]
    child_args = list(argv)
    commands = {
        'discovery': child('shared_affine_cuda_identity.py', ['--discover-output', str(discovery_output)]),
        'preflight': child('check_daily_training_runtime.py', ['--backend', 'gpu', '--output', str(preflight_output)]),
        'probe': child('probe_shared_current_T7_precision_admission.py', child_args),
    }
    scope = dict(scope_id='current-T7-precision-' + os.environ.get('SLURM_JOB_ID', 'unbound'),
        gpu_host_index=None, gpu_uuid=None, caps=dict(CAPS, total_seconds=TOTAL_SECONDS,
        release_reserve_seconds=CAPS['reserve'], memory_limit_mib=24576,
        load_seconds=CAPS['load'], preflight_seconds=CAPS['preflight'],
        lower_seconds=CAPS['lower'], compile_seconds=CAPS['compile'], calls_seconds=CAPS['calls']))
    return dict(scope=scope, checkout=str(ROOT), output=str(output), ledger=str(ledger),
        supervisor_output=str(supervisor_output), preflight_output=str(preflight_output),
        discovery_output=str(discovery_output), allocation_diagnostic_output=str(allocation_diagnostic_output),
        shell_started_epoch=os.environ.get('CURRENT_T7_SHELL_STARTED_EPOCH'),
        commands=commands, phase_environment={}, paid_startup_observation=paid.startup_observation(os.environ))


class CurrentSupervisor(paid.PaidSupervisor):
    """90-minute owner reusing paid discovery, phase and UUID monitor behavior."""
    def __init__(self, plan, **kwargs):
        super().__init__(plan, **kwargs)
        self.record.update(schema_version='current_T7_precision_supervisor_v1',
            reserved_gpu_seconds=TOTAL_SECONDS, reserved_allocation_seconds=TOTAL_SECONDS,
            GPU_executed=False, inherited_old_budget=False)

    def phase(self, name, seconds, **kwargs):
        if name == 'preflight':
            seconds = min(seconds, CAPS['preflight'])
        if name == 'probe':
            seconds = min(seconds, WORK_SECONDS - CAPS['preflight'])
        environment = dict(kwargs.pop('environment', {}) or {})
        original_cvd = os.environ.get('ORCJAX_SLURM_VISIBLE_DEVICES', '')
        if not (original_cvd.isdigit() or paid.cuda_identity.valid_uuid(original_cvd)):
            raise ValueError('single original scheduler GPU mask required for every child phase')
        if name == 'discovery':
            expected_cvd = original_cvd
        else:
            expected_cvd = self.plan['scope'].get('gpu_uuid')
            if not paid.cuda_identity.valid_uuid(expected_cvd):
                raise ValueError('discovered GPU UUID required for post-discovery child phases')
        inherited_cvd = environment.get('CUDA_VISIBLE_DEVICES')
        if inherited_cvd is not None and inherited_cvd != expected_cvd:
            raise ValueError('child CUDA_VISIBLE_DEVICES differs from its admitted phase identity')
        environment['CUDA_VISIBLE_DEVICES'] = expected_cvd
        environment['ORCJAX_CURRENT_T7_SUPERVISED'] = '1'
        started = self.plan.get('shell_started_epoch', self.plan['allocation']['start_epoch'])
        environment['CURRENT_T7_SHELL_STARTED_EPOCH'] = str(int(started))
        kwargs['environment'] = environment
        return super().phase(name, seconds, **kwargs)

    def run(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open('x', encoding='utf-8') as stream:
            json.dump(dict(schema_version='current_T7_precision_one_shot_ledger_v1',
                status='reserved', reservation_seconds=TOTAL_SECONDS, retry_allowed=False), stream)
        if self.watchdog_enabled:
            import threading
            threading.Thread(target=self.watchdog, daemon=True).start()
        primary = ('runtime_failed', 1)
        release_verified = False
        try:
            self.snapshot()
            primary = self.startup()
            if primary is None:
                primary = self.sample(initial=True)
                if primary is None:
                    primary = self.phase('preflight', CAPS['preflight'])
            if primary[1] == 0:
                preflight = self.read_report(self.plan['preflight_output'], 'preflight_result_admission')
                expected_uuid = self.plan['scope'].get('gpu_uuid')
                from scripts.hpc.shared_affine_cuda_identity import binding_report_passes
                binding = preflight.get('affine_gpu_identity', {})
                if (preflight.get('passed') is not True or preflight.get('actual_backend') != 'gpu'
                        or preflight.get('jax_version') != '0.4.38'
                        or len(preflight.get('devices', [])) != 1 or not expected_uuid
                        or not binding_report_passes(binding, expected_uuid,
                            self.plan.get('phase_environment', {}).get('ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT'))):
                    raise ValueError('GPU JAX 0.4.38/single-device UUID-bound preflight failed')
                primary = self.phase('probe', WORK_SECONDS - CAPS['preflight'])
        except (TimeoutError, subprocess.TimeoutExpired) as error:
            primary = ('deadline_exceeded', 124)
            self.record['error'] = str(error)
        except Exception as error:
            primary = ('runtime_failed', 1)
            self.record.update(error_type=type(error).__name__, error=str(error))
        finally:
            groups_released = self.cleanup()
            try:
                final = self.monitor.sample()
                self.record['samples'].append(final)
                release_verified = (groups_released and not final['processes']
                    and final['memory_mib'] == 0 and final['utilization_percent'] == 0
                    and self.clock() <= self.deadline)
                self.record['release_observation'] = final
            except Exception as error:
                self.record['release_error'] = str(error)
            elapsed = max(time.time() - self.plan['allocation']['start_epoch'], 0)
            cost = quote_cost(elapsed)
            result = primary[1] if primary[1] else 0 if release_verified else 125
            self.record.update(primary_status=primary[0], primary_exit_code=primary[1],
                release_verified=release_verified, process_groups_released=groups_released,
                occupied_seconds=elapsed, occupied_gpu_seconds=elapsed, cost=cost,
                within_total_cap=elapsed <= TOTAL_SECONDS, ended_epoch=time.time(),
                status=primary[0] if result == primary[1] else 'resource_unverified', exit_code=result)
            ledger = dict(schema_version='current_T7_precision_one_shot_ledger_v1',
                status='closed' if release_verified else 'closed_release_unverified',
                reservation_seconds=TOTAL_SECONDS, actual_allocation_seconds=elapsed,
                allocation_cost_estimate_cny=cost['allocation_cost_estimate_cny'],
                approved_cost_ceiling_cny=MAX_COST_CNY, actual_bill_known=False,
                primary_status=primary[0],
                exit_code=result, release_verified=release_verified,
                supervisor_receipt=str(self.out), retry_allowed=False)
            free.atomic_json(self.ledger, ledger)
            self.snapshot()
            self.done.set()
        return result


def run_live(args, argv, *, allocation_reader=None, lock_factory=AffineScopeLock,
             supervisor_factory=CurrentSupervisor):
    approval = probe.check_approval(args, argv)
    if allocation_reader is None:
        diagnostic_output = args.output.resolve().with_suffix('.allocation-diagnostic.json')
        allocation = read_live_allocation(diagnostic_output=diagnostic_output)
    else:
        allocation = allocation_reader()
    plan = build_plan(args, argv)
    plan['allocation'] = allocation
    plan['slurm_allocation'] = allocation
    plan['shell_started_epoch'] = int(allocation['start_epoch'])
    observation = plan['paid_startup_observation']
    lock = lock_factory(SCOPE_LOCK, plan['scope']['scope_id']).acquire()
    supervisor = None
    release_verified = False
    try:
        existing = (args.output, Path(plan['ledger']), Path(plan['supervisor_output']),
                    Path(plan['preflight_output']), Path(plan['discovery_output']))
        if any(path.exists() for path in existing):
            raise FileExistsError('preserve existing one-shot output; retry prohibited')
        plan['phase_environment'] = {
            'ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT': paid.GPU_EXECUTION_CONTRACT,
        }
        plan['paid_startup_observation']['environment'] = {
            key: os.environ.get(key) for key in paid.EVIDENCE_ENVIRONMENT
        }
        supervisor = supervisor_factory(plan)
        supervisor.record['allocation'] = allocation
        supervisor.record['approval_sha256'] = probe.sha(args.approval_report)
        supervisor.record['approved_cost_ceiling_cny'] = MAX_COST_CNY
        supervisor.record['approval_checked'] = bool(approval)
        result = supervisor.run()
        release_verified = supervisor.record.get('release_verified') is True
        return result
    finally:
        lock.release(release_verified=release_verified or supervisor is None)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = probe.parser().parse_args(argv)
    if args.stage == 'dry-run':
        return subprocess.run([sys.executable, str(Path(probe.__file__).resolve()), *argv], check=False).returncode
    if args.check_approval_only:
        probe.check_approval(args, argv)
        print(json.dumps({'approval_checked': True, 'GPU_executed': False}))
        return 0
    return run_live(args, argv)


if __name__ == '__main__':
    raise SystemExit(main())
