import importlib.util
import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[2] / 'scripts/hpc/run_shared_current_T7_precision_admission.py'
_SPEC = importlib.util.spec_from_file_location('run_shared_current_T7_precision_admission', _PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def cli_args(stage='execute', output=None):
    output = Path(output or 'result.json')
    return [
        '--stage', stage,
        '--parameters', 'parameters.pkl',
        '--prepared-context', 'context.pkl',
        '--training-prepared-manifest', 'training.json',
        '--dataset-manifest', 'dataset.json',
        '--saved-identity-receipt', 'identity.json',
        '--state-input-metadata', 'state.json',
        '--turnover-overlay', 'turnover.py',
        '--approval-report', str(output.parent / 'approval.json'),
        '--runtime-preflight', 'preflight.json',
        '--output', str(output),
    ]


def valid_allocation(now=1_800_000_000):
    start = now - 30
    end = start + runner.TOTAL_SECONDS
    fmt = lambda epoch: datetime.fromtimestamp(epoch, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S+00:00')
    env = {
        'SLURM_JOB_ID': '77881', 'SLURM_CPUS_PER_TASK': '4', 'SLURM_NTASKS': '1',
        'SLURM_JOB_NUM_NODES': '1', 'SLURM_JOB_PARTITION': 'gnall',
        'SLURM_GPUS_ON_NODE': '1', 'SLURM_JOB_GPUS': '0',
    }
    job = {
        'JobId': '77881', 'JobState': 'RUNNING', 'Partition': 'gnall', 'NodeList': 'gln01',
        'NumNodes': '1', 'NumCPUs': '4', 'NumTasks': '1', 'Requeue': '0',
        'TimeLimit': '01:30:00', 'OverSubscribe': 'OK', 'RunTime': '00:00:30',
        'StartTime': fmt(start), 'EndTime': fmt(end),
        'AllocTRES': 'cpu=4,gres/gpu=1,gres/gpu:v100=1,node=1',
        'GresDetail': 'gpu:v100:1(IDX:0)',
    }
    return env, job, now


def test_dry_run_forwards_exact_arguments_to_host_probe(monkeypatch, tmp_path):
    argv = cli_args('dry-run', tmp_path / 'out.json')
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=7))
    assert runner.main(argv) == 7
    assert calls == [([runner.sys.executable, str(Path(runner.probe.__file__).resolve()), *argv], {'check': False})]


def test_approval_is_checked_before_allocation_and_gpu_supervisor(monkeypatch, tmp_path):
    events = []
    args = runner.probe.parser().parse_args(cli_args(output=tmp_path / 'out.json'))
    args.approval_report.write_text('{}')
    monkeypatch.setattr(runner.probe, 'check_approval', lambda parsed, argv: events.append('approval') or {'approved': True})
    monkeypatch.setattr(runner, 'read_live_allocation', lambda: events.append('allocation') or {'start_epoch': 1})

    class Lock:
        def __init__(self, *args):
            events.append('lock_constructed')
        def acquire(self):
            events.append('lock_acquired')
            return self
        def release(self, *, release_verified):
            events.append(('lock_released', release_verified))

    class Supervisor:
        def __init__(self, plan):
            events.append('supervisor_constructed')
            self.record = {'release_verified': True}
        def run(self):
            events.append('run')
            return 0

    assert runner.run_live(args, cli_args(output=args.output), allocation_reader=lambda: events.append('allocation') or {'start_epoch': 1}, lock_factory=Lock,
                           supervisor_factory=Supervisor) == 0
    assert events[:4] == ['approval', 'allocation', 'lock_constructed', 'lock_acquired']
    assert events.index('approval') < events.index('allocation') < events.index('supervisor_constructed')
    assert events[-1] == ('lock_released', True)


def test_allocation_rejects_wrong_wall_limit_and_nonexclusive_job():
    env, job, now = valid_allocation()
    job['TimeLimit'] = '01:29:00'
    with pytest.raises(ValueError, match='resources'):
        runner.validate_allocation(env, job, 'gln01', now=now)
    env, job, now = valid_allocation()
    job['OverSubscribe'] = 'NO'
    with pytest.raises(ValueError, match='resources'):
        runner.validate_allocation(env, job, 'gln01', now=now)


def test_live_allocation_rejection_stops_before_lock_or_supervisor(monkeypatch, tmp_path):
    events = []
    args = runner.probe.parser().parse_args(cli_args(output=tmp_path / 'out.json'))
    monkeypatch.setattr(runner.probe, 'check_approval', lambda *args: events.append('approval') or {})

    def reject_allocation():
        events.append('allocation')
        raise ValueError('allocation rejected')

    class Unexpected:
        def __init__(self, *args):
            events.append('lock_or_supervisor_constructed')

    with pytest.raises(ValueError, match='allocation rejected'):
        runner.run_live(args, cli_args(output=args.output), allocation_reader=reject_allocation,
                        lock_factory=Unexpected, supervisor_factory=Unexpected)
    assert events == ['approval', 'allocation']


def test_allocation_accepts_exact_scope_and_cost_is_capped():
    env, job, now = valid_allocation()
    allocation = runner.validate_allocation(env, job, 'gln01', now=now)
    assert allocation['physical_gpu'] == '0'
    assert runner.quote_cost(runner.TOTAL_SECONDS)['allocation_cost_estimate_cny'] == pytest.approx(3.72)
    overrun = runner.quote_cost(runner.TOTAL_SECONDS + 500)
    assert overrun['allocation_cost_estimate_cny'] > 3.72
    assert overrun['ceiling_exposure_cny'] == pytest.approx(3.72)
    assert overrun['actual_bill_known'] is False
    assert runner.quote_cost(runner.TOTAL_SECONDS + 500)['allocation_seconds'] == runner.TOTAL_SECONDS + 500


def _plan(tmp_path):
    output = tmp_path / 'result.json'
    allocation = {'start_epoch': __import__('time').time() - 10, 'end_epoch': __import__('time').time() + 5390}
    return {
        'scope': {'scope_id': 'test-scope', 'gpu_host_index': None, 'gpu_uuid': None,
            'caps': dict(runner.CAPS, total_seconds=runner.TOTAL_SECONDS,
                release_reserve_seconds=runner.CAPS['reserve'], memory_limit_mib=24576,
                load_seconds=runner.CAPS['load'], preflight_seconds=runner.CAPS['preflight'],
                lower_seconds=runner.CAPS['lower'], compile_seconds=runner.CAPS['compile'],
                calls_seconds=runner.CAPS['calls'])},
        'checkout': str(tmp_path), 'output': str(output),
        'ledger': str(output.with_suffix('.ledger.json')),
        'supervisor_output': str(output.with_suffix('.supervisor.json')),
        'preflight_output': str(output.with_suffix('.runtime-preflight.json')),
        'discovery_output': str(output.with_suffix('.discovery.json')),
        'commands': {'probe': ['mock-probe'], 'preflight': ['mock-preflight']},
        'phase_environment': {}, 'paid_startup_observation': {'files': [], 'commands': []}, 'allocation': allocation,
    }


class IdleMonitor:
    def sample(self):
        return {'processes': [], 'memory_mib': 0, 'utilization_percent': 0, 'gpu_uuid': 'GPU-test'}


class PassingSupervisor(runner.CurrentSupervisor):
    def startup(self):
        self.plan['scope']['gpu_uuid'] = 'GPU-test'
        self.monitor = IdleMonitor()
        return None

    def phase(self, name, seconds, **kwargs):
        self.phase_names.append((name, seconds))
        if name == 'preflight':
            Path(self.plan['preflight_output']).write_text(json.dumps({
                'passed': True, 'actual_backend': 'gpu', 'jax_version': '0.4.38',
                'devices': ['gpu:0'], 'affine_gpu_identity': {'passed': True},
            }))
        return 'passed', 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_names = []


def test_inherited_startup_uses_scheduler_job_and_accepts_discovery_mapping(monkeypatch, tmp_path):
    from scripts.hpc import shared_affine_cuda_identity

    plan = _plan(tmp_path)
    plan['slurm_allocation'] = {'job': {'JobId': '77881'}}
    plan['shell_started_epoch'] = 1_800_000_000
    uuid = 'GPU-00000000-0000-0000-0000-000000000001'
    monkeypatch.setenv('ORCJAX_SLURM_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('SLURM_JOB_GPUS', '0')
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    monkeypatch.setattr(runner.paid.os, 'major', lambda value: 195, raising=False)
    monkeypatch.setattr(runner.paid.os, 'minor', lambda value: 0, raising=False)
    monkeypatch.setattr(runner.paid, 'observe_gpu_mapping', lambda *args, **kwargs: (
        {'minor_number': 0, 'gpu_host_index': 3}, {'status': 'mapped'}))
    monkeypatch.setattr(runner.paid, '_kernel_stat', lambda *args, **kwargs: SimpleNamespace(
        st_mode=stat.S_IFCHR, st_rdev=0))
    monkeypatch.setattr(runner.paid, 'read_cgroup_rules', lambda *args, **kwargs: ('/devices.list', []))
    monkeypatch.setattr(runner.paid, 'admit_gpu_mapping', lambda *args, **kwargs: {
        'gpu_host_index': 3, 'gpu_uuid': uuid, 'status': 'admitted'})

    supervisor = runner.CurrentSupervisor(plan, watchdog=False, clock=lambda: 0)
    shared_preflight_deadline = supervisor.startup_preflight_deadline
    phase_calls = []

    def discovery_phase(name, seconds, **kwargs):
        phase_calls.append((name, seconds, kwargs))
        Path(plan['discovery_output']).write_text(json.dumps({
            'status': 'passed',
            'identity': {'passed': True, 'original_CUDA_VISIBLE_DEVICES': '0',
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'slurm_gpu_id': '0',
                'allocation_association': 'trusted_scheduler_original_CVD_same_runtime_namespace',
                'gpu_uuid': uuid},
            'runtime_identity': {'runtime_device_count': 1, 'runtime_visible_ordinal': 0,
                'runtime_pci_bus_id': '0000:3b:00.0'},
            'driver_identity': {'driver_device_count': 1, 'driver_gpu_uuid': uuid,
                'driver_pci_bus_id': '0000:3b:00.0'},
        }))
        return 'passed', 0

    supervisor.phase = discovery_phase
    assert runner.paid.PaidSupervisor.startup(supervisor) is None
    assert supervisor.startup_preflight_deadline == shared_preflight_deadline
    assert phase_calls[0][0:2] == ('discovery', runner.CAPS['preflight'])
    assert phase_calls[0][2]['environment']['CUDA_VISIBLE_DEVICES'] == '0'
    assert plan['scope']['gpu_host_index'] == 3
    assert plan['scope']['gpu_uuid'] == uuid
    assert plan['slurm_gpu_mapping']['gpu_uuid'] == uuid
    assert isinstance(supervisor.monitor, runner.paid.PaidNvidiaMonitor)
    assert supervisor.monitor.select_by_uuid is True
    del supervisor.phase
    captured = {}
    monkeypatch.setattr(runner.free.Supervisor, 'phase', lambda self, name, seconds, **kwargs:
        captured.update(name=name, seconds=seconds, environment=kwargs['environment']) or ('passed', 0))
    assert supervisor.phase('preflight', runner.CAPS['preflight']) == ('passed', 0)
    assert captured['environment']['CUDA_VISIBLE_DEVICES'] == uuid
    assert captured['environment']['ORCJAX_CURRENT_T7_SUPERVISED'] == '1'
    assert captured['environment']['CURRENT_T7_SHELL_STARTED_EPOCH'] == str(plan['shell_started_epoch'])


@pytest.mark.parametrize('phase_name,expected_cvd', [
    ('discovery', '0'),
    ('preflight', 'GPU-00000000-0000-0000-0000-000000000001'),
    ('probe', 'GPU-00000000-0000-0000-0000-000000000001'),
])
def test_phase_forwards_stage_identity_allocation_start_and_supervised_marker(
        monkeypatch, tmp_path, phase_name, expected_cvd):
    plan = _plan(tmp_path)
    plan['shell_started_epoch'] = 1_800_000_000
    plan['scope']['gpu_uuid'] = 'GPU-00000000-0000-0000-0000-000000000001'
    monkeypatch.setenv('ORCJAX_SLURM_VISIBLE_DEVICES', '0')
    plan['allocation']['start_epoch'] = 1_800_000_000
    captured = {}
    monkeypatch.setattr(runner.paid.PaidSupervisor, 'phase', lambda self, name, seconds, **kwargs:
        captured.update(name=name, seconds=seconds, environment=kwargs['environment']) or ('passed', 0))
    supervisor = runner.CurrentSupervisor(plan, watchdog=False, clock=lambda: 0)
    environment = {'CUDA_VISIBLE_DEVICES': expected_cvd}
    assert supervisor.phase(phase_name, 20, environment=environment) == ('passed', 0)
    assert captured['environment'] == {
        'CUDA_VISIBLE_DEVICES': expected_cvd, 'ORCJAX_CURRENT_T7_SUPERVISED': '1',
        'CURRENT_T7_SHELL_STARTED_EPOCH': '1800000000'}


def test_phase_rejects_cvd_that_differs_from_admitted_phase_identity(monkeypatch, tmp_path):
    plan = _plan(tmp_path)
    plan['scope']['gpu_uuid'] = 'GPU-00000000-0000-0000-0000-000000000001'
    monkeypatch.setenv('ORCJAX_SLURM_VISIBLE_DEVICES', '0')
    supervisor = runner.CurrentSupervisor(plan, watchdog=False, clock=lambda: 0)
    with pytest.raises(ValueError, match='differs from its admitted phase identity'):
        supervisor.phase('preflight', 20, environment={'CUDA_VISIBLE_DEVICES': 'GPU-other'})


def test_supervisor_uses_one_probe_phase_and_exclusive_ledger(monkeypatch, tmp_path):
    plan = _plan(tmp_path)
    from scripts.hpc import shared_affine_cuda_identity
    monkeypatch.setattr(shared_affine_cuda_identity, 'binding_report_passes', lambda *args: True)
    supervisor = PassingSupervisor(plan, watchdog=False, clock=lambda: 0)
    assert supervisor.run() == 0
    assert [name for name, _ in supervisor.phase_names] == ['preflight', 'probe']
    assert supervisor.phase_names[-1][1] == runner.WORK_SECONDS - runner.CAPS['preflight']
    ledger = json.loads(Path(plan['ledger']).read_text())
    assert ledger['status'] == 'closed'
    assert ledger['retry_allowed'] is False
    retry = PassingSupervisor(plan, watchdog=False, clock=lambda: 0)
    with pytest.raises(FileExistsError):
        retry.run()


def test_unknown_release_keeps_shared_lock(monkeypatch, tmp_path):
    args = runner.probe.parser().parse_args(cli_args(output=tmp_path / 'out.json'))
    args.approval_report.write_text('{}')
    events = []
    monkeypatch.setattr(runner.probe, 'check_approval', lambda *args: {'approved': True})
    monkeypatch.setattr(runner, 'read_live_allocation', lambda: {'start_epoch': 1})

    class Lock:
        def __init__(self, *args):
            pass
        def acquire(self):
            return self
        def release(self, *, release_verified):
            events.append(release_verified)

    class Supervisor:
        def __init__(self, plan):
            self.record = {'release_verified': False}
        def run(self):
            return 125

    assert runner.run_live(args, cli_args(output=args.output), allocation_reader=lambda: {'start_epoch': 1}, lock_factory=Lock,
                           supervisor_factory=Supervisor) == 125
    assert events == [False]


def test_no_inherited_paid_run_is_used(monkeypatch, tmp_path):
    plan = _plan(tmp_path)
    monkeypatch.setattr(runner.free.Supervisor, 'run', lambda self: (_ for _ in ()).throw(AssertionError('inherited run called')))
    from scripts.hpc import shared_affine_cuda_identity
    monkeypatch.setattr(shared_affine_cuda_identity, 'binding_report_passes', lambda *args: True)
    assert PassingSupervisor(plan, watchdog=False, clock=lambda: 0).run() == 0
