"""Private CPU fixtures for paid evidence; no scheduler, driver or GPU runtime."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.hpc import run_shared_latent_T7_batch_numerics_paid as paid
from scripts.hpc import shared_affine_cuda_identity as cuda
from scripts.hpc import check_daily_training_runtime as runtime

UUID = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
XML = f'<nvidia_smi_log><gpu><uuid>{UUID}</uuid><product_name>Tesla V100</product_name><minor_number>5</minor_number></gpu></nvidia_smi_log>'


def main_fixture(tmp_path, monkeypatch, *, rules='c 195:5 rwm\n'):
    diag = tmp_path / 'diagnostics'
    diag.mkdir()
    output, ledger, scope = diag / 'result.v1.json', diag / 'ledger.json', diag / 'scope.v1.json'
    scope.write_text(json.dumps(dict(schema_version=paid.SCHEMA, scope_id=paid.SCOPE_ID,
        status='open', execution_authorized=True, approval_note='private CPU fixture only')))
    args = SimpleNamespace(dry_run=False, scope_json=scope, output=output, ledger=ledger)
    env = dict(SLURM_JOB_ID='123456', SLURM_JOB_GPUS='3', ORCJAX_SLURM_VISIBLE_DEVICES='0',
               CUDA_VISIBLE_DEVICES='', SECRET_TOKEN='must-not-be-saved')
    monkeypatch.setattr(paid, 'os', SimpleNamespace(name='posix', environ=env,
        major=lambda _device: 195, minor=lambda device: device))
    for key, value in [('OUTPUT_DIR', diag), ('OUTPUT', output), ('LEDGER', ledger),
                       ('SCOPE_LOCK', tmp_path / 'locks' / 'scope.lock')]:
        monkeypatch.setattr(paid, key, value)
    monkeypatch.setattr(paid.free, 'parser', lambda: SimpleNamespace(parse_args=lambda _argv: args))
    def prepare(_args):
        _args._paid_startup_observation['slurm_job'] = dict(JobId='123456', NumCPUs='4', GRES='gpu:v100:1(IDX:3)')
        return dict(slurm_allocation=dict(job=dict(JobId='123456')), scope=dict(paid.RESOURCES))
    monkeypatch.setattr(paid, 'prepare', prepare)
    monkeypatch.setattr(paid, 'read_free_admission', lambda **_kwargs: dict(status='known_free_unopened_unreserved'))
    monkeypatch.setattr(paid.free, 'Supervisor', lambda *_args: pytest.fail('Supervisor must not start before admission'))
    original_stat = Path.stat
    def stat(path, *args, **kwargs):
        if path.as_posix() in ('/dev/nvidia5', '/dev/nvidiactl'):
            return SimpleNamespace(st_mode=paid.stat.S_IFCHR, st_rdev=5 if path.name == 'nvidia5' else 255)
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', stat)
    proc = tmp_path / 'proc-self-cgroup'
    proc.write_text('7:devices:/slurm/uid_1/job_123456/step_batch\n2:cpu,cpuacct:/slurm/job_123456\n')
    base = tmp_path / 'devices'
    devices = base / 'slurm/uid_1/job_123456/step_batch/devices.list'
    devices.parent.mkdir(parents=True)
    devices.write_text(rules)
    reader = paid.read_cgroup_rules
    monkeypatch.setattr(paid, 'read_cgroup_rules', lambda job, **kwargs:
        reader(job, proc_path=proc, cgroup_base=base, **kwargs))
    queries = []
    def query(command, **kwargs):
        queries.append(command)
        assert command[0] == 'nvidia-smi', 'no scheduler/retry/compute subprocess allowed'
        return subprocess.CompletedProcess(command, 0,
            f'3, {UUID}, Tesla V100\n' if len(queries) == 1 else XML,
            'fixture CSV stderr' if len(queries) == 1 else 'fixture XML stderr')
    monkeypatch.setattr(paid.subprocess, 'run', query)
    return SimpleNamespace(args=args, diag=diag, output=output, ledger=ledger, proc=proc,
                           devices=devices, queries=queries, env=env)


@pytest.mark.parametrize('rules', ['a *:* rwm\n', 'c *:255 rw\n'])
def test_startup_wildcard_rejection_retains_rule_kernel_cgroup_and_prior_stages(tmp_path, monkeypatch, capsys, rules):
    fixture = main_fixture(tmp_path, monkeypatch, rules=rules)
    assert paid.main([]) == 2
    result = json.loads(capsys.readouterr().out)
    evidence = result['paid_startup_observation']
    assert result['error'] == 'wildcard GPU access does not prove Slurm device isolation'
    assert result['GPU_executed'] is False
    assert evidence['slurm_job']['NumCPUs'] == '4'
    assert evidence['environment']['SLURM_JOB_GPUS'] == '3'
    assert 'SECRET_TOKEN' not in evidence['environment']
    assert 'must-not-be-saved' not in json.dumps(result)
    reads = {row['label']: row for row in evidence['files'] if 'label' in row}
    assert reads['authorization_scope']['text'] == fixture.args.scope_json.read_text()
    assert reads['proc_self_cgroup']['text'] == fixture.proc.read_text()
    assert reads['job_devices_list']['text'] == rules
    assert evidence['cgroup']['devices_list_path'] == str(fixture.devices.resolve())
    assert evidence['cgroup']['controllers'][0]['membership'].endswith('job_123456/step_batch')
    kernel = next(row for row in evidence['files'] if row['path'] == str(Path('/dev/nvidia5')))
    assert kernel['is_character_device'] and (kernel['major'], kernel['minor']) == (195, 5)
    rule = evidence['current_device_rule']
    assert rule['raw'] == rules.rstrip('\n') and rule['line_number'] == 1
    assert rule['selected_kernel_pair'] == [195, 5]
    assert [195, 255] in rule['control_pairs']
    assert [command['stdout'] for command in result['gpu_observation']['commands']] == [f'3, {UUID}, Tesla V100\n', XML]
    assert not fixture.ledger.exists() and not paid.SCOPE_LOCK.exists()
    attempt = fixture.output.with_suffix('.job-attempt.json').read_bytes()
    assert json.loads(attempt)['retry_allowed'] is False
    assert paid.main([]) == 2
    second = json.loads(capsys.readouterr().out)
    assert second['error_type'] == 'FileExistsError'
    assert len(fixture.queries) == 2 and fixture.output.with_suffix('.job-attempt.json').read_bytes() == attempt


@pytest.mark.parametrize('failure', ['devices_read', 'proc_read', 'kernel_stat'])
def test_startup_file_failure_preserves_path_errno_and_completed_queries(tmp_path, monkeypatch, capsys, failure):
    fixture = main_fixture(tmp_path, monkeypatch)
    original_read, original_stat = Path.read_text, Path.stat
    denied = fixture.devices if failure == 'devices_read' else fixture.proc
    def read(path, *args, **kwargs):
        if failure != 'kernel_stat' and path == denied:
            raise PermissionError(13, 'fixture permission denied', str(path))
        return original_read(path, *args, **kwargs)
    def stat(path, *args, **kwargs):
        if failure == 'kernel_stat' and path.as_posix() == '/dev/nvidia5':
            raise PermissionError(13, 'fixture stat denied', str(path))
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(Path, 'stat', stat)
    assert paid.main([]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['error_type'] == 'PermissionError'
    failed = [row for row in result['paid_startup_observation']['files'] if row.get('status') == 'failed']
    assert len(failed) == 1 and failed[0]['errno'] == 13
    assert failed[0]['path'] == (str(Path('/dev/nvidia5')) if failure == 'kernel_stat' else str(denied.resolve()))
    assert len(result['gpu_observation']['commands']) == 2
    assert not fixture.ledger.exists() and not paid.SCOPE_LOCK.exists()


def test_scheduler_clock_failure_keeps_raw_command_environment_and_count(monkeypatch, capsys):
    monkeypatch.setenv('SLURM_JOB_ID', '123456')
    command = [paid.SLURM, 'show', 'job', '--details', '-o', '123456']
    def run(argv, **_kwargs):
        assert argv == command
        return subprocess.CompletedProcess(argv, 0, 'JobId=123456 JobState=PENDING TimeLimit=00:20:00 NumCPUs=8', 'fixture Slurm stderr')
    monkeypatch.setattr(paid.subprocess, 'run', run)
    with pytest.raises(paid.PaidStartupError):
        paid.scheduler_allocation_epochs()
    result = json.loads(capsys.readouterr().err)['paid_startup_observation']
    assert result['slurm_job']['NumCPUs'] == '8'
    assert result['commands'][0]['stdout'].endswith('NumCPUs=8')
    assert result['commands'][0]['stderr'] == 'fixture Slurm stderr'
    assert result['environment']['SLURM_JOB_ID'] == '123456'


def test_main_allocation_rejection_keeps_actual_request_before_gpu_or_supervisor(tmp_path, monkeypatch, capsys):
    fixture = main_fixture(tmp_path, monkeypatch)
    calls = []
    raw = 'JobId=123456 JobState=RUNNING Partition=gnall NodeList=ibc13b03n01 NumNodes=1 NumTasks=1 NumCPUs=8 Requeue=0 TimeLimit=00:20:00'
    def scheduler(command, **_kwargs):
        calls.append(command)
        assert command[0] == paid.SLURM
        return subprocess.CompletedProcess(command, 0, raw, 'fixture scheduler stderr')
    monkeypatch.setattr(paid.subprocess, 'run', scheduler)
    def prepare(args):
        observation = args._paid_startup_observation
        result = paid._record_command([paid.SLURM, 'show', 'job', '--details', '-o', '123456'], observation)
        return paid.validate_allocation(dict(resources=paid.RESOURCES, rates=paid.RATES), fixture.env,
            paid.parse_job(result.stdout), 'ibc13b03n01', observation=observation)
    monkeypatch.setattr(paid, 'prepare', prepare)
    assert paid.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    observation = report['paid_startup_observation']
    assert observation['slurm_job']['NumCPUs'] == '8'
    assert observation['commands'][0]['stdout'] == raw
    assert observation['expected_resources']['allocated_slurm_cpus'] == 4
    assert len(calls) == 1 and not fixture.queries and not fixture.ledger.exists()
    assert fixture.output.with_suffix('.job-attempt.json').exists() and not paid.SCOPE_LOCK.exists()


def test_preflight_driver_count_failure_preserves_pjrt_and_driver_without_science(tmp_path, monkeypatch, capsys):
    expected = UUID
    monkeypatch.setenv('ORCJAX_AFFINE_EXPECTED_GPU_UUID', expected)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', expected)
    device = SimpleNamespace(platform='gpu', local_hardware_id=0)
    jax = SimpleNamespace(__version__='0.4.38', default_backend=lambda: 'gpu')
    monkeypatch.setattr(runtime, '_initialize_runtime', lambda _backend: (jax, object(), [device]))
    monkeypatch.setattr(runtime, 'check_runtime', lambda *_: pytest.fail('synthetic AD and affine science must not run'))
    import ctypes
    class Function:
        def __init__(self, callback): self.callback = callback
        def __call__(self, *args): return self.callback(*args)
    def count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 2
        return 0
    driver = SimpleNamespace(cuInit=Function(lambda *_: 0), cuDeviceGetCount=Function(count))
    monkeypatch.setattr(cuda.ctypes, 'CDLL', lambda name: driver)
    output = tmp_path / 'preflight.json'
    assert runtime.main(['--backend', 'gpu', '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    binding = report['paid_startup_observation']['affine_gpu_identity']
    assert report['error'] == 'CUDA driver must independently expose exactly one device'
    assert binding['expected_gpu_uuid'] == expected and binding['pjrt_device_count'] == 1
    assert binding['driver']['driver_device_count'] == 2
    assert binding['driver']['calls'] == [{'api': 'cuInit', 'returncode': 0}, {'api': 'cuDeviceGetCount', 'returncode': 0}]
    assert report['real_training_updates'] == 0 and report['passed'] is False
    capsys.readouterr()


@pytest.mark.parametrize('release_ok', [True, False])
def test_supervisor_later_preflight_rejection_preserves_raw_report_and_release_semantics(tmp_path, monkeypatch, capsys, release_ok):
    real_supervisor = paid.free.Supervisor
    fixture = main_fixture(tmp_path, monkeypatch, rules='c 195:* m\nc 195:5 rw\nc 195:255 rw\n')
    raw = '{"passed":true,"actual_backend":"gpu","jax_version":"wrong-version","devices":["fixture-GPU"]}'
    preflight = fixture.output.with_suffix('.runtime-preflight.json')
    def prepare(args):
        return dict(slurm_allocation=dict(job=dict(JobId='123456')), scope=dict(caps=dict(paid.free.CAPS)),
            output=str(fixture.output), ledger=str(fixture.ledger), checkout=str(tmp_path),
            shell_started_epoch=paid.time.time(), preflight_output=str(preflight), commands={})
    monkeypatch.setattr(paid, 'prepare', prepare)
    monkeypatch.setattr(paid.free, 'Supervisor', lambda plan: real_supervisor(plan, watchdog=False))
    samples = []
    def sample(_self):
        samples.append(1)
        return dict(processes=[], memory_mib=0 if release_ok or len(samples) == 1 else 1, utilization_percent=0)
    monkeypatch.setattr(paid.PaidNvidiaMonitor, 'sample', sample)
    phases = []
    def phase(_self, name, seconds):
        phases.append((name, seconds))
        preflight.write_text(raw)
        return 'passed', 0
    monkeypatch.setattr(real_supervisor, 'phase', phase)
    monkeypatch.setattr(paid.signal, 'signal', lambda *_: None)
    assert paid.main([]) == 1
    report = json.loads(fixture.output.with_suffix('.supervisor.json').read_text())
    assert report['error'] == 'actual target JAX0.4.38/single-GPU preflight gate failed'
    read = next(row for row in report['plan']['paid_startup_observation']['files'] if row.get('label') == 'preflight_result_admission')
    assert read['text'] == raw
    assert report['release_verified'] is release_ok and paid.SCOPE_LOCK.exists() is not release_ok
    assert phases == [('preflight', 240)]
    assert json.loads(fixture.ledger.read_text())['status'] == ('closed' if release_ok else 'closed_release_unverified')
    assert fixture.output.with_suffix('.job-attempt.json').exists()
    assert not fixture.output.exists() and len(fixture.queries) == 2
    capsys.readouterr()


def test_existing_monitor_response_reaches_gate_but_new_evidence_redacts_process_identifiers(monkeypatch):
    observation = {}
    monitor = paid.PaidNvidiaMonitor(3, UUID, 100, lambda: 0, observation=observation)
    raw = f'{UUID}, 98765, 4096, other-user-private-process-name\n'
    command = ['nvidia-smi', '-i', UUID, '--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name', '--format=csv,noheader,nounits']
    calls = []
    def run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, raw, '')
    monkeypatch.setattr(paid.subprocess, 'run', run)
    assert monitor.command(command) == raw
    assert calls == [command]
    saved = observation['monitor_commands'][0]
    assert saved['stdout'] is None and saved['stdout_retention'] == 'process identifiers redacted'
    assert saved['process_rows'] == [dict(field_count=4, gpu_uuid_matches_expected=True)]
    assert '98765' not in json.dumps(observation) and 'other-user-private-process-name' not in json.dumps(observation)


def test_monitor_timeout_retains_status_without_partial_process_identifiers(monkeypatch):
    observation = {}
    monitor = paid.PaidNvidiaMonitor(3, UUID, 100, lambda: 0, observation=observation)
    command = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name']
    raw = f'{UUID}, 98765, 4096, other-user-private-process-name\n'.encode()
    def run(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 4, output=raw, stderr=b'fixture timeout')
    monkeypatch.setattr(paid.subprocess, 'run', run)
    with pytest.raises(subprocess.TimeoutExpired):
        monitor.command(command)
    saved = observation['monitor_commands'][0]
    assert saved['status'] == 'failed' and saved['error_type'] == 'TimeoutExpired'
    assert saved['stdout'] is None and saved['stderr'] == 'fixture timeout'
    assert saved['process_rows'] == [dict(field_count=4, gpu_uuid_matches_expected=True)]
    assert '98765' not in json.dumps(observation) and 'other-user-private-process-name' not in json.dumps(observation)
