"""Independent CPU-only API, namespace, association and lifecycle fixtures."""
import ctypes
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from scripts.hpc import shared_affine_cuda_identity as cuda
from scripts.hpc import run_shared_latent_T7_batch_numerics_paid as paid
from scripts.hpc import run_shared_latent_T7_batch_numerics as free

UUID = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
PCI = '0000:65:00.0'


class Call:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *args):
        return self.callback(*args)


class FakeRuntime:
    def __init__(self, count=1, pci=PCI):
        self.ordinals = []
        def write_count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = count
            return 0
        def write_bus(buffer, length, ordinal):
            self.ordinals.append(ordinal)
            ctypes.memmove(buffer, pci.encode()+b'\0', len(pci)+1)
            return 0
        self.cudaGetDeviceCount = Call(write_count)
        self.cudaDeviceGetPCIBusId = Call(write_bus)


class FakePhysicalDriver:
    def __init__(self, count=8, pci=PCI, uuid_bytes=bytes.fromhex('0123456789abcdef0123456789abcdef')):
        self.by_pci = []
        self.uuid_handles = []
        self.cuInit = Call(lambda *_: 0)
        def write_count(pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = count
            return 0
        def write_handle(pointer, address):
            self.by_pci.append(address)
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 6
            return 0
        def write_bus(buffer, length, handle):
            assert handle.value == 6
            ctypes.memmove(buffer, pci.encode()+b'\0', len(pci)+1)
            return 0
        def write_uuid(pointer, handle):
            self.uuid_handles.append(handle.value)
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte*16)).contents[:] = uuid_bytes
            return 0
        self.cuDeviceGetCount = Call(write_count)
        self.cuDeviceGetByPCIBusId = Call(write_handle)
        self.cuDeviceGetPCIBusId = Call(write_bus)
        self.cuDeviceGetUuid_v2 = Call(write_uuid)
        self.cuDeviceGet = Call(lambda *_: pytest.fail('do not select driver ordinal zero'))


def discovery_environment(monkeypatch, visible='0'):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setenv('ORCJAX_SLURM_VISIBLE_DEVICES', visible)
    monkeypatch.setenv('SLURM_JOB_GPUS', '5')
    monkeypatch.setenv(cuda.CONTRACT_ENVIRONMENT_KEY, cuda.DISCOVERY_GPU_CONTRACT)
    monkeypatch.delenv(cuda.ENVIRONMENT_KEY, raising=False)
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'FASTEST_FIRST')


def test_runtime_local_zero_binds_nonzero_physical_driver_handle(monkeypatch):
    discovery_environment(monkeypatch)
    runtime, driver = FakeRuntime(), FakePhysicalDriver()
    result = cuda.discover_visible_gpu(observation={},
        runtime_provider=lambda: cuda.runtime_identity(runtime),
        driver_provider=lambda **kwargs: cuda.driver_identity(driver, **kwargs))
    assert runtime.ordinals == [0] and driver.by_pci == [PCI.encode()] and driver.uuid_handles == [6]
    assert result['gpu_uuid'] == UUID and result['slurm_gpu_id'] == '5'
    assert result['driver_device_count'] == 8 and result['driver_visible_ordinal'] is None
    assert result['runtime_device_count'] == 1 and not result['independent_GRES_index_to_UUID_verified']


@pytest.mark.parametrize('count', [0, 2, 8])
def test_multivisible_never_selects_first_runtime_card(count):
    runtime = FakeRuntime(count)
    with pytest.raises(ValueError, match='exactly one'):
        cuda.runtime_identity(runtime)
    assert runtime.ordinals == []


@pytest.mark.parametrize('visible', ['', '0,1', '-1', 'GPU-partial'])
def test_unknown_original_visibility_rejects_before_loading_cuda(monkeypatch, visible):
    discovery_environment(monkeypatch, visible)
    with pytest.raises(ValueError, match='unknown'):
        cuda.discover_visible_gpu(observation={}, runtime_provider=lambda: pytest.fail('must not initialize CUDA'))


def test_full_scheduler_uuid_conflict_rejects(monkeypatch):
    discovery_environment(monkeypatch, 'GPU-ffffffff-ffff-ffff-ffff-ffffffffffff')
    with pytest.raises(ValueError, match='scheduler original UUID'):
        cuda.discover_visible_gpu(observation={}, runtime_provider=lambda: cuda.runtime_identity(FakeRuntime()),
            driver_provider=lambda **kwargs: cuda.driver_identity(FakePhysicalDriver(), **kwargs))


def test_driver_pci_roundtrip_conflict_rejects():
    with pytest.raises(ValueError, match='PCI handle differs'):
        cuda.driver_identity(FakePhysicalDriver(pci='0000:66:00.0'), pci_bus_id=PCI)


def mapping_row():
    return dict(index=0, uuid=UUID, name='Tesla V100', minor_number=7, device_major=195, device_minor=7,
        control_devices=[dict(major=195, minor=255)], slurm_gpu_id='5', discovery_gpu_uuid=UUID,
        allocation_association='trusted_scheduler_original_CVD_same_runtime_namespace')


@pytest.mark.parametrize('visible,rules,basis', [
    ('0', 'c 195:7 rw\nc 195:255 rw\nc 195:* m\n', 'numeric_scheduler_CVD_and_unique_actual_job_data_device_pair'),
    (UUID, 'a *:* rwm\n', 'scheduler_original_UUID_matches_runtime_driver_NVML_kernel')])
def test_association_distinguishes_all_four_identity_spaces(visible, rules, basis):
    mapping = paid.admit_gpu_mapping(dict(SLURM_JOB_GPUS='5', ORCJAX_SLURM_VISIBLE_DEVICES=visible),
        [mapping_row()], rules, contract=paid.GPU_EXECUTION_CONTRACT)
    assert mapping['slurm_gpu_id'] == '5' and mapping['gpu_host_index'] == 0 and mapping['actual_GPU_minor'] == 7
    assert mapping['child_visible_ordinal'] == 0 and mapping['allocation_association_basis'] == basis
    assert mapping['os_device_access_assessment']['os_other_gpu_open_denial'] == 'not_verified'


@pytest.mark.parametrize('rules', ['a *:* rw\n', 'c 195:* rw\n', 'c 195:7 rw\nc *:255 rw\n', 'c 195:7 rw\nc 195:8 r\n'])
def test_numeric_mask_broad_or_second_data_permissions_are_unknown(rules):
    with pytest.raises(ValueError, match='physical_allocation_mapping_unknown'):
        paid.admit_gpu_mapping(dict(SLURM_JOB_GPUS='5', ORCJAX_SLURM_VISIBLE_DEVICES='0'),
            [mapping_row()], rules, contract=paid.GPU_EXECUTION_CONTRACT)


@pytest.mark.parametrize('change', ['slurm_gpu_id', 'discovery_gpu_uuid', 'device_minor'])
def test_conflicting_association_rejects(change):
    row = mapping_row()
    row[change] = 8 if change == 'device_minor' else 'conflict'
    with pytest.raises(ValueError):
        paid.admit_gpu_mapping(dict(SLURM_JOB_GPUS='5', ORCJAX_SLURM_VISIBLE_DEVICES='0'),
            [row], 'c 195:7 rw\n', contract=paid.GPU_EXECUTION_CONTRACT)


def test_uuid_only_nvml_selector_and_independent_xml_pci_minor(monkeypatch):
    calls = []
    xml = f'<nvidia_smi_log><gpu><uuid>{UUID}</uuid><product_name>Tesla V100</product_name><minor_number>7</minor_number><pci><pci_bus_id>00000000:65:00.0</pci_bus_id></pci></gpu></nvidia_smi_log>'
    def query(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, f'0, {UUID}, Tesla V100\n' if len(calls) == 1 else xml, '')
    monkeypatch.setattr(paid.subprocess, 'run', query)
    row, _ = paid.observe_gpu_mapping(dict(SLURM_JOB_GPUS='5'), UUID, pci_bus_id=PCI)
    assert row['index'] == '0' and row['minor_number'] == 7
    assert all(argv[2] == UUID for argv in calls)


def test_v2_actual_binding_count_visibility_not_driver_inventory(monkeypatch):
    discovery_environment(monkeypatch, UUID)
    monkeypatch.setenv(cuda.ENVIRONMENT_KEY, UUID)
    runtime_fn = cuda.runtime_identity
    monkeypatch.setattr(cuda, 'runtime_identity', lambda **_: runtime_fn(FakeRuntime()))
    driver = FakePhysicalDriver()
    binding = cuda.verify_expected_gpu([SimpleNamespace(platform='gpu', local_hardware_id=0)], UUID,
        identity_provider=lambda **kwargs: cuda.driver_identity(driver, **kwargs))
    assert cuda.binding_report_passes(binding, UUID, cuda.DISCOVERY_GPU_CONTRACT)
    assert binding['driver_device_count'] == 8
    assert not cuda.binding_report_passes(dict(binding, runtime_device_count=2), UUID, cuda.DISCOVERY_GPU_CONTRACT)
    assert not cuda.binding_report_passes(dict(binding, driver_gpu_uuid='GPU-wrong'), UUID, cuda.DISCOVERY_GPU_CONTRACT)


def cuda_runtime_fixture():
    return cuda.runtime_identity(FakeRuntime())


def test_unknown_release_monitor_cannot_certify_cleanup():
    with pytest.raises(ValueError, match='release cannot be verified'):
        paid.IdentityPendingMonitor().sample()


def test_discovery_shares_preflight_budget_and_phase_ownership(tmp_path, monkeypatch):
    from tests.unit.test_shared_latent_T7_batch_numerics_launcher import minimal_plan, FakeGroups
    plan = minimal_plan(tmp_path)
    plan['paid_startup_observation'] = paid.startup_observation({})
    plan['shell_started_epoch'] = 1000
    monkeypatch.setattr(free.time, 'time', lambda: 1100)
    now = [10.]
    supervisor = paid.PaidSupervisor(plan, clock=lambda: now[0], groups=FakeGroups(), watchdog=False)
    assert supervisor.startup_preflight_deadline == 150.  # 100s used of startup240
    calls = []
    monkeypatch.setattr(free.Supervisor, 'phase', lambda self, name, cap, **kwargs: calls.append((name, cap, kwargs)) or ('passed', 0))
    assert supervisor.phase('discovery', 240, monitor=False) == ('passed', 0)
    now[0] = 40
    assert supervisor.phase('preflight', 240) == ('passed', 0)
    assert calls[0][1] == 140 and calls[1][1] == 110
    now[0] = 150
    assert supervisor.phase('preflight', 240) == ('deadline_exceeded', 124)
    assert supervisor.phase('probe', 780) == ('passed', 0)
    assert calls[-1][1] == 780


@pytest.mark.parametrize('outcome', ['passed', 'unknown', 'broad_numeric', 'query_failed'])
def test_full_paid_lifecycle_owns_discovery_and_releases_or_retains_mutex(tmp_path, monkeypatch, outcome):
    from pathlib import Path
    from tests.unit.test_shared_latent_T7_batch_numerics_launcher import minimal_plan, FakeGroups
    plan = minimal_plan(tmp_path)
    plan['scope']['gpu_execution_contract'] = paid.GPU_EXECUTION_CONTRACT
    plan['slurm_allocation'] = dict(job=dict(JobId='1234'))
    plan['commands']['discovery'] = ['fixture-discovery']
    plan['discovery_output'] = str(tmp_path/'discovery.json')
    discovery_environment(monkeypatch)
    groups, launched = FakeGroups(live=[777]), []
    identity = dict(passed=True, gpu_uuid=UUID, original_CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='FASTEST_FIRST',
        slurm_gpu_id='5', allocation_association='trusted_scheduler_original_CVD_same_runtime_namespace')
    runtime = dict(runtime_device_count=1, runtime_visible_ordinal=0, runtime_pci_bus_id=PCI)
    driver = dict(driver_device_count=8, driver_gpu_uuid=UUID, driver_pci_bus_id=PCI)
    binding = dict(passed=True, expected_gpu_uuid=UUID, pjrt_local_hardware_id=0,
        gpu_execution_contract=paid.GPU_EXECUTION_CONTRACT, **runtime, **driver)
    def popen(argv, **kwargs):
        name = 'discovery' if argv == ['fixture-discovery'] else 'preflight' if argv == ['fake-preflight'] else 'probe'
        launched.append((name, kwargs['env']))
        if name == 'discovery':
            report = dict(status='failed') if outcome == 'unknown' else dict(status='passed', identity=identity, runtime_identity=runtime, driver_identity=driver)
            Path(plan['discovery_output']).write_text(json.dumps(report))
        elif name == 'preflight':
            Path(plan['preflight_output']).write_text(json.dumps(dict(passed=True, actual_backend='gpu', jax_version='0.4.38', devices=['GPU'], affine_gpu_identity=binding)))
        else:
            output = Path(plan['output'])
            arrays = output.with_suffix('.results.npz')
            arrays.write_bytes(b'private fixture, no numerical computation')
            output.write_text(json.dumps(dict(status='passed', passed=True, backend='gpu', jax_version='0.4.38',
                operand_sha256=free.OPERAND_SHA, measurement_count=12, preserved_array_count=91, total_small_graph_compiles=4,
                tolerances=dict(rtol=1e-5, atol=1e-6), results_container=str(arrays), affine_gpu_identity=binding)))
        return SimpleNamespace(pid=101+len(launched), poll=lambda: 2 if outcome == 'unknown' else 0, wait=lambda **_: 0)
    queries = []
    def query(argv, **kwargs):
        queries.append(argv)
        assert argv[2] == UUID, 'every host selector must be UUID'
        if '--query-gpu=index,uuid,name' in argv:
            if outcome == 'query_failed':
                raise subprocess.CalledProcessError(6, argv, output='No devices were found\n', stderr='fixture denied')
            stdout = f'0, {UUID}, Tesla V100\n'
        elif '-x' in argv:
            stdout = f'<nvidia_smi_log><gpu><uuid>{UUID}</uuid><product_name>Tesla V100</product_name><minor_number>7</minor_number><pci><pci_bus_id>{PCI}</pci_bus_id></pci></gpu></nvidia_smi_log>'
        elif any(arg.startswith('--query-compute-apps=') for arg in argv):
            stdout = ''
        else:
            stdout = f'0, {UUID}, Tesla V100, 0, 0\n'
        return subprocess.CompletedProcess(argv, 0, stdout, '')
    constructor = paid.PaidSupervisor
    monkeypatch.setattr(paid, 'PaidSupervisor', lambda plan: constructor(plan, popen=popen, groups=groups, watchdog=False))
    monkeypatch.setattr(paid, 'SCOPE_LOCK', tmp_path/'paid.lock')
    monkeypatch.setattr(paid, 'read_free_admission', lambda **_: dict(status='fixture_unchanged'))
    monkeypatch.setattr(paid.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(paid.subprocess, 'run', query)
    monkeypatch.setattr(paid, 'os', SimpleNamespace(environ=os.environ, major=lambda _: 195, minor=lambda _: 7))
    monkeypatch.setattr(paid, '_kernel_stat', lambda *_: SimpleNamespace(st_mode=paid.stat.S_IFCHR, st_rdev=7))
    monkeypatch.setattr(paid, 'read_cgroup_rules', lambda *_, **__: ('/fixture/job_1234/devices.list', 'a *:* rwm\n' if outcome == 'broad_numeric' else 'c 195:7 rw\n'))
    assert paid.run_live(plan) == (0 if outcome == 'passed' else 2 if outcome == 'unknown' else 1)
    ledger = json.loads(Path(plan['ledger']).read_text())
    record = json.loads(Path(plan['output']).with_suffix('.supervisor.json').read_text())
    assert not any(group == 777 for group, _ in groups.killed), 'never kill unowned groups'
    assert not record['active_owned_groups']
    assert launched[0][1]['CUDA_VISIBLE_DEVICES'] == '0'
    assert cuda.ENVIRONMENT_KEY not in launched[0][1]
    if outcome == 'passed':
        assert [name for name, _ in launched] == ['discovery', 'preflight', 'probe']
        assert all(env['CUDA_VISIBLE_DEVICES'] == UUID for _, env in launched[1:])
        assert record['release_verified'] and not paid.SCOPE_LOCK.exists()
        assert ledger['charged_gpu_seconds'] < 1200
    else:
        assert [name for name, _ in launched] == ['discovery']
        assert record['numerical_status'] == 'not_executed'
        if outcome == 'unknown':
            assert not queries and not record['release_verified']
            assert ledger['charged_gpu_seconds'] == 1200 and paid.SCOPE_LOCK.exists()
        elif outcome == 'query_failed':
            evidence = record['plan']['paid_startup_observation']['gpu_observation']['commands'][0]
            assert evidence['stdout'] == 'No devices were found\n' and evidence['stderr'] == 'fixture denied'
