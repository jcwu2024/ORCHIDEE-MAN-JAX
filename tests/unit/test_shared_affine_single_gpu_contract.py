"""Private CPU contract checks; all GPU/driver boundaries are mocked."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hpc import run_shared_latent_T7_batch_numerics_paid as paid
from scripts.hpc import probe_shared_latent_T7_batch_numerics as probe
from scripts.hpc import shared_affine_cuda_identity as cuda
from tests.unit.test_shared_latent_T7_batch_numerics_paid import FakeCudaDriver, scope_json
from tests.unit.test_shared_affine_gpu_discovery import FakeRuntime, FakePhysicalDriver
from tests.unit.test_shared_latent_T7_batch_numerics_launcher import minimal_plan, FakeMonitor, FakeGroups, gpu_row

UUID='GPU-01234567-89ab-cdef-0123-456789abcdef'
ENV=dict(SLURM_JOB_GPUS='3', ORCJAX_SLURM_VISIBLE_DEVICES=UUID)
ROW=dict(index=3, uuid=UUID, name='Tesla V100', minor_number=5,
    device_major=195, device_minor=5, control_devices=[dict(major=195,minor=255)],
    slurm_gpu_id='3', discovery_gpu_uuid=UUID,
    allocation_association='trusted_scheduler_original_CVD_same_runtime_namespace')


@pytest.fixture(autouse=True)
def mocked_cuda_runtime(monkeypatch):
    original = cuda.runtime_identity
    monkeypatch.setattr(cuda, 'runtime_identity', lambda **kwargs: original(FakeRuntime(), **kwargs))


@pytest.mark.parametrize('rules,broad', [
    ('c 195:5 rwm\nc *:255 rw\n',False),
    ('c 195:* rwm\n',True),
    ('a *:* rwm\n',True),
    ('c 195:5 rwm\nc 195:6 rw\n',True),
    ('c 195:* m\nc 195:5 rw\nc 195:255 rw\n',False)])
def test_new_contract_retains_rules_and_correct_driver_pjrt_without_claiming_os_denial(tmp_path,monkeypatch,rules,broad):
    observation={}
    mapping=paid.admit_gpu_mapping(ENV,[ROW],rules,contract=paid.GPU_EXECUTION_CONTRACT,observation=observation)
    assessment=observation['os_device_access_assessment']
    assert assessment['raw_devices_rules']==rules
    assert bool(assessment['broader_or_ambiguous_access_rules']) is broad
    assert assessment['os_other_gpu_open_denial']=='not_verified'
    assert assessment['selected_device_read_write_permitted']
    monkeypatch.setenv(cuda.CONTRACT_ENVIRONMENT_KEY,paid.GPU_EXECUTION_CONTRACT)
    monkeypatch.setenv(cuda.ENVIRONMENT_KEY,mapping['gpu_uuid'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',mapping['gpu_uuid'])
    monkeypatch.setattr(cuda.ctypes,'CDLL',lambda _name:FakePhysicalDriver())
    binding=probe.verify_probe_gpu_binding('gpu',[SimpleNamespace(platform='gpu',local_hardware_id=0)],tmp_path/'result.json')
    assert binding['passed'] and binding['driver_device_count']==8 and binding['runtime_device_count']==1 and binding['driver_gpu_uuid']==UUID
    evidence=json.loads((tmp_path/'result.startup-identity.json').read_text())
    assert evidence['status']=='passed' and evidence['affine_science_executed'] is False
    assert [row['api'] for row in evidence['driver']['calls']]==['cuInit','cuDeviceGetCount','cuDeviceGetByPCIBusId','cuDeviceGetPCIBusId','cuDeviceGetUuid_v2']


@pytest.mark.parametrize('rules',['c 195:* rw\n','a *:* rwm\n','c 195:5 rw\nc *:255 rw\n'])
def test_old_strict_default_still_rejects_wildcards(rules):
    with pytest.raises(ValueError,match='wildcard'):
        paid.admit_gpu_mapping(ENV,[ROW],rules)


def test_fixed_control_minor_wildcard_major_has_same_data_major_classification():
    controls = {(195, 255)}
    explicit = paid.assess_device_rules('c 195:7 rw\nc 195:255 rw\n', (195, 7), controls)
    wildcard = paid.assess_device_rules('c 195:7 rw\nc *:255 rw\n', (195, 7), controls)
    for field in ('selected_device_read_write_permitted', 'broader_or_ambiguous_access_rules', 'observed_data_major_access'):
        assert explicit[field] == wildcard[field]
    assert not explicit['cross_major_unknown_access_rules']
    assert wildcard['cross_major_unknown_access_rules'][0]['line_number'] == 2
    assert wildcard['os_other_gpu_open_denial'] == 'not_verified'


def test_unrecorded_fixed_minor_wildcard_major_remains_same_major_unknown():
    result = paid.assess_device_rules('c 195:7 rw\nc *:8 rw\n', (195, 7), {(195, 255)})
    assert result['broader_or_ambiguous_access_rules'] and result['cross_major_unknown_access_rules']


@pytest.mark.parametrize('environment,row,rules',[
    (dict(ENV,SLURM_JOB_GPUS='2'),ROW,'a *:* rwm\n'),
    (dict(ENV,ORCJAX_SLURM_VISIBLE_DEVICES='GPU-wrong'),ROW,'a *:* rwm\n'),
    (ENV,dict(ROW,device_minor=6),'a *:* rwm\n'),
    (ENV,dict(ROW,control_devices=[dict(major=195,minor=5)]),'a *:* rwm\n'),
    (ENV,ROW,'c *:255 rw\n')])
def test_new_contract_keeps_allocation_uuid_kernel_and_selected_access_gates(environment,row,rules):
    with pytest.raises(ValueError):
        paid.admit_gpu_mapping(environment,[row],rules,contract=paid.GPU_EXECUTION_CONTRACT)


@pytest.mark.parametrize('change',['missing','strict','require_os'])
def test_scope_never_silently_migrates_old_contract(tmp_path,monkeypatch,change):
    args,scope,path,_,_=scope_json(tmp_path,monkeypatch)
    if change=='missing':scope.pop('gpu_execution_contract')
    elif change=='strict':scope['gpu_execution_contract']=paid.STRICT_GPU_CONTRACT
    else:scope['os_device_open_isolation_required']=True
    path.write_text(json.dumps(scope))
    with pytest.raises(ValueError,match='contract differs'):paid.prepare(args)


@pytest.mark.parametrize('failure',['second_pjrt','second_runtime','wrong_uuid','wrong_cvd'])
def test_actual_affine_child_rejects_before_upload_or_science_and_saves_identity(tmp_path,monkeypatch,failure):
    monkeypatch.setenv(cuda.CONTRACT_ENVIRONMENT_KEY,paid.GPU_EXECUTION_CONTRACT)
    monkeypatch.setenv(cuda.ENVIRONMENT_KEY,UUID)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-wrong' if failure=='wrong_cvd' else UUID)
    driver=FakePhysicalDriver(
        uuid_bytes=bytes.fromhex('f'*32) if failure=='wrong_uuid' else bytes.fromhex('0123456789abcdef0123456789abcdef'))
    if failure=='second_runtime':
        monkeypatch.setattr(cuda, 'runtime_identity', lambda **_: dict(runtime_device_count=2, runtime_visible_ordinal=0, runtime_pci_bus_id='0000:65:00.0'))
    monkeypatch.setattr(cuda.ctypes,'CDLL',lambda _name:driver)
    devices=[SimpleNamespace(platform='gpu',local_hardware_id=0)]*(2 if failure=='second_pjrt' else 1)
    jax=SimpleNamespace(device_put=lambda *_:pytest.fail('identity rejection must precede upload/affine'))
    monkeypatch.setattr(probe,'initialize_probe_runtime',lambda _backend:(jax,devices))
    monkeypatch.setattr(probe,'load_operands',lambda _path:{})
    operands=tmp_path/'operands.npz';operands.touch()
    output=tmp_path/'result.json'
    with pytest.raises(ValueError):probe.run(SimpleNamespace(repeats=3,output=output,operands=operands,backend='gpu'))
    saved=json.loads(output.with_suffix('.startup-identity.json').read_text())
    assert saved['status']=='failed' and saved['affine_science_executed'] is False
    assert saved['expected_gpu_uuid']==UUID and saved['pjrt_device_count']==len(devices)
    assert not output.exists()


def test_missing_mandatory_uuid_rejects_before_runtime_initialization(tmp_path,monkeypatch):
    monkeypatch.setenv(cuda.CONTRACT_ENVIRONMENT_KEY,paid.GPU_EXECUTION_CONTRACT)
    monkeypatch.delenv(cuda.ENVIRONMENT_KEY,raising=False)
    monkeypatch.setattr(probe,'initialize_probe_runtime',lambda *_:pytest.fail('must reject before runtime init'))
    monkeypatch.setattr(probe,'load_operands',lambda _path:{})
    with pytest.raises(ValueError,match='requires expected'):
        probe.run(SimpleNamespace(repeats=3,output=tmp_path/'result.json',operands=tmp_path/'input.npz',backend='gpu'))
    saved=json.loads((tmp_path/'result.startup-identity.json').read_text())
    assert saved['stage']=='affine_child_contract_before_runtime' and saved['affine_science_executed'] is False


@pytest.mark.parametrize('actual_binding',[True,False])
def test_supervisor_requires_actual_affine_identity_not_only_separate_preflight(tmp_path,monkeypatch,actual_binding):
    launcher=paid.free
    plan=minimal_plan(tmp_path)
    plan['phase_environment']={'ORCJAX_AFFINE_EXPECTED_GPU_UUID':UUID}
    plan['paid_startup_observation']={'files':[]}
    supervisor=launcher.Supervisor(plan,monitor=FakeMonitor(gpu_row(),gpu_row()),groups=FakeGroups(),watchdog=False)
    binding=dict(passed=True,expected_gpu_uuid=UUID,driver_gpu_uuid=UUID,driver_device_count=1,pjrt_local_hardware_id=0)
    def phase(_self,name,seconds):
        if name=='preflight':
            Path(plan['preflight_output']).write_text(json.dumps(dict(passed=True,actual_backend='gpu',jax_version='0.4.38',
                devices=['GPU'],affine_gpu_identity=binding)))
        else:
            result=Path(plan['output']);container=result.with_suffix('.results.npz');container.write_bytes(b'private fixture')
            result.write_text(json.dumps(dict(status='passed',passed=True,backend='gpu',jax_version='0.4.38',
                operand_sha256=launcher.OPERAND_SHA,measurement_count=12,preserved_array_count=91,total_small_graph_compiles=4,
                tolerances=dict(rtol=1e-5,atol=1e-6),results_container=str(container),
                affine_gpu_identity=binding if actual_binding else {})))
        return 'passed',0
    monkeypatch.setattr(launcher.Supervisor,'phase',phase)
    assert supervisor.run()==(0 if actual_binding else 1)
    record=json.loads(Path(plan['output']).with_suffix('.supervisor.json').read_text())
    assert record['release_verified']
    if not actual_binding:assert record['error']=='actual affine child CUDA-driver/PJRT UUID binding gate failed'
    assert len(plan['paid_startup_observation']['files'])==2
