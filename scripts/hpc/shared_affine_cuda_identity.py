"""Paid visibility/physical identity gates; CUDA is loaded only in owned children."""
from __future__ import annotations
import ctypes
import os
import re

DISCOVERY_GPU_CONTRACT = 'T7_affine_allocated_single_gpu_v2'

ENVIRONMENT_KEY = 'ORCJAX_AFFINE_EXPECTED_GPU_UUID'
CONTRACT_ENVIRONMENT_KEY = 'ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT'
GPU_EXECUTION_CONTRACT = 'T7_affine_allocated_single_gpu_v1'


def validate_contract_environment():
    contract = os.environ.get(CONTRACT_ENVIRONMENT_KEY)
    expected = os.environ.get(ENVIRONMENT_KEY)
    if contract is not None:
        if contract not in (GPU_EXECUTION_CONTRACT, DISCOVERY_GPU_CONTRACT):
            raise ValueError('unknown child GPU execution contract')
        if not expected:
            raise ValueError('single-GPU child contract requires expected physical UUID')
    return expected

class CudaIdentityError(ValueError):
    def __init__(self, cause, observation):
        super().__init__(str(cause))
        self.cause, self.observation = cause, observation


def valid_uuid(value):
    return isinstance(value, str) and bool(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value))


def pci_address(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', value):
        raise ValueError('physical PCI bus address is invalid or unknown')
    domain, bus, device = value.lower().split(':')
    slot, function = device.split('.')
    return tuple(int(field, 16) for field in (domain, bus, slot, function))


def runtime_identity(runtime=None, *, observation=None):
    """Read documented CUDA-runtime visibility; never treat a Slurm ID as ordinal.

    cudaGetDeviceCount honors CUDA_VISIBLE_DEVICES. Driver enumeration is a
    separate mandatory check below; this function does not make claims about
    cuDeviceGetCount's visibility semantics or allocate arrays/compile JAX.
    """
    import importlib.util
    from pathlib import Path
    if observation is not None:
        observation['calls'] = []
    if runtime is None:
        library = 'libcudart.so.12'
        try:
            spec = importlib.util.find_spec('nvidia.cuda_runtime')
        except ModuleNotFoundError:
            spec = None
        if spec is not None and spec.submodule_search_locations:
            candidates = [Path(root)/'lib/libcudart.so.12' for root in spec.submodule_search_locations]
            candidates = [path for path in candidates if path.is_file()]
            if len(candidates) > 1:
                raise ValueError('ambiguous CUDA runtime library candidates')
            if candidates:
                library = str(candidates[0])
        if observation is not None:
            observation['library'] = library
        runtime = ctypes.CDLL(library)
    def call(name, argtypes, *arguments):
        row = dict(api=name)
        if observation is not None:
            observation['calls'].append(row)
        function = getattr(runtime, name)
        function.argtypes, function.restype = argtypes, ctypes.c_int
        code = function(*arguments)
        row['returncode'] = code
        if code != 0:
            raise RuntimeError(f'{name} failed with CUDA runtime result {code}')
    count = ctypes.c_int()
    call('cudaGetDeviceCount', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    if observation is not None:
        observation['runtime_device_count'] = count.value
    if count.value != 1:
        raise ValueError('CUDA runtime must expose exactly one scheduler-visible device; allocation association unknown')
    bus_id = ctypes.create_string_buffer(32)
    call('cudaDeviceGetPCIBusId', [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], bus_id, len(bus_id), 0)
    bus = bus_id.value.decode('ascii')
    pci_address(bus)
    if observation is not None:
        observation['runtime_pci_bus_id'] = bus
    return dict(runtime_device_count=1, runtime_visible_ordinal=0, runtime_pci_bus_id=bus)


def discover_visible_gpu(*, observation, runtime_provider=runtime_identity, driver_provider=None):
    """Owned startup child: resolve the original mask in the actual runtime namespace.

    Allocation association trusts the unchanged scheduler-supplied CVD. This
    is operational provenance, not an independently measured GRES File/UUID
    relation or OS denial of opening other cards. Uncertainty stays explicit.
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    observation['environment'] = {key: os.environ.get(key) for key in
        ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'ORCJAX_SLURM_VISIBLE_DEVICES',
         CONTRACT_ENVIRONMENT_KEY, ENVIRONMENT_KEY, 'SLURM_JOB_GPUS')}
    if (os.environ.get(CONTRACT_ENVIRONMENT_KEY) != DISCOVERY_GPU_CONTRACT
            or os.environ.get(ENVIRONMENT_KEY) or visible != os.environ.get('ORCJAX_SLURM_VISIBLE_DEVICES')
            or not (visible.isdigit() or valid_uuid(visible))):
        raise ValueError('discovery requires original single scheduler CVD and explicit v2 contract; allocation association unknown')
    observation.update(runtime={}, driver={})
    identity = (runtime_identity(observation=observation['runtime'])
        if runtime_provider is runtime_identity else runtime_provider())
    observation['runtime_identity'] = identity
    if identity.get('runtime_device_count') != 1 or identity.get('runtime_visible_ordinal') != 0:
        raise ValueError('runtime identity is unknown or ambiguous')
    driver_provider = driver_identity if driver_provider is None else driver_provider
    driver = driver_provider(pci_bus_id=identity.get('runtime_pci_bus_id'), observation=observation['driver'])
    observation['driver_identity'] = driver
    uuid = driver.get('driver_gpu_uuid')
    if (driver.get('driver_device_count', 0) < 1 or not valid_uuid(uuid)
            or pci_address(driver.get('driver_pci_bus_id')) != pci_address(identity.get('runtime_pci_bus_id'))):
        raise ValueError('independent PCI-selected CUDA-driver UUID discovery gate failed')
    if valid_uuid(visible) and uuid.lower() != visible.lower():
        raise ValueError('PCI-selected driver UUID differs from scheduler original UUID')
    return dict(passed=True, gpu_uuid=uuid, original_CUDA_VISIBLE_DEVICES=visible,
        CUDA_DEVICE_ORDER=os.environ.get('CUDA_DEVICE_ORDER'),
        slurm_gpu_id=os.environ.get('SLURM_JOB_GPUS'),
        allocation_association='trusted_scheduler_original_CVD_same_runtime_namespace',
        independent_GRES_index_to_UUID_verified=False,
        UUID_source='CUDA_runtime_ordinal_PCI_to_driver_physical_handle_UUID',
        os_other_gpu_open_denial='not_verified', **identity, **driver)


def driver_identity(driver=None, *, observation=None, pci_bus_id=None):
    """Observe driver-visible devices independently of Slurm/NVML indices.

    Legacy v1 requires one driver-enumerated device. V2 selects a handle by
    the runtime-measured PCI address and records total enumeration separately.
    Neither path infers UUIDs from Slurm/NVML/PJRT ordinal numbers.
    """
    if observation is not None:
        observation.update(library='libcuda.so.1', calls=[])
    driver = driver if driver is not None else ctypes.CDLL('libcuda.so.1')
    def call(name, argtypes, *arguments):
        row = dict(api=name)
        if observation is not None:
            observation['calls'].append(row)
        function = getattr(driver, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        code = function(*arguments)
        if observation is not None:
            row['returncode'] = code
        if code != 0:
            raise RuntimeError(f'{name} failed with CUDA driver result {code}')
    call('cuInit', [ctypes.c_uint], 0)
    count = ctypes.c_int()
    call('cuDeviceGetCount', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    if observation is not None:
        observation['driver_device_count'] = count.value
    if pci_bus_id is None and count.value != 1:
        raise ValueError('CUDA driver must independently expose exactly one device')
    if count.value < 1:
        raise ValueError('CUDA driver exposes no physical devices')
    device = ctypes.c_int()
    if pci_bus_id is None:
        call('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
    else:
        pci_address(pci_bus_id)
        call('cuDeviceGetByPCIBusId', [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p], ctypes.byref(device), pci_bus_id.encode('ascii'))
        actual_bus = ctypes.create_string_buffer(32)
        call('cuDeviceGetPCIBusId', [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], actual_bus, len(actual_bus), device)
        if pci_address(actual_bus.value.decode('ascii')) != pci_address(pci_bus_id):
            raise ValueError('CUDA driver selected PCI handle differs from runtime PCI address')
    if observation is not None:
        observation['driver_device_handle'] = device.value
    uuid_bytes = (ctypes.c_ubyte * 16)()
    # Prefer the current UUID API; the older UUID API is valid for these
    # non-MIG V100 allocations and supplies the same physical UUID bytes.
    name = 'cuDeviceGetUuid_v2' if hasattr(driver, 'cuDeviceGetUuid_v2') else 'cuDeviceGetUuid'
    if observation is not None:
        observation['driver_uuid_api'] = name
    call(name, [ctypes.c_void_p, ctypes.c_int], ctypes.byref(uuid_bytes), device)
    encoded = bytes(uuid_bytes).hex()
    uuid = 'GPU-' + '-'.join((encoded[:8], encoded[8:12], encoded[12:16], encoded[16:20], encoded[20:]))
    if observation is not None:
        observation['driver_gpu_uuid'] = uuid
    result = dict(driver_device_count=count.value, driver_visible_ordinal=0 if pci_bus_id is None else None,
                  driver_gpu_uuid=uuid, driver_uuid_api=name)
    if pci_bus_id is not None:
        result.update(driver_physical_handle=device.value, driver_pci_bus_id=actual_bus.value.decode('ascii'))
    return result

def verify_expected_gpu(devices, expected_uuid, *, identity_provider=driver_identity, observation=None):
    if observation is None:
        return _verify_expected_gpu(devices, expected_uuid, identity_provider=identity_provider)
    observation.update(expected_gpu_uuid=expected_uuid,
        child_CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        pjrt_devices=[dict(device=str(device), platform=getattr(device, 'platform', None),
            local_hardware_id=getattr(device, 'local_hardware_id', None)) for device in devices],
        pjrt_device_count=len(devices), driver={})
    try:
        return _verify_expected_gpu(devices, expected_uuid, identity_provider=identity_provider, observation=observation)
    except Exception as error:
        observation['error'] = dict(type=type(error).__name__, message=str(error))
        raise CudaIdentityError(error, observation) from error


def _verify_expected_gpu(devices, expected_uuid, *, identity_provider, observation=None):
    if not valid_uuid(expected_uuid):
        raise ValueError('expected physical GPU UUID is invalid')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_uuid:
        raise ValueError('child CUDA_VISIBLE_DEVICES differs from admitted physical UUID')
    if len(devices) != 1 or getattr(devices[0], 'platform', None) != 'gpu' or getattr(devices[0], 'local_hardware_id', None) != 0:
        raise ValueError('PJRT must expose the single admitted GPU at local hardware ordinal zero')
    v2 = os.environ.get(CONTRACT_ENVIRONMENT_KEY) == DISCOVERY_GPU_CONTRACT
    runtime = None
    if v2:
        runtime_observation = {} if observation is None else observation.setdefault('runtime', {})
        runtime = runtime_identity(observation=runtime_observation)
        if runtime.get('runtime_device_count') != 1 or runtime.get('runtime_visible_ordinal') != 0:
            raise ValueError('CUDA runtime must expose exactly one selected device')
        identity = identity_provider(pci_bus_id=runtime['runtime_pci_bus_id'],
            observation=None if observation is None else observation['driver'])
    else:
        identity = (driver_identity(observation=observation['driver'])
            if observation is not None and identity_provider is driver_identity else identity_provider())
    if observation is not None:
        observation['driver_identity'] = identity
    if ((not v2 and (identity.get('driver_device_count') != 1 or identity.get('driver_visible_ordinal') != 0))
            or (v2 and (identity.get('driver_device_count', 0) < 1
                or pci_address(identity.get('driver_pci_bus_id')) != pci_address(runtime['runtime_pci_bus_id'])))
            or str(identity.get('driver_gpu_uuid', '')).lower() != expected_uuid.lower()):
        raise ValueError('independent CUDA-driver UUID differs from admitted physical GPU')
    return dict(passed=True, expected_gpu_uuid=expected_uuid, gpu_execution_contract=os.environ.get(CONTRACT_ENVIRONMENT_KEY),
                child_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                pjrt_local_hardware_id=0, pjrt_device=str(devices[0]), **identity, **(runtime or {}))


def binding_report_passes(binding, expected_uuid, contract=None):
    """Validate already-read preflight/actual affine evidence, without GPU calls."""
    if (binding.get('passed') is not True or binding.get('expected_gpu_uuid') != expected_uuid
            or str(binding.get('driver_gpu_uuid', '')).lower() != expected_uuid.lower()
            or binding.get('pjrt_local_hardware_id') != 0):
        return False
    if contract != DISCOVERY_GPU_CONTRACT:
        return binding.get('driver_device_count') == 1
    try:
        return (binding.get('gpu_execution_contract') == DISCOVERY_GPU_CONTRACT
            and binding.get('runtime_device_count') == 1 and binding.get('runtime_visible_ordinal') == 0
            and binding.get('driver_device_count', 0) >= 1
            and pci_address(binding.get('driver_pci_bus_id')) == pci_address(binding.get('runtime_pci_bus_id')))
    except (ValueError, TypeError):
        return False


def main(argv=None):
    """Only the deadline-owning paid runtime-discovery shell invokes this CLI."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discover-output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.discover_output.exists():
        raise FileExistsError('GPU discovery output already exists; no overwrite/retry')
    report = dict(schema_version='T7_affine_gpu_discovery_v1', affine_science_executed=False)
    try:
        report['identity'] = discover_visible_gpu(observation=report)
        report['status'] = 'passed'
        code = 0
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        code = 2
    with args.discover_output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
