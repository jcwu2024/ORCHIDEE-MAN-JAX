"""Optional paid CUDA-driver/PJRT identity gate; no JAX import or compilation."""
from __future__ import annotations
import ctypes
import os
import re

ENVIRONMENT_KEY = 'ORCJAX_AFFINE_EXPECTED_GPU_UUID'

class CudaIdentityError(ValueError):
    def __init__(self, cause, observation):
        super().__init__(str(cause))
        self.cause, self.observation = cause, observation


def driver_identity(driver=None, *, observation=None):
    """Observe driver-visible devices independently of Slurm/NVML indices.

    Unsupported driver enumeration fails closed. This does not infer UUIDs
    from CUDA_VISIBLE_DEVICES or from PJRT's printed ordinal.
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
    if count.value != 1:
        raise ValueError('CUDA driver must independently expose exactly one device')
    device = ctypes.c_int()
    call('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
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
    return dict(driver_device_count=1, driver_visible_ordinal=0,
                driver_gpu_uuid=uuid, driver_uuid_api=name)

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
    if not isinstance(expected_uuid, str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', expected_uuid):
        raise ValueError('expected physical GPU UUID is invalid')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_uuid:
        raise ValueError('child CUDA_VISIBLE_DEVICES differs from admitted physical UUID')
    if len(devices) != 1 or getattr(devices[0], 'platform', None) != 'gpu' or getattr(devices[0], 'local_hardware_id', None) != 0:
        raise ValueError('PJRT must expose the single admitted GPU at local hardware ordinal zero')
    identity = (driver_identity(observation=observation['driver'])
        if observation is not None and identity_provider is driver_identity else identity_provider())
    if observation is not None:
        observation['driver_identity'] = identity
    if (identity.get('driver_device_count') != 1 or identity.get('driver_visible_ordinal') != 0
            or str(identity.get('driver_gpu_uuid', '')).lower() != expected_uuid.lower()):
        raise ValueError('independent CUDA-driver UUID differs from admitted physical GPU')
    return dict(passed=True, expected_gpu_uuid=expected_uuid,
                child_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                pjrt_local_hardware_id=0, pjrt_device=str(devices[0]), **identity)
