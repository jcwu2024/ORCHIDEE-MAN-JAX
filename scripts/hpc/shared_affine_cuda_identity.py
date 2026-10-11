"""Optional paid CUDA-driver/PJRT identity gate; no JAX import or compilation."""
from __future__ import annotations
import ctypes
import os
import re

ENVIRONMENT_KEY = 'ORCJAX_AFFINE_EXPECTED_GPU_UUID'

def driver_identity(driver=None):
    """Observe driver-visible devices independently of Slurm/NVML indices.

    Unsupported driver enumeration fails closed. This does not infer UUIDs
    from CUDA_VISIBLE_DEVICES or from PJRT's printed ordinal.
    """
    driver = driver if driver is not None else ctypes.CDLL('libcuda.so.1')
    def call(name, argtypes, *arguments):
        function = getattr(driver, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        code = function(*arguments)
        if code != 0:
            raise RuntimeError(f'{name} failed with CUDA driver result {code}')
    call('cuInit', [ctypes.c_uint], 0)
    count = ctypes.c_int()
    call('cuDeviceGetCount', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    if count.value != 1:
        raise ValueError('CUDA driver must independently expose exactly one device')
    device = ctypes.c_int()
    call('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
    uuid_bytes = (ctypes.c_ubyte * 16)()
    # Prefer the current UUID API; the older UUID API is valid for these
    # non-MIG V100 allocations and supplies the same physical UUID bytes.
    name = 'cuDeviceGetUuid_v2' if hasattr(driver, 'cuDeviceGetUuid_v2') else 'cuDeviceGetUuid'
    call(name, [ctypes.c_void_p, ctypes.c_int], ctypes.byref(uuid_bytes), device)
    encoded = bytes(uuid_bytes).hex()
    uuid = 'GPU-' + '-'.join((encoded[:8], encoded[8:12], encoded[12:16], encoded[16:20], encoded[20:]))
    return dict(driver_device_count=1, driver_visible_ordinal=0,
                driver_gpu_uuid=uuid, driver_uuid_api=name)

def verify_expected_gpu(devices, expected_uuid, *, identity_provider=driver_identity):
    if not isinstance(expected_uuid, str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', expected_uuid):
        raise ValueError('expected physical GPU UUID is invalid')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_uuid:
        raise ValueError('child CUDA_VISIBLE_DEVICES differs from admitted physical UUID')
    if len(devices) != 1 or getattr(devices[0], 'platform', None) != 'gpu' or getattr(devices[0], 'local_hardware_id', None) != 0:
        raise ValueError('PJRT must expose the single admitted GPU at local hardware ordinal zero')
    identity = identity_provider()
    if (identity.get('driver_device_count') != 1 or identity.get('driver_visible_ordinal') != 0
            or str(identity.get('driver_gpu_uuid', '')).lower() != expected_uuid.lower()):
        raise ValueError('independent CUDA-driver UUID differs from admitted physical GPU')
    return dict(passed=True, expected_gpu_uuid=expected_uuid,
                child_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                pjrt_local_hardware_id=0, pjrt_device=str(devices[0]), **identity)
