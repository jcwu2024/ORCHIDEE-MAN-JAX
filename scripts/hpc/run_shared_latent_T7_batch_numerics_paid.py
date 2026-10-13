"""CPU-only admission for exactly one proposed paid Slurm affine diagnostic.

Scientific probe computation is unchanged. Live execution requires approval,
an immutable committed manifest, exact scheduler/cgroup admission, the declared
paid-only launch management chain and a cooperative shared mutex.
"""
from __future__ import annotations
import datetime
import csv
from decimal import Decimal, ROUND_CEILING
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.hpc import run_shared_latent_T7_batch_numerics as free
from scripts.hpc import shared_affine_cuda_identity as cuda_identity
from scripts.hpc.shared_affine_gpu_admission import (
    AffineScopeLock, SCOPE_LOCK, MANAGEMENT, read_free_admission)

SCOPE_ID = 'T7-summary-affine-paid-single-gpu-20261012-v2'
CHECKOUT_TARGET = free.REMOTE / 'checkouts/T7-affine-paid-single-gpu-20261012-v2'
SCHEMA = 'T7_affine_paid_slurm_scope_v1'
STRICT_GPU_CONTRACT = 'T7_affine_strict_device_isolation_v1'
GPU_EXECUTION_CONTRACT = cuda_identity.DISCOVERY_GPU_CONTRACT
OUTPUT_DIR = free.REMOTE / 'outputs/diagnostics' / SCOPE_ID
OUTPUT = OUTPUT_DIR / 'result.v1.json'
LEDGER = OUTPUT_DIR / (SCOPE_ID + '.ledger.json')
SLURM = '/rmprog/slurm/v22.05.7/bin/scontrol'
NODE = 'ibc13b03n04'
SOURCE_PATHS = (*free.SOURCE_PATHS,
    'scripts/hpc/freeze_shared_affine_paid_scope.py',
    'scripts/hpc/run_shared_latent_T7_batch_numerics_paid.py',
    'scripts/hpc/run_shared_latent_T7_batch_numerics_paid.sh',
    'scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh')
RESOURCES = dict(partition='gnall', node=NODE, nodes=1, tasks=1,
                 allocated_slurm_cpus=4, gpus=1, wall_seconds=1200,
                 nonexclusive=True, requeue=False)
RATES = dict(cpu_cny_per_allocated_cpu_hour='0.07', gpu_cny_per_gpu_hour='2.20')
EVIDENCE_ENVIRONMENT = (
    'SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_CPUS_PER_TASK',
    'SLURM_NTASKS', 'SLURM_JOB_NUM_NODES', 'SLURM_JOB_PARTITION',
    'SLURM_JOB_START_TIME', 'SLURM_JOB_END_TIME', 'ORCJAX_SLURM_VISIBLE_DEVICES',
    'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'ORCJAX_AFFINE_EXPECTED_GPU_UUID', 'ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'JAX_PLATFORMS')


def startup_observation(environment):
    return dict(schema_version='T7_paid_startup_observation_v1', stage='scope',
                environment={key: environment.get(key) for key in EVIDENCE_ENVIRONMENT},
                files=[], commands=[])


class PaidStartupError(ValueError):
    def __init__(self, message, observation, cause=None):
        super().__init__(message)
        self.observation, self.cause = observation, cause


def _read_text(path, observation, label):
    row = dict(label=label, path=str(path), operation='read_text')
    observation['files'].append(row)
    try:
        row['text'] = Path(path).read_text(encoding='utf-8')
        row['status'] = 'read'
        return row['text']
    except OSError as error:
        row.update(status='failed', error_type=type(error).__name__, error=str(error),
                   errno=error.errno)
        raise PaidStartupError(str(error), observation, error) from error


def _record_command(command, observation):
    row = dict(command=command, status='started')
    observation['commands'].append(row)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=4, check=True)
    except (subprocess.SubprocessError, OSError) as error:
        row.update(status='failed', **(subprocess_failure_detail(error) or
            dict(error_type=type(error).__name__, error=str(error))))
        raise PaidStartupError(str(error), observation, error) from error
    row.update(status='completed', returncode=getattr(result, 'returncode', 0),
               stdout=result.stdout, stderr=getattr(result, 'stderr', ''))
    return result

def quote():
    cpu = Decimal('4') * Decimal('1200') / Decimal('3600') * Decimal('0.07')
    gpu = Decimal('1200') / Decimal('3600') * Decimal('2.20')
    total = cpu + gpu
    return dict(cpu_cny=str(cpu), gpu_cny=str(gpu), total_cny=str(total),
                display_ceiling_cny=str(total.quantize(Decimal('.01'), rounding=ROUND_CEILING)),
                proposed_authorization_ceiling_cny='1.00')

def parse_job(text):
    fields = {}
    for token in shlex.split(text):
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        if key in fields:
            raise ValueError('duplicate Slurm job field: ' + key)
        fields[key] = value
    return fields

def seconds(value):
    days, sep, clock = value.partition('-')
    if not sep:
        clock, days = days, '0'
    fields = clock.split(':')
    if len(fields) != 3 or not all(part.isdigit() for part in [days, *fields]):
        raise ValueError('invalid finite Slurm duration')
    hour, minute, second = map(int, fields)
    if minute >= 60 or second >= 60:
        raise ValueError('invalid Slurm duration components')
    return int(days)*86400 + hour*3600 + minute*60 + second

def tres(value):
    result = {}
    for item in value.split(','):
        key, sep, number = item.partition('=')
        if not sep or not key or key in result:
            raise ValueError('invalid or duplicate allocation TRES')
        result[key] = number
    return result

def allocation_epochs(environment, job, *, observation=None):
    """Scheduler clock; 22.05.7 batch env does not supply JOB_START/END_TIME.

    Original env tokens, when present, are additional exact crosschecks.
    Neither shell/Python startup nor a renewed deadline is an admissible clock.
    """
    if observation is not None:
        observation['allocation_clock'] = dict(job=job,
            original_start=environment.get('SLURM_JOB_START_TIME'),
            original_end=environment.get('SLURM_JOB_END_TIME'), protected_release_seconds=180)
    if (not str(environment.get('SLURM_JOB_ID', '')).isdigit()
            or job.get('JobId') != environment['SLURM_JOB_ID']
            or job.get('JobState') != 'RUNNING' or job.get('TimeLimit') != '00:20:00'):
        raise ValueError('actual running Slurm job and exact wall limit required')
    epochs = []
    for field, token in [('StartTime', 'SLURM_JOB_START_TIME'), ('EndTime', 'SLURM_JOB_END_TIME')]:
        text = job.get(field, '')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?', text):
            raise ValueError('invalid scheduler allocation ' + field)
        epoch = int(datetime.datetime.fromisoformat(text).timestamp())
        if epoch <= 0:
            raise ValueError('invalid scheduler allocation epoch')
        if token in environment:
            original = environment[token]
            if not re.fullmatch(r'[0-9]+', original) or int(original) != epoch:
                raise ValueError('original allocation epoch differs from scheduler ' + field)
        epochs.append(epoch)
    start, end = epochs
    runtime = seconds(job.get('RunTime', ''))
    now = time.time()
    if observation is not None:
        observation['allocation_clock']['observed_epoch'] = now
    if end - start != 1200:
        raise ValueError('allocation end epoch differs from exact paid wall limit')
    if start > now or now >= end or runtime >= 1020:
        raise ValueError('allocation has exhausted protected release time or is not current')
    if abs(now - start - runtime) > 15:
        raise ValueError('Slurm allocation start/runtime/current clock binding differs')
    return start, end


def allocated_tres(job):
    """Allocated fields only, with actual allocation detail for legacy TRES.

    22.05.7 job_info.c:696 emits tres_alloc_str, falling back to tres_req_str
    without allocation. Nodes/CPU_IDs/GRES at 808/851 are allocated detail from
    job_resrcs/core_bitmap/gres_detail_str. Require RUNNING and these records;
    never accept ReqTRES/JOB_GRES as allocation proof.
    """
    sources = [key for key in ('AllocTRES', 'TRES') if key in job]
    if not sources:
        raise ValueError('actual allocated TRES unavailable')
    values = [tres(job[key]) for key in sources]
    if any(value != values[0] for value in values[1:]):
        raise ValueError('conflicting actual allocated TRES fields')
    if 'TRES' in sources:
        if job.get('JobState') != 'RUNNING' or job.get('Nodes') != NODE:
            raise ValueError('legacy TRES lacks actual single-node allocation detail')
        ids = set()
        for item in job.get('CPU_IDs', '').split(','):
            if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', item):
                raise ValueError('actual allocated CPU IDs unavailable')
            bounds = list(map(int, item.split('-')))
            lo, hi = bounds[0], bounds[-1]
            if hi < lo or hi >= 112 or ids.intersection(range(lo, hi + 1)):
                raise ValueError('invalid actual allocated CPU IDs')
            ids.update(range(lo, hi + 1))
        if len(ids) != 4:
            raise ValueError('actual allocated CPU IDs differ from four Slurm CPUs')
        if not job.get('GRES'):
            raise ValueError('legacy TRES lacks actual allocated GPU detail')
    return values[0], sources


def scheduler_allocation_epochs():
    observation = startup_observation(os.environ)
    observation['stage'] = 'scheduler_allocation_clock'
    try:
        return _scheduler_allocation_epochs(observation)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        cause = error.cause if isinstance(error, PaidStartupError) and error.cause else error
        observation['error'] = dict(type=type(cause).__name__, message=str(cause))
        raise PaidStartupError(str(cause), observation, cause) from error
    finally:
        # stdout remains the exact two-token shell protocol; stderr is evidence.
        print(json.dumps(dict(paid_startup_observation=observation)), file=sys.stderr)


def _scheduler_allocation_epochs(observation):
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise ValueError('numeric actual Slurm job ID required')
    result = _record_command([SLURM, 'show', 'job', '--details', '-o',
        os.environ.get('SLURM_JOB_ID', 'missing')], observation)
    job = parse_job(result.stdout)
    observation['slurm_job'] = job
    start, end = allocation_epochs(os.environ, job, observation=observation)
    print(start, end)
    return 0


def validate_allocation(scope, environment, job, hostname, *, observation=None):
    """No GPU query: exact paid scheduling/charge admission, never CPU fallback."""
    if observation is not None:
        observation.update(slurm_job=job, hostname=hostname,
            expected_resources=RESOURCES, expected_rates=RATES, expected_quote=quote())
    if scope.get('resources') != RESOURCES or scope.get('rates') != RATES:
        raise ValueError('paid resource/rate contract differs')
    required = dict(JobId=environment.get('SLURM_JOB_ID'), JobState='RUNNING',
        Partition='gnall', NodeList=NODE, NumNodes='1', NumCPUs='4', NumTasks='1',
        Requeue='0', TimeLimit='00:20:00')
    if (not required['JobId'] or hostname.split('.')[0] != NODE
            or any(job.get(key) != value for key, value in required.items())
            or environment.get('SLURM_CPUS_PER_TASK') != '4'
            or environment.get('SLURM_NTASKS') != '1'
            or environment.get('SLURM_JOB_NUM_NODES') != '1'
            or environment.get('SLURM_JOB_PARTITION') != 'gnall'):
        raise ValueError('actual Slurm job/node/resources differ from paid scope')
    allocated, allocation_fields = allocated_tres(job)
    if allocated.get('cpu') != '4' or allocated.get('gres/gpu') != '1' or allocated.get('node') != '1':
        raise ValueError('actual allocated TRES CPU/GPU count differs')
    typed_gpus = [value for key, value in allocated.items() if key.startswith('gres/gpu:')]
    if len(typed_gpus) > 1 or any(value != '1' for value in typed_gpus):
        raise ValueError('typed GPU allocation differs')
    details = [job[key] for key in ('GresDetail', 'GRES') if key in job]
    if not details or len(set(details)) != 1 or any(not re.fullmatch(r'gpu(?::[A-Za-z0-9_.-]+)?:1\(IDX:[0-9]+\)', detail)
                          or re.findall(r'IDX:([0-9]+)', detail) != [environment.get('SLURM_JOB_GPUS')]
                          for detail in details):
        raise ValueError('Slurm GresDetail physical allocation differs from environment')
    if job.get('OverSubscribe') != 'OK':
        raise ValueError('unexpected job resource sharing policy')
    start, end = allocation_epochs(environment, job, observation=observation)
    maximum = Decimal(scope.get('approved_cost_ceiling_cny', '0'))
    if maximum != Decimal('1.00') or Decimal(quote()['total_cny']) > maximum:
        raise ValueError('paid exact quote or approved ceiling differs')
    return dict(job=job, allocation_start_epoch=start, allocation_end_epoch=end,
                allocation_clock_source='scheduler StartTime/EndTime; optional original env crosschecks',
                actual_NumCPUs=4, allocation_TRES_fields=allocation_fields,
                actual_AllocTRES=allocated, quote=quote(), hostname=hostname,
                original_visible_devices=environment.get('ORCJAX_SLURM_VISIBLE_DEVICES'),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

class GpuObservationError(ValueError):
    def __init__(self, message, observation, subprocess_failure=None, cause=None):
        super().__init__(message)
        self.observation = observation
        self.subprocess_failure = subprocess_failure
        self.cause = cause

def _gpu_query(command, observation, timeout_seconds=4):
    try:
        if timeout_seconds <= 0:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=True)
    except (subprocess.SubprocessError, OSError) as error:
        failure = subprocess_failure_detail(error)
        if failure is None:
            failure = dict(command=command, stdout=None, stderr=None,
                           error_type=type(error).__name__, error=str(error))
        observation['commands'].append(dict(status='failed', **failure))
        raise GpuObservationError('GPU identity query command failed', observation, failure, error) from error
    completed = dict(command=command, returncode=result.returncode,
                     stdout=result.stdout, stderr=result.stderr, status='completed')
    observation['commands'].append(completed)
    return completed

def observe_gpu_mapping(environment, expected_uuid=None, *, pci_bus_id=None, deadline=None, clock=time.monotonic):
    """Query the runtime-discovered UUID; Slurm GRES is never an NVML selector."""
    observation = dict(schema_version='T7_paid_gpu_observation_v1', commands=[])
    slurm_id = environment.get('SLURM_JOB_GPUS', '')
    if not slurm_id.isdigit():
        raise GpuObservationError('single physical SLURM_JOB_GPUS required before observation', observation)
    if expected_uuid is None:
        expected_uuid = environment.get('ORCJAX_SLURM_VISIBLE_DEVICES', '')
    if not str(expected_uuid).startswith('GPU-') or ',' in expected_uuid:
        raise GpuObservationError('runtime-discovered GPU UUID required; allocation association unknown', observation)
    observation.update(slurm_gpu_id=slurm_id, selector_uuid=expected_uuid)
    def remaining():
        return 4 if deadline is None else min(4, deadline-clock())
    try:
        csv_result = _gpu_query(['nvidia-smi', '-i', expected_uuid, '--query-gpu=index,uuid,name',
            '--format=csv,noheader,nounits'], observation, remaining())
        rows = list(csv.reader(csv_result['stdout'].splitlines()))
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError('nvidia-smi CSV did not return exactly one index/UUID/name row')
        host_index, gpu_uuid, gpu_name = (field.strip() for field in rows[0])
        if (not host_index.isdigit() or gpu_uuid != expected_uuid
                or not gpu_name):
            raise ValueError('discovered GPU UUID/NVML index/name mapping differs')

        xml_result = _gpu_query(['nvidia-smi', '-i', gpu_uuid, '-q', '-x'], observation, remaining())
        root = ET.fromstring(xml_result['stdout'])
        if root.tag != 'nvidia_smi_log':
            raise ValueError('nvidia-smi XML root is not nvidia_smi_log')
        gpus = root.findall('gpu')
        if len(gpus) != 1 or len(root.findall('.//gpu')) != 1:
            raise ValueError('nvidia-smi XML did not return exactly one GPU')
        gpu = gpus[0]
        uuids = gpu.findall('uuid')
        names = gpu.findall('product_name')
        minors = gpu.findall('minor_number')
        if len(uuids) != 1 or len(names) != 1 or len(minors) != 1:
            raise ValueError('nvidia-smi XML UUID/name/minor fields must each occur exactly once')
        xml_uuid = (uuids[0].text or '').strip()
        xml_name = (names[0].text or '').strip()
        minor_text = (minors[0].text or '').strip()
        if xml_uuid != gpu_uuid:
            raise ValueError('nvidia-smi XML UUID differs from CSV-selected GPU')
        if not xml_name or xml_name != gpu_name:
            raise ValueError('nvidia-smi XML product name differs from CSV-selected GPU')
        if not re.fullmatch(r'[0-9]+', minor_text):
            raise ValueError('nvidia-smi XML Linux minor_number is unavailable or invalid')
        if pci_bus_id is not None:
            pci_fields = gpu.findall('pci/pci_bus_id')
            if (len(pci_fields) != 1 or
                    cuda_identity.pci_address((pci_fields[0].text or '').strip()) != cuda_identity.pci_address(pci_bus_id)):
                raise ValueError('NVML XML physical PCI address differs from runtime-selected device')
        return dict(index=host_index, uuid=gpu_uuid, name=gpu_name, minor_number=int(minor_text),
                    slurm_gpu_id=slurm_id), observation
    except GpuObservationError:
        raise
    except (ValueError, ET.ParseError) as error:
        raise GpuObservationError(str(error), observation) from error

def admit_gpu_mapping(environment, gpu_rows, devices_rules, *, observation=None,
                      contract=STRICT_GPU_CONTRACT):
    """Bind Slurm physical GPU ID, UUID and cgroup minor, separately from CVD.

    The original visible ordinal is retained as evidence, never used as a host
    index. The child selects the admitted UUID inside the Slurm device cgroup.
    The strict default rejects wildcard access. The explicitly selected new
    contract retains permission assessment separately from execution binding.
    """
    if observation is not None:
        observation['mapping_inputs'] = dict(gpu_rows=gpu_rows, devices_rules=devices_rules,
            SLURM_JOB_GPUS=environment.get('SLURM_JOB_GPUS'),
            ORCJAX_SLURM_VISIBLE_DEVICES=environment.get('ORCJAX_SLURM_VISIBLE_DEVICES'))
    selected = environment.get('SLURM_JOB_GPUS', '')
    visible = environment.get('ORCJAX_SLURM_VISIBLE_DEVICES', '')
    if not selected.isdigit() or not visible or ',' in visible or len(gpu_rows) != 1:
        raise ValueError('single Slurm physical GPU and original visible mapping required')
    row = gpu_rows[0]
    host_index = int(row['index'])
    minor = int(row['minor_number'])
    device_pair = (int(row['device_major']), int(row['device_minor']))
    if (host_index < 0 or not str(row['uuid']).startswith('GPU-')
            or 'V100' not in row['name']):
        raise ValueError('Slurm physical GPU/UUID/V100 mapping differs')
    if minor < 0 or device_pair[1] != minor:
        raise ValueError('XML minor_number and actual kernel GPU device minor differ')
    if visible.startswith('GPU-') and visible != row['uuid']:
        raise ValueError('visible GPU UUID differs')
    if not visible.isdigit() and not visible.startswith('GPU-'):
        raise ValueError('unsupported visible GPU mapping')
    permitted = set()
    controls = {(int(item['major']), int(item['minor'])) for item in row.get('control_devices', [])}
    if device_pair in controls:
        raise ValueError('GPU data device cannot be a driver control device')
    if contract == GPU_EXECUTION_CONTRACT:
        association = row.get('allocation_association')
        if (row.get('slurm_gpu_id') != selected
                or association != 'trusted_scheduler_original_CVD_same_runtime_namespace'
                or row.get('discovery_gpu_uuid') != row['uuid']):
            raise ValueError('allocated GPU association is unknown or conflicts with discovery')
        assessment = assess_device_rules(devices_rules, device_pair, controls)
        if observation is not None:
            observation['os_device_access_assessment'] = assessment
        if not assessment['selected_device_read_write_permitted']:
            raise ValueError('cgroup does not permit allocated GPU read/write access')
        if cuda_identity.valid_uuid(visible):
            basis = 'scheduler_original_UUID_matches_runtime_driver_NVML_kernel'
        else:
            basis = 'scheduler_original_numeric_mask_same_runtime_driver_NVML_kernel'
        return dict(gpu_execution_contract=contract, gpu_host_index=host_index,
            gpu_uuid=row['uuid'], gpu_name=row['name'],
            slurm_gpu_id=selected, allocation_association=association,
            independent_GRES_index_to_UUID_verified=False,
            allocation_association_basis=basis,
            original_CUDA_VISIBLE_DEVICES=visible, child_CUDA_VISIBLE_DEVICES=row['uuid'],
            child_visible_ordinal=0, actual_GPU_major=device_pair[0], actual_GPU_minor=device_pair[1],
            cgroup_devices_rules=devices_rules, os_device_access_assessment=assessment)
    if contract != STRICT_GPU_CONTRACT:
        raise ValueError('unknown GPU execution contract; no automatic fallback')
    if host_index != int(selected):
        raise ValueError('legacy strict Slurm/NVML index contract differs')
    for number, line in enumerate(devices_rules.splitlines(), 1):
        if observation is not None:
            observation['current_device_rule'] = dict(line_number=number, raw=line,
                selected_kernel_pair=list(device_pair), control_pairs=[list(pair) for pair in sorted(controls)])
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('invalid cgroup devices rule')
        kind, device, permissions = parts
        major, sep, minor = device.partition(':')
        if observation is not None:
            observation['current_device_rule'].update(kind=kind, major=major, minor=minor, permissions=permissions)
        if (not sep or kind not in ('a', 'b', 'c') or not set(permissions) <= set('rwm')
                or not permissions or any(value != '*' and not value.isdigit() for value in (major, minor))):
            raise ValueError('invalid cgroup device permission rule')
        # m-only wildcard grants mknod, not access to a GPU. Any wildcard
        # read/write permission for the GPU's actual major cannot prove the
        # allocated single-device boundary, including c *:N.
        access = bool(set(permissions) & set('rw'))
        if access and (kind == 'a' or (kind == 'c' and major in ('*', str(device_pair[0])) and (major == '*' or minor == '*'))):
            if observation is not None:
                observation['current_device_rule']['rejection_rationale'] = 'unchanged conservative wildcard read/write predicate; not evidence of another GPU computation'
            raise ValueError('wildcard GPU access does not prove Slurm device isolation')
        if access and kind == 'c' and major == str(device_pair[0]):
            pair = (int(major), int(minor))
            if pair not in controls:
                if not set('rw') <= set(permissions):
                    raise ValueError('GPU data device lacks required read/write permission')
                permitted.add(pair)
    if permitted != {device_pair}:
        raise ValueError('cgroup permitted GPU minor differs from single allocated GPU')
    return dict(gpu_host_index=host_index, gpu_uuid=row['uuid'], gpu_name=row['name'],
                original_CUDA_VISIBLE_DEVICES=visible, child_CUDA_VISIBLE_DEVICES=row['uuid'],
                child_visible_ordinal=0, actual_GPU_major=device_pair[0], actual_GPU_minor=device_pair[1],
                cgroup_GPU_minors=[pair[1] for pair in sorted(permitted)],
                cgroup_devices_rules=devices_rules)


def assess_device_rules(raw_rules, selected_pair, controls):
    """Bounded permission evidence, never a full hardware-isolation claim.

    Intersect each rule with the selected data major before excluding recorded
    controls. Cross-major wildcard uncertainty is separate from that intersection.
    Does not open devices or enumerate hardware; width is not allocation proof.
    """
    rules, selected_permissions, broader, cross_major = [], set(), [], []
    for number, line in enumerate(raw_rules.splitlines(), 1):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('invalid cgroup devices rule')
        kind, device, permissions = parts
        major, sep, minor = device.partition(':')
        if (not sep or kind not in ('a', 'b', 'c') or not permissions
                or not set(permissions) <= set('rwm')
                or any(value != '*' and not value.isdigit() for value in (major, minor))):
            raise ValueError('invalid cgroup device permission rule')
        row = dict(line_number=number, raw=line, kind=kind, major=major,
                   minor=minor, permissions=permissions)
        rules.append(row)
        matching = kind in ('a', 'c') and major in ('*', str(selected_pair[0]))
        if matching and minor in ('*', str(selected_pair[1])):
            selected_permissions.update(permissions)
        if matching and set(permissions) & set('rw'):
            if major == '*':
                cross_major.append(dict(line_number=number,
                    reason='wildcard major permits other majors; hardware roles not established'))
            if minor == '*':
                broader.append(dict(line_number=number, reason='wildcard same-major device access; hardware inventory not established'))
            elif (selected_pair[0], int(minor)) not in {selected_pair, *controls}:
                broader.append(dict(line_number=number, reason='additional same-major device permission; hardware role not established'))
    return dict(raw_devices_rules=raw_rules, parsed_rules=rules,
        selected_kernel_pair=list(selected_pair), control_pairs=[list(pair) for pair in sorted(controls)],
        selected_device_read_write_permitted=set('rw') <= selected_permissions,
        broader_or_ambiguous_access_rules=broader,
        cross_major_unknown_access_rules=cross_major,
        observed_data_major_access='broader_or_ambiguous' if broader else 'selected_and_recorded_controls_only',
        os_other_gpu_open_denial='not_verified', os_isolation_required=False,
        limitation='rule analysis only; no complete device inventory/open test; not evidence of computation on another GPU')

def read_cgroup_rules(job_id, *, proc_path=Path('/proc/self/cgroup'), cgroup_base=Path('/sys/fs/cgroup/devices'), observation=None):
    observation = observation if observation is not None else startup_observation({})
    observation['cgroup'] = dict(job_id=str(job_id), proc_path=str(proc_path), devices_base=str(cgroup_base), controllers=[])
    if not str(job_id).isdigit():
        raise ValueError('numeric actual Slurm job ID required for cgroup binding')
    memberships = []
    for line in _read_text(proc_path, observation, 'proc_self_cgroup').splitlines():
        _, controllers, member = line.split(':', 2)
        observation['cgroup']['controllers'].append(dict(controllers=controllers.split(','), membership=member))
        if 'devices' in controllers.split(','):
            memberships.append(member)
    if len(memberships) != 1:
        raise ValueError('Slurm job device cgroup-v1 membership unverified')
    job_components = [item for item in memberships[0].split('/') if item.startswith('job_')]
    if job_components != ['job_' + str(job_id)]:
        raise ValueError('device cgroup belongs to a different actual Slurm job')
    base = cgroup_base.resolve()
    path = (base / memberships[0].lstrip('/') / 'devices.list').resolve()
    observation['cgroup'].update(resolved_base=str(base), devices_list_path=str(path))
    path.relative_to(base)
    return str(path), _read_text(path, observation, 'job_devices_list')

def source_manifest(*, observation=None):
    return {path: free.small_sha(ROOT / path, 1000000, observation=observation) for path in SOURCE_PATHS}

def proposed_scope():
    """Preparation snapshot. No invented candidate-commit identity."""
    return dict(schema_version=SCHEMA, scope_id=SCOPE_ID, status='proposed',
        execution_authorized=False, approval_note='', code_git_head=None,
        gpu_execution_contract=GPU_EXECUTION_CONTRACT, os_device_open_isolation_required=False,
        source_files=source_manifest(), caps=free.CAPS, resources=RESOURCES, rates=RATES,
        quote=quote(), approved_cost_ceiling_cny='1.00', repeats=3,
        cards=1, nonexclusive=True, output=OUTPUT.as_posix(), ledger=LEDGER.as_posix(),
        management_admission=MANAGEMENT, shared_mutex=SCOPE_LOCK.as_posix(),
        input_sha256=dict(operand=free.OPERAND_SHA, receipt=free.RECEIPT_SHA,
                          capture_source=free.CAPTURE_SHA),
        identity_status='candidate source hashes only; code_git_head binds real accepted commit after versioning')

def prepare(args):
    observation = getattr(args, '_paid_startup_observation', None)
    observation = observation if observation is not None else startup_observation(os.environ)
    observation['stage'] = 'capture_identity'
    observation['arguments'] = {key: str(getattr(args, key, None)) for key in
        ('checkout_root', 'operands', 'capture_receipt', 'capture_source', 'scope_json', 'output', 'ledger', 'shell_started_epoch', 'dry_run')}
    args._paid_startup_observation = observation
    identities, capture = free.admit_capture(args)
    observation['capture_identities'] = identities
    observation['stage'] = 'scope_contract'
    scope = json.loads(_read_text(args.scope_json, observation, 'prepared_scope'))
    observation['scope'] = scope
    if (scope.get('schema_version') != SCHEMA or scope.get('scope_id') != SCOPE_ID
            or scope.get('gpu_execution_contract') != GPU_EXECUTION_CONTRACT
            or scope.get('os_device_open_isolation_required') is not False
            or scope.get('status') not in ('proposed', 'open') or scope.get('caps') != free.CAPS
            or scope.get('resources') != RESOURCES or scope.get('rates') != RATES
            or scope.get('quote') != quote() or scope.get('approved_cost_ceiling_cny') != '1.00'
            or scope.get('repeats') != 3 or scope.get('cards') != 1
            or scope.get('nonexclusive') is not True
            or scope.get('input_sha256') != identities
            or scope.get('management_admission') != MANAGEMENT
            or scope.get('shared_mutex') != SCOPE_LOCK.as_posix()):
        raise ValueError('paid independent scope/resource/identity contract differs')
    if args.output.resolve() != Path(scope['output']).resolve() or args.ledger.resolve() != Path(scope['ledger']).resolve():
        raise ValueError('paid output/ledger paths differ')
    output = args.output.resolve()
    destinations = [args.ledger, output, *[output.with_suffix(suffix) for suffix in
        ('.results.npz', '.partial.npz', '.supervisor.json', '.runtime-preflight.json', '.outer-timeout.json')]]
    destinations.append(output.with_suffix('.startup-identity.json'))
    destinations.append(output.with_suffix('.discovery.json'))
    destinations += [output.with_suffix('.' + route + '.optimized-hlo.txt.gz') for route in ('b1', 'vmap', 'map')]
    observation['destination_checks'] = []
    def destination_exists(path):
        row = dict(path=str(path), operation='exists')
        observation['destination_checks'].append(row)
        row['exists'] = Path(path).exists()
        return row['exists']
    if any(destination_exists(path) for path in destinations):
        raise FileExistsError('paid one-shot already consumed or evidence exists; no retry')
    if args.dry_run and output.with_suffix('.job-attempt.json').exists():
        raise FileExistsError('paid scheduler attempt already consumed; no retry')
    if args.dry_run and (output.parent / 'slurm-attempt.v1.json').exists():
        raise FileExistsError('paid Slurm launch already consumed; no retry')
    shell = ROOT / 'scripts/hpc/run_shared_latent_T7_batch_numerics_paid.sh'
    gpu_python = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python'
    child = lambda name, argv: ['bash', str(shell), '--runtime-child', gpu_python, str(ROOT / 'scripts/hpc' / name), *argv]
    plan = dict(scope=scope, identities=identities, source_files=source_manifest(observation=observation),
        checkout=str(ROOT), output=str(output), ledger=str(args.ledger.resolve()),
        scope_path=str(args.scope_json.resolve()), shell_started_epoch=args.shell_started_epoch,
        preflight_output=str(output.with_suffix('.runtime-preflight.json')),
        planned_measurements=12, planned_arrays=91, planned_small_compiles=4,
        cotangent='fixed diagnostic cotangent; not original T7 loss',
        captured_execution_head=capture['current_execution']['git_head'], GPU_executed=False,
        readiness='paid CPU preparation only; actual allocation/cgroup/container/PJRT/GPU unverified',
        management_admission=MANAGEMENT,
        discovery_output=str(output.with_suffix('.discovery.json')),
        commands={'discovery': ['bash', str(shell), '--runtime-discovery', gpu_python,
                      str(ROOT/'scripts/hpc/shared_affine_cuda_identity.py'), '--discover-output', str(output.with_suffix('.discovery.json'))],
                  'preflight': child('check_daily_training_runtime.py', ['--backend', 'gpu', '--output', str(output.with_suffix('.runtime-preflight.json'))]),
                  'probe': child('probe_shared_latent_T7_batch_numerics.py', ['--backend', 'gpu', '--operands', str(args.operands.resolve()), '--output', str(output), '--repeats', '3', '--seed', '20261011'])})
    observation.update(actual_source_files=plan['source_files'], expected_caps=free.CAPS,
                       child_commands=plan['commands'])
    plan['paid_startup_observation'] = observation
    if scope.get('source_files') != plan['source_files']:
        raise ValueError('paid source manifest differs from actual execution sources')
    actual_head = None
    if scope.get('code_git_head') is not None or not args.dry_run:
        actual_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, timeout=4, text=True).strip()
        observation['actual_git_head'] = actual_head
        if actual_head != scope.get('code_git_head'):
            raise ValueError('paid exact accepted commit binding differs')
    if args.dry_run:
        return plan
    if (os.name != 'posix' or scope.get('execution_authorized') is not True
            or scope.get('status') != 'open' or not str(scope.get('approval_note', '')).strip()):
        raise ValueError('paid live execution requires specific final approval; proposed scope is not runnable')
    if args.output.resolve() != OUTPUT or args.ledger.resolve() != LEDGER or args.scope_json.resolve() != OUTPUT_DIR / 'scope.v1.json':
        raise ValueError('paid live paths differ from one unique independent scope')
    for path in (ROOT, args.operands.resolve(), args.capture_receipt.resolve(), args.capture_source.resolve(), output, args.scope_json.resolve(), args.ledger.resolve()):
        path.relative_to(free.REMOTE)
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, timeout=4, text=True)
    observation['git_status_porcelain'] = dirty
    if dirty.strip():
        raise ValueError('paid checkout must be clean and immutable')
    observation['stage'] = 'slurm_allocation'
    job_result = _record_command([SLURM, 'show', 'job', '--details', '-o', os.environ.get('SLURM_JOB_ID', 'missing')], observation)
    job = parse_job(job_result.stdout)
    plan['slurm_allocation'] = validate_allocation(scope, os.environ, job, socket.gethostname(), observation=observation)
    observation['slurm_allocation'] = plan['slurm_allocation']
    if args.shell_started_epoch != plan['slurm_allocation']['allocation_start_epoch']:
        raise ValueError('outer shell deadline differs from scheduler allocation start')
    observation['stage'] = 'scheduler_attempt_binding'
    launch = json.loads(_read_text(OUTPUT_DIR / 'slurm-attempt.v1.json', observation, 'slurm_attempt'))
    if (launch.get('scope_id') != SCOPE_ID or launch.get('job_id') != os.environ['SLURM_JOB_ID']
            or launch.get('status') != 'scheduler_started' or launch.get('retry_allowed') is not False):
        raise ValueError('one-shot Slurm launch receipt differs from actual job')
    plan['scheduler_launch_receipt'] = launch
    plan['shell_started_epoch'] = plan['slurm_allocation']['allocation_start_epoch']
    return plan


def _kernel_stat(path, observation):
    row = dict(path=str(path), operation='stat')
    observation['files'].append(row)
    try:
        node = Path(path).stat()
        row.update(status='stat', mode=node.st_mode, is_character_device=stat.S_ISCHR(node.st_mode),
                   rdev=node.st_rdev, major=os.major(node.st_rdev), minor=os.minor(node.st_rdev))
        return node
    except OSError as error:
        row.update(status='failed', error_type=type(error).__name__, error=str(error), errno=error.errno)
        raise PaidStartupError(str(error), observation, error) from error


class PaidNvidiaMonitor(free.NvidiaMonitor):
    """Retain the existing monitor calls; no additional GPU observation."""
    def __init__(self, *args, observation, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation = observation
        self.select_by_uuid = True

    def command(self, argv):
        remaining = self.deadline - self.clock()
        row = dict(command=argv, remaining_seconds=remaining, status='started')
        self.observation.setdefault('monitor_commands', []).append(row)
        if remaining <= 0:
            row['status'] = 'deadline_exhausted'
            raise TimeoutError('total GPU occupation deadline reached')
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=min(4., remaining))
        except (OSError, subprocess.SubprocessError) as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error),
                **(subprocess_failure_detail(error) or {}))
            self.retain_stdout(row, argv, row.get('stdout'))
            raise
        row.update(status='completed', returncode=result.returncode, stderr=result.stderr)
        self.retain_stdout(row, argv, result.stdout)
        if result.returncode:
            raise RuntimeError('nvidia-smi failed: ' + result.stderr.strip())
        return result.stdout

    def retain_stdout(self, row, argv, stdout):
        if any(argument.startswith('--query-compute-apps=') for argument in argv):
            # Existing sample/gates still receive the complete response. The new
            # evidence container does not duplicate foreign PIDs/process names.
            row.update(stdout=None, stdout_retention='process identifiers redacted',
                process_rows=[dict(field_count=len(fields),
                    gpu_uuid_matches_expected=bool(fields and fields[0].strip() == self.uuid))
                    for fields in csv.reader((_diagnostic_text(stdout) or '').strip().splitlines())])
        else:
            row['stdout'] = stdout

class IdentityPendingMonitor:
    def sample(self):
        raise ValueError('discovery UUID unknown; live GPU release cannot be verified')


class PaidSupervisor(free.Supervisor):
    """One existing reservation owns discovery, preflight, probe and release."""
    def __init__(self, plan, **kwargs):
        super().__init__(plan, monitor=IdentityPendingMonitor(), **kwargs)
        self.startup_preflight_deadline = min(self.work_deadline,
            self.started + max(0., plan['scope']['caps']['preflight_seconds'] - self.shell_elapsed))

    def phase(self, name, seconds, **kwargs):
        if name in ('discovery', 'preflight'):
            seconds = min(seconds, self.startup_preflight_deadline-self.clock())
            if seconds <= 0:
                return 'deadline_exceeded', 124
        return super().phase(name, seconds, **kwargs)

    def startup(self):
        startup = self.plan['paid_startup_observation']
        visible = os.environ['ORCJAX_SLURM_VISIBLE_DEVICES']
        startup['stage'] = 'owned_runtime_discovery'
        result = self.phase('discovery', self.plan['scope']['caps']['preflight_seconds'],
            environment={'CUDA_VISIBLE_DEVICES': visible, 'ORCJAX_AFFINE_EXPECTED_GPU_UUID': None,
                'ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT': GPU_EXECUTION_CONTRACT}, monitor=False)
        # Preserve completed child failures. A valid measured runtime UUID may
        # select cleanup observations, but is not permission to run science.
        try:
            discovery = self.read_report(self.plan['discovery_output'], 'discovery_result_admission')
        except (OSError, ValueError):
            if result[1]:
                return result
            raise
        startup['discovery'] = discovery
        identity = discovery.get('identity', {})
        observed_uuid = discovery.get('driver_identity', {}).get('driver_gpu_uuid')
        if cuda_identity.valid_uuid(observed_uuid):
            self.monitor = PaidNvidiaMonitor(None, observed_uuid, self.deadline, self.clock, observation=startup)
        if result[1]:
            return result
        runtime = discovery.get('runtime_identity', {})
        driver = discovery.get('driver_identity', {})
        if (discovery.get('status') != 'passed' or identity.get('passed') is not True
                or identity.get('original_CUDA_VISIBLE_DEVICES') != visible
                or identity.get('CUDA_DEVICE_ORDER') != os.environ.get('CUDA_DEVICE_ORDER')
                or identity.get('slurm_gpu_id') != os.environ.get('SLURM_JOB_GPUS')
                or identity.get('allocation_association') != 'trusted_scheduler_original_CVD_same_runtime_namespace'
                or runtime.get('runtime_device_count') != 1 or runtime.get('runtime_visible_ordinal') != 0
                or not cuda_identity.valid_uuid(observed_uuid) or identity.get('gpu_uuid') != observed_uuid
                or driver.get('driver_device_count', 0) < 1
                or str(driver.get('driver_gpu_uuid', '')).lower() != observed_uuid.lower()
                or cuda_identity.pci_address(driver.get('driver_pci_bus_id')) != cuda_identity.pci_address(runtime.get('runtime_pci_bus_id'))):
            raise ValueError('runtime discovery identity/environment association unknown or conflicting')
        startup['stage'] = 'GPU_UUID_CSV_XML'
        try:
            row, observation = observe_gpu_mapping(os.environ, observed_uuid,
                pci_bus_id=runtime['runtime_pci_bus_id'], deadline=self.startup_preflight_deadline, clock=self.clock)
        except GpuObservationError as error:
            startup['gpu_observation'] = error.observation
            raise
        startup['gpu_observation'] = observation
        row.update(discovery_gpu_uuid=observed_uuid,
            allocation_association=identity['allocation_association'])
        startup['stage'] = 'kernel_device_mapping'
        device = _kernel_stat(Path('/dev/nvidia' + str(row['minor_number'])), startup)
        if not stat.S_ISCHR(device.st_mode):
            raise ValueError('XML-selected GPU kernel node is not a character device')
        row.update(device_major=os.major(device.st_rdev), device_minor=os.minor(device.st_rdev), control_devices=[])
        for name in ('/dev/nvidiactl', '/dev/nvidia-modeset'):
            node = Path(name)
            exists = node.exists()
            startup['files'].append(dict(path=name, operation='exists', exists=exists))
            if exists:
                control = _kernel_stat(node, startup)
                if not stat.S_ISCHR(control.st_mode):
                    raise ValueError('GPU driver control path is not a character device')
                row['control_devices'].append(dict(name=name, major=os.major(control.st_rdev), minor=os.minor(control.st_rdev)))
        startup['stage'] = 'job_device_cgroup'
        cgroup_path, rules = read_cgroup_rules(self.plan['slurm_allocation']['job']['JobId'], observation=startup)
        startup['stage'] = 'cgroup_access_assessment_and_GPU_binding'
        mapping = admit_gpu_mapping(os.environ, [row], rules, observation=startup, contract=GPU_EXECUTION_CONTRACT)
        mapping['cgroup_devices_path'] = cgroup_path
        self.plan.update(slurm_gpu_mapping=mapping, slurm_gpu_observation=observation,
            phase_environment={'ORCJAX_AFFINE_EXPECTED_GPU_UUID': mapping['gpu_uuid'],
                'ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT': GPU_EXECUTION_CONTRACT})
        self.plan['scope'].update(gpu_host_index=mapping['gpu_host_index'], gpu_uuid=mapping['gpu_uuid'])
        self.monitor = PaidNvidiaMonitor(mapping['gpu_host_index'], mapping['gpu_uuid'], self.deadline, self.clock, observation=startup)
        self.record.update(paid_gpu_mapping=mapping, paid_gpu_observation=observation)
        startup['stage'] = 'discovery_admitted'
        return None


def run_live(plan):
    # Cooperative exclusion and bounded read-only management admission precede
    # GPU observation/ledger/preflight. They do not constrain old b676 copies.
    observation = plan.setdefault('paid_startup_observation', startup_observation(os.environ))
    observation.update(stage='mutex', mutex_path=str(SCOPE_LOCK), management_contract=MANAGEMENT)
    lock = AffineScopeLock(SCOPE_LOCK, SCOPE_ID).acquire()
    observation['mutex_acquired'] = True
    supervisor = None
    release_safe = False
    try:
        observation['stage'] = 'free_management'
        plan['free_management_state'] = read_free_admission(observation=observation)
        observation['free_management_state'] = plan['free_management_state']
        visible = os.environ.get('ORCJAX_SLURM_VISIBLE_DEVICES', '')
        if (not (visible.isdigit() or cuda_identity.valid_uuid(visible))
                or visible != observation.get('environment', {}).get('ORCJAX_SLURM_VISIBLE_DEVICES')):
            raise ValueError('single original scheduler CVD required; physical_allocation_mapping_unknown')
        plan['scope'] = dict(plan['scope'], gpu_host_index=None, gpu_uuid='')
        observation['stage'] = 'supervisor'
        # Use the same clock/deadline origin as Supervisor's existing monitor.
        supervisor = PaidSupervisor(plan)
        observation['supervisor_limits'] = dict(total_deadline=supervisor.deadline,
            work_deadline=getattr(supervisor, 'work_deadline', None),
            shell_elapsed_seconds=getattr(supervisor, 'shell_elapsed', None), caps=plan['scope'].get('caps'))
        supervisor.record['paid_allocation'] = plan['slurm_allocation']
        supervisor.record['actual_cost_upper_bound_cny'] = quote()['total_cny']
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda number, frame: setattr(supervisor, 'interrupted', number))
        result = supervisor.run()
        observation['stage'] = 'supervisor_finished'
        supervisor.record['numerical_status'] = ('executed' if any(
            phase['name'] == 'probe' for phase in supervisor.record.get('phases', [])) else 'not_executed')
        supervisor.snapshot()
        release_safe = supervisor.record.get('release_verified') is True
        return result
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        startup = plan['paid_startup_observation']
        if isinstance(error, GpuObservationError):
            error.startup_observation = startup
            raise
        if isinstance(error, PaidStartupError):
            raise
        raise PaidStartupError(str(error), startup, error) from error
    finally:
        # Admission before any GPU child safely releases mutual exclusion.
        # Any uncertain child cleanup retains it persistently for manual audit.
        if supervisor is None:
            release_safe = True
        plan['paid_startup_observation']['mutex_release_verified'] = release_safe
        plan['paid_startup_observation']['mutex_released'] = lock.release(release_verified=release_safe)

def _diagnostic_text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)

def subprocess_failure_detail(error):
    if isinstance(error, subprocess.CalledProcessError):
        return dict(command=error.cmd, returncode=error.returncode,
            stdout=_diagnostic_text(error.stdout if error.stdout is not None else error.output),
            stderr=_diagnostic_text(error.stderr))
    if isinstance(error, subprocess.TimeoutExpired):
        return dict(command=error.cmd, timeout_seconds=error.timeout,
            stdout=_diagnostic_text(error.stdout if error.stdout is not None else error.output),
            stderr=_diagnostic_text(error.stderr))
    return None

def main(argv=None):
    args = free.parser().parse_args(argv)
    observation = startup_observation(os.environ)
    args._paid_startup_observation = observation
    attempt = None
    try:
        if not args.dry_run:
            # A paid scheduler launch is consumed even if CPU admission fails.
            # Never silently resubmit after a rejected allocation or bootstrap.
            scope = json.loads(_read_text(args.scope_json, observation, 'authorization_scope'))
            observation['scope'] = scope
            if (os.name != 'posix' or not os.environ.get('SLURM_JOB_ID')
                    or args.scope_json.resolve() != OUTPUT_DIR / 'scope.v1.json'
                    or args.output.resolve() != OUTPUT or args.ledger.resolve() != LEDGER
                    or scope.get('schema_version') != SCHEMA or scope.get('scope_id') != SCOPE_ID
                    or scope.get('gpu_execution_contract') != GPU_EXECUTION_CONTRACT
                    or scope.get('os_device_open_isolation_required') is not False
                    or scope.get('status') != 'open' or scope.get('execution_authorized') is not True
                    or not str(scope.get('approval_note', '')).strip()):
                raise ValueError('specific approved paid scope and Slurm launch required')
            attempt = OUTPUT.with_suffix('.job-attempt.json')
            observation.update(stage='consume_attempt', attempt_path=str(attempt))
            with attempt.open('x', encoding='utf-8') as stream:
                json.dump(dict(scope_id=SCOPE_ID, status='scheduler_attempt_consumed',
                    job_id=os.environ['SLURM_JOB_ID'], recorded_epoch=time.time(),
                    retry_allowed=False, GPU_executed=False), stream, indent=2)
        plan = prepare(args)
        plan['paid_startup_observation'] = observation
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        return run_live(plan)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        cause = error.cause if isinstance(error, (GpuObservationError, PaidStartupError)) and error.cause is not None else error
        rejection = dict(status='paid_admission_rejected', GPU_executed=False,
            scheduler_attempt_receipt=str(attempt) if attempt else None,
            error_type=type(cause).__name__, error=str(cause), paid_startup_observation=observation)
        if isinstance(error, GpuObservationError):
            rejection['gpu_observation'] = error.observation
        subprocess_detail = (error.subprocess_failure if isinstance(error, GpuObservationError)
                             and error.subprocess_failure is not None else subprocess_failure_detail(cause))
        if subprocess_detail is not None:
            rejection['subprocess_failure'] = subprocess_detail
        print(json.dumps(rejection))
        return 2

if __name__ == '__main__':
    if sys.argv[1:] == ['--scheduler-allocation-epochs']:
        try:
            raise SystemExit(scheduler_allocation_epochs())
        except (ValueError, KeyError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(2)
    raise SystemExit(main())
