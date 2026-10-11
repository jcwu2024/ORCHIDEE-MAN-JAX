"""CPU-only admission for exactly one proposed paid Slurm affine diagnostic.

Scientific probe computation is unchanged. Live execution requires approval,
an immutable committed manifest, exact scheduler/cgroup admission, the declared
paid-only launch management chain and a cooperative shared mutex.
"""
from __future__ import annotations
import datetime
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.hpc import run_shared_latent_T7_batch_numerics as free
from scripts.hpc.shared_affine_gpu_admission import (
    AffineScopeLock, SCOPE_LOCK, MANAGEMENT, read_free_admission)

SCOPE_ID = 'T7-summary-affine-paid-gpu-20261011-v2'
CHECKOUT_TARGET = free.REMOTE / 'checkouts/T7-affine-paid-20261011-v2'
SCHEMA = 'T7_affine_paid_slurm_scope_v1'
OUTPUT_DIR = free.REMOTE / 'outputs/diagnostics' / SCOPE_ID
OUTPUT = OUTPUT_DIR / 'result.v1.json'
LEDGER = OUTPUT_DIR / (SCOPE_ID + '.ledger.json')
SLURM = '/rmprog/slurm/v22.05.7/bin/scontrol'
NODE = 'ibc13b03n01'
SOURCE_PATHS = (*free.SOURCE_PATHS,
    'scripts/hpc/run_shared_latent_T7_batch_numerics_paid.py',
    'scripts/hpc/run_shared_latent_T7_batch_numerics_paid.sh',
    'scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh')
RESOURCES = dict(partition='gnall', node=NODE, nodes=1, tasks=1,
                 allocated_slurm_cpus=4, gpus=1, wall_seconds=1200,
                 nonexclusive=True, requeue=False)
RATES = dict(cpu_cny_per_allocated_cpu_hour='0.07', gpu_cny_per_gpu_hour='2.20')

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

def allocation_epochs(environment, job):
    """Scheduler clock; 22.05.7 batch env does not supply JOB_START/END_TIME.

    Original env tokens, when present, are additional exact crosschecks.
    Neither shell/Python startup nor a renewed deadline is an admissible clock.
    """
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
    if not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise ValueError('numeric actual Slurm job ID required')
    result = subprocess.run([SLURM, 'show', 'job', '--details', '-o',
        os.environ.get('SLURM_JOB_ID', 'missing')], capture_output=True,
        text=True, timeout=4, check=True)
    start, end = allocation_epochs(os.environ, parse_job(result.stdout))
    print(start, end)
    return 0


def validate_allocation(scope, environment, job, hostname):
    """No GPU query: exact paid scheduling/charge admission, never CPU fallback."""
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
    start, end = allocation_epochs(environment, job)
    maximum = Decimal(scope.get('approved_cost_ceiling_cny', '0'))
    if maximum != Decimal('1.00') or Decimal(quote()['total_cny']) > maximum:
        raise ValueError('paid exact quote or approved ceiling differs')
    return dict(job=job, allocation_start_epoch=start, allocation_end_epoch=end,
                allocation_clock_source='scheduler StartTime/EndTime; optional original env crosschecks',
                actual_NumCPUs=4, allocation_TRES_fields=allocation_fields,
                actual_AllocTRES=allocated, quote=quote(), hostname=hostname,
                original_visible_devices=environment.get('ORCJAX_SLURM_VISIBLE_DEVICES'),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

def admit_gpu_mapping(environment, gpu_rows, devices_rules):
    """Bind Slurm physical GPU ID, UUID and cgroup minor, separately from CVD.

    The original visible ordinal is retained as evidence, never used as a host
    index. The child selects the admitted UUID inside the Slurm device cgroup.
    Unsupported cgroup-v2 or wildcard GPU access is a failed admission.
    """
    selected = environment.get('SLURM_JOB_GPUS', '')
    visible = environment.get('ORCJAX_SLURM_VISIBLE_DEVICES', '')
    if not selected.isdigit() or not visible or ',' in visible or len(gpu_rows) != 1:
        raise ValueError('single Slurm physical GPU and original visible mapping required')
    row = gpu_rows[0]
    host_index = int(selected)
    minor = int(row['minor_number'])
    device_pair = (int(row['device_major']), int(row['device_minor']))
    if (int(row['index']) != host_index or not str(row['uuid']).startswith('GPU-')
            or 'V100' not in row['name']):
        raise ValueError('Slurm physical GPU/UUID/V100 mapping differs')
    if minor < 0 or device_pair[1] != minor:
        raise ValueError('NVML minor_number and actual kernel GPU device minor differ')
    if visible.startswith('GPU-') and visible != row['uuid']:
        raise ValueError('visible GPU UUID differs')
    if not visible.isdigit() and not visible.startswith('GPU-'):
        raise ValueError('unsupported visible GPU mapping')
    permitted = set()
    controls = {(int(item['major']), int(item['minor'])) for item in row.get('control_devices', [])}
    if device_pair in controls:
        raise ValueError('GPU data device cannot be a driver control device')
    for line in devices_rules.splitlines():
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('invalid cgroup devices rule')
        kind, device, permissions = parts
        major, sep, minor = device.partition(':')
        if (not sep or kind not in ('a', 'b', 'c') or not set(permissions) <= set('rwm')
                or not permissions or any(value != '*' and not value.isdigit() for value in (major, minor))):
            raise ValueError('invalid cgroup device permission rule')
        # m-only wildcard grants mknod, not access to a GPU. Any wildcard
        # read/write permission for the GPU's actual major cannot prove the
        # allocated single-device boundary, including c *:N.
        access = bool(set(permissions) & set('rw'))
        if access and (kind == 'a' or (kind == 'c' and major in ('*', str(device_pair[0])) and (major == '*' or minor == '*'))):
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

def read_cgroup_rules(job_id, *, proc_path=Path('/proc/self/cgroup'), cgroup_base=Path('/sys/fs/cgroup/devices')):
    if not str(job_id).isdigit():
        raise ValueError('numeric actual Slurm job ID required for cgroup binding')
    memberships = []
    for line in proc_path.read_text().splitlines():
        _, controllers, member = line.split(':', 2)
        if 'devices' in controllers.split(','):
            memberships.append(member)
    if len(memberships) != 1:
        raise ValueError('Slurm job device cgroup-v1 membership unverified')
    job_components = [item for item in memberships[0].split('/') if item.startswith('job_')]
    if job_components != ['job_' + str(job_id)]:
        raise ValueError('device cgroup belongs to a different actual Slurm job')
    base = cgroup_base.resolve()
    path = (base / memberships[0].lstrip('/') / 'devices.list').resolve()
    path.relative_to(base)
    return str(path), path.read_text()

def source_manifest():
    return {path: free.small_sha(ROOT / path, 1000000) for path in SOURCE_PATHS}

def proposed_scope():
    """Preparation snapshot. No invented candidate-commit identity."""
    return dict(schema_version=SCHEMA, scope_id=SCOPE_ID, status='proposed',
        execution_authorized=False, approval_note='', code_git_head=None,
        source_files=source_manifest(), caps=free.CAPS, resources=RESOURCES, rates=RATES,
        quote=quote(), approved_cost_ceiling_cny='1.00', repeats=3,
        cards=1, nonexclusive=True, output=OUTPUT.as_posix(), ledger=LEDGER.as_posix(),
        management_admission=MANAGEMENT, shared_mutex=SCOPE_LOCK.as_posix(),
        input_sha256=dict(operand=free.OPERAND_SHA, receipt=free.RECEIPT_SHA,
                          capture_source=free.CAPTURE_SHA),
        identity_status='candidate source hashes only; code_git_head binds real accepted commit after versioning')

def prepare(args):
    identities, capture = free.admit_capture(args)
    scope = json.loads(args.scope_json.read_text(encoding='utf-8'))
    if (scope.get('schema_version') != SCHEMA or scope.get('scope_id') != SCOPE_ID
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
    destinations += [output.with_suffix('.' + route + '.optimized-hlo.txt.gz') for route in ('b1', 'vmap', 'map')]
    if any(Path(path).exists() for path in destinations):
        raise FileExistsError('paid one-shot already consumed or evidence exists; no retry')
    if args.dry_run and output.with_suffix('.job-attempt.json').exists():
        raise FileExistsError('paid scheduler attempt already consumed; no retry')
    if args.dry_run and (output.parent / 'slurm-attempt.v1.json').exists():
        raise FileExistsError('paid Slurm launch already consumed; no retry')
    shell = ROOT / 'scripts/hpc/run_shared_latent_T7_batch_numerics_paid.sh'
    gpu_python = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python'
    child = lambda name, argv: ['bash', str(shell), '--runtime-child', gpu_python, str(ROOT / 'scripts/hpc' / name), *argv]
    plan = dict(scope=scope, identities=identities, source_files=source_manifest(),
        checkout=str(ROOT), output=str(output), ledger=str(args.ledger.resolve()),
        scope_path=str(args.scope_json.resolve()), shell_started_epoch=args.shell_started_epoch,
        preflight_output=str(output.with_suffix('.runtime-preflight.json')),
        planned_measurements=12, planned_arrays=91, planned_small_compiles=4,
        cotangent='fixed diagnostic cotangent; not original T7 loss',
        captured_execution_head=capture['current_execution']['git_head'], GPU_executed=False,
        readiness='paid CPU preparation only; actual allocation/cgroup/container/PJRT/GPU unverified',
        management_admission=MANAGEMENT,
        commands={'preflight': child('check_daily_training_runtime.py', ['--backend', 'gpu', '--output', str(output.with_suffix('.runtime-preflight.json'))]),
                  'probe': child('probe_shared_latent_T7_batch_numerics.py', ['--backend', 'gpu', '--operands', str(args.operands.resolve()), '--output', str(output), '--repeats', '3', '--seed', '20261011'])})
    if scope.get('source_files') != plan['source_files']:
        raise ValueError('paid source manifest differs from actual execution sources')
    actual_head = None
    if scope.get('code_git_head') is not None or not args.dry_run:
        actual_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, timeout=4, text=True).strip()
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
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, timeout=4, text=True).strip():
        raise ValueError('paid checkout must be clean and immutable')
    job_result = subprocess.run([SLURM, 'show', 'job', '--details', '-o', os.environ.get('SLURM_JOB_ID', 'missing')], capture_output=True, text=True, timeout=4, check=True)
    job = parse_job(job_result.stdout)
    plan['slurm_allocation'] = validate_allocation(scope, os.environ, job, socket.gethostname())
    if args.shell_started_epoch != plan['slurm_allocation']['allocation_start_epoch']:
        raise ValueError('outer shell deadline differs from scheduler allocation start')
    launch = json.loads((OUTPUT_DIR / 'slurm-attempt.v1.json').read_text(encoding='utf-8'))
    if (launch.get('scope_id') != SCOPE_ID or launch.get('job_id') != os.environ['SLURM_JOB_ID']
            or launch.get('status') != 'scheduler_started' or launch.get('retry_allowed') is not False):
        raise ValueError('one-shot Slurm launch receipt differs from actual job')
    plan['scheduler_launch_receipt'] = launch
    plan['shell_started_epoch'] = plan['slurm_allocation']['allocation_start_epoch']
    return plan

def run_live(plan):
    # Cooperative exclusion and bounded read-only management admission precede
    # GPU observation/ledger/preflight. They do not constrain old b676 copies.
    lock = AffineScopeLock(SCOPE_LOCK, SCOPE_ID).acquire()
    supervisor = None
    release_safe = False
    try:
        plan['free_management_state'] = read_free_admission()
        index = os.environ.get('SLURM_JOB_GPUS', '')
        if not index.isdigit():
            raise ValueError('single physical SLURM_JOB_GPUS required before observation')
        observed = subprocess.run(['nvidia-smi', '-i', index, '--query-gpu=index,uuid,name,minor_number', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4, check=True)
        import csv
        rows = [dict(zip(('index', 'uuid', 'name', 'minor_number'), [part.strip() for part in row], strict=True)) for row in csv.reader(observed.stdout.strip().splitlines())]
        for row in rows:
            device = Path('/dev/nvidia' + str(int(row['minor_number']))).stat()
            if not stat.S_ISCHR(device.st_mode):
                raise ValueError('NVML-selected GPU kernel node is not a character device')
            row.update(device_major=os.major(device.st_rdev), device_minor=os.minor(device.st_rdev), control_devices=[])
            for name in ('/dev/nvidiactl', '/dev/nvidia-modeset'):
                node = Path(name)
                if node.exists():
                    control = node.stat()
                    if not stat.S_ISCHR(control.st_mode):
                        raise ValueError('GPU driver control path is not a character device')
                    row['control_devices'].append(dict(name=name, major=os.major(control.st_rdev), minor=os.minor(control.st_rdev)))
        cgroup_path, rules = read_cgroup_rules(plan['slurm_allocation']['job']['JobId'])
        mapping = admit_gpu_mapping(os.environ, rows, rules)
        mapping['cgroup_devices_path'] = cgroup_path
        plan['slurm_gpu_mapping'] = mapping
        plan['phase_environment'] = {'ORCJAX_AFFINE_EXPECTED_GPU_UUID': mapping['gpu_uuid']}
        plan['scope'] = dict(plan['scope'], gpu_host_index=mapping['gpu_host_index'], gpu_uuid=mapping['gpu_uuid'])
        supervisor = free.Supervisor(plan)
        supervisor.record['paid_allocation'] = plan['slurm_allocation']
        supervisor.record['paid_gpu_mapping'] = mapping
        supervisor.record['actual_cost_upper_bound_cny'] = quote()['total_cny']
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda number, frame: setattr(supervisor, 'interrupted', number))
        result = supervisor.run()
        release_safe = supervisor.record.get('release_verified') is True
        return result
    finally:
        # Admission before any GPU child safely releases mutual exclusion.
        # Any uncertain child cleanup retains it persistently for manual audit.
        if supervisor is None:
            release_safe = True
        lock.release(release_verified=release_safe)

def main(argv=None):
    args = free.parser().parse_args(argv)
    attempt = None
    try:
        if not args.dry_run:
            # A paid scheduler launch is consumed even if CPU admission fails.
            # Never silently resubmit after a rejected allocation or bootstrap.
            scope = json.loads(args.scope_json.read_text(encoding='utf-8'))
            if (os.name != 'posix' or not os.environ.get('SLURM_JOB_ID')
                    or args.scope_json.resolve() != OUTPUT_DIR / 'scope.v1.json'
                    or args.output.resolve() != OUTPUT or args.ledger.resolve() != LEDGER
                    or scope.get('schema_version') != SCHEMA or scope.get('scope_id') != SCOPE_ID
                    or scope.get('status') != 'open' or scope.get('execution_authorized') is not True
                    or not str(scope.get('approval_note', '')).strip()):
                raise ValueError('specific approved paid scope and Slurm launch required')
            attempt = OUTPUT.with_suffix('.job-attempt.json')
            with attempt.open('x', encoding='utf-8') as stream:
                json.dump(dict(scope_id=SCOPE_ID, status='scheduler_attempt_consumed',
                    job_id=os.environ['SLURM_JOB_ID'], recorded_epoch=time.time(),
                    retry_allowed=False, GPU_executed=False), stream, indent=2)
        plan = prepare(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        return run_live(plan)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(status='paid_admission_rejected', GPU_executed=False,
                              scheduler_attempt_receipt=str(attempt) if attempt else None,
                              error_type=type(error).__name__, error=str(error))))
        return 2

if __name__ == '__main__':
    if sys.argv[1:] == ['--scheduler-allocation-epochs']:
        try:
            raise SystemExit(scheduler_allocation_epochs())
        except (ValueError, KeyError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(2)
    raise SystemExit(main())
