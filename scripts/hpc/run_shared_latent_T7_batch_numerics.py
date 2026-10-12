"""One-shot supervisor for the admitted early affine diagnostic, never T7.

No JAX import here. Container startup is a supervised child, not a prerequisite
outside the charged interval. The old shared_gpu_budget ledger is never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.hpc import probe_shared_latent_T7_batch_numerics as probe
from scripts.hpc.shared_affine_gpu_admission import AffineScopeLock, SCOPE_LOCK

REMOTE = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime')
SCOPE_ID = 'T7-summary-affine-gpu-20261011-v1'
SCHEMA = 'T7_affine_gpu_scope_v1'
OPERAND_SHA = 'bf36a098b4e6a89d555f8d18521555f2da66135425d9435cfe01878adeec6129'
RECEIPT_SHA = 'f863a2d964f626cec327981d86a8832e73aa45af62d708a342805c4aa4220e00'
CAPTURE_SHA = '70619b0a6445cde87eccaeb31feac46bbb34cdc0d7a6f68c68938c4d09901072'
HISTORICAL_PROBE_SHA = 'c175841fcf9131122adebebc683cf33ca6f980e145cdecfdff0cd77e1bf2e2e8'
CAPTURE_PROBE_SHA = '05e2ee66b1905fd238722d3a4cf645e74639f7134de4af5a700cf0b657305ee0'
CAPS = dict(total_seconds=1200, preflight_seconds=240, probe_seconds=780,
            release_reserve_seconds=180, memory_limit_mib=24576)
SOURCE_PATHS = ('scripts/hpc/run_shared_latent_T7_batch_numerics.py',
    'scripts/hpc/run_shared_latent_T7_batch_numerics.sh',
    'scripts/hpc/probe_shared_latent_T7_batch_numerics.py',
    'scripts/hpc/orcjax_gpu_runtime.py', 'scripts/hpc/check_daily_training_runtime.py',
    'scripts/hpc/capture_shared_latent_T7_summary_operands.py',
    'scripts/hpc/shared_affine_gpu_admission.py',
    'scripts/hpc/shared_affine_cuda_identity.py',
    'research/daily_coarse_graining/daily_operator/strong_baseline_architecture.py')


def atomic_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def small_sha(path, maximum, *, observation=None):
    path = Path(path).resolve()
    row = dict(path=str(path), operation='bounded_sha256', maximum_bytes=maximum)
    if observation is not None:
        observation.setdefault('files', []).append(row)
    try:
        row['is_file'] = path.is_file()
        if row['is_file']:
            row['size_bytes'] = path.stat().st_size
        if not row['is_file'] or row['size_bytes'] > maximum:
            raise ValueError('missing or oversized bounded input: ' + str(path))
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        row.update(error_type=type(error).__name__, error=str(error), errno=error.errno)
        raise
    return row['sha256']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkout-root', 'operands', 'capture-receipt', 'capture-source', 'output', 'scope-json', 'ledger'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--shell-started-epoch', type=float)
    p.add_argument('--dry-run', action='store_true')
    return p


def validate_source_manifest(scope, source_files):
    if set(source_files) != set(SOURCE_PATHS) or scope.get('source_files') != source_files:
        raise ValueError('deployed execution sources differ from frozen scope manifest')


def validate_live_host(hostname, platform):
    if platform != 'posix' or hostname.split('.')[0] not in ('gln01', 'ibc13b02n04'):
        raise ValueError('live execution requires the gln01 GPU test node on POSIX')


def admit_capture(args):
    """The shared, unchanged thin-capture identity and numeric-byte admission."""
    code = args.checkout_root.resolve()
    if code != ROOT or not code.is_dir():
        raise ValueError('launcher must resolve from its own declared checkout')
    observation = getattr(args, '_paid_startup_observation', None)
    identities = {'operand': small_sha(args.operands, 100000, observation=observation),
                  'receipt': small_sha(args.capture_receipt, 100000, observation=observation),
                  'capture_source': small_sha(args.capture_source, 100000, observation=observation)}
    if observation is not None:
        observation['capture_file_identities'] = identities
    if identities != {'operand': OPERAND_SHA, 'receipt': RECEIPT_SHA, 'capture_source': CAPTURE_SHA}:
        raise ValueError('original NPZ/receipt/capture-source identity differs')
    capture_text = args.capture_receipt.read_text(encoding='utf-8')
    if observation is not None:
        observation['capture_receipt_text'] = capture_text
    capture = json.loads(capture_text)
    bundle = probe.load_operands(args.operands)
    meta = bundle['metadata']
    if observation is not None:
        observation['capture_boundary'] = dict(metadata=meta,
            x_shape=list(bundle['x'].shape), kernel_shape=list(bundle['kernel'].shape), array_identities={})
    if (meta['origin'] != 'captured_component_operands' or meta['capture_receipt_sha256'] != RECEIPT_SHA
            or meta['capture_source_sha256'] != CAPTURE_SHA or meta['direction_coordinate'] != 434
            or meta['direction_sample'] != 3 or meta['window_index'] != 0 or meta['day_index'] != 0
            or bundle['x'].shape != (4, 257, 4) or bundle['kernel'].shape != (4, 384)
            or capture['status'] != 'operands_captured'
            or capture['array_identities']['x']['shape'] != [4, 257, 4]
            or capture['current_execution']['source_files']['scripts/hpc/probe_shared_latent_T7_batch_numerics.py']['sha256'] != CAPTURE_PROBE_SHA):
        raise ValueError('fixed early recovery capture boundary or capture probe binding differs')
    # Historical source files authenticate the capture, not current bootstrap
    # code. Preserve that receipt unchanged and record current sources separately.
    for key in probe.ARRAY_KEYS:
        value = bundle[key]
        identity = {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'sha256_bytes_C_order': hashlib.sha256(value.tobytes(order='C')).hexdigest()}
        if observation is not None:
            observation['capture_boundary']['array_identities'][key] = identity
        if identity != capture['array_identities'][key]:
            raise ValueError('numeric operand/receipt binding differs: ' + key)
    return identities, capture


def prepare(args):
    """Identity and argv admission only; does not initialize any GPU or model."""
    code = args.checkout_root.resolve()
    identities, capture = admit_capture(args)
    small_sha(args.scope_json, 20000)
    scope = json.loads(args.scope_json.read_text(encoding='utf-8'))
    if (scope.get('schema_version') != SCHEMA or scope.get('scope_id') != SCOPE_ID
            or scope.get('caps') != CAPS or scope.get('node') != 'gln01'
            or scope.get('cards') != 1 or scope.get('nonexclusive') is not True
            or scope.get('worst_case_cost_cny') != 0 or scope.get('repeats') != 3
            or scope.get('status') not in ('proposed', 'open')
            or scope.get('gpu_host_index') not in (0, 1)
            or not str(scope.get('gpu_uuid', '')).startswith('GPU-')):
        raise ValueError('scope differs or is closed; old grants are not admissible')
    if args.ledger.resolve() != Path(scope['ledger']).resolve() or args.output.resolve() != Path(scope['output']).resolve():
        raise ValueError('scope output/one-shot ledger paths differ')
    if args.ledger.exists():
        raise ValueError('one-shot scope already reserved or closed; no retry')
    output = args.output.resolve()
    destinations = [output, output.with_suffix('.results.npz'), output.with_suffix('.partial.npz'),
                    output.with_suffix('.supervisor.json'), output.with_suffix('.runtime-preflight.json'),
                    output.with_suffix('.outer-timeout.json')]
    destinations += [output.with_suffix('.' + name + '.optimized-hlo.txt.gz') for name in ('b1', 'vmap', 'map')]
    if any(path.exists() for path in destinations):
        raise FileExistsError('use the new authorized output only; preserve existing evidence')
    shell = code / 'scripts/hpc/run_shared_latent_T7_batch_numerics.sh'
    gpu_python = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python'
    child = lambda entry, argv: ['bash', str(shell), '--runtime-child', gpu_python, str(code / 'scripts/hpc' / entry), *argv]
    preflight_output = output.with_suffix('.runtime-preflight.json')
    commands = {
        'preflight': child('check_daily_training_runtime.py', ['--backend', 'gpu', '--output', str(preflight_output)]),
        'probe': child('probe_shared_latent_T7_batch_numerics.py', ['--backend', 'gpu', '--operands', str(args.operands.resolve()), '--output', str(output), '--repeats', '3', '--seed', '20261011'])}
    sources = [code / relative for relative in SOURCE_PATHS]
    plan = dict(scope=scope, identities=identities, commands=commands, output=str(output),
        ledger=str(args.ledger.resolve()), checkout=str(code), scope_path=str(args.scope_json.resolve()),
        source_files={str(path.relative_to(code).as_posix()): small_sha(path, 1000000) for path in sources},
        shell_started_epoch=args.shell_started_epoch, preflight_output=str(preflight_output),
        historical_probe_sha256=HISTORICAL_PROBE_SHA, captured_execution_head=capture['current_execution']['git_head'],
        recovery_capture_probe_sha256=CAPTURE_PROBE_SHA,
        input_identity='fresh recovery capture; six numeric arrays equal preincident C-order hashes; original files not restored',
        GPU_executed=False, planned_measurements=12, planned_arrays=91, planned_small_compiles=4,
        cotangent='fixed diagnostic cotangent; not original T7 loss',
        readiness='local preparation only; target shell/container/bootstrap/preflight not verified')
    if not args.dry_run:
        validate_live_host(socket.gethostname(), os.name)
        if (scope.get('status') != 'open' or scope.get('execution_authorized') is not True
                or not str(scope.get('approval_note', '')).strip()):
            raise ValueError('live execution requires a separately approved open scope on gln01/POSIX')
        for path in (code, output, args.ledger.resolve(), args.scope_json.resolve(), args.operands.resolve(), args.capture_receipt.resolve(), args.capture_source.resolve()):
            path.relative_to(REMOTE)
        expected_ledger = REMOTE / 'outputs/diagnostics/T7-summary-affine-gpu-20261011' / (SCOPE_ID + '.ledger.json')
        if args.ledger.resolve() != expected_ledger:
            raise ValueError('independent scope has one fixed ledger; old/alternate balances forbidden')
        validate_source_manifest(scope, plan['source_files'])
    return plan


class PosixGroups:
    """Only groups created by this supervisor are ever sent a signal."""
    @staticmethod
    def alive(group):
        try:
            os.killpg(group, 0)
            return True
        except ProcessLookupError:
            return False

    @staticmethod
    def kill(group, signum):
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass

    @staticmethod
    def owner(pid, groups):
        try:
            return os.getpgid(pid) in groups
        except ProcessLookupError:
            return False


class NvidiaMonitor:
    """Host UUID-based observations. Every nvidia-smi call has a timeout."""
    def __init__(self, host_index, uuid, deadline, clock=time.monotonic):
        self.index, self.uuid, self.deadline, self.clock = host_index, uuid, deadline, clock

    def command(self, argv):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError('total GPU occupation deadline reached')
        result = subprocess.run(argv, capture_output=True, text=True, timeout=min(4., remaining))
        if result.returncode:
            raise RuntimeError('nvidia-smi failed: ' + result.stderr.strip())
        return result.stdout

    def sample(self):
        gpu = self.command(['nvidia-smi', '-i', str(self.index), '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
        rows = list(csv.reader(gpu.strip().splitlines()))
        if len(rows) != 1 or len(rows[0]) != 5:
            raise ValueError('ambiguous host GPU observation')
        index, uuid, name, memory, utilization = [value.strip() for value in rows[0]]
        if int(index) != self.index or uuid != self.uuid or 'V100' not in name:
            raise ValueError('host index/UUID/V100 identity differs')
        apps = self.command(['nvidia-smi', '-i', uuid, '--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name', '--format=csv,noheader,nounits'])
        processes = []
        for row in csv.reader(apps.strip().splitlines()):
            if len(row) != 4 or row[0].strip() != uuid:
                raise ValueError('unknown or differently selected GPU process')
            processes.append(dict(pid=int(row[1]), memory_mib=int(row[2]), name=row[3].strip()))
        return dict(epoch=time.time(), host_index=int(index), gpu_uuid=uuid, name=name,
                    memory_mib=int(memory), utilization_percent=int(utilization), processes=processes)


class Supervisor:
    def __init__(self, plan, *, monitor=None, clock=time.monotonic, sleeper=time.sleep,
                 popen=subprocess.Popen, groups=None, watchdog=True):
        self.plan, self.clock, self.sleep, self.popen = plan, clock, sleeper, popen
        self.groups = groups or PosixGroups()
        self.started = clock()
        elapsed = max(0., time.time() - plan['shell_started_epoch']) if plan.get('shell_started_epoch') else 0.
        self.shell_elapsed = elapsed
        self.deadline = self.started + plan['scope']['caps']['total_seconds'] - elapsed
        self.work_deadline = self.deadline - plan['scope']['caps']['release_reserve_seconds']
        self.monitor = monitor or NvidiaMonitor(plan['scope']['gpu_host_index'], plan['scope']['gpu_uuid'], self.deadline, clock)
        self.owned = set()
        self.group_history = []
        self.processes = []
        self.done = threading.Event()
        self.watchdog_enabled = watchdog
        self.interrupted = None
        self.record = dict(schema_version='T7_affine_gpu_supervisor_v1', plan=plan, phases=[], samples=[],
            reserved_gpu_seconds=1200, cards=1, started_epoch=time.time() - elapsed,
            status='reserved', primary_status='not_executed', primary_exit_code=None,
            node=socket.gethostname(), cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            host_load=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
            python=sys.executable, environment={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES')},
            inherited_old_budget=False, GPU_executed=False)
        self.out = Path(plan['output']).with_suffix('.supervisor.json')
        self.ledger = Path(plan['ledger'])

    def snapshot(self):
        self.record['peak_memory_mib'] = max((row['memory_mib'] for row in self.record['samples']), default=None)
        atomic_json(self.out, self.record)

    def read_report(self, path, label):
        """Retain already-read startup inputs for the opt-in paid observer."""
        observation = self.plan.get('paid_startup_observation')
        row = dict(label=label, path=str(path), operation='read_text')
        if observation is not None:
            observation['stage'] = label
            observation['files'].append(row)
        try:
            row['text'] = Path(path).read_text(encoding='utf-8')
            row['status'] = 'read'
        except OSError as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error), errno=error.errno)
            raise
        return json.loads(row['text'])

    def watchdog(self):
        if not self.done.wait(max(0., self.deadline - self.clock() - 8.)):
            self.interrupted = 'hard_deadline'
            for group in tuple(self.owned):
                self.groups.kill(group, signal.SIGTERM)
            if not self.done.wait(2.):
                for group in tuple(self.owned):
                    self.groups.kill(group, signal.SIGKILL)

    def sample(self, initial=False):
        row = self.monitor.sample()
        self.record['samples'].append(row)
        self.snapshot()
        if row['memory_mib'] > self.plan['scope']['caps']['memory_limit_mib']:
            return 'memory_exceeded', 137
        foreign = [item for item in row['processes'] if not self.groups.owner(item['pid'], self.owned)]
        if foreign:
            return 'foreign_gpu_process', 125
        if initial and (row['memory_mib'] != 0 or row['utilization_percent'] != 0 or row['processes']):
            return 'gpu_not_idle', 125
        return None

    def cleanup(self):
        for group in tuple(self.owned):
            self.groups.kill(group, signal.SIGTERM)
        until = min(self.deadline - 8., self.clock() + 10.)
        while self.clock() < until and any(self.groups.alive(group) for group in self.owned):
            self.sleep(.1)
        for group in tuple(self.owned):
            if self.groups.alive(group):
                self.groups.kill(group, signal.SIGKILL)
        for process in self.processes:
            try:
                process.wait(timeout=max(.01, min(2., self.deadline - self.clock())))
            except subprocess.TimeoutExpired:
                pass
        for group in tuple(self.owned):
            if not self.groups.alive(group):
                self.owned.remove(group)
                self.group_history.append(group)
        self.record['active_owned_groups'] = sorted(self.owned)
        self.record['released_group_history'] = list(self.group_history)
        return not self.owned

    def phase(self, name, seconds):
        if self.clock() >= self.work_deadline:
            return 'deadline_exceeded', 124
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', ORCHIDEE_REPO_ROOT=self.plan['checkout'],
            CUDA_VISIBLE_DEVICES=self.plan['scope']['gpu_uuid'], JAX_PLATFORMS='cuda,cpu',
            XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        env['ORCJAX_NUMERICS_SHELL_STARTED_EPOCH'] = str(self.plan.get('shell_started_epoch') or time.time())
        env.update(self.plan.get('phase_environment', {}))
        stem = Path(self.plan['output']).with_suffix('.' + name)
        row = dict(name=name, argv=self.plan['commands'][name], started_epoch=time.time(),
                   cap_seconds=seconds, stdout=str(stem)+'.stdout.txt', stderr=str(stem)+'.stderr.txt')
        self.record['phases'].append(row)
        started = self.clock()
        end = min(started + seconds, self.work_deadline)
        with Path(row['stdout']).open('xb') as stdout, Path(row['stderr']).open('xb') as stderr:
            process = self.popen(row['argv'], cwd=self.plan['checkout'], env=env,
                stdout=stdout, stderr=stderr, start_new_session=True)
            self.owned.add(process.pid)
            self.processes.append(process)
            row['pid'] = process.pid
            self.record['GPU_executed'] = True
            self.snapshot()
            status = None
            while process.poll() is None:
                if self.interrupted:
                    status = ('deadline_exceeded', 124) if self.interrupted == 'hard_deadline' else ('interrupted', 143)
                    break
                if self.clock() >= end:
                    status = ('deadline_exceeded', 124)
                    break
                status = self.sample()
                if status:
                    break
                self.sleep(min(.5, max(.01, end - self.clock())))
            code = process.poll()
            if status is None:
                if self.clock() >= end:
                    status = ('deadline_exceeded', 124)
                else:
                    status = ('passed', 0) if code == 0 else ('numerical_failed', 3) if code == 3 else ('runtime_failed', code if code and code > 0 else 1)
            row.update(child_exit_code=code, primary_status=status[0], primary_exit_code=status[1], elapsed_seconds=self.clock()-started)
            self.snapshot()
            # Includes surviving descendants even after a normal parent exit.
            if not self.cleanup() and status[1] == 0:
                status = ('resource_unverified', 125)
            return status

    def run(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open('x', encoding='utf-8') as stream:
            json.dump(dict(schema_version='T7_affine_gpu_one_shot_ledger_v1', scope_id=self.plan['scope'].get('scope_id', SCOPE_ID),
                status='reserved', reserved_gpu_seconds=1200, old_budget_read=False), stream)
        if self.watchdog_enabled:
            threading.Thread(target=self.watchdog, daemon=True).start()
        primary = ('runtime_failed', 1)
        release_verified = False
        try:
            self.snapshot()
            if self.clock() >= self.work_deadline:
                primary = ('deadline_exceeded', 124)
            else:
                primary = self.sample(initial=True) or self.phase('preflight', self.plan['scope']['caps']['preflight_seconds'])
                if primary[1] == 0:
                    preflight = self.read_report(self.plan['preflight_output'], 'preflight_result_admission')
                    if (preflight.get('passed') is not True or preflight.get('actual_backend') != 'gpu'
                            or preflight.get('jax_version') != '0.4.38' or len(preflight.get('devices', [])) != 1):
                        raise ValueError('actual target JAX0.4.38/single-GPU preflight gate failed')
                    expected_uuid = self.plan.get('phase_environment', {}).get('ORCJAX_AFFINE_EXPECTED_GPU_UUID')
                    if expected_uuid is not None:
                        binding = preflight.get('affine_gpu_identity', {})
                        if (binding.get('passed') is not True or binding.get('expected_gpu_uuid') != expected_uuid
                                or binding.get('driver_gpu_uuid', '').lower() != expected_uuid.lower()
                                or binding.get('driver_device_count') != 1 or binding.get('pjrt_local_hardware_id') != 0):
                            raise ValueError('paid CUDA-driver/PJRT UUID binding preflight gate failed')
                    primary = self.phase('probe', self.plan['scope']['caps']['probe_seconds'])
                    if primary[1] == 0:
                        report = self.read_report(self.plan['output'], 'affine_result_admission')
                        if (report.get('status') != 'passed' or report.get('passed') is not True
                                or report.get('backend') != 'gpu' or report.get('jax_version') != '0.4.38'
                                or report.get('operand_sha256') != OPERAND_SHA
                                or report.get('measurement_count') != 12 or report.get('preserved_array_count') != 91
                                or report.get('total_small_graph_compiles') != 4
                                or report.get('tolerances') != {'rtol':1e-5, 'atol':1e-6}
                                or not Path(report['results_container']).is_file()):
                            raise ValueError('target affine result/count/identity admission failed')
                        if expected_uuid is not None:
                            binding = report.get('affine_gpu_identity', {})
                            if (binding.get('passed') is not True or binding.get('expected_gpu_uuid') != expected_uuid
                                    or binding.get('driver_gpu_uuid', '').lower() != expected_uuid.lower()
                                    or binding.get('driver_device_count') != 1 or binding.get('pjrt_local_hardware_id') != 0):
                                raise ValueError('actual affine child CUDA-driver/PJRT UUID binding gate failed')
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
                release_verified = (groups_released and not final['processes'] and final['memory_mib'] == 0
                    and final['utilization_percent'] == 0 and self.clock() <= self.deadline)
                self.record['release_observation'] = final
            except Exception as error:
                self.record['release_error'] = str(error)
            elapsed = self.clock() - self.started + self.shell_elapsed
            self.record.update(primary_status=primary[0], primary_exit_code=primary[1],
                release_verified=release_verified, process_groups_released=groups_released,
                occupied_seconds=elapsed, occupied_gpu_seconds=elapsed,
                within_total_cap=elapsed <= self.plan['scope']['caps']['total_seconds'],
                ended_epoch=time.time(), interrupted_signal=self.interrupted)
            result = primary[1] if primary[1] else 0 if release_verified else 125
            self.record.update(status=primary[0] if result == primary[1] else 'resource_unverified', exit_code=result)
            # Unknown release remains charged at the entire reservation and
            # one-shot closed; no successful accounting hides a primary failure.
            atomic_json(self.ledger, dict(schema_version='T7_affine_gpu_one_shot_ledger_v1', scope_id=self.plan['scope'].get('scope_id', SCOPE_ID),
                status='closed' if release_verified else 'closed_release_unverified', reserved_gpu_seconds=1200,
                charged_gpu_seconds=elapsed if release_verified else 1200, primary_status=primary[0],
                primary_exit_code=primary[1], exit_code=result, release_verified=release_verified,
                supervisor_receipt=str(self.out), old_budget_read=False))
            self.snapshot()
            self.done.set()
        return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = prepare(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        started = args.shell_started_epoch or time.time()
        remaining = CAPS['total_seconds'] - max(0., time.time()-started)
        if remaining <= CAPS['release_reserve_seconds']:
            raise ValueError('shell startup exhausted protected occupation time')
        actual_head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,
            timeout=min(4., remaining), text=True).strip()
        if actual_head != plan['scope'].get('code_git_head'):
            raise ValueError('deployed checkout HEAD differs from separately approved code version')
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
            cwd=ROOT, timeout=min(4., remaining), text=True)
        if dirty.strip():
            raise ValueError('live execution requires a clean frozen checkout')
        supervisor = Supervisor(plan)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda number, frame: setattr(supervisor, 'interrupted', number))
        # New free and paid entries cooperate through one persistent atomic
        # directory lock. The unchanged b676 entry does not use it; exclusion
        # covers new entries and relies on the declared sole launch owner.
        lock = AffineScopeLock(SCOPE_LOCK, SCOPE_ID).acquire()
        try:
            return supervisor.run()
        finally:
            lock.release(release_verified=supervisor.record.get('release_verified') is True)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(json.dumps(dict(status='admission_rejected', GPU_executed=False,
            error_type=type(error).__name__, error=str(error))))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
