"""Prepare a proposed 360-second free-GPU component window; no paid fallback."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
try:
    import run_shared_bank_adjoint_T7 as shared
except ModuleNotFoundError as error:
    if error.name != 'run_shared_bank_adjoint_T7':
        raise
    helper = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_adjoint_T7.py')
    spec = importlib.util.spec_from_file_location('component_shared_launcher', helper)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)

ROOT = shared.ROOT
CHECKOUT = shared.CHECKOUT
OVERLAY = ROOT / 'runtime/transfer/normalized-state-fanout-component-20261003'
FAMILY = ROOT / 'runtime/outputs/diagnostics/normalized-state-fanout-component-20261003'
PREPARED = ROOT / 'runtime/outputs/diagnostics/shared-bank-adjoint-proved-binary-T7-20261003/next-component-preparation'
TOTAL_CAP, MAIN_CAP, GRACE = 360, 350, 10
HEAD = 'c599236bcf57b745e9bafed678298010f4c3b58f'
shared.GLOBAL_CAP = TOTAL_CAP


def reusable_preflight(prior, family=FAMILY, checkout=CHECKOUT):
    """Reuse only passing gate bytes recorded in this same unchanged window."""
    path = family / 'preflight.json'
    if not path.is_file() or not any(row.get('phase') in ('preflight', 'preflight_reused')
            and row.get('exit_status') == 0 for row in prior.get('records', [])):
        return False
    report = json.loads(path.read_text())
    if report.get('passed') is not True or report.get('actual_backend') != 'gpu':
        return False
    for filename in ('check_daily_training_runtime.py', 'orcjax_gpu_runtime.py'):
        snapshot = family / ('preflight-source-' + filename)
        if not snapshot.is_file() or snapshot.read_bytes() != (checkout / 'scripts/hpc' / filename).read_bytes():
            return False
    return True


def commands(gpu, seconds=MAIN_CAP):
    existing = shared.commands(gpu)
    model = existing['model']
    old_entry = str(shared.OVERLAY / 'probe_shared_bank_adjoint_T7.py')
    prefix = model[:model.index(old_entry)]
    prefix = [value.replace('SINGULARITYENV_PYTHONPATH=' + str(CHECKOUT) + ':',
        'SINGULARITYENV_PYTHONPATH=' + str(CHECKOUT) + ':' + str(OVERLAY) + ':') for value in prefix]
    return {
        'preflight': prefix + [str(CHECKOUT / 'scripts/hpc/check_daily_training_runtime.py'),
            '--backend', 'gpu', '--output', str(FAMILY / 'preflight.json')],
        'component': prefix + [str(OVERLAY / 'probe_normalized_state_fanout_component.py'),
            '--root', str(ROOT), '--prepared-dir', str(PREPARED), '--runtime-preflight',
            str(FAMILY / 'preflight.json'), '--output-dir', str(FAMILY / 'model'), '--process-seconds', str(seconds)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('1',), default='1')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='ordinary preparation repair within original deadline; no compile retry')
    args = parser.parse_args(argv)
    if args.dry_run:
        if any(name == 'jax' or name.startswith('jax.') for name in sys.modules):
            raise RuntimeError('pure launcher dry-run imported JAX')
        metadata = dict(status='new_explicit_GPU_window_required', total_cap_seconds=TOTAL_CAP,
            resume_requested=args.resume,
            main_cap_seconds=MAIN_CAP, kill_grace_seconds=GRACE, preflight_cap_seconds=90,
            component_inputs_cap_seconds=90, compile_cap_seconds_per_arm=60,
            max_compilation_attempts_per_arm=1, call_cap_seconds=15, matched_pairs=3,
            actual_optimized_IR_capture_cap_per_arm=15, actual_IR_review_required=True,
            GPU_calls=0, Teacher_reads=0, model_values=0, paid_jobs=0, CNY=0,
            source_checkout=str(CHECKOUT), overlay=str(OVERLAY), output=str(FAMILY),
            prepared_dir=str(PREPARED), GPU_compile_time='unmeasured; caps are limits, not completion promises',
            deadline_rule='all stages and ordinary repairs share original 360-second deadline',
            resume_preflight_rule='reuse passing same-window preflight when checker/runtime source bytes are unchanged; repair failed gate within original deadline',
            hypothesis='native normalized-state cross-consumer fan-in F64; summary consumers/reducers remain F32',
            public_loss_cotangents=False, training_updates=0, reference_adoption=False)
        print(json.dumps(metadata, indent=2))
        for label, command in commands(args.gpu).items():
            print(label + ': ' + shlex.join(command))
        return 0
    if socket.gethostname().split('.')[0] != 'ibc13b02n04' or os.environ.get('SLURM_JOB_ID'):
        raise ValueError('actual run requires free gln01 GPU1 outside paid Slurm')
    original_start = time.time()
    started = time.perf_counter()
    prior_records = []
    reuse_preflight = False
    policy_source = OVERLAY / 'normalized_state_fanout_component.py'
    if args.resume:
        prior = json.loads((FAMILY / 'actual-exit.json').read_text())
        original_start = prior['original_start_UTC_seconds']
        elapsed = time.time() - original_start
        if elapsed < 0 or elapsed >= MAIN_CAP:
            raise TimeoutError('original component allowance exhausted; no deadline reset')
        if prior.get('compile_attempts', {}).get('original', 0) or prior.get('compile_attempts', {}).get('candidate', 0):
            raise ValueError('one compile per arm already attempted; no automatic retry')
        if prior.get('component_status') in ('hard_admission_failed', 'component_numerical_discrepancy', 'component_numerical_passed_IR_review_pending'):
            raise ValueError('scientific/hard result is terminal')
        if (FAMILY / 'candidate-policy-source.py').read_bytes() != policy_source.read_bytes():
            raise ValueError('ordinary repair cannot change the declared candidate policy')
        started -= elapsed
        prior_records = prior['records']
        reuse_preflight = reusable_preflight(prior)
        preserved = FAMILY / ('preserved-precompile-attempt-' + str(len(prior_records)))
        preserved.mkdir(exist_ok=False)
        names = ['actual-exit.json', 'component.log', 'model']
        if not reuse_preflight:
            names.extend(('preflight.log', 'preflight.json'))
        for name in names:
            source = FAMILY / name
            if source.exists():
                source.rename(preserved / name)
    else:
        FAMILY.mkdir(parents=True, exist_ok=False)
        (FAMILY / 'candidate-policy-source.py').write_bytes(policy_source.read_bytes())
    records = []
    status = 1
    context = {}
    exception = None
    try:
        head = shared.setup_command(['git', 'rev-parse', 'HEAD'], started, records, cwd=CHECKOUT).strip()
        if head != HEAD:
            raise ValueError('declared immutable checkout differs')
        context = shared.read_context(args.gpu, started, records)
        if reuse_preflight:
            records.append(dict(phase='preflight_reused', exit_status=0,
                start_UTC_seconds=time.time(), start_family_elapsed_seconds=time.perf_counter() - started,
                report=str(FAMILY / 'preflight.json'), same_original_window=True,
                unchanged_checker_runtime_source_bytes=True, new_GPU_preflight_calls=0))
        else:
            for filename in ('check_daily_training_runtime.py', 'orcjax_gpu_runtime.py'):
                (FAMILY / ('preflight-source-' + filename)).write_bytes((CHECKOUT / 'scripts/hpc' / filename).read_bytes())
            status = shared.run_phase(commands(args.gpu)['preflight'], 'preflight', started, 90, FAMILY, records)
            if status:
                return status
        remaining = shared.effective_limit(started, MAIN_CAP)
        status = shared.run_phase(commands(args.gpu, remaining)['component'], 'component', started, MAIN_CAP, FAMILY, records)
        if list((FAMILY / 'model').glob('*.timeout.json')):
            status = 124
        return status
    except BaseException as error:
        exception = repr(error)
        status = 124 if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else 1
        raise
    finally:
        report_path = FAMILY / 'model/model.json'
        component = json.loads(report_path.read_text()) if report_path.exists() else {}
        elapsed = time.perf_counter() - started
        ledger = dict(exit_status=status, family_seconds=elapsed, card_hours=elapsed / 3600,
            original_start_UTC_seconds=original_start, hard_deadline_UTC_seconds=original_start + TOTAL_CAP,
            total_cap_including_grace=TOTAL_CAP, kill_grace_seconds=GRACE,
            compile_attempts=component.get('compile_attempts', {'original': 0, 'candidate': 0}),
            component_status=component.get('status'), records=prior_records + records,
            runtime_preflight_reused=reuse_preflight,
            execution_context=context, family_exception=exception, GPU_count=1, physical_GPU=1,
            paid_jobs=0, CNY=0, training_updates=0, reference_adoption=False, production_adoption=False)
        (FAMILY / 'actual-exit.json').write_text(json.dumps(ledger, indent=2) + '\n')
        print(json.dumps(ledger, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
