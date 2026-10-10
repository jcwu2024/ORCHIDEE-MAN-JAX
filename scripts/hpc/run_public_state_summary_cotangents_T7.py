"""One finite free GPU1 original-FP32 public T7 cotangent window."""
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
    path = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-bank-adjoint-accumulation-20261002/run_shared_bank_adjoint_T7.py')
    spec = importlib.util.spec_from_file_location('public_cotangent_shared_launcher', path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)

ROOT, CHECKOUT = shared.ROOT, shared.CHECKOUT
OVERLAY = ROOT / 'runtime/transfer/public-state-summary-cotangents-T7-20261003'
FAMILY = ROOT / 'runtime/outputs/diagnostics/public-state-summary-cotangents-T7-20261003'
TOTAL_CAP, MAIN_CAP, GRACE = 3600, 3590, 10
HEAD = 'c599236bcf57b745e9bafed678298010f4c3b58f'
SOURCES = ('public_state_summary_cotangents_T7.py', 'probe_public_state_summary_cotangents_T7.py',
           'run_public_state_summary_cotangents_T7.py', 'run_public_state_summary_cotangents_T7.sh')
shared.GLOBAL_CAP = TOTAL_CAP


def commands(gpu, seconds=MAIN_CAP):
    model = shared.commands(gpu)['model']
    prefix = model[:model.index(str(shared.OVERLAY / 'probe_shared_bank_adjoint_T7.py'))]
    prefix = [value.replace('SINGULARITYENV_PYTHONPATH=' + str(CHECKOUT) + ':',
              'SINGULARITYENV_PYTHONPATH=' + str(CHECKOUT) + ':' + str(OVERLAY) + ':') for value in prefix]
    return {
        'preflight': prefix + [str(CHECKOUT / 'scripts/hpc/check_daily_training_runtime.py'), '--backend', 'gpu', '--output', str(FAMILY / 'preflight.json')],
        'capture': prefix + [str(OVERLAY / 'probe_public_state_summary_cotangents_T7.py'), '--root', str(ROOT),
            '--runtime-preflight', str(FAMILY / 'preflight.json'), '--output-dir', str(FAMILY / 'model'), '--process-seconds', str(seconds)]}


def reusable_preflight(prior):
    path = FAMILY / 'preflight.json'
    if not path.is_file() or not any(r.get('phase') in ('preflight', 'preflight_reused') and r.get('exit_status') == 0 for r in prior.get('records', [])):
        return False
    report = json.loads(path.read_text())
    if report.get('status') != 'passed' or not report.get('passed') or report.get('actual_backend') != 'gpu' or report.get('jax_version') != '0.4.38':
        return False
    return all((FAMILY / ('preflight-source-' + name)).read_bytes() == (CHECKOUT / 'scripts/hpc' / name).read_bytes()
               for name in ('check_daily_training_runtime.py', 'orcjax_gpu_runtime.py'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=('1',), default='1')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='precompile repair within exact original deadline only')
    args = parser.parse_args(argv)
    if args.dry_run:
        if 'jax' in sys.modules:
            raise RuntimeError('dry-run imported JAX')
        print(json.dumps(dict(status='prepared_parent_method_confirmation_required', resume=args.resume,
            total_seconds=TOTAL_CAP, main_seconds=MAIN_CAP, kill_grace_seconds=GRACE,
            preflight_seconds=90, setup_assets_original_replay_seconds=300, combined_lower_backend_seconds=3100,
            capture_call_raw_save_seconds=90, optional_binary_save_seconds=30, optimized_IR_export=False,
            maximum_capture_compilations=1, maximum_capture_calls=3, matched_timing_pairs=0,
            all_stages_clipped_to_original_deadline=True, origin='Bash timestamp before Python imports; resumes preserve first origin',
            source_checkout=str(CHECKOUT), overlay=str(OVERLAY), output=str(FAMILY),
            delta_shape=[7,8,3854], delta_dtype='float32', native_public_T7=True,
            full_parameter_coordinates=64884481, full_initial_state_coordinates=3854,
            scientific_changes=False, GPU_calls=0, Teacher_reads=0, training_updates=0, paid_jobs=0, CNY=0), indent=2))
        for label, command in commands(args.gpu).items():
            print(label + ': ' + shlex.join(command))
        return 0
    if socket.gethostname().split('.')[0] != 'ibc13b02n04' or os.environ.get('SLURM_JOB_ID'):
        raise ValueError('requires free gln01 GPU1 outside paid Slurm')
    if not os.environ.get('PUBLIC_COTANGENT_ORIGIN_UTC_SECONDS'):
        raise ValueError('use Bash entry: first origin required before Python imports')
    origin = float(os.environ['PUBLIC_COTANGENT_ORIGIN_UTC_SECONDS'])
    prior_records = []
    reuse = False
    if args.resume:
        prior = json.loads((FAMILY / 'actual-exit.json').read_text())
        origin = prior['original_start_UTC_seconds']
        if prior.get('capture_compile_attempts', 0) or (FAMILY / 'model/public-cotangents.result.pkl').exists():
            raise ValueError('capture graph attempted or raw result saved; terminal, no retry')
        if (FAMILY / 'declared-policy-source.py').read_bytes() != (OVERLAY / SOURCES[0]).read_bytes():
            raise ValueError('ordinary repair cannot change declared instrumentation')
        if prior.get('capture_status') in ('original_hard_admission_failed', 'strict_zero_delta_or_hard_bridge_failed', 'public_cotangents_captured_review_pending'):
            raise ValueError('hard/scientific terminal result cannot resume')
        prior_records = prior['records']
        reuse = reusable_preflight(prior)
    elapsed = time.time() - origin
    if elapsed < 0 or elapsed >= MAIN_CAP:
        raise TimeoutError('original dispatch allowance exhausted; no deadline reset')
    started = time.perf_counter() - elapsed
    if args.resume:
        preserved = FAMILY / ('preserved-precompile-attempt-' + str(len(prior_records)))
        preserved.mkdir(exist_ok=False)
        for name in ('actual-exit.json', 'capture.log', 'model') + (() if reuse else ('preflight.log', 'preflight.json')):
            source = FAMILY / name
            if source.exists():
                source.rename(preserved / name)
    else:
        FAMILY.mkdir(parents=True, exist_ok=False)
        (FAMILY / 'declared-policy-source.py').write_bytes((OVERLAY / SOURCES[0]).read_bytes())
        (FAMILY / 'origin.json').write_text(json.dumps(dict(original_start_UTC_seconds=origin, hard_deadline_UTC_seconds=origin+TOTAL_CAP)))
    snapshot = FAMILY / ('source-attempt-' + str(len(prior_records)))
    snapshot.mkdir(exist_ok=False)
    for name in SOURCES:
        (snapshot / name).write_bytes((OVERLAY / name).read_bytes())
    records, context, exception, status = [], {}, None, 1
    try:
        head = shared.setup_command(['git', 'rev-parse', 'HEAD'], started, records, cwd=CHECKOUT).strip()
        if head != HEAD:
            raise ValueError('immutable source checkout differs')
        context = shared.read_context(args.gpu, started, records)
        if reuse:
            records.append(dict(phase='preflight_reused', exit_status=0, new_GPU_calls=0, same_original_window=True))
        else:
            for name in ('check_daily_training_runtime.py', 'orcjax_gpu_runtime.py'):
                (FAMILY / ('preflight-source-' + name)).write_bytes((CHECKOUT / 'scripts/hpc' / name).read_bytes())
            status = shared.run_phase(commands(args.gpu)['preflight'], 'preflight', started, 90, FAMILY, records)
            if status:
                return status
        remaining = shared.effective_limit(started, MAIN_CAP)
        status = shared.run_phase(commands(args.gpu, remaining)['capture'], 'capture', started, MAIN_CAP, FAMILY, records)
        return status
    except BaseException as error:
        exception = repr(error)
        status = 124 if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else 1
        raise
    finally:
        model_path = FAMILY / 'model/model.json'
        model = json.loads(model_path.read_text()) if model_path.exists() else {}
        elapsed = time.perf_counter() - started
        ledger = dict(exit_status=status, original_start_UTC_seconds=origin, hard_deadline_UTC_seconds=origin+TOTAL_CAP,
            family_seconds=elapsed, card_hours=elapsed/3600, total_cap_seconds=TOTAL_CAP, kill_grace_seconds=GRACE,
            capture_compile_attempts=model.get('capture_compile_attempts',0), capture_status=model.get('status'),
            records=prior_records+records, execution_context=context, family_exception=exception,
            runtime_preflight_reused=reuse, physical_GPU=1, GPU_count=1, paid_jobs=0, CNY=0,
            training_updates=0, reference_adoption=False, production_adoption=False)
        (FAMILY / 'actual-exit.json').write_text(json.dumps(ledger, indent=2)+'\n')
        print(json.dumps(ledger, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
