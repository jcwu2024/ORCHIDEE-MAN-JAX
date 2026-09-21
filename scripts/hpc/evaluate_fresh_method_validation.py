"""Evaluate one declared fresh-method checkpoint on the held-out 48-window panel.

This entry point is forward-only. It restores parameters from a retained
mixed-learning milestone, but never creates a gradient or optimizer update.
"""
from __future__ import annotations

import gc
import json
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.hpc import run_fresh_start_one_day as prototype
from scripts.hpc.run_mixed_learning import (
    FRESH_LITTER_SCHEMA,
    configure_execution,
    validate_configuration,
)

SCHEMA_VERSION = 'fresh_method_validation_evaluation_v1'


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise TypeError(f'{name} must contain a JSON object')
    return value


def training_method_from_report(
    report: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    model_protocol: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the method identity written by the mixed-learning runner."""
    if configuration.get('schema_version') != FRESH_LITTER_SCHEMA:
        raise ValueError('validation evaluator requires the fresh litter method')
    if (report.get('configuration') != configuration
            or report.get('model_protocol') != model_protocol
            or report.get('execution_policy') != execution_policy):
        raise ValueError('training report configuration or execution method differs')
    if report.get('scope') != 'one_pool_pass_learning':
        raise ValueError('validation requires a one-pool-pass training report')
    if report.get('device_execution') is None:
        raise ValueError('training report lacks its declared device execution')
    method = dict(
        implementation='mixed_learning_v1',
        configuration=dict(configuration),
        model_protocol=dict(model_protocol),
        scope='one_pool_pass_learning',
        integration_updates=None,
        source_git_head=report['source_git_head'],
        execution_policy=dict(execution_policy),
        recurrent_execution=execution_policy['recurrent_execution'],
        device_execution=report['device_execution'],
    )
    if report.get('learning_schedule') is not None:
        method['learning_schedule'] = report['learning_schedule']
    return method


def checkpoint_identity(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    checkpoint_path: Path,
    expected_update: int,
    configuration: Mapping[str, Any],
    model_protocol: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the trained snapshot and return strict loader identities."""
    milestones = tuple(int(value) for value in configuration['milestones'])
    if expected_update <= 0 or expected_update not in milestones:
        raise ValueError('expected update is not a declared trained milestone')
    final_update = int(configuration['updates'])
    completed = report.get('completed_updates')
    if (not isinstance(completed, int) or completed < expected_update
            or report.get('scientific_acceptance') is not False
            or report.get('checkpoint_inherited') is not False):
        raise ValueError('training report is not the declared fresh milestone snapshot')
    if report.get('data_plan') != plan['summary']:
        raise ValueError('training report data plan differs from validated inputs')

    source = Path(checkpoint_path).resolve()
    matches = [
        row for row in report.get('checkpoints', ())
        if row.get('update') == expected_update
        and Path(row.get('path', '')).resolve() == source
    ]
    if len(matches) != 1:
        raise ValueError('checkpoint path and update are not recorded by the training report')
    fresh = dict(
        seed=model_protocol['seed'],
        initialization=model_protocol['initialization'],
        parameter_count=model_protocol['parameter_count'],
        inherited_weights=False,
    )
    return dict(
        expected_method=training_method_from_report(
            report,
            configuration=configuration,
            model_protocol=model_protocol,
            execution_policy=execution_policy,
        ),
        expected_source_identities=plan['identities'],
        expected_schedule_identity=plan['identities'],
        expected_fresh_initialization_identity=fresh,
        maximum_updates=final_update,
        source_report=str(Path(report_path).resolve()),
        source_report_state=dict(
            status=report.get('status'),
            completed_updates=completed,
            latest_checkpoint_forward_replay=report.get('checkpoint_forward_replay'),
        ),
        checkpoint_record=dict(matches[0]),
    )


def validate_panel_identity(pool: Any) -> None:
    panel = pool.panel
    method = panel.get('method', {})
    if (panel.get('schema_version') != 'fresh_method_validation_panel_v1'
            or method.get('method_id') != FRESH_LITTER_SCHEMA
            or int(panel.get('windows_per_slice_per_horizon', -1)) != 8
            or int(method.get('windows_per_slice_per_horizon', -1)) != 8
            or len(pool.rows) != 48):
        raise ValueError('prepared panel is not the current 48-window fresh-method validation')


def run(args: Any) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.mixed_training_state import (
        load_mixed_training_checkpoint,
    )
    from scripts.hpc.fresh_method_validation_pool import (
        FreshMethodValidationPool,
        evaluate_validation_panel,
    )
    from scripts.hpc.mixed_learning_monitor import build_mixed_rollout_monitor
    from scripts.hpc.mixed_learning_plan import load_plan
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.trained_screen_common import json_safe

    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError('fresh validation evaluation requires a new output')
    execution_policy = configure_execution(args)
    if execution_policy is None:
        raise ValueError('fresh validation evaluation requires the admitted execution policy')
    configuration = _read_json(args.mixed_protocol, name='mixed protocol')
    model_protocol = _read_json(args.protocol, name='model protocol')
    validate_configuration(configuration, model_protocol)
    if ((args.mixed_protocol.parent / configuration['model_protocol']).resolve()
            != args.protocol.resolve()):
        raise ValueError('CLI and mixed model protocols differ')
    plan = load_plan(args, configuration)
    source_report = _read_json(args.training_report, name='training report')
    identity = checkpoint_identity(
        source_report,
        report_path=args.training_report,
        checkpoint_path=args.training_checkpoint,
        expected_update=args.expected_update,
        configuration=configuration,
        model_protocol=model_protocol,
        execution_policy=execution_policy,
        plan=plan,
    )
    loader_identity = {
        key: identity[key]
        for key in (
            'expected_method', 'expected_source_identities',
            'expected_schedule_identity', 'expected_fresh_initialization_identity',
            'maximum_updates',
        )
    }
    validation_pool = FreshMethodValidationPool(args.validation_manifest)
    validate_panel_identity(validation_pool)

    print('fresh_method_validation stage=bootstrap', flush=True)
    runtime, head, template, scales, temporary_parameters, _, _ = prototype.bootstrap(
        args, model_protocol)
    jax = runtime['jax']
    devices = tuple(jax.local_devices())
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise ValueError('fresh validation evaluation requires exactly one visible GPU')
    device = devices[0]
    del temporary_parameters
    gc.collect()

    print('fresh_method_validation stage=checkpoint_load', flush=True)
    checkpoint = load_mixed_training_checkpoint(
        args.training_checkpoint, **loader_identity)
    if (checkpoint['completed_updates'] != args.expected_update
            or checkpoint['daily_sampler_cursor'] != {
                'epoch': 0, 'batch_offset': args.expected_update}
            or checkpoint['recurrent_schedule_cursor'] != args.expected_update):
        raise ValueError('checkpoint progress differs from the declared milestone')
    host_parameters = checkpoint['parameters']
    checkpoint_progress = dict(
        completed_updates=checkpoint['completed_updates'],
        daily_sampler_cursor=checkpoint['daily_sampler_cursor'],
        recurrent_schedule_cursor=checkpoint['recurrent_schedule_cursor'],
    )
    del checkpoint
    gc.collect()
    parameters = jax.device_put(host_parameters, device)
    del host_parameters
    gc.collect()

    monitor = build_mixed_rollout_monitor(
        runtime, head, template, scales, model_protocol,
        compiled_memory_limit_bytes=18 * 1024**3,
    )
    print('fresh_method_validation stage=panel_forward', flush=True)
    validation = evaluate_validation_panel(monitor, parameters, validation_pool)
    passed = bool(
        validation['physical_gate_passed']
        and validation['metric_population_complete']
    )
    result = json_safe(dict(
        schema_version=SCHEMA_VERSION,
        status='passed' if passed else 'failed',
        scope='forward_only_held_out_fresh_method_validation',
        source_git_head=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        training_source_git_head=source_report['source_git_head'],
        training_report=str(args.training_report.resolve()),
        training_checkpoint=str(args.training_checkpoint.resolve()),
        expected_update=args.expected_update,
        checkpoint_record=identity['checkpoint_record'],
        checkpoint_progress=checkpoint_progress,
        source_report_state=identity['source_report_state'],
        configuration=configuration,
        model_protocol=model_protocol,
        execution_policy=execution_policy,
        training_device_execution=source_report['device_execution'],
        evaluation_device=str(device),
        validation_manifest=str(args.validation_manifest.resolve()),
        validation=validation,
        parameter_updates=0,
        gradient_evaluations=0,
        optimizer_transferred_to_device=False,
        checkpoint_inherited_for_training=False,
        sealed_test_used=False,
        scientific_acceptance=False,
        elapsed_seconds=time.perf_counter() - started,
    ))
    _atomic_json(args.output, result)
    print(json.dumps(result, separators=(',', ':'), allow_nan=False), flush=True)
    if not passed:
        raise ValueError('fresh validation physical or metric-population gate failed')
    return result


def parser():
    result = prototype.parser()
    for name in (
        'mixed-protocol', 'catalogue-report', 'recurrent-schedule', 'selection',
        'execution-policy', 'training-report', 'training-checkpoint',
        'validation-manifest',
    ):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--expected-update', type=int, required=True)
    result.set_defaults(
        compare_checked_recurrent=False,
        checked_reference_report=None,
        dual_device_profile=False,
        dual_device_training=False,
    )
    return result


if __name__ == '__main__':
    run(parser().parse_args())
