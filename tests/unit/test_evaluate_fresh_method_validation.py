import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.hpc.evaluate_fresh_method_validation import (
    checkpoint_identity,
    parser,
    training_method_from_report,
    validate_panel_identity,
)
from scripts.hpc.mixed_learning_dual_device import DUAL_DEVICE_TRAINING_LAYOUT

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = ROOT / 'manifests' / 'coarse_graining'


def _inputs(tmp_path):
    configuration = json.loads(
        (MANIFEST_ROOT / 'mixed_learning_fresh_litter_v1.json').read_text()
    )
    model = json.loads((MANIFEST_ROOT / configuration['model_protocol']).read_text())
    execution = json.loads(
        (MANIFEST_ROOT / 'mixed_learning_execution_fp32_checked_v1.json').read_text()
    )
    checkpoint = (tmp_path / 'report.update-001000.checkpoint.pkl').resolve()
    report_path = (tmp_path / 'report.json').resolve()
    summary = {'updates': 5415, 'daily_draws': 346560}
    identities = {'daily_source': 'daily', 'recurrent_schedule': 'schedule'}
    report = {
        'status': 'stopped_for_recovery',
        'configuration': configuration,
        'model_protocol': model,
        'scope': 'one_pool_pass_learning',
        'source_git_head': 'training-head',
        'execution_policy': execution,
        'device_execution': deepcopy(DUAL_DEVICE_TRAINING_LAYOUT),
        'completed_updates': 1000,
        'resource_segment_endpoint': 1000,
        'checkpoint_forward_replay': True,
        'scientific_acceptance': False,
        'checkpoint_inherited': False,
        'data_plan': deepcopy(summary),
        'checkpoints': [{'update': 1000, 'path': str(checkpoint), 'bytes': 123}],
    }
    return configuration, model, execution, report, report_path, checkpoint, {
        'summary': summary,
        'identities': identities,
    }


def test_checkpoint_identity_reconstructs_exact_fresh_training_method(tmp_path):
    configuration, model, execution, report, report_path, checkpoint, plan = _inputs(
        tmp_path
    )
    identity = checkpoint_identity(
        report,
        report_path=report_path,
        checkpoint_path=checkpoint,
        expected_update=1000,
        configuration=configuration,
        model_protocol=model,
        execution_policy=execution,
        plan=plan,
    )
    method = identity['expected_method']
    assert method == {
        'implementation': 'mixed_learning_v1',
        'configuration': configuration,
        'model_protocol': model,
        'scope': 'one_pool_pass_learning',
        'integration_updates': None,
        'source_git_head': 'training-head',
        'execution_policy': execution,
        'recurrent_execution': 'split_checked_v1',
        'device_execution': DUAL_DEVICE_TRAINING_LAYOUT,
    }
    assert identity['expected_source_identities'] == plan['identities']
    assert identity['expected_schedule_identity'] == plan['identities']
    assert identity['expected_fresh_initialization_identity'] == {
        'seed': 20260911,
        'initialization': model['initialization'],
        'parameter_count': 109464375,
        'inherited_weights': False,
    }
    assert identity['maximum_updates'] == 5415
    assert identity['checkpoint_record']['update'] == 1000


def test_integration_report_cannot_be_used_as_trained_validation_snapshot(tmp_path):
    configuration, model, execution, report, _, _, _ = _inputs(tmp_path)
    report.update(scope='integration_only', completed_updates=2, status='completed')
    with pytest.raises(ValueError, match='one-pool-pass'):
        training_method_from_report(
            report,
            configuration=configuration,
            model_protocol=model,
            execution_policy=execution,
        )


def test_older_epoch_method_cannot_cross_fresh_validation_boundary(tmp_path):
    _, model, execution, report, _, _, _ = _inputs(tmp_path)
    configuration = json.loads(
        (MANIFEST_ROOT / 'mixed_learning_epoch_v1.json').read_text()
    )
    report['configuration'] = configuration
    with pytest.raises(ValueError, match='fresh litter method'):
        training_method_from_report(
            report,
            configuration=configuration,
            model_protocol=model,
            execution_policy=execution,
        )


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        (lambda report, checkpoint: report['data_plan'].update(daily_draws=1),
         'data plan differs'),
        (lambda report, checkpoint: report['checkpoints'][0].update(path=str(checkpoint) + '.other'),
         'path and update'),
        (lambda report, checkpoint: report.update(completed_updates=999),
         'milestone snapshot'),
    ],
)
def test_checkpoint_identity_rejects_mismatched_source_and_snapshot(
    tmp_path, change, message
):
    configuration, model, execution, report, report_path, checkpoint, plan = _inputs(
        tmp_path
    )
    change(report, checkpoint)
    with pytest.raises(ValueError, match=message):
        checkpoint_identity(
            report,
            report_path=report_path,
            checkpoint_path=checkpoint,
            expected_update=1000,
            configuration=configuration,
            model_protocol=model,
            execution_policy=execution,
            plan=plan,
        )


def test_only_positive_declared_milestones_can_be_evaluated(tmp_path):
    configuration, model, execution, report, report_path, checkpoint, plan = _inputs(
        tmp_path
    )
    for update in (0, 999):
        with pytest.raises(ValueError, match='declared trained milestone'):
            checkpoint_identity(
                report,
                report_path=report_path,
                checkpoint_path=checkpoint,
                expected_update=update,
                configuration=configuration,
                model_protocol=model,
                execution_policy=execution,
                plan=plan,
            )


def test_later_source_report_can_retain_and_validate_earlier_milestone(tmp_path):
    configuration, model, execution, report, report_path, checkpoint, plan = _inputs(
        tmp_path
    )
    report.update(
        status='failed', completed_updates=2500, resource_segment_endpoint=4000,
        checkpoint_forward_replay=False,
    )
    identity = checkpoint_identity(
        report,
        report_path=report_path,
        checkpoint_path=checkpoint,
        expected_update=1000,
        configuration=configuration,
        model_protocol=model,
        execution_policy=execution,
        plan=plan,
    )
    assert identity['checkpoint_record']['update'] == 1000
    assert identity['source_report_state'] == {
        'status': 'failed',
        'completed_updates': 2500,
        'latest_checkpoint_forward_replay': False,
    }


def test_validation_panel_identity_is_current_method_and_exactly_48_windows():
    panel = {
        'schema_version': 'fresh_method_validation_panel_v1',
        'windows_per_slice_per_horizon': 8,
        'method': {
            'method_id': 'mixed_learning_fresh_litter_v1',
            'windows_per_slice_per_horizon': 8,
        },
    }
    pool = type('Pool', (), {'panel': panel, 'rows': [{}] * 48})()
    validate_panel_identity(pool)
    for change in (
        lambda value: value.panel.update(schema_version='old'),
        lambda value: value.panel['method'].update(method_id='old'),
        lambda value: setattr(value, 'rows', [{}] * 47),
    ):
        changed = type('Pool', (), {
            'panel': deepcopy(panel), 'rows': [{}] * 48,
        })()
        change(changed)
        with pytest.raises(ValueError, match='current 48-window'):
            validate_panel_identity(changed)


def test_parser_declares_forward_only_single_device_execution_defaults():
    cli = parser()
    required = [action for action in cli._actions if action.required]
    arguments = [
        value
        for action in required
        for value in (
            action.option_strings[0],
            '1000' if action.dest == 'expected_update' else 'unused',
        )
    ]
    args = cli.parse_args(arguments)
    assert args.expected_update == 1000
    assert args.dual_device_training is False
    assert args.dual_device_profile is False
    assert args.admission_only is False
    assert args.shape_only is False


def test_shell_dry_run_expands_actual_forward_only_command():
    checkout = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/fresh-validation'
    result = subprocess.run(
        [
            'bash', '-c',
            f'CODE_ROOT={checkout} GPU_DEVICE=1 EXPECTED_UPDATE=2500 '
            'bash scripts/hpc/run_fresh_method_validation.sh --dry-run',
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    command = result.stdout
    assert 'PYTHON_MODULE=scripts.hpc.evaluate_fresh_method_validation' in command
    assert 'GPU_DEVICE=1' in command
    assert 'RUN_TIMEOUT=90m' in command
    assert '--expected-update 2500' in command
    assert 'report.update-002500.checkpoint.pkl' in command
    assert 'fresh-method-validation-20260921/prepared_manifest.json' in command
    assert 'mixed_learning_execution_fp32_checked_v1.json' in command
    assert '--dual-device-training' not in command
    assert '--integration-updates' not in command
