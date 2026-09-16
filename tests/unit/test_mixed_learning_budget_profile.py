from __future__ import annotations

import subprocess
from copy import copy
from pathlib import Path

import pytest

from scripts.hpc.run_mixed_learning import (
    configure_execution,
    parser,
    validate_learning_budget_mode,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = (
    ROOT
    / 'manifests'
    / 'coarse_graining'
    / 'mixed_learning_execution_fp32_checked_v1.json'
)


def _arguments(*extra):
    cli = parser()
    required = [action for action in cli._actions if action.required]
    base = [value for action in required for value in (action.option_strings[0], 'unused')]
    return cli.parse_args(base + list(extra))


def test_budget_mode_requires_fixed_checkpoint_and_checked_execution_policy():
    arguments = _arguments(
        '--profile-checkpoint', 'checkpoint.pkl',
        '--profile-training-report', 'training.json',
        '--execution-policy', str(POLICY),
        '--learning-budget-profile',
    )
    policy = configure_execution(arguments)
    validate_learning_budget_mode(arguments, profiling=True, execution_policy=policy)
    assert arguments.checked_recurrent and arguments.separate_checked_validation

    no_profile = copy(arguments)
    with pytest.raises(ValueError, match='fixed-checkpoint'):
        validate_learning_budget_mode(no_profile, profiling=False, execution_policy=policy)
    with pytest.raises(ValueError, match='admitted execution policy'):
        validate_learning_budget_mode(arguments, profiling=True, execution_policy=None)


@pytest.mark.parametrize(
    ('attribute', 'value'),
    (
        ('profile_trace', True),
        ('precision_stage', 'daily'),
        ('dual_device_profile', True),
        ('compare_checked_recurrent', True),
        ('checked_reference_report', Path('reference.json')),
        ('learning_schedule', Path('stages.json')),
    ),
)
def test_budget_mode_rejects_other_diagnostic_and_training_modes(attribute, value):
    arguments = _arguments(
        '--profile-checkpoint', 'checkpoint.pkl',
        '--profile-training-report', 'training.json',
        '--execution-policy', str(POLICY),
        '--learning-budget-profile',
    )
    policy = configure_execution(arguments)
    setattr(arguments, attribute, value)
    with pytest.raises(ValueError, match='separate from trace'):
        validate_learning_budget_mode(arguments, profiling=True, execution_policy=policy)


def test_free_budget_launcher_expands_single_gpu_fixed_checkpoint_command():
    launcher = ROOT / 'scripts' / 'hpc' / 'run_mixed_learning_budget_profile.sh'
    source = launcher.read_text()
    assert 'sbatch' not in source
    assert '--profile-trace' not in source
    bash_launcher = 'scripts/hpc/run_mixed_learning_budget_profile.sh'
    checkout = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/budget'
    completed = subprocess.run(
        ['bash', '-c',
         f'CODE_ROOT={checkout} PROFILE_REPEATS=5 bash {bash_launcher} --dry-run'],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    command = completed.stdout
    assert 'GPU_DEVICE=0' in command
    assert '--learning-budget-profile' in command
    assert '--profile-repeats 5' in command
    assert 'mixed_learning_execution_fp32_checked_v1.json' in command
    assert 'report.update-000200.checkpoint.pkl' in command

    failed = subprocess.run(
        ['bash', '-c',
         f'CODE_ROOT={checkout} PROFILE_REPEATS=2 bash {bash_launcher} --dry-run'],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert failed.returncode == 2
