from copy import deepcopy

from scripts.hpc.run_dual_training_integration import compare_records, integration_commands


def records():
    return [dict(update=i+1, daily_batch_index=i, recurrent_schedule_index=i,
                 integration_data={'daily_rows': [i], 'recurrent_rows': {'T7': i, 'T30': i}},
                 daily_loss=1., T7=2., T30=3., mixed_loss=3.5, norm=4., clip=.1,
                 gradient_norms={'daily': 1., 'T7': 2., 'T30': 3.},
                 daily_hard={'finite': 0}, recurrent_hard={'T7': {'finite': 0}, 'T30': {'finite': 0}})
            for i in range(3)]


def test_recovery_keeps_declared_budget_and_slurm_gpu_mapping(tmp_path):
    commands, paths = integration_commands(tmp_path/'report.json', ['--protocol', 'model.json'], ['3', '5'])
    assert [r['gpu'] for r in commands] == ['3', '3,5', '3,5', '3,5']
    for row in commands:
        command = row['command']
        assert command[command.index('--integration-updates')+1] == '3'
    stop, resume = commands[-2]['command'], commands[-1]['command']
    assert stop[stop.index('--stop-after-update')+1] == '2'
    assert resume[resume.index('--resume')+1] == str(paths['recovered'].with_suffix('.recovery.checkpoint.pkl'))
    assert stop[stop.index('--output')+1] == resume[resume.index('--output')+1]


def test_all_four_commands_parse_through_real_training_entry(tmp_path):
    from scripts.hpc.run_mixed_learning import parser

    cli = parser()
    common = [v for action in cli._actions if action.required and action.dest != 'output'
              for v in (action.option_strings[0], 'unused')]
    legs, _ = integration_commands(tmp_path/'result.json', common, ['3', '5'])
    for leg in legs:
        args = cli.parse_args(leg['command'][3:])
        assert args.integration_updates == 3
        assert args.dual_device_training == (leg['name'] != 'serial')


def test_comparison_requires_same_data_physics_and_numerics():
    expected = records()
    assert compare_records(expected, deepcopy(expected))['passed']
    for change in ('data', 'physical', 'numeric'):
        actual = deepcopy(expected)
        if change == 'data':
            actual[1]['integration_data']['recurrent_rows']['T7'] = 99
        elif change == 'physical':
            actual[1]['recurrent_hard']['T30']['finite'] = 1
        else:
            actual[1]['mixed_loss'] = 8.
        assert not compare_records(expected, actual)['passed']


def test_full_checkpoint_comparison_uses_shared_state_and_preserves_layout_identity(tmp_path):
    import json

    import jax
    import numpy as np

    from research.daily_coarse_graining.daily_operator.mixed_training_state import save_mixed_training_checkpoint
    from research.daily_coarse_graining.daily_operator.target_oriented_training import TargetOrientedAdamWState
    from scripts.hpc.run_dual_training_integration import compare_runs

    output = tmp_path/'admission.json'
    _, paths = integration_commands(output, [], ['0', '1'])
    for arm, path in paths.items():
        method = {'updates': 200, 'integration_updates': 3}
        if arm != 'serial':
            method['device_execution'] = {'primary': 0, 'recurrent': 1}
        save_mixed_training_checkpoint(
            path.with_suffix('.recovery.checkpoint.pkl'), jax=jax,
            parameters={'w': np.ones(2, dtype=np.float32)},
            optimizer_state=TargetOrientedAdamWState(3, {'w': np.zeros(2)}, {'w': np.ones(2)}), completed_updates=3,
            daily_sampler_cursor={'epoch': 0, 'batch_offset': 3}, recurrent_schedule_cursor=3,
            method=method, source_identities={'daily': 'pool'}, schedule_identity={'seed': 42},
            fresh_initialization_identity={'seed': 20260911})
        report = dict(status='completed', completed_updates=3, scope='integration_only',
                      checkpoint_forward_replay=True, configuration={}, model_protocol={},
                      source_git_head='same', execution_policy={}, data_plan={},
                      updates=[dict(r, seconds=1.) for r in records()])
        if arm == 'recovered':
            report['resumed_from'] = 'update2'
        path.write_text(json.dumps(report))
    result = compare_runs(output, paths)
    assert result['status'] == 'passed'
    assert not result['scientific_acceptance']
