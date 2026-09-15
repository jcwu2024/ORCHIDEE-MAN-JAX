from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.fresh_start_training import make_update
from research.daily_coarse_graining.daily_operator.mixed_objective import make_mixed_gradient
from research.daily_coarse_graining.daily_operator.one_day_training_state import (
    make_gradient_accumulator,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
)
from scripts.hpc.run_mixed_learning import (
    checkpoint_path,
    configure_execution,
    lowered_precision_summary,
    masked_precision_daily_auxiliary,
    newest_checkpoint,
    precision_reference_identity,
    precision_reference_path,
    precision_stage_components,
    profile_source_method,
    saved_gradient_path,
    trim_resume_report,
)

jax.config.update("jax_enable_x64", True)


def test_profile_uses_completed_checkpoint_method_not_diagnostic_head():
    report = dict(status="completed", completed_updates=200, configuration={"x": 1},
                  model_protocol={"seed": 2}, scope="bounded_200_update_learning",
                  source_git_head="original_training_head")
    method = profile_source_method(report, {"x": 1}, {"seed": 2})
    assert method["source_git_head"] == "original_training_head"
    assert method["scope"] == "bounded_200_update_learning"
    assert method["integration_updates"] is None
    with pytest.raises(ValueError):
        profile_source_method(dict(report, completed_updates=100), {"x": 1}, {"seed": 2})


def test_declared_combined_policy_binds_real_parser_and_checkpoint_method():
    from pathlib import Path

    from scripts.hpc.run_mixed_learning import parser

    policy_path = Path(__file__).resolve().parents[2] / 'manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json'
    # Populate inherited required parser fields without loading any Teacher data.
    cli = parser()
    required = [action for action in cli._actions if action.required]
    base = [value for action in required for value in (action.option_strings[0], 'unused')]
    train_args = cli.parse_args(base + ['--execution-policy', str(policy_path)])
    policy = configure_execution(train_args)
    assert train_args.attention_score_dtype == 'fp32'
    assert train_args.checked_recurrent and train_args.separate_checked_validation
    compare_args = cli.parse_args(base + ['--execution-policy', str(policy_path),
                                         '--compare-checked-recurrent'])
    assert configure_execution(compare_args) == policy
    assert not compare_args.checked_recurrent
    assert compare_args.separate_checked_validation
    report = dict(status='completed', completed_updates=200, configuration={},
                  model_protocol={}, scope='bounded_200_update_learning',
                  source_git_head='trained', execution_policy=policy)
    method = profile_source_method(report, {}, {})
    assert method['execution_policy'] == policy
    assert method['recurrent_execution'] == 'split_checked_v1'
    legacy_args = cli.parse_args(base)
    assert configure_execution(legacy_args) is None
    assert legacy_args.attention_score_dtype == 'legacy'
    assert not legacy_args.checked_recurrent


def test_execution_policy_cannot_silently_change_physical_precision(tmp_path):
    import json
    from argparse import Namespace

    path = tmp_path / 'execution.json'
    path.write_text(json.dumps(dict(schema_version='mixed_learning_execution_v1',
        attention_score_dtype='fp32', recurrent_execution='split_checked_v1',
        source_family_pooling_dtype='legacy', physical_dtype='float32')))
    with pytest.raises(ValueError, match='unsupported mixed-learning execution policy'):
        configure_execution(Namespace(execution_policy=path))


def test_static_precision_summary_reads_real_lowered_matrix_result_types():
    fn = jax.jit(lambda a, b: (a @ a, b @ b))
    lowered = fn.lower(jnp.ones((2, 2), jnp.float32), jnp.ones((2, 2), jnp.float64))
    summary = lowered_precision_summary(lowered)
    assert summary["matrix_operation_result_types"] == {"f32": 1, "f64": 1}


def test_reference_reuse_keeps_original_paths_across_interrupted_attempts(tmp_path):
    original = tmp_path / 'first.T7.reference.pkl'
    later = tmp_path / 'second.json'
    assert saved_gradient_path(later, {'reference_gradient': str(original)}, 'T7') == original
    assert saved_gradient_path(later, {'reference_gradient': str(original)}, 'T30') == later.with_suffix('.T30.reference.pkl')
    record = {'checked_comparisons': {'T7': {'reference_gradient': str(original)}}}
    assert saved_gradient_path(later, record, 'T7') == original


def test_precision_stages_and_reference_paths_are_component_specific(tmp_path):
    from research.daily_coarse_graining.daily_operator.one_day_sampling import OneDayPreparedRow

    assert precision_stage_components('daily') == ('daily',)
    assert precision_stage_components('T7') == ('T7',)
    assert precision_stage_components('full') == ('daily', 'T7', 'T30')
    with pytest.raises(ValueError, match='unsupported precision stage'):
        precision_stage_components('weekly')

    daily = tmp_path / 'legacy.daily.precision-reference.pkl'
    t7 = tmp_path / 'legacy.T7.reference.pkl'
    report = {
        'precision_comparisons': {'daily': {'reference_gradient': str(daily)}},
        'checked_comparisons': {'T7': {'reference_gradient': str(t7)}},
    }
    report_path = tmp_path / 'legacy.json'
    assert precision_reference_path(report_path, report, 'daily') == daily
    assert precision_reference_path(report_path, report, 'T7') == t7
    assert precision_reference_path(report_path, report, 'T30') == report_path.with_suffix('.T30.reference.pkl')

    rows = [{'artifact_id': 1, 'batch_row': 2}]
    assert precision_reference_identity('daily', checkpoint='checkpoint.pkl', rows=rows) == {
        'component': 'daily', 'checkpoint': 'checkpoint.pkl', 'rows': rows, 'window': None,
    }
    with pytest.raises(ValueError, match='daily precision'):
        precision_reference_identity('daily', checkpoint='checkpoint.pkl')
    prepared = OneDayPreparedRow(artifact_id=1, batch_row=2, record_index=0,
        row_index_within_record=2, landpoint_id='001', year=1962, start_row=3)
    identity = precision_reference_identity('daily', checkpoint='checkpoint.pkl', rows=[prepared])
    assert identity['rows'] == [prepared.as_metadata()]


def _loss(parameters, features, targets):
    prediction = features @ parameters["weight"] + parameters["bias"]
    return jnp.mean((prediction - targets) ** 2)


def test_precision_auxiliary_masks_only_undefined_values_and_retains_masks():
    valid = np.asarray([True, False])
    auxiliary = {
        'terms': np.asarray([[1., 2., 3.], [4., 5., 6.]]),
        'physical': {'fluxes': {'gpp': {
            'predicted': np.asarray([7., np.nan]),
            'teacher': np.asarray([8., np.nan]), 'defined': valid}}},
        'canopy_stock_change': {'row_valid': valid, 'scale_valid': np.ones(2, dtype=bool),
                                'predicted_change': np.asarray([.1, np.nan])},
        'litterpart_feedback': {'valid': valid[:, None],
                               'predicted': np.asarray([[.3], [np.nan]])},
    }
    result = masked_precision_daily_auxiliary(jax, auxiliary)
    np.testing.assert_array_equal(result['physical']['fluxes']['gpp']['predicted'], [7., 0.])
    np.testing.assert_array_equal(result['physical']['fluxes']['gpp']['defined'], valid)
    np.testing.assert_array_equal(result['litterpart_feedback']['predicted'], [[.3], [0.]])
    np.testing.assert_array_equal(result['canopy_stock_change']['predicted_change'], [.1, 0.])
    # An invalid numerical value in a defined slot must still fail comparison.
    auxiliary['physical']['fluxes']['gpp']['predicted'][0] = np.nan
    assert np.isnan(masked_precision_daily_auxiliary(jax, auxiliary)['physical']['fluxes']['gpp']['predicted'][0])


def _assert_tree_allclose(left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for actual, expected in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_three_component_gradient_and_single_clipped_adamw_match_direct_scalar():
    parameters = {
        "weight": jnp.asarray([0.4, -0.7, 0.2], dtype=jnp.float64),
        "bias": jnp.asarray(-0.15, dtype=jnp.float64),
    }
    index = jnp.arange(64, dtype=jnp.float64)
    daily_features = jnp.stack(
        (0.1 + index / 50.0, jnp.sin(index / 7.0), jnp.cos(index / 11.0)),
        axis=1,
    )
    daily_targets = 0.2 - index / 100.0
    t7_features = jnp.asarray([[2.0, -1.0, 0.5]], dtype=jnp.float64)
    t7_targets = jnp.asarray([-0.6], dtype=jnp.float64)
    t30_features = jnp.asarray([[-0.4, 1.7, 0.9]], dtype=jnp.float64)
    t30_targets = jnp.asarray([1.2], dtype=jnp.float64)

    gradient = jax.grad(_loss)
    add, finish = make_gradient_accumulator(jax, jnp)
    daily_total = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    for start in range(0, 64, 8):
        microgradient = gradient(
            parameters,
            daily_features[start : start + 8],
            daily_targets[start : start + 8],
        )
        daily_total = add(daily_total, microgradient, 8)
    daily_mean = finish(daily_total, 64)
    t7_gradient = gradient(parameters, t7_features, t7_targets)
    t30_gradient = gradient(parameters, t30_features, t30_targets)

    combine = make_mixed_gradient(jax, jnp)
    mixed = combine(daily_mean, t7_gradient, 1.0, 0.5)
    mixed = combine(mixed, t30_gradient, 1.0, 0.5)
    direct = jax.grad(
        lambda current: _loss(current, daily_features, daily_targets)
        + 0.5 * _loss(current, t7_features, t7_targets)
        + 0.5 * _loss(current, t30_features, t30_targets)
    )(parameters)
    _assert_tree_allclose(mixed, direct)

    protocol = {
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "learning_rate": 1.0e-5,
            "clip_norm": 1.0,
        }
    }
    update = make_update(protocol)
    optimizer = initialize_target_oriented_adamw(parameters)
    actual = update(parameters, mixed, optimizer)
    expected = update(parameters, direct, optimizer)
    _assert_tree_allclose(actual[0], expected[0])
    _assert_tree_allclose(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[3], expected[3], rtol=1e-12, atol=1e-12)
    assert float(actual[3]) < 1.0


def test_resume_report_discards_state_ahead_of_checkpoint_and_stale_failure():
    report = {
        "status": "failed",
        "stage": "failed",
        "completed_updates": 12,
        "updates": [{"update": value} for value in range(1, 13)],
        "monitors": [{"update": value} for value in (0, 10)],
        "checkpoints": [
            {"update": value, "path": f"update-{value}.pkl"}
            for value in (0, 10, 12)
        ],
        "failed_training_window": {"update": 12},
        "error": "ValueError: interrupted after report write",
        "next_update": 13,
        "monitor_update": 10,
        "scientific_acceptance": False,
    }

    result = trim_resume_report(report, 10)

    assert result["completed_updates"] == 10
    assert [row["update"] for row in result["updates"]] == list(range(1, 11))
    assert [row["update"] for row in result["monitors"]] == [0, 10]
    assert [row["update"] for row in result["checkpoints"]] == [0, 10]
    assert result["status"] == "running"
    for stale in (
        "failed_training_window",
        "error",
        "next_update",
        "monitor_update",
    ):
        assert stale not in result


def test_checkpoint_names_separate_rolling_recovery_and_retained_milestones(tmp_path):
    output = tmp_path / "mixed-learning.json"
    assert checkpoint_path(output).name == "mixed-learning.recovery.checkpoint.pkl"
    assert checkpoint_path(output, 0).name == "mixed-learning.update-000000.checkpoint.pkl"
    assert checkpoint_path(output, 200).name == "mixed-learning.update-000200.checkpoint.pkl"


def test_resume_report_cannot_silently_omit_completed_updates():
    with pytest.raises(ValueError, match="missing completed update records"):
        trim_resume_report({"updates": [{"update": 2}], "monitors": []}, 2)


def test_resume_prefers_newest_retained_milestone_over_stale_rolling_file(tmp_path):
    output = tmp_path / "mixed-learning.json"
    rolling = checkpoint_path(output)
    rolling.touch()
    checkpoint_path(output, 50).touch()
    checkpoint_path(output, 100).touch()

    assert newest_checkpoint(output, rolling, 90) == checkpoint_path(output, 100)
    assert newest_checkpoint(output, rolling, 100) == rolling
