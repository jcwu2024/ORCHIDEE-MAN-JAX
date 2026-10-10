"""Independent short recurrences through the public checkpointed loss path."""
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import shared_latent_T7_batch_ad as batch
from scripts.hpc.probe_shared_latent_T7_batch import bounded_failure_evidence, backend_budget_gate
import research.daily_coarse_graining.daily_operator.shared_recurrent_training as public
from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import STOCK_KEYS, SharedComponentTrajectories, SharedTrajectoryHardCounts


class Terms(NamedTuple):
    flux: object
    stock: object
    living_stock: object
    nonliving_stock: object
    hard_counts: object


@pytest.fixture
def public_cases(monkeypatch):
    """Replace only the physical transition/science fixtures, keep public AD."""
    def step_factory(_head, _retained):
        def step(parameters, state, discrete, forcing):
            new = parameters["rate"] * state + forcing
            return new, discrete, {"defined": jnp.ones_like(new, dtype=bool)}
        return step

    class Science:
        def terms(self, parameters, initial, observations, targets, context):
            del parameters, initial, context
            residual = jnp.sum((observations["state"]-targets["state"])**2)
            zero = jnp.asarray(0, dtype=jnp.int32)
            return Terms(residual, 2*residual, 0.0, 0.0, SharedTrajectoryHardCounts(zero, zero, zero, zero, zero))

    monkeypatch.setattr(public, "make_canonical_step", step_factory)
    monkeypatch.setattr(public, "make_shared_trajectory_loss", lambda **kwargs: Science())
    monkeypatch.setattr(public, "physical_counts", lambda obs, targets, *_args: {
        "mask": jnp.sum(obs["defined"] != targets["defined"], dtype=jnp.int32),
        "negative": jnp.sum(obs["state"] < 0, dtype=jnp.int32)})
    monkeypatch.setattr(public, "shared_component_trajectories", lambda *_args: SharedComponentTrajectories(
        fluxes={}, stocks={key: MaskedTrajectory(jnp.ones((1, 1)), jnp.ones((1, 1), dtype=bool)) for key in STOCK_KEYS}))
    head = SimpleNamespace(contract=object())
    scales = SimpleNamespace(state_declared_dynamic=SimpleNamespace(other=jnp.asarray([False, True])))
    rollout, objective = public.make_shared_recurrent_objective(head, object(), scales,
        {"flux_scales": {}, "stock_change_scales": {}})
    initial_masks = {key: jnp.asarray([True]) for key in STOCK_KEYS}
    cases = []
    # Two different short forcing trajectories, with four distinct starts.
    for index in range(4):
        forcing = jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.2, 0.1]]) + 0.07*(index % 2)
        initial = jnp.asarray([0.2, 0.5]) + 0.11*index
        context = public.SharedRecurrentLossContext(expected_discrete={"flag": jnp.zeros(3, dtype=jnp.int32)},
            templates=jnp.zeros(3), initial_stock_defined=initial_masks)
        targets = {"state": jnp.full((3, 2), 0.13*index), "defined": jnp.ones((3, 2), dtype=bool),
            "initial_stock_defined": initial_masks}
        cases.append((initial, {"flag": jnp.asarray(0, dtype=jnp.int32)}, forcing, targets, context))
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *cases)
    return rollout, objective, {"rate": jnp.asarray(0.6)}, tuple(cases), stacked


def _analytic(rate, initial, forcing, targets):
    """Scalar recurrence sensitivities computed without autodiff or batching."""
    state = np.asarray(initial).copy()
    d_parameter = np.zeros_like(state)
    d_initial = np.ones_like(state)
    loss = parameter = 0.0
    state_gradient = np.zeros_like(state)
    for value, target in zip(forcing, targets, strict=True):
        d_parameter = state + rate*d_parameter
        d_initial = rate*d_initial
        state = rate*state+value
        residual = state-target
        loss += np.sum(residual**2)/3
        parameter += np.sum(2*residual*d_parameter)/3
        state_gradient += 2*residual*d_initial/3
    return loss, parameter, state_gradient


def test_true_mean_full_ad_and_quarter_state_through_public_checkpoint(public_cases):
    rollout, objective, parameters, cases, stacked = public_cases
    actual = jax.jit(batch.make_batch_value_and_grad(jax, objective))(parameters, *stacked)
    analytic = [_analytic(float(parameters["rate"]), case[0], case[2], case[3]["state"]) for case in cases]
    np.testing.assert_allclose(actual[0][0], np.mean([value[0] for value in analytic]), rtol=2e-6)
    np.testing.assert_allclose(actual[1][0]["rate"], np.mean([value[1] for value in analytic]), rtol=2e-6)
    np.testing.assert_allclose(actual[1][1], np.stack([value[2] for value in analytic])/4, rtol=2e-6)
    assert np.shape(actual[1][0]["rate"]) == ()
    assert np.shape(actual[1][1]) == (4, 2)
    assert np.shape(actual[0][1]["flux"]) == (4,)
    assert all(np.all(value == 0) for value in actual[0][1]["hard"].values())
    # A separate plain public rollout verifies checkpointed objective AD.
    singles = [jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(parameters, *case) for case in cases]
    merged = jax.jit(lambda *rows: batch.merge_serial_results(jax, rows))(*singles)
    for value, expected in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(merged), strict=True):
        np.testing.assert_allclose(value, expected, rtol=2e-6, atol=1e-7)
    for case, single in zip(cases, singles, strict=True):
        plain = jax.value_and_grad(rollout.loss, argnums=(0, 1))(parameters, *case)
        np.testing.assert_allclose(single[0][0], plain[0], rtol=2e-6)
        np.testing.assert_allclose(single[1][1], plain[1][1], rtol=2e-6)


def test_one_sample_changes_only_its_state_ad_and_auxiliary(public_cases):
    _, objective, parameters, _, stacked = public_cases
    backward = jax.jit(batch.make_batch_value_and_grad(jax, objective))
    baseline = backward(parameters, *stacked)
    changed = list(stacked)
    changed[0] = changed[0].at[1].add(0.23)
    result = backward(parameters, *changed)
    np.testing.assert_array_equal(np.asarray(result[1][1])[[0, 2, 3]], np.asarray(baseline[1][1])[[0, 2, 3]])
    np.testing.assert_array_equal(np.asarray(result[0][1]["flux"])[[0, 2, 3]], np.asarray(baseline[0][1]["flux"])[[0, 2, 3]])
    assert not np.allclose(result[1][1][1], baseline[1][1][1])
    assert not np.allclose(result[1][0]["rate"], baseline[1][0]["rate"])


def test_per_sample_masks_and_physics_survive_batching(public_cases):
    _, objective, parameters, _, stacked = public_cases
    changed = list(stacked)
    changed[3] = dict(changed[3])
    changed[3]["defined"] = changed[3]["defined"].at[2, 1, 0].set(False)
    changed[0] = changed[0].at[3].set(jnp.asarray([-5.0, -5.0]))
    result = jax.jit(batch.make_batch_value_and_grad(jax, objective))(parameters, *changed)
    np.testing.assert_array_equal(result[0][1]["hard"]["mask"], [0, 0, 1, 0])
    assert int(result[0][1]["hard"]["negative"][3]) > 0
    assert np.all(np.isfinite(result[1][1]))
    with pytest.raises(ValueError, match="physical/mask"):
        batch.validate_batch_result(jax, jax.device_get(result), parameters, changed[0], SimpleNamespace())


def test_shape_estimate_is_not_a_tape_or_linear_memory_guarantee(public_cases):
    _, objective, parameters, _, stacked = public_cases
    args = (parameters, *stacked)
    shape = jax.eval_shape(batch.make_batch_value_and_grad(jax, objective), *args)
    estimate = batch.shape_memory_estimate(jax, shape, args)
    assert estimate["state_gradient_shape"] == [4, 2]
    assert estimate["parameter_gradient_coordinates"] == 1
    assert estimate["linear_memory_scaling_guaranteed"] is False
    assert estimate["reverse_tape_workspace_and_compiler_peak_estimated"] is False


def test_saved_executable_abi_rejects_wrong_state_shape_or_dtype(public_cases):
    _, objective, parameters, _, stacked = public_cases
    args = (parameters, *stacked)
    executable = jax.jit(batch.make_batch_value_and_grad(jax, objective)).lower(*args).compile()
    serialized_tree = (b"unused binary", jax.tree_util.tree_structure((args, {})), None)
    batch.validate_batch_abi(jax, serialized_tree, executable, args, state_width=2)
    wrong_shape = (parameters, stacked[0][:3], *stacked[1:])
    with pytest.raises(ValueError, match="state shape"):
        batch.validate_batch_abi(jax, serialized_tree, executable, wrong_shape, state_width=2)
    wrong_dtype = (parameters, stacked[0].astype(jnp.int32), *stacked[1:])
    with pytest.raises(ValueError, match="shape/dtype"):
        batch.validate_batch_abi(jax, serialized_tree, executable, wrong_dtype, state_width=2)


def test_all_nan_failure_metadata_keeps_complete_counts_and_bounded_positions():
    host = {"parameter_gradient": np.full((32, 128), np.nan),
            "finite_auxiliary": np.ones(7), "state_gradient": np.asarray([1.0, np.inf])}
    evidence = bounded_failure_evidence(jax, host)
    assert evidence["counts_complete"] is True
    assert evidence["nonfinite_coordinates"] == 4097
    assert evidence["finite_coordinates"] == 8
    leaves = {row["path"]: row for row in evidence["nonfinite_leaves"]}
    parameter = leaves["['parameter_gradient']"]
    assert parameter["nonfinite_coordinates"] == 4096
    assert len(parameter["position_samples"]) == 16
    assert parameter["position_samples_complete"] is False
    assert parameter["position_samples"][0]["index"] == [0, 0]
    assert leaves["['state_gradient']"]["position_samples_complete"] is True


def test_backend_budget_gate_preserves_minimum_added_time_before_compile():
    required = 5543.315643
    short = backend_budget_gate(required-123.25, required)
    assert short["passed"] is False
    assert short["required_backend_seconds"] == required
    assert short["available_backend_seconds"] == required-123.25
    assert short["minimum_additional_seconds"] == 123.25
    equal = backend_budget_gate(required, required)
    assert equal["passed"] is True
    assert equal["minimum_additional_seconds"] == 0.0
    assert equal["B4_backend_duration_guaranteed"] is False
    for invalid in (np.nan, np.inf, 0.0, -1.0):
        with pytest.raises(ValueError, match="positive measured B1"):
            backend_budget_gate(required, invalid)
