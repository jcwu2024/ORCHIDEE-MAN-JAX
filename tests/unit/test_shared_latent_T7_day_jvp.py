"""Tiny analytic contracts for same-precision one-day JVP localization."""
from typing import NamedTuple
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import shared_latent_T7_day_jvp as day_jvp


class _Case(NamedTuple):
    initial_continuous: object
    initial_discrete: object
    days: object
    targets: object
    context: object


def _analytic_step(parameters, state, discrete, day):
    scale = parameters["observation_scale"]
    next_state = state + day["forcing"]
    next_discrete = discrete + day["day_code"]
    observations = {
        "flux": scale * state + day["forcing"],
        "scalar": jnp.sum(scale * state),
        "mask": day["mask"],
        "count": jnp.sum(day["mask"], dtype=jnp.int32),
        "discrete": next_discrete,
        "budgets": jnp.stack((jnp.sum(next_state), jnp.sum(state - next_state))),
        "learned_output_nonfinite": jnp.asarray(0, dtype=jnp.int32),
    }
    return next_state, next_discrete, observations


def _cases():
    jax.config.update("jax_enable_x64", True)
    cases = []
    for sample in range(day_jvp.BATCH_SIZE):
        days = {
            "forcing": jnp.arange(21, dtype=jnp.float64).reshape(7, 3) * (sample + 1) / 10,
            "day_code": jnp.arange(7, dtype=jnp.int32) + 1,
            "mask": (jnp.arange(21).reshape(7, 3) + sample) % 2 == 0,
        }
        cases.append(_Case(
            jnp.asarray([sample + 0.1, sample + 0.2, sample + 0.3], jnp.float64),
            jnp.asarray(sample, jnp.int32),
            days,
            {"unused": jnp.asarray(sample, jnp.int32)},
            {"unused": jnp.asarray(sample, jnp.int32)},
        ))
    stacked = jax.tree_util.tree_map(lambda *values: jnp.stack(values), *cases)
    return tuple(cases), stacked


def test_first_day_selector_uses_day_axis_zero_for_b1_and_axis_one_for_b4():
    cases, stacked = _cases()
    b1 = day_jvp.first_day_inputs(jax, cases[3], batched=False)
    b4 = day_jvp.first_day_inputs(jax, stacked, batched=True)
    assert b1.initial_continuous.shape == (3,)
    assert b4.initial_continuous.shape == (4, 3)
    np.testing.assert_array_equal(b1.day["forcing"], b4.day["forcing"][3])
    np.testing.assert_array_equal(b1.day["mask"], b4.day["mask"][3])
    np.testing.assert_array_equal(b1.day["day_code"], b4.day["day_code"][3])
    with pytest.raises(ValueError, match="four-case batch axis"):
        day_jvp.first_day_inputs(jax, cases[0], batched=True)


def test_b1_and_b4_full_first_day_jvps_match_without_mean_loss_scaling():
    cases, stacked = _cases()
    b1_inputs = day_jvp.first_day_inputs(jax, cases[3], batched=False)
    b4_inputs = day_jvp.first_day_inputs(jax, stacked, batched=True)
    direction = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64)
    directions = jnp.zeros_like(stacked.initial_continuous).at[3].set(direction)
    b1, b4 = day_jvp.route_functions(jax, _analytic_step)
    parameters = {"observation_scale": jnp.asarray(0.5, dtype=jnp.float32)}

    result_b1 = jax.jit(b1)(parameters, b1_inputs.initial_continuous, b1_inputs.initial_discrete,
                           b1_inputs.day, direction)
    result_b4 = jax.jit(b4)(parameters, b4_inputs.initial_continuous, b4_inputs.initial_discrete,
                           b4_inputs.day, directions)
    comparison = day_jvp.compare_results(jax, result_b1, result_b4, sample_index=3)

    assert comparison["passed"]
    np.testing.assert_array_equal(result_b1.tangent.next_continuous, direction)
    np.testing.assert_array_equal(result_b4.tangent.next_continuous[3], direction)
    np.testing.assert_array_equal(result_b1.tangent.observations["flux"], direction * 0.5)
    np.testing.assert_array_equal(result_b4.tangent.observations["flux"][3], direction * 0.5)
    assert comparison["auxiliary_mismatches"] == 0
    assert all(row["zero_gate"] for row in comparison["auxiliary_leaves"]
               if "zero_gate" in row)
    assert all(row["finite_gate"] for row in comparison["float_leaves"]
               if row["path"].endswith("['budgets']"))


def test_auxiliary_outputs_have_no_tangent_and_masks_are_exact():
    cases, _ = _cases()
    inputs = day_jvp.first_day_inputs(jax, cases[3], batched=False)
    b1, _ = day_jvp.route_functions(jax, _analytic_step)
    direction = jnp.ones((3,), dtype=jnp.float64)
    result = b1({"observation_scale": jnp.asarray(0.25, jnp.float32)},
                inputs.initial_continuous, inputs.initial_discrete, inputs.day, direction)
    assert result.auxiliary[0].dtype == jnp.int32
    np.testing.assert_array_equal(result.auxiliary[0], jnp.asarray(4, jnp.int32))
    np.testing.assert_array_equal(result.auxiliary[1]["mask"], inputs.day["mask"])
    np.testing.assert_array_equal(result.auxiliary[1]["count"], jnp.sum(inputs.day["mask"]))
    assert result.auxiliary[1]["learned_output_nonfinite"] == 0


def test_comparison_preserves_strict_failure_and_records_nonfinite_values():
    cases, _ = _cases()
    inputs = day_jvp.first_day_inputs(jax, cases[3], batched=False)
    b1, _ = day_jvp.route_functions(jax, _analytic_step)
    result = b1({"observation_scale": jnp.asarray(0.5, jnp.float32)},
                inputs.initial_continuous, inputs.initial_discrete, inputs.day,
                jnp.ones((3,), jnp.float64))
    altered_primal = result.primal._replace(
        next_continuous=result.primal.next_continuous.at[1].add(1.0e-2))
    altered = result._replace(primal=altered_primal)
    failed = day_jvp.compare_results(jax, altered, result)
    assert failed["passed"] is False
    assert failed["failed_float_coordinates"] >= 1
    row = next(row for row in failed["float_leaves"] if row["path"] == "primal.next_continuous")
    assert row["failed_coordinates"] == 1
    assert row["worst_position"] == [1]
    json.dumps(failed)

    altered_scalar_observations = {**result.primal.observations,
                                   "scalar": result.primal.observations["scalar"] + 1.0e-2}
    scalar_failed = result._replace(primal=result.primal._replace(
        observations=altered_scalar_observations))
    scalar_report = day_jvp.compare_results(jax, scalar_failed, result)
    scalar_row = next(row for row in scalar_report["float_leaves"]
                      if row["path"] == "primal.observations['scalar']")
    assert scalar_row["failed_coordinates"] == 1
    assert scalar_row["worst_position"] == []
    json.dumps(scalar_report)

    nonfinite_primal = result.primal._replace(
        observations={**result.primal.observations,
                      "flux": result.primal.observations["flux"].at[0].set(jnp.nan)})
    nonfinite = result._replace(primal=nonfinite_primal)
    report = day_jvp.compare_results(jax, nonfinite, result)
    assert report["passed"] is False
    assert report["nonfinite_float_coordinates"] == 1
    json.dumps(report)


def test_b4_sample_selection_fails_closed_and_auxiliary_gates_preserve_failures():
    cases, stacked = _cases()
    b1_inputs = day_jvp.first_day_inputs(jax, cases[3], batched=False)
    b4_inputs = day_jvp.first_day_inputs(jax, stacked, batched=True)
    b1, b4 = day_jvp.route_functions(jax, _analytic_step)
    parameters = {"observation_scale": jnp.asarray(0.5, jnp.float32)}
    direction = jnp.asarray([1.0, 0.0, 0.0], jnp.float64)
    directions = jnp.zeros_like(stacked.initial_continuous).at[3].set(direction)
    result_b1 = b1(parameters, b1_inputs.initial_continuous, b1_inputs.initial_discrete,
                   b1_inputs.day, direction)
    result_b4 = b4(parameters, b4_inputs.initial_continuous, b4_inputs.initial_discrete,
                   b4_inputs.day, directions)

    malformed = result_b4._replace(primal=result_b4.primal._replace(
        next_continuous=result_b4.primal.next_continuous[3]))
    with pytest.raises(ValueError, match="four-case batch axis"):
        day_jvp.compare_results(jax, result_b1, malformed, sample_index=3)
    with pytest.raises(ValueError, match="sample index"):
        day_jvp._select_sample(jax, {}, sample_index=4, batch_size=4)

    auxiliary = result_b4.auxiliary
    observations = dict(auxiliary[1])
    observations["mask"] = observations["mask"].at[3, 0].set(~observations["mask"][3, 0])
    mask_mismatch = result_b4._replace(auxiliary=(auxiliary[0], observations))
    mask_report = day_jvp.compare_results(jax, result_b1, mask_mismatch, sample_index=3)
    assert mask_report["passed"] is False
    mask_row = next(row for row in mask_report["auxiliary_leaves"] if row["path"].endswith("['mask']"))
    assert mask_row["exact_match"] is False

    b1_auxiliary = result_b1.auxiliary
    bad_observations = dict(b1_auxiliary[1])
    bad_observations["learned_output_nonfinite"] = jnp.asarray(2, jnp.int32)
    nonfinite_count = result_b1._replace(auxiliary=(b1_auxiliary[0], bad_observations))
    count_report = day_jvp.compare_results(jax, nonfinite_count, nonfinite_count)
    assert count_report["passed"] is False
    count_row = next(row for row in count_report["auxiliary_leaves"]
                     if row["path"].endswith("['learned_output_nonfinite']"))
    assert count_row["zero_gate"] is False
    assert count_row["actual_values"] == 2
    json.dumps(count_report)


def test_shape_dtype_and_nonfinite_state_direction_gates_fail_closed():
    cases, _ = _cases()
    inputs = day_jvp.first_day_inputs(jax, cases[0], batched=False)
    b1, _ = day_jvp.route_functions(jax, _analytic_step)
    parameters = {"observation_scale": jnp.asarray(0.5, jnp.float32)}
    with pytest.raises(ValueError, match="shape"):
        b1(parameters, inputs.initial_continuous, inputs.initial_discrete,
           inputs.day, jnp.ones((2,), jnp.float64))
    with pytest.raises(ValueError, match="FP64"):
        b1(parameters, inputs.initial_continuous, inputs.initial_discrete,
           inputs.day, jnp.ones((3,), jnp.float32))
