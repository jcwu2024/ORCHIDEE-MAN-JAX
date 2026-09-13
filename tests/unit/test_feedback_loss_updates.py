from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.probe_feedback_loss_calibration import _make_batch_loss
from scripts.hpc.probe_feedback_loss_updates import (
    CONFIG_SCHEMA_VERSION,
    _calibration_complete,
    _coefficients,
    _directional_record,
    _validate_inputs,
    parser,
)


def _model_protocol():
    return {
        "schema_version": "fresh_start_one_day_probe_v3",
        "seed": 20260911,
        "parameter_count": 109_464_375,
        "trainable_scope": "all_neural_parameters",
    }


def _pool_protocol():
    return {
        "sampling": {
            "seed": 20260912,
            "batch_size": 32,
            "microbatch_size": 4,
            "max_artifacts_per_batch": 4,
            "epochs": 1,
            "scope": "prepared_one_day_extension_1001_3000",
        }
    }


def _completed_calibration():
    return {
        "schema_version": "feedback_loss_gradient_calibration_v1",
        "status": "completed",
        "training_updates": 0,
        "coefficient_selected": None,
        "all_reported_losses_finite": True,
        "all_gradient_values_finite": True,
        "all_hard_counts_zero": True,
        "batches": [{"completed": True} for _ in range(4)],
    }


def test_coefficients_must_be_external_complete_and_keep_base_fixed():
    valid = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "coefficients": {
            "base": 1.0,
            "canopy_stock_change": 3e-4,
            "litterpart_feedback": 1e-4,
        },
        "description": "descriptive root-owned fields are allowed",
    }
    assert _coefficients(valid) == (1.0, 3e-4, 1e-4)

    missing = {**valid, "coefficients": {**valid["coefficients"], "canopy_stock_change": None}}
    with pytest.raises((TypeError, ValueError)):
        _coefficients(missing)

    changed_base = {**valid, "coefficients": {**valid["coefficients"], "base": 0.5}}
    with pytest.raises(ValueError, match="base-loss coefficient"):
        _coefficients(changed_base)


def test_execution_requires_the_complete_finite_zero_hard_calibration():
    completed = _completed_calibration()
    assert _calibration_complete(completed)
    _validate_inputs(_pool_protocol(), _model_protocol(), completed, shape_only=False)

    incomplete = {**completed, "all_hard_counts_zero": False}
    assert not _calibration_complete(incomplete)
    with pytest.raises(ValueError, match="calibration"):
        _validate_inputs(_pool_protocol(), _model_protocol(), incomplete, shape_only=False)

    # Shape tracing uses explicitly supplied synthetic coefficients and makes
    # no science-execution claim, so it may be prepared while calibration runs.
    _validate_inputs(_pool_protocol(), _model_protocol(), incomplete, shape_only=True)


def test_shared_combined_scalar_has_two_bounded_reverse_jvp_agreements():
    base_weights = {
        "independent_daily_flux_mse": 1.0,
        "one_day_state_anchor": 0.1,
        "necessary_process_supervision": 0.01,
    }

    def objective(parameters, sample):
        prediction = parameters["weight"] * sample + parameters["bias"]
        return prediction * 0.0, {
            "flux_loss": (prediction - 1.0) ** 2,
            "state_loss": (prediction + 1.0) ** 2,
            "process_loss": prediction**2,
            "canopy_stock_change": {"loss": (prediction - 2.0) ** 2},
            "litterpart_feedback": {"loss": (prediction + 3.0) ** 2},
            "hard": {"synthetic": jnp.asarray(0, dtype=jnp.int32)},
            "physical": {},
        }

    parameters = {
        "weight": jnp.asarray(0.25),
        "bias": jnp.asarray(-0.5),
    }
    samples = jnp.asarray([-2.0, -0.5, 1.0, 3.0])
    coefficients = jnp.asarray([1.0, 3e-4, 1e-4])
    scalar = _make_batch_loss(jax, jnp, objective, base_weights)
    value, gradient = jax.value_and_grad(scalar)(parameters, samples, coefficients)

    directions = (
        {"weight": jnp.asarray(0.6), "bias": jnp.asarray(0.8)},
        {"weight": jnp.asarray(-0.8), "bias": jnp.asarray(0.6)},
    )
    records = []
    for seed, direction in zip((20260913, 20260914), directions, strict=True):
        primal, forward = jax.jvp(
            lambda current: scalar(current, samples, coefficients),
            (parameters,),
            (direction,),
        )
        reverse = sum(jnp.vdot(gradient[name], direction[name]) for name in parameters)
        records.append(_directional_record(primal, reverse, forward, seed=seed))

    np.testing.assert_allclose(float(value), records[0]["primal"])
    assert all(record["passed"] for record in records)
    assert [record["seed"] for record in records] == [20260913, 20260914]


def test_cli_keeps_v3_default_and_requires_both_evidence_inputs(tmp_path):
    base = [
        "--dataset-manifest",
        str(tmp_path / "dataset.json"),
        "--acceptance-report",
        str(tmp_path / "acceptance.json"),
        "--base-statistics",
        str(tmp_path / "base.npz"),
        "--process-statistics",
        str(tmp_path / "process.npz"),
        "--fast-statistics",
        str(tmp_path / "fast.npz"),
        "--prepared-manifest",
        str(tmp_path / "prepared.json"),
        "--output",
        str(tmp_path / "output.json"),
        "--catalogue-report",
        str(tmp_path / "catalogue.json"),
        "--admission-config",
        str(tmp_path / "admission.json"),
        "--calibration-report",
        str(tmp_path / "calibration.json"),
    ]
    args = parser().parse_args(base)
    assert args.protocol.name == "fresh_start_one_day_probe_v3.json"
    assert args.admission_config == tmp_path / "admission.json"
    assert args.calibration_report == tmp_path / "calibration.json"
