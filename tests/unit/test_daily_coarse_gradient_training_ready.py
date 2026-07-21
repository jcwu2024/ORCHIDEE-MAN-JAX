from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver.orchestration import DriverCompiledHalfHourForcing
from research.daily_coarse_graining.gradient_training_ready import (
    ConditionedDayInput,
    _adam_init,
    _checkpoint_payload,
    _family_metrics,
    _load_resume,
    deterministic_boundary_fields,
    encode_day,
    initialize_model,
    model_apply,
    train_model,
)
from research.daily_coarse_graining.supervised_learnability_pilot import (
    BoundaryLeafSpec,
    BoundaryVectorSpec,
)


def _forcing(temp, rain, snow):
    steps = temp.shape[0]
    zero = np.zeros((steps, 1), dtype=np.float64)
    return DriverCompiledHalfHourForcing(
        zlev=zero,
        zlevuv=zero,
        u=zero,
        v=zero,
        qair=zero,
        temp_air=temp,
        pb=zero,
        precip_rain=rain,
        precip_snow=snow,
        lwdown=zero,
        swdown=zero,
        ccanopy=zero,
        salinity=zero,
        tide_height=zero,
    )


def _small_spec():
    leaves = (
        BoundaryLeafSpec("hydrol", "hydrol_previous_step_state", "mc", (2,), 0, 2),
        BoundaryLeafSpec("daily_interface", None, "gpp_daily", (1,), 2, 3),
    )
    return BoundaryVectorSpec(leaves, (("hydrol", 0, 2), ("daily_interface", 2, 3)), 3)


def _conditioned(parameter_shift=0.0, state_shift=0.0):
    return ConditionedDayInput(
        day_start_state=jnp.linspace(-1.0, 1.0, 70, dtype=jnp.float32) + state_shift,
        forcing_48=jnp.linspace(-0.5, 0.5, 48 * 3, dtype=jnp.float32).reshape(48, 3),
        parameters=jnp.asarray([0.2 + parameter_shift, 0.4, 0.6, 0.8], dtype=jnp.float32),
        landpoint_physics=jnp.asarray([0.1, 0.3, 0.5, 0.7, 0.9], dtype=jnp.float32),
        calendar=jnp.asarray([0.0, 1.0, 0.25, 1.0], dtype=jnp.float32),
    )


def test_forcing_owned_daily_fields_follow_teacher_source_order_exactly():
    temp = np.asarray([[270.0], [272.0], [271.0]], dtype=np.float64)
    rain = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64)
    snow = np.asarray([[0.5], [0.25], [0.0]], dtype=np.float64)
    actual = deterministic_boundary_fields(
        _forcing(temp, rain, snow), dt_sechiba=1800.0, dt_stomate=5400.0
    )
    np.testing.assert_array_equal(actual["t2m_daily"], np.asarray([271.0]))
    np.testing.assert_array_equal(actual["snowfall_daily"], np.asarray([0.25]))
    np.testing.assert_array_equal(actual["t2m_min_daily"], np.asarray([270.0]))
    np.testing.assert_array_equal(actual["t2m_max_daily"], np.asarray([272.0]))
    np.testing.assert_array_equal(actual["t2mdiag"], np.asarray([271.0]))
    np.testing.assert_array_equal(actual["precip_daily"], np.asarray([108.0]))


def test_parameter_arrays_are_dynamic_model_inputs_and_change_output():
    spec = _small_spec()
    inputs = _conditioned()
    parameters = initialize_model(inputs, spec, seed=8)
    baseline = model_apply(parameters, inputs)
    changed_input = _conditioned(parameter_shift=0.125)
    changed = model_apply(parameters, changed_input)
    assert not np.array_equal(
        np.asarray(encode_day(parameters.encoder, inputs)),
        np.asarray(encode_day(parameters.encoder, changed_input)),
    )
    assert not np.array_equal(np.asarray(baseline), np.asarray(changed))
    jaxpr = str(jax.make_jaxpr(model_apply)(parameters, inputs))
    assert "tanh" in jaxpr


def test_pure_jax_adam_updates_encoder_and_decoder_and_reduces_loss():
    spec = _small_spec()
    inputs = (_conditioned(state_shift=0.0), _conditioned(state_shift=0.2))
    targets = np.asarray([[0.25, -0.5, 1.0], [-0.75, 0.4, -0.2]], dtype=np.float32)
    result = train_model(
        inputs,
        targets,
        np.ones_like(targets, dtype=bool),
        spec,
        {"steps": 80, "log_every": 20, "learning_rate": 0.01},
        seed=4,
    )
    assert result["initial_gradient_norm"] > 0.0
    assert result["encoder_parameter_change_norm"] > 0.0
    assert result["decoder_parameter_change_norm"] > 0.0
    assert result["final_loss"] < result["initial_loss"] * 0.05
    metrics = _family_metrics(
        result["predictions"], targets, np.ones_like(targets, dtype=bool), spec
    )
    assert metrics["overall"]["normalized_rmse"] < 0.1


def test_checkpoint_resume_contract_roundtrips_parameter_and_optimizer_pytrees(tmp_path):
    spec = _small_spec()
    parameters = initialize_model(_conditioned(), spec, seed=9)
    optimizer = _adam_init(parameters)
    path = tmp_path / "checkpoint.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            _checkpoint_payload(parameters, optimizer, None, spec, {"steps": 1}),
            handle,
        )
    resumed = _load_resume(path, spec)
    assert resumed["teacher_commit"].startswith("7333b46")
    assert resumed["learned_total_size"] == spec.total_size
    assert len(jax.tree_util.tree_leaves(resumed["parameters"])) == len(
        jax.tree_util.tree_leaves(parameters)
    )
