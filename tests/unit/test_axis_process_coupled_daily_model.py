from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalModelConfig,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
    canonical_multistep_rollout,
)
from research.daily_coarse_graining.canonical_training import (
    FastDayTargetRepresentation,
)
from research.daily_coarse_graining.daily_model_architecture import (
    AXIS_PROCESS_COUPLED_V1,
    build_daily_model_definition,
    verify_checkpoint_architecture,
)
from research.daily_coarse_graining.daily_process_axis_layout import (
    process_axis_layout_from_contract,
)
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    TrainingStatistics,
)


def _leaf(component, field, index, *, axes=("npts",)):
    return {
        "component": component,
        "path": [field],
        "shape": [1],
        "start": index,
        "stop": index + 1,
        "axis_names": list(axes),
    }


def _target(family, component, field, index, *, axes=("npts",)):
    return {
        "family": family,
        "component": component,
        "path": [field],
        "shape": [1],
        "start": index,
        "stop": index + 1,
        "axis_names": list(axes),
    }


def _contract_metadata():
    return {
        "continuous_state_width": 8,
        "fast_day_target_width": 8,
        "state_leaves": [
            _leaf("slowproc_stomate_previous_step_state", "gpp_daily", 0),
            _leaf("slowproc_stomate_previous_step_state", "biomass", 1),
            _leaf("slowproc_stomate_previous_step_state", "litter", 2),
            _leaf("slowproc_stomate_previous_step_state", "carbon", 3),
            _leaf("slowproc_stomate_previous_step_state", "herbivores", 4),
            _leaf("hydrol_previous_step_state", "soil_moisture", 5),
            _leaf("thermosoil_previous_step_state", "soil_temperature", 6),
            _leaf("diffuco_previous_step_state", "rveget", 7),
        ],
        "fast_day_target_leaves": [
            _target("driver", "driver_previous_step_state", "albedo", 0),
            _target(
                "diffuco_enerbil",
                "diffuco_previous_step_state",
                "gpp",
                1,
            ),
            _target("hydrol", "hydrol_previous_step_state", "mc", 2),
            _target(
                "thermosoil",
                "thermosoil_previous_step_state",
                "ptn",
                3,
            ),
            _target("daily_interface", None, "gpp_daily", 4),
            _target("ok_leak", None, "carbon_32l", 5, axes=("ncarb",)),
            _target("final_diagnostics", None, "t2mdiag", 6, axes=()),
            _target(
                "sechiba_finalize",
                "sechiba_finalize_state",
                "leaf_ci",
                7,
                axes=("nlai",),
            ),
        ],
    }


def _config():
    return CanonicalModelConfig(
        state_width=8,
        forcing_width=2,
        parameter_width=4,
        landpoint_static_width=5,
        annual_condition_width=1,
        fast_day_target_width=8,
        dynamic_undefined_width=2,
        state_latent_width=7,
        forcing_latent_width=5,
        condition_latent_width=4,
        hidden_width=9,
    )


def _batch(*, mask_first=False, masked_state_value=0.0, parameter_shift=0.0):
    batch_size = 3
    state = jnp.arange(batch_size * 8, dtype=jnp.float32).reshape(batch_size, 8)
    state = state.at[:, 0].set(masked_state_value)
    state_finite = jnp.ones_like(state, dtype=bool)
    if mask_first:
        state_finite = state_finite.at[:, 0].set(False)
    forcing = jnp.arange(
        batch_size * 5 * 2, dtype=jnp.float32
    ).reshape(batch_size, 5, 2)
    return CanonicalDayBatch(
        state=state / 20.0,
        state_finite=state_finite,
        normalized_fast_day_baseline=jnp.zeros((batch_size, 8)),
        forcing_native=forcing / 10.0,
        forcing_finite=jnp.ones_like(forcing, dtype=bool),
        parameters=jnp.ones((batch_size, 4)) + parameter_shift,
        parameters_finite=jnp.ones((batch_size, 4), dtype=bool),
        landpoint_static=jnp.ones((batch_size, 5)),
        landpoint_static_finite=jnp.ones((batch_size, 5), dtype=bool),
        annual_conditions=jnp.ones((batch_size, 1)),
        annual_conditions_finite=jnp.ones((batch_size, 1), dtype=bool),
        calendar=jnp.zeros((batch_size, 4)),
    )


def _statistics():
    shapes = {
        "state": (8,),
        "forcing_native": (2,),
        "parameters": (4,),
        "landpoint_static": (5,),
        "annual_conditions": (1,),
        "fast_day_target": (8,),
    }
    arrays = {
        name: FiniteColumnStatistics(
            count=np.ones(shape, dtype=np.uint64),
            mean=np.zeros(shape),
            variance=np.ones(shape),
            scale=np.ones(shape),
        )
        for name, shape in shapes.items()
    }
    return TrainingStatistics("test", "test", "test", 1, ("test",), {}, arrays)


def test_process_axis_layout_is_exhaustive_stable_and_owner_bound():
    layout = process_axis_layout_from_contract(_contract_metadata())
    repeated = process_axis_layout_from_contract(_contract_metadata())

    assert layout.sha256 == repeated.sha256
    assert [len(group.state_indices) for group in layout.process_groups] == [1] * 8
    assert [len(family.target_indices) for family in layout.target_families] == [
        1
    ] * 8
    assert layout.target_families[4].source_process_ids == (
        "stomate_carbon_flux",
    )
    assert layout.target_families[5].source_process_ids == (
        "stomate_carbon_storage",
    )


def test_process_axis_layout_rejects_unknown_or_overlapping_contract_axes():
    unknown = _contract_metadata()
    unknown["state_leaves"][0]["axis_names"] = ["new_axis"]
    with pytest.raises(ValueError, match="unknown axes"):
        process_axis_layout_from_contract(unknown)

    overlap = _contract_metadata()
    overlap["state_leaves"][1]["start"] = 0
    with pytest.raises(ValueError, match="overlapping state"):
        process_axis_layout_from_contract(overlap)


def test_axis_process_model_jits_masks_inputs_and_uses_conditions():
    definition = build_daily_model_definition(
        AXIS_PROCESS_COUPLED_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = definition.initialize(seed=17)
    baseline = jax.jit(definition.apply)(
        parameters, _batch(mask_first=True, masked_state_value=1.0)
    )
    masked_change = jax.jit(definition.apply)(
        parameters, _batch(mask_first=True, masked_state_value=1.0e12)
    )
    condition_change = jax.jit(definition.apply)(
        parameters,
        _batch(
            mask_first=True,
            masked_state_value=1.0,
            parameter_shift=0.5,
        ),
    )

    assert baseline.normalized_fast_day_target.shape == (3, 8)
    assert baseline.dynamic_undefined_flip_logits.shape == (3, 2)
    np.testing.assert_array_equal(
        baseline.normalized_fast_day_target,
        masked_change.normalized_fast_day_target,
    )
    assert not np.array_equal(
        np.asarray(baseline.normalized_fast_day_target),
        np.asarray(condition_change.normalized_fast_day_target),
    )


def test_axis_process_model_has_finite_gradients_for_every_structured_path():
    definition = build_daily_model_definition(
        AXIS_PROCESS_COUPLED_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = definition.initialize(seed=19)
    gradients = jax.jit(
        jax.grad(
            lambda value: jnp.sum(
                definition.apply(value, _batch()).normalized_fast_day_target
            )
        )
    )(parameters)

    assert all(
        np.all(np.isfinite(np.asarray(value)))
        for value in jax.tree_util.tree_leaves(gradients)
    )
    assert all(
        np.any(np.asarray(encoder.weight) != 0.0)
        for group in gradients.state_axis_encoders
        for encoder in group
    )
    assert all(
        np.any(np.asarray(head.weight) != 0.0)
        for head in gradients.target_family_outputs
    )


def test_axis_process_checkpoint_identity_and_parameter_roundtrip():
    definition = build_daily_model_definition(
        AXIS_PROCESS_COUPLED_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = definition.initialize(seed=23)
    identity = {"model_architecture": definition.identity()}
    verify_checkpoint_architecture(identity, definition)

    drifted = {"model_architecture": dict(definition.identity())}
    drifted["model_architecture"]["process_axis_layout_sha256"] = "bad"
    with pytest.raises(ValueError, match="drifted"):
        verify_checkpoint_architecture(drifted, definition)

    restored = pickle.loads(pickle.dumps(parameters))
    expected = definition.apply(parameters, _batch())
    observed = definition.apply(restored, _batch())
    np.testing.assert_array_equal(
        expected.normalized_fast_day_target,
        observed.normalized_fast_day_target,
    )


def test_axis_process_model_has_no_hidden_cross_day_restart_state():
    definition = build_daily_model_definition(
        AXIS_PROCESS_COUPLED_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = definition.initialize(seed=29)
    batch = _batch()

    first = definition.apply(parameters, batch)
    restarted = definition.apply(pickle.loads(pickle.dumps(parameters)), batch)

    np.testing.assert_array_equal(
        first.normalized_fast_day_target,
        restarted.normalized_fast_day_target,
    )
    assert definition.identity()["cross_day_memory"] == "canonical_state_only"


def test_axis_process_model_jits_and_differentiates_through_multiday_scan():
    definition = build_daily_model_definition(
        AXIS_PROCESS_COUPLED_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = definition.initialize(seed=31)
    days = 3
    sequence = CanonicalMultistepSequence(
        forcing_native=jnp.ones((days, 5, 2)),
        parameters=jnp.ones((days, 4)),
        landpoint_static=jnp.ones((days, 5)),
        annual_conditions=jnp.ones((days, 1)),
        year=jnp.full((days,), 1962),
        day_index=jnp.arange(2, 2 + days),
        teacher_state=jnp.zeros((days, 8)),
        teacher_fast_day_target=jnp.full((days, 8), 0.2),
        teacher_next_state=jnp.full((days, 8), 0.4),
        retained_tail_inputs=jnp.zeros((days, 1)),
    )
    representation = FastDayTargetRepresentation(
        np.arange(8, dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1.0e20, 1.0e20]),
    )

    def tail(state, discrete, fast_day, forcing, year, day_index):
        del forcing, year, day_index
        return state + 0.1 * fast_day, discrete

    def loss(value):
        return canonical_multistep_rollout(
            value,
            jnp.zeros((8,)),
            {"flag": jnp.asarray([True])},
            sequence,
            statistics=_statistics(),
            representation=representation,
            fast_day_weights=jnp.full((8,), 1.0 / 8.0),
            retained_tail_transition=tail,
            model_apply=definition.apply,
        ).loss

    value, gradients = jax.jit(jax.value_and_grad(loss))(parameters)

    assert np.isfinite(np.asarray(value))
    assert all(
        np.all(np.isfinite(np.asarray(gradient)))
        for gradient in jax.tree_util.tree_leaves(gradients)
    )
