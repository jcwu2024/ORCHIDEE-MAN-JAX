from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalModelConfig,
    canonical_model_apply,
    initialize_canonical_model,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
    canonical_multistep_rollout,
)
from research.daily_coarse_graining.canonical_training import (
    FastDayTargetRepresentation,
)
from research.daily_coarse_graining.daily_model_architecture import (
    CANONICAL_FLAT_V1,
    STRUCTURED_PROCESS_FILM_V1,
    build_daily_model_definition,
    checkpoint_architecture_id,
    verify_checkpoint_architecture,
)
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    TrainingStatistics,
)


def _contract_metadata():
    fields = (
        ("slowproc_stomate_previous_step_state", "gpp_daily"),
        ("slowproc_stomate_previous_step_state", "biomass"),
        ("slowproc_stomate_previous_step_state", "litter"),
        ("slowproc_stomate_previous_step_state", "DOC"),
        ("slowproc_stomate_previous_step_state", "herbivores"),
        ("hydrol_previous_step_state", "soil_moisture"),
        ("thermosoil_previous_step_state", "soil_temperature"),
        ("diffuco_previous_step_state", "rveget"),
    )
    return {
        "continuous_state_width": len(fields),
        "state_leaves": [
            {
                "component": component,
                "path": [field],
                "start": index,
                "stop": index + 1,
            }
            for index, (component, field) in enumerate(fields)
        ],
    }


def _config():
    return CanonicalModelConfig(
        state_width=8,
        forcing_width=2,
        parameter_width=4,
        landpoint_static_width=5,
        annual_condition_width=1,
        fast_day_target_width=6,
        dynamic_undefined_width=2,
        state_latent_width=7,
        forcing_latent_width=5,
        condition_latent_width=4,
        hidden_width=9,
    )


def _batch(*, parameter_shift=0.0):
    batch_size = 3
    state = jnp.arange(batch_size * 8, dtype=jnp.float32).reshape(batch_size, 8)
    forcing = jnp.arange(
        batch_size * 5 * 2, dtype=jnp.float32
    ).reshape(batch_size, 5, 2)
    return CanonicalDayBatch(
        state=state / 20.0,
        state_finite=jnp.ones_like(state, dtype=bool),
        normalized_fast_day_baseline=jnp.zeros((batch_size, 6)),
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
        "fast_day_target": (6,),
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


def test_zero_initialized_structured_candidate_exactly_preserves_flat_baseline():
    config = _config()
    canonical = initialize_canonical_model(config, seed=5)
    definition = build_daily_model_definition(
        STRUCTURED_PROCESS_FILM_V1,
        config,
        _contract_metadata(),
    )
    structured = definition.initialize(seed=9, canonical_parameters=canonical)

    baseline = jax.jit(canonical_model_apply)(canonical, _batch())
    candidate = jax.jit(definition.apply)(structured, _batch())

    np.testing.assert_array_equal(
        candidate.normalized_fast_day_target,
        baseline.normalized_fast_day_target,
    )
    np.testing.assert_array_equal(
        candidate.dynamic_undefined_flip_logits,
        baseline.dynamic_undefined_flip_logits,
    )


def test_structured_residual_paths_receive_finite_nonzero_gradients():
    definition = build_daily_model_definition(
        STRUCTURED_PROCESS_FILM_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = definition.initialize(seed=11)

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
    assert any(
        np.any(np.asarray(output.weight) != 0.0)
        for output in gradients.state_group_outputs
    )
    assert np.any(np.asarray(gradients.fusion_input_shift.weight) != 0.0)
    assert np.any(np.asarray(gradients.fusion_hidden_shift.weight) != 0.0)


def test_trained_condition_modulation_can_respond_to_parameter_changes():
    definition = build_daily_model_definition(
        STRUCTURED_PROCESS_FILM_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = definition.initialize(seed=13)
    shift = parameters.fusion_hidden_shift._replace(
        weight=jnp.full_like(parameters.fusion_hidden_shift.weight, 1.0e-2)
    )
    parameters = parameters._replace(fusion_hidden_shift=shift)

    baseline = definition.apply(parameters, _batch())
    changed = definition.apply(parameters, _batch(parameter_shift=0.5))

    assert not np.array_equal(
        np.asarray(baseline.normalized_fast_day_target),
        np.asarray(changed.normalized_fast_day_target),
    )


def test_checkpoint_architecture_defaults_old_assets_and_rejects_drift():
    assert checkpoint_architecture_id({}) == CANONICAL_FLAT_V1
    definition = build_daily_model_definition(
        STRUCTURED_PROCESS_FILM_V1,
        _config(),
        _contract_metadata(),
    )
    identity = {"model_architecture": definition.identity()}
    verify_checkpoint_architecture(identity, definition)

    drifted = {"model_architecture": dict(definition.identity())}
    drifted["model_architecture"]["state_groups_sha256"] = "bad"
    with pytest.raises(ValueError, match="drifted"):
        verify_checkpoint_architecture(drifted, definition)


def test_structured_model_jits_and_differentiates_through_multiday_scan():
    definition = build_daily_model_definition(
        STRUCTURED_PROCESS_FILM_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = definition.initialize(seed=23)
    days = 2
    sequence = CanonicalMultistepSequence(
        forcing_native=jnp.ones((days, 5, 2)),
        parameters=jnp.ones((days, 4)),
        landpoint_static=jnp.ones((days, 5)),
        annual_conditions=jnp.ones((days, 1)),
        year=jnp.full((days,), 1962),
        day_index=jnp.arange(2, 2 + days),
        teacher_state=jnp.zeros((days, 8)),
        teacher_fast_day_target=jnp.full((days, 6), 0.2),
        teacher_next_state=jnp.full((days, 8), 0.4),
        retained_tail_inputs=jnp.zeros((days, 1)),
    )
    representation = FastDayTargetRepresentation(
        np.arange(6, dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1.0e20, 1.0e20]),
    )

    def tail(state, discrete, fast_day, forcing, year, day_index):
        del forcing, year, day_index
        return state.at[:6].add(0.1 * fast_day), discrete

    def loss(value):
        return canonical_multistep_rollout(
            value,
            jnp.zeros((8,)),
            {"flag": jnp.asarray([True])},
            sequence,
            statistics=_statistics(),
            representation=representation,
            fast_day_weights=jnp.full((6,), 1.0 / 6.0),
            retained_tail_transition=tail,
            model_apply=definition.apply,
        ).loss

    value, gradients = jax.jit(jax.value_and_grad(loss))(parameters)

    assert np.isfinite(np.asarray(value))
    assert all(
        np.all(np.isfinite(np.asarray(gradient)))
        for gradient in jax.tree_util.tree_leaves(gradients)
    )
