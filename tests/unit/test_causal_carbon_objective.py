from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalPrediction,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
)
from research.daily_coarse_graining.canonical_training import (
    CanonicalTrainingBatch,
    FastDayTargetRepresentation,
)
from research.daily_coarse_graining.causal_carbon_objective import (
    causal_carbon_candidate_loss,
    causal_carbon_interface_loss,
    causal_carbon_objective_layout_from_contract,
    causal_carbon_state_losses,
)
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    TrainingStatistics,
)


def _leaf(component, name, start):
    return {
        "component": component,
        "path": [name],
        "shape": [1, 2],
        "full_shape": [1, 14],
        "start": start,
        "stop": start + 2,
        "axis_names": ["npts", "nvm"],
        "selected_pft_indices": [0, 13],
    }


def _fast_leaf(family, name, start, shape, axes):
    stop = start + int(np.prod(shape))
    return {
        "family": family,
        "component": None,
        "path": [name],
        "shape": list(shape),
        "full_shape": list(shape),
        "start": start,
        "stop": stop,
        "axis_names": list(axes),
        "selected_pft_indices": [0, 13],
    }


def _contract():
    state_fields = (
        "npp_daily",
        "resp_growth",
        "resp_maint",
        "biomass",
        "lai",
        "carbon_32l",
        "deepC_peat",
        "litter",
        "DOC",
    )
    return {
        "continuous_state_width": 2 * len(state_fields),
        "fast_day_target_width": 14,
        "state_leaves": [
            _leaf(
                "slowproc_stomate_previous_step_state",
                name,
                2 * index,
            )
            for index, name in enumerate(state_fields)
        ],
        "fast_day_target_leaves": [
            _fast_leaf(
                "daily_interface",
                "gpp_daily",
                0,
                (1, 2),
                ("npts", "nvm"),
            ),
            _fast_leaf(
                "daily_interface",
                "resp_maint_part",
                2,
                (1, 2, 2),
                ("npts", "nvm", "nparts"),
            ),
            _fast_leaf(
                "ok_leak",
                "carbon_32l",
                6,
                (1, 2, 2),
                ("npts", "nvm", "ndeep"),
            ),
            _fast_leaf(
                "ok_leak",
                "deepC_peat",
                10,
                (1, 2, 2),
                ("npts", "ndeep", "nvm"),
            ),
        ],
    }


def _statistics(state_width):
    arrays = {
        name: FiniteColumnStatistics(
            count=np.ones(shape, dtype=np.uint64),
            mean=np.zeros(shape),
            variance=np.ones(shape),
            scale=np.ones(shape),
        )
        for name, shape in {
            "state": (state_width,),
            "forcing_native": (1,),
            "parameters": (1,),
            "landpoint_static": (1,),
            "annual_conditions": (1,),
            "fast_day_target": (14,),
        }.items()
    }
    return TrainingStatistics("test", "test", "test", 1, ("test",), {}, arrays)


def _model_input(batch_size=2):
    return CanonicalDayBatch(
        state=jnp.zeros((batch_size, 18)),
        state_finite=jnp.ones((batch_size, 18), dtype=bool),
        normalized_fast_day_baseline=jnp.zeros((batch_size, 14)),
        forcing_native=jnp.zeros((batch_size, 1, 1)),
        forcing_finite=jnp.ones((batch_size, 1, 1), dtype=bool),
        parameters=jnp.zeros((batch_size, 1)),
        parameters_finite=jnp.ones((batch_size, 1), dtype=bool),
        landpoint_static=jnp.zeros((batch_size, 1)),
        landpoint_static_finite=jnp.ones((batch_size, 1), dtype=bool),
        annual_conditions=jnp.zeros((batch_size, 1)),
        annual_conditions_finite=jnp.ones((batch_size, 1), dtype=bool),
        calendar=jnp.zeros((batch_size, 4)),
    )


def _training_batch(batch_size=2):
    target = jnp.zeros((batch_size, 14))
    return CanonicalTrainingBatch(
        model_input=_model_input(batch_size),
        normalized_fast_day_target=target,
        fast_day_target_finite=jnp.ones_like(target, dtype=bool),
        fast_day_target_undefined=jnp.zeros_like(target, dtype=bool),
        fast_day_target_undefined_values=target,
        persistent_fast_day_undefined=jnp.zeros_like(target, dtype=bool),
        persistent_fast_day_undefined_values=target,
        persistent_dynamic_undefined=jnp.zeros((batch_size, 0), dtype=bool),
        dynamic_undefined_flip_target=jnp.zeros((batch_size, 0)),
    )


def test_objective_layout_selects_pft14_and_balances_fields():
    layout = causal_carbon_objective_layout_from_contract(_contract())

    assert layout.fast_field_ids == (
        "daily_interface.gpp_daily",
        "daily_interface.resp_maint_part",
        "ok_leak.carbon_32l",
        "ok_leak.deepC_peat",
    )
    assert tuple(np.sum(layout.fast_field_masks, axis=1)) == (1, 2, 2, 2)
    np.testing.assert_allclose(
        layout.fast_field_masks @ layout.fast_weights,
        np.full((4,), 0.25),
    )
    np.testing.assert_allclose(
        layout.primary_state_field_masks @ layout.primary_state_weights,
        np.full((7,), 1.0 / 7.0),
    )
    assert layout.metadata["day_end_gpp_state_supervision"] is False


def test_interface_loss_ignores_protected_columns_and_reaches_each_field():
    layout = causal_carbon_objective_layout_from_contract(_contract())
    selected = np.flatnonzero(np.any(layout.fast_field_masks, axis=0))
    protected = np.flatnonzero(~np.any(layout.fast_field_masks, axis=0))

    def apply(parameters, batch):
        return CanonicalPrediction(
            normalized_fast_day_target=(batch.normalized_fast_day_baseline + parameters[None, :]),
            dynamic_undefined_flip_logits=jnp.zeros((batch.state.shape[0], 0)),
        )

    parameters = jnp.zeros((14,)).at[selected].set(1.0)
    parameters = parameters.at[protected].set(1.0e6)
    batch = _training_batch()
    evaluate = jax.jit(
        jax.value_and_grad(
            lambda value: causal_carbon_interface_loss(
                value,
                batch,
                layout=layout,
                model_apply=apply,
            )
        )
    )
    value, gradient = evaluate(parameters)

    np.testing.assert_allclose(value, 0.5)
    assert np.all(np.asarray(gradient)[selected] != 0.0)
    assert np.all(np.asarray(gradient)[protected] == 0.0)


def test_state_losses_separate_absolute_error_from_signed_bias():
    layout = causal_carbon_objective_layout_from_contract(_contract())
    width = _contract()["continuous_state_width"]
    teacher_state = jnp.zeros((2, 2, width))
    teacher_next = jnp.zeros((2, 2, width))
    initial = jnp.zeros((2, width))
    predicted = jnp.zeros((2, 2, width))

    npp = np.flatnonzero(layout.flux_bias_field_masks[0])
    biomass = np.flatnonzero(layout.stock_tendency_field_masks[0])
    predicted = predicted.at[0, :, npp].set(1.0)
    predicted = predicted.at[1, :, npp].set(-1.0)
    predicted = predicted.at[0, 0, biomass].set(1.0)
    predicted = predicted.at[0, 1, biomass].set(2.0)
    predicted = predicted.at[1, 0, biomass].set(-1.0)
    predicted = predicted.at[1, 1, biomass].set(-2.0)

    losses = causal_carbon_state_losses(
        predicted,
        initial,
        teacher_state,
        teacher_next,
        statistics=_statistics(width),
        state_delta_scale=jnp.ones((width,)),
        layout=layout,
    )

    assert float(losses.L_next) > 0.0
    assert float(losses.L_rollout) > 0.0
    np.testing.assert_allclose(losses.L_flux_bias, 0.0)
    np.testing.assert_allclose(losses.L_stock_tendency_bias, 0.0)
    np.testing.assert_allclose(losses.L_guard, 0.0)


def test_candidate_loss_jits_through_two_day_retained_transition():
    layout = causal_carbon_objective_layout_from_contract(_contract())
    state_width = _contract()["continuous_state_width"]
    batch_size = 2
    horizon = 2
    anchor = _training_batch(batch_size)
    initial_states = jnp.zeros((batch_size, state_width))
    initial_discrete = {"flag": jnp.zeros((batch_size, 1), dtype=bool)}
    sequence = CanonicalMultistepSequence(
        forcing_native=jnp.zeros((batch_size, horizon, 1, 1)),
        parameters=jnp.zeros((batch_size, horizon, 1)),
        landpoint_static=jnp.zeros((batch_size, horizon, 1)),
        annual_conditions=jnp.zeros((batch_size, horizon, 1)),
        year=jnp.full((batch_size, horizon), 1962),
        day_index=jnp.broadcast_to(jnp.arange(2, 4), (batch_size, horizon)),
        teacher_state=jnp.zeros((batch_size, horizon, state_width)),
        teacher_fast_day_target=jnp.zeros((batch_size, horizon, 14)),
        teacher_next_state=jnp.zeros((batch_size, horizon, state_width)),
        retained_tail_inputs=jnp.zeros((batch_size, horizon, 1)),
    )
    teacher_discrete = {
        "flag": jnp.zeros((batch_size, horizon, 1), dtype=bool)
    }
    state_indices = np.full((14,), -1, dtype=np.int32)
    state_indices[0] = 0
    representation = FastDayTargetRepresentation(
        state_indices=state_indices,
        dynamic_undefined_indices=np.asarray([0], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )
    adapted = np.flatnonzero(np.any(layout.fast_field_masks, axis=0))
    state_mask = jnp.asarray(
        np.any(layout.primary_state_field_masks, axis=0),
        dtype=jnp.float64,
    )

    def apply(parameters, batch):
        return CanonicalPrediction(
            normalized_fast_day_target=(
                batch.normalized_fast_day_baseline + parameters[None, :]
            ),
            dynamic_undefined_flip_logits=jnp.full(
                (batch.state.shape[0], 1),
                -20.0,
            ),
        )

    def transition(state, discrete, fast_day, inputs, year, day):
        del inputs, year, day
        delta = 1.0e-3 * jnp.sum(jnp.take(fast_day, adapted))
        return state + delta * state_mask, discrete

    def objective(parameters):
        result = causal_carbon_candidate_loss(
            parameters,
            anchor,
            initial_states,
            initial_discrete,
            sequence,
            coefficients={
                "L_next": 1.0,
                "L_rollout": 1.0,
                "L_flux_bias": 1.0,
                "L_stock_tendency_bias": 1.0,
            },
            layout=layout,
            statistics=_statistics(state_width),
            representation=representation,
            retained_tail_transition=transition,
            state_delta_scale=jnp.ones((state_width,)),
            teacher_next_discrete_states=teacher_discrete,
            model_apply=apply,
        )
        return result.loss, result.components

    parameters = jnp.zeros((14,)).at[adapted].set(0.1)
    (loss, components), gradient = jax.jit(
        jax.value_and_grad(objective, has_aux=True)
    )(parameters)

    assert np.isfinite(np.asarray(loss))
    assert all(
        np.isfinite(np.asarray(getattr(components, name)))
        for name in components._fields
    )
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(gradient)[adapted] != 0.0)
