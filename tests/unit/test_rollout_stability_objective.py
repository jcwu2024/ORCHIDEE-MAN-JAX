from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    canonical_model_apply,
    initialize_canonical_model,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
    canonical_multistep_rollout,
)
from research.daily_coarse_graining.canonical_retained_tail import (
    CanonicalRetainedTailDayInputs,
    canonical_retained_tail_day_inputs,
)
from research.daily_coarse_graining.canonical_state_objective import (
    stabilized_state_delta_scale,
    state_process_weighting_from_contract,
)
from research.daily_coarse_graining.canonical_training import (
    FastDayTargetRepresentation,
    prepare_canonical_batch,
)
from research.daily_coarse_graining.canonical_training_run import _adam_init
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    TrainingStatistics,
)
from research.daily_coarse_graining.rollout_stability_objective import (
    SCIENCE_FIELDS,
    canonical_fast_day_anchor_loss,
    rollout_stability_candidate_loss,
    rollout_stability_components,
    rollout_stability_loss,
    science_objective_layout_from_contract,
)
from research.daily_coarse_graining.rollout_stability_run import (
    build_arm_checkpoint,
    make_candidate_update_step,
    make_coefficient_calibration_step,
    make_control_update_step,
    verify_arm_checkpoint,
)


def _contract_metadata():
    fields = [
        ("slowproc_stomate_previous_step_state", "gpp_daily"),
        ("slowproc_stomate_previous_step_state", "npp_daily"),
        ("slowproc_stomate_previous_step_state", "resp_growth"),
        ("slowproc_stomate_previous_step_state", "resp_maint"),
        ("slowproc_stomate_previous_step_state", "resp_hetero"),
        ("slowproc_stomate_previous_step_state", "biomass"),
        ("slowproc_stomate_previous_step_state", "lai"),
        ("slowproc_stomate_previous_step_state", "height"),
        ("slowproc_stomate_previous_step_state", "litter"),
        ("slowproc_stomate_previous_step_state", "carbon_32l"),
        ("slowproc_stomate_previous_step_state", "DOC"),
        ("slowproc_stomate_previous_step_state", "deepC_peat"),
        ("slowproc_stomate_previous_step_state", "herbivores"),
        ("hydrol_previous_step_state", "soil_moisture"),
        ("thermosoil_previous_step_state", "soil_temperature"),
        ("diffuco_previous_step_state", "rveget"),
    ]
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


def _statistics(width: int):
    shapes = {
        "state": (width,),
        "forcing_native": (1,),
        "parameters": (1,),
        "landpoint_static": (1,),
        "annual_conditions": (1,),
        "fast_day_target": (width,),
        "state_delta": (width,),
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


def _sequence(batch: int, horizon: int, width: int):
    single = CanonicalMultistepSequence(
        forcing_native=jnp.ones((horizon, 5, 1), dtype=jnp.float32),
        parameters=jnp.ones((horizon, 1), dtype=jnp.float32),
        landpoint_static=jnp.ones((horizon, 1), dtype=jnp.float32),
        annual_conditions=jnp.ones((horizon, 1), dtype=jnp.float32),
        year=jnp.full((horizon,), 1962, dtype=jnp.int32),
        day_index=jnp.arange(2, horizon + 2, dtype=jnp.int32),
        teacher_state=jnp.full((horizon, width), 0.1),
        teacher_fast_day_target=jnp.full((horizon, width), 0.2),
        teacher_next_state=jnp.full((horizon, width), 0.3),
        retained_tail_inputs=jnp.ones((horizon, 1), dtype=jnp.float32),
    )
    return jax.tree_util.tree_map(
        lambda value: jnp.stack([value] * batch),
        single,
    )


def _tail(state, discrete, fast_day, inputs, year, day_index):
    del year, day_index
    return state + 0.1 * fast_day + 0.001 * inputs[0], discrete


def _setup(horizon: int):
    metadata = _contract_metadata()
    width = metadata["continuous_state_width"]
    config = CanonicalModelConfig(
        width,
        1,
        1,
        1,
        1,
        width,
        1,
        8,
        4,
        4,
        12,
    )
    parameters = initialize_canonical_model(config, seed=19)
    representation = FastDayTargetRepresentation(
        state_indices=np.arange(width, dtype=np.int32),
        dynamic_undefined_indices=np.asarray([0], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )
    statistics = _statistics(width)
    scale, _ = stabilized_state_delta_scale(statistics, floor_ratio=0.01)
    return {
        "parameters": parameters,
        "initial_states": jnp.full((2, width), 0.1),
        "initial_discrete_states": {"flag": jnp.asarray([[True], [False]])},
        "sequences": _sequence(2, horizon, width),
        "statistics": statistics,
        "representation": representation,
        "fast_day_weights": jnp.full((width,), 1.0 / width),
        "retained_tail_transition": _tail,
        "process_weighting": state_process_weighting_from_contract(metadata),
        "state_delta_scale": scale,
        "science_layout": science_objective_layout_from_contract(metadata),
        "teacher_next_discrete_states": {
            "flag": jnp.broadcast_to(
                jnp.asarray([[True], [False]])[:, None, :],
                (2, horizon, 1),
            )
        },
    }


def _anchor_batch(setup):
    sequence = setup["sequences"]
    return prepare_canonical_batch(
        {
            "state": np.asarray(setup["initial_states"]),
            "forcing_native": np.asarray(sequence.forcing_native[:, 0]),
            "parameters": np.asarray(sequence.parameters[:, 0]),
            "landpoint_static": np.asarray(sequence.landpoint_static[:, 0]),
            "annual_conditions": np.asarray(sequence.annual_conditions[:, 0]),
            "year": np.asarray(sequence.year[:, 0]),
            "day_index": np.asarray(sequence.day_index[:, 0]),
            "fast_day_target": np.asarray(
                sequence.teacher_fast_day_target[:, 0]
            ),
        },
        setup["statistics"],
        setup["representation"],
    )


def test_science_layout_is_complete_nonoverlapping_and_hash_stable():
    layout = science_objective_layout_from_contract(_contract_metadata())

    assert layout.field_masks.shape == (len(SCIENCE_FIELDS), 16)
    assert np.all(layout.field_masks.sum(axis=1) == 1)
    assert np.count_nonzero(layout.nonnegative_state_mask) == 3
    assert len(layout.sha256) == 64

    missing = _contract_metadata()
    missing["state_leaves"] = missing["state_leaves"][:-5]
    with pytest.raises(ValueError, match="missing science fields"):
        science_objective_layout_from_contract(missing)


def test_rollout_stability_components_are_finite_differentiable_and_separate():
    setup = _setup(3)

    def objective(parameters):
        result = rollout_stability_loss(
            parameters,
            coefficients={
                "L_next": 1.0,
                "L_rollout": 1.0,
                "L_bias": 0.25,
                "L_science": 0.5,
            },
            **{name: value for name, value in setup.items() if name != "parameters"},
        )
        return result.loss, result.components

    (loss, components), gradients = jax.jit(
        jax.value_and_grad(objective, has_aux=True)
    )(setup["parameters"])

    assert np.isfinite(np.asarray(loss))
    for name in ("L_fast", "L_next", "L_rollout", "L_bias", "L_science"):
        assert np.isfinite(np.asarray(getattr(components, name)))
    assert float(components.L_rollout) > 0.0
    assert int(components.discrete_state_mismatches) == 0
    assert int(components.nonfinite_defined_values) == 0
    assert all(
        np.all(np.isfinite(np.asarray(value)))
        for value in jax.tree_util.tree_leaves(gradients)
    )


def test_one_day_rollout_component_is_exactly_zero():
    setup = _setup(1)
    components = rollout_stability_components(**setup)

    assert float(components.L_rollout) == 0.0
    assert float(components.L_next) > 0.0


def test_candidate_uses_matched_anchor_without_double_counting_rollout_fast_loss():
    setup = _setup(3)
    anchor_batch = _anchor_batch(setup)
    anchor = canonical_fast_day_anchor_loss(
        setup["parameters"],
        anchor_batch,
        fast_day_weights=setup["fast_day_weights"],
    )
    result = rollout_stability_candidate_loss(
        setup["parameters"],
        anchor_batch,
        setup["initial_states"],
        setup["initial_discrete_states"],
        setup["sequences"],
        coefficients={
            "L_next": 0.0,
            "L_rollout": 0.0,
            "L_bias": 0.0,
            "L_science": 0.0,
        },
        statistics=setup["statistics"],
        representation=setup["representation"],
        fast_day_weights=setup["fast_day_weights"],
        retained_tail_transition=setup["retained_tail_transition"],
        process_weighting=setup["process_weighting"],
        state_delta_scale=setup["state_delta_scale"],
        science_layout=setup["science_layout"],
        teacher_next_discrete_states=setup["teacher_next_discrete_states"],
    )

    np.testing.assert_allclose(result.loss, anchor, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.components.L_fast,
        anchor,
        rtol=0.0,
        atol=0.0,
    )


def _candidate_kwargs(setup):
    return {
        "statistics": setup["statistics"],
        "representation": setup["representation"],
        "retained_tail_transition": setup["retained_tail_transition"],
        "process_weighting": setup["process_weighting"],
        "state_delta_scale": setup["state_delta_scale"],
        "science_layout": setup["science_layout"],
    }


def test_zero_treatment_candidate_matches_control_optimizer_update_exactly():
    setup = _setup(3)
    anchor_batch = _anchor_batch(setup)
    optimizer = _adam_init(setup["parameters"])
    control = make_control_update_step(
        fast_day_weights=setup["fast_day_weights"],
        undefined_loss_weight=0.1,
        model_apply=canonical_model_apply,
    )
    candidate = make_candidate_update_step(
        coefficients={
            "L_next": 0.0,
            "L_rollout": 0.0,
            "L_bias": 0.0,
            "L_science": 0.0,
        },
        fast_day_weights=setup["fast_day_weights"],
        undefined_loss_weight=0.1,
        rematerialize=False,
        model_apply=canonical_model_apply,
        **_candidate_kwargs(setup),
    )
    control_result = control(
        setup["parameters"],
        optimizer,
        anchor_batch,
        jnp.asarray(3.0e-5),
    )
    candidate_result = candidate(
        setup["parameters"],
        optimizer,
        anchor_batch,
        setup["initial_states"],
        setup["initial_discrete_states"],
        setup["sequences"],
        setup["teacher_next_discrete_states"],
        jnp.asarray(3.0e-5),
    )

    assert bool(control_result.update_applied)
    assert bool(candidate_result.update_applied)
    np.testing.assert_allclose(
        control_result.loss,
        candidate_result.loss,
        rtol=0.0,
        atol=0.0,
    )
    for control_leaf, candidate_leaf in zip(
        jax.tree_util.tree_leaves(
            (control_result.parameters, control_result.optimizer)
        ),
        jax.tree_util.tree_leaves(
            (candidate_result.parameters, candidate_result.optimizer)
        ),
        strict=True,
    ):
        np.testing.assert_allclose(
            control_leaf,
            candidate_leaf,
            rtol=0.0,
            atol=0.0,
        )


def test_candidate_hard_constraint_failure_preserves_parameters_and_optimizer():
    setup = _setup(1)
    anchor_batch = _anchor_batch(setup)
    optimizer = _adam_init(setup["parameters"])
    candidate = make_candidate_update_step(
        coefficients={
            "L_next": 1.0,
            "L_rollout": 1.0,
            "L_bias": 0.25,
            "L_science": 0.5,
        },
        fast_day_weights=setup["fast_day_weights"],
        undefined_loss_weight=0.1,
        rematerialize=False,
        model_apply=canonical_model_apply,
        **_candidate_kwargs(setup),
    )
    mismatched_discrete = {
        "flag": ~setup["teacher_next_discrete_states"]["flag"]
    }
    result = candidate(
        setup["parameters"],
        optimizer,
        anchor_batch,
        setup["initial_states"],
        setup["initial_discrete_states"],
        setup["sequences"],
        mismatched_discrete,
        jnp.asarray(3.0e-5),
    )

    assert not bool(result.update_applied)
    assert int(result.components.discrete_state_mismatches) > 0
    for before, after in zip(
        jax.tree_util.tree_leaves((setup["parameters"], optimizer)),
        jax.tree_util.tree_leaves((result.parameters, result.optimizer)),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


def test_paired_gradient_calibration_reports_zero_one_day_rollout_gradient():
    setup = _setup(1)
    calibration = make_coefficient_calibration_step(
        fast_day_weights=setup["fast_day_weights"],
        undefined_loss_weight=0.1,
        rematerialize=False,
        model_apply=canonical_model_apply,
        **_candidate_kwargs(setup),
    )
    values, gradient_norms, components = calibration(
        setup["parameters"],
        _anchor_batch(setup),
        setup["initial_states"],
        setup["initial_discrete_states"],
        setup["sequences"],
        setup["teacher_next_discrete_states"],
    )

    assert values.shape == (5,)
    assert gradient_norms.shape == (5,)
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.all(np.isfinite(np.asarray(gradient_norms)))
    assert float(values[2]) == 0.0
    assert float(gradient_norms[2]) == 0.0
    assert int(components.discrete_state_mismatches) == 0


def test_control_checkpoint_resume_is_exactly_identical_to_uninterrupted_updates():
    setup = _setup(1)
    step = make_control_update_step(
        fast_day_weights=setup["fast_day_weights"],
        undefined_loss_weight=0.1,
        model_apply=canonical_model_apply,
    )
    anchor = _anchor_batch(setup)
    rate = jnp.asarray(3.0e-5)
    initial_optimizer = _adam_init(setup["parameters"])

    def advance(parameters, optimizer, count):
        for _ in range(count):
            result = step(parameters, optimizer, anchor, rate)
            parameters, optimizer = result.parameters, result.optimizer
        return parameters, optimizer

    uninterrupted = advance(setup["parameters"], initial_optimizer, 4)
    halfway = advance(setup["parameters"], initial_optimizer, 2)
    schedule = {
        "canonical_sha256": "schedule",
        "horizon_counts": {"1": 4},
        "entries": [
            {"update": index, "reference_index": 0, "horizon": 1}
            for index in range(4)
        ],
    }
    identity = {"experiment": "resume"}
    checkpoint = build_arm_checkpoint(
        arm_id="one_step_continuation_control",
        identity=identity,
        parameters=halfway[0],
        optimizer=halfway[1],
        next_update=2,
        schedule=schedule,
        history=[],
    )
    verify_arm_checkpoint(
        checkpoint,
        arm_id="one_step_continuation_control",
        identity=identity,
        schedule=schedule,
    )
    resumed = advance(
        jax.tree_util.tree_map(jnp.asarray, checkpoint["parameters"]),
        jax.tree_util.tree_map(jnp.asarray, checkpoint["optimizer"]),
        2,
    )

    for expected, actual in zip(
        jax.tree_util.tree_leaves(uninterrupted),
        jax.tree_util.tree_leaves(resumed),
        strict=True,
    ):
        np.testing.assert_array_equal(expected, actual)


def test_multistep_rematerialization_preserves_value_and_gradient():
    setup = _setup(3)

    def loss(parameters, rematerialize):
        sequence = jax.tree_util.tree_map(lambda value: value[0], setup["sequences"])
        result = canonical_multistep_rollout(
            parameters,
            setup["initial_states"][0],
            {"flag": setup["initial_discrete_states"]["flag"][0]},
            sequence,
            statistics=setup["statistics"],
            representation=setup["representation"],
            fast_day_weights=setup["fast_day_weights"],
            retained_tail_transition=_tail,
            rematerialize=rematerialize,
        )
        return result.loss

    plain_value, plain_gradient = jax.value_and_grad(loss)(setup["parameters"], False)
    remat_value, remat_gradient = jax.value_and_grad(loss)(setup["parameters"], True)

    np.testing.assert_allclose(plain_value, remat_value, rtol=0.0, atol=0.0)
    for plain, remat in zip(
        jax.tree_util.tree_leaves(plain_gradient),
        jax.tree_util.tree_leaves(remat_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(plain, remat, rtol=0.0, atol=0.0)


def test_retained_tail_day_inputs_excludes_trace_static_controls():
    static = {
        "metadata": SimpleNamespace(
            pref_soil_veg=np.asarray([1.0]),
            lalo=np.asarray([[2.0, 3.0]]),
        ),
        "stomate_parameter_values": {"value": np.asarray([2.0])},
        "hydrol_table_arrays": SimpleNamespace(value=np.asarray([3.0])),
        "landpoint_payload": {"value": np.asarray([4.0])},
        "stomate_restart_template": SimpleNamespace(value=np.asarray([5.0])),
        "stomate_season_values": {"value": np.asarray([6.0])},
        "diffuco_parameter_values": SimpleNamespace(value=np.asarray([7.0])),
        "mineral_imin": 1,
        "mineral_imax": 2,
        "daily_carbon_dispatch": {"static": True},
        "season": SimpleNamespace(provenance=("source",)),
    }

    inputs = canonical_retained_tail_day_inputs(np.asarray([8.0]), static)

    assert isinstance(inputs, CanonicalRetainedTailDayInputs)
    assert np.array_equal(inputs.compiled_forcing, np.asarray([8.0]))
    assert np.array_equal(inputs.metadata.pref_soil_veg, np.asarray([1.0]))
    assert not hasattr(inputs, "daily_carbon_dispatch")
    assert not hasattr(inputs, "mineral_imin")
