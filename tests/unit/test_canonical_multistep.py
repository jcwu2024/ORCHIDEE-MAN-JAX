from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    initialize_canonical_model,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
    canonical_multistep_batch_loss,
    canonical_multistep_rollout,
    canonical_pushforward_prefix,
    sequence_from_markov_window,
)
from research.daily_coarse_graining.canonical_training import (
    FastDayTargetRepresentation,
)
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    TrainingStatistics,
)


def _statistics():
    shapes = {
        "state": (2,),
        "forcing_native": (1,),
        "parameters": (1,),
        "landpoint_static": (1,),
        "annual_conditions": (1,),
        "fast_day_target": (2,),
        "state_delta": (2,),
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
    return TrainingStatistics(
        dataset_id="test",
        teacher_git_head="test",
        contract_sha256="test",
        sample_count=1,
        source_shards=("test",),
        observations_per_column={name: 1 for name in arrays},
        arrays=arrays,
    )


def _sequence(days: int):
    return CanonicalMultistepSequence(
        forcing_native=jnp.ones((days, 5, 1), dtype=jnp.float32),
        parameters=jnp.ones((days, 1), dtype=jnp.float32),
        landpoint_static=jnp.ones((days, 1), dtype=jnp.float32),
        annual_conditions=jnp.ones((days, 1), dtype=jnp.float32),
        year=jnp.full((days,), 1961, dtype=jnp.int32),
        day_index=jnp.arange(2, days + 2, dtype=jnp.int32),
        teacher_state=jnp.zeros((days, 2)),
        teacher_fast_day_target=jnp.full((days, 2), 0.25),
        teacher_next_state=jnp.full((days, 2), 0.5),
        retained_tail_inputs=jnp.arange(days, dtype=jnp.float32)[:, None],
    )


def _tail(state, discrete, fast_day, tail_inputs, year, day_index):
    del year, day_index
    next_state = state + 0.1 * fast_day + 0.001 * tail_inputs[0]
    return next_state, discrete


def test_multistep_scan_is_jittable_and_differentiates_through_recursive_state():
    config = CanonicalModelConfig(
        state_width=2,
        forcing_width=1,
        parameter_width=1,
        landpoint_static_width=1,
        annual_condition_width=1,
        fast_day_target_width=2,
        dynamic_undefined_width=1,
        state_latent_width=4,
        forcing_latent_width=3,
        condition_latent_width=3,
        hidden_width=5,
    )
    parameters = initialize_canonical_model(config, seed=3)
    representation = FastDayTargetRepresentation(
        state_indices=np.asarray([0, 1], dtype=np.int32),
        dynamic_undefined_indices=np.asarray([1], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )

    def loss(model_parameters):
        result = canonical_multistep_rollout(
            model_parameters,
            jnp.asarray([0.1, 0.2]),
            {"flag": jnp.asarray([True])},
            _sequence(3),
            statistics=_statistics(),
            representation=representation,
            fast_day_weights=jnp.asarray([0.5, 0.5]),
            retained_tail_transition=_tail,
        )
        return result.loss, result

    (loss_value, result), gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))(
        parameters
    )
    assert np.isfinite(np.asarray(loss_value))
    assert result.steps.continuous_state.shape == (3, 2)
    assert result.steps.physical_fast_day_target.shape == (3, 2)
    assert not np.array_equal(
        np.asarray(result.steps.continuous_state[0]),
        np.asarray(result.steps.continuous_state[-1]),
    )
    leaves = jax.tree_util.tree_leaves(gradients)
    assert all(np.all(np.isfinite(np.asarray(value))) for value in leaves)
    assert sum(float(jnp.sum(jnp.abs(value))) for value in leaves) > 0.0
    assert np.all(np.asarray(result.steps.state_increment_loss) == 0.0)
    np.testing.assert_allclose(
        result.loss,
        jnp.mean(
            result.steps.fast_day_loss
            + 0.1 * result.steps.undefined_loss
            + result.steps.state_loss
        ),
    )


def test_multistep_scan_supports_one_three_and_seven_day_curriculum():
    config = CanonicalModelConfig(
        state_width=2,
        forcing_width=1,
        parameter_width=1,
        landpoint_static_width=1,
        annual_condition_width=1,
        fast_day_target_width=2,
        dynamic_undefined_width=1,
        state_latent_width=2,
        forcing_latent_width=2,
        condition_latent_width=2,
        hidden_width=3,
    )
    parameters = initialize_canonical_model(config, seed=4)
    representation = FastDayTargetRepresentation(
        state_indices=np.asarray([0, 1], dtype=np.int32),
        dynamic_undefined_indices=np.asarray([1], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )
    run = jax.jit(
        lambda sequence: canonical_multistep_rollout(
            parameters,
            jnp.asarray([0.1, 0.2]),
            {"flag": jnp.asarray([True])},
            sequence,
            statistics=_statistics(),
            representation=representation,
            fast_day_weights=jnp.asarray([0.5, 0.5]),
            retained_tail_transition=_tail,
        )
    )
    for days in (1, 3, 7):
        result = run(_sequence(days))
        assert result.steps.continuous_state.shape == (days, 2)
        assert np.isfinite(np.asarray(result.loss))


def test_pushforward_prefix_detaches_terminal_state_from_parameter_gradients():
    config = CanonicalModelConfig(2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3)
    parameters = initialize_canonical_model(config, seed=5)
    representation = FastDayTargetRepresentation(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1.0e20]),
    )

    def terminal_sum(model_parameters):
        terminal = canonical_pushforward_prefix(
            model_parameters,
            jnp.asarray([0.1, 0.2]),
            {"flag": jnp.asarray([True])},
            _sequence(3),
            statistics=_statistics(),
            representation=representation,
            retained_tail_transition=_tail,
        )
        return jnp.sum(terminal.continuous_state)

    value, gradients = jax.jit(jax.value_and_grad(terminal_sum))(parameters)

    assert np.isfinite(np.asarray(value))
    assert all(
        np.all(np.asarray(gradient) == 0.0)
        for gradient in jax.tree_util.tree_leaves(gradients)
    )


def test_process_increment_objective_is_finite_and_adds_daily_change_loss():
    config = CanonicalModelConfig(
        state_width=2,
        forcing_width=1,
        parameter_width=1,
        landpoint_static_width=1,
        annual_condition_width=1,
        fast_day_target_width=2,
        dynamic_undefined_width=1,
        state_latent_width=2,
        forcing_latent_width=2,
        condition_latent_width=2,
        hidden_width=3,
    )
    parameters = initialize_canonical_model(config, seed=6)
    representation = FastDayTargetRepresentation(
        state_indices=np.asarray([0, 1], dtype=np.int32),
        dynamic_undefined_indices=np.asarray([1], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )
    common = {
        "statistics": _statistics(),
        "representation": representation,
        "fast_day_weights": jnp.asarray([0.5, 0.5]),
        "retained_tail_transition": _tail,
        "state_weights": jnp.asarray([0.75, 0.25]),
    }

    baseline = canonical_multistep_rollout(
        parameters,
        jnp.asarray([0.1, 0.2]),
        {"flag": jnp.asarray([True])},
        _sequence(3),
        **common,
    )
    candidate = canonical_multistep_rollout(
        parameters,
        jnp.asarray([0.1, 0.2]),
        {"flag": jnp.asarray([True])},
        _sequence(3),
        state_delta_scale=jnp.asarray([0.1, 0.2]),
        state_increment_loss_weight=1.0,
        **common,
    )

    assert np.all(np.asarray(candidate.steps.state_increment_loss) > 0.0)
    np.testing.assert_allclose(
        candidate.loss,
        baseline.loss + jnp.mean(candidate.steps.state_increment_loss),
    )
    with np.testing.assert_raises_regex(ValueError, "state-delta scale"):
        canonical_multistep_rollout(
            parameters,
            jnp.asarray([0.1, 0.2]),
            {"flag": jnp.asarray([True])},
            _sequence(1),
            state_increment_loss_weight=1.0,
            **common,
        )


def test_process_increment_objective_masks_undefined_state_changes():
    config = CanonicalModelConfig(2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3)
    parameters = initialize_canonical_model(config, seed=9)
    representation = FastDayTargetRepresentation(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1.0e20]),
    )
    sequence = _sequence(1)._replace(
        teacher_state=jnp.asarray([[0.1, 1.0e20]]),
        teacher_next_state=jnp.asarray([[0.2, 1.0e20]]),
    )

    result = canonical_multistep_rollout(
        parameters,
        jnp.asarray([0.1, 1.0e20]),
        {"flag": jnp.asarray([True])},
        sequence,
        statistics=_statistics(),
        representation=representation,
        fast_day_weights=jnp.asarray([0.5, 0.5]),
        retained_tail_transition=_tail,
        state_weights=jnp.asarray([0.5, 0.5]),
        state_delta_scale=jnp.asarray([0.1, 0.1]),
        state_increment_loss_weight=1.0,
    )

    assert np.isfinite(np.asarray(result.loss))
    assert np.isfinite(np.asarray(result.steps.state_increment_loss)).all()


def test_multistep_batch_loss_vmaps_same_runtime_windows_and_gradients():
    config = CanonicalModelConfig(
        state_width=2,
        forcing_width=1,
        parameter_width=1,
        landpoint_static_width=1,
        annual_condition_width=1,
        fast_day_target_width=2,
        dynamic_undefined_width=1,
        state_latent_width=2,
        forcing_latent_width=2,
        condition_latent_width=2,
        hidden_width=3,
    )
    parameters = initialize_canonical_model(config, seed=8)
    representation = FastDayTargetRepresentation(
        state_indices=np.asarray([0, 1], dtype=np.int32),
        dynamic_undefined_indices=np.asarray([1], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )
    sequences = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value)),
        _sequence(3),
    )

    def loss(model_parameters):
        return canonical_multistep_batch_loss(
            model_parameters,
            jnp.asarray([[0.1, 0.2], [0.2, 0.3]]),
            {"flag": jnp.asarray([[True], [False]])},
            sequences,
            statistics=_statistics(),
            representation=representation,
            fast_day_weights=jnp.asarray([0.5, 0.5]),
            retained_tail_transition=_tail,
        )

    loss_value, gradients = jax.jit(jax.value_and_grad(loss))(parameters)
    assert np.isfinite(np.asarray(loss_value))
    assert all(
        np.all(np.isfinite(np.asarray(value)))
        for value in jax.tree_util.tree_leaves(gradients)
    )


def test_verified_markov_window_adapter_requires_matching_tail_horizon():
    window = {
        "forcing_native": np.ones((3, 5, 1)),
        "parameters": np.ones((3, 1)),
        "landpoint_static": np.ones((3, 1)),
        "annual_conditions": np.ones((3, 1)),
        "year": np.full((3,), 1962),
        "day_index": np.asarray([2, 3, 4]),
        "state_trajectory": np.ones((4, 2)),
        "teacher_fast_day_target": np.ones((3, 2)),
        "teacher_next_state": np.ones((3, 2)),
    }
    sequence = sequence_from_markov_window(
        window,
        retained_tail_inputs={"forcing": np.ones((3, 48, 1))},
    )
    assert sequence.teacher_next_state.shape == (3, 2)
    with np.testing.assert_raises_regex(ValueError, "retained-tail inputs"):
        sequence_from_markov_window(
            window,
            retained_tail_inputs={"forcing": np.ones((2, 48, 1))},
        )
