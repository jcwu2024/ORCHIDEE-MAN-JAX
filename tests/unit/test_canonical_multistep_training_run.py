from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    initialize_canonical_model,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
)
from research.daily_coarse_graining.canonical_multistep_training_run import (
    _make_train_step,
    parse_curriculum,
)
from research.daily_coarse_graining.canonical_training import (
    FastDayTargetRepresentation,
)
from research.daily_coarse_graining.canonical_training_run import _adam_init
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


def _tail(state, discrete, fast_day, forcing, year, day_index):
    del forcing, year, day_index
    return state + 0.1 * fast_day, discrete


def test_curriculum_parser_requires_unique_increasing_supported_horizons():
    assert parse_curriculum("1:8,3:4,7:2") == (
        parse_curriculum("1:8")[0],
        parse_curriculum("3:4")[0],
        parse_curriculum("7:2")[0],
    )
    with pytest.raises(ValueError, match="unique and increasing"):
        parse_curriculum("3:1,1:1")
    with pytest.raises(ValueError, match="1, 3, or 7"):
        parse_curriculum("2:1")


def test_compiled_multistep_train_step_updates_parameters_with_finite_gradient():
    config = CanonicalModelConfig(2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3)
    parameters = initialize_canonical_model(config, seed=2)
    optimizer = _adam_init(parameters)
    representation = FastDayTargetRepresentation(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1.0e20]),
    )
    batch = 2
    horizon = 3
    sequence = CanonicalMultistepSequence(
        forcing_native=jnp.ones((batch, horizon, 5, 1)),
        parameters=jnp.ones((batch, horizon, 1)),
        landpoint_static=jnp.ones((batch, horizon, 1)),
        annual_conditions=jnp.ones((batch, horizon, 1)),
        year=jnp.full((batch, horizon), 1962),
        day_index=jnp.tile(jnp.arange(2, 5), (batch, 1)),
        teacher_fast_day_target=jnp.full((batch, horizon, 2), 0.2),
        teacher_next_state=jnp.full((batch, horizon, 2), 0.4),
        retained_tail_inputs=jnp.ones((batch, horizon, 1)),
    )
    step = _make_train_step(
        statistics=_statistics(),
        representation=representation,
        weights=jnp.asarray([0.5, 0.5]),
        transition=_tail,
        state_loss_weight=1.0,
        undefined_loss_weight=0.1,
    )
    updated, _, loss, gradient_norm = step(
        parameters,
        optimizer,
        jnp.asarray([[0.1, 0.2], [0.2, 0.3]]),
        {"flag": jnp.asarray([[True], [False]])},
        sequence,
        jnp.asarray(1.0e-3),
    )
    assert np.isfinite(np.asarray(loss))
    assert float(gradient_norm) > 0.0
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(parameters),
            jax.tree_util.tree_leaves(updated),
            strict=True,
        )
    )
