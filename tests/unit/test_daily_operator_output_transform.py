from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.neural_architecture import (
    NeuralOutputNormalization,
    NeuralOutputSchema,
    OutputFieldSpec,
)
from research.daily_coarse_graining.daily_operator.output_transform import (
    normalized_output_to_physical,
)
from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
    training_representation_outputs,
)


def _inputs(day_start: float) -> SimpleNamespace:
    return SimpleNamespace(
        canonical_input=SimpleNamespace(continuous_state=jnp.asarray([day_start], dtype=jnp.float64))
    )


def _field(constraint: str) -> OutputFieldSpec:
    return OutputFieldSpec(
        name="endpoint",
        shape=(1,),
        label_index=0,
        constraint=constraint,
        value_representation="normalized_day_end_minus_day_start_owner",
        day_start_owner="state",
        day_start_state_start=0,
        day_start_state_stop=1,
    )


def _represented(endpoint: object, inputs: object, field: OutputFieldSpec) -> object:
    schema = NeuralOutputSchema(
        fields=(field,),
        labels=("label",),
        active_pft_indices=(0,),
    )
    return training_representation_outputs({"endpoint": endpoint}, inputs, schema)["endpoint"]


def test_nonnegative_endpoint_decoder_represents_negative_delta() -> None:
    field = _field("nonnegative")
    inputs = _inputs(10.0)
    normalization = NeuralOutputNormalization(
        mean={"endpoint": np.asarray([0.0])},
        scale={"endpoint": np.asarray([1.0])},
    )
    baseline = 10.0 + np.log(-np.expm1(-10.0))
    target_raw = 4.0 + np.log(-np.expm1(-4.0)) - baseline
    endpoint = normalized_output_to_physical(
        jnp.asarray([target_raw]),
        jnp.asarray([True]),
        field=field,
        normalization=normalization,
        inputs=inputs,
    )

    np.testing.assert_allclose(endpoint, [4.0], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(_represented(endpoint, inputs, field), [-6.0], rtol=0.0, atol=1.0e-12)


def test_unit_interval_endpoint_decoder_represents_negative_delta() -> None:
    field = _field("unit_interval")
    inputs = _inputs(0.8)
    normalization = NeuralOutputNormalization(
        mean={"endpoint": np.asarray([0.0])},
        scale={"endpoint": np.asarray([1.0])},
    )
    local_slope = 0.8 * 0.2
    target_raw = ((np.log(0.2) - np.log(0.8)) - (np.log(0.8) - np.log(0.2))) * local_slope
    endpoint = normalized_output_to_physical(
        jnp.asarray([target_raw]),
        jnp.asarray([True]),
        field=field,
        normalization=normalization,
        inputs=inputs,
    )

    np.testing.assert_allclose(endpoint, [0.2], rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(_represented(endpoint, inputs, field), [-0.6], rtol=0.0, atol=1.0e-12)
