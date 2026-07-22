"""Pure-JAX neural operator for canonical daily Markov Teacher shards."""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np


class CanonicalModelConfig(NamedTuple):
    state_width: int
    forcing_width: int
    parameter_width: int
    landpoint_static_width: int
    annual_condition_width: int
    diagnostic_width: int
    state_latent_width: int = 128
    forcing_latent_width: int = 64
    condition_latent_width: int = 64
    hidden_width: int = 256


class CanonicalDayBatch(NamedTuple):
    """Normalized numerical input with explicit finite masks and no point ID."""

    state: Any
    state_finite: Any
    forcing_native: Any
    forcing_finite: Any
    parameters: Any
    parameters_finite: Any
    landpoint_static: Any
    landpoint_static_finite: Any
    annual_conditions: Any
    annual_conditions_finite: Any
    calendar: Any


class DenseParameters(NamedTuple):
    weight: Any
    bias: Any


class CanonicalModelParameters(NamedTuple):
    state_encoder: DenseParameters
    forcing_input: DenseParameters
    forcing_recurrent: DenseParameters
    condition_encoder: DenseParameters
    fusion_input: DenseParameters
    fusion_hidden: DenseParameters
    state_delta_head: DenseParameters
    diagnostic_head: DenseParameters


class CanonicalPrediction(NamedTuple):
    normalized_state_delta: Any
    normalized_diagnostics: Any


def _dense_init(key, input_width: int, output_width: int, *, scale: float = 1.0):
    limit = scale * np.sqrt(6.0 / (input_width + output_width))
    weight = jax.random.uniform(
        key,
        (input_width, output_width),
        minval=-limit,
        maxval=limit,
        dtype=jnp.float32,
    )
    return DenseParameters(weight=weight, bias=jnp.zeros((output_width,), dtype=jnp.float32))


def initialize_canonical_model(
    config: CanonicalModelConfig,
    *,
    seed: int,
) -> CanonicalModelParameters:
    """Initialize a persistence-centered residual model."""

    _validate_config(config)
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 8))
    condition_width = (
        2
        * (
            config.parameter_width
            + config.landpoint_static_width
            + config.annual_condition_width
        )
        + 4
    )
    fused_width = (
        config.state_latent_width
        + config.forcing_latent_width
        + config.condition_latent_width
    )
    return CanonicalModelParameters(
        state_encoder=_dense_init(
            next(keys), 2 * config.state_width, config.state_latent_width
        ),
        forcing_input=_dense_init(
            next(keys), 2 * config.forcing_width + 2, config.forcing_latent_width
        ),
        forcing_recurrent=_dense_init(
            next(keys), config.forcing_latent_width, config.forcing_latent_width
        ),
        condition_encoder=_dense_init(
            next(keys), condition_width, config.condition_latent_width
        ),
        fusion_input=_dense_init(next(keys), fused_width, config.hidden_width),
        fusion_hidden=_dense_init(
            next(keys), config.hidden_width, config.hidden_width
        ),
        state_delta_head=_dense_init(
            next(keys), config.hidden_width, config.state_width, scale=1.0e-2
        ),
        diagnostic_head=_dense_init(
            next(keys), config.hidden_width, config.diagnostic_width, scale=1.0e-2
        ),
    )


def _validate_config(config: CanonicalModelConfig) -> None:
    widths = tuple(int(value) for value in config)
    if any(value < 1 for value in widths):
        raise ValueError("canonical model widths must all be positive")


def _dense(value, parameters: DenseParameters):
    return value @ parameters.weight + parameters.bias


def _masked_features(value, finite):
    finite = jnp.asarray(finite, dtype=bool)
    safe = jnp.where(finite, jnp.asarray(value, dtype=jnp.float32), 0.0)
    return jnp.concatenate((safe, finite.astype(jnp.float32)), axis=-1)


def _forcing_position(length: int):
    position = jnp.arange(length, dtype=jnp.float32)
    denominator = jnp.maximum(jnp.asarray(length - 1, dtype=jnp.float32), 1.0)
    phase = 2.0 * jnp.pi * position / denominator
    return jnp.stack((jnp.sin(phase), jnp.cos(phase)), axis=-1)


def _encode_forcing_one(
    parameters: CanonicalModelParameters,
    forcing_native,
    forcing_finite,
):
    records = _masked_features(forcing_native, forcing_finite)
    records = jnp.concatenate((records, _forcing_position(records.shape[0])), axis=-1)

    def step(carry, record):
        hidden = jnp.tanh(
            _dense(record, parameters.forcing_input)
            + _dense(carry, parameters.forcing_recurrent)
        )
        return hidden, hidden

    initial = jnp.zeros_like(parameters.forcing_recurrent.bias)
    final, sequence = jax.lax.scan(step, initial, records)
    return 0.5 * (final + jnp.mean(sequence, axis=0))


def canonical_model_apply(
    parameters: CanonicalModelParameters,
    batch: CanonicalDayBatch,
) -> CanonicalPrediction:
    """Predict normalized state tendencies and same-day diagnostics."""

    state = jax.nn.silu(
        _dense(
            _masked_features(batch.state, batch.state_finite),
            parameters.state_encoder,
        )
    )
    forcing = jax.vmap(_encode_forcing_one, in_axes=(None, 0, 0))(
        parameters,
        batch.forcing_native,
        batch.forcing_finite,
    )
    condition = jnp.concatenate(
        (
            _masked_features(batch.parameters, batch.parameters_finite),
            _masked_features(
                batch.landpoint_static, batch.landpoint_static_finite
            ),
            _masked_features(
                batch.annual_conditions, batch.annual_conditions_finite
            ),
            jnp.asarray(batch.calendar, dtype=jnp.float32),
        ),
        axis=-1,
    )
    condition = jax.nn.silu(_dense(condition, parameters.condition_encoder))
    hidden = jnp.concatenate((state, forcing, condition), axis=-1)
    hidden = jax.nn.silu(_dense(hidden, parameters.fusion_input))
    hidden = hidden + jax.nn.silu(_dense(hidden, parameters.fusion_hidden))
    return CanonicalPrediction(
        normalized_state_delta=_dense(hidden, parameters.state_delta_head),
        normalized_diagnostics=_dense(hidden, parameters.diagnostic_head),
    )


def equal_group_weights(
    width: int,
    ranges: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Give every declared process group equal total loss weight."""

    if width < 1 or not ranges:
        raise ValueError("loss weight width and ranges must be non-empty")
    weights = np.zeros((width,), dtype=np.float32)
    occupied = np.zeros((width,), dtype=bool)
    for start, stop in ranges:
        if start < 0 or stop <= start or stop > width or np.any(occupied[start:stop]):
            raise ValueError(f"invalid or overlapping loss group [{start}, {stop})")
        weights[start:stop] = 1.0 / (len(ranges) * (stop - start))
        occupied[start:stop] = True
    if not np.all(occupied):
        raise ValueError("loss groups must cover every output element exactly once")
    return weights


def masked_huber_loss(prediction, target, finite, weights, *, delta: float = 1.0):
    if delta <= 0.0:
        raise ValueError("Huber delta must be positive")
    finite = jnp.asarray(finite, dtype=bool)
    error = jnp.where(finite, prediction - target, 0.0)
    absolute = jnp.abs(error)
    element = jnp.where(
        absolute <= delta,
        0.5 * error * error,
        delta * (absolute - 0.5 * delta),
    )
    weighted_mask = finite.astype(element.dtype) * jnp.asarray(weights)[None, :]
    denominator = jnp.maximum(jnp.sum(weighted_mask, axis=1), 1.0e-12)
    return jnp.mean(jnp.sum(element * weighted_mask, axis=1) / denominator)


def canonical_one_step_loss(
    parameters: CanonicalModelParameters,
    batch: CanonicalDayBatch,
    *,
    normalized_state_delta,
    state_delta_finite,
    normalized_diagnostics,
    diagnostics_finite,
    state_weights,
    diagnostic_weights,
    diagnostic_loss_weight: float = 0.25,
):
    prediction = canonical_model_apply(parameters, batch)
    state_loss = masked_huber_loss(
        prediction.normalized_state_delta,
        normalized_state_delta,
        state_delta_finite,
        state_weights,
    )
    diagnostic_loss = masked_huber_loss(
        prediction.normalized_diagnostics,
        normalized_diagnostics,
        diagnostics_finite,
        diagnostic_weights,
    )
    return state_loss + diagnostic_loss_weight * diagnostic_loss


def parameter_count(parameters: CanonicalModelParameters) -> int:
    return int(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(parameters)))
