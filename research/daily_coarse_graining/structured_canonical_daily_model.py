"""Process-aware, persistently conditioned canonical daily operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalModelConfig,
    CanonicalModelParameters,
    CanonicalPrediction,
    DenseParameters,
    _dense,
    _dense_init,
    _encode_forcing_one,
    _masked_features,
    initialize_canonical_model,
)
from research.daily_coarse_graining.canonical_state_objective import (
    state_process_weighting_from_contract,
)


@dataclass(frozen=True)
class StructuredCanonicalModelSpec:
    """Static architecture and source-derived state-group definition."""

    base_config: CanonicalModelConfig
    state_group_ids: tuple[str, ...]
    state_group_indices: tuple[tuple[int, ...], ...]
    state_groups_sha256: str
    state_group_latent_width: int = 32
    parameter_latent_width: int = 32
    landpoint_static_latent_width: int = 64
    annual_calendar_latent_width: int = 16

    def identity(self) -> dict[str, Any]:
        return {
            "id": "structured_process_film_v1",
            "state_group_ids": list(self.state_group_ids),
            "state_group_widths": [len(value) for value in self.state_group_indices],
            "state_groups_sha256": self.state_groups_sha256,
            "state_group_latent_width": self.state_group_latent_width,
            "parameter_latent_width": self.parameter_latent_width,
            "landpoint_static_latent_width": self.landpoint_static_latent_width,
            "annual_calendar_latent_width": self.annual_calendar_latent_width,
            "conditioning": "residual_film_after_both_fusion_layers",
            "initialization": "zero_residual_exact_canonical_flat_v1",
        }


class StructuredCanonicalModelParameters(NamedTuple):
    """Old canonical trunk plus zero-initialized structured residual paths."""

    canonical: CanonicalModelParameters
    state_group_encoders: tuple[DenseParameters, ...]
    state_group_outputs: tuple[DenseParameters, ...]
    parameter_encoder: DenseParameters
    landpoint_static_encoder: DenseParameters
    annual_calendar_encoder: DenseParameters
    fusion_input_scale: DenseParameters
    fusion_input_shift: DenseParameters
    fusion_hidden_scale: DenseParameters
    fusion_hidden_shift: DenseParameters


def structured_spec_from_contract(
    config: CanonicalModelConfig,
    contract_metadata: Mapping[str, Any],
) -> StructuredCanonicalModelSpec:
    """Build the eight exhaustive process groups from contract provenance."""

    weighting = state_process_weighting_from_contract(contract_metadata)
    group_ids = []
    group_indices = []
    for group in weighting.metadata["groups"]:
        group_ids.append(str(group["id"]))
        indices = []
        for leaf in group["leaves"]:
            indices.extend(range(int(leaf["start"]), int(leaf["stop"])))
        group_indices.append(tuple(indices))
    spec = StructuredCanonicalModelSpec(
        base_config=config,
        state_group_ids=tuple(group_ids),
        state_group_indices=tuple(group_indices),
        state_groups_sha256=weighting.sha256,
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: StructuredCanonicalModelSpec) -> None:
    if len(spec.state_group_ids) != len(spec.state_group_indices):
        raise ValueError("structured state group IDs and indices do not match")
    if not spec.state_group_ids or len(set(spec.state_group_ids)) != len(
        spec.state_group_ids
    ):
        raise ValueError("structured state group IDs must be unique and non-empty")
    flat = tuple(index for group in spec.state_group_indices for index in group)
    if sorted(flat) != list(range(spec.base_config.state_width)):
        raise ValueError("structured state groups must cover state exactly once")
    widths = (
        spec.state_group_latent_width,
        spec.parameter_latent_width,
        spec.landpoint_static_latent_width,
        spec.annual_calendar_latent_width,
    )
    if any(width < 1 for width in widths):
        raise ValueError("structured latent widths must be positive")


def _zero_dense(input_width: int, output_width: int) -> DenseParameters:
    return DenseParameters(
        weight=jnp.zeros((input_width, output_width), dtype=jnp.float32),
        bias=jnp.zeros((output_width,), dtype=jnp.float32),
    )


def initialize_structured_canonical_model(
    spec: StructuredCanonicalModelSpec,
    *,
    seed: int,
    canonical_parameters: CanonicalModelParameters | None = None,
) -> StructuredCanonicalModelParameters:
    """Initialize residual adapters with exactly zero effect on the old model."""

    _validate_spec(spec)
    config = spec.base_config
    if canonical_parameters is None:
        canonical_parameters = initialize_canonical_model(config, seed=seed)
    keys = iter(
        jax.random.split(
            jax.random.PRNGKey(seed + 104729),
            2 * len(spec.state_group_indices) + 3,
        )
    )
    state_group_encoders = tuple(
        _dense_init(
            next(keys),
            2 * len(indices),
            spec.state_group_latent_width,
        )
        for indices in spec.state_group_indices
    )
    state_group_outputs = tuple(
        _zero_dense(spec.state_group_latent_width, config.state_latent_width)
        for _ in spec.state_group_indices
    )
    parameter_encoder = _dense_init(
        next(keys),
        2 * config.parameter_width,
        spec.parameter_latent_width,
    )
    landpoint_static_encoder = _dense_init(
        next(keys),
        2 * config.landpoint_static_width,
        spec.landpoint_static_latent_width,
    )
    annual_calendar_encoder = _dense_init(
        next(keys),
        2 * config.annual_condition_width + 4,
        spec.annual_calendar_latent_width,
    )
    condition_width = (
        spec.parameter_latent_width
        + spec.landpoint_static_latent_width
        + spec.annual_calendar_latent_width
    )
    return StructuredCanonicalModelParameters(
        canonical=canonical_parameters,
        state_group_encoders=state_group_encoders,
        state_group_outputs=state_group_outputs,
        parameter_encoder=parameter_encoder,
        landpoint_static_encoder=landpoint_static_encoder,
        annual_calendar_encoder=annual_calendar_encoder,
        fusion_input_scale=_zero_dense(condition_width, config.hidden_width),
        fusion_input_shift=_zero_dense(condition_width, config.hidden_width),
        fusion_hidden_scale=_zero_dense(condition_width, config.hidden_width),
        fusion_hidden_shift=_zero_dense(condition_width, config.hidden_width),
    )


def _structured_condition(
    parameters: StructuredCanonicalModelParameters,
    batch: CanonicalDayBatch,
):
    annual_calendar = jnp.concatenate(
        (
            _masked_features(
                batch.annual_conditions, batch.annual_conditions_finite
            ),
            jnp.asarray(batch.calendar, dtype=jnp.float32),
        ),
        axis=-1,
    )
    return jnp.concatenate(
        (
            jax.nn.silu(
                _dense(
                    _masked_features(batch.parameters, batch.parameters_finite),
                    parameters.parameter_encoder,
                )
            ),
            jax.nn.silu(
                _dense(
                    _masked_features(
                        batch.landpoint_static, batch.landpoint_static_finite
                    ),
                    parameters.landpoint_static_encoder,
                )
            ),
            jax.nn.silu(
                _dense(annual_calendar, parameters.annual_calendar_encoder)
            ),
        ),
        axis=-1,
    )


def _residual_film(hidden, condition, scale, shift):
    return hidden + hidden * _dense(condition, scale) + _dense(condition, shift)


def structured_canonical_model_apply(
    parameters: StructuredCanonicalModelParameters,
    batch: CanonicalDayBatch,
    spec: StructuredCanonicalModelSpec,
) -> CanonicalPrediction:
    """Apply process adapters and persistent condition modulation."""

    base = parameters.canonical
    state = jax.nn.silu(
        _dense(_masked_features(batch.state, batch.state_finite), base.state_encoder)
    )
    adjustments = []
    for indices, encoder, output in zip(
        spec.state_group_indices,
        parameters.state_group_encoders,
        parameters.state_group_outputs,
        strict=True,
    ):
        index = jnp.asarray(indices, dtype=jnp.int32)
        group = _masked_features(
            jnp.take(batch.state, index, axis=-1),
            jnp.take(batch.state_finite, index, axis=-1),
        )
        adjustments.append(_dense(jax.nn.silu(_dense(group, encoder)), output))
    state = state + sum(adjustments, jnp.zeros_like(state))

    forcing = jax.vmap(_encode_forcing_one, in_axes=(None, 0, 0))(
        base,
        batch.forcing_native,
        batch.forcing_finite,
    )
    flat_condition = jnp.concatenate(
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
    flat_condition = jax.nn.silu(_dense(flat_condition, base.condition_encoder))
    structured_condition = _structured_condition(parameters, batch)

    hidden = jnp.concatenate((state, forcing, flat_condition), axis=-1)
    hidden = jax.nn.silu(_dense(hidden, base.fusion_input))
    hidden = _residual_film(
        hidden,
        structured_condition,
        parameters.fusion_input_scale,
        parameters.fusion_input_shift,
    )
    hidden = hidden + jax.nn.silu(_dense(hidden, base.fusion_hidden))
    hidden = _residual_film(
        hidden,
        structured_condition,
        parameters.fusion_hidden_scale,
        parameters.fusion_hidden_shift,
    )
    return CanonicalPrediction(
        normalized_fast_day_target=(
            batch.normalized_fast_day_baseline
            + _dense(hidden, base.fast_day_target_head)
        ),
        dynamic_undefined_flip_logits=_dense(
            hidden, base.dynamic_undefined_head
        ),
    )
