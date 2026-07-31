"""Axis-partitioned, process-coupled canonical daily neural operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalModelConfig,
    CanonicalPrediction,
    DenseParameters,
    _dense,
    _dense_init,
    _encode_forcing_one,
    _masked_features,
)
from research.daily_coarse_graining.daily_process_axis_layout import (
    DailyProcessAxisLayout,
    process_axis_layout_from_contract,
)


@dataclass(frozen=True)
class AxisProcessCoupledModelSpec:
    """Static, checkpoint-bound architecture derived from Contract v5."""

    base_config: CanonicalModelConfig
    layout: DailyProcessAxisLayout
    process_latent_width: int = 88
    forcing_latent_width: int = 64
    parameter_latent_width: int = 32
    landpoint_static_latent_width: int = 64
    annual_calendar_latent_width: int = 16
    target_head_width: int = 160
    coupling_blocks: int = 2

    @property
    def condition_width(self) -> int:
        return self.parameter_latent_width + self.landpoint_static_latent_width + self.annual_calendar_latent_width

    def identity(self) -> dict[str, Any]:
        return {
            "id": "axis_process_coupled_v1",
            "process_axis_layout_sha256": self.layout.sha256,
            "process_group_ids": [group.id for group in self.layout.process_groups],
            "process_group_widths": [len(group.state_indices) for group in self.layout.process_groups],
            "axis_partition_widths": {
                group.id: {partition.encoder_kind: len(partition.indices) for partition in group.axis_partitions}
                for group in self.layout.process_groups
            },
            "target_family_widths": {family.id: len(family.target_indices) for family in self.layout.target_families},
            "target_family_sources": {
                family.id: list(family.source_process_ids) for family in self.layout.target_families
            },
            "process_latent_width": self.process_latent_width,
            "forcing_latent_width": self.forcing_latent_width,
            "parameter_latent_width": self.parameter_latent_width,
            "landpoint_static_latent_width": self.landpoint_static_latent_width,
            "annual_calendar_latent_width": self.annual_calendar_latent_width,
            "target_head_width": self.target_head_width,
            "coupling_blocks": self.coupling_blocks,
            "cross_day_memory": "canonical_state_only",
            "initialization": "from_scratch",
        }


class AxisProcessCoupledModelParameters(NamedTuple):
    state_axis_encoders: tuple[tuple[DenseParameters, ...], ...]
    process_embeddings: Any
    forcing_input: DenseParameters
    forcing_recurrent: DenseParameters
    parameter_encoder: DenseParameters
    landpoint_static_encoder: DenseParameters
    annual_calendar_encoder: DenseParameters
    condition_to_process: DenseParameters
    forcing_to_process: DenseParameters
    coupling_self: tuple[DenseParameters, ...]
    coupling_global: tuple[DenseParameters, ...]
    target_family_inputs: tuple[DenseParameters, ...]
    target_family_outputs: tuple[DenseParameters, ...]
    dynamic_undefined_head: DenseParameters


class AxisProcessFeatures(NamedTuple):
    """Reusable latent boundary before the family-specific output heads."""

    tokens: Any
    global_token: Any
    forcing: Any
    condition: Any


def axis_process_spec_from_contract(
    config: CanonicalModelConfig,
    contract_metadata: Mapping[str, Any],
) -> AxisProcessCoupledModelSpec:
    spec = AxisProcessCoupledModelSpec(
        base_config=config,
        layout=process_axis_layout_from_contract(contract_metadata),
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: AxisProcessCoupledModelSpec) -> None:
    config = spec.base_config
    layout = spec.layout
    if layout.state_width != config.state_width:
        raise ValueError("process-axis state width does not match model config")
    if layout.target_width != config.fast_day_target_width:
        raise ValueError("process-axis target width does not match model config")
    widths = (
        spec.process_latent_width,
        spec.forcing_latent_width,
        spec.parameter_latent_width,
        spec.landpoint_static_latent_width,
        spec.annual_calendar_latent_width,
        spec.target_head_width,
        spec.coupling_blocks,
    )
    if any(width < 1 for width in widths):
        raise ValueError("axis-process architecture widths must be positive")
    state_indices = [
        index for group in layout.process_groups for partition in group.axis_partitions for index in partition.indices
    ]
    target_indices = [index for family in layout.target_families for index in family.target_indices]
    if sorted(state_indices) != list(range(config.state_width)):
        raise ValueError("axis partitions must cover model state exactly once")
    if sorted(target_indices) != list(range(config.fast_day_target_width)):
        raise ValueError("target families must cover model output exactly once")


def initialize_axis_process_coupled_model(
    spec: AxisProcessCoupledModelSpec,
    *,
    seed: int,
) -> AxisProcessCoupledModelParameters:
    """Initialize the source-structured operator without a flat-model trunk."""

    _validate_spec(spec)
    config = spec.base_config
    partition_count = sum(len(group.axis_partitions) for group in spec.layout.process_groups)
    key_count = partition_count + 9 + 2 * spec.coupling_blocks + 2 * len(spec.layout.target_families)
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), key_count))
    state_axis_encoders = tuple(
        tuple(
            _dense_init(
                next(keys),
                2 * len(partition.indices),
                spec.process_latent_width,
            )
            for partition in group.axis_partitions
        )
        for group in spec.layout.process_groups
    )
    process_embeddings = 0.01 * jax.random.normal(
        next(keys),
        (len(spec.layout.process_groups), spec.process_latent_width),
        dtype=jnp.float32,
    )
    forcing_input = _dense_init(
        next(keys),
        2 * config.forcing_width + 2,
        spec.forcing_latent_width,
    )
    forcing_recurrent = _dense_init(
        next(keys),
        spec.forcing_latent_width,
        spec.forcing_latent_width,
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
    process_count = len(spec.layout.process_groups)
    condition_to_process = _dense_init(
        next(keys),
        spec.condition_width,
        process_count * spec.process_latent_width,
    )
    forcing_to_process = _dense_init(
        next(keys),
        spec.forcing_latent_width,
        process_count * spec.process_latent_width,
    )
    coupling_self = tuple(
        _dense_init(
            next(keys),
            spec.process_latent_width,
            spec.process_latent_width,
        )
        for _ in range(spec.coupling_blocks)
    )
    coupling_global = tuple(
        _dense_init(
            next(keys),
            spec.process_latent_width,
            spec.process_latent_width,
        )
        for _ in range(spec.coupling_blocks)
    )
    process_indices = {group.id: index for index, group in enumerate(spec.layout.process_groups)}
    target_family_inputs = []
    target_family_outputs = []
    for family in spec.layout.target_families:
        source_width = len(family.source_process_ids) * spec.process_latent_width
        input_width = source_width + spec.process_latent_width + spec.forcing_latent_width + spec.condition_width
        if any(source not in process_indices for source in family.source_process_ids):
            raise ValueError(f"target family {family.id} has an unknown process source")
        target_family_inputs.append(_dense_init(next(keys), input_width, spec.target_head_width))
        target_family_outputs.append(
            _dense_init(
                next(keys),
                spec.target_head_width,
                len(family.target_indices),
                scale=1.0e-2,
            )
        )
    dynamic_undefined_head = _dense_init(
        next(keys),
        spec.process_latent_width + spec.forcing_latent_width + spec.condition_width,
        config.dynamic_undefined_width,
        scale=1.0e-2,
    )
    dynamic_undefined_head = dynamic_undefined_head._replace(bias=jnp.full_like(dynamic_undefined_head.bias, -4.0))
    try:
        next(keys)
    except StopIteration:
        pass
    else:
        raise AssertionError("axis-process initializer left unused random keys")
    return AxisProcessCoupledModelParameters(
        state_axis_encoders=state_axis_encoders,
        process_embeddings=process_embeddings,
        forcing_input=forcing_input,
        forcing_recurrent=forcing_recurrent,
        parameter_encoder=parameter_encoder,
        landpoint_static_encoder=landpoint_static_encoder,
        annual_calendar_encoder=annual_calendar_encoder,
        condition_to_process=condition_to_process,
        forcing_to_process=forcing_to_process,
        coupling_self=coupling_self,
        coupling_global=coupling_global,
        target_family_inputs=tuple(target_family_inputs),
        target_family_outputs=tuple(target_family_outputs),
        dynamic_undefined_head=dynamic_undefined_head,
    )


def _condition(
    parameters: AxisProcessCoupledModelParameters,
    batch: CanonicalDayBatch,
):
    annual_calendar = jnp.concatenate(
        (
            _masked_features(batch.annual_conditions, batch.annual_conditions_finite),
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
                    _masked_features(batch.landpoint_static, batch.landpoint_static_finite),
                    parameters.landpoint_static_encoder,
                )
            ),
            jax.nn.silu(_dense(annual_calendar, parameters.annual_calendar_encoder)),
        ),
        axis=-1,
    )


def axis_process_coupled_features(
    parameters: AxisProcessCoupledModelParameters,
    batch: CanonicalDayBatch,
    spec: AxisProcessCoupledModelSpec,
) -> AxisProcessFeatures:
    """Encode one day while preserving the process-token ownership boundary."""

    process_tokens = []
    for group, encoders in zip(
        spec.layout.process_groups,
        parameters.state_axis_encoders,
        strict=True,
    ):
        axis_tokens = []
        for partition, encoder in zip(group.axis_partitions, encoders, strict=True):
            indices = jnp.asarray(partition.indices, dtype=jnp.int32)
            features = _masked_features(
                jnp.take(batch.state, indices, axis=-1),
                jnp.take(batch.state_finite, indices, axis=-1),
            )
            axis_tokens.append(jax.nn.silu(_dense(features, encoder)))
        process_tokens.append(jnp.mean(jnp.stack(axis_tokens, axis=1), axis=1))
    tokens = jnp.stack(process_tokens, axis=1)
    tokens = tokens + parameters.process_embeddings[None, :, :]

    forcing = jax.vmap(_encode_forcing_one, in_axes=(None, 0, 0))(
        parameters,
        batch.forcing_native,
        batch.forcing_finite,
    )
    condition = _condition(parameters, batch)
    process_shape = (
        tokens.shape[0],
        len(spec.layout.process_groups),
        spec.process_latent_width,
    )
    tokens = tokens + jnp.reshape(_dense(condition, parameters.condition_to_process), process_shape)
    tokens = tokens + jnp.reshape(_dense(forcing, parameters.forcing_to_process), process_shape)
    for self_parameters, global_parameters in zip(
        parameters.coupling_self,
        parameters.coupling_global,
        strict=True,
    ):
        global_token = jnp.mean(tokens, axis=1)
        update = _dense(tokens, self_parameters) + _dense(global_token, global_parameters)[:, None, :]
        tokens = tokens + jax.nn.silu(update)
    global_token = jnp.mean(tokens, axis=1)
    return AxisProcessFeatures(
        tokens=tokens,
        global_token=global_token,
        forcing=forcing,
        condition=condition,
    )


def axis_process_coupled_prediction_from_features(
    parameters: AxisProcessCoupledModelParameters,
    batch: CanonicalDayBatch,
    spec: AxisProcessCoupledModelSpec,
    features: AxisProcessFeatures,
) -> CanonicalPrediction:
    """Decode all Contract-v5 target families from one encoded day."""

    process_indices = {group.id: index for index, group in enumerate(spec.layout.process_groups)}
    tokens = features.tokens
    global_token = features.global_token
    forcing = features.forcing
    condition = features.condition
    residual = jnp.zeros_like(batch.normalized_fast_day_baseline)
    for family, input_parameters, output_parameters in zip(
        spec.layout.target_families,
        parameters.target_family_inputs,
        parameters.target_family_outputs,
        strict=True,
    ):
        source_indices = jnp.asarray(
            [process_indices[source] for source in family.source_process_ids],
            dtype=jnp.int32,
        )
        source_tokens = jnp.take(tokens, source_indices, axis=1).reshape(tokens.shape[0], -1)
        features = jnp.concatenate((source_tokens, global_token, forcing, condition), axis=-1)
        family_residual = _dense(
            jax.nn.silu(_dense(features, input_parameters)),
            output_parameters,
        )
        target_indices = jnp.asarray(family.target_indices, dtype=jnp.int32)
        residual = residual.at[:, target_indices].set(family_residual)

    undefined_features = jnp.concatenate((global_token, forcing, condition), axis=-1)
    return CanonicalPrediction(
        normalized_fast_day_target=(batch.normalized_fast_day_baseline + residual),
        dynamic_undefined_flip_logits=_dense(undefined_features, parameters.dynamic_undefined_head),
    )


def axis_process_coupled_model_apply(
    parameters: AxisProcessCoupledModelParameters,
    batch: CanonicalDayBatch,
    spec: AxisProcessCoupledModelSpec,
) -> CanonicalPrediction:
    """Predict one complete fast-day boundary from explicit Markov state."""

    features = axis_process_coupled_features(parameters, batch, spec)
    return axis_process_coupled_prediction_from_features(
        parameters,
        batch,
        spec,
        features,
    )
