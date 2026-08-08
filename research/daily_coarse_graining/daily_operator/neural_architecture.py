"""Pure-JAX architecture for the first formal neural process-head A/B.

The candidate and control deliberately share one parameter tree.  The
candidate applies attention independently within the PFT, water, carbon, and
thermal token axes and lets bounded process queries attend to four pooled
domain summaries.  The control consumes every identically shaped parameter
as token-local transforms plus masked pooling, so its parameter count is
exactly matched without introducing cross-token interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.constrained_daily_replay import (
    CARBON_INVENTORY_FIELDS,
    THERMAL_FIELDS,
    WATER_INVENTORY_FIELDS,
)

from .input_assembly import CanonicalDailyOperatorInputAssembler
from .state_features import assemble_local_pft_features
from .types import AssembledDailyOperatorInput

AXIAL_CANDIDATE = "axial_process_operator_v1"
FACTORIZED_CONTROL = "factorized_process_control_v1"
DOMAIN_NAMES = ("pft", "water", "carbon", "thermal")


@dataclass(frozen=True)
class NeuralArchitectureConfig:
    d_model: int = 192
    attention_heads: int = 6
    ffn_width: int = 768
    forcing_blocks: int = 2
    process_blocks: int = 4
    state_rank: int = 64
    decoder_width: int = 192

    def validate(self) -> None:
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.forcing_blocks != 2 or self.process_blocks != 4:
            raise ValueError("the first-candidate depth is frozen at 2 forcing and 4 process blocks")
        if self.d_model != 192 or self.attention_heads != 6 or self.ffn_width != 768:
            raise ValueError("the first-candidate width/head/FFN contract drifted")


@dataclass(frozen=True)
class OutputFieldSpec:
    name: str
    shape: tuple[int, ...]
    label_index: int
    constraint: str
    pft_axis: int | None = None
    full_pft_axis: bool = False


@dataclass(frozen=True)
class NeuralOutputSchema:
    fields: tuple[OutputFieldSpec, ...]
    labels: tuple[str, ...]
    active_pft_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.fields or not self.labels:
            raise ValueError("neural output schema cannot be empty")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("neural output field names must be unique")
        for field in self.fields:
            if field.label_index < 0 or field.label_index >= len(self.labels):
                raise ValueError(f"invalid label index for {field.name}")
            if field.constraint not in {"signed", "nonnegative", "unit_interval", "signed_unit"}:
                raise ValueError(f"invalid output constraint for {field.name}")
            if field.pft_axis is not None and not 0 <= field.pft_axis < len(field.shape):
                raise ValueError(f"invalid PFT axis for {field.name}")


@dataclass(frozen=True, eq=False)
class AxisDomainLayout:
    name: str
    pool_matrix: np.ndarray
    field_index: np.ndarray
    role_index: np.ndarray
    coordinate: np.ndarray
    pft_index: np.ndarray
    element_pft_index: np.ndarray

    def __post_init__(self) -> None:
        tokens = self.pool_matrix.shape[0]
        if self.pool_matrix.ndim != 2 or any(
            value.shape != (tokens,)
            for value in (
                self.field_index,
                self.role_index,
                self.coordinate,
                self.pft_index,
            )
        ):
            raise ValueError(f"invalid semantic axis layout for {self.name}")
        if self.element_pft_index.shape != (self.pool_matrix.shape[1],):
            raise ValueError(f"invalid semantic element/PFT layout for {self.name}")


@dataclass(frozen=True, eq=False)
class ProcessAxisLayout:
    domains: tuple[AxisDomainLayout, AxisDomainLayout, AxisDomainLayout]
    field_names: tuple[str, ...]
    role_names: tuple[str, ...]


def _state_leaf(assembler: CanonicalDailyOperatorInputAssembler, component: str, field: str) -> Any:
    matches = tuple(
        leaf
        for leaf in assembler.contract.state_leaves
        if leaf.component == component and leaf.path == (field,)
    )
    if len(matches) != 1:
        raise ValueError(f"semantic axis layout requires one {component}.{field} leaf")
    return matches[0]


def build_process_axis_layout(
    assembler: CanonicalDailyOperatorInputAssembler,
) -> ProcessAxisLayout:
    """Build scientific-axis pooling from frozen Markov leaf metadata."""

    segment_specs: dict[str, list[tuple[str, tuple[int, ...], tuple[str, ...], bool]]] = {
        "water": [],
        "carbon": [],
        "thermal": [],
    }
    for field in WATER_INVENTORY_FIELDS:
        leaf = _state_leaf(assembler, "hydrol_previous_step_state", field)
        names = (f"{field}.storage", f"{field}.debt") if field == "qsintveg" else (field,)
        segment_specs["water"].extend(
            (name, tuple(leaf.shape), tuple(leaf.axis_names), False) for name in names
        )
    for field in CARBON_INVENTORY_FIELDS:
        leaf = _state_leaf(assembler, "slowproc_stomate_previous_step_state", field)
        segment_specs["carbon"].append(
            (field, tuple(leaf.shape), tuple(leaf.axis_names), False)
        )
    for component, field, _bound in THERMAL_FIELDS:
        key = f"{component}.{field}"
        if key not in assembler.thermal_keys:
            continue
        leaf = _state_leaf(assembler, component, field)
        segment_specs["thermal"].append(
            (key, tuple(leaf.full_shape), tuple(leaf.axis_names), True)
        )

    role_names = ["field"]
    for specs in segment_specs.values():
        for _field, _shape, axes, _full in specs:
            for axis in axes:
                if axis != "npts" and axis not in role_names:
                    role_names.append(axis)
    role_index = {name: index for index, name in enumerate(role_names)}
    field_names = []
    domains = []
    active_full = tuple(int(index) for index in assembler.active_pft_indices)
    for domain_name in ("water", "carbon", "thermal"):
        specs = segment_specs[domain_name]
        total = sum(int(np.prod(shape)) for _name, shape, _axes, _full in specs)
        rows = []
        fields = []
        roles = []
        coordinates = []
        pft_indices = []
        element_pft_indices = np.full((total,), -1, dtype=np.int32)
        offset = 0
        for field_name, shape, axes, full_pft in specs:
            field_id = len(field_names)
            field_names.append(f"{domain_name}.{field_name}")
            size = int(np.prod(shape))
            grid = np.indices(shape, dtype=np.int32).reshape((len(shape), -1)).T
            base_valid = np.ones((size,), dtype=np.bool_)
            if "nvm" in axes and full_pft:
                pft_axis = axes.index("nvm")
                base_valid &= np.isin(grid[:, pft_axis], np.asarray(active_full))
            if "nvm" in axes:
                pft_axis = axes.index("nvm")
                if full_pft:
                    for compact, coordinate in enumerate(active_full):
                        selected = grid[:, pft_axis] == coordinate
                        element_pft_indices[offset : offset + size][selected] = compact
                else:
                    element_pft_indices[offset : offset + size] = grid[:, pft_axis]

            def add_row(local: np.ndarray, role: str, coordinate: int, pft_index: int) -> None:
                row = np.zeros((total,), dtype=np.bool_)
                row[offset : offset + size] = local & base_valid
                rows.append(row)
                fields.append(field_id)
                roles.append(role_index[role])
                coordinates.append(coordinate)
                pft_indices.append(pft_index)

            add_row(np.ones((size,), dtype=np.bool_), "field", 0, -1)
            for axis, (axis_name, extent) in enumerate(zip(axes, shape, strict=True)):
                if axis_name == "npts" or extent <= 1:
                    continue
                if axis_name == "nvm":
                    values = active_full if full_pft else tuple(range(extent))
                    for compact, coordinate in enumerate(values):
                        add_row(grid[:, axis] == coordinate, axis_name, 0, compact)
                else:
                    for coordinate in range(extent):
                        add_row(grid[:, axis] == coordinate, axis_name, coordinate, -1)
            offset += size
        layout = AxisDomainLayout(
            name=domain_name,
            pool_matrix=np.stack(rows),
            field_index=np.asarray(fields, dtype=np.int32),
            role_index=np.asarray(roles, dtype=np.int32),
            coordinate=np.asarray(coordinates, dtype=np.int32),
            pft_index=np.asarray(pft_indices, dtype=np.int32),
            element_pft_index=element_pft_indices,
        )
        if layout.pool_matrix.shape[1] != total:
            raise ValueError(f"{domain_name} semantic layout width drift")
        domains.append(layout)
    return ProcessAxisLayout(
        domains=tuple(domains),
        field_names=tuple(field_names),
        role_names=tuple(role_names),
    )


class NeuralInputNormalization(NamedTuple):
    continuous_mean: Any
    continuous_scale: Any
    forcing_mean: Any
    forcing_scale: Any
    static_mean: Any
    static_scale: Any
    annual_mean: Any
    annual_scale: Any
    water_scale: Any
    carbon_scale: Any
    thermal_scale: Any


class NeuralOutputNormalization(NamedTuple):
    mean: Mapping[str, Any]
    scale: Mapping[str, Any]


class NeuralProcessContext(NamedTuple):
    forcing_tokens: Any
    forcing_mask: Any
    domain_tokens: tuple[Any, Any, Any, Any]
    domain_masks: tuple[Any, Any, Any, Any]
    global_token: Any


def unit_input_normalization(inputs: AssembledDailyOperatorInput) -> NeuralInputNormalization:
    """Identity normalization for plumbing tests, never formal training."""

    return NeuralInputNormalization(
        continuous_mean=np.zeros_like(np.asarray(inputs.state.other), dtype=np.float64),
        continuous_scale=np.ones_like(np.asarray(inputs.state.other), dtype=np.float64),
        forcing_mean=np.zeros((np.asarray(inputs.forcing.values).shape[-1],), dtype=np.float64),
        forcing_scale=np.ones((np.asarray(inputs.forcing.values).shape[-1],), dtype=np.float64),
        static_mean=np.zeros_like(np.asarray(inputs.static_conditions), dtype=np.float64),
        static_scale=np.ones_like(np.asarray(inputs.static_conditions), dtype=np.float64),
        annual_mean=np.zeros_like(np.asarray(inputs.annual_conditions), dtype=np.float64),
        annual_scale=np.ones_like(np.asarray(inputs.annual_conditions), dtype=np.float64),
        water_scale=np.ones_like(np.asarray(inputs.state.water), dtype=np.float64),
        carbon_scale=np.ones_like(np.asarray(inputs.state.carbon), dtype=np.float64),
        thermal_scale=np.ones_like(np.asarray(inputs.state.thermal), dtype=np.float64),
    )


def unit_output_normalization(schema: NeuralOutputSchema) -> NeuralOutputNormalization:
    """Identity output scales for plumbing tests, never formal training."""

    return NeuralOutputNormalization(
        mean={field.name: np.zeros(field.shape, dtype=np.float64) for field in schema.fields},
        scale={field.name: np.ones(field.shape, dtype=np.float64) for field in schema.fields},
    )


def _normal(key: Any, shape: tuple[int, ...], *, fan_in: int) -> Any:
    return jax.random.normal(key, shape, dtype=jnp.float64) / np.sqrt(float(max(fan_in, 1)))


def _linear_parameters(key: Any, in_width: int, out_width: int) -> dict[str, Any]:
    kernel_key, _ = jax.random.split(key)
    return {
        "kernel": _normal(kernel_key, (in_width, out_width), fan_in=in_width),
        "bias": jnp.zeros((out_width,), dtype=jnp.float64),
    }


def _attention_parameters(key: Any, width: int) -> dict[str, Any]:
    keys = jax.random.split(key, 4)
    return {
        name: _linear_parameters(item, width, width)
        for name, item in zip(("query", "key", "value", "output"), keys, strict=True)
    }


def _ffn_parameters(key: Any, width: int, hidden: int) -> dict[str, Any]:
    first, second = jax.random.split(key)
    return {
        "first": _linear_parameters(first, width, hidden),
        "second": _linear_parameters(second, hidden, width),
    }


def _block_parameters(key: Any, config: NeuralArchitectureConfig) -> dict[str, Any]:
    axis_key, cross_key, axis_ffn_key, query_ffn_key = jax.random.split(key, 4)
    return {
        "axis_attention": _attention_parameters(axis_key, config.d_model),
        "query_attention": _attention_parameters(cross_key, config.d_model),
        "axis_ffn": _ffn_parameters(axis_ffn_key, config.d_model, config.ffn_width),
        "query_ffn": _ffn_parameters(query_ffn_key, config.d_model, config.ffn_width),
        "axis_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float64),
        "axis_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float64),
        "query_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float64),
        "query_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float64),
    }


def initialize_neural_process_parameters(
    key: Any,
    *,
    schema: NeuralOutputSchema,
    axis_layout: ProcessAxisLayout,
    inputs: AssembledDailyOperatorInput,
    config: NeuralArchitectureConfig = NeuralArchitectureConfig(),
) -> Mapping[str, Any]:
    """Initialize the shared candidate/control tree without historical weights."""

    config.validate()
    keys = iter(jax.random.split(key, 17 + config.forcing_blocks + config.process_blocks))
    forcing_width = int(np.asarray(inputs.forcing.values).shape[-1]) + 4
    pft_width = int(np.asarray(assemble_local_pft_features(inputs.pft)).shape[-1])
    return {
        "forcing_input": _linear_parameters(next(keys), forcing_width, config.d_model),
        "pft_input": _linear_parameters(next(keys), pft_width, config.d_model),
        "scalar_input": _linear_parameters(next(keys), 1, config.d_model),
        "state_rank": _linear_parameters(next(keys), 4, config.state_rank),
        "static_rank": _linear_parameters(next(keys), 4, config.state_rank),
        "annual_rank": _linear_parameters(next(keys), 4, config.state_rank),
        "global_input": _linear_parameters(next(keys), 3 * config.state_rank, config.d_model),
        "domain_embedding": _normal(next(keys), (len(DOMAIN_NAMES), config.d_model), fan_in=config.d_model),
        "input_field_embedding": _normal(
            next(keys),
            (len(axis_layout.field_names), config.d_model),
            fan_in=config.d_model,
        ),
        "input_role_embedding": _normal(
            next(keys),
            (len(axis_layout.role_names), config.d_model),
            fan_in=config.d_model,
        ),
        "input_position_embedding": _normal(next(keys), (32, config.d_model), fan_in=config.d_model),
        "process_queries": _normal(next(keys), (len(schema.labels), config.d_model), fan_in=config.d_model),
        "field_embedding": _normal(next(keys), (len(schema.fields), config.d_model), fan_in=config.d_model),
        "axis_ordinal_embedding": _normal(next(keys), (6, config.d_model), fan_in=config.d_model),
        "output_position_embedding": _normal(next(keys), (32, config.d_model), fan_in=config.d_model),
        "forcing_blocks": tuple(
            _block_parameters(next(keys), config) for _ in range(config.forcing_blocks)
        ),
        "process_blocks": tuple(
            _block_parameters(next(keys), config) for _ in range(config.process_blocks)
        ),
        "decoder_first": _linear_parameters(next(keys), config.d_model, config.decoder_width),
        "decoder_second": _linear_parameters(next(keys), config.decoder_width, 1),
    }


def parameter_count(parameters: Mapping[str, Any]) -> int:
    return int(sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(parameters)))


def _linear(parameters: Mapping[str, Any], value: Any) -> Any:
    return jnp.asarray(value, dtype=jnp.float64) @ parameters["kernel"] + parameters["bias"]


def _layer_norm(value: Any, scale: Any, bias: Any) -> Any:
    value = jnp.asarray(value, dtype=jnp.float64)
    mean = jnp.mean(value, axis=-1, keepdims=True)
    variance = jnp.mean((value - mean) ** 2, axis=-1, keepdims=True)
    return (value - mean) * jax.lax.rsqrt(variance + 1.0e-8) * scale + bias


def _ffn(parameters: Mapping[str, Any], value: Any) -> Any:
    return _linear(parameters["second"], jax.nn.gelu(_linear(parameters["first"], value)))


def _attention(
    parameters: Mapping[str, Any],
    query: Any,
    key_value: Any,
    *,
    heads: int,
    attention_mask: Any,
) -> Any:
    query = _linear(parameters["query"], query)
    key = _linear(parameters["key"], key_value)
    value = _linear(parameters["value"], key_value)
    width = int(query.shape[-1])
    head_width = width // heads
    query = query.reshape(query.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    key = key.reshape(key.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    value = value.reshape(value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
    logits = jnp.einsum("...hqd,...hkd->...hqk", query, key) / np.sqrt(float(head_width))
    mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
    logits = jnp.where(mask[..., None, :, :], logits, -jnp.inf)
    row_valid = jnp.any(mask, axis=-1, keepdims=True)
    safe_logits = jnp.where(row_valid[..., None, :, :], logits, 0.0)
    weights = jax.nn.softmax(safe_logits, axis=-1)
    weights = jnp.where(mask[..., None, :, :], weights, 0.0)
    result = jnp.einsum("...hqk,...hkd->...hqd", weights, value)
    result = result.swapaxes(-3, -2).reshape(result.shape[:-3] + (result.shape[-2], width))
    return _linear(parameters["output"], result)


def _candidate_axis_update(parameters: Mapping[str, Any], tokens: Any, mask: Any, config: Any) -> Any:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    pair_mask = mask[:, None] & mask[None, :]
    attended = _attention(
        parameters["axis_attention"],
        tokens,
        tokens,
        heads=config.attention_heads,
        attention_mask=pair_mask,
    )
    updated = tokens + attended
    normalized = _layer_norm(
        updated,
        parameters["axis_norm_scale"],
        parameters["axis_norm_bias"],
    )
    return jnp.where(mask[:, None], normalized + _ffn(parameters["axis_ffn"], normalized), 0.0)


def _control_axis_update(parameters: Mapping[str, Any], tokens: Any, mask: Any) -> Any:
    attention = parameters["axis_attention"]
    local = (
        jax.nn.gelu(_linear(attention["query"], tokens))
        + jax.nn.gelu(_linear(attention["key"], tokens))
        + jax.nn.gelu(_linear(attention["value"], tokens))
    ) / 3.0
    local = _linear(attention["output"], local)
    updated = tokens + local
    normalized = _layer_norm(updated, parameters["axis_norm_scale"], parameters["axis_norm_bias"])
    return jnp.where(mask[:, None], normalized + _ffn(parameters["axis_ffn"], normalized), 0.0)


def _masked_mean(value: Any, mask: Any) -> Any:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    count = jnp.maximum(jnp.sum(mask, dtype=jnp.float64), 1.0)
    return jnp.sum(jnp.where(mask[:, None], value, 0.0), axis=0) / count


def _standardize(value: Any, mean: Any, scale: Any) -> Any:
    value = jnp.asarray(value, dtype=jnp.float64)
    mean = jnp.asarray(mean, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    valid = jnp.isfinite(value) & (jnp.abs(value) < 0.5e20)
    safe_value = jnp.where(valid, value, mean)
    standardized = (safe_value - mean) / jnp.where(
        jnp.isfinite(scale) & (scale > 0.0),
        scale,
        1.0,
    )
    return jnp.where(valid & jnp.isfinite(standardized), standardized, 0.0)


def _symmetric_summary(value: Any, defined: Any | None = None) -> Any:
    value = jnp.asarray(value, dtype=jnp.float64).reshape(-1)
    valid = jnp.isfinite(value) & (jnp.abs(value) < 0.5e20)
    if defined is not None:
        valid &= jnp.asarray(defined, dtype=jnp.bool_).reshape(-1)
    safe = jnp.where(valid, value, 0.0)
    count = jnp.maximum(jnp.sum(valid, dtype=jnp.float64), 1.0)
    mean = jnp.sum(safe, dtype=jnp.float64) / count
    variance = jnp.sum(jnp.where(valid, (safe - mean) ** 2, 0.0), dtype=jnp.float64) / count
    minimum = jnp.min(jnp.where(valid, value, jnp.inf))
    maximum = jnp.max(jnp.where(valid, value, -jnp.inf))
    any_valid = jnp.any(valid)
    return jnp.stack(
        (
            mean,
            jnp.sqrt(jnp.maximum(variance, 0.0)),
            jnp.where(any_valid, minimum, 0.0),
            jnp.where(any_valid, maximum, 0.0),
        )
    )


def _semantic_domain_tokens(
    parameters: Mapping[str, Any],
    layout: AxisDomainLayout,
    values: Any,
    defined: Any,
    pft_tokens: Any,
    pft_mask: Any,
    *,
    domain_index: int,
) -> tuple[Any, Any]:
    values = jnp.asarray(values, dtype=jnp.float64).reshape(-1)
    defined = jnp.asarray(defined, dtype=jnp.bool_).reshape(-1)
    pft_mask = jnp.asarray(pft_mask, dtype=jnp.bool_)
    element_pft = jnp.asarray(layout.element_pft_index, dtype=jnp.int32)
    element_has_pft = element_pft >= 0
    element_active = (~element_has_pft) | pft_mask[jnp.maximum(element_pft, 0)]
    defined &= element_active
    pool = jnp.asarray(layout.pool_matrix, dtype=jnp.bool_) & defined[None, :]
    count = jnp.sum(pool, axis=1, dtype=jnp.float64)
    token_mask = count > 0.0
    safe_values = jnp.where(defined, values, 0.0)
    pooled = jnp.sum(jnp.where(pool, safe_values[None, :], 0.0), axis=1) / jnp.maximum(count, 1.0)
    tokens = _linear(parameters["scalar_input"], pooled[:, None])
    tokens += parameters["domain_embedding"][domain_index]
    tokens += parameters["input_field_embedding"][jnp.asarray(layout.field_index)]
    tokens += parameters["input_role_embedding"][jnp.asarray(layout.role_index)]
    pft_index = jnp.asarray(layout.pft_index, dtype=jnp.int32)
    is_pft = pft_index >= 0
    token_mask &= (~is_pft) | pft_mask[jnp.maximum(pft_index, 0)]
    safe_pft_index = jnp.maximum(pft_index, 0)
    tokens += jnp.where(is_pft[:, None], pft_tokens[safe_pft_index], 0.0)
    coordinate = jnp.asarray(layout.coordinate, dtype=jnp.int32)
    tokens += jnp.where(
        is_pft[:, None],
        0.0,
        parameters["input_position_embedding"][coordinate],
    )
    return jnp.where(token_mask[:, None], tokens, 0.0), token_mask


def assemble_neural_process_context(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    *,
    normalization: NeuralInputNormalization,
    axis_layout: ProcessAxisLayout,
    config: NeuralArchitectureConfig = NeuralArchitectureConfig(),
) -> NeuralProcessContext:
    """Assemble only supplied native forcing and explicit process-axis tokens."""

    forcing = inputs.forcing
    values = _standardize(forcing.values, normalization.forcing_mean, normalization.forcing_scale)
    duration = jnp.asarray(forcing.duration_seconds, dtype=jnp.float64)[:, None] / 86400.0
    time = jnp.asarray(forcing.record_time_seconds, dtype=jnp.float64)[:, None] / 86400.0
    forcing_features = jnp.concatenate(
        (
            values,
            time,
            duration,
            jnp.asarray(forcing.predecessor_mask, dtype=jnp.float64)[:, None],
            jnp.asarray(forcing.record_mask, dtype=jnp.float64)[:, None],
        ),
        axis=1,
    )
    forcing_tokens = _linear(parameters["forcing_input"], forcing_features)
    forcing_mask = jnp.asarray(forcing.record_mask, dtype=jnp.bool_)

    pft_features = jnp.asarray(assemble_local_pft_features(inputs.pft), dtype=jnp.float64)
    pft_features = jnp.where(
        jnp.isfinite(pft_features) & (jnp.abs(pft_features) < 0.5e20),
        pft_features,
        0.0,
    )
    pft = _linear(parameters["pft_input"], pft_features)
    pft_mask = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
    domain_values = (
        _standardize(inputs.state.water, 0.0, normalization.water_scale),
        _standardize(inputs.state.carbon, 0.0, normalization.carbon_scale),
        _standardize(inputs.state.thermal, 0.0, normalization.thermal_scale),
    )
    domain_defined = (
        jnp.asarray(inputs.state_defined.water, dtype=jnp.bool_),
        jnp.asarray(inputs.state_defined.carbon, dtype=jnp.bool_),
        jnp.asarray(inputs.state_defined.thermal, dtype=jnp.bool_),
    )
    domain_tokens = [pft + parameters["domain_embedding"][0]]
    domain_masks = [pft_mask]
    for index, (layout, value, defined) in enumerate(
        zip(axis_layout.domains, domain_values, domain_defined, strict=True),
        start=1,
    ):
        tokens, mask = _semantic_domain_tokens(
            parameters,
            layout,
            value,
            defined,
            pft,
            pft_mask,
            domain_index=index,
        )
        domain_tokens.append(tokens)
        domain_masks.append(mask)

    continuous = _standardize(
        inputs.state.other,
        normalization.continuous_mean,
        normalization.continuous_scale,
    ).reshape(-1)
    continuous = jnp.where(
        jnp.asarray(inputs.state_defined.other, dtype=jnp.bool_).reshape(-1),
        continuous,
        0.0,
    )
    static = _standardize(inputs.static_conditions, normalization.static_mean, normalization.static_scale).reshape(-1)
    annual = _standardize(inputs.annual_conditions, normalization.annual_mean, normalization.annual_scale).reshape(-1)
    global_token = _linear(
        parameters["global_input"],
        jnp.concatenate(
            (
                _linear(parameters["state_rank"], _symmetric_summary(continuous)),
                _linear(parameters["static_rank"], _symmetric_summary(static)),
                _linear(parameters["annual_rank"], _symmetric_summary(annual)),
            )
        ),
    )
    return NeuralProcessContext(
        forcing_tokens=forcing_tokens,
        forcing_mask=forcing_mask,
        domain_tokens=tuple(domain_tokens),
        domain_masks=tuple(domain_masks),
        global_token=global_token,
    )


def _forcing_encode(parameters: Mapping[str, Any], context: NeuralProcessContext, config: Any) -> Any:
    tokens = context.forcing_tokens
    mask = context.forcing_mask
    causal = jnp.tril(jnp.ones((tokens.shape[0], tokens.shape[0]), dtype=jnp.bool_))
    pair_mask = causal & mask[:, None] & mask[None, :]
    for block in parameters["forcing_blocks"]:
        attended = _attention(
            block["axis_attention"],
            tokens,
            tokens,
            heads=config.attention_heads,
            attention_mask=pair_mask,
        )
        tokens = tokens + attended
        tokens = _layer_norm(tokens, block["axis_norm_scale"], block["axis_norm_bias"])
        tokens = jnp.where(mask[:, None], tokens + _ffn(block["axis_ffn"], tokens), 0.0)
    return _masked_mean(tokens, mask)


def encode_process_queries(
    parameters: Mapping[str, Any],
    context: NeuralProcessContext,
    *,
    arm: str,
    config: NeuralArchitectureConfig = NeuralArchitectureConfig(),
) -> tuple[Any, tuple[Any, Any, Any, Any]]:
    """Run the candidate or its exactly parameter-matched factorized control."""

    if arm not in {AXIAL_CANDIDATE, FACTORIZED_CONTROL}:
        raise ValueError(f"unknown process-head arm: {arm}")
    forcing_summary = _forcing_encode(parameters, context, config)
    domains = tuple(
        tokens + context.global_token + forcing_summary
        for tokens in context.domain_tokens
    )
    queries = parameters["process_queries"] + context.global_token + forcing_summary
    for block in parameters["process_blocks"]:
        if arm == AXIAL_CANDIDATE:
            domains = tuple(
                _candidate_axis_update(block, tokens, mask, config)
                for tokens, mask in zip(domains, context.domain_masks, strict=True)
            )
            summaries = jnp.stack(
                tuple(
                    _masked_mean(tokens, mask)
                    for tokens, mask in zip(domains, context.domain_masks, strict=True)
                )
            )
            query_mask = jnp.ones((queries.shape[0], summaries.shape[0]), dtype=jnp.bool_)
            interacted = _attention(
                block["query_attention"],
                queries,
                summaries,
                heads=config.attention_heads,
                attention_mask=query_mask,
            )
        else:
            domains = tuple(
                _control_axis_update(block, tokens, mask)
                for tokens, mask in zip(domains, context.domain_masks, strict=True)
            )
            pooled = jnp.mean(
                jnp.stack(
                    tuple(
                        _masked_mean(tokens, mask)
                        for tokens, mask in zip(domains, context.domain_masks, strict=True)
                    )
                ),
                axis=0,
            )
            attention = block["query_attention"]
            interacted = _linear(
                attention["output"],
                (
                    jax.nn.gelu(_linear(attention["query"], queries))
                    + jax.nn.gelu(_linear(attention["key"], pooled))[None, :]
                    + jax.nn.gelu(_linear(attention["value"], pooled))[None, :]
                )
                / 3.0,
            )
        queries = queries + interacted
        queries = _layer_norm(queries, block["query_norm_scale"], block["query_norm_bias"])
        queries = queries + _ffn(block["query_ffn"], queries)
    return queries, domains


def _field_coordinates(field: OutputFieldSpec) -> np.ndarray:
    if not field.shape:
        return np.zeros((1, 0), dtype=np.int32)
    return np.stack(np.unravel_index(np.arange(np.prod(field.shape)), field.shape), axis=1).astype(np.int32)


def _decode_field(
    parameters: Mapping[str, Any],
    queries: Any,
    pft_tokens: Any,
    field: OutputFieldSpec,
    field_index: int,
    schema: NeuralOutputSchema,
    normalization: NeuralOutputNormalization,
) -> Any:
    coordinates = jnp.asarray(_field_coordinates(field), dtype=jnp.int32)
    features = jnp.broadcast_to(
        queries[field.label_index] + parameters["field_embedding"][field_index],
        (coordinates.shape[0], queries.shape[-1]),
    )
    active_mask = jnp.ones((coordinates.shape[0],), dtype=jnp.bool_)
    for axis in range(coordinates.shape[1]):
        if axis == field.pft_axis:
            coordinate = coordinates[:, axis]
            if field.full_pft_axis:
                active = jnp.asarray(schema.active_pft_indices, dtype=jnp.int32)
                matches = coordinate[:, None] == active[None, :]
                active_mask = jnp.any(matches, axis=1)
                compact = jnp.argmax(matches, axis=1)
            else:
                compact = coordinate
                active_mask = active_mask & (compact < pft_tokens.shape[0])
            features = features + jnp.where(active_mask[:, None], pft_tokens[compact], 0.0)
        else:
            coordinate = coordinates[:, axis]
            features = features + parameters["axis_ordinal_embedding"][axis]
            features = features + parameters["output_position_embedding"][coordinate]
    raw = _linear(
        parameters["decoder_second"],
        jax.nn.gelu(_linear(parameters["decoder_first"], features)),
    )[:, 0]
    mean = jnp.asarray(normalization.mean[field.name], dtype=jnp.float64).reshape(-1)
    scale = jnp.asarray(normalization.scale[field.name], dtype=jnp.float64).reshape(-1)
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    if field.constraint == "unit_interval":
        value = jax.nn.sigmoid(raw)
    elif field.constraint == "signed_unit":
        value = jnp.tanh(raw)
    elif field.constraint == "nonnegative":
        ratio = jnp.maximum(mean / scale, 1.0e-12)
        baseline = ratio + jnp.log(-jnp.expm1(-ratio))
        value = jax.nn.softplus(baseline + raw) * scale
    else:
        value = mean + scale * raw
    value = jnp.where(active_mask, value, 0.0)
    return value.reshape(field.shape)


def neural_process_outputs(
    parameters: Mapping[str, Any],
    context: NeuralProcessContext,
    *,
    schema: NeuralOutputSchema,
    normalization: NeuralOutputNormalization,
    arm: str,
    config: NeuralArchitectureConfig = NeuralArchitectureConfig(),
) -> dict[str, Any]:
    queries, domains = encode_process_queries(parameters, context, arm=arm, config=config)
    pft_tokens = domains[0]
    return {
        field.name: _decode_field(
            parameters,
            queries,
            pft_tokens,
            field,
            index,
            schema,
            normalization,
        )
        for index, field in enumerate(schema.fields)
    }


def _constraint_for_field(name: str) -> str:
    if name.endswith(".outgoing_fraction"):
        return "unit_interval"
    if name.endswith(".incoming_amount") or name in {
        "daily.fast_gpp",
        "daily.fast_maintenance_respiration",
    }:
        return "nonnegative"
    if name.startswith("thermal."):
        return "signed_unit"
    return "signed"


def _pft_axis_for_field(name: str, shape: tuple[int, ...], typed_axes: Mapping[str, Sequence[str]]) -> tuple[int | None, bool]:
    if name.startswith("typed."):
        axes = tuple(typed_axes[name.removeprefix("typed.")])
        if "pft" not in axes:
            return None, False
        axis = axes.index("pft")
        return axis, shape[axis] == 14
    if name in {"daily.fast_gpp", "daily.fast_maintenance_respiration"}:
        return 1, False
    if name.startswith("inventory.water.qsintveg"):
        return 1, False
    if name.startswith("inventory.carbon.litter_above") or name.startswith("inventory.carbon.litter_below"):
        return 2, False
    if name.startswith("inventory.carbon.carbon_32l"):
        return 2, False
    if name.startswith("inventory.carbon.DOC") or name.startswith("inventory.carbon.interception_storage"):
        return 1, False
    if name == "thermal.thermosoil_previous_step_state.ptn.normalized_tendency":
        return 2, True
    return None, False


def build_neural_output_schema(
    targets: Mapping[str, Any],
    *,
    field_labels: Mapping[str, str],
    typed_axes: Mapping[str, Sequence[str]],
    active_pft_indices: Sequence[int],
) -> NeuralOutputSchema:
    """Freeze the named decoder registry from host-side contract metadata."""

    if set(targets) != set(field_labels):
        raise ValueError("neural output schema target/label registry drift")
    labels = tuple(sorted(set(field_labels.values())))
    label_index = {name: index for index, name in enumerate(labels)}
    fields = []
    for name, target in targets.items():
        value = target.value if hasattr(target, "value") else target
        shape = tuple(np.asarray(value).shape)
        pft_axis, full = _pft_axis_for_field(name, shape, typed_axes)
        fields.append(
            OutputFieldSpec(
                name=name,
                shape=shape,
                label_index=label_index[field_labels[name]],
                constraint=_constraint_for_field(name),
                pft_axis=pft_axis,
                full_pft_axis=full,
            )
        )
    return NeuralOutputSchema(
        fields=tuple(fields),
        labels=labels,
        active_pft_indices=tuple(int(index) for index in active_pft_indices),
    )
