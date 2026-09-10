"""Source-routed field-query architecture for the successor process-head A/B."""

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
from .input_conditioning import standardize_continuous
from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
    ProcessAxisLayout,
    build_process_axis_layout,
)
from .output_transform import field_coordinates, normalized_output_to_physical
from .parameter_routing import apply_fast_parameter_routes
from .process_derivations import NativeForcingFieldLayout
from .state_features import assemble_local_pft_features
from .successor_contract import SuccessorCandidateContract
from .types import AssembledDailyOperatorInput

SOURCE_ROUTED_ATTENTION = "source_routed_process_attention_v2"
SOURCE_ROUTED_GRAPH_CONTROL = "source_routed_graph_mlp_control_v2"
SUCCESSOR_FINAL_DECODER_KERNEL_GAIN = 1.0e-3


@dataclass(frozen=True)
class SuccessorArchitectureConfig:
    d_model: int = 256
    attention_heads: int = 8
    ffn_width: int = 1536
    forcing_blocks: int = 2
    process_blocks: int = 3
    decoder_width: int = 256

    def validate(self) -> None:
        if self.d_model != 256 or self.attention_heads != 8:
            raise ValueError("successor width/head contract drift")
        if self.ffn_width != 1536 or self.decoder_width != 256:
            raise ValueError("successor FFN/decoder contract drift")
        if self.forcing_blocks != 2 or self.process_blocks != 3:
            raise ValueError("successor depth contract drift")
        if self.d_model % self.attention_heads:
            raise ValueError("successor d_model must be divisible by attention heads")


@dataclass(frozen=True, eq=False)
class SummaryTokenLayout:
    pool_matrix: np.ndarray
    field_index: np.ndarray
    family_index: np.ndarray
    pft_index: np.ndarray
    element_pft_index: np.ndarray
    field_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.pool_matrix.ndim != 2:
            raise ValueError("summary token pool must be a matrix")
        tokens, width = self.pool_matrix.shape
        if tokens == 0:
            raise ValueError("summary token layout cannot be empty")
        if any(value.shape != (tokens,) for value in (self.field_index, self.family_index, self.pft_index)):
            raise ValueError("summary token metadata shape drift")
        if self.element_pft_index.shape != (width,):
            raise ValueError("summary token element/PFT layout drift")


@dataclass(frozen=True, eq=False)
class SuccessorTokenLayout:
    axis_layout: ProcessAxisLayout
    state: SummaryTokenLayout
    static: SummaryTokenLayout
    annual: SummaryTokenLayout
    family_names: tuple[str, ...]
    forcing_family_index: np.ndarray
    process_static_family_index: np.ndarray
    field_allowed_family_mask: np.ndarray
    field_graph_mask: np.ndarray
    t2m_longterm_start: int
    t2m_longterm_stop: int
    model_interval_seconds: float

    def family_index(self, name: str) -> int:
        try:
            return self.family_names.index(name)
        except ValueError as error:
            raise ValueError(f"unknown successor source family {name!r}") from error


class SuccessorProcessContext(NamedTuple):
    input_tokens: Any
    input_mask: Any
    input_family_index: Any
    pft_tokens: Any
    pft_mask: Any


def _contract_groups(contract: SuccessorCandidateContract) -> Mapping[str, Any]:
    return contract.payload["input_source_families"]


def _family_registry(contract: SuccessorCandidateContract) -> tuple[str, ...]:
    groups = _contract_groups(contract)
    names = []
    for key in (
        "forcing_fields",
        "state_components",
        "static_fields",
        "annual_fields",
        "process_static_fields",
    ):
        names.extend(groups[key])
    names.extend(groups["structural"])
    if len(set(names)) != len(names):
        raise ValueError("successor source-family names must be unique")
    return tuple(names)


def _reverse_groups(groups: Mapping[str, Sequence[str]]) -> dict[str, str]:
    result = {}
    for family, members in groups.items():
        for member in members:
            if member in result:
                raise ValueError(f"source member {member!r} has multiple families")
            result[str(member)] = str(family)
    return result


def _represented_state_keys(assembler: CanonicalDailyOperatorInputAssembler) -> set[str]:
    keys = {f"hydrol_previous_step_state.{name}" for name in WATER_INVENTORY_FIELDS}
    keys.update(f"slowproc_stomate_previous_step_state.{name}" for name in CARBON_INVENTORY_FIELDS)
    keys.update(
        f"{component}.{field}"
        for component, field, _bound in THERMAL_FIELDS
        if f"{component}.{field}" in assembler.thermal_keys
    )
    keys.update(f"slowproc_stomate_previous_step_state.{name}" for name in ("veget_max", "lai", "biomass"))
    return keys


def _state_summary_layout(
    assembler: CanonicalDailyOperatorInputAssembler,
    family_names: tuple[str, ...],
    component_families: Mapping[str, str],
) -> SummaryTokenLayout:
    width = int(assembler.contract.continuous_state_width)
    represented = _represented_state_keys(assembler)
    rows: list[np.ndarray] = []
    field_indices: list[int] = []
    family_indices: list[int] = []
    pft_indices: list[int] = []
    field_names: list[str] = []
    element_pft = np.full((width,), -1, dtype=np.int32)
    active_count = len(assembler.active_pft_indices)

    for leaf in assembler.contract.state_leaves:
        if leaf.start is None or leaf.stop is None:
            continue
        if leaf.key in represented:
            continue
        family = component_families.get(leaf.component)
        if family is None:
            raise ValueError(f"state component lacks successor source family: {leaf.component}")
        field_id = len(field_names)
        field_names.append(leaf.key)
        local_size = int(leaf.stop - leaf.start)
        local_shape = tuple(int(value) for value in leaf.shape)
        grid = (
            np.indices(local_shape, dtype=np.int32).reshape((len(local_shape), -1)).T
            if local_shape
            else np.zeros((1, 0), dtype=np.int32)
        )
        if grid.shape[0] != local_size:
            raise ValueError(f"state leaf shape/width drift for {leaf.key}")
        nvm_axis = tuple(leaf.axis_names).index("nvm") if "nvm" in leaf.axis_names else None
        if nvm_axis is not None and local_shape[nvm_axis] == active_count:
            element_pft[leaf.start : leaf.stop] = grid[:, nvm_axis]

        def add(local: np.ndarray, pft_index: int) -> None:
            row = np.zeros((width,), dtype=np.bool_)
            row[leaf.start : leaf.stop] = local
            rows.append(row)
            field_indices.append(field_id)
            family_indices.append(family_names.index(family))
            pft_indices.append(pft_index)

        add(np.ones((local_size,), dtype=np.bool_), -1)
        if nvm_axis is not None and local_shape[nvm_axis] == active_count:
            for compact in range(active_count):
                add(grid[:, nvm_axis] == compact, compact)

    return SummaryTokenLayout(
        pool_matrix=np.stack(rows),
        field_index=np.asarray(field_indices, dtype=np.int32),
        family_index=np.asarray(family_indices, dtype=np.int32),
        pft_index=np.asarray(pft_indices, dtype=np.int32),
        element_pft_index=element_pft,
        field_names=tuple(field_names),
    )


def _condition_summary_layout(
    leaves: Sequence[Any],
    *,
    width: int,
    family_names: tuple[str, ...],
    member_families: Mapping[str, str],
    active_pft_indices: tuple[int, ...],
    pft_indexed_fields: set[str],
) -> SummaryTokenLayout:
    rows = []
    fields = []
    families = []
    pfts = []
    names = []
    element_pft = np.full((width,), -1, dtype=np.int32)
    for leaf in leaves:
        family = member_families.get(leaf.name)
        if family is None:
            raise ValueError(f"condition leaf lacks successor source family: {leaf.name}")
        field_id = len(names)
        names.append(str(leaf.name))
        local_size = int(leaf.stop - leaf.start)
        base_valid = np.ones((local_size,), dtype=np.bool_)
        if leaf.name in pft_indexed_fields:
            if tuple(leaf.shape) != (14,):
                raise ValueError(f"PFT-indexed static leaf shape drift for {leaf.name}")
            base_valid = np.zeros((14,), dtype=np.bool_)
            base_valid[np.asarray(active_pft_indices, dtype=np.int32)] = True
            for compact, full in enumerate(active_pft_indices):
                element_pft[leaf.start + int(full)] = compact

        def add(local: np.ndarray, pft_index: int) -> None:
            row = np.zeros((width,), dtype=np.bool_)
            row[leaf.start : leaf.stop] = local
            rows.append(row)
            fields.append(field_id)
            families.append(family_names.index(family))
            pfts.append(pft_index)

        add(base_valid, -1)
        if leaf.name in pft_indexed_fields:
            for compact, full in enumerate(active_pft_indices):
                local = np.zeros((14,), dtype=np.bool_)
                local[int(full)] = True
                add(local, compact)
    return SummaryTokenLayout(
        pool_matrix=np.stack(rows),
        field_index=np.asarray(fields, dtype=np.int32),
        family_index=np.asarray(families, dtype=np.int32),
        pft_index=np.asarray(pfts, dtype=np.int32),
        element_pft_index=element_pft,
        field_names=tuple(names),
    )


def _route_key(label: str) -> str:
    if label == "carbon.fast_gpp":
        return label
    if label == "carbon.fast_maintenance_respiration":
        return label
    if label.startswith("water."):
        return "water"
    if label.startswith("energy."):
        return "energy"
    if label.startswith("carbon."):
        return "carbon.other"
    raise ValueError(f"successor output label has no source route: {label}")


def _field_source_route(field: Any, schema: NeuralOutputSchema) -> str:
    if field.source_route is not None:
        return field.source_route
    return _route_key(schema.labels[field.label_index])


def _field_process_domain(field: Any, schema: NeuralOutputSchema) -> str:
    if field.process_domain is not None:
        return field.process_domain
    return schema.labels[field.label_index].split(".", 1)[0]


def build_successor_token_layout(
    assembler: CanonicalDailyOperatorInputAssembler,
    schema: NeuralOutputSchema,
    successor_contract: SuccessorCandidateContract,
) -> SuccessorTokenLayout:
    """Build complete source-family and process-graph masks before initialization."""

    if assembler.contract.sha256 != successor_contract.payload["data"]["markov_contract_sha256"]:
        raise ValueError("successor and Markov contracts differ")
    family_names = _family_registry(successor_contract)
    groups = _contract_groups(successor_contract)
    forcing_members = _reverse_groups(groups["forcing_fields"])
    state_members = _reverse_groups(groups["state_components"])
    static_members = _reverse_groups(groups["static_fields"])
    annual_members = _reverse_groups(groups["annual_fields"])
    process_static_members = _reverse_groups(groups["process_static_fields"])

    forcing_fields = tuple(assembler.contract.native_forcing.fields)
    if set(forcing_fields) != set(forcing_members):
        raise ValueError("successor forcing fields do not cover the Markov contract")
    forcing_family = np.asarray(
        [family_names.index(forcing_members[name]) for name in forcing_fields],
        dtype=np.int32,
    )
    axis_layout = build_process_axis_layout(assembler)
    state_layout = _state_summary_layout(assembler, family_names, state_members)
    static_leaves = assembler.contract.static_conditions.landpoint_static_leaves
    if {leaf.name for leaf in static_leaves} != set(static_members):
        raise ValueError("successor static fields do not cover the Markov contract")
    static_layout = _condition_summary_layout(
        static_leaves,
        width=int(assembler.contract.static_conditions.landpoint_static_width),
        family_names=family_names,
        member_families=static_members,
        active_pft_indices=assembler.active_pft_indices,
        pft_indexed_fields={
            "hydrol_humcste",
            "hydrol_throughfall_by_pft",
            "ext_coeff_vegetfrac",
            "pref_soil_veg",
        },
    )
    annual_leaves = assembler.contract.static_conditions.annual_condition_leaves
    if {leaf.name for leaf in annual_leaves} != set(annual_members):
        raise ValueError("successor annual fields do not cover the Markov contract")
    annual_layout = _condition_summary_layout(
        annual_leaves,
        width=int(assembler.contract.static_conditions.annual_condition_width),
        family_names=family_names,
        member_families=annual_members,
        active_pft_indices=assembler.active_pft_indices,
        pft_indexed_fields=set(),
    )
    process_static_names = ("rprof", "z_soil", "litterfrac", "cue", "frac_carb", "sro_bottom")
    if set(process_static_names) != set(process_static_members):
        raise ValueError("successor process-static fields drift")
    process_static_family = np.asarray(
        [family_names.index(process_static_members[name]) for name in process_static_names],
        dtype=np.int32,
    )

    routes = successor_contract.payload["output_source_routes"]
    allowed = np.zeros((len(schema.fields), len(family_names)), dtype=np.bool_)
    domains = []
    for field_index, field in enumerate(schema.fields):
        route = _field_source_route(field, schema)
        if route not in routes:
            raise ValueError(f"successor output source route is unknown: {route}")
        allowed[field_index, [family_names.index(name) for name in routes[route]]] = True
        domains.append(_field_process_domain(field, schema))
    graph_rules = successor_contract.payload["process_graph"]
    graph = np.zeros((len(schema.fields), len(schema.fields)), dtype=np.bool_)
    for receiver, domain in enumerate(domains):
        admitted = set(graph_rules[f"{domain}_receives_from"])
        graph[receiver] = np.asarray([source in admitted for source in domains])
    if not np.all(np.any(allowed, axis=1)) or not np.all(np.any(graph, axis=1)):
        raise ValueError("successor source masks contain an empty query row")

    tlong = tuple(
        leaf
        for leaf in assembler.contract.state_leaves
        if leaf.key == "slowproc_stomate_previous_step_state.t2m_longterm"
    )
    if len(tlong) != 1 or tuple(tlong[0].shape) != (1,):
        raise ValueError("successor requires one scalar t2m_longterm state owner")
    return SuccessorTokenLayout(
        axis_layout=axis_layout,
        state=state_layout,
        static=static_layout,
        annual=annual_layout,
        family_names=family_names,
        forcing_family_index=forcing_family,
        process_static_family_index=process_static_family,
        field_allowed_family_mask=allowed,
        field_graph_mask=graph,
        t2m_longterm_start=int(tlong[0].start),
        t2m_longterm_stop=int(tlong[0].stop),
        model_interval_seconds=float(assembler.contract.native_forcing.model_interval_seconds),
    )


def _normal(key: Any, shape: tuple[int, ...], fan_in: int) -> Any:
    return jax.random.normal(key, shape, dtype=jnp.float64) / np.sqrt(float(max(fan_in, 1)))


def _linear_parameters(key: Any, in_width: int, out_width: int) -> dict[str, Any]:
    return {
        "kernel": _normal(key, (in_width, out_width), in_width),
        "bias": jnp.zeros((out_width,), dtype=jnp.float64),
    }


def _final_decoder_parameters(key: Any, width: int) -> dict[str, Any]:
    parameters = _linear_parameters(key, width, 1)
    return {
        "kernel": parameters["kernel"] * SUCCESSOR_FINAL_DECODER_KERNEL_GAIN,
        "bias": parameters["bias"],
    }


def _mixer_parameters(key: Any, width: int) -> dict[str, Any]:
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


def _forcing_block_parameters(key: Any, config: SuccessorArchitectureConfig) -> dict[str, Any]:
    mix, ffn = jax.random.split(key)
    return {
        "mixer": _mixer_parameters(mix, config.d_model),
        "ffn": _ffn_parameters(ffn, config.d_model, config.ffn_width),
        "mix_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float64),
        "mix_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float64),
        "ffn_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float64),
        "ffn_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float64),
    }


def _process_block_parameters(key: Any, config: SuccessorArchitectureConfig) -> dict[str, Any]:
    input_mix, graph_mix, input_ffn, graph_ffn = jax.random.split(key, 4)
    return {
        "input_mixer": _mixer_parameters(input_mix, config.d_model),
        "graph_mixer": _mixer_parameters(graph_mix, config.d_model),
        "input_ffn": _ffn_parameters(input_ffn, config.d_model, config.ffn_width),
        "graph_ffn": _ffn_parameters(graph_ffn, config.d_model, config.ffn_width),
        "input_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float64),
        "input_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float64),
        "input_ffn_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float64),
        "input_ffn_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float64),
        "graph_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float64),
        "graph_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float64),
        "graph_ffn_norm_scale": jnp.ones((config.d_model,), dtype=jnp.float64),
        "graph_ffn_norm_bias": jnp.zeros((config.d_model,), dtype=jnp.float64),
    }


def initialize_successor_parameters(
    key: Any,
    *,
    schema: NeuralOutputSchema,
    layout: SuccessorTokenLayout,
    inputs: AssembledDailyOperatorInput,
    arm: str,
    config: SuccessorArchitectureConfig = SuccessorArchitectureConfig(),
) -> Mapping[str, Any]:
    """Initialize a new successor tree without any first-pair parameter reuse."""

    config.validate()
    if arm not in {SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL}:
        raise ValueError(f"unknown successor arm {arm!r}")
    key_count = 21 + config.forcing_blocks + config.process_blocks
    keys = iter(jax.random.split(key, key_count))
    pft_width = int(np.asarray(assemble_local_pft_features(inputs.pft)).shape[-1])
    return {
        "forcing_input": _linear_parameters(next(keys), 5, config.d_model),
        "summary_input": _linear_parameters(next(keys), 4, config.d_model),
        "pft_input": _linear_parameters(next(keys), pft_width, config.d_model),
        "calendar_input": _linear_parameters(next(keys), 4, config.d_model),
        "family_embedding": _normal(next(keys), (len(layout.family_names), config.d_model), config.d_model),
        "forcing_field_embedding": _normal(
            next(keys),
            (np.asarray(layout.forcing_family_index).size, config.d_model),
            config.d_model,
        ),
        "forcing_position_embedding": _normal(next(keys), (7, config.d_model), config.d_model),
        "semantic_field_embedding": _normal(
            next(keys),
            (len(layout.axis_layout.field_names), config.d_model),
            config.d_model,
        ),
        "semantic_role_embedding": _normal(
            next(keys),
            (len(layout.axis_layout.role_names), config.d_model),
            config.d_model,
        ),
        "semantic_position_embedding": _normal(next(keys), (64, config.d_model), config.d_model),
        "state_field_embedding": _normal(next(keys), (len(layout.state.field_names), config.d_model), config.d_model),
        "static_field_embedding": _normal(next(keys), (len(layout.static.field_names), config.d_model), config.d_model),
        "annual_field_embedding": _normal(next(keys), (len(layout.annual.field_names), config.d_model), config.d_model),
        "process_static_field_embedding": _normal(next(keys), (6, config.d_model), config.d_model),
        "field_queries": _normal(next(keys), (len(schema.fields), config.d_model), config.d_model),
        "label_embedding": _normal(next(keys), (len(schema.labels), config.d_model), config.d_model),
        "output_field_embedding": _normal(next(keys), (len(schema.fields), config.d_model), config.d_model),
        "axis_ordinal_embedding": _normal(next(keys), (6, config.d_model), config.d_model),
        "output_position_embedding": _normal(next(keys), (64, config.d_model), config.d_model),
        "forcing_blocks": tuple(_forcing_block_parameters(next(keys), config) for _ in range(config.forcing_blocks)),
        "process_blocks": tuple(_process_block_parameters(next(keys), config) for _ in range(config.process_blocks)),
        "decoder_first": _linear_parameters(next(keys), config.d_model, config.decoder_width),
        "decoder_second": _final_decoder_parameters(next(keys), config.decoder_width),
    }


def successor_parameter_count(parameters: Mapping[str, Any]) -> int:
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


def _fixed_mixer(
    parameters: Mapping[str, Any],
    query: Any,
    key_value: Any,
    mask: Any,
) -> Any:
    query_local = jax.nn.gelu(_linear(parameters["query"], query))
    key_local = jax.nn.gelu(_linear(parameters["key"], key_value))
    gate = jax.nn.sigmoid(_linear(parameters["value"], key_value))
    message = key_local * gate
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    count = jnp.maximum(jnp.sum(mask, axis=-1, dtype=jnp.float64), 1.0)
    pooled = jnp.einsum("qk,kd->qd", mask.astype(jnp.float64), message) / count[:, None]
    return _linear(parameters["output"], (query_local + pooled) / 2.0)


def _standardize(value: Any, mean: Any, scale: Any) -> Any:
    value = jnp.asarray(value, dtype=jnp.float64)
    mean = jnp.asarray(mean, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    valid = jnp.isfinite(value) & (jnp.abs(value) < 0.5e20)
    safe = jnp.where(valid, value, mean)
    return jnp.where(valid, (safe - mean) / jnp.where(scale > 0.0, scale, 1.0), 0.0)


@jax.custom_jvp
def _sqrt_with_zero_tangent(value: Any) -> Any:
    """Preserve sqrt primals while defining the zero-variance tangent."""

    return jnp.sqrt(value)


@_sqrt_with_zero_tangent.defjvp
def _sqrt_with_zero_tangent_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
    (value,) = primals
    (tangent,) = tangents
    root = jnp.sqrt(value)
    positive = value > 0.0
    denominator = jnp.where(positive, 2.0 * root, 1.0)
    return root, jnp.where(positive, tangent / denominator, 0.0)


def _summary_tokens(
    parameters: Mapping[str, Any],
    layout: SummaryTokenLayout,
    values: Any,
    defined: Any,
    pft_tokens: Any,
    pft_mask: Any,
    field_embedding: Any,
) -> tuple[Any, Any, Any]:
    values = jnp.asarray(values, dtype=jnp.float64).reshape(-1)
    defined = jnp.asarray(defined, dtype=jnp.bool_).reshape(-1)
    pft_mask = jnp.asarray(pft_mask, dtype=jnp.bool_)
    element_pft = jnp.asarray(layout.element_pft_index, dtype=jnp.int32)
    has_pft = element_pft >= 0
    element_active = (~has_pft) | pft_mask[jnp.maximum(element_pft, 0)]
    valid = defined & element_active & jnp.isfinite(values)
    pool = jnp.asarray(layout.pool_matrix, dtype=jnp.bool_) & valid[None, :]
    count = jnp.sum(pool, axis=1, dtype=jnp.float64)
    token_mask = count > 0.0
    safe = jnp.where(valid, values, 0.0)
    mean = jnp.sum(jnp.where(pool, safe[None, :], 0.0), axis=1) / jnp.maximum(count, 1.0)
    centered = jnp.where(pool, safe[None, :] - mean[:, None], 0.0)
    std = _sqrt_with_zero_tangent(jnp.sum(centered * centered, axis=1) / jnp.maximum(count, 1.0))
    minimum = jnp.min(jnp.where(pool, safe[None, :], jnp.inf), axis=1)
    maximum = jnp.max(jnp.where(pool, safe[None, :], -jnp.inf), axis=1)
    summary = jnp.stack(
        (
            mean,
            std,
            jnp.where(token_mask, minimum, 0.0),
            jnp.where(token_mask, maximum, 0.0),
        ),
        axis=1,
    )
    tokens = _linear(parameters["summary_input"], summary)
    tokens += field_embedding[jnp.asarray(layout.field_index, dtype=jnp.int32)]
    family_index = jnp.asarray(layout.family_index, dtype=jnp.int32)
    tokens += parameters["family_embedding"][family_index]
    pft_index = jnp.asarray(layout.pft_index, dtype=jnp.int32)
    is_pft = pft_index >= 0
    safe_pft = jnp.maximum(pft_index, 0)
    token_mask &= (~is_pft) | pft_mask[safe_pft]
    tokens += jnp.where(is_pft[:, None], pft_tokens[safe_pft], 0.0)
    return jnp.where(token_mask[:, None], tokens, 0.0), token_mask, family_index


def _semantic_domain_tokens(
    parameters: Mapping[str, Any],
    domain: Any,
    values: Any,
    defined: Any,
    pft_tokens: Any,
    pft_mask: Any,
    family_index: int,
) -> tuple[Any, Any, Any]:
    summary_layout = SummaryTokenLayout(
        pool_matrix=domain.pool_matrix,
        field_index=domain.field_index,
        family_index=np.full(domain.field_index.shape, family_index, dtype=np.int32),
        pft_index=domain.pft_index,
        element_pft_index=domain.element_pft_index,
        field_names=tuple(),
    )
    tokens, mask, families = _summary_tokens(
        parameters,
        summary_layout,
        values,
        defined,
        pft_tokens,
        pft_mask,
        parameters["semantic_field_embedding"],
    )
    tokens += parameters["semantic_role_embedding"][jnp.asarray(domain.role_index)]
    coordinate = jnp.asarray(domain.coordinate, dtype=jnp.int32)
    is_pft = jnp.asarray(domain.pft_index, dtype=jnp.int32) >= 0
    tokens += jnp.where(
        is_pft[:, None],
        0.0,
        parameters["semantic_position_embedding"][coordinate],
    )
    return jnp.where(mask[:, None], tokens, 0.0), mask, families


def _process_static_tokens(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    layout: SuccessorTokenLayout,
) -> tuple[Any, Any, Any]:
    summaries = []
    masks = []
    pft_mask = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
    for name in ("rprof", "z_soil", "litterfrac", "cue", "frac_carb", "sro_bottom"):
        value = getattr(inputs.process_static, name)
        array = jnp.asarray(value, dtype=jnp.float64).reshape(-1)
        valid = jnp.isfinite(array)
        if name == "rprof":
            original_shape = jnp.asarray(value).shape
            if not original_shape or original_shape[-1] != pft_mask.shape[0]:
                raise ValueError("rprof must end in the compact PFT axis")
            valid &= jnp.broadcast_to(pft_mask, original_shape).reshape(-1)
        count = jnp.sum(valid, dtype=jnp.float64)
        safe = jnp.where(valid, array, 0.0)
        mean = jnp.sum(safe) / jnp.maximum(count, 1.0)
        centered = jnp.where(valid, array - mean, 0.0)
        summaries.append(
            jnp.stack(
                (
                    mean,
                    _sqrt_with_zero_tangent(jnp.sum(centered * centered) / jnp.maximum(count, 1.0)),
                    jnp.min(jnp.where(valid, array, jnp.inf)),
                    jnp.max(jnp.where(valid, array, -jnp.inf)),
                )
            )
        )
        masks.append(count > 0.0)
    summary = jnp.stack(tuple(summaries))
    families = jnp.asarray(layout.process_static_family_index, dtype=jnp.int32)
    tokens = _linear(parameters["summary_input"], summary)
    tokens += parameters["process_static_field_embedding"]
    tokens += parameters["family_embedding"][families]
    mask = jnp.asarray(masks, dtype=jnp.bool_) & jnp.all(jnp.isfinite(summary), axis=1)
    return jnp.where(mask[:, None], tokens, 0.0), mask, families


def _forcing_tokens(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    layout: SuccessorTokenLayout,
    normalization: NeuralInputNormalization,
    config: SuccessorArchitectureConfig,
) -> tuple[Any, Any, Any]:
    forcing = inputs.forcing
    values = _standardize(
        forcing.values,
        normalization.forcing_mean,
        normalization.forcing_scale,
    )
    records, fields = values.shape
    if fields != layout.forcing_family_index.size or records > 7:
        raise ValueError("successor native forcing shape drift")
    value = values.reshape(-1)
    time = jnp.repeat(jnp.asarray(forcing.record_time_seconds) / 86400.0, fields)
    duration = jnp.repeat(jnp.asarray(forcing.duration_seconds) / 86400.0, fields)
    predecessor = jnp.repeat(jnp.asarray(forcing.predecessor_mask, dtype=jnp.float64), fields)
    record_valid = jnp.repeat(jnp.asarray(forcing.record_mask, dtype=jnp.bool_), fields)
    field_index = jnp.tile(jnp.arange(fields, dtype=jnp.int32), records)
    record_index = jnp.repeat(jnp.arange(records, dtype=jnp.int32), fields)
    features = jnp.stack((value, time, duration, predecessor, record_valid.astype(jnp.float64)), axis=1)
    family = jnp.tile(jnp.asarray(layout.forcing_family_index, dtype=jnp.int32), records)
    tokens = _linear(parameters["forcing_input"], features)
    tokens += parameters["forcing_field_embedding"][field_index]
    tokens += parameters["forcing_position_embedding"][record_index]
    tokens += parameters["family_embedding"][family]
    pair_mask = record_valid[:, None] & record_valid[None, :]
    for block in parameters["forcing_blocks"]:
        mixed = _attention(
            block["mixer"],
            tokens,
            tokens,
            heads=config.attention_heads,
            attention_mask=pair_mask,
        )
        tokens = _layer_norm(
            tokens + mixed,
            block["mix_norm_scale"],
            block["mix_norm_bias"],
        )
        updated = tokens + _ffn(block["ffn"], tokens)
        tokens = _layer_norm(
            updated,
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
        tokens = jnp.where(record_valid[:, None], tokens, 0.0)
    return tokens, record_valid, family


def assemble_successor_process_context(
    parameters: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    *,
    normalization: NeuralInputNormalization,
    layout: SuccessorTokenLayout,
    config: SuccessorArchitectureConfig = SuccessorArchitectureConfig(),
) -> SuccessorProcessContext:
    """Assemble source-typed tokens without an opaque global state summary."""

    config.validate()
    pft_features = jnp.asarray(assemble_local_pft_features(inputs.pft), dtype=jnp.float64)
    pft_features = jnp.where(jnp.isfinite(pft_features), pft_features, 0.0)
    pft_mask = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
    pft_family = layout.family_index("pft_structure")
    pft_tokens = _linear(parameters["pft_input"], pft_features)
    pft_tokens += parameters["family_embedding"][pft_family]
    pft_tokens = jnp.where(pft_mask[:, None], pft_tokens, 0.0)

    token_groups = []
    mask_groups = []
    family_groups = []
    forcing = _forcing_tokens(parameters, inputs, layout, normalization, config)
    token_groups.append(forcing[0])
    mask_groups.append(forcing[1])
    family_groups.append(forcing[2])
    token_groups.append(pft_tokens)
    mask_groups.append(pft_mask)
    family_groups.append(jnp.full((pft_tokens.shape[0],), pft_family, dtype=jnp.int32))

    semantic_values = (
        _standardize(inputs.state.water, 0.0, normalization.water_scale),
        _standardize(inputs.state.carbon, 0.0, normalization.carbon_scale),
        _standardize(inputs.state.thermal, 0.0, normalization.thermal_scale),
    )
    semantic_defined = (
        inputs.state_defined.water,
        inputs.state_defined.carbon,
        inputs.state_defined.thermal,
    )
    for domain, values, defined, family_name in zip(
        layout.axis_layout.domains,
        semantic_values,
        semantic_defined,
        ("state_water", "state_carbon", "state_thermal"),
        strict=True,
    ):
        group = _semantic_domain_tokens(
            parameters,
            domain,
            values,
            defined,
            pft_tokens,
            pft_mask,
            layout.family_index(family_name),
        )
        token_groups.append(group[0])
        mask_groups.append(group[1])
        family_groups.append(group[2])

    state_values = standardize_continuous(
        inputs.state.other,
        normalization.continuous_mean,
        normalization.continuous_scale,
        normalization.continuous_tail_threshold,
        dtype=jnp.float64,
        legacy_standardize=_standardize,
    )
    state_group = _summary_tokens(
        parameters,
        layout.state,
        state_values,
        inputs.state_defined.other,
        pft_tokens,
        pft_mask,
        parameters["state_field_embedding"],
    )
    token_groups.append(state_group[0])
    mask_groups.append(state_group[1])
    family_groups.append(state_group[2])

    static_values = _standardize(
        inputs.static_conditions,
        normalization.static_mean,
        normalization.static_scale,
    )
    static_group = _summary_tokens(
        parameters,
        layout.static,
        static_values,
        jnp.ones_like(static_values, dtype=jnp.bool_),
        pft_tokens,
        pft_mask,
        parameters["static_field_embedding"],
    )
    token_groups.append(static_group[0])
    mask_groups.append(static_group[1])
    family_groups.append(static_group[2])

    annual_values = _standardize(
        inputs.annual_conditions,
        normalization.annual_mean,
        normalization.annual_scale,
    )
    annual_group = _summary_tokens(
        parameters,
        layout.annual,
        annual_values,
        jnp.ones_like(annual_values, dtype=jnp.bool_),
        pft_tokens,
        pft_mask,
        parameters["annual_field_embedding"],
    )
    token_groups.append(annual_group[0])
    mask_groups.append(annual_group[1])
    family_groups.append(annual_group[2])

    process_static = _process_static_tokens(parameters, inputs, layout)
    token_groups.append(process_static[0])
    mask_groups.append(process_static[1])
    family_groups.append(process_static[2])

    day = jnp.asarray(inputs.day_index, dtype=jnp.float64)
    year = jnp.asarray(inputs.year, dtype=jnp.float64)
    phase = 2.0 * jnp.pi * day / 365.0
    calendar_features = jnp.stack((year / 1000.0, day / 365.0, jnp.sin(phase), jnp.cos(phase)))
    calendar_family = layout.family_index("calendar")
    calendar = _linear(parameters["calendar_input"], calendar_features)[None, :]
    calendar += parameters["family_embedding"][calendar_family]
    token_groups.append(calendar)
    mask_groups.append(jnp.ones((1,), dtype=jnp.bool_))
    family_groups.append(jnp.asarray([calendar_family], dtype=jnp.int32))

    return SuccessorProcessContext(
        input_tokens=jnp.concatenate(tuple(token_groups), axis=0),
        input_mask=jnp.concatenate(tuple(mask_groups), axis=0),
        input_family_index=jnp.concatenate(tuple(family_groups), axis=0),
        pft_tokens=pft_tokens,
        pft_mask=pft_mask,
    )


def _encode_field_queries(
    parameters: Mapping[str, Any],
    context: SuccessorProcessContext,
    *,
    schema: NeuralOutputSchema,
    layout: SuccessorTokenLayout,
    arm: str,
    config: SuccessorArchitectureConfig,
) -> Any:
    if arm not in {SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL}:
        raise ValueError(f"unknown successor arm {arm!r}")
    label_indices = jnp.asarray([field.label_index for field in schema.fields], dtype=jnp.int32)
    queries = parameters["field_queries"] + parameters["label_embedding"][label_indices]
    family_allowed = jnp.asarray(layout.field_allowed_family_mask, dtype=jnp.bool_)
    input_family = jnp.asarray(context.input_family_index, dtype=jnp.int32)
    input_mask = family_allowed[:, input_family] & context.input_mask[None, :]
    graph_mask = jnp.asarray(layout.field_graph_mask, dtype=jnp.bool_)
    for block in parameters["process_blocks"]:
        if arm == SOURCE_ROUTED_ATTENTION:
            input_message = _attention(
                block["input_mixer"],
                queries,
                context.input_tokens,
                heads=config.attention_heads,
                attention_mask=input_mask,
            )
        else:
            input_message = _fixed_mixer(
                block["input_mixer"],
                queries,
                context.input_tokens,
                input_mask,
            )
        queries = _layer_norm(
            queries + input_message,
            block["input_norm_scale"],
            block["input_norm_bias"],
        )
        queries = _layer_norm(
            queries + _ffn(block["input_ffn"], queries),
            block["input_ffn_norm_scale"],
            block["input_ffn_norm_bias"],
        )
        if arm == SOURCE_ROUTED_ATTENTION:
            graph_message = _attention(
                block["graph_mixer"],
                queries,
                queries,
                heads=config.attention_heads,
                attention_mask=graph_mask,
            )
        else:
            graph_message = _fixed_mixer(
                block["graph_mixer"],
                queries,
                queries,
                graph_mask,
            )
        queries = _layer_norm(
            queries + graph_message,
            block["graph_norm_scale"],
            block["graph_norm_bias"],
        )
        queries = _layer_norm(
            queries + _ffn(block["graph_ffn"], queries),
            block["graph_ffn_norm_scale"],
            block["graph_ffn_norm_bias"],
        )
    return queries


def _decode_field(
    parameters: Mapping[str, Any],
    queries: Any,
    context: SuccessorProcessContext,
    field: Any,
    field_index: int,
    schema: NeuralOutputSchema,
    normalization: NeuralOutputNormalization,
    inputs: AssembledDailyOperatorInput,
) -> Any:
    coordinates = jnp.asarray(field_coordinates(field.shape), dtype=jnp.int32)
    features = jnp.broadcast_to(
        queries[field_index] + parameters["output_field_embedding"][field_index],
        (coordinates.shape[0], queries.shape[-1]),
    )
    active_mask = jnp.ones((coordinates.shape[0],), dtype=jnp.bool_)
    for axis in range(coordinates.shape[1]):
        coordinate = coordinates[:, axis]
        if axis == field.pft_axis:
            if field.full_pft_axis:
                active = jnp.asarray(schema.active_pft_indices, dtype=jnp.int32)
                matches = coordinate[:, None] == active[None, :]
                compact = jnp.argmax(matches, axis=1)
                active_mask &= jnp.any(matches, axis=1)
            else:
                compact = coordinate
                active_mask &= compact < context.pft_tokens.shape[0]
            safe_compact = jnp.minimum(compact, context.pft_tokens.shape[0] - 1)
            active_mask &= context.pft_mask[safe_compact]
            features += jnp.where(active_mask[:, None], context.pft_tokens[safe_compact], 0.0)
        else:
            features += parameters["axis_ordinal_embedding"][axis]
            features += parameters["output_position_embedding"][coordinate]
    raw = _linear(
        parameters["decoder_second"],
        jax.nn.gelu(_linear(parameters["decoder_first"], features)),
    )[:, 0]
    return normalized_output_to_physical(
        raw,
        active_mask,
        field=field,
        normalization=normalization,
        inputs=inputs,
    )


def successor_process_outputs(
    parameters: Mapping[str, Any],
    context: SuccessorProcessContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: SuccessorTokenLayout,
    normalization: NeuralOutputNormalization,
    forcing_layout: NativeForcingFieldLayout,
    arm: str,
    config: SuccessorArchitectureConfig = SuccessorArchitectureConfig(),
) -> dict[str, Any]:
    queries = _encode_field_queries(
        parameters,
        context,
        schema=schema,
        layout=layout,
        arm=arm,
        config=config,
    )
    outputs = {
        field.name: _decode_field(
            parameters,
            queries,
            context,
            field,
            index,
            schema,
            normalization,
            inputs,
        )
        for index, field in enumerate(schema.fields)
    }
    return apply_fast_parameter_routes(outputs, inputs, layout, forcing_layout)
