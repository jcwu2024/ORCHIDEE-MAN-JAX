"""Coordinate-preserving carbon-state feedback for the structured GPP head."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import DailyMarkovContract

from .neural_architecture import NeuralInputNormalization, NeuralOutputNormalization, NeuralOutputSchema
from .parameter_routing import apply_fast_parameter_routes
from .process_derivations import NativeForcingFieldLayout
from .strong_baseline_architecture import (
    StrongBaselineConfig,
    StrongBaselineContext,
    StrongBaselineLayout,
    _attention,
    _ffn,
    _layer_norm,
    _linear,
    _linear_parameters,
    _normal,
    _residual_block_parameters,
    _standardize,
    strong_baseline_parameter_count,
)
from .strong_baseline_contract import StrongBaselineContract, canonical_axis_aliases
from .target_oriented_attention import TargetOrientedAttentionConfig, _target_oriented_attention_decoded_outputs
from .target_oriented_contract import TargetOrientedContract
from .target_oriented_distribution_aware_maintenance import (
    DistributionAwareMaintenanceConfig,
    DistributionAwareMaintenanceContract,
    distribution_aware_maintenance_corrections,
)
from .target_oriented_source_shaped_maintenance import (
    MAINTENANCE_FIELD,
    SourceShapedMaintenanceConfig,
    contract_sha256,
    materialize_part_resolved_source_shaped_maintenance,
)
from .target_oriented_structured_gpp import (
    GPP_FIELD,
    StructuredGPPConfig,
    StructuredGPPContract,
    StructuredGPPQueryState,
    StructuredGPPStateLayout,
    structured_gpp_base_response,
    structured_gpp_query_state,
)
from .types import AssembledDailyOperatorInput

SCHEMA_VERSION = "target_oriented_coupled_gpp_carbon_feedback_v1"
FROZEN_CONTRACT_SHA256 = "c7f81436ff6ba1179b01adad0f76b47ddd2aab68208e97719c2c5464f3e7e74d"
CARBON_COMPONENT = "slowproc_stomate_previous_step_state"
PARAMETER_FAMILY = "coupled_gpp_carbon_feedback"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_carbon_feedback_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPCarbonFeedbackContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


class CarbonFeedbackStateLayout(NamedTuple):
    state_indices: np.ndarray
    field_index: np.ndarray
    pft_owner_index: np.ndarray
    axis_role_index: np.ndarray
    axis_position_index: np.ndarray
    field_names: tuple[str, ...]
    axis_role_names: tuple[str, ...]
    axis_role_extents: tuple[int, ...]


@dataclass(frozen=True)
class CoupledGPPCarbonFeedbackConfig:
    width: int = 384
    attention_heads: int = 8
    ffn_width: int = 768
    cross_attention_blocks: int = 2
    maximum_non_pft_axis_roles: int = 5
    residual_initialization_scale: float = 1.0e-3

    def validate(
        self,
        contract: CoupledGPPCarbonFeedbackContract,
        structured_config: StructuredGPPConfig,
    ) -> None:
        architecture = contract.payload["architecture"]
        observed = (
            self.width,
            self.attention_heads,
            self.ffn_width,
            self.cross_attention_blocks,
            self.maximum_non_pft_axis_roles,
            self.residual_initialization_scale,
        )
        declared = (
            int(architecture["width"]),
            int(architecture["attention_heads"]),
            int(architecture["ffn_width"]),
            int(architecture["cross_attention_blocks"]),
            int(contract.payload["state_tokenization"]["maximum_non_pft_axis_roles_per_coordinate"]),
            float(architecture["residual_injection_kernel_initialization_scale"]),
        )
        if observed != declared:
            raise ValueError("coupled GPP/carbon-feedback architecture drift")
        if self.width != structured_config.width or self.width % self.attention_heads:
            raise ValueError("coupled GPP/carbon-feedback width drift")


def load_coupled_gpp_carbon_feedback_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPCarbonFeedbackContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    expected_status = "frozen_after_user_architecture_approval_for_local_implementation_and_zero_update_admission"
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != expected_status:
        raise ValueError("coupled GPP/carbon-feedback contract schema drift")
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("coupled GPP/carbon-feedback contract is not frozen v1")
    state = payload["state_tokenization"]
    architecture = payload["architecture"]
    if state["selected_component"] != CARBON_COMPONENT:
        raise ValueError("coupled GPP/carbon-feedback component drift")
    if int(state["continuous_coordinate_count"]) != 2_386 or int(state["continuous_field_count"]) != 107:
        raise ValueError("coupled GPP/carbon-feedback coordinate inventory drift")
    if int(architecture["new_adapter_parameter_count"]) != 3_080_832:
        raise ValueError("coupled GPP/carbon-feedback adapter count drift")
    if int(architecture["candidate_parameter_count"]) != 109_464_359:
        raise ValueError("coupled GPP/carbon-feedback candidate count drift")
    return CoupledGPPCarbonFeedbackContract(payload, source, observed)


def build_carbon_feedback_state_layout(
    markov_contract: DailyMarkovContract,
    *,
    baseline_layout: StrongBaselineLayout,
    architecture_contract: StrongBaselineContract,
    coupled_contract: CoupledGPPCarbonFeedbackContract,
) -> CarbonFeedbackStateLayout:
    """Derive carbon coordinates and semantic axes from the accepted contracts."""

    leaves = tuple(
        leaf
        for leaf in markov_contract.state_leaves
        if not leaf.discrete and leaf.component == CARBON_COMPONENT
    )
    state_spec = coupled_contract.payload["state_tokenization"]
    expected_fields = int(state_spec["continuous_field_count"])
    expected_coordinates = int(state_spec["continuous_coordinate_count"])
    if len(leaves) != expected_fields:
        raise ValueError(f"carbon-feedback field count is {len(leaves)}; expected {expected_fields}")

    aliases = canonical_axis_aliases(architecture_contract)
    role_names = tuple(str(value) for value in state_spec["axis_role_names"])
    role_extents = tuple(int(value) for value in state_spec["axis_role_extents"])
    role_lookup = {name: index for index, name in enumerate(role_names)}
    derived_extents = {name: 0 for name in role_names}
    max_roles = int(state_spec["maximum_non_pft_axis_roles_per_coordinate"])

    state_indices: list[np.ndarray] = []
    field_indices: list[np.ndarray] = []
    role_rows: list[np.ndarray] = []
    position_rows: list[np.ndarray] = []
    for field, leaf in enumerate(leaves):
        start = int(leaf.start)
        stop = int(leaf.stop)
        shape = tuple(int(value) for value in leaf.shape)
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if stop - start != size:
            raise ValueError(f"carbon-feedback state slice size drift: {leaf.key}")
        grid = np.indices(shape, dtype=np.int32).reshape((len(shape), -1)).T if shape else np.zeros((1, 0), dtype=np.int32)
        roles = np.full((size, max_roles), -1, dtype=np.int32)
        positions = np.full((size, max_roles), -1, dtype=np.int32)
        local_role = 0
        for axis, (raw_name, extent) in enumerate(zip(leaf.axis_names, shape, strict=True)):
            canonical = aliases.get(raw_name, raw_name)
            if canonical == "pft":
                continue
            if canonical not in role_lookup:
                raise ValueError(f"carbon-feedback axis role is not frozen: {raw_name}")
            if local_role >= max_roles:
                raise ValueError(f"carbon-feedback state rank exceeds the frozen maximum: {leaf.key}")
            role = role_lookup[canonical]
            roles[:, local_role] = role
            positions[:, local_role] = grid[:, axis]
            derived_extents[canonical] = max(derived_extents[canonical], extent)
            local_role += 1
        state_indices.append(np.arange(start, stop, dtype=np.int32))
        field_indices.append(np.full((size,), field, dtype=np.int32))
        role_rows.append(roles)
        position_rows.append(positions)

    indices = np.concatenate(state_indices)
    if indices.size != expected_coordinates or np.unique(indices).size != indices.size:
        raise ValueError("carbon-feedback coordinate inventory is not exact")
    observed_extents = tuple(derived_extents[name] for name in role_names)
    if observed_extents != role_extents:
        raise ValueError(f"carbon-feedback axis extents are {observed_extents}; expected {role_extents}")
    owners = np.asarray(baseline_layout.state_element_pft_index[indices], dtype=np.int32)
    if np.any(owners < -1):
        raise ValueError("carbon-feedback selected an unsupported PFT coordinate")
    return CarbonFeedbackStateLayout(
        state_indices=indices,
        field_index=np.concatenate(field_indices),
        pft_owner_index=owners,
        axis_role_index=np.concatenate(role_rows),
        axis_position_index=np.concatenate(position_rows),
        field_names=tuple(leaf.key for leaf in leaves),
        axis_role_names=role_names,
        axis_role_extents=role_extents,
    )


def initialize_coupled_gpp_carbon_feedback_parameters(
    parent_parameters: Mapping[str, Any],
    key: Any,
    *,
    state_layout: CarbonFeedbackStateLayout,
    coupled_contract: CoupledGPPCarbonFeedbackContract,
    structured_contract: StructuredGPPContract,
    structured_config: StructuredGPPConfig = StructuredGPPConfig(),
    config: CoupledGPPCarbonFeedbackConfig = CoupledGPPCarbonFeedbackConfig(),
) -> Mapping[str, Any]:
    """Attach the approved adapter without changing inherited parameters."""

    config.validate(coupled_contract, structured_config)
    expected_parent = int(coupled_contract.payload["architecture"]["parent_candidate_parameter_count"])
    observed_parent = strong_baseline_parameter_count(parent_parameters)
    if observed_parent != expected_parent:
        raise ValueError(f"coupled GPP/carbon-feedback parent has {observed_parent} parameters; expected {expected_parent}")
    structured = parent_parameters["target_oriented"].get("structured_gpp")
    expected_structured = int(coupled_contract.payload["architecture"]["trainable_structured_gpp_parameter_count"])
    if strong_baseline_parameter_count(structured) != expected_structured:
        raise ValueError("coupled GPP/carbon-feedback structured parent drift")
    if structured_contract.sha256 != coupled_contract.payload["identity"]["parent_candidate_contract_sha256"]:
        raise ValueError("coupled GPP/carbon-feedback parent contract drift")

    keys = iter(jax.random.split(key, 32))
    injections = []
    for _ in range(3):
        projection = _linear_parameters(next(keys), config.width, config.width)
        projection["kernel"] = projection["kernel"] * config.residual_initialization_scale
        injections.append(projection)
    adapter = {
        "value_input": _linear_parameters(next(keys), 2, config.width),
        "field_embedding": _normal(next(keys), (len(state_layout.field_names), config.width), config.width),
        "axis_role_embedding": _normal(next(keys), (len(state_layout.axis_role_names), config.width), config.width),
        "axis_position_embedding": tuple(
            _normal(next(keys), (extent, config.width), config.width)
            for extent in state_layout.axis_role_extents
        ),
        "query_projection": _linear_parameters(next(keys), config.width, config.width),
        "blocks": tuple(
            _residual_block_parameters(next(keys), config.width, config.ffn_width)
            for _ in range(config.cross_attention_blocks)
        ),
        "residual_injections": tuple(injections),
    }
    expected_adapter = int(coupled_contract.payload["architecture"]["new_adapter_parameter_count"])
    observed_adapter = strong_baseline_parameter_count(adapter)
    if observed_adapter != expected_adapter:
        raise ValueError(f"coupled GPP/carbon-feedback adapter has {observed_adapter} parameters; expected {expected_adapter}")

    parameters = dict(parent_parameters)
    target = dict(parent_parameters["target_oriented"])
    if PARAMETER_FAMILY in target:
        raise ValueError("coupled GPP/carbon-feedback adapter already exists")
    target[PARAMETER_FAMILY] = adapter
    parameters["target_oriented"] = target
    expected_total = int(coupled_contract.payload["architecture"]["candidate_parameter_count"])
    observed_total = strong_baseline_parameter_count(parameters)
    if observed_total != expected_total:
        raise ValueError(f"coupled GPP/carbon-feedback candidate has {observed_total} parameters; expected {expected_total}")
    return parameters


def carbon_feedback_tokens(
    adapter: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    normalization: NeuralInputNormalization,
    state_layout: CarbonFeedbackStateLayout,
) -> tuple[Any, Any]:
    indices = jnp.asarray(state_layout.state_indices, dtype=jnp.int32)
    raw = jnp.asarray(inputs.state.other, dtype=jnp.float32)[indices]
    mean = jnp.asarray(normalization.continuous_mean, dtype=jnp.float32)[indices]
    scale = jnp.asarray(normalization.continuous_scale, dtype=jnp.float32)[indices]
    defined = jnp.asarray(inputs.state_defined.other, dtype=jnp.bool_)[indices] & jnp.isfinite(raw)
    normalized = _standardize(jnp.where(defined, raw, mean), mean, scale)
    features = jnp.stack((normalized, defined.astype(jnp.float32)), axis=1)
    tokens = _linear(adapter["value_input"], features)
    tokens += adapter["field_embedding"][jnp.asarray(state_layout.field_index, dtype=jnp.int32)]

    roles = jnp.asarray(state_layout.axis_role_index, dtype=jnp.int32)
    positions = jnp.asarray(state_layout.axis_position_index, dtype=jnp.int32)
    valid_roles = roles >= 0
    safe_roles = jnp.maximum(roles, 0)
    role_tokens = adapter["axis_role_embedding"][safe_roles]
    tokens += jnp.sum(jnp.where(valid_roles[..., None], role_tokens, 0.0), axis=1)
    for role, embedding in enumerate(adapter["axis_position_embedding"]):
        selected = roles == role
        safe_positions = jnp.where(selected, positions, 0)
        position_tokens = embedding[safe_positions]
        tokens += jnp.sum(jnp.where(selected[..., None], position_tokens, 0.0), axis=1)
    return jnp.where(defined[:, None], tokens, 0.0), defined


def carbon_feedback_query_residuals(
    parameters: Mapping[str, Any],
    base_query: Any,
    active: Any,
    inputs: AssembledDailyOperatorInput,
    *,
    state_layout: CarbonFeedbackStateLayout,
    input_normalization: NeuralInputNormalization,
    config: CoupledGPPCarbonFeedbackConfig = CoupledGPPCarbonFeedbackConfig(),
) -> tuple[Any, Any, Any]:
    """Read global and same-PFT carbon coordinates with shared query weights."""

    adapter = parameters["target_oriented"][PARAMETER_FAMILY]
    tokens, defined = carbon_feedback_tokens(adapter, inputs, input_normalization, state_layout)
    owners = jnp.asarray(state_layout.pft_owner_index, dtype=jnp.int32)
    n_pft = int(base_query.shape[0])
    query_pft = jnp.arange(n_pft, dtype=jnp.int32)[:, None]
    bank_mask = defined[None, :] & ((owners[None, :] == -1) | (owners[None, :] == query_pft))
    bank_mask &= jnp.asarray(active, dtype=jnp.bool_)[:, None]

    encoded = _linear(adapter["query_projection"], base_query)[:, None, :]
    bank = jnp.broadcast_to(tokens[None, :, :], (n_pft, tokens.shape[0], config.width))
    for block in adapter["blocks"]:
        encoded = _layer_norm(
            encoded
            + _attention(
                block["attention"],
                encoded,
                bank,
                heads=config.attention_heads,
                mask=bank_mask[:, None, :],
            ),
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        encoded = _layer_norm(
            encoded + _ffn(block["ffn"], encoded),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
    carbon_context = jnp.where(jnp.asarray(active, dtype=jnp.bool_)[:, None], encoded[:, 0, :], 0.0)
    return tuple(_linear(projection, carbon_context) for projection in adapter["residual_injections"])


def coupled_gpp_carbon_feedback_base_response(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    gpp_hidden: Any,
    inputs: AssembledDailyOperatorInput,
    *,
    layout: StrongBaselineLayout,
    structured_state_layout: StructuredGPPStateLayout,
    carbon_state_layout: CarbonFeedbackStateLayout,
    input_normalization: NeuralInputNormalization,
    output_normalization: NeuralOutputNormalization,
    structured_config: StructuredGPPConfig = StructuredGPPConfig(),
    config: CoupledGPPCarbonFeedbackConfig = CoupledGPPCarbonFeedbackConfig(),
) -> Any:
    query_state: StructuredGPPQueryState = structured_gpp_query_state(
        parameters,
        context,
        gpp_hidden,
        inputs,
        state_layout=structured_state_layout,
        input_normalization=input_normalization,
        config=structured_config,
    )
    residuals = carbon_feedback_query_residuals(
        parameters,
        query_state.base_query,
        query_state.local_active,
        inputs,
        state_layout=carbon_state_layout,
        input_normalization=input_normalization,
        config=config,
    )
    return structured_gpp_base_response(
        parameters,
        context,
        gpp_hidden,
        inputs,
        layout=layout,
        state_layout=structured_state_layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        config=structured_config,
        query_state=query_state,
        query_residuals=residuals,
    )


def target_oriented_coupled_gpp_carbon_feedback_process_outputs(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    structured_state_layout: StructuredGPPStateLayout,
    carbon_state_layout: CarbonFeedbackStateLayout,
    input_normalization: NeuralInputNormalization,
    output_normalization: NeuralOutputNormalization,
    forcing_layout: NativeForcingFieldLayout,
    target_contract: TargetOrientedContract,
    distribution_contract: DistributionAwareMaintenanceContract,
    coeff_maint_zero: Any,
    ext_coeff_vegetfrac: Any,
    message_gate_scale: float = 1.0,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    attention_config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
    distribution_config: DistributionAwareMaintenanceConfig = DistributionAwareMaintenanceConfig(),
    source_config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig(),
    structured_config: StructuredGPPConfig = StructuredGPPConfig(),
    config: CoupledGPPCarbonFeedbackConfig = CoupledGPPCarbonFeedbackConfig(),
) -> Mapping[str, Any]:
    """Materialize the parent outputs and replace only GPP with carbon feedback."""

    del distribution_contract
    outputs, hidden = _target_oriented_attention_decoded_outputs(
        parameters,
        context,
        inputs,
        schema=schema,
        layout=layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        forcing_layout=forcing_layout,
        target_contract=target_contract,
        message_gate_scale=message_gate_scale,
        baseline_config=baseline_config,
        config=attention_config,
        deferred_primary_fields=frozenset((GPP_FIELD, MAINTENANCE_FIELD)),
    )
    maintenance_correction = distribution_aware_maintenance_corrections(
        parameters,
        context,
        hidden[MAINTENANCE_FIELD],
        config=distribution_config,
    )
    outputs[MAINTENANCE_FIELD] = materialize_part_resolved_source_shaped_maintenance(
        inputs,
        outputs,
        coeff_maint_zero,
        ext_coeff_vegetfrac,
        maintenance_correction,
        layout=layout,
        forcing_layout=forcing_layout,
        config=source_config,
    )
    outputs[GPP_FIELD] = coupled_gpp_carbon_feedback_base_response(
        parameters,
        context,
        hidden[GPP_FIELD],
        inputs,
        layout=layout,
        structured_state_layout=structured_state_layout,
        carbon_state_layout=carbon_state_layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        structured_config=structured_config,
        config=config,
    )
    return apply_fast_parameter_routes(
        outputs,
        inputs,
        layout,
        forcing_layout,
        maintenance_already_routed=True,
    )


def coupled_gpp_carbon_feedback_parameter_count(parameters: Mapping[str, Any]) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "CARBON_COMPONENT",
    "CarbonFeedbackStateLayout",
    "CoupledGPPCarbonFeedbackConfig",
    "CoupledGPPCarbonFeedbackContract",
    "build_carbon_feedback_state_layout",
    "carbon_feedback_query_residuals",
    "carbon_feedback_tokens",
    "coupled_gpp_carbon_feedback_base_response",
    "coupled_gpp_carbon_feedback_parameter_count",
    "initialize_coupled_gpp_carbon_feedback_parameters",
    "load_coupled_gpp_carbon_feedback_contract",
    "target_oriented_coupled_gpp_carbon_feedback_process_outputs",
]
