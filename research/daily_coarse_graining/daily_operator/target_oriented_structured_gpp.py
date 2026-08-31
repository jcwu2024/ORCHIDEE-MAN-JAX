"""Structured daily GPP specialist over the accepted target-oriented parent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import DailyMarkovContract

from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .parameter_routing import (
    DEFAULT_PFT_PARAMETER_ROUTING,
    PFT_TRAIT_BARE_SOIL,
    apply_fast_parameter_routes,
)
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
from .target_oriented_attention import (
    TargetOrientedAttentionConfig,
    _target_oriented_attention_decoded_outputs,
)
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
from .types import AssembledDailyOperatorInput

SCHEMA_VERSION = "target_oriented_structured_gpp_v1"
FROZEN_CONTRACT_SHA256 = "ce211fa7c6f5a09bd900e02e31a864b61fb32f6b06ee96d4262402c513d620ea"
GPP_FIELD = "daily.fast_gpp"
FORCING_FIELDS = (
    "Tair",
    "PSurf",
    "Qair",
    "Wind_E",
    "Wind_N",
    "Rainf",
    "Snowf",
    "SWdown",
    "LWdown",
)
POTENTIAL_FORCING_FIELDS = frozenset(("Tair", "PSurf", "Qair", "Wind_E", "Wind_N", "SWdown", "LWdown"))
POTENTIAL_FORCING_MASK = np.tile(
    np.asarray(
        [name in POTENTIAL_FORCING_FIELDS for name in FORCING_FIELDS],
        dtype=np.bool_,
    ),
    5,
)
DIRECT_STATE_KEYS = (
    "slowproc_stomate_previous_step_state.assim_param",
    "slowproc_stomate_previous_step_state.temp_growth",
    "hydrol_previous_step_state.humrel",
    "hydrol_previous_step_state.qsintveg",
    "diffuco_previous_step_state.control_salinity",
    "diffuco_previous_step_state.control_inudate",
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "manifests" / "coarse_graining" / "target_oriented_structured_gpp_v1.json"
)


@dataclass(frozen=True)
class StructuredGPPContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


class StructuredGPPStateSlice(NamedTuple):
    start: int
    stop: int
    shape: tuple[int, ...]


class StructuredGPPStateLayout(NamedTuple):
    assim_param: StructuredGPPStateSlice
    temp_growth: StructuredGPPStateSlice
    humrel: StructuredGPPStateSlice
    qsintveg: StructuredGPPStateSlice
    control_salinity: StructuredGPPStateSlice
    control_inudate: StructuredGPPStateSlice


class StructuredGPPQueryState(NamedTuple):
    pft_tokens: Any
    parent_tokens: Any
    local_tokens: Any
    local_active: Any
    base_query: Any


@dataclass(frozen=True)
class StructuredGPPConfig:
    width: int = 384
    attention_heads: int = 8
    ffn_width: int = 1536
    forcing_blocks: int = 2
    potential_blocks: int = 2
    limitation_blocks: int = 2
    correction_blocks: int = 1
    local_feature_width: int = 12
    parent_hidden_width: int = 768
    correction_log_bound: float = float(np.log(4.0))
    final_kernel_initialization_scale: float = 1.0e-3

    def validate(
        self,
        contract: StructuredGPPContract,
        attention_config: TargetOrientedAttentionConfig,
    ) -> None:
        architecture = contract.payload["architecture"]
        observed = (
            self.width,
            self.attention_heads,
            self.ffn_width,
            self.forcing_blocks,
            self.potential_blocks,
            self.limitation_blocks,
            self.correction_blocks,
        )
        declared = (
            int(architecture["width"]),
            int(architecture["attention_heads"]),
            int(architecture["ffn_width"]),
            int(architecture["forcing_attention_blocks"]),
            int(architecture["potential_cross_attention_blocks"]),
            int(architecture["limitation_cross_attention_blocks"]),
            int(architecture["correction_cross_attention_blocks"]),
        )
        if observed != declared:
            raise ValueError("structured GPP architecture drift")
        if self.width != attention_config.d_model or self.parent_hidden_width != attention_config.primary_decoder_width:
            raise ValueError("structured GPP parent width drift")
        if self.width % self.attention_heads:
            raise ValueError("structured GPP width must divide its head count")
        if self.final_kernel_initialization_scale != float(architecture["final_kernel_initialization_scale"]):
            raise ValueError("structured GPP final initialization drift")
        correction = tuple(
            float(value) for value in contract.payload["output_factorization"]["multiplicative_correction_range"]
        )
        if correction != (0.25, 4.0) or not np.isclose(self.correction_log_bound, np.log(4.0)):
            raise ValueError("structured GPP correction bound drift")


def load_structured_gpp_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_after_user_architecture_approval_before_local_implementation"
    ):
        raise ValueError("structured GPP contract schema drift")
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("structured GPP contract is not frozen v1")
    boundary = payload["input_boundary"]
    if tuple(boundary["native_forcing_fields"]) != FORCING_FIELDS:
        raise ValueError("structured GPP forcing registry drift")
    if tuple(boundary["direct_dynamic_state"]) != DIRECT_STATE_KEYS:
        raise ValueError("structured GPP state registry drift")
    if int(payload["architecture"]["new_parameter_count"]) != 12_879_363:
        raise ValueError("structured GPP parameter count drift")
    if int(payload["architecture"]["candidate_parameter_count"]) != 106_383_527:
        raise ValueError("structured GPP candidate count drift")
    return StructuredGPPContract(payload, source, observed)


def _state_slice(contract: DailyMarkovContract, key: str) -> StructuredGPPStateSlice:
    matches = tuple(leaf for leaf in contract.state_leaves if leaf.key == key)
    if len(matches) != 1 or matches[0].discrete:
        raise ValueError(f"structured GPP requires one continuous state owner for {key}")
    leaf = matches[0]
    return StructuredGPPStateSlice(int(leaf.start), int(leaf.stop), tuple(int(v) for v in leaf.shape))


def build_structured_gpp_state_layout(
    contract: DailyMarkovContract,
) -> StructuredGPPStateLayout:
    """Resolve the six named source-owned state slices once on the host."""

    slices = {key: _state_slice(contract, key) for key in DIRECT_STATE_KEYS}
    n_pft = len(contract.active_pft_indices)
    expected = {
        DIRECT_STATE_KEYS[0]: (1, n_pft, 1),
        DIRECT_STATE_KEYS[1]: (1,),
        DIRECT_STATE_KEYS[2]: (1, n_pft),
        DIRECT_STATE_KEYS[3]: (1, n_pft),
        DIRECT_STATE_KEYS[4]: (1,),
        DIRECT_STATE_KEYS[5]: (1,),
    }
    for key, shape in expected.items():
        if slices[key].shape != shape:
            raise ValueError(f"structured GPP state owner {key} has shape {slices[key].shape}; expected {shape}")
    return StructuredGPPStateLayout(
        assim_param=slices[DIRECT_STATE_KEYS[0]],
        temp_growth=slices[DIRECT_STATE_KEYS[1]],
        humrel=slices[DIRECT_STATE_KEYS[2]],
        qsintveg=slices[DIRECT_STATE_KEYS[3]],
        control_salinity=slices[DIRECT_STATE_KEYS[4]],
        control_inudate=slices[DIRECT_STATE_KEYS[5]],
    )


def resolve_gpp_field_index(
    schema: NeuralOutputSchema,
    target_contract: TargetOrientedContract,
) -> int:
    del target_contract
    matches = tuple(index for index, field in enumerate(schema.fields) if field.name == GPP_FIELD)
    if len(matches) != 1 or schema.fields[matches[0]].shape != (1, len(schema.active_pft_indices)):
        raise ValueError("structured GPP requires one compact-PFT daily.fast_gpp field")
    return matches[0]


def initialize_structured_gpp_parameters(
    parent_parameters: Mapping[str, Any],
    key: Any,
    *,
    schema: NeuralOutputSchema,
    target_contract: TargetOrientedContract,
    structured_contract: StructuredGPPContract,
    attention_config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
    config: StructuredGPPConfig = StructuredGPPConfig(),
) -> Mapping[str, Any]:
    """Remove the old scalar GPP projection and install the approved specialist."""

    config.validate(structured_contract, attention_config)
    observed_parent = strong_baseline_parameter_count(parent_parameters)
    expected_parent = int(structured_contract.payload["identity"]["parent_parameter_count"])
    if observed_parent != expected_parent:
        raise ValueError(f"structured GPP parent has {observed_parent} parameters; expected {expected_parent}")

    field_index = resolve_gpp_field_index(schema, target_contract)
    parameters = dict(parent_parameters)
    target = dict(parent_parameters["target_oriented"])
    primary_heads = list(target["primary_heads"])
    removed = primary_heads[field_index]
    if strong_baseline_parameter_count(removed) != 769:
        raise ValueError("structured GPP removed head size drift")
    primary_heads[field_index] = {}
    target["primary_heads"] = tuple(primary_heads)

    keys = iter(jax.random.split(key, 32))
    output_parameters = []
    for _ in range(3):
        output = _linear_parameters(next(keys), config.width, 1)
        output["kernel"] = output["kernel"] * config.final_kernel_initialization_scale
        output_parameters.append(output)
    target["structured_gpp"] = {
        "forcing_input": _linear_parameters(next(keys), 5, config.width),
        "forcing_field_embedding": _normal(next(keys), (len(FORCING_FIELDS), config.width), config.width),
        "forcing_position_embedding": _normal(next(keys), (5, config.width), config.width),
        "branch_embedding": _normal(next(keys), (3, config.width), config.width),
        "parent_hidden_projection": _linear_parameters(next(keys), config.parent_hidden_width, config.width),
        "local_input": _linear_parameters(next(keys), config.local_feature_width, config.width),
        "pft_query_projection": _linear_parameters(next(keys), config.width, config.width),
        "forcing_blocks": tuple(
            _residual_block_parameters(next(keys), config.width, config.ffn_width) for _ in range(config.forcing_blocks)
        ),
        "potential_blocks": tuple(
            _residual_block_parameters(next(keys), config.width, config.ffn_width)
            for _ in range(config.potential_blocks)
        ),
        "limitation_blocks": tuple(
            _residual_block_parameters(next(keys), config.width, config.ffn_width)
            for _ in range(config.limitation_blocks)
        ),
        "correction_blocks": tuple(
            _residual_block_parameters(next(keys), config.width, config.ffn_width)
            for _ in range(config.correction_blocks)
        ),
        "potential_output": output_parameters[0],
        "limitation_output": output_parameters[1],
        "correction_output": output_parameters[2],
    }
    parameters["target_oriented"] = target

    specialist_count = strong_baseline_parameter_count(target["structured_gpp"])
    expected_specialist = int(structured_contract.payload["architecture"]["new_parameter_count"])
    if specialist_count != expected_specialist:
        raise ValueError(f"structured GPP specialist has {specialist_count} parameters; expected {expected_specialist}")
    observed = strong_baseline_parameter_count(parameters)
    expected = int(structured_contract.payload["architecture"]["candidate_parameter_count"])
    if observed != expected:
        raise ValueError(f"structured GPP candidate has {observed} parameters; expected {expected}")
    return parameters


def _structured_forcing_tokens(
    specialist: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    normalization: NeuralInputNormalization,
    config: StructuredGPPConfig,
) -> tuple[Any, Any, Any]:
    forcing = inputs.forcing
    values = _standardize(forcing.values, normalization.forcing_mean, normalization.forcing_scale)
    records, fields = values.shape
    if records != 5 or fields != len(FORCING_FIELDS):
        raise ValueError("structured GPP requires exactly five by nine native forcing values")
    time = jnp.repeat(jnp.asarray(forcing.record_time_seconds, dtype=jnp.float32) / 86400.0, fields)
    duration = jnp.repeat(jnp.asarray(forcing.duration_seconds, dtype=jnp.float32) / 86400.0, fields)
    predecessor = jnp.repeat(jnp.asarray(forcing.predecessor_mask, dtype=jnp.float32), fields)
    valid = jnp.repeat(jnp.asarray(forcing.record_mask, dtype=jnp.bool_), fields)
    features = jnp.stack((values.reshape(-1), time, duration, predecessor, valid.astype(jnp.float32)), axis=1)
    field_index = jnp.tile(jnp.arange(fields, dtype=jnp.int32), records)
    record_index = jnp.repeat(jnp.arange(records, dtype=jnp.int32), fields)
    tokens = _linear(specialist["forcing_input"], features)
    tokens += specialist["forcing_field_embedding"][field_index]
    tokens += specialist["forcing_position_embedding"][record_index]
    pair_mask = valid[:, None] & valid[None, :]
    for block in specialist["forcing_blocks"]:
        tokens = _layer_norm(
            tokens
            + _attention(
                block["attention"],
                tokens,
                tokens,
                heads=config.attention_heads,
                mask=pair_mask,
            ),
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        tokens = _layer_norm(
            tokens + _ffn(block["ffn"], tokens),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
        tokens = jnp.where(valid[:, None], tokens, 0.0)
    return tokens, valid, field_index


def _normalized_state_slice(
    inputs: AssembledDailyOperatorInput,
    normalization: NeuralInputNormalization,
    spec: StructuredGPPStateSlice,
) -> tuple[Any, Any, Any]:
    raw = jnp.asarray(inputs.state.other[spec.start : spec.stop], dtype=jnp.float32).reshape(spec.shape)
    mean = jnp.asarray(normalization.continuous_mean[spec.start : spec.stop], dtype=jnp.float32).reshape(spec.shape)
    scale = jnp.asarray(normalization.continuous_scale[spec.start : spec.stop], dtype=jnp.float32).reshape(spec.shape)
    defined = jnp.asarray(inputs.state_defined.other[spec.start : spec.stop], dtype=jnp.bool_).reshape(spec.shape)
    normalized = _standardize(jnp.where(defined, raw, mean), mean, scale)
    return raw, normalized, defined


def structured_gpp_local_tokens(
    specialist: Mapping[str, Any],
    inputs: AssembledDailyOperatorInput,
    normalization: NeuralInputNormalization,
    state_layout: StructuredGPPStateLayout,
) -> tuple[Any, Any]:
    """Encode six named dynamic source owners without exposing other parameters."""

    assim_raw, _, assim_defined = _normalized_state_slice(inputs, normalization, state_layout.assim_param)
    _, temp_growth, temp_defined = _normalized_state_slice(inputs, normalization, state_layout.temp_growth)
    _, humrel, humrel_defined = _normalized_state_slice(inputs, normalization, state_layout.humrel)
    _, qsintveg, qsint_defined = _normalized_state_slice(inputs, normalization, state_layout.qsintveg)
    _, salinity, salinity_defined = _normalized_state_slice(inputs, normalization, state_layout.control_salinity)
    _, inudate, inudate_defined = _normalized_state_slice(inputs, normalization, state_layout.control_inudate)

    n_pft = inputs.pft.pft_parameters.shape[0]
    vcmax_index = DEFAULT_PFT_PARAMETER_ROUTING.index("vcmax25")
    vcmax25 = jnp.asarray(inputs.pft.pft_parameters[:, vcmax_index], dtype=jnp.float32)
    dynamic = assim_raw.reshape((n_pft,))
    dynamic_defined = assim_defined.reshape((n_pft,)) & jnp.isfinite(vcmax25) & (vcmax25 > 0.0)
    denominator = jnp.where(vcmax25 > 1.0e-6, vcmax25, 1.0)
    dynamic_ratio = jnp.log1p(jnp.maximum(dynamic, 0.0) / denominator)

    values = jnp.stack(
        (
            dynamic_ratio,
            humrel.reshape((n_pft,)),
            qsintveg.reshape((n_pft,)),
            jnp.broadcast_to(temp_growth.reshape(()), (n_pft,)),
            jnp.broadcast_to(salinity.reshape(()), (n_pft,)),
            jnp.broadcast_to(inudate.reshape(()), (n_pft,)),
        ),
        axis=1,
    )
    defined = jnp.stack(
        (
            dynamic_defined,
            humrel_defined.reshape((n_pft,)),
            qsint_defined.reshape((n_pft,)),
            jnp.broadcast_to(temp_defined.reshape(()), (n_pft,)),
            jnp.broadcast_to(salinity_defined.reshape(()), (n_pft,)),
            jnp.broadcast_to(inudate_defined.reshape(()), (n_pft,)),
        ),
        axis=1,
    )
    features = jnp.concatenate((jnp.where(defined, values, 0.0), defined.astype(jnp.float32)), axis=1)
    return _linear(specialist["local_input"], features), jnp.all(defined, axis=1)


def _run_gpp_branch(
    specialist: Mapping[str, Any],
    query: Any,
    bank: Any,
    bank_mask: Any,
    blocks: tuple[Mapping[str, Any], ...],
    *,
    active: Any,
    config: StructuredGPPConfig,
) -> Any:
    encoded = query[:, None, :]
    for block in blocks:
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
    return jnp.where(active[:, None], encoded[:, 0, :], 0.0)


def structured_gpp_query_state(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    gpp_hidden: Any,
    inputs: AssembledDailyOperatorInput,
    *,
    state_layout: StructuredGPPStateLayout,
    input_normalization: NeuralInputNormalization,
    config: StructuredGPPConfig = StructuredGPPConfig(),
) -> StructuredGPPQueryState:
    """Build the shared per-PFT query without changing the output factorization."""

    specialist = parameters["target_oriented"]["structured_gpp"]
    local_tokens, local_defined = structured_gpp_local_tokens(
        specialist,
        inputs,
        input_normalization,
        state_layout,
    )
    pft_mask = jnp.asarray(context.pft_mask, dtype=jnp.bool_)
    n_pft = pft_mask.shape[0]
    hidden = jnp.asarray(gpp_hidden, dtype=jnp.float32)
    if hidden.shape != (1, n_pft, config.parent_hidden_width):
        raise ValueError("structured GPP parent hidden shape drift")
    parent_tokens = _linear(specialist["parent_hidden_projection"], hidden[0])
    pft_tokens = _linear(specialist["pft_query_projection"], context.pft_tokens)
    local_active = pft_mask & local_defined
    return StructuredGPPQueryState(
        pft_tokens=pft_tokens,
        parent_tokens=parent_tokens,
        local_tokens=local_tokens,
        local_active=local_active,
        base_query=pft_tokens + parent_tokens + local_tokens,
    )


def structured_gpp_base_response(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    gpp_hidden: Any,
    inputs: AssembledDailyOperatorInput,
    *,
    layout: StrongBaselineLayout,
    state_layout: StructuredGPPStateLayout,
    input_normalization: NeuralInputNormalization,
    output_normalization: NeuralOutputNormalization,
    config: StructuredGPPConfig = StructuredGPPConfig(),
    query_state: StructuredGPPQueryState | None = None,
    query_residuals: tuple[Any, Any, Any] | None = None,
) -> Any:
    """Return pre-VCMAX positive daily GPP for every compact PFT."""

    specialist = parameters["target_oriented"]["structured_gpp"]
    forcing_tokens, forcing_valid, forcing_field = _structured_forcing_tokens(
        specialist, inputs, input_normalization, config
    )
    query_state = (
        structured_gpp_query_state(
            parameters,
            context,
            gpp_hidden,
            inputs,
            state_layout=state_layout,
            input_normalization=input_normalization,
            config=config,
        )
        if query_state is None
        else query_state
    )
    pft_tokens = query_state.pft_tokens
    parent_tokens = query_state.parent_tokens
    local_tokens = query_state.local_tokens
    local_active = query_state.local_active
    base_query = query_state.base_query
    n_pft = int(base_query.shape[0])

    forcing_count = forcing_tokens.shape[0]
    parent_source = jnp.asarray(context.source_tokens[forcing_count:], dtype=jnp.float32)
    parent_source_mask = jnp.asarray(context.source_mask[forcing_count:], dtype=jnp.bool_)
    parent_source_family = jnp.asarray(context.source_family_index[forcing_count:], dtype=jnp.int32)
    canopy_index = layout.expert_names.index("canopy_gpp")
    allowed_family = jnp.asarray(layout.source_allowed[canopy_index], dtype=jnp.bool_)
    parent_source_mask &= allowed_family[parent_source_family]
    axis_tokens = jnp.asarray(context.expert_axis_tokens[canopy_index], dtype=jnp.float32)
    axis_mask = jnp.asarray(context.expert_axis_mask[canopy_index], dtype=jnp.bool_)

    shared_bank = jnp.concatenate(
        (forcing_tokens, parent_source, axis_tokens),
        axis=0,
    )
    shared_bank = jnp.broadcast_to(
        shared_bank[None, :, :],
        (n_pft, shared_bank.shape[0], config.width),
    )
    local_bank = jnp.stack((pft_tokens, local_tokens, parent_tokens), axis=1)
    bank = jnp.concatenate((shared_bank, local_bank), axis=1)
    potential_forcing = jnp.asarray(POTENTIAL_FORCING_MASK, dtype=jnp.bool_)
    if potential_forcing.shape != forcing_field.shape:
        raise ValueError("structured GPP forcing-field registry drift")
    common_tail_mask = jnp.concatenate(
        (
            parent_source_mask,
            axis_mask,
            jnp.ones((3,), dtype=jnp.bool_),
        )
    )
    potential_mask = jnp.broadcast_to(
        jnp.concatenate((forcing_valid & potential_forcing, common_tail_mask))[None, :],
        (n_pft, bank.shape[1]),
    )
    full_mask = jnp.broadcast_to(
        jnp.concatenate((forcing_valid, common_tail_mask))[None, :],
        (n_pft, bank.shape[1]),
    )
    potential_mask &= local_active[:, None]
    full_mask &= local_active[:, None]

    if query_residuals is None:
        query_residuals = tuple(jnp.zeros_like(base_query) for _ in range(3))
    if len(query_residuals) != 3 or any(value.shape != base_query.shape for value in query_residuals):
        raise ValueError("structured GPP query residual shape drift")
    potential_hidden = _run_gpp_branch(
        specialist,
        base_query + specialist["branch_embedding"][0] + query_residuals[0],
        bank,
        potential_mask,
        specialist["potential_blocks"],
        active=local_active,
        config=config,
    )
    limitation_hidden = _run_gpp_branch(
        specialist,
        base_query + specialist["branch_embedding"][1] + query_residuals[1],
        bank,
        full_mask,
        specialist["limitation_blocks"],
        active=local_active,
        config=config,
    )
    correction_hidden = _run_gpp_branch(
        specialist,
        base_query + specialist["branch_embedding"][2] + query_residuals[2],
        bank,
        full_mask,
        specialist["correction_blocks"],
        active=local_active,
        config=config,
    )
    potential = jax.nn.softplus(_linear(specialist["potential_output"], potential_hidden)[:, 0])
    limitation = jax.nn.sigmoid(_linear(specialist["limitation_output"], limitation_hidden)[:, 0])
    correction = jnp.exp(
        jnp.asarray(config.correction_log_bound, dtype=jnp.float64)
        * jnp.tanh(_linear(specialist["correction_output"], correction_hidden)[:, 0])
    )
    scale = jnp.asarray(output_normalization.scale[GPP_FIELD], dtype=jnp.float64).reshape((n_pft,))
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    traits = jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)
    fraction = jnp.asarray(inputs.pft.pft_fraction, dtype=jnp.float64)
    pft_mask = jnp.asarray(context.pft_mask, dtype=jnp.bool_)
    vegetation = pft_mask & (fraction > 0.0) & (traits[:, PFT_TRAIT_BARE_SOIL] <= 0.5)
    base = scale * jnp.asarray(potential, dtype=jnp.float64) * jnp.asarray(limitation, dtype=jnp.float64) * correction
    return jnp.where(vegetation, base, 0.0)[None, :]


def target_oriented_structured_gpp_process_outputs(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    state_layout: StructuredGPPStateLayout,
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
    config: StructuredGPPConfig = StructuredGPPConfig(),
) -> Mapping[str, Any]:
    """Materialize only GPP and maintenance through their accepted specialists."""

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
    outputs[GPP_FIELD] = structured_gpp_base_response(
        parameters,
        context,
        hidden[GPP_FIELD],
        inputs,
        layout=layout,
        state_layout=state_layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        config=config,
    )
    return apply_fast_parameter_routes(
        outputs,
        inputs,
        layout,
        forcing_layout,
        maintenance_already_routed=True,
    )


def structured_gpp_parameter_count(parameters: Mapping[str, Any]) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "DIRECT_STATE_KEYS",
    "FORCING_FIELDS",
    "GPP_FIELD",
    "StructuredGPPConfig",
    "StructuredGPPContract",
    "StructuredGPPQueryState",
    "StructuredGPPStateLayout",
    "build_structured_gpp_state_layout",
    "initialize_structured_gpp_parameters",
    "load_structured_gpp_contract",
    "resolve_gpp_field_index",
    "structured_gpp_base_response",
    "structured_gpp_local_tokens",
    "structured_gpp_parameter_count",
    "structured_gpp_query_state",
    "target_oriented_structured_gpp_process_outputs",
]
