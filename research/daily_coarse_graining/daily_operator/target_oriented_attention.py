"""Target-oriented hierarchical coordinate Attention candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .output_transform import normalized_output_to_physical
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
    encode_strong_baseline_experts,
    initialize_strong_baseline_parameters,
    strong_baseline_field_context,
    strong_baseline_parameter_count,
)
from .strong_baseline_contract import StrongBaselineContract
from .target_oriented_contract import (
    AUXILIARY_SUPERVISION,
    LEARNED_CATEGORIES,
    PRIMARY_SCIENCE_DIRECT,
    STATE_BUDGET_ESSENTIAL,
    TargetOrientedContract,
)
from .types import AssembledDailyOperatorInput

CANDIDATE_NAME = "target_oriented_hierarchical_coordinate_attention_v1"
SCIENCE_TOKEN_NAMES = (
    "gpp",
    "autotrophic_respiration",
    "water_stress",
    "temperature",
    "slow_carbon",
)
DOMAIN_REFINEMENT_DEPTHS = {
    "canopy_gpp": 2,
    "surface_energy": 1,
    "hydrology_snow": 2,
    "soil_thermal": 2,
    "maintenance_respiration": 2,
    "litter_poc_doc": 2,
    "daily_interface_finalize": 1,
}
SCIENCE_DOMAIN_ADJACENCY = {
    "gpp": ("canopy_gpp", "surface_energy", "hydrology_snow", "soil_thermal"),
    "autotrophic_respiration": (
        "canopy_gpp",
        "maintenance_respiration",
        "soil_thermal",
        "litter_poc_doc",
    ),
    "water_stress": ("canopy_gpp", "surface_energy", "hydrology_snow", "soil_thermal"),
    "temperature": (
        "surface_energy",
        "hydrology_snow",
        "soil_thermal",
        "maintenance_respiration",
    ),
    "slow_carbon": (
        "canopy_gpp",
        "hydrology_snow",
        "soil_thermal",
        "maintenance_respiration",
        "litter_poc_doc",
    ),
}


@dataclass(frozen=True)
class TargetOrientedAttentionConfig:
    d_model: int = 384
    attention_heads: int = 8
    ffn_width: int = 1536
    science_cross_blocks: int = 2
    science_mixer_blocks: int = 2
    primary_blocks: int = 3
    state_blocks: int = 2
    primary_decoder_width: int = 768
    state_decoder_width: int = 512
    auxiliary_bottleneck_width: int = 192
    auxiliary_decoder_width: int = 128
    minimum_nonnegative_ratio: float = 0.01
    minimum_unit_interval_center: float = 0.0001
    minimum_parameter_count: int = 80_000_000
    target_parameter_count_maximum: int = 100_000_000
    hard_parameter_count_maximum: int = 120_000_000

    def validate(self, baseline: StrongBaselineConfig) -> None:
        if self.d_model != baseline.d_model or self.ffn_width != baseline.ffn_width:
            raise ValueError("target-oriented and source-routed backbone widths must match")
        if self.attention_heads != baseline.attention_heads:
            raise ValueError("target-oriented and source-routed head counts must match")
        if self.d_model % self.attention_heads:
            raise ValueError("target-oriented width must divide its head count")
        if (self.science_cross_blocks, self.science_mixer_blocks) != (2, 2):
            raise ValueError("target-oriented science-token depth drift")
        if (self.primary_blocks, self.state_blocks) != (3, 2):
            raise ValueError("target-oriented category decoder depth drift")
        if not (
            self.primary_decoder_width >= self.state_decoder_width > self.d_model
            and self.auxiliary_decoder_width < self.auxiliary_bottleneck_width < self.d_model
        ):
            raise ValueError("target-oriented category capacity ordering drift")
        if not (self.minimum_parameter_count < self.target_parameter_count_maximum < self.hard_parameter_count_maximum):
            raise ValueError("target-oriented parameter bounds are not ordered")
        if not 0.0 < self.minimum_nonnegative_ratio < 1.0:
            raise ValueError("target-oriented nonnegative ratio must lie in (0, 1)")
        if not 0.0 < self.minimum_unit_interval_center < 0.5:
            raise ValueError("target-oriented bounded center must lie in (0, 0.5)")


def target_oriented_field_categories(
    schema: NeuralOutputSchema,
    contract: TargetOrientedContract,
) -> tuple[str, ...]:
    """Resolve the frozen category of every learned field."""

    label_category = {label: category for category, labels in contract.labels_by_category.items() for label in labels}
    categories = []
    for field in schema.fields:
        label = schema.labels[field.label_index]
        if label not in label_category:
            raise ValueError(f"target-oriented output label is unclassified: {label}")
        categories.append(label_category[label])
    counts = {category: sum(value == category for value in categories) for category in LEARNED_CATEGORIES}
    scalar_counts = {
        category: sum(
            int(np.prod(field.shape, dtype=np.int64))
            for field, value in zip(schema.fields, categories, strict=True)
            if value == category
        )
        for category in LEARNED_CATEGORIES
    }
    if counts != {
        PRIMARY_SCIENCE_DIRECT: 2,
        STATE_BUDGET_ESSENTIAL: 93,
        AUXILIARY_SUPERVISION: 66,
    }:
        raise ValueError("target-oriented learned field counts drift")
    if scalar_counts != {
        PRIMARY_SCIENCE_DIRECT: 26,
        STATE_BUDGET_ESSENTIAL: 3_682,
        AUXILIARY_SUPERVISION: 35_298,
    }:
        raise ValueError("target-oriented learned scalar counts drift")
    return tuple(categories)


def _science_domain_mask(layout: StrongBaselineLayout) -> np.ndarray:
    if set(layout.expert_names) != set(DOMAIN_REFINEMENT_DEPTHS):
        raise ValueError("target-oriented process-domain registry drift")
    return np.asarray(
        [[expert in SCIENCE_DOMAIN_ADJACENCY[name] for expert in layout.expert_names] for name in SCIENCE_TOKEN_NAMES],
        dtype=np.bool_,
    )


def initialize_target_oriented_attention_parameters(
    key: Any,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    inputs: AssembledDailyOperatorInput,
    architecture_contract: StrongBaselineContract,
    target_contract: TargetOrientedContract,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
) -> Mapping[str, Any]:
    """Initialize a fresh target-oriented tree without historical checkpoints."""

    config.validate(baseline_config)
    categories = target_oriented_field_categories(schema, target_contract)
    _science_domain_mask(layout)
    baseline_key, candidate_key = jax.random.split(key)
    parameters = dict(
        initialize_strong_baseline_parameters(
            baseline_key,
            schema=schema,
            layout=layout,
            inputs=inputs,
            contract=architecture_contract,
            config=baseline_config,
        )
    )
    parameters.pop("label_heads")
    keys = iter(jax.random.split(candidate_key, 1024))

    domain_blocks = tuple(
        tuple(
            _residual_block_parameters(next(keys), config.d_model, config.ffn_width)
            for _ in range(DOMAIN_REFINEMENT_DEPTHS[name])
        )
        for name in layout.expert_names
    )
    primary_heads = []
    state_heads = []
    for category in categories:
        primary = (
            _linear_parameters(next(keys), config.primary_decoder_width, 1)
            if (category == PRIMARY_SCIENCE_DIRECT)
            else {}
        )
        state = (
            _linear_parameters(next(keys), config.state_decoder_width, 1)
            if (category == STATE_BUDGET_ESSENTIAL)
            else {}
        )
        if primary:
            primary["kernel"] = primary["kernel"] * 1.0e-3
        if state:
            state["kernel"] = state["kernel"] * 1.0e-3
        primary_heads.append(primary)
        state_heads.append(state)

    auxiliary_labels = {
        field.label_index
        for field, category in zip(schema.fields, categories, strict=True)
        if category == AUXILIARY_SUPERVISION
    }
    auxiliary_label_heads = []
    for label_index in range(len(schema.labels)):
        if label_index not in auxiliary_labels:
            auxiliary_label_heads.append({})
            continue
        first = _linear_parameters(
            next(keys),
            config.auxiliary_bottleneck_width,
            config.auxiliary_decoder_width,
        )
        second = _linear_parameters(next(keys), config.auxiliary_decoder_width, 1)
        second["kernel"] = second["kernel"] * 1.0e-3
        auxiliary_label_heads.append({"first": first, "second": second})

    parameters["target_oriented"] = {
        "domain_embedding": _normal(
            next(keys),
            (len(layout.expert_names), config.d_model),
            config.d_model,
        ),
        "domain_blocks": domain_blocks,
        "science_query": _normal(
            next(keys),
            (len(SCIENCE_TOKEN_NAMES), config.d_model),
            config.d_model,
        ),
        "science_embedding": _normal(
            next(keys),
            (len(SCIENCE_TOKEN_NAMES), config.d_model),
            config.d_model,
        ),
        "science_cross_blocks": tuple(
            _residual_block_parameters(next(keys), config.d_model, config.ffn_width)
            for _ in range(config.science_cross_blocks)
        ),
        "science_mixer_blocks": tuple(
            _residual_block_parameters(next(keys), config.d_model, config.ffn_width)
            for _ in range(config.science_mixer_blocks)
        ),
        "category_embedding": _normal(
            next(keys),
            (len(LEARNED_CATEGORIES), config.d_model),
            config.d_model,
        ),
        "primary_blocks": tuple(
            _residual_block_parameters(next(keys), config.d_model, config.ffn_width)
            for _ in range(config.primary_blocks)
        ),
        "state_blocks": tuple(
            _residual_block_parameters(next(keys), config.d_model, config.ffn_width) for _ in range(config.state_blocks)
        ),
        "primary_decoder": _linear_parameters(next(keys), config.d_model, config.primary_decoder_width),
        "state_decoder": _linear_parameters(next(keys), config.d_model, config.state_decoder_width),
        "primary_heads": tuple(primary_heads),
        "state_heads": tuple(state_heads),
        "auxiliary_science_projection": _linear_parameters(next(keys), config.d_model, config.d_model),
        "auxiliary_bottleneck": _linear_parameters(next(keys), config.d_model, config.auxiliary_bottleneck_width),
        "auxiliary_label_heads": tuple(auxiliary_label_heads),
    }
    count = strong_baseline_parameter_count(parameters)
    if count < config.minimum_parameter_count:
        raise ValueError(f"target-oriented candidate has {count} parameters, below {config.minimum_parameter_count}")
    if count > config.hard_parameter_count_maximum:
        raise ValueError(
            f"target-oriented candidate has {count} parameters, above {config.hard_parameter_count_maximum}"
        )
    return parameters


def _source_family_bank(
    context: StrongBaselineContext,
    layout: StrongBaselineLayout,
    expert_index: int,
) -> tuple[Any, Any]:
    family_count = len(layout.source_family_names)
    family = jnp.asarray(context.source_family_index, dtype=jnp.int32)
    selected = family[None, :] == jnp.arange(family_count, dtype=jnp.int32)[:, None]
    selected &= jnp.asarray(context.source_mask, dtype=jnp.bool_)[None, :]
    selected &= jnp.asarray(layout.source_allowed[expert_index], dtype=jnp.bool_)[:, None]
    count = jnp.sum(selected, axis=1, dtype=jnp.float32)
    tokens = jnp.sum(
        selected[:, :, None].astype(jnp.float64) * jnp.asarray(context.source_tokens, dtype=jnp.float64)[None, :, :],
        axis=1,
        dtype=jnp.float64,
    ) / jnp.maximum(count[:, None], 1.0)
    valid = count > 0.0
    return jnp.asarray(jnp.where(valid[:, None], tokens, 0.0), dtype=jnp.float32), valid


def encode_target_oriented_domains(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    *,
    layout: StrongBaselineLayout,
    message_gate_scale: float,
    baseline_config: StrongBaselineConfig,
    config: TargetOrientedAttentionConfig,
) -> tuple[Any, Any]:
    """Refine process domains and form the five cross-domain science tokens."""

    base_domains = encode_strong_baseline_experts(
        parameters,
        context,
        layout=layout,
        message_gate_scale=message_gate_scale,
        config=baseline_config,
    )
    target = parameters["target_oriented"]
    family = jnp.asarray(context.source_family_index, dtype=jnp.int32)
    allowed = jnp.asarray(layout.source_allowed, dtype=jnp.bool_)
    graph = jnp.asarray(layout.graph_edge_mask, dtype=jnp.bool_)
    refined = []
    for expert_index, blocks in enumerate(target["domain_blocks"]):
        source_mask = allowed[expert_index, family] & context.source_mask
        domain_mask = graph[:, expert_index].at[expert_index].set(True)
        bank = jnp.concatenate(
            (
                context.source_tokens,
                context.expert_axis_tokens[expert_index],
                base_domains,
            ),
            axis=0,
        )
        mask = jnp.concatenate(
            (
                source_mask,
                context.expert_axis_mask[expert_index],
                domain_mask,
            )
        )
        query = (base_domains[expert_index] + target["domain_embedding"][expert_index])[None, :]
        for block in blocks:
            message = _attention(
                block["attention"],
                query,
                bank,
                heads=config.attention_heads,
                mask=mask[None, :],
            )
            query = _layer_norm(
                query + message,
                block["attention_norm_scale"],
                block["attention_norm_bias"],
            )
            query = _layer_norm(
                query + _ffn(block["ffn"], query),
                block["ffn_norm_scale"],
                block["ffn_norm_bias"],
            )
        refined.append(query[0])
    domain_tokens = jnp.stack(tuple(refined))

    science = target["science_query"] + target["science_embedding"]
    science_mask = jnp.asarray(_science_domain_mask(layout), dtype=jnp.bool_)
    for block in target["science_cross_blocks"]:
        message = _attention(
            block["attention"],
            science,
            domain_tokens,
            heads=config.attention_heads,
            mask=science_mask,
        )
        science = _layer_norm(
            science + message,
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        science = _layer_norm(
            science + _ffn(block["ffn"], science),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
    full_mask = jnp.ones(
        (len(SCIENCE_TOKEN_NAMES), len(SCIENCE_TOKEN_NAMES)),
        dtype=jnp.bool_,
    )
    for block in target["science_mixer_blocks"]:
        message = _attention(
            block["attention"],
            science,
            science,
            heads=config.attention_heads,
            mask=full_mask,
        )
        science = _layer_norm(
            science + message,
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        science = _layer_norm(
            science + _ffn(block["ffn"], science),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
    return domain_tokens, science


def _coordinate_attention(
    parameters: Mapping[str, Any],
    field_context: Any,
    family_tokens: Any,
    family_mask: Any,
    science_tokens: Any,
    *,
    category_index: int,
    blocks: tuple[Mapping[str, Any], ...],
    config: TargetOrientedAttentionConfig,
) -> Any:
    size = field_context.features.shape[0]
    family_bank = jnp.broadcast_to(
        family_tokens[None, :, :],
        (size, family_tokens.shape[0], config.d_model),
    )
    science_bank = jnp.broadcast_to(
        science_tokens[None, :, :],
        (size, science_tokens.shape[0], config.d_model),
    )
    bank = jnp.concatenate(
        (
            family_bank,
            field_context.axis_tokens,
            field_context.local_state_token[:, None, :],
            science_bank,
        ),
        axis=1,
    )
    mask = jnp.concatenate(
        (
            field_context.active[:, None] & family_mask[None, :],
            field_context.active[:, None] & field_context.axis_mask,
            field_context.local_state_mask[:, None],
            jnp.broadcast_to(
                field_context.active[:, None],
                (size, science_tokens.shape[0]),
            ),
        ),
        axis=1,
    )
    target = parameters["target_oriented"]
    query = (field_context.features + target["category_embedding"][category_index])[:, None, :]
    for block in blocks:
        message = _attention(
            block["attention"],
            query,
            bank,
            heads=config.attention_heads,
            mask=mask[:, None, :],
        )
        query = _layer_norm(
            query + message,
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        query = _layer_norm(
            query + _ffn(block["ffn"], query),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
    return jnp.where(field_context.active[:, None], query[:, 0, :], 0.0)


def _target_oriented_attention_decoded_outputs(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    output_normalization: NeuralOutputNormalization,
    forcing_layout: NativeForcingFieldLayout,
    target_contract: TargetOrientedContract,
    message_gate_scale: float = 1.0,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
    deferred_primary_fields: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode fields while retaining hidden coordinates for explicit materializers."""

    config.validate(baseline_config)
    categories = target_oriented_field_categories(schema, target_contract)
    domain_tokens, science_tokens = encode_target_oriented_domains(
        parameters,
        context,
        layout=layout,
        message_gate_scale=message_gate_scale,
        baseline_config=baseline_config,
        config=config,
    )
    family_banks = tuple(
        _source_family_bank(context, layout, expert_index) for expert_index in range(len(layout.expert_names))
    )
    target = parameters["target_oriented"]
    science_summary = jnp.mean(science_tokens, axis=0)
    auxiliary_science = _linear(
        target["auxiliary_science_projection"],
        science_summary,
    )
    outputs = {}
    deferred_hidden = {}
    for field_index, (field, ownership, category) in enumerate(
        zip(schema.fields, layout.ledger, categories, strict=True)
    ):
        expert_index = int(layout.field_expert_index[field_index])
        field_context = strong_baseline_field_context(
            parameters,
            domain_tokens,
            context,
            inputs,
            field=field,
            ownership=ownership,
            field_index=field_index,
            layout=layout,
            normalization=input_normalization,
        )
        if category == PRIMARY_SCIENCE_DIRECT:
            encoded = _coordinate_attention(
                parameters,
                field_context,
                *family_banks[expert_index],
                science_tokens,
                category_index=LEARNED_CATEGORIES.index(category),
                blocks=target["primary_blocks"],
                config=config,
            )
            hidden = jax.nn.gelu(_linear(target["primary_decoder"], encoded))
            if field.name in deferred_primary_fields:
                deferred_hidden[field.name] = hidden.reshape((*field.shape, config.primary_decoder_width))
                continue
            raw = _linear(target["primary_heads"][field_index], hidden)[:, 0]
        elif category == STATE_BUDGET_ESSENTIAL:
            encoded = _coordinate_attention(
                parameters,
                field_context,
                *family_banks[expert_index],
                science_tokens,
                category_index=LEARNED_CATEGORIES.index(category),
                blocks=target["state_blocks"],
                config=config,
            )
            hidden = jax.nn.gelu(_linear(target["state_decoder"], encoded))
            raw = _linear(target["state_heads"][field_index], hidden)[:, 0]
        else:
            features = (
                field_context.features
                + target["category_embedding"][LEARNED_CATEGORIES.index(category)]
                + auxiliary_science
            )
            bottleneck = jax.nn.gelu(_linear(target["auxiliary_bottleneck"], features))
            head = target["auxiliary_label_heads"][field.label_index]
            raw = _linear(head["second"], jax.nn.gelu(_linear(head["first"], bottleneck)))[:, 0]
        outputs[field.name] = normalized_output_to_physical(
            raw,
            field_context.active,
            field=field,
            normalization=output_normalization,
            inputs=inputs,
            minimum_nonnegative_ratio=config.minimum_nonnegative_ratio,
            minimum_unit_interval_center=config.minimum_unit_interval_center,
            rescale_unit_interval_delta=True,
        )
    if set(deferred_hidden) != set(deferred_primary_fields):
        missing = sorted(set(deferred_primary_fields) - set(deferred_hidden))
        raise ValueError(f"deferred target-oriented primary fields are invalid: {missing}")
    return outputs, deferred_hidden


def target_oriented_attention_process_outputs(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    inputs: AssembledDailyOperatorInput,
    *,
    schema: NeuralOutputSchema,
    layout: StrongBaselineLayout,
    input_normalization: NeuralInputNormalization,
    output_normalization: NeuralOutputNormalization,
    forcing_layout: NativeForcingFieldLayout,
    target_contract: TargetOrientedContract,
    message_gate_scale: float = 1.0,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
) -> Mapping[str, Any]:
    """Decode the complete learned schema through category-specific heads."""

    outputs, _hidden = _target_oriented_attention_decoded_outputs(
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
        config=config,
    )
    return apply_fast_parameter_routes(outputs, inputs, layout, forcing_layout)


def target_oriented_attention_parameter_count(parameters: Mapping[str, Any]) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "CANDIDATE_NAME",
    "DOMAIN_REFINEMENT_DEPTHS",
    "SCIENCE_DOMAIN_ADJACENCY",
    "SCIENCE_TOKEN_NAMES",
    "TargetOrientedAttentionConfig",
    "encode_target_oriented_domains",
    "initialize_target_oriented_attention_parameters",
    "target_oriented_attention_parameter_count",
    "target_oriented_attention_process_outputs",
    "target_oriented_field_categories",
]
