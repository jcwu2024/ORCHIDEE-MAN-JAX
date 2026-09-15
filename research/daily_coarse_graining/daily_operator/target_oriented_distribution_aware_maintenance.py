"""Part-resolved maintenance adapter over the accepted target Attention body."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from jax_orchidee.stomate.carbon_kernels import NPARTS

from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .parameter_routing import apply_fast_parameter_routes
from .process_derivations import NativeForcingFieldLayout
from .strong_baseline_architecture import (
    NeuralPrecisionPolicy,
    StrongBaselineConfig,
    StrongBaselineContext,
    StrongBaselineLayout,
    _attention,
    _ffn,
    _layer_norm,
    _linear,
    _linear_parameters,
    _residual_block_parameters,
    strong_baseline_parameter_count,
)
from .target_oriented_attention import (
    TargetOrientedAttentionConfig,
    _target_oriented_attention_decoded_outputs,
)
from .target_oriented_contract import TargetOrientedContract
from .target_oriented_source_shaped_maintenance import (
    MAINTENANCE_FIELD,
    SourceShapedMaintenanceConfig,
    contract_sha256,
    materialize_part_resolved_source_shaped_maintenance,
    resolve_maintenance_field_index,
)
from .types import AssembledDailyOperatorInput

SCHEMA_VERSION = "target_oriented_distribution_aware_maintenance_v1"
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_distribution_aware_maintenance_v1.json"
)


@dataclass(frozen=True)
class DistributionAwareMaintenanceContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class DistributionAwareMaintenanceConfig:
    context_width: int = 384
    attention_width: int = 768
    attention_heads: int = 8
    ffn_width: int = 1536
    attention_blocks: int = 2
    maximum_absolute_correction: float = 1.0
    final_kernel_initialization_scale: float = 1.0e-3

    def validate(
        self,
        contract: DistributionAwareMaintenanceContract,
        attention_config: TargetOrientedAttentionConfig,
    ) -> None:
        mechanism = contract.payload["mechanism"]
        projection = tuple(int(value) for value in mechanism["context_projection_shape"])
        observed = (
            self.context_width,
            self.attention_width,
            self.attention_heads,
            self.ffn_width,
            self.attention_blocks,
        )
        declared = (
            projection[0],
            projection[1],
            int(mechanism["attention_heads"]),
            int(mechanism["ffn_width"]),
            int(mechanism["attention_blocks"]),
        )
        if observed != declared:
            raise ValueError("distribution-aware maintenance architecture drift")
        if self.attention_width % self.attention_heads:
            raise ValueError("distribution-aware maintenance width must divide its head count")
        if self.attention_width != attention_config.primary_decoder_width:
            raise ValueError("distribution-aware maintenance parent hidden width drift")
        if self.maximum_absolute_correction != float(mechanism["maximum_absolute_correction"]):
            raise ValueError("distribution-aware maintenance correction bound drift")
        if self.final_kernel_initialization_scale != float(mechanism["final_kernel_initialization_scale"]):
            raise ValueError("distribution-aware maintenance final initialization drift")
        if int(mechanism["output_coordinates"]) != 2 * NPARTS:
            raise ValueError("distribution-aware maintenance coordinate count drift")
        if bool(mechanism["coordinate_pooling_before_output"]):
            raise ValueError("distribution-aware maintenance forbids coordinate pooling")


def load_distribution_aware_maintenance_contract(
    path: str | Path = _CONTRACT_PATH,
) -> DistributionAwareMaintenanceContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("distribution-aware maintenance schema version drift")
    if payload.get("contract_sha256") != observed:
        raise ValueError("distribution-aware maintenance contract SHA256 mismatch")
    if int(payload["mechanism"]["new_parameter_count"]) != 9_750_529:
        raise ValueError("distribution-aware maintenance adapter parameter count drift")
    if int(payload["mechanism"]["candidate_parameter_count"]) != 93_504_933:
        raise ValueError("distribution-aware maintenance candidate parameter count drift")
    return DistributionAwareMaintenanceContract(
        payload=payload,
        source_path=source,
        sha256=observed,
    )


def initialize_distribution_aware_maintenance_parameters(
    parent_parameters: Mapping[str, Any],
    key: Any,
    *,
    schema: NeuralOutputSchema,
    target_contract: TargetOrientedContract,
    distribution_contract: DistributionAwareMaintenanceContract,
    attention_config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
    config: DistributionAwareMaintenanceConfig = DistributionAwareMaintenanceConfig(),
) -> Mapping[str, Any]:
    """Replace only the parent's direct maintenance projection."""

    config.validate(distribution_contract, attention_config)
    expected_parent = int(distribution_contract.payload["identity"]["parent_parameter_count"])
    observed_parent = strong_baseline_parameter_count(parent_parameters)
    if observed_parent != expected_parent:
        raise ValueError(
            f"distribution-aware maintenance parent has {observed_parent} parameters; expected {expected_parent}"
        )
    index = resolve_maintenance_field_index(schema, target_contract)
    parameters = dict(parent_parameters)
    target = dict(parent_parameters["target_oriented"])
    primary_heads = list(target["primary_heads"])
    removed = primary_heads[index]
    if strong_baseline_parameter_count(removed) != 769:
        raise ValueError("distribution-aware maintenance removed head size drift")
    primary_heads[index] = {}
    target["primary_heads"] = tuple(primary_heads)

    projection_key, block_key_0, block_key_1, output_key = jax.random.split(key, 4)
    output = _linear_parameters(output_key, config.attention_width, 1)
    output["kernel"] = output["kernel"] * config.final_kernel_initialization_scale
    target["distribution_aware_maintenance"] = {
        "context_projection": _linear_parameters(
            projection_key,
            config.context_width,
            config.attention_width,
        ),
        "blocks": (
            _residual_block_parameters(block_key_0, config.attention_width, config.ffn_width),
            _residual_block_parameters(block_key_1, config.attention_width, config.ffn_width),
        ),
        "output": output,
    }
    parameters["target_oriented"] = target

    observed = strong_baseline_parameter_count(parameters)
    expected = int(distribution_contract.payload["mechanism"]["candidate_parameter_count"])
    if observed != expected:
        raise ValueError(f"distribution-aware maintenance candidate has {observed} parameters; expected {expected}")
    return parameters


def distribution_aware_maintenance_corrections(
    parameters: Mapping[str, Any],
    context: StrongBaselineContext,
    maintenance_hidden: Any,
    *,
    config: DistributionAwareMaintenanceConfig = DistributionAwareMaintenanceConfig(),
    neural_precision: NeuralPrecisionPolicy = NeuralPrecisionPolicy(),
) -> Any:
    """Return one bounded correction for every compact PFT/part coordinate."""

    hidden = jnp.asarray(maintenance_hidden, dtype=jnp.float32)
    if (
        hidden.ndim != 4
        or hidden.shape[0] != 1
        or hidden.shape[2:]
        != (
            NPARTS,
            config.attention_width,
        )
    ):
        raise ValueError("maintenance hidden states must have shape (1, n_pft, 12, 768)")
    n_pft = hidden.shape[1]
    pft_mask = jnp.asarray(context.pft_mask, dtype=jnp.bool_)
    if pft_mask.shape != (n_pft,):
        raise ValueError("distribution-aware maintenance PFT mask shape drift")

    adapter = parameters["target_oriented"]["distribution_aware_maintenance"]
    source_tokens = _linear(adapter["context_projection"], context.source_tokens)
    pft_tokens = _linear(adapter["context_projection"], context.pft_tokens)
    coordinate_tokens = hidden.reshape((n_pft * NPARTS, config.attention_width))
    coordinate_mask = jnp.repeat(pft_mask, NPARTS)
    bank = jnp.concatenate((source_tokens, pft_tokens, coordinate_tokens), axis=0)
    bank_mask = jnp.concatenate(
        (
            jnp.asarray(context.source_mask, dtype=jnp.bool_),
            pft_mask,
            coordinate_mask,
        )
    )
    attention_mask = jnp.broadcast_to(
        bank_mask[None, :],
        (coordinate_tokens.shape[0], bank.shape[0]),
    )

    query = coordinate_tokens
    for block in adapter["blocks"]:
        query = _layer_norm(
            query
            + _attention(
                block["attention"],
                query,
                bank,
                heads=config.attention_heads,
                mask=attention_mask,
                score_dtype=neural_precision.attention_score_dtype,
            ),
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        query = _layer_norm(
            query + _ffn(block["ffn"], query),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
    raw = _linear(adapter["output"], query)[:, 0]
    correction = config.maximum_absolute_correction * jnp.tanh(raw)
    correction = jnp.where(coordinate_mask, correction, 0.0)
    return correction.reshape((n_pft, NPARTS))


def target_oriented_distribution_aware_maintenance_process_outputs(
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
    distribution_contract: DistributionAwareMaintenanceContract,
    coeff_maint_zero: Any,
    ext_coeff_vegetfrac: Any,
    message_gate_scale: float = 1.0,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    attention_config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
    config: DistributionAwareMaintenanceConfig = DistributionAwareMaintenanceConfig(),
    source_config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig(),
) -> Mapping[str, Any]:
    """Decode parent outputs and materialize part-resolved maintenance."""

    config.validate(distribution_contract, attention_config)
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
        deferred_primary_fields=frozenset((MAINTENANCE_FIELD,)),
    )
    correction = distribution_aware_maintenance_corrections(
        parameters,
        context,
        hidden[MAINTENANCE_FIELD],
        config=config,
        neural_precision=baseline_config.neural_precision,
    )
    outputs[MAINTENANCE_FIELD] = materialize_part_resolved_source_shaped_maintenance(
        inputs,
        outputs,
        coeff_maint_zero,
        ext_coeff_vegetfrac,
        correction,
        layout=layout,
        forcing_layout=forcing_layout,
        config=source_config,
    )
    return apply_fast_parameter_routes(
        outputs,
        inputs,
        layout,
        forcing_layout,
        maintenance_already_routed=True,
    )


def distribution_aware_maintenance_parameter_count(parameters: Mapping[str, Any]) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "SCHEMA_VERSION",
    "DistributionAwareMaintenanceConfig",
    "DistributionAwareMaintenanceContract",
    "distribution_aware_maintenance_corrections",
    "distribution_aware_maintenance_parameter_count",
    "initialize_distribution_aware_maintenance_parameters",
    "load_distribution_aware_maintenance_contract",
    "target_oriented_distribution_aware_maintenance_process_outputs",
]
