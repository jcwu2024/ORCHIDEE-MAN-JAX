"""Q/K-conditioned potential attention for the coupled daily-GPP head."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .process_derivations import NativeForcingFieldLayout
from .strong_baseline_architecture import (
    NeuralPrecisionPolicy,
    StrongBaselineConfig,
    StrongBaselineContext,
    StrongBaselineLayout,
    strong_baseline_parameter_count,
)
from .strong_baseline_contract import contract_sha256
from .target_oriented_attention import TargetOrientedAttentionConfig
from .target_oriented_contract import TargetOrientedContract
from .target_oriented_coupled_gpp_carbon_feedback import (
    CarbonFeedbackStateLayout,
    CoupledGPPCarbonFeedbackConfig,
    coupled_gpp_carbon_feedback_factorization,
    target_oriented_coupled_gpp_carbon_feedback_process_outputs,
)
from .target_oriented_distribution_aware_maintenance import (
    DistributionAwareMaintenanceConfig,
    DistributionAwareMaintenanceContract,
)
from .target_oriented_source_shaped_maintenance import SourceShapedMaintenanceConfig
from .target_oriented_structured_gpp import (
    StructuredGPPConfig,
    StructuredGPPFactorization,
    StructuredGPPStateLayout,
)
from .types import AssembledDailyOperatorInput

SCHEMA_VERSION = "target_oriented_coupled_gpp_conditioned_attention_v1"
FROZEN_CONTRACT_SHA256 = "b9bdd9dde51a8f6f1de2c21cd6678f5a51232011397501459d614b549ca560d4"
PARAMETER_FAMILY = "coupled_gpp_conditioning"
TRAINABLE_FAMILIES = (
    "structured_gpp",
    "coupled_gpp_carbon_feedback",
    PARAMETER_FAMILY,
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_conditioned_attention_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPConditionedAttentionContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class CoupledGPPConditionedAttentionConfig:
    potential_blocks: int = 2
    attention_heads: int = 8
    head_width: int = 48
    normalization_epsilon: float = 1.0e-6
    temperature_initial_value: float = float(np.sqrt(48.0))
    temperature_maximum: float = 12.0

    def validate(
        self,
        contract: CoupledGPPConditionedAttentionContract,
        structured_config: StructuredGPPConfig,
    ) -> None:
        mechanism = contract.payload["mechanism"]
        observed = (
            self.potential_blocks,
            self.attention_heads,
            self.head_width,
            self.normalization_epsilon,
            self.temperature_initial_value,
            self.temperature_maximum,
        )
        declared = (
            int(mechanism["conditioned_attention_blocks"]),
            int(mechanism["attention_heads"]),
            int(mechanism["head_width"]),
            float(mechanism["normalization_epsilon"]),
            float(mechanism["temperature_initial_value"]),
            float(mechanism["temperature_maximum"]),
        )
        if not all(np.isclose(left, right, rtol=0.0, atol=1.0e-12) for left, right in zip(observed, declared)):
            raise ValueError("conditioned GPP attention architecture drift")
        if (
            self.potential_blocks != structured_config.potential_blocks
            or self.attention_heads != structured_config.attention_heads
            or self.head_width * self.attention_heads != structured_config.width
        ):
            raise ValueError("conditioned GPP attention parent geometry drift")
        if not 0.0 < self.normalization_epsilon < 1.0:
            raise ValueError("conditioned GPP attention epsilon is invalid")
        if not 0.0 < self.temperature_initial_value < self.temperature_maximum:
            raise ValueError("conditioned GPP attention temperature interval is invalid")


def load_coupled_gpp_conditioned_attention_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPConditionedAttentionContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    expected_status = (
        "frozen_after_user_conditioning_architecture_approval_for_local_implementation_and_zero_update_admission"
    )
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != expected_status:
        raise ValueError("conditioned GPP attention contract schema drift")
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("conditioned GPP attention contract is not frozen v1")
    ownership = payload["parameter_ownership"]
    if (
        int(ownership["new_parameter_count"]) != 16
        or int(ownership["parent_candidate_parameter_count"]) != 109_464_359
        or int(ownership["candidate_parameter_count"]) != 109_464_375
        or int(ownership["first_stage_trainable_parameter_count"]) != 15_960_211
    ):
        raise ValueError("conditioned GPP attention parameter ownership drift")
    return CoupledGPPConditionedAttentionContract(payload, source, observed)


def _initial_temperature_logit(config: CoupledGPPConditionedAttentionConfig) -> float:
    fraction = config.temperature_initial_value / config.temperature_maximum
    return float(np.log(fraction / (1.0 - fraction)))


def initialize_coupled_gpp_conditioned_attention_parameters(
    parent_parameters: Mapping[str, Any],
    *,
    conditioning_contract: CoupledGPPConditionedAttentionContract,
    structured_config: StructuredGPPConfig = StructuredGPPConfig(),
    config: CoupledGPPConditionedAttentionConfig = CoupledGPPConditionedAttentionConfig(),
) -> Mapping[str, Any]:
    """Attach 16 per-block/head temperature logits to the unchanged parent."""

    config.validate(conditioning_contract, structured_config)
    ownership = conditioning_contract.payload["parameter_ownership"]
    observed_parent = strong_baseline_parameter_count(parent_parameters)
    expected_parent = int(ownership["parent_candidate_parameter_count"])
    if observed_parent != expected_parent:
        raise ValueError(f"conditioned GPP parent has {observed_parent} parameters; expected {expected_parent}")
    target = dict(parent_parameters["target_oriented"])
    if PARAMETER_FAMILY in target:
        raise ValueError("conditioned GPP attention parameters already exist")
    target[PARAMETER_FAMILY] = {
        "potential_temperature_logits": jnp.full(
            (config.potential_blocks, config.attention_heads),
            _initial_temperature_logit(config),
            dtype=jnp.float32,
        )
    }
    parameters = dict(parent_parameters, target_oriented=target)
    observed = strong_baseline_parameter_count(parameters)
    expected = int(ownership["candidate_parameter_count"])
    if observed != expected:
        raise ValueError(f"conditioned GPP candidate has {observed} parameters; expected {expected}")
    return parameters


def coupled_gpp_conditioned_attention_temperatures(
    parameters: Mapping[str, Any],
    *,
    config: CoupledGPPConditionedAttentionConfig = CoupledGPPConditionedAttentionConfig(),
) -> Any:
    """Map unconstrained scalar parameters into the frozen open temperature interval."""

    raw = jnp.asarray(
        parameters["target_oriented"][PARAMETER_FAMILY]["potential_temperature_logits"],
        dtype=jnp.float32,
    )
    expected = (config.potential_blocks, config.attention_heads)
    if raw.shape != expected:
        raise ValueError(f"conditioned GPP temperature shape is {raw.shape}; expected {expected}")
    return jnp.asarray(config.temperature_maximum, dtype=jnp.float32) * jax.nn.sigmoid(raw)


def fixed_reference_attention_temperatures(
    *,
    config: CoupledGPPConditionedAttentionConfig = CoupledGPPConditionedAttentionConfig(),
) -> Any:
    """Return the matched reference's fixed standard-scale temperatures."""

    return jnp.full(
        (config.potential_blocks, config.attention_heads),
        config.temperature_initial_value,
        dtype=jnp.float32,
    )


def conditioned_continuity_gate(
    *,
    parent_lower_gpp: float,
    parent_upper_gpp: float,
    candidate_lower_gpp: float,
    candidate_upper_gpp: float,
    reference_lower_gpp: float,
    reference_upper_gpp: float,
    maximum_absolute_logit: float,
    jump_fraction_maximum: float,
    logit_absolute_maximum: float,
    reference_rtol: float,
    reference_atol: float,
) -> Mapping[str, Any]:
    """Evaluate the frozen zero-training conditioning and matched-reference gate."""

    values = np.asarray(
        (
            parent_lower_gpp,
            parent_upper_gpp,
            candidate_lower_gpp,
            candidate_upper_gpp,
            reference_lower_gpp,
            reference_upper_gpp,
            maximum_absolute_logit,
        ),
        dtype=np.float64,
    )
    parent_jump = float(parent_upper_gpp) - float(parent_lower_gpp)
    if not np.all(np.isfinite(values)) or abs(parent_jump) <= 1.0e-12:
        raise ValueError("conditioned continuity gate requires finite values and a nonzero parent jump")
    candidate_jump = float(candidate_upper_gpp) - float(candidate_lower_gpp)
    jump_fraction = abs(candidate_jump) / abs(parent_jump)
    reference_match = bool(
        np.allclose(
            (candidate_lower_gpp, candidate_upper_gpp),
            (reference_lower_gpp, reference_upper_gpp),
            rtol=reference_rtol,
            atol=reference_atol,
        )
    )
    nonnegative = bool(np.all(values[:6] >= 0.0))
    jump_passed = bool(jump_fraction <= float(jump_fraction_maximum))
    logit_passed = bool(abs(float(maximum_absolute_logit)) <= float(logit_absolute_maximum))
    passed = bool(nonnegative and reference_match and jump_passed and logit_passed)
    return {
        "passed": passed,
        "parent_jump": parent_jump,
        "candidate_jump": candidate_jump,
        "conditioned_jump_fraction_of_parent": jump_fraction,
        "candidate_and_fixed_reference_match": reference_match,
        "all_gpp_nonnegative": nonnegative,
        "maximum_absolute_conditioned_logit": float(maximum_absolute_logit),
        "jump_reduction_gate_passed": jump_passed,
        "logit_bound_gate_passed": logit_passed,
        "thresholds": {
            "conditioned_jump_fraction_of_parent_maximum": float(jump_fraction_maximum),
            "conditioned_logit_absolute_maximum": float(logit_absolute_maximum),
            "reference_rtol": float(reference_rtol),
            "reference_atol": float(reference_atol),
        },
    }


def conditioned_coupled_gpp_trainable_tree(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    target = parameters["target_oriented"]
    return {name: target[name] for name in TRAINABLE_FAMILIES}


def replace_conditioned_coupled_gpp_trainable(
    parameters: Mapping[str, Any],
    trainable: Mapping[str, Any],
) -> Mapping[str, Any]:
    keys = frozenset(trainable)
    if keys not in (frozenset(), frozenset(TRAINABLE_FAMILIES)):
        raise ValueError("conditioned coupled-GPP trainable family drift")
    target = dict(parameters["target_oriented"])
    for name in TRAINABLE_FAMILIES:
        target.pop(name, None)
    target.update(trainable)
    return dict(parameters, target_oriented=target)


def conditioned_coupled_gpp_factorization(
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
    coupled_config: CoupledGPPCarbonFeedbackConfig = CoupledGPPCarbonFeedbackConfig(),
    config: CoupledGPPConditionedAttentionConfig = CoupledGPPConditionedAttentionConfig(),
    neural_precision: NeuralPrecisionPolicy = NeuralPrecisionPolicy(),
) -> StructuredGPPFactorization:
    return coupled_gpp_carbon_feedback_factorization(
        parameters,
        context,
        gpp_hidden,
        inputs,
        layout=layout,
        structured_state_layout=structured_state_layout,
        carbon_state_layout=carbon_state_layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        structured_config=structured_config,
        config=coupled_config,
        neural_precision=neural_precision,
        potential_attention_temperatures=coupled_gpp_conditioned_attention_temperatures(
            parameters,
            config=config,
        ),
        potential_attention_normalization_epsilon=config.normalization_epsilon,
    )


def target_oriented_coupled_gpp_conditioned_attention_process_outputs(
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
    coupled_config: CoupledGPPCarbonFeedbackConfig = CoupledGPPCarbonFeedbackConfig(),
    config: CoupledGPPConditionedAttentionConfig = CoupledGPPConditionedAttentionConfig(),
) -> Mapping[str, Any]:
    """Run the coupled head with bounded cosine attention only in potential blocks."""

    return target_oriented_coupled_gpp_carbon_feedback_process_outputs(
        parameters,
        context,
        inputs,
        schema=schema,
        layout=layout,
        structured_state_layout=structured_state_layout,
        carbon_state_layout=carbon_state_layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        forcing_layout=forcing_layout,
        target_contract=target_contract,
        distribution_contract=distribution_contract,
        coeff_maint_zero=coeff_maint_zero,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        message_gate_scale=message_gate_scale,
        baseline_config=baseline_config,
        attention_config=attention_config,
        distribution_config=distribution_config,
        source_config=source_config,
        structured_config=structured_config,
        config=coupled_config,
        potential_attention_temperatures=coupled_gpp_conditioned_attention_temperatures(
            parameters,
            config=config,
        ),
        potential_attention_normalization_epsilon=config.normalization_epsilon,
    )


def coupled_gpp_conditioned_attention_parameter_count(parameters: Mapping[str, Any]) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "CoupledGPPConditionedAttentionConfig",
    "CoupledGPPConditionedAttentionContract",
    "PARAMETER_FAMILY",
    "TRAINABLE_FAMILIES",
    "conditioned_coupled_gpp_factorization",
    "conditioned_coupled_gpp_trainable_tree",
    "conditioned_continuity_gate",
    "coupled_gpp_conditioned_attention_parameter_count",
    "coupled_gpp_conditioned_attention_temperatures",
    "fixed_reference_attention_temperatures",
    "initialize_coupled_gpp_conditioned_attention_parameters",
    "load_coupled_gpp_conditioned_attention_contract",
    "replace_conditioned_coupled_gpp_trainable",
    "target_oriented_coupled_gpp_conditioned_attention_process_outputs",
]
