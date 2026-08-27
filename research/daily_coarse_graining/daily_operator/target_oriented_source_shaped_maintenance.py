"""Source-shaped maintenance head over the accepted target Attention body."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.stomate.carbon_kernels import (
    IAGRHRTPN,
    IAGRHRTST,
    IAGRSAPPN,
    IAGRSAPST,
    ICARBON,
    ICARBRES,
    IFRUIT,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    NPARTS,
    root_temperature,
)

from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
)
from .parameter_routing import (
    PFT_TRAIT_BARE_SOIL,
    PFT_TRAIT_IS_TREE,
    apply_fast_parameter_routes,
    fast_process_parameter_factors,
)
from .process_derivations import (
    NativeForcingFieldLayout,
    derive_native_daily_deterministic_fields,
)
from .strong_baseline_architecture import (
    StrongBaselineConfig,
    StrongBaselineContext,
    StrongBaselineLayout,
    _linear,
    _linear_parameters,
    strong_baseline_parameter_count,
)
from .target_oriented_attention import (
    TargetOrientedAttentionConfig,
    _target_oriented_attention_decoded_outputs,
    target_oriented_field_categories,
)
from .target_oriented_contract import PRIMARY_SCIENCE_DIRECT, TargetOrientedContract
from .types import AssembledDailyOperatorInput

SCHEMA_VERSION = "target_oriented_source_shaped_maintenance_v1"
MAINTENANCE_FIELD = "daily.fast_maintenance_respiration"
ABOVEGROUND_PARTS = (
    ILEAF,
    ISAPABOVE,
    IHEARTABOVE,
    IFRUIT,
    IAGRSAPST,
    IAGRSAPPN,
    IAGRHRTST,
    IAGRHRTPN,
)
BELOWGROUND_PARTS = (ISAPBELOW, IHEARTBELOW, IROOT)
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_source_shaped_maintenance_v1.json"
)


def contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SourceShapedMaintenanceContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class SourceShapedMaintenanceConfig:
    branch_width: int = 256
    maximum_absolute_correction: float = 1.0
    biomass_threshold: float = 1.0e-8
    lai_threshold: float = 1.0e-8

    def validate(
        self,
        contract: SourceShapedMaintenanceContract,
        attention_config: TargetOrientedAttentionConfig,
    ) -> None:
        mechanism = contract.payload["mechanism"]
        materialization = contract.payload["source_shaped_materialization"]
        if self.branch_width != int(mechanism["branch_mlp_width"]):
            raise ValueError("source-shaped maintenance branch width drift")
        if self.maximum_absolute_correction != float(mechanism["maximum_absolute_correction"]):
            raise ValueError("source-shaped maintenance correction bound drift")
        if (
            self.biomass_threshold,
            self.lai_threshold,
        ) != (
            float(materialization["biomass_threshold"]),
            float(materialization["lai_threshold"]),
        ):
            raise ValueError("source-shaped maintenance threshold drift")
        if attention_config.primary_decoder_width != 768:
            raise ValueError("source-shaped maintenance parent hidden width drift")


def load_source_shaped_maintenance_contract(
    path: str | Path = _CONTRACT_PATH,
) -> SourceShapedMaintenanceContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("source-shaped maintenance schema version drift")
    if payload.get("contract_sha256") != observed:
        raise ValueError("source-shaped maintenance contract SHA256 mismatch")
    if payload["mechanism"]["candidate_parameter_count"] != 84_148_646:
        raise ValueError("source-shaped maintenance parameter count drift")
    return SourceShapedMaintenanceContract(
        payload=payload,
        source_path=source,
        sha256=observed,
    )


def resolve_maintenance_field_index(
    schema: NeuralOutputSchema,
    target_contract: TargetOrientedContract,
) -> int:
    matches = tuple(index for index, field in enumerate(schema.fields) if field.name == MAINTENANCE_FIELD)
    if len(matches) != 1:
        raise ValueError("learned schema requires one maintenance field")
    index = matches[0]
    categories = target_oriented_field_categories(schema, target_contract)
    if categories[index] != PRIMARY_SCIENCE_DIRECT:
        raise ValueError("maintenance field is not owned by the primary decoder")
    if tuple(schema.fields[index].shape) != (1, 2, NPARTS):
        raise ValueError("source-shaped maintenance compact output shape drift")
    return index


def initialize_source_shaped_maintenance_parameters(
    parent_parameters: Mapping[str, Any],
    key: Any,
    *,
    schema: NeuralOutputSchema,
    target_contract: TargetOrientedContract,
    source_contract: SourceShapedMaintenanceContract,
    attention_config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
    config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig(),
) -> Mapping[str, Any]:
    """Replace only the parent's direct maintenance projection."""

    config.validate(source_contract, attention_config)
    expected_parent = int(source_contract.payload["identity"]["parent_parameter_count"])
    observed_parent = strong_baseline_parameter_count(parent_parameters)
    if observed_parent != expected_parent:
        raise ValueError(
            f"source-shaped maintenance parent has {observed_parent} parameters; expected {expected_parent}"
        )
    index = resolve_maintenance_field_index(schema, target_contract)
    parameters = dict(parent_parameters)
    target = dict(parent_parameters["target_oriented"])
    primary_heads = list(target["primary_heads"])
    removed = primary_heads[index]
    if strong_baseline_parameter_count(removed) != 769:
        raise ValueError("source-shaped maintenance removed head size drift")
    primary_heads[index] = {}
    target["primary_heads"] = tuple(primary_heads)

    branches = {}
    keys = iter(jax.random.split(key, 4))
    for name in ("aboveground", "belowground"):
        first = _linear_parameters(
            next(keys),
            attention_config.primary_decoder_width,
            config.branch_width,
        )
        second = _linear_parameters(next(keys), config.branch_width, 1)
        second["kernel"] = jnp.zeros_like(second["kernel"])
        second["bias"] = jnp.zeros_like(second["bias"])
        branches[name] = {"first": first, "second": second}
    target["source_shaped_maintenance_heads"] = branches
    parameters["target_oriented"] = target

    observed = strong_baseline_parameter_count(parameters)
    expected = int(source_contract.payload["mechanism"]["candidate_parameter_count"])
    if observed != expected:
        raise ValueError(f"source-shaped maintenance candidate has {observed} parameters; expected {expected}")
    return parameters


def compact_coeff_maint_zero_from_retained_tail(retained_tail: Any) -> Any:
    """Select exact maintenance coefficients from their retained static owner."""

    owner = retained_tail.static_owner
    full = np.asarray(
        owner.context.stomate_static.parameters.coeff_maint_zero,
        dtype=np.float64,
    )
    if full.ndim != 2 or full.shape[1] != NPARTS:
        raise ValueError("retained coeff_maint_zero must have shape (nvm, 12)")
    indices = tuple(int(index) for index in owner.contract.active_pft_indices)
    if not indices or min(indices) < 0 or max(indices) >= full.shape[0]:
        raise ValueError("retained coeff_maint_zero active PFT indices are invalid")
    return jnp.asarray(full[np.asarray(indices, dtype=np.int32)], dtype=jnp.float64)


def maintenance_temperature_corrections(
    parameters: Mapping[str, Any],
    maintenance_hidden: Any,
    is_tree: Any,
    *,
    config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig(),
) -> tuple[Any, Any]:
    """Map part-group hidden states to bounded per-PFT temperature corrections."""

    hidden = jnp.asarray(maintenance_hidden, dtype=jnp.float32)
    if hidden.ndim != 4 or hidden.shape[0] != 1 or hidden.shape[2:] != (NPARTS, 768):
        raise ValueError("maintenance hidden states must have shape (1, n_pft, 12, 768)")
    tree = jnp.asarray(is_tree, dtype=jnp.bool_)
    if tree.shape != (hidden.shape[1],):
        raise ValueError("is_tree must contain one value per compact PFT")
    heads = parameters["target_oriented"]["source_shaped_maintenance_heads"]

    def branch(name: str, parts: tuple[int, ...], reserve_mask: Any) -> Any:
        part_mask = jnp.zeros((hidden.shape[1], NPARTS), dtype=jnp.bool_)
        part_mask = part_mask.at[:, jnp.asarray(parts, dtype=jnp.int32)].set(True)
        part_mask = part_mask.at[:, ICARBRES].set(reserve_mask)
        count = jnp.sum(part_mask, axis=1, dtype=jnp.float32)
        context = (
            jnp.sum(
                jnp.where(part_mask[None, :, :, None], hidden, 0.0),
                axis=2,
                dtype=jnp.float32,
            )
            / count[None, :, None]
        )
        head = heads[name]
        raw = _linear(head["second"], jax.nn.gelu(_linear(head["first"], context)))[..., 0]
        return jnp.asarray(config.maximum_absolute_correction, dtype=raw.dtype) * jnp.tanh(raw)

    return branch("aboveground", ABOVEGROUND_PARTS, tree), branch("belowground", BELOWGROUND_PARTS, ~tree)


def materialize_source_shaped_maintenance(
    inputs: AssembledDailyOperatorInput,
    decoded_outputs: Mapping[str, Any],
    coeff_maint_zero: Any,
    ext_coeff_vegetfrac: Any,
    aboveground_correction: Any,
    belowground_correction: Any,
    *,
    layout: StrongBaselineLayout,
    forcing_layout: NativeForcingFieldLayout,
    config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig(),
) -> Any:
    """Evaluate the daily source formula with bounded learned corrections."""

    pft = inputs.pft
    state = jnp.asarray(pft.pft_state, dtype=jnp.float64)
    n_pft = state.shape[0]
    biomass_width = state.shape[1] - 2 if state.ndim == 2 else -1
    if biomass_width <= 0 or biomass_width % NPARTS:
        raise ValueError("pft_state must contain veget_max, lai, and part-resolved biomass")
    element_count = biomass_width // NPARTS
    if element_count <= ICARBON:
        raise ValueError("pft_state biomass is missing its carbon element")
    biomass_carbon = state[:, 2:].reshape((n_pft, NPARTS, element_count))[:, :, ICARBON]
    lai = state[:, 1]
    coefficient = jnp.asarray(coeff_maint_zero, dtype=jnp.float64)
    if coefficient.shape != (n_pft, NPARTS):
        raise ValueError("compact coeff_maint_zero shape drift")
    if tuple(layout.active_pft_indices) != (0, 13):
        raise ValueError("source-shaped maintenance is scientifically admitted only for bare soil plus PFT14")
    traits = jnp.asarray(pft.pft_traits, dtype=jnp.float64)
    bare = traits[:, PFT_TRAIT_BARE_SOIL] > 0.5
    coefficient = jnp.where(
        bare[:, None],
        coefficient[0][None, :],
        coefficient[1][None, :],
    )

    full_ext = jnp.asarray(ext_coeff_vegetfrac, dtype=jnp.float64)
    active_indices = jnp.asarray(layout.active_pft_indices, dtype=jnp.int32)
    ext_coeff = jnp.take(full_ext, active_indices, axis=0)
    if ext_coeff.shape != (n_pft,):
        raise ValueError("compact ext_coeff_vegetfrac shape drift")

    deterministic = derive_native_daily_deterministic_fields(
        inputs.forcing,
        forcing_layout,
        model_interval_seconds=layout.model_interval_seconds,
    )
    air_k = jnp.asarray(deterministic.t2m_daily, dtype=jnp.float64).reshape(-1)[0]
    soil_k = jnp.asarray(decoded_outputs["daily.temperature_context.tsoil"], dtype=jnp.float64)
    root_k = root_temperature(
        soil_k[:, None, :],
        jnp.asarray(inputs.process_static.z_soil, dtype=jnp.float64),
        jnp.asarray(inputs.process_static.rprof, dtype=jnp.float64),
    )[0]

    continuous = jnp.asarray(inputs.canonical_input.continuous_state, dtype=jnp.float64)
    tlong_k = continuous[layout.t2m_longterm_start : layout.t2m_longterm_stop][0]
    factors = fast_process_parameter_factors(
        pft.pft_parameters,
        jnp.broadcast_to(tlong_k - 273.15, (n_pft,)),
    )

    is_tree = traits[:, PFT_TRAIT_IS_TREE] > 0.5
    above_delta = jnp.asarray(aboveground_correction, dtype=jnp.float64).reshape((n_pft,))
    below_delta = jnp.asarray(belowground_correction, dtype=jnp.float64).reshape((n_pft,))
    part_temperature = jnp.broadcast_to(air_k, (n_pft, NPARTS))
    part_delta = jnp.broadcast_to(above_delta[:, None], (n_pft, NPARTS))
    below = jnp.asarray(BELOWGROUND_PARTS, dtype=jnp.int32)
    part_temperature = part_temperature.at[:, below].set(root_k[:, None])
    part_delta = part_delta.at[:, below].set(below_delta[:, None])
    reserve_temperature = jnp.where(is_tree, air_k, root_k)
    reserve_delta = jnp.where(is_tree, above_delta, below_delta)
    part_temperature = part_temperature.at[:, ICARBRES].set(reserve_temperature)
    part_delta = part_delta.at[:, ICARBRES].set(reserve_delta)

    temperature_baseline = jnp.maximum(
        1.0 + factors.maintenance_temperature_slope[:, None] * (part_temperature - 273.15),
        0.0,
    )
    temperature_effective = jnp.maximum(
        temperature_baseline + part_delta,
        0.0,
    )
    base = biomass_carbon * coefficient
    safe_lai = jnp.where(lai > config.lai_threshold, lai, 1.0)
    leaf_factor = (0.3 * lai + 1.4 * (1.0 - jnp.exp(-ext_coeff * lai))) / safe_lai
    leaf_base = base[:, ILEAF] * leaf_factor
    leaf_active = (biomass_carbon[:, ILEAF] > config.biomass_threshold) & (lai > config.lai_threshold)
    base = base.at[:, ILEAF].set(jnp.where(leaf_active, leaf_base, 0.0))

    active = jnp.asarray(pft.active_pft_mask, dtype=jnp.bool_)
    fraction = jnp.asarray(pft.pft_fraction, dtype=jnp.float64)
    vegetation = active & (fraction > 0.0) & ~bare
    maintenance = jnp.where(
        vegetation[:, None],
        base * temperature_effective,
        0.0,
    )
    return maintenance[None, ...]


def target_oriented_source_shaped_maintenance_process_outputs(
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
    source_contract: SourceShapedMaintenanceContract,
    coeff_maint_zero: Any,
    ext_coeff_vegetfrac: Any,
    message_gate_scale: float = 1.0,
    baseline_config: StrongBaselineConfig = StrongBaselineConfig(),
    attention_config: TargetOrientedAttentionConfig = TargetOrientedAttentionConfig(),
    config: SourceShapedMaintenanceConfig = SourceShapedMaintenanceConfig(),
) -> Mapping[str, Any]:
    """Decode the parent outputs and materialize maintenance from exact sources."""

    config.validate(source_contract, attention_config)
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
    above, below = maintenance_temperature_corrections(
        parameters,
        hidden[MAINTENANCE_FIELD],
        jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)[:, PFT_TRAIT_IS_TREE] > 0.5,
        config=config,
    )
    outputs[MAINTENANCE_FIELD] = materialize_source_shaped_maintenance(
        inputs,
        outputs,
        coeff_maint_zero,
        ext_coeff_vegetfrac,
        above,
        below,
        layout=layout,
        forcing_layout=forcing_layout,
        config=config,
    )
    return apply_fast_parameter_routes(
        outputs,
        inputs,
        layout,
        forcing_layout,
        maintenance_already_routed=True,
    )


def source_shaped_maintenance_parameter_count(parameters: Mapping[str, Any]) -> int:
    return strong_baseline_parameter_count(parameters)


__all__ = [
    "ABOVEGROUND_PARTS",
    "BELOWGROUND_PARTS",
    "MAINTENANCE_FIELD",
    "SCHEMA_VERSION",
    "SourceShapedMaintenanceConfig",
    "SourceShapedMaintenanceContract",
    "compact_coeff_maint_zero_from_retained_tail",
    "contract_sha256",
    "initialize_source_shaped_maintenance_parameters",
    "load_source_shaped_maintenance_contract",
    "maintenance_temperature_corrections",
    "materialize_source_shaped_maintenance",
    "resolve_maintenance_field_index",
    "source_shaped_maintenance_parameter_count",
    "target_oriented_source_shaped_maintenance_process_outputs",
]
