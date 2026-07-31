"""Frozen-base adapter for the causal carbon interface entering daily STOMATE."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.axis_process_coupled_daily_model import (
    AxisProcessCoupledModelParameters,
    AxisProcessCoupledModelSpec,
    AxisProcessFeatures,
    axis_process_coupled_features,
    axis_process_coupled_prediction_from_features,
)
from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalPrediction,
    DenseParameters,
    _dense,
    _dense_init,
)

CAUSAL_CARBON_ADAPTER_V1 = "causal_carbon_adapter_v1"
PFT14_INDEX = 13

_GROUP_FIELDS = {
    "carbon_flux_interface": (
        ("daily_interface", "gpp_daily"),
        ("daily_interface", "resp_maint_part"),
    ),
    "carbon_stock_interface": (
        ("ok_leak", "carbon_32l"),
        ("ok_leak", "deepC_peat"),
    ),
}
_GROUP_PROCESS_SOURCES = {
    "carbon_flux_interface": (
        "stomate_carbon_flux",
        "hydrology",
        "thermal_energy",
    ),
    "carbon_stock_interface": (
        "stomate_carbon_storage",
        "stomate_litter_turnover",
        "hydrology",
    ),
}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CausalCarbonTargetGroup:
    id: str
    target_indices: tuple[int, ...]
    source_process_ids: tuple[str, ...]
    fields: tuple[str, ...]


@dataclass(frozen=True)
class CausalCarbonInterfaceLayout:
    target_width: int
    pft_index: int
    groups: tuple[CausalCarbonTargetGroup, ...]
    metadata: Mapping[str, Any]
    sha256: str

    @property
    def target_indices(self) -> tuple[int, ...]:
        return tuple(index for group in self.groups for index in group.target_indices)


def _pft_target_indices(leaf: Mapping[str, Any], *, pft_index: int) -> tuple[int, ...]:
    axes = tuple(str(value) for value in leaf.get("axis_names", ()))
    if "nvm" not in axes:
        raise ValueError(f"causal carbon field has no PFT axis: {leaf['path'][0]}")
    selected = tuple(int(value) for value in leaf.get("selected_pft_indices", ()))
    if pft_index not in selected:
        raise ValueError(f"causal carbon field does not carry PFT{pft_index + 1}: {leaf['path'][0]}")
    shape = tuple(int(value) for value in leaf["shape"])
    start = int(leaf["start"])
    stop = int(leaf["stop"])
    if int(np.prod(shape, dtype=np.int64)) != stop - start:
        raise ValueError(f"causal carbon field shape drift: {leaf['path'][0]}")
    local = np.arange(stop - start, dtype=np.int32).reshape(shape)
    pft_axis = axes.index("nvm")
    compact_index = selected.index(pft_index)
    indices = np.take(local, compact_index, axis=pft_axis).reshape(-1) + start
    return tuple(int(value) for value in indices)


def causal_carbon_interface_layout_from_contract(
    contract_metadata: Mapping[str, Any],
    base_spec: AxisProcessCoupledModelSpec,
    *,
    pft_index: int = PFT14_INDEX,
) -> CausalCarbonInterfaceLayout:
    """Resolve only source-reachable PFT14 fields consumed by daily carbon."""

    target_width = int(contract_metadata["fast_day_target_width"])
    required_keys = {key for field_keys in _GROUP_FIELDS.values() for key in field_keys}
    leaves = {}
    for leaf in contract_metadata["fast_day_target_leaves"]:
        key = (str(leaf["family"]), str(leaf["path"][0]))
        if key not in required_keys:
            continue
        if key in leaves:
            raise ValueError(f"duplicate causal carbon target leaf {key}")
        leaves[key] = leaf
    available_processes = {group.id for group in base_spec.layout.process_groups}
    groups = []
    metadata_groups = []
    occupied: set[int] = set()
    for group_id, field_keys in _GROUP_FIELDS.items():
        field_metadata = []
        target_indices = []
        for key in field_keys:
            if key not in leaves:
                raise ValueError(f"missing causal carbon target {key[0]}.{key[1]}")
            leaf = leaves[key]
            if leaf.get("component") is not None:
                raise ValueError(f"causal carbon target unexpectedly has a component: {key}")
            indices = _pft_target_indices(leaf, pft_index=pft_index)
            if any(index < 0 or index >= target_width for index in indices):
                raise ValueError(f"causal carbon target index is out of range: {key}")
            overlap = occupied.intersection(indices)
            if overlap:
                raise ValueError(f"overlapping causal carbon target indices: {overlap}")
            occupied.update(indices)
            target_indices.extend(indices)
            field_metadata.append(
                {
                    "id": f"{key[0]}.{key[1]}",
                    "owner": str(leaf["owner"]),
                    "full_target_range": [int(leaf["start"]), int(leaf["stop"])],
                    "adapted_target_indices": list(indices),
                    "axis_names": [str(value) for value in leaf.get("axis_names", ())],
                    "selected_pft_indices": [int(value) for value in leaf.get("selected_pft_indices", ())],
                }
            )
        sources = _GROUP_PROCESS_SOURCES[group_id]
        missing_sources = tuple(source for source in sources if source not in available_processes)
        if missing_sources:
            raise ValueError(f"causal carbon group {group_id} has missing process sources {missing_sources}")
        group = CausalCarbonTargetGroup(
            id=group_id,
            target_indices=tuple(target_indices),
            source_process_ids=sources,
            fields=tuple(f"{family}.{field}" for family, field in field_keys),
        )
        groups.append(group)
        metadata_groups.append(
            {
                "id": group.id,
                "target_width": len(group.target_indices),
                "source_process_ids": list(group.source_process_ids),
                "fields": field_metadata,
            }
        )
    metadata = {
        "schema_version": "causal_carbon_interface_layout_v1",
        "fast_day_target_width": target_width,
        "pft_index": pft_index,
        "pft_label": f"PFT{pft_index + 1}",
        "groups": metadata_groups,
        "protected_target_width": target_width - len(occupied),
        "base_process_axis_layout_sha256": base_spec.layout.sha256,
    }
    return CausalCarbonInterfaceLayout(
        target_width=target_width,
        pft_index=pft_index,
        groups=tuple(groups),
        metadata=metadata,
        sha256=_canonical_sha256(metadata),
    )


@dataclass(frozen=True)
class CausalCarbonAdapterModelSpec:
    base_spec: AxisProcessCoupledModelSpec
    interface_layout: CausalCarbonInterfaceLayout
    adapter_hidden_width: int = 64

    def identity(self) -> dict[str, Any]:
        return {
            "id": CAUSAL_CARBON_ADAPTER_V1,
            "base_model_architecture": self.base_spec.identity(),
            "causal_carbon_interface_layout_sha256": self.interface_layout.sha256,
            "adapter_hidden_width": self.adapter_hidden_width,
            "adapter_group_widths": {group.id: len(group.target_indices) for group in self.interface_layout.groups},
            "adapter_group_sources": {
                group.id: list(group.source_process_ids) for group in self.interface_layout.groups
            },
            "base_parameters_frozen": True,
            "adapter_output_initialization": "exact_zero",
            "cross_day_memory": "canonical_state_only",
        }


class CausalCarbonAdapterParameters(NamedTuple):
    base: AxisProcessCoupledModelParameters
    group_inputs: tuple[DenseParameters, ...]
    group_outputs: tuple[DenseParameters, ...]


class CausalCarbonAdapterTrainableParameters(NamedTuple):
    """The optimizer-owned adapter leaves, excluding the frozen parent."""

    group_inputs: tuple[DenseParameters, ...]
    group_outputs: tuple[DenseParameters, ...]


def causal_carbon_adapter_trainable_parameters(
    parameters: CausalCarbonAdapterParameters,
) -> CausalCarbonAdapterTrainableParameters:
    return CausalCarbonAdapterTrainableParameters(
        group_inputs=parameters.group_inputs,
        group_outputs=parameters.group_outputs,
    )


def assemble_causal_carbon_adapter_parameters(
    base_parameters: AxisProcessCoupledModelParameters,
    trainable_parameters: CausalCarbonAdapterTrainableParameters,
) -> CausalCarbonAdapterParameters:
    """Combine an immutable parent with the optimizer-owned adapter leaves."""

    return CausalCarbonAdapterParameters(
        base=base_parameters,
        group_inputs=trainable_parameters.group_inputs,
        group_outputs=trainable_parameters.group_outputs,
    )


def causal_carbon_adapter_spec_from_contract(
    base_spec: AxisProcessCoupledModelSpec,
    contract_metadata: Mapping[str, Any],
) -> CausalCarbonAdapterModelSpec:
    layout = causal_carbon_interface_layout_from_contract(
        contract_metadata,
        base_spec,
    )
    return CausalCarbonAdapterModelSpec(base_spec, layout)


def initialize_causal_carbon_adapter(
    spec: CausalCarbonAdapterModelSpec,
    *,
    base_parameters: AxisProcessCoupledModelParameters,
    seed: int,
) -> CausalCarbonAdapterParameters:
    """Attach an exact-zero adapter without changing the parent prediction."""

    if spec.adapter_hidden_width < 1:
        raise ValueError("causal carbon adapter width must be positive")
    process_width = spec.base_spec.process_latent_width
    shared_width = process_width + spec.base_spec.forcing_latent_width + spec.base_spec.condition_width
    keys = iter(
        jax.random.split(
            jax.random.PRNGKey(seed),
            2 * len(spec.interface_layout.groups),
        )
    )
    inputs = []
    outputs = []
    for group in spec.interface_layout.groups:
        input_width = len(group.source_process_ids) * process_width + shared_width
        inputs.append(
            _dense_init(
                next(keys),
                input_width,
                spec.adapter_hidden_width,
            )
        )
        output = _dense_init(
            next(keys),
            spec.adapter_hidden_width,
            len(group.target_indices),
        )
        outputs.append(
            output._replace(
                weight=jnp.zeros_like(output.weight),
                bias=jnp.zeros_like(output.bias),
            )
        )
    return CausalCarbonAdapterParameters(
        base=base_parameters,
        group_inputs=tuple(inputs),
        group_outputs=tuple(outputs),
    )


def _adapter_group_features(
    features: AxisProcessFeatures,
    group: CausalCarbonTargetGroup,
    spec: CausalCarbonAdapterModelSpec,
):
    process_indices = {process.id: index for index, process in enumerate(spec.base_spec.layout.process_groups)}
    source_indices = jnp.asarray(
        [process_indices[source] for source in group.source_process_ids],
        dtype=jnp.int32,
    )
    sources = jnp.take(features.tokens, source_indices, axis=1).reshape(
        features.tokens.shape[0],
        -1,
    )
    return jnp.concatenate(
        (
            sources,
            features.global_token,
            features.forcing,
            features.condition,
        ),
        axis=-1,
    )


def causal_carbon_adapter_model_apply(
    parameters: CausalCarbonAdapterParameters,
    batch: CanonicalDayBatch,
    spec: CausalCarbonAdapterModelSpec,
) -> CanonicalPrediction:
    """Apply the frozen base and modify only the declared causal columns."""

    frozen_base = jax.tree_util.tree_map(jax.lax.stop_gradient, parameters.base)
    features = axis_process_coupled_features(
        frozen_base,
        batch,
        spec.base_spec,
    )
    base = axis_process_coupled_prediction_from_features(
        frozen_base,
        batch,
        spec.base_spec,
        features,
    )
    correction = jnp.zeros_like(base.normalized_fast_day_target)
    for group, input_parameters, output_parameters in zip(
        spec.interface_layout.groups,
        parameters.group_inputs,
        parameters.group_outputs,
        strict=True,
    ):
        group_features = _adapter_group_features(features, group, spec)
        group_correction = _dense(
            jax.nn.silu(_dense(group_features, input_parameters)),
            output_parameters,
        )
        correction = correction.at[
            :,
            jnp.asarray(group.target_indices, dtype=jnp.int32),
        ].set(group_correction)
    return CanonicalPrediction(
        normalized_fast_day_target=(base.normalized_fast_day_target + correction),
        dynamic_undefined_flip_logits=base.dynamic_undefined_flip_logits,
    )
