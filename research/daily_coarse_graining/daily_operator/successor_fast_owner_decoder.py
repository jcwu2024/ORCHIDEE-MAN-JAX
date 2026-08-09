"""Inference-safe decoder schema for the complete fast-owner boundary."""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import DailyMarkovContract

from .fast_owner_reconstruction import (
    ROLE_NEURAL,
    FastOwnerLeaf,
    FastOwnerReconstructionContract,
)
from .neural_architecture import (
    NeuralOutputNormalization,
    NeuralOutputSchema,
    OutputFieldSpec,
)

FAST_OWNER_LABELS = (
    "fast_owner.driver",
    "fast_owner.diffuco",
    "fast_owner.enerbil",
    "fast_owner.hydrol",
    "fast_owner.thermosoil",
    "daily_summary",
    "ok_leak_aux",
    "fast_owner.sechiba_finalize",
)
DAY_START_DELTA = "normalized_day_end_minus_day_start_owner"
ABSOLUTE_TARGET = "normalized_absolute_teacher_native"
DEFINED_LOGIT = "defined_logit"
RVEGET_KEY = "diffuco_previous_step_state.rveget"
RVEGET_LOGIT_KEY = "endpoint.fast_state.diffuco_previous_step_state.rveget.defined_logit"
UNDEFINED_THRESHOLD = 0.5e20


def _constraint(value: str) -> str:
    mapping = {
        "finite_teacher_native": "signed",
        "nonnegative": "nonnegative",
        "unit_interval": "unit_interval",
        "dynamic_defined_nonnegative": "nonnegative",
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported fast-owner scientific constraint: {value}") from error


def _source_route(leaf: FastOwnerLeaf) -> str:
    domain = leaf.proposed_process_domain
    if domain in {"water", "energy"}:
        return str(domain)
    if domain != "carbon":
        raise ValueError(f"fast-owner endpoint lacks a process domain: {leaf.key}")
    if leaf.key == "daily_interface.resp_maint_radia":
        return "carbon.fast_maintenance_respiration"
    if leaf.component == "diffuco_previous_step_state" or leaf.key == ("sechiba_finalize_state.leaf_ci"):
        return "carbon.fast_gpp"
    return "carbon.other"


def _day_start_owner_key(leaf: FastOwnerLeaf) -> str:
    if leaf.key == "final_diagnostics.temp_sol":
        return "enerbil_previous_step_state.temp_sol"
    return leaf.key


def _state_leaves(contract: DailyMarkovContract) -> Mapping[str, Any]:
    result = {leaf.key: leaf for leaf in contract.state_leaves}
    if len(result) != len(contract.state_leaves):
        raise ValueError("Markov state leaf keys are not unique")
    return result


def _neural_leaves(
    owner_contract: FastOwnerReconstructionContract,
) -> tuple[FastOwnerLeaf, ...]:
    result = tuple(leaf for leaf in owner_contract.leaves if leaf.role == ROLE_NEURAL)
    if len(result) != 79:
        raise ValueError("fast-owner decoder requires exactly 79 value endpoints")
    return result


def build_successor_fast_owner_schema(
    base_schema: NeuralOutputSchema,
    *,
    markov_contract: DailyMarkovContract,
    owner_contract: FastOwnerReconstructionContract,
) -> NeuralOutputSchema:
    """Append the accepted 79 value queries and one dynamic-status query."""

    if markov_contract.sha256 != owner_contract.markov_contract_sha256:
        raise ValueError("fast-owner decoder contracts have different Markov identities")
    if tuple(base_schema.active_pft_indices) != tuple(markov_contract.active_pft_indices):
        raise ValueError("fast-owner decoder PFT registry drift")
    if set(base_schema.labels) & set(FAST_OWNER_LABELS):
        raise ValueError("fast-owner loss labels collide with the base schema")
    labels = base_schema.labels + FAST_OWNER_LABELS
    label_index = {name: index for index, name in enumerate(labels)}
    state_leaves = _state_leaves(markov_contract)
    fields = list(base_schema.fields)
    existing_names = {field.name for field in fields}

    dynamic_leaf: FastOwnerLeaf | None = None
    for leaf in _neural_leaves(owner_contract):
        if leaf.output_key is None or leaf.loss_group not in label_index:
            raise ValueError(f"fast-owner endpoint metadata is incomplete: {leaf.key}")
        if leaf.output_key in existing_names:
            raise ValueError(f"fast-owner output collides with the base schema: {leaf.output_key}")
        if leaf.value_representation not in {DAY_START_DELTA, ABSOLUTE_TARGET}:
            raise ValueError(f"fast-owner representation drift for {leaf.key}")
        owner_key = None
        state_start = None
        state_stop = None
        if leaf.value_representation == DAY_START_DELTA:
            owner_key = _day_start_owner_key(leaf)
            state_leaf = state_leaves.get(owner_key)
            if state_leaf is None:
                raise ValueError(f"fast-owner delta lacks state owner: {leaf.key}")
            if tuple(state_leaf.shape) != leaf.shape:
                raise ValueError(f"fast-owner day-start shape drift for {leaf.key}")
            state_start = int(state_leaf.start)
            state_stop = int(state_leaf.stop)
        field = OutputFieldSpec(
            name=leaf.output_key,
            shape=leaf.shape,
            label_index=label_index[str(leaf.loss_group)],
            constraint=_constraint(leaf.scientific_constraint),
            pft_axis=leaf.pft_axis,
            full_pft_axis=False,
            process_domain=leaf.proposed_process_domain,
            source_route=_source_route(leaf),
            value_representation=str(leaf.value_representation),
            day_start_owner=owner_key,
            day_start_state_start=state_start,
            day_start_state_stop=state_stop,
        )
        fields.append(field)
        existing_names.add(field.name)
        if leaf.defined_output_key is not None:
            if dynamic_leaf is not None:
                raise ValueError("fast-owner decoder admits only one dynamic status head")
            dynamic_leaf = leaf

    if dynamic_leaf is None or dynamic_leaf.key != RVEGET_KEY:
        raise ValueError("fast-owner decoder requires the accepted rveget status head")
    if dynamic_leaf.defined_output_key != RVEGET_LOGIT_KEY:
        raise ValueError("fast-owner rveget logit identity drift")
    fields.append(
        OutputFieldSpec(
            name=RVEGET_LOGIT_KEY,
            shape=dynamic_leaf.shape,
            label_index=label_index[str(dynamic_leaf.loss_group)],
            constraint="logit",
            pft_axis=dynamic_leaf.pft_axis,
            full_pft_axis=False,
            process_domain=dynamic_leaf.proposed_process_domain,
            source_route=_source_route(dynamic_leaf),
            value_representation=DEFINED_LOGIT,
        )
    )
    return NeuralOutputSchema(
        fields=tuple(fields),
        labels=labels,
        active_pft_indices=base_schema.active_pft_indices,
    )


def schema_field_labels(schema: NeuralOutputSchema) -> dict[str, str]:
    return {field.name: schema.labels[field.label_index] for field in schema.fields}


def _canonical_input(inputs: Any) -> Any:
    return getattr(inputs, "canonical_input", inputs)


def _day_start_base(inputs: Any, field: OutputFieldSpec) -> Any:
    if field.value_representation != DAY_START_DELTA:
        return jnp.zeros(field.shape, dtype=jnp.float64)
    state = jnp.asarray(_canonical_input(inputs).continuous_state, dtype=jnp.float64)
    start = int(field.day_start_state_start)
    stop = int(field.day_start_state_stop)
    value = state[start:stop].reshape(field.shape)
    defined = jnp.isfinite(value) & (jnp.abs(value) < UNDEFINED_THRESHOLD)
    return jnp.where(defined, value, 0.0)


def training_representation_outputs(
    physical_outputs: Mapping[str, Any],
    inputs: Any,
    schema: NeuralOutputSchema,
) -> dict[str, Any]:
    """Convert physical materializer endpoints back to their loss representation."""

    if set(physical_outputs) != {field.name for field in schema.fields}:
        raise ValueError("fast-owner physical output registry drift")
    result = {}
    for field in schema.fields:
        value = jnp.asarray(physical_outputs[field.name], dtype=jnp.float64)
        if field.value_representation == DAY_START_DELTA:
            value = value - _day_start_base(inputs, field)
        result[field.name] = value
    return result


def validate_successor_output_normalization(
    schema: NeuralOutputSchema,
    normalization: NeuralOutputNormalization,
) -> None:
    expected = {field.name for field in schema.fields}
    if set(normalization.mean) != expected or set(normalization.scale) != expected:
        raise ValueError("successor output normalization registry drift")
    for field in schema.fields:
        mean = np.asarray(normalization.mean[field.name], dtype=np.float64)
        scale = np.asarray(normalization.scale[field.name], dtype=np.float64)
        if mean.shape != field.shape or scale.shape != field.shape:
            raise ValueError(f"successor output normalization shape drift for {field.name}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale) & (scale > 0.0)):
            raise ValueError(f"successor output normalization is invalid for {field.name}")
        if field.value_representation == DEFINED_LOGIT:
            if np.any(mean != 0.0) or np.any(scale != 1.0):
                raise ValueError("defined logits require exact zero/one normalization metadata")


def extend_successor_output_normalization(
    base: NeuralOutputNormalization,
    schema: NeuralOutputSchema,
    *,
    added_mean: Mapping[str, Any],
    added_scale: Mapping[str, Any],
) -> NeuralOutputNormalization:
    """Add fitted value statistics; only the Bernoulli logits use fixed unit scale."""

    base_names = {field.name for field in schema.fields if field.value_representation == "direct"}
    added_names = {
        field.name for field in schema.fields if field.value_representation in {DAY_START_DELTA, ABSOLUTE_TARGET}
    }
    if set(base.mean) != base_names or set(base.scale) != base_names:
        raise ValueError("base successor normalization does not match the 101-field schema")
    if set(added_mean) != added_names or set(added_scale) != added_names:
        raise ValueError("fitted fast-owner normalization registry drift")
    mean = dict(base.mean) | dict(added_mean)
    scale = dict(base.scale) | dict(added_scale)
    logit = next(field for field in schema.fields if field.value_representation == DEFINED_LOGIT)
    mean[logit.name] = np.zeros(logit.shape, dtype=np.float64)
    scale[logit.name] = np.ones(logit.shape, dtype=np.float64)
    result = NeuralOutputNormalization(mean=mean, scale=scale)
    validate_successor_output_normalization(schema, result)
    return result
