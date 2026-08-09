"""Host-only targets and loss for the complete successor fast-owner schema."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    _compiled_select_pft_axes,
)

from .fast_owner_reconstruction import (
    FastOwnerLeaf,
    FastOwnerReconstructionContract,
)
from .neural_architecture import NeuralOutputSchema, OutputFieldSpec
from .successor_fast_owner_decoder import (
    DAY_START_DELTA,
    DEFINED_LOGIT,
    RVEGET_KEY,
    RVEGET_LOGIT_KEY,
    UNDEFINED_THRESHOLD,
    _canonical_input,
    _day_start_base,
    _neural_leaves,
    schema_field_labels,
    training_representation_outputs,
)
from .training_data import MaskedProcessTarget, ProcessTrainingExample
from .training_loss import ProcessLossResult, masked_equal_label_macro_mse


class SuccessorFastOwnerLossResult(NamedTuple):
    total: Any
    label_loss: Mapping[str, Any]
    label_count: Mapping[str, Any]
    value_loss: ProcessLossResult
    defined_bce: Mapping[str, Any]
    defined_count: Mapping[str, Any]


def _get_path(root: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = root
    for name in path:
        value = value[name]
    return value


def _retained_value(example: ProcessTrainingExample, leaf: FastOwnerLeaf) -> Any:
    boundary = example.targets.retained_boundary
    if leaf.component is not None:
        root = boundary.fast_state_fields[leaf.component]
    elif leaf.family == "daily_interface":
        root = boundary.daily_fields
    elif leaf.family == "ok_leak":
        root = boundary.ok_leak_result
    elif leaf.family == "final_diagnostics":
        root = boundary.final_entry_diagnostics
    else:
        raise ValueError(f"unknown fast-owner target family for {leaf.key}")
    return _get_path(root, leaf.path)


def _structural_mask(inputs: Any, field: OutputFieldSpec) -> np.ndarray:
    mask = np.ones(field.shape, dtype=np.bool_)
    if field.pft_axis is None:
        return mask
    active = np.asarray(_canonical_input(inputs).pft.active_pft_mask, dtype=np.bool_)
    if field.shape[field.pft_axis] != active.size:
        raise ValueError(f"fast-owner target PFT width drift for {field.name}")
    view = [1] * len(field.shape)
    view[field.pft_axis] = active.size
    return mask & np.broadcast_to(active.reshape(view), field.shape)


def build_successor_fast_owner_targets(
    example: ProcessTrainingExample,
    *,
    schema: NeuralOutputSchema,
    markov_contract: DailyMarkovContract,
    owner_contract: FastOwnerReconstructionContract,
) -> dict[str, MaskedProcessTarget]:
    """Build 181 named targets without exposing an anonymous fast vector."""

    if markov_contract.sha256 != owner_contract.markov_contract_sha256:
        raise ValueError("fast-owner target contracts have different Markov identities")
    fields_by_name = {field.name: field for field in schema.fields}
    markov_leaves = {leaf.key: leaf for leaf in markov_contract.fast_day_target_leaves}
    result = dict(example.targets.supervision)
    rveget_defined: np.ndarray | None = None
    rveget_eligible: np.ndarray | None = None

    for leaf in _neural_leaves(owner_contract):
        field = fields_by_name.get(str(leaf.output_key))
        spec = markov_leaves.get(leaf.key)
        if field is None or spec is None:
            raise ValueError(f"fast-owner target schema is incomplete for {leaf.key}")
        absolute = np.asarray(
            _compiled_select_pft_axes(_retained_value(example, leaf), spec),
            dtype=np.float64,
        )
        if absolute.shape != field.shape:
            raise ValueError(f"fast-owner target shape drift for {leaf.key}")
        teacher_defined = np.isfinite(absolute) & (np.abs(absolute) < UNDEFINED_THRESHOLD)
        eligible = _structural_mask(example.inputs, field)
        if field.value_representation == DAY_START_DELTA:
            base = np.asarray(_day_start_base(example.inputs, field), dtype=np.float64)
            value = np.where(teacher_defined, absolute - base, 0.0)
        else:
            value = np.where(teacher_defined, absolute, 0.0)
        result[field.name] = MaskedProcessTarget(
            value=value,
            defined=teacher_defined & eligible,
        )
        if leaf.key == RVEGET_KEY:
            rveget_defined = teacher_defined
            rveget_eligible = eligible

    if rveget_defined is None or rveget_eligible is None:
        raise ValueError("fast-owner targets lack rveget dynamic status")
    result[RVEGET_LOGIT_KEY] = MaskedProcessTarget(
        value=rveget_defined.astype(np.float64),
        defined=rveget_eligible,
    )
    if set(result) != set(fields_by_name):
        missing = sorted(set(fields_by_name) - set(result))
        extra = sorted(set(result) - set(fields_by_name))
        raise ValueError(f"fast-owner target registry drift; missing={missing}, extra={extra}")
    return result


def successor_fast_owner_loss(
    physical_outputs: Mapping[str, Any],
    targets: Mapping[str, MaskedProcessTarget],
    *,
    inputs: Any,
    schema: NeuralOutputSchema,
    scales: Mapping[str, Any],
) -> SuccessorFastOwnerLossResult:
    """Equal-label value MSE with rveget BCE folded into its process group."""

    expected = {field.name for field in schema.fields}
    if set(targets) != expected:
        raise ValueError("fast-owner loss target registry drift")
    represented = training_representation_outputs(physical_outputs, inputs, schema)
    logit_fields = tuple(field for field in schema.fields if field.value_representation == DEFINED_LOGIT)
    if len(logit_fields) != 1 or logit_fields[0].name != RVEGET_LOGIT_KEY:
        raise ValueError("fast-owner loss dynamic-logit registry drift")
    value_names = tuple(field.name for field in schema.fields if field.name != RVEGET_LOGIT_KEY)
    if set(scales) != set(value_names):
        raise ValueError("fast-owner loss scale registry drift")
    labels = schema_field_labels(schema)
    value_loss = masked_equal_label_macro_mse(
        {name: represented[name] for name in value_names},
        {name: targets[name] for name in value_names},
        scales=scales,
        field_labels={name: labels[name] for name in value_names},
    )

    logits = jnp.asarray(represented[RVEGET_LOGIT_KEY], dtype=jnp.float64)
    logit_target = targets[RVEGET_LOGIT_KEY]
    wanted = jnp.asarray(logit_target.value, dtype=jnp.float64)
    eligible = jnp.asarray(logit_target.defined, dtype=jnp.bool_)
    if logits.shape != wanted.shape or eligible.shape != wanted.shape:
        raise ValueError("rveget BCE shape drift")
    element = jnp.maximum(logits, 0.0) - logits * wanted + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    bce_count = jnp.sum(eligible, dtype=jnp.float64)
    bce = jnp.where(
        bce_count > 0.0,
        jnp.sum(jnp.where(eligible, element, 0.0), dtype=jnp.float64) / bce_count,
        0.0,
    )
    bce_label = labels[RVEGET_LOGIT_KEY]
    if bce_label != "fast_owner.diffuco" or bce_label not in value_loss.label_mse:
        raise ValueError("rveget BCE loss-group drift")

    label_loss = dict(value_loss.label_mse)
    label_count = dict(value_loss.label_count)
    value_count = label_count[bce_label]
    both = (value_count > 0.0) & (bce_count > 0.0)
    label_loss[bce_label] = jnp.where(
        both,
        0.5 * (label_loss[bce_label] + bce),
        jnp.where(bce_count > 0.0, bce, label_loss[bce_label]),
    )
    label_count[bce_label] = value_count + bce_count
    ordered = tuple(name for name in schema.labels if name in label_loss)
    active = jnp.stack(tuple(label_count[name] > 0.0 for name in ordered))
    values = jnp.stack(tuple(label_loss[name] for name in ordered))
    active_count = jnp.sum(active, dtype=jnp.float64)
    total = jnp.where(
        active_count > 0.0,
        jnp.sum(jnp.where(active, values, 0.0), dtype=jnp.float64) / active_count,
        0.0,
    )
    return SuccessorFastOwnerLossResult(
        total=total,
        label_loss=label_loss,
        label_count=label_count,
        value_loss=value_loss,
        defined_bce={RVEGET_LOGIT_KEY: bce},
        defined_count={RVEGET_LOGIT_KEY: bce_count},
    )
