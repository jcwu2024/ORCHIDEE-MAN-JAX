"""Zero-training attainability oracle for mechanistic-output coordinates."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import (
    reconstruct_state_fields_compiled,
)

from .conservative_update import apply_conservative_update
from .mechanistic_output_reconstruction import invert_doc_baseline_residual
from .neural_architecture import (
    NeuralOutputNormalization,
    NeuralOutputSchema,
    OutputFieldSpec,
)
from .output_transform import field_coordinates, normalized_output_to_physical
from .successor_fast_owner_decoder import (
    DAY_START_DELTA,
    RVEGET_LOGIT_KEY,
    physical_outputs_from_training_representation,
)
from .training_data import MaskedProcessTarget
from .types import ConservativeUpdateResult, ProcessPrediction

CLASS_LOGIT_MAGNITUDE = 20.0
POSITIVE_INVERSE_FLOOR = 1.0e-30


class AttainabilityOracleResult(NamedTuple):
    prediction: ProcessPrediction
    fast_update: ConservativeUpdateResult
    learned_outputs: Mapping[str, Any]
    doc_coordinate_fixed_point_residual: Any


def _softplus_inverse(value: Any) -> Any:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(value > 20.0, value, jnp.log(jnp.expm1(value)))


def _open_unit_interval(value: Any) -> Any:
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    lower = jnp.asarray(POSITIVE_INVERSE_FLOOR, dtype=jnp.float64)
    upper = jnp.nextafter(one, zero)
    return jnp.minimum(jnp.maximum(value, lower), upper)


def _open_signed_unit_interval(value: Any) -> Any:
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    lower = jnp.nextafter(-one, zero)
    upper = jnp.nextafter(one, zero)
    return jnp.minimum(jnp.maximum(value, lower), upper)


def _field_active_mask(
    field: OutputFieldSpec,
    schema: NeuralOutputSchema,
    inputs: Any,
) -> Any:
    coordinates = jnp.asarray(field_coordinates(field.shape), dtype=jnp.int32)
    active = jnp.ones((coordinates.shape[0],), dtype=jnp.bool_)
    if field.pft_axis is None:
        return active
    coordinate = coordinates[:, int(field.pft_axis)]
    pft_mask = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
    if field.full_pft_axis:
        admitted = jnp.asarray(schema.active_pft_indices, dtype=jnp.int32)
        matches = coordinate[:, None] == admitted[None, :]
        compact = jnp.argmax(matches, axis=1)
        return active & jnp.any(matches, axis=1) & pft_mask[compact]
    safe = jnp.minimum(coordinate, pft_mask.size - 1)
    return active & (coordinate < pft_mask.size) & pft_mask[safe]


def _day_start_base(field: OutputFieldSpec, inputs: Any) -> Any:
    if field.value_representation != DAY_START_DELTA:
        return jnp.zeros(field.shape, dtype=jnp.float64)
    state = jnp.asarray(inputs.canonical_input.continuous_state, dtype=jnp.float64)
    start = int(field.day_start_state_start)
    stop = int(field.day_start_state_stop)
    value = state[start:stop].reshape(field.shape)
    defined = jnp.isfinite(value) & (jnp.abs(value) < 0.5e20)
    return jnp.where(defined, value, 0.0)


def normalized_coordinate_for_physical(
    target: Any,
    *,
    field: OutputFieldSpec,
    normalization: NeuralOutputNormalization,
    inputs: Any,
) -> Any:
    """Invert the accepted output transform without invoking a neural body."""

    target = jnp.asarray(target, dtype=jnp.float64).reshape(-1)
    if field.constraint == "logit":
        return target
    mean = jnp.asarray(normalization.mean[field.name], dtype=jnp.float64).reshape(-1)
    scale = jnp.asarray(normalization.scale[field.name], dtype=jnp.float64).reshape(-1)
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    positive_floor = jnp.asarray(POSITIVE_INVERSE_FLOOR, dtype=jnp.float64)
    if field.value_representation == "direct":
        if field.constraint == "unit_interval":
            bounded = _open_unit_interval(target)
            return jnp.log(bounded) - jnp.log1p(-bounded)
        if field.constraint == "signed_unit":
            return jnp.arctanh(_open_signed_unit_interval(target))
        if field.constraint == "nonnegative":
            ratio = jnp.maximum(mean / scale, 1.0e-12)
            baseline = ratio + jnp.log(-jnp.expm1(-ratio))
            return _softplus_inverse(jnp.maximum(target / scale, positive_floor)) - baseline
        return (target - mean) / scale

    base = _day_start_base(field, inputs).reshape(-1)
    center = base + mean
    if field.constraint == "unit_interval":
        bounded_center = jnp.clip(center, 1.0e-6, 1.0 - 1.0e-6)
        slope = jnp.maximum(bounded_center * (1.0 - bounded_center), 1.0e-6)
        baseline = jnp.log(bounded_center) - jnp.log1p(-bounded_center)
        bounded_target = _open_unit_interval(target)
        target_logit = jnp.log(bounded_target) - jnp.log1p(-bounded_target)
        return (target_logit - baseline) * slope / scale
    if field.constraint == "nonnegative":
        positive_center = jnp.maximum(center, scale * 1.0e-12)
        baseline = _softplus_inverse(positive_center / scale)
        return _softplus_inverse(jnp.maximum(target / scale, positive_floor)) - baseline
    if field.constraint == "signed_unit":
        bounded_center = jnp.clip(center, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
        slope = jnp.maximum(1.0 - bounded_center * bounded_center, 1.0e-6)
        bounded_target = _open_signed_unit_interval(target)
        return (jnp.arctanh(bounded_target) - jnp.arctanh(bounded_center)) * slope / scale
    return (target - center) / scale


def _roundtrip_target(
    target: Any,
    *,
    field: OutputFieldSpec,
    schema: NeuralOutputSchema,
    normalization: NeuralOutputNormalization,
    inputs: Any,
) -> Any:
    raw = normalized_coordinate_for_physical(
        target,
        field=field,
        normalization=normalization,
        inputs=inputs,
    )
    return normalized_output_to_physical(
        raw,
        _field_active_mask(field, schema, inputs),
        field=field,
        normalization=normalization,
        inputs=inputs,
    )


def target_canopy_endpoint(targets: Mapping[str, MaskedProcessTarget], inputs: Any) -> Any:
    storage_out = jnp.asarray(
        targets["inventory.water.qsintveg.storage.outgoing_fraction"].value,
        dtype=jnp.float64,
    )
    storage_in = jnp.asarray(
        targets["inventory.water.qsintveg.storage.incoming_amount"].value,
        dtype=jnp.float64,
    )
    debt_out = jnp.asarray(
        targets["inventory.water.qsintveg.debt.outgoing_fraction"].value,
        dtype=jnp.float64,
    )
    debt_in = jnp.asarray(
        targets["inventory.water.qsintveg.debt.incoming_amount"].value,
        dtype=jnp.float64,
    )
    size = int(np.prod(storage_out.shape, dtype=np.int64))
    start = jnp.asarray(inputs.state.water, dtype=jnp.float64)
    start_storage = start[:size].reshape(storage_out.shape)
    start_debt = start[size : 2 * size].reshape(storage_out.shape)
    end_storage = start_storage * (1.0 - storage_out) + storage_in
    end_debt = start_debt * (1.0 - debt_out) + debt_in
    return end_storage - end_debt


def oracle_learned_outputs(
    head: Any,
    targets: Mapping[str, MaskedProcessTarget],
    inputs: Any,
) -> tuple[dict[str, Any], Any]:
    """Construct one per-day decoded coordinate for every learned field."""

    complete_schema = head.complete_schema
    learned_schema = head.learned_schema
    expected = {field.name for field in complete_schema.fields}
    if set(targets) != expected:
        raise ValueError("attainability oracle requires the complete target boundary")
    represented = {name: target.value for name, target in targets.items()}
    desired = physical_outputs_from_training_representation(
        represented,
        inputs,
        complete_schema,
    )
    outputs = {
        field.name: _roundtrip_target(
            desired[field.name],
            field=field,
            schema=learned_schema,
            normalization=head.output_normalization,
            inputs=inputs,
        )
        for field in learned_schema.fields
    }

    fields = reconstruct_state_fields_compiled(
        inputs.canonical_input.continuous_state,
        inputs.canonical_input.discrete_state,
        head.input_assembler.contract,
    )
    active_indices = jnp.asarray(head.input_assembler.active_pft_indices, dtype=jnp.int32)
    nroot_name = head.mechanism_contract.nroot_coordinate_field
    start_nroot = jnp.take(
        jnp.asarray(fields["hydrol_previous_step_state"]["nroot"], dtype=jnp.float64),
        active_indices,
        axis=1,
    )
    allowed = (
        jnp.isfinite(start_nroot)
        & (jnp.abs(start_nroot) < 0.5e20)
        & (start_nroot > 0.0)
        & jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)[None, :, None]
    )
    normalized_start = jnp.where(allowed, start_nroot, 0.0)
    start_sum = jnp.sum(normalized_start, axis=-1, keepdims=True)
    normalized_start = jnp.where(
        start_sum > 0.0,
        normalized_start / jnp.where(start_sum > 0.0, start_sum, 1.0),
        0.0,
    )
    outputs[nroot_name] = jnp.asarray(desired[nroot_name], dtype=jnp.float64) - normalized_start

    target_endpoint = target_canopy_endpoint(targets, inputs)
    start_qsintveg = jnp.take(
        jnp.asarray(fields["hydrol_previous_step_state"]["qsintveg"], dtype=jnp.float64),
        active_indices,
        axis=1,
    )
    capacity = head.reconstructor.canopy_capacity(fields, inputs)
    endpoint_scale = jnp.asarray(
        head.output_normalization.scale[head.mechanism_contract.canopy_debt_scale_field],
        dtype=jnp.float64,
    )
    endpoint_scale = jnp.maximum(
        capacity,
        jnp.where(
            jnp.isfinite(endpoint_scale) & (endpoint_scale > 0.0),
            endpoint_scale,
            1.0e-6,
        ),
    )
    outputs[head.mechanism_contract.canopy_endpoint_field] = (
        target_endpoint - start_qsintveg
    ) / endpoint_scale
    status_target = jnp.where(
        target_endpoint > 1.0e-10,
        0,
        jnp.where(target_endpoint < -1.0e-10, 2, 1),
    )
    for index, status in enumerate(("storage", "zero", "debt")):
        outputs[head.mechanism_contract.canopy_status_fields[status]] = jnp.where(
            status_target == index,
            CLASS_LOGIT_MAGNITUDE,
            -CLASS_LOGIT_MAGNITUDE,
        )

    rveget_target = jnp.asarray(targets[RVEGET_LOGIT_KEY].value, dtype=jnp.float64)
    rveget_field = next(field for field in learned_schema.fields if field.name == RVEGET_LOGIT_KEY)
    outputs[RVEGET_LOGIT_KEY] = normalized_output_to_physical(
        jnp.where(
            rveget_target >= 0.5,
            CLASS_LOGIT_MAGNITUDE,
            -CLASS_LOGIT_MAGNITUDE,
        ).reshape(-1),
        _field_active_mask(rveget_field, learned_schema, inputs),
        field=rveget_field,
        normalization=head.output_normalization,
        inputs=inputs,
    )

    doc_name = head.mechanism_contract.doc_field
    target_doc = jnp.asarray(desired[doc_name], dtype=jnp.float64)
    outputs[doc_name] = invert_doc_baseline_residual(
        target_doc,
        head.reconstructor.doc_start_baseline(inputs),
        head.output_normalization.scale[doc_name],
    )
    return outputs, target_doc


def materialize_attainability_oracle(
    head: Any,
    targets: Mapping[str, MaskedProcessTarget],
    inputs: Any,
    *,
    doc_fixed_point_iterations: int = 8,
) -> AttainabilityOracleResult:
    """Materialize one complete boundary with no learned network parameters."""

    if doc_fixed_point_iterations < 1:
        raise ValueError("DOC fixed-point iteration count must be positive")
    learned, target_doc = oracle_learned_outputs(head, targets, inputs)
    doc_name = head.mechanism_contract.doc_field
    doc_scale = head.output_normalization.scale[doc_name]
    doc_mask = head.reconstructor.doc_active_mask(inputs, target_doc.ndim)

    def update_doc(_index: Any, coordinate: Any) -> Any:
        current = dict(learned)
        current[doc_name] = coordinate
        primitive = head.prediction_from_learned_outputs(current, inputs)
        update = apply_conservative_update(inputs, primitive)
        baseline = head.reconstructor.doc_trapezoid_baseline(
            primitive,
            update,
            inputs,
        )
        solved = invert_doc_baseline_residual(target_doc, baseline, doc_scale)
        return jnp.where(doc_mask, solved, 0.0)

    coordinate = jax.lax.fori_loop(
        0,
        doc_fixed_point_iterations,
        update_doc,
        learned[doc_name],
    )
    learned = dict(learned)
    learned[doc_name] = coordinate
    primitive = head.prediction_from_learned_outputs(learned, inputs)
    fast_update = apply_conservative_update(inputs, primitive)
    next_coordinate = update_doc(doc_fixed_point_iterations, coordinate)
    complete = head.materialize_post_update(
        {},
        None,
        inputs,
        primitive,
        fast_update,
    )
    residual = jnp.max(jnp.where(doc_mask, jnp.abs(next_coordinate - coordinate), 0.0))
    return AttainabilityOracleResult(
        prediction=complete,
        fast_update=fast_update,
        learned_outputs=learned,
        doc_coordinate_fixed_point_residual=residual,
    )


__all__ = [
    "AttainabilityOracleResult",
    "materialize_attainability_oracle",
    "normalized_coordinate_for_physical",
    "oracle_learned_outputs",
    "target_canopy_endpoint",
]
