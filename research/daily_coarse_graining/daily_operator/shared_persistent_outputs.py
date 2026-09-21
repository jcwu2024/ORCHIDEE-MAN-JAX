"""Shared-only residual coordinates for persistent OK_LEAK endpoint readouts.

Source ownership: orchestration.py half_hour_ok_leak carry at 11079-11085,
slowproc packet/write at 9021-9030 and 9210-9238; source and 512-row identity
evidence: shared_remaining_state_source_audit_20260921.md. This changes only
decoder coordinates, never the canonical absolute endpoint or its loss.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .neural_architecture import NeuralOutputNormalization, OutputFieldSpec
from .successor_fast_owner_decoder import ABSOLUTE_TARGET, DAY_START_DELTA
from .training_statistics import OUTPUT_MINIMUM_SCALE, MaskedMoments, MomentStatistics

PERSISTENT_OUTPUT_OWNERS = {
    f"endpoint.ok_leak_aux.{name}": f"slowproc_stomate_previous_step_state.{name}"
    for name in (
        "lignin_struc_above", "lignin_struc_below", "litterpart", "dead_leaves",
        "fuel_1hr", "fuel_10hr", "fuel_100hr", "fuel_1000hr",
    )
}


@dataclass(frozen=True)
class SharedPersistentCoordinates:
    fields: Mapping[str, OutputFieldSpec]
    normalization: NeuralOutputNormalization


def build_shared_persistent_coordinates(schema, contract, state_delta):
    """Read existing full-TRAIN population moments, with no Teacher IO.

    State slices and compact PFT order must match the canonical schema. Pool
    count-weighted within/between-PFT variance, not already-fallback scales.
    """
    if tuple(schema.active_pft_indices) != tuple(contract.active_pft_indices):
        raise ValueError("persistent output/state PFT order differs")
    leaves = {leaf.key: leaf for leaf in contract.state_leaves}
    specs = {field.name: field for field in schema.fields}
    fields, means, scales = {}, {}, {}
    for name, owner in PERSISTENT_OUTPUT_OWNERS.items():
        field, leaf = specs[name], leaves[owner]
        if (field.value_representation != ABSOLUTE_TARGET or field.full_pft_axis
                or tuple(leaf.shape) != tuple(field.shape)
                or leaf.start is None or leaf.stop - leaf.start != np.prod(field.shape)):
            raise ValueError(f"persistent endpoint/state ownership drift: {name}")
        selected = tuple(leaf.selected_pft_indices)
        if field.pft_axis is not None and selected != tuple(schema.active_pft_indices):
            raise ValueError(f"persistent state PFT selection differs: {name}")
        values = {
            key: np.asarray(getattr(state_delta, key)).reshape(-1)[leaf.start:leaf.stop].reshape(field.shape)
            for key in ("count", "mean", "variance")
        }
        if (np.any(values["count"] <= 0) or not np.all(np.isfinite(values["mean"]))
                or not np.all(np.isfinite(values["variance"])) or np.any(values["variance"] < 0)):
            raise ValueError(f"invalid TRAIN residual moments: {name}")
        moments = MomentStatistics(**values, m2=values["variance"] * values["count"],
                                   scale=np.ones(field.shape))
        indices = tuple(range(field.shape[field.pft_axis])) if field.pft_axis is not None else ()
        pooled = MaskedMoments.from_statistics(
            moments, pft_axis=field.pft_axis, pft_indices=indices,
        ).finalize(minimum_scale=OUTPUT_MINIMUM_SCALE)
        fields[name] = replace(
            field, value_representation=DAY_START_DELTA, day_start_owner=owner,
            day_start_state_start=leaf.start, day_start_state_stop=leaf.stop,
        )
        means[name] = pooled.mean
        scales[name] = np.maximum(np.sqrt(pooled.variance), OUTPUT_MINIMUM_SCALE)
    return SharedPersistentCoordinates(fields, NeuralOutputNormalization(mean=means, scale=scales))
