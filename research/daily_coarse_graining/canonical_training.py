"""Training-side adapters for canonical daily Markov neural operators."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, NamedTuple, Sequence

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalModelConfig,
)
from research.daily_coarse_graining.markov_dataset import (
    ORCHIDEE_UNDEFINED_THRESHOLD,
    MarkovDatasetIndex,
    TrainingStatistics,
    defined_numeric_mask,
    denormalize,
    load_markov_shard,
    normalize_finite,
)


class CanonicalTrainingBatch(NamedTuple):
    model_input: CanonicalDayBatch
    normalized_fast_day_target: Any
    fast_day_target_finite: Any
    fast_day_target_undefined: Any
    fast_day_target_undefined_values: Any
    persistent_fast_day_undefined: Any
    persistent_fast_day_undefined_values: Any
    persistent_dynamic_undefined: Any
    dynamic_undefined_flip_target: Any


class CanonicalInferenceBatch(NamedTuple):
    """Target-free model input and persistence metadata for autoregressive use."""

    model_input: CanonicalDayBatch
    persistent_fast_day_undefined: Any
    persistent_fast_day_undefined_values: Any
    persistent_dynamic_undefined: Any


class FastDayTargetRepresentation(NamedTuple):
    """Contract mapping from learned outputs to same-owner day-start state."""

    state_indices: np.ndarray
    dynamic_undefined_indices: np.ndarray
    dynamic_undefined_fill_values: np.ndarray
    nonnegative_indices: tuple[int, ...] = ()


_DYNAMIC_UNDEFINED_OUTPUTS = {
    "diffuco_previous_step_state.rveget": 1.0e20,
}

# These are material stocks, not signed fluxes. The source assumes they remain
# nonnegative before soilcarbon_leak decomposition and PERMA_PEAT redistribution.
# Fortran provenance: stomate_soilcarbon.f90 soilcarbon_leak lines 1348-1437,
# 1605-1838, and 2264-2303; cryoturbate_doc_POC lines 2916-3091 explicitly
# diagnoses negative carbon_32l/DOC as invalid.
_NONNEGATIVE_FAST_DAY_OUTPUTS = frozenset(
    {
        "ok_leak.carbon_32l",
        "ok_leak.DOC",
        "ok_leak.deepC_peat",
    }
)


def fast_day_target_representation_from_contract(
    contract_metadata: Mapping[str, Any],
) -> FastDayTargetRepresentation:
    width = int(contract_metadata["fast_day_target_width"])
    state_by_owner = {}
    for leaf in contract_metadata["state_leaves"]:
        if leaf.get("start") is None:
            continue
        key = (str(leaf["component"]), tuple(str(item) for item in leaf["path"]))
        if key in state_by_owner:
            raise ValueError(f"duplicate continuous state owner {key}")
        state_by_owner[key] = leaf

    state_indices = np.full(width, -1, dtype=np.int32)
    dynamic_indices = []
    dynamic_fill_values = []
    nonnegative_indices = []
    slow_component = "slowproc_stomate_previous_step_state"
    for leaf in contract_metadata["fast_day_target_leaves"]:
        family = str(leaf["family"])
        component = leaf.get("component")
        path = tuple(str(item) for item in leaf["path"])
        key = str(
            leaf.get("key")
            or ".".join((str(component or family), *path))
        )
        if key in _DYNAMIC_UNDEFINED_OUTPUTS:
            dynamic_indices.extend(range(int(leaf["start"]), int(leaf["stop"])))
            dynamic_fill_values.extend(
                [_DYNAMIC_UNDEFINED_OUTPUTS[key]]
                * (int(leaf["stop"]) - int(leaf["start"]))
            )
        if key in _NONNEGATIVE_FAST_DAY_OUTPUTS:
            nonnegative_indices.extend(
                range(int(leaf["start"]), int(leaf["stop"]))
            )
        owner = None
        if component is not None:
            owner = state_by_owner.get((str(component), path))
        elif family == "ok_leak":
            owner = state_by_owner.get((slow_component, path))
        if owner is None:
            continue
        target_start = int(leaf["start"])
        target_stop = int(leaf["stop"])
        state_start = int(owner["start"])
        state_stop = int(owner["stop"])
        if target_stop - target_start != state_stop - state_start:
            raise ValueError(
                "persistence owner width mismatch for "
                f"{component or family}.{'.'.join(path)}"
            )
        state_indices[target_start:target_stop] = np.arange(
            state_start, state_stop, dtype=np.int32
        )
    if not dynamic_indices:
        raise ValueError("fast-day contract is missing dynamic rveget outputs")
    return FastDayTargetRepresentation(
        state_indices=state_indices,
        dynamic_undefined_indices=np.asarray(dynamic_indices, dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray(
            dynamic_fill_values, dtype=np.float64
        ),
        nonnegative_indices=tuple(nonnegative_indices),
    )


def _calendar_features(year: np.ndarray, day_index: np.ndarray) -> np.ndarray:
    year = np.asarray(year, dtype=np.int32)
    day = np.asarray(day_index, dtype=np.float64)
    period = 365.0
    phase = 2.0 * np.pi * (day - 1.0) / period
    return np.stack(
        (
            np.sin(phase),
            np.cos(phase),
            day / period,
            np.zeros_like(year, dtype=np.float64),
        ),
        axis=-1,
    )


def prepare_canonical_batch(
    batch: Mapping[str, Any],
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
) -> CanonicalTrainingBatch:
    """Normalize one collated batch using train-only finite statistics."""

    inference = prepare_canonical_inference_batch(
        batch,
        statistics,
        representation,
    )
    required = (
        "fast_day_target",
    )
    missing = tuple(name for name in required if name not in batch)
    if missing:
        raise ValueError(f"canonical training batch is missing {missing}")
    normalized_target, finite_target = normalize_finite(
        np.asarray(batch["fast_day_target"]),
        statistics.arrays["fast_day_target"],
    )
    target = np.asarray(batch["fast_day_target"], dtype=np.float64)
    state = np.asarray(batch["state"], dtype=np.float64)
    indices = np.asarray(representation.state_indices, dtype=np.int32)
    if indices.shape != (target.shape[-1],):
        raise ValueError("fast-day target representation width mismatch")
    matched = indices >= 0
    selected_indices = np.maximum(indices, 0)
    persisted = state[:, selected_indices]
    target_defined = defined_numeric_mask(target)
    persisted_undefined = ~defined_numeric_mask(persisted) & matched[None, :]
    target_undefined = ~target_defined
    same_undefined_value = (
        (np.isnan(target) & np.isnan(persisted))
        | (np.isfinite(target) & np.isfinite(persisted) & (target == persisted))
    )
    undefined_mismatch = (target_undefined != persisted_undefined) | (
        target_undefined & persisted_undefined & ~same_undefined_value
    )
    dynamic_columns = np.zeros(target.shape[-1], dtype=bool)
    dynamic_columns[representation.dynamic_undefined_indices] = True
    unexpected_mismatch = undefined_mismatch & ~dynamic_columns[None, :]
    if np.any(unexpected_mismatch):
        rows, columns = np.nonzero(unexpected_mismatch)
        first = int(columns[0])
        raise ValueError(
            "fast-day undefined value is not persistent from day-start state: "
            f"column={first}, mismatches={rows.size}"
        )
    dynamic_indices = representation.dynamic_undefined_indices
    persistent_dynamic_undefined = persisted_undefined[:, dynamic_indices]
    target_dynamic_undefined = target_undefined[:, dynamic_indices]
    return CanonicalTrainingBatch(
        model_input=inference.model_input,
        normalized_fast_day_target=normalized_target.astype(np.float32),
        fast_day_target_finite=finite_target,
        fast_day_target_undefined=target_undefined,
        fast_day_target_undefined_values=np.where(
            target_undefined, target, 0.0
        ),
        persistent_fast_day_undefined=persisted_undefined,
        persistent_fast_day_undefined_values=np.where(
            persisted_undefined, persisted, 0.0
        ),
        persistent_dynamic_undefined=persistent_dynamic_undefined,
        dynamic_undefined_flip_target=np.logical_xor(
            persistent_dynamic_undefined,
            target_dynamic_undefined,
        ),
    )


def prepare_canonical_inference_batch(
    batch: Mapping[str, Any],
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
) -> CanonicalInferenceBatch:
    """Build canonical model inputs without reading a Teacher target."""

    required = (
        "state",
        "forcing_native",
        "parameters",
        "landpoint_static",
        "annual_conditions",
        "year",
        "day_index",
    )
    missing = tuple(name for name in required if name not in batch)
    if missing:
        raise ValueError(f"canonical inference batch is missing {missing}")
    normalized = {}
    finite = {}
    for name in required[:5]:
        if name not in statistics.arrays:
            raise ValueError(f"training statistics are missing {name}")
        normalized[name], finite[name] = normalize_finite(
            np.asarray(batch[name]), statistics.arrays[name]
        )

    state = np.asarray(batch["state"], dtype=np.float64)
    indices = np.asarray(representation.state_indices, dtype=np.int32)
    matched = indices >= 0
    selected_indices = np.maximum(indices, 0)
    persisted = state[:, selected_indices]
    persisted_defined = defined_numeric_mask(persisted) & matched[None, :]
    persisted_undefined = ~defined_numeric_mask(persisted) & matched[None, :]
    target_statistics = statistics.arrays["fast_day_target"]
    baseline = np.zeros(persisted.shape, dtype=np.float64)
    np.subtract(
        persisted,
        target_statistics.mean,
        out=baseline,
        where=persisted_defined,
    )
    np.divide(
        baseline,
        target_statistics.scale,
        out=baseline,
        where=persisted_defined,
    )
    dynamic_indices = representation.dynamic_undefined_indices
    return CanonicalInferenceBatch(
        model_input=CanonicalDayBatch(
            state=normalized["state"].astype(np.float32),
            state_finite=finite["state"],
            normalized_fast_day_baseline=baseline.astype(np.float32),
            forcing_native=normalized["forcing_native"].astype(np.float32),
            forcing_finite=finite["forcing_native"],
            parameters=normalized["parameters"].astype(np.float32),
            parameters_finite=finite["parameters"],
            landpoint_static=normalized["landpoint_static"].astype(np.float32),
            landpoint_static_finite=finite["landpoint_static"],
            annual_conditions=normalized["annual_conditions"].astype(np.float32),
            annual_conditions_finite=finite["annual_conditions"],
            calendar=_calendar_features(batch["year"], batch["day_index"]).astype(
                np.float32
            ),
        ),
        persistent_fast_day_undefined=persisted_undefined,
        persistent_fast_day_undefined_values=np.where(
            persisted_undefined, persisted, 0.0
        ),
        persistent_dynamic_undefined=persisted_undefined[:, dynamic_indices],
    )


def _defined_numeric_mask_compiled(values):
    values = jnp.asarray(values)
    return jnp.isfinite(values) & (
        jnp.abs(values) < jnp.asarray(ORCHIDEE_UNDEFINED_THRESHOLD, values.dtype)
    )


def _normalize_finite_compiled(values, statistics):
    values = jnp.asarray(values, dtype=jnp.float64)
    finite = _defined_numeric_mask_compiled(values)
    mean = jnp.asarray(statistics.mean)
    scale = jnp.asarray(statistics.scale)
    safe_values = jnp.where(finite, values, 0.0)
    safe_mean = jnp.where(finite, mean, 0.0)
    safe_scale = jnp.where(finite, scale, 1.0)
    normalized = (safe_values - safe_mean) / safe_scale
    normalized = jnp.where(
        finite,
        normalized,
        0.0,
    )
    return normalized, finite


def prepare_canonical_inference_batch_compiled(
    batch: Mapping[str, Any],
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
) -> CanonicalInferenceBatch:
    """JAX-traceable target-free adapter for recursive daily training."""

    required = (
        "state",
        "forcing_native",
        "parameters",
        "landpoint_static",
        "annual_conditions",
        "year",
        "day_index",
    )
    missing = tuple(name for name in required if name not in batch)
    if missing:
        raise ValueError(f"canonical inference batch is missing {missing}")
    normalized = {}
    finite = {}
    for name in required[:5]:
        if name not in statistics.arrays:
            raise ValueError(f"training statistics are missing {name}")
        normalized[name], finite[name] = _normalize_finite_compiled(
            batch[name], statistics.arrays[name]
        )

    state = jnp.asarray(batch["state"], dtype=jnp.float64)
    indices = jnp.asarray(representation.state_indices, dtype=jnp.int32)
    matched = indices >= 0
    selected_indices = jnp.maximum(indices, 0)
    persisted = jnp.take(state, selected_indices, axis=-1)
    persisted_defined = _defined_numeric_mask_compiled(persisted) & matched
    persisted_undefined = ~_defined_numeric_mask_compiled(persisted) & matched
    target_statistics = statistics.arrays["fast_day_target"]
    baseline = jnp.where(
        persisted_defined,
        (persisted - jnp.asarray(target_statistics.mean))
        / jnp.asarray(target_statistics.scale),
        0.0,
    )
    dynamic_indices = jnp.asarray(
        representation.dynamic_undefined_indices, dtype=jnp.int32
    )
    year = jnp.asarray(batch["year"], dtype=jnp.float64)
    day = jnp.asarray(batch["day_index"], dtype=jnp.float64)
    del year
    phase = 2.0 * jnp.pi * (day - 1.0) / 365.0
    calendar = jnp.stack(
        (
            jnp.sin(phase),
            jnp.cos(phase),
            day / 365.0,
            jnp.zeros_like(day),
        ),
        axis=-1,
    )
    return CanonicalInferenceBatch(
        model_input=CanonicalDayBatch(
            state=normalized["state"].astype(jnp.float32),
            state_finite=finite["state"],
            normalized_fast_day_baseline=baseline.astype(jnp.float32),
            forcing_native=normalized["forcing_native"].astype(jnp.float32),
            forcing_finite=finite["forcing_native"],
            parameters=normalized["parameters"].astype(jnp.float32),
            parameters_finite=finite["parameters"],
            landpoint_static=normalized["landpoint_static"].astype(jnp.float32),
            landpoint_static_finite=finite["landpoint_static"],
            annual_conditions=normalized["annual_conditions"].astype(jnp.float32),
            annual_conditions_finite=finite["annual_conditions"],
            calendar=calendar.astype(jnp.float32),
        ),
        persistent_fast_day_undefined=persisted_undefined,
        persistent_fast_day_undefined_values=jnp.where(
            persisted_undefined, persisted, 0.0
        ),
        persistent_dynamic_undefined=jnp.take(
            persisted_undefined, dynamic_indices, axis=-1
        ),
    )


def restore_fast_day_inference_prediction(
    normalized_prediction: np.ndarray,
    dynamic_undefined_flip_logits: np.ndarray,
    batch: CanonicalInferenceBatch,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
) -> np.ndarray:
    """Restore one target-free model prediction to physical Teacher units."""

    physical = denormalize(
        np.asarray(normalized_prediction), statistics.arrays["fast_day_target"]
    )
    if representation.nonnegative_indices:
        indices = np.asarray(representation.nonnegative_indices, dtype=np.int32)
        physical[..., indices] = np.maximum(physical[..., indices], 0.0)
    undefined = np.asarray(batch.persistent_fast_day_undefined).copy()
    undefined_values = np.asarray(
        batch.persistent_fast_day_undefined_values
    ).copy()
    dynamic_flip = np.asarray(dynamic_undefined_flip_logits) >= 0.0
    dynamic = np.logical_xor(
        np.asarray(batch.persistent_dynamic_undefined),
        dynamic_flip,
    )
    undefined[:, representation.dynamic_undefined_indices] = dynamic
    undefined_values[:, representation.dynamic_undefined_indices] = (
        representation.dynamic_undefined_fill_values[None, :]
    )
    return np.where(undefined, undefined_values, physical)


def restore_fast_day_inference_prediction_compiled(
    normalized_prediction,
    dynamic_undefined_flip_logits,
    batch: CanonicalInferenceBatch,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
):
    """JAX-traceable inverse target representation for recursive rollout."""

    target_statistics = statistics.arrays["fast_day_target"]
    physical = (
        jnp.asarray(normalized_prediction, dtype=jnp.float64)
        * jnp.asarray(target_statistics.scale)
        + jnp.asarray(target_statistics.mean)
    )
    nonnegative_indices = jnp.asarray(
        representation.nonnegative_indices, dtype=jnp.int32
    )
    physical = physical.at[..., nonnegative_indices].set(
        jnp.maximum(jnp.take(physical, nonnegative_indices, axis=-1), 0.0)
    )
    dynamic_indices = jnp.asarray(
        representation.dynamic_undefined_indices, dtype=jnp.int32
    )
    dynamic_flip = jnp.asarray(dynamic_undefined_flip_logits) >= 0.0
    dynamic_undefined = jnp.logical_xor(
        jnp.asarray(batch.persistent_dynamic_undefined),
        dynamic_flip,
    )
    undefined = jnp.asarray(batch.persistent_fast_day_undefined).at[
        ..., dynamic_indices
    ].set(dynamic_undefined)
    dynamic_fill = jnp.asarray(representation.dynamic_undefined_fill_values)
    undefined_values = jnp.asarray(
        batch.persistent_fast_day_undefined_values
    ).at[..., dynamic_indices].set(dynamic_fill)
    return jnp.where(undefined, undefined_values, physical)


def restore_fast_day_prediction(
    normalized_prediction: np.ndarray,
    dynamic_undefined_flip_logits: np.ndarray,
    batch: CanonicalTrainingBatch,
    statistics: TrainingStatistics,
    representation: FastDayTargetRepresentation,
) -> np.ndarray:
    """Restore physical values and source-defined undefined input values."""

    inference = CanonicalInferenceBatch(
        model_input=batch.model_input,
        persistent_fast_day_undefined=batch.persistent_fast_day_undefined,
        persistent_fast_day_undefined_values=(
            batch.persistent_fast_day_undefined_values
        ),
        persistent_dynamic_undefined=batch.persistent_dynamic_undefined,
    )
    return restore_fast_day_inference_prediction(
        normalized_prediction,
        dynamic_undefined_flip_logits,
        inference,
        statistics,
        representation,
    )


def restore_fast_day_target(
    normalized_target: np.ndarray,
    batch: CanonicalTrainingBatch,
    statistics: TrainingStatistics,
) -> np.ndarray:
    """Restore physical Teacher values including their exact undefined kind."""

    physical = denormalize(
        np.asarray(normalized_target), statistics.arrays["fast_day_target"]
    )
    return np.where(
        np.asarray(batch.fast_day_target_undefined),
        np.asarray(batch.fast_day_target_undefined_values),
        physical,
    )


def audit_fast_day_target_representation(
    index: MarkovDatasetIndex,
    contract_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit that every undefined learned output is restorable from S[d]."""

    representation = fast_day_target_representation_from_contract(
        contract_metadata
    )
    indices = np.asarray(representation.state_indices, dtype=np.int32)
    matched = indices >= 0
    selected_indices = np.maximum(indices, 0)
    leaves = tuple(contract_metadata["fast_day_target_leaves"])
    leaf_counts = {
        str(leaf["key"]): {
            "undefined_values": 0,
            "nan_values": 0,
            "sentinel_values": 0,
            "persistence_mismatches": 0,
        }
        for leaf in leaves
    }
    transitions = 0
    undefined_values = 0
    persistence_mismatches = 0
    for reference in index.shards:
        shard = load_markov_shard(reference.path)
        target = np.asarray(shard.fast_day_target, dtype=np.float64)
        state = np.asarray(shard.state_trajectory[:-1], dtype=np.float64)
        if target.shape[-1] != indices.size:
            raise ValueError(
                "fast-day target audit width mismatch at "
                f"{reference.landpoint_id}:{reference.year}"
            )
        persisted = state[:, selected_indices]
        target_undefined = ~defined_numeric_mask(target)
        persisted_undefined = (
            ~defined_numeric_mask(persisted) & matched[None, :]
        )
        same_undefined_value = (
            (np.isnan(target) & np.isnan(persisted))
            | (
                np.isfinite(target)
                & np.isfinite(persisted)
                & (target == persisted)
            )
        )
        mismatch = (target_undefined != persisted_undefined) | (
            target_undefined & persisted_undefined & ~same_undefined_value
        )
        transitions += shard.days
        undefined_values += int(np.count_nonzero(target_undefined))
        dynamic_columns = np.zeros(target.shape[-1], dtype=bool)
        dynamic_columns[representation.dynamic_undefined_indices] = True
        unexpected_mismatch = mismatch & ~dynamic_columns[None, :]
        persistence_mismatches += int(
            np.count_nonzero(unexpected_mismatch)
        )
        for leaf in leaves:
            start = int(leaf["start"])
            stop = int(leaf["stop"])
            selected = target[:, start:stop]
            selected_undefined = target_undefined[:, start:stop]
            counts = leaf_counts[str(leaf["key"])]
            counts["undefined_values"] += int(
                np.count_nonzero(selected_undefined)
            )
            counts["nan_values"] += int(np.count_nonzero(np.isnan(selected)))
            counts["sentinel_values"] += int(
                np.count_nonzero(
                    np.isfinite(selected) & ~defined_numeric_mask(selected)
                )
            )
            counts["persistence_mismatches"] += int(
                np.count_nonzero(unexpected_mismatch[:, start:stop])
            )
    active_leaves = {
        key: value
        for key, value in leaf_counts.items()
        if value["undefined_values"] or value["persistence_mismatches"]
    }
    return {
        "dataset_id": index.dataset_id,
        "teacher_git_head": index.teacher_git_head,
        "markov_contract_sha256": index.contract_sha256,
        "shards": len(index.shards),
        "transitions": transitions,
        "target_width": int(indices.size),
        "persistence_mapped_columns": int(np.count_nonzero(matched)),
        "mean_centered_columns": int(np.count_nonzero(~matched)),
        "undefined_values": undefined_values,
        "persistence_mismatches": persistence_mismatches,
        "dynamic_undefined_columns": int(
            representation.dynamic_undefined_indices.size
        ),
        "status": "passed" if persistence_mismatches == 0 else "failed",
        "undefined_leaves": active_leaves,
    }


def model_config_from_batch(
    batch: CanonicalTrainingBatch,
    **architecture_widths: int,
) -> CanonicalModelConfig:
    inputs = batch.model_input
    return CanonicalModelConfig(
        state_width=int(inputs.state.shape[-1]),
        forcing_width=int(inputs.forcing_native.shape[-1]),
        parameter_width=int(inputs.parameters.shape[-1]),
        landpoint_static_width=int(inputs.landpoint_static.shape[-1]),
        annual_condition_width=int(inputs.annual_conditions.shape[-1]),
        fast_day_target_width=int(batch.normalized_fast_day_target.shape[-1]),
        dynamic_undefined_width=int(batch.dynamic_undefined_flip_target.shape[-1]),
        **architecture_widths,
    )


def _balanced_slice_weights(
    width: int,
    slices: Sequence[tuple[str, int, int]],
) -> np.ndarray:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    occupied = np.zeros((width,), dtype=bool)
    for group, start, stop in slices:
        if start < 0 or stop <= start or stop > width or np.any(occupied[start:stop]):
            raise ValueError(f"invalid or overlapping contract slice {group}[{start}:{stop}]")
        groups[group].append((start, stop))
        occupied[start:stop] = True
    if not groups or not np.all(occupied):
        raise ValueError("contract slices must cover every learned output")
    result = np.zeros((width,), dtype=np.float32)
    for ranges in groups.values():
        group_width = sum(stop - start for start, stop in ranges)
        value = 1.0 / (len(groups) * group_width)
        for start, stop in ranges:
            result[start:stop] = value
    return result


def loss_weights_from_contract(
    contract_metadata: Mapping[str, Any],
) -> np.ndarray:
    target_width = int(contract_metadata["fast_day_target_width"])
    target_slices = tuple(
        (str(leaf["family"]), int(leaf["start"]), int(leaf["stop"]))
        for leaf in contract_metadata["fast_day_target_leaves"]
    )
    return _balanced_slice_weights(target_width, target_slices)


def audit_discrete_persistence(index: MarkovDatasetIndex) -> dict[str, Any]:
    """Measure which exact state leaves require a non-persistence update rule."""

    changes: dict[str, int] = defaultdict(int)
    values: dict[str, int] = defaultdict(int)
    transitions = 0
    schema: tuple[str, ...] | None = None
    for reference in index.shards:
        shard = load_markov_shard(reference.path)
        observed = tuple(sorted(shard.discrete_trajectories))
        if schema is None:
            schema = observed
        elif observed != schema:
            raise ValueError(f"discrete-state schema drift at {reference.landpoint_id}:{reference.year}")
        for name, trajectory in shard.discrete_trajectories.items():
            left = np.asarray(trajectory[:-1])
            right = np.asarray(trajectory[1:])
            changed = left != right
            if changed.ndim > 1:
                changed_transition = np.any(changed.reshape((changed.shape[0], -1)), axis=1)
            else:
                changed_transition = changed
            changes[name] += int(np.count_nonzero(changed_transition))
            values[name] += int(changed.size)
        transitions += shard.days
    return {
        "shards": len(index.shards),
        "transitions": transitions,
        "leaves": {
            name: {
                "changed_transitions": changes[name],
                "persistence_transition_fraction": 1.0 - changes[name] / transitions,
                "compared_values": values[name],
            }
            for name in schema or ()
        },
        "all_persistent": all(changes[name] == 0 for name in schema or ()),
    }
