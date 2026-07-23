"""Training-side adapters for canonical daily Markov neural operators."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalModelConfig,
)
from research.daily_coarse_graining.markov_dataset import (
    MarkovDatasetIndex,
    TrainingStatistics,
    load_markov_shard,
    normalize_finite,
)


class CanonicalTrainingBatch(NamedTuple):
    model_input: CanonicalDayBatch
    normalized_fast_day_target: Any
    fast_day_target_finite: Any


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
) -> CanonicalTrainingBatch:
    """Normalize one collated batch using train-only finite statistics."""

    required = (
        "state",
        "forcing_native",
        "parameters",
        "landpoint_static",
        "annual_conditions",
        "fast_day_target",
        "year",
        "day_index",
    )
    missing = tuple(name for name in required if name not in batch)
    if missing:
        raise ValueError(f"canonical training batch is missing {missing}")
    normalized = {}
    finite = {}
    for name in required[:6]:
        if name not in statistics.arrays:
            raise ValueError(f"training statistics are missing {name}")
        normalized[name], finite[name] = normalize_finite(
            np.asarray(batch[name]), statistics.arrays[name]
        )
    model_input = CanonicalDayBatch(
        state=normalized["state"].astype(np.float32),
        state_finite=finite["state"],
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
    )
    return CanonicalTrainingBatch(
        model_input=model_input,
        normalized_fast_day_target=normalized["fast_day_target"].astype(
            np.float32
        ),
        fast_day_target_finite=finite["fast_day_target"],
    )


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
