"""Streaming train/train statistics for the first neural process head.

This host-only module vectorizes target construction over each coherent v6
shard.  It deliberately follows :mod:`training_data` field ownership without
calling the single-example adapter millions of times.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.constrained_daily_replay import (
    CARBON_INVENTORY_FIELDS,
    THERMAL_FIELDS,
    WATER_INVENTORY_FIELDS,
    encode_inventory_endpoints,
    encode_signed_inventory_endpoints,
    encode_thermal_endpoints,
)
from research.daily_coarse_graining.daily_markov_contract import MarkovShard
from research.daily_coarse_graining.markov_dataset import TrainingStatistics
from research.daily_coarse_graining.typed_sidecar import (
    TypedSidecarIndex,
    read_typed_sidecar_shard,
)

from .neural_architecture import (
    NeuralInputNormalization,
    NeuralOutputNormalization,
    NeuralOutputSchema,
    OutputFieldSpec,
    build_neural_output_schema,
    build_process_axis_layout,
)
from .training_contract import FirstCandidateContract
from .training_data import DailyProcessTrainingAdapter, MaskedProcessTarget

SCHEMA_VERSION = "daily_process_head_training_statistics_v1"
ORCHIDEE_UNDEFINED_THRESHOLD = 0.5e20
OUTPUT_MINIMUM_SCALE = 1.0e-12
SEMANTIC_INPUT_MINIMUM_SCALE = 1.0e-6
INPUT_NAMES = (
    "continuous",
    "forcing",
    "static",
    "annual",
    "water",
    "carbon",
    "thermal",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MomentStatistics:
    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray
    variance: np.ndarray
    scale: np.ndarray


class MaskedMoments:
    """Mergeable population moments with optional PFT-shared finalization."""

    def __init__(
        self,
        feature_shape: Sequence[int],
        *,
        pft_axis: int | None = None,
        pft_indices: Sequence[int] = (),
    ) -> None:
        self.feature_shape = tuple(int(value) for value in feature_shape)
        self.width = int(np.prod(self.feature_shape, dtype=np.int64))
        self.width = max(self.width, 1)
        self.pft_axis = pft_axis
        self.pft_indices = tuple(int(value) for value in pft_indices)
        if pft_axis is not None:
            if not 0 <= pft_axis < len(self.feature_shape):
                raise ValueError("PFT moment axis is outside the feature shape")
            if not self.pft_indices:
                raise ValueError("PFT-shared moments require active PFT indices")
            extent = self.feature_shape[pft_axis]
            if any(index < 0 or index >= extent for index in self.pft_indices):
                raise ValueError("PFT moment index is outside the feature shape")
        self.count = np.zeros(self.width, dtype=np.uint64)
        self.mean = np.zeros(self.width, dtype=np.float64)
        self.m2 = np.zeros(self.width, dtype=np.float64)

    def _merge_arrays(
        self,
        batch_count: np.ndarray,
        batch_mean: np.ndarray,
        batch_m2: np.ndarray,
    ) -> None:
        batch_count = np.asarray(batch_count, dtype=np.uint64).reshape(self.width)
        batch_mean = np.asarray(batch_mean, dtype=np.float64).reshape(self.width)
        batch_m2 = np.asarray(batch_m2, dtype=np.float64).reshape(self.width)
        total = self.count + batch_count
        active = batch_count > 0
        delta = batch_mean - self.mean
        weight = np.zeros(self.width, dtype=np.float64)
        np.divide(batch_count, total, out=weight, where=active)
        cross = np.zeros(self.width, dtype=np.float64)
        np.divide(self.count * batch_count, total, out=cross, where=active)
        self.mean = np.where(active, self.mean + delta * weight, self.mean)
        self.m2 = np.where(active, self.m2 + batch_m2 + delta * delta * cross, self.m2)
        self.count = total

    def update(
        self,
        values: Any,
        defined: Any,
        *,
        chunk_rows: int = 64,
    ) -> None:
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        values = np.asarray(values, dtype=np.float64)
        defined = np.asarray(defined, dtype=np.bool_)
        expected_tail = self.feature_shape
        if values.ndim != len(expected_tail) + 1 or values.shape[1:] != expected_tail:
            raise ValueError(
                f"moment value shape drift: {values.shape} does not end in {expected_tail}"
            )
        if defined.shape != values.shape:
            raise ValueError("moment defined-mask shape drift")

        if self.pft_axis is not None:
            feature_axis = self.pft_axis + 1
            active = np.zeros(self.feature_shape[self.pft_axis], dtype=np.bool_)
            active[np.asarray(self.pft_indices, dtype=np.int64)] = True
            view = [1] * values.ndim
            view[feature_axis] = active.size
            defined = defined & active.reshape(view)

        rows = values.reshape((-1, self.width))
        masks = defined.reshape((-1, self.width))
        for start in range(0, rows.shape[0], chunk_rows):
            chunk = rows[start : start + chunk_rows]
            mask = masks[start : start + chunk_rows]
            valid = (
                mask
                & np.isfinite(chunk)
                & (np.abs(chunk) < ORCHIDEE_UNDEFINED_THRESHOLD)
            )
            safe = np.where(valid, chunk, 0.0)
            count = valid.sum(axis=0, dtype=np.uint64)
            mean = np.zeros(self.width, dtype=np.float64)
            np.divide(safe.sum(axis=0), count, out=mean, where=count > 0)
            centered = np.where(valid, chunk - mean, 0.0)
            self._merge_arrays(count, mean, np.sum(centered * centered, axis=0))

    def merge(self, other: "MaskedMoments") -> None:
        identity = (self.feature_shape, self.pft_axis, self.pft_indices)
        expected = (other.feature_shape, other.pft_axis, other.pft_indices)
        if identity != expected:
            raise ValueError("cannot merge moments with different feature/PFT layouts")
        self._merge_arrays(other.count, other.mean, other.m2)

    @classmethod
    def from_statistics(
        cls,
        statistics: MomentStatistics,
        *,
        pft_axis: int | None = None,
        pft_indices: Sequence[int] = (),
    ) -> "MaskedMoments":
        result = cls(
            statistics.mean.shape,
            pft_axis=pft_axis,
            pft_indices=pft_indices,
        )
        result.count = np.asarray(statistics.count, dtype=np.uint64).reshape(-1).copy()
        result.mean = np.asarray(statistics.mean, dtype=np.float64).reshape(-1).copy()
        result.m2 = np.asarray(statistics.m2, dtype=np.float64).reshape(-1).copy()
        return result

    @staticmethod
    def _pooled_populations(
        count: np.ndarray,
        mean: np.ndarray,
        m2: np.ndarray,
        *,
        axis: int,
        indices: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        moved_count = np.moveaxis(count, axis, 0)
        moved_mean = np.moveaxis(mean, axis, 0)
        moved_m2 = np.moveaxis(m2, axis, 0)
        selected_count = moved_count[np.asarray(indices)]
        selected_mean = moved_mean[np.asarray(indices)]
        selected_m2 = moved_m2[np.asarray(indices)]
        total = selected_count.sum(axis=0, dtype=np.uint64)
        weighted = np.sum(selected_mean * selected_count, axis=0, dtype=np.float64)
        pooled_mean = np.zeros_like(weighted)
        np.divide(weighted, total, out=pooled_mean, where=total > 0)
        pooled_m2 = np.sum(
            selected_m2
            + selected_count * (selected_mean - pooled_mean[None, ...]) ** 2,
            axis=0,
            dtype=np.float64,
        )
        result_count = np.zeros_like(moved_count)
        result_mean = np.zeros_like(moved_mean)
        result_m2 = np.zeros_like(moved_m2)
        for index in indices:
            result_count[index] = total
            result_mean[index] = pooled_mean
            result_m2[index] = pooled_m2
        return (
            np.moveaxis(result_count, 0, axis),
            np.moveaxis(result_mean, 0, axis),
            np.moveaxis(result_m2, 0, axis),
        )

    def finalize(self, *, minimum_scale: float) -> MomentStatistics:
        if minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")
        shape = self.feature_shape
        count = self.count.reshape(shape).copy()
        mean = self.mean.reshape(shape).copy()
        m2 = self.m2.reshape(shape).copy()
        if self.pft_axis is not None:
            count, mean, m2 = self._pooled_populations(
                count,
                mean,
                m2,
                axis=self.pft_axis,
                indices=self.pft_indices,
            )
        variance = np.zeros(shape, dtype=np.float64)
        np.divide(m2, count, out=variance, where=count > 0)
        variance = np.maximum(variance, 0.0)
        standard_deviation = np.sqrt(variance)
        scale = np.where(
            (count > 1) & np.isfinite(standard_deviation) & (standard_deviation > minimum_scale),
            standard_deviation,
            1.0,
        )
        return MomentStatistics(
            count=count,
            mean=mean,
            m2=m2,
            variance=variance,
            scale=scale,
        )


def _share_semantic_pft_moments(
    statistics: MomentStatistics,
    element_pft_index: np.ndarray,
    *,
    minimum_scale: float,
) -> MomentStatistics:
    pft = np.asarray(element_pft_index, dtype=np.int32).reshape(-1)
    slots = tuple(int(value) for value in np.unique(pft) if value >= 0)
    if not slots:
        return statistics
    groups = tuple(np.flatnonzero(pft == slot) for slot in slots)
    if len({group.size for group in groups}) != 1:
        raise ValueError("semantic PFT layout does not have matched slot coordinates")
    count = np.asarray(statistics.count).reshape(-1).copy()
    mean = np.asarray(statistics.mean).reshape(-1).copy()
    m2 = np.asarray(statistics.m2).reshape(-1).copy()
    for coordinates in zip(*groups, strict=True):
        selected = np.asarray(coordinates, dtype=np.int64)
        pooled_count = count[selected].sum(dtype=np.uint64)
        if pooled_count:
            pooled_mean = float(np.sum(count[selected] * mean[selected]) / pooled_count)
            pooled_m2 = float(
                np.sum(m2[selected] + count[selected] * (mean[selected] - pooled_mean) ** 2)
            )
        else:
            pooled_mean = 0.0
            pooled_m2 = 0.0
        count[selected] = pooled_count
        mean[selected] = pooled_mean
        m2[selected] = pooled_m2
    variance = np.zeros_like(mean)
    np.divide(m2, count, out=variance, where=count > 0)
    variance = np.maximum(variance, 0.0)
    standard_deviation = np.sqrt(variance)
    scale = np.where(
        (count > 1) & np.isfinite(standard_deviation) & (standard_deviation > minimum_scale),
        standard_deviation,
        1.0,
    )
    shape = statistics.mean.shape
    return MomentStatistics(
        count=count.reshape(shape),
        mean=mean.reshape(shape),
        m2=m2.reshape(shape),
        variance=variance.reshape(shape),
        scale=scale.reshape(shape),
    )


def _state_leaf(adapter: DailyProcessTrainingAdapter, component: str, field: str) -> Any:
    matches = tuple(
        leaf
        for leaf in adapter.contract.state_leaves
        if not leaf.discrete and leaf.component == component and leaf.path == (field,)
    )
    if len(matches) != 1:
        raise ValueError(f"process statistics require one state leaf {component}.{field}")
    return matches[0]


def _fast_leaf(
    adapter: DailyProcessTrainingAdapter,
    *,
    component: str | None,
    family: str | None,
    field: str,
) -> Any:
    matches = tuple(
        leaf
        for leaf in adapter.contract.fast_day_target_leaves
        if leaf.component == component
        and (family is None or leaf.family == family)
        and leaf.path == (field,)
    )
    if len(matches) != 1:
        owner = component if component is not None else family
        raise ValueError(f"process statistics require one fast leaf {owner}.{field}")
    return matches[0]


def _state_values(shard: MarkovShard, leaf: Any) -> np.ndarray:
    return shard.state_trajectory[:-1, leaf.start : leaf.stop].reshape(
        (shard.days, *leaf.shape)
    )


def _fast_values(shard: MarkovShard, leaf: Any) -> np.ndarray:
    return shard.fast_day_target[:, leaf.start : leaf.stop].reshape(
        (shard.days, *leaf.shape)
    )


def _inflate_pft_batch(values: np.ndarray, leaf: Any) -> np.ndarray:
    values = np.asarray(values)
    if not leaf.selected_pft_indices:
        return values.reshape((values.shape[0], *leaf.full_shape))
    axes = tuple(index for index, name in enumerate(leaf.axis_names) if name == "nvm")
    if len(axes) != 1:
        raise ValueError(f"unsupported PFT-axis count for {leaf.key}: {axes}")
    axis = axes[0]
    result = np.zeros((values.shape[0], *leaf.full_shape), dtype=values.dtype)
    for source_index, target_index in enumerate(leaf.selected_pft_indices):
        target = [slice(None)] * result.ndim
        source = [slice(None)] * values.ndim
        target[axis + 1] = target_index
        source[axis + 1] = source_index
        result[tuple(target)] = values[tuple(source)]
    return result


def _defined(values: Any) -> np.ndarray:
    values = np.asarray(values)
    return np.isfinite(values) & (np.abs(values) < ORCHIDEE_UNDEFINED_THRESHOLD)


def _add_inventory(
    result: dict[str, MaskedProcessTarget],
    prefix: str,
    transition: Any,
) -> None:
    result[f"{prefix}.outgoing_fraction"] = MaskedProcessTarget(
        np.asarray(transition.outgoing_fraction, dtype=np.float64),
        np.asarray(transition.defined, dtype=np.bool_),
    )
    result[f"{prefix}.incoming_amount"] = MaskedProcessTarget(
        np.asarray(transition.incoming_amount, dtype=np.float64),
        np.asarray(transition.defined, dtype=np.bool_),
    )


def vectorized_process_targets(
    shard: MarkovShard,
    typed_values: Mapping[str, np.ndarray],
    typed_defined: Mapping[str, np.ndarray],
    adapter: DailyProcessTrainingAdapter,
) -> dict[str, MaskedProcessTarget]:
    """Construct the 101 named target batches for one coherent annual shard."""

    expected_typed = set(adapter.typed_contract.fields_by_path)
    if set(typed_values) != expected_typed or set(typed_defined) != expected_typed:
        raise ValueError("process statistics typed registry drift")
    result: dict[str, MaskedProcessTarget] = {}

    for field in WATER_INVENTORY_FIELDS:
        start_leaf = _state_leaf(adapter, "hydrol_previous_step_state", field)
        end_leaf = _fast_leaf(
            adapter,
            component="hydrol_previous_step_state",
            family="hydrol",
            field=field,
        )
        start = _state_values(shard, start_leaf)
        end = _fast_values(shard, end_leaf)
        if field == "qsintveg":
            transition = encode_signed_inventory_endpoints(start, end, name=f"water.{field}")
            _add_inventory(result, f"inventory.water.{field}.storage", transition.positive)
            _add_inventory(result, f"inventory.water.{field}.debt", transition.debt)
        else:
            transition = encode_inventory_endpoints(start, end, name=f"water.{field}")
            _add_inventory(result, f"inventory.water.{field}", transition)

    for field in CARBON_INVENTORY_FIELDS:
        start_leaf = _state_leaf(adapter, "slowproc_stomate_previous_step_state", field)
        end_leaf = _fast_leaf(adapter, component=None, family="ok_leak", field=field)
        transition = encode_inventory_endpoints(
            _state_values(shard, start_leaf),
            _fast_values(shard, end_leaf),
            name=f"carbon.{field}",
        )
        _add_inventory(result, f"inventory.carbon.{field}", transition)

    available = {leaf.key for leaf in adapter.contract.state_leaves}
    for component, field, bound in THERMAL_FIELDS:
        if f"{component}.{field}" not in available:
            continue
        start_leaf = _state_leaf(adapter, component, field)
        end_leaf = _fast_leaf(
            adapter,
            component=component,
            family=None,
            field=field,
        )
        transition = encode_thermal_endpoints(
            _inflate_pft_batch(_state_values(shard, start_leaf), start_leaf),
            _inflate_pft_batch(_fast_values(shard, end_leaf), end_leaf),
            name=f"energy.{component}.{field}",
            bound=bound,
        )
        result[f"thermal.{component}.{field}.normalized_tendency"] = MaskedProcessTarget(
            np.asarray(transition.normalized_tendency, dtype=np.float64),
            np.asarray(transition.defined, dtype=np.bool_),
        )

    for source, key in (
        ("gpp_daily", "daily.fast_gpp"),
        ("resp_maint_part", "daily.fast_maintenance_respiration"),
        ("tsurf_daily", "daily.temperature_context.tsurf"),
        ("tsoil_daily", "daily.temperature_context.tsoil"),
    ):
        leaf = _fast_leaf(adapter, component=None, family="daily_interface", field=source)
        value = _fast_values(shard, leaf)
        result[key] = MaskedProcessTarget(value, _defined(value))

    for field in adapter.typed_contract.fields:
        value = np.asarray(typed_values[field.path])
        defined = np.asarray(typed_defined[field.path])
        expected_shape = (shard.days, *field.feature_shape)
        if value.shape != expected_shape or value.dtype != np.dtype(field.dtype):
            raise ValueError(f"typed process value shape/dtype drift for {field.path}")
        if defined.shape != expected_shape or defined.dtype != np.dtype(np.bool_):
            raise ValueError(f"typed process mask drift for {field.path}")
        result[f"typed.{field.path}"] = MaskedProcessTarget(value, defined)

    expected = adapter.field_labels()
    if set(result) != set(expected):
        raise ValueError("vectorized process target registry does not match the adapter")
    return result


def vectorized_semantic_state(
    shard: MarkovShard,
    adapter: DailyProcessTrainingAdapter,
) -> dict[str, MaskedProcessTarget]:
    """Construct the three semantic state domains used by the neural tokenizer."""

    water = []
    for field in WATER_INVENTORY_FIELDS:
        leaf = _state_leaf(adapter, "hydrol_previous_step_state", field)
        value = _state_values(shard, leaf)
        if field == "qsintveg":
            water.extend((np.maximum(value, 0.0), np.maximum(-value, 0.0)))
        else:
            water.append(value)
    carbon = [
        _state_values(
            shard,
            _state_leaf(adapter, "slowproc_stomate_previous_step_state", field),
        )
        for field in CARBON_INVENTORY_FIELDS
    ]
    thermal = []
    available = {leaf.key for leaf in adapter.contract.state_leaves}
    for component, field, _bound in THERMAL_FIELDS:
        if f"{component}.{field}" not in available:
            continue
        leaf = _state_leaf(adapter, component, field)
        thermal.append(_inflate_pft_batch(_state_values(shard, leaf), leaf))

    values = {
        "water": np.concatenate(tuple(value.reshape((shard.days, -1)) for value in water), axis=1),
        "carbon": np.concatenate(tuple(value.reshape((shard.days, -1)) for value in carbon), axis=1),
        "thermal": np.concatenate(tuple(value.reshape((shard.days, -1)) for value in thermal), axis=1),
    }
    return {
        name: MaskedProcessTarget(value, _defined(value)) for name, value in values.items()
    }


@dataclass(frozen=True)
class ProcessTrainingStatistics:
    candidate_contract_sha256: str
    dataset_id: str
    teacher_git_head: str
    markov_contract_sha256: str
    typed_contract_sha256: str
    dataset_manifest_sha256: str
    base_statistics_json_sha256: str
    base_statistics_npz_sha256: str
    selection: Mapping[str, str]
    partition_index: int
    partition_count: int
    sample_count: int
    source_shard_count: int
    source_inventory_sha256: str
    schema: NeuralOutputSchema
    field_labels: Mapping[str, str]
    inputs: Mapping[str, MomentStatistics]
    outputs: Mapping[str, MomentStatistics]
    loss_scales: Mapping[str, np.ndarray]

    def input_normalization(self) -> NeuralInputNormalization:
        return NeuralInputNormalization(
            continuous_mean=self.inputs["continuous"].mean,
            continuous_scale=self.inputs["continuous"].scale,
            forcing_mean=self.inputs["forcing"].mean,
            forcing_scale=self.inputs["forcing"].scale,
            static_mean=self.inputs["static"].mean,
            static_scale=self.inputs["static"].scale,
            annual_mean=self.inputs["annual"].mean,
            annual_scale=self.inputs["annual"].scale,
            water_scale=self.inputs["water"].scale,
            carbon_scale=self.inputs["carbon"].scale,
            thermal_scale=self.inputs["thermal"].scale,
        )

    def output_normalization(self) -> NeuralOutputNormalization:
        return NeuralOutputNormalization(
            mean={name: value.mean for name, value in self.outputs.items()},
            scale={name: value.scale for name, value in self.outputs.items()},
        )


def _source_inventory(references: Sequence[Any]) -> tuple[int, str]:
    entries = [
        {
            "landpoint_id": reference.parent.landpoint_id,
            "year": reference.parent.year,
            "base_sha256": reference.parent.sha256,
            "typed_sha256": reference.sha256,
            "day_count": reference.day_count,
        }
        for reference in references
    ]
    return len(entries), _canonical_sha256(entries)


def _base_input_statistics(statistics: TrainingStatistics) -> dict[str, MomentStatistics]:
    mapping = {
        "continuous": "state",
        "forcing": "forcing_native",
        "static": "landpoint_static",
        "annual": "annual_conditions",
    }
    result = {}
    for output_name, source_name in mapping.items():
        source = statistics.arrays[source_name]
        result[output_name] = MomentStatistics(
            count=np.asarray(source.count),
            mean=np.asarray(source.mean),
            m2=np.asarray(source.variance) * np.asarray(source.count),
            variance=np.asarray(source.variance),
            scale=np.asarray(source.scale),
        )
    return result


def _validate_fit_identity(
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    base_statistics: TrainingStatistics,
    candidate_contract: FirstCandidateContract,
    *,
    base_statistics_json_sha256: str,
    base_statistics_npz_sha256: str,
) -> None:
    data = candidate_contract.payload["data"]
    observed = {
        "dataset_manifest_sha256": index.parent_manifest_sha256,
        "markov_contract_sha256": index.parent_index.contract_sha256,
        "typed_contract_sha256": index.contract.sha256,
        "training_statistics_json_sha256": base_statistics_json_sha256,
        "training_statistics_npz_sha256": base_statistics_npz_sha256,
    }
    drift = {
        name: (value, data[name])
        for name, value in observed.items()
        if value != data[name]
    }
    if drift:
        raise ValueError(f"process statistics data identity drift: {drift}")
    if adapter.contract.sha256 != index.parent_index.contract_sha256:
        raise ValueError("process adapter Markov contract drift")
    if adapter.typed_contract.sha256 != index.contract.sha256:
        raise ValueError("process adapter typed contract drift")
    if (
        base_statistics.dataset_id != index.parent_index.dataset_id
        or base_statistics.teacher_git_head != index.parent_index.teacher_git_head
        or base_statistics.contract_sha256 != index.parent_index.contract_sha256
    ):
        raise ValueError("base training statistics dataset identity drift")


def fit_process_training_statistics(
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    base_statistics: TrainingStatistics,
    candidate_contract: FirstCandidateContract,
    *,
    base_statistics_json_sha256: str,
    base_statistics_npz_sha256: str,
    partition_index: int = 0,
    partition_count: int = 1,
    chunk_rows: int = 64,
) -> ProcessTrainingStatistics:
    """Fit one mergeable partition of train/train-only process statistics."""

    if partition_count < 1 or not 0 <= partition_index < partition_count:
        raise ValueError("invalid process statistics partition")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    _validate_fit_identity(
        index,
        adapter,
        base_statistics,
        candidate_contract,
        base_statistics_json_sha256=base_statistics_json_sha256,
        base_statistics_npz_sha256=base_statistics_npz_sha256,
    )
    all_references = index.select(spatial_split="train", temporal_split="train")
    expected_full_samples = sum(int(reference.day_count) for reference in all_references)
    if base_statistics.sample_count != expected_full_samples:
        raise ValueError("accepted base statistics sample count does not match train/train v6")
    references = all_references[partition_index::partition_count]
    if not references:
        raise ValueError("process statistics partition has no train/train shards")

    output_moments: dict[str, MaskedMoments] | None = None
    domain_moments: dict[str, MaskedMoments] | None = None
    schema: NeuralOutputSchema | None = None
    sample_count = 0
    axis_layout = build_process_axis_layout(adapter.input_assembler)
    for reference in references:
        from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

        shard = load_markov_shard(reference.parent.path, contract=adapter.contract)
        typed_values, typed_defined, day_index = read_typed_sidecar_shard(
            reference.path,
            contract=adapter.typed_contract,
            expected_days=reference.day_count,
        )
        if not np.array_equal(day_index, shard.day_index):
            raise ValueError("coherent process-statistics day index drift")
        try:
            targets = vectorized_process_targets(
                shard,
                typed_values,
                typed_defined,
                adapter,
            )
        except ValueError as error:
            raise ValueError(
                "process target construction failed for "
                f"{reference.parent.landpoint_id}:{reference.parent.year}"
            ) from error
        domains = vectorized_semantic_state(shard, adapter)
        if output_moments is None:
            template = {
                name: MaskedProcessTarget(target.value[0], target.defined[0])
                for name, target in targets.items()
            }
            typed_axes = {
                field.path: field.axes[1:] for field in adapter.typed_contract.fields
            }
            schema = build_neural_output_schema(
                template,
                field_labels=adapter.field_labels(),
                typed_axes=typed_axes,
                active_pft_indices=adapter.active_pft_indices,
            )
            output_moments = {
                field.name: MaskedMoments(
                    field.shape,
                    pft_axis=field.pft_axis,
                    pft_indices=(
                        adapter.active_pft_indices
                        if field.full_pft_axis
                        else tuple(range(field.shape[field.pft_axis]))
                        if field.pft_axis is not None
                        else ()
                    ),
                )
                for field in schema.fields
            }
            domain_moments = {
                name: MaskedMoments(target.value.shape[1:])
                for name, target in domains.items()
            }
        assert output_moments is not None and domain_moments is not None
        for name, target in targets.items():
            output_moments[name].update(target.value, target.defined, chunk_rows=chunk_rows)
        for name, target in domains.items():
            domain_moments[name].update(target.value, target.defined, chunk_rows=chunk_rows)
        sample_count += shard.days

    assert schema is not None and output_moments is not None and domain_moments is not None
    outputs = {
        name: value.finalize(minimum_scale=OUTPUT_MINIMUM_SCALE)
        for name, value in output_moments.items()
    }
    domain_layout = {
        layout.name: layout.element_pft_index for layout in axis_layout.domains
    }
    inputs = _base_input_statistics(base_statistics)
    for name, value in domain_moments.items():
        inputs[name] = _share_semantic_pft_moments(
            value.finalize(minimum_scale=SEMANTIC_INPUT_MINIMUM_SCALE),
            domain_layout[name],
            minimum_scale=SEMANTIC_INPUT_MINIMUM_SCALE,
        )
    constraints = {field.name: field.constraint for field in schema.fields}
    loss_scales = {
        name: (
            np.ones(value.mean.shape, dtype=np.float64)
            if constraints[name] in {"unit_interval", "signed_unit"}
            else value.scale.copy()
        )
        for name, value in outputs.items()
    }
    source_count, source_hash = _source_inventory(references)
    return ProcessTrainingStatistics(
        candidate_contract_sha256=candidate_contract.sha256,
        dataset_id=index.parent_index.dataset_id,
        teacher_git_head=index.parent_index.teacher_git_head,
        markov_contract_sha256=index.parent_index.contract_sha256,
        typed_contract_sha256=index.contract.sha256,
        dataset_manifest_sha256=index.parent_manifest_sha256,
        base_statistics_json_sha256=base_statistics_json_sha256,
        base_statistics_npz_sha256=base_statistics_npz_sha256,
        selection={"spatial_split": "train", "temporal_split": "train"},
        partition_index=partition_index,
        partition_count=partition_count,
        sample_count=sample_count,
        source_shard_count=source_count,
        source_inventory_sha256=source_hash,
        schema=schema,
        field_labels=adapter.field_labels(),
        inputs=inputs,
        outputs=outputs,
        loss_scales=loss_scales,
    )


def _moment_arrays(prefix: str, values: Mapping[str, MomentStatistics]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}__{member}": np.asarray(getattr(value, member))
        for name, value in values.items()
        for member in ("count", "mean", "m2", "variance", "scale")
    }


def write_process_training_statistics(
    statistics: ProcessTrainingStatistics,
    path: str | Path,
) -> Path:
    """Write a self-describing, hash-linked JSON/NPZ statistics artifact."""

    metadata_path = Path(path)
    if metadata_path.suffix != ".json":
        metadata_path = metadata_path.with_suffix(".json")
    arrays_path = metadata_path.with_suffix(".npz")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _moment_arrays("input", statistics.inputs)
    arrays.update(_moment_arrays("output", statistics.outputs))
    arrays.update(
        {
            f"loss_scale__{name}": np.asarray(value)
            for name, value in statistics.loss_scales.items()
        }
    )
    np.savez_compressed(arrays_path, **arrays)
    fields_by_name = {field.name: field for field in statistics.schema.fields}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_contract_sha256": statistics.candidate_contract_sha256,
        "dataset": {
            "dataset_id": statistics.dataset_id,
            "teacher_git_head": statistics.teacher_git_head,
            "dataset_manifest_sha256": statistics.dataset_manifest_sha256,
            "markov_contract_sha256": statistics.markov_contract_sha256,
            "typed_contract_sha256": statistics.typed_contract_sha256,
            "base_statistics_json_sha256": statistics.base_statistics_json_sha256,
            "base_statistics_npz_sha256": statistics.base_statistics_npz_sha256,
        },
        "selection": dict(statistics.selection),
        "partition": {
            "index": statistics.partition_index,
            "count": statistics.partition_count,
        },
        "sample_count": statistics.sample_count,
        "source_shard_count": statistics.source_shard_count,
        "source_inventory_sha256": statistics.source_inventory_sha256,
        "statistics_npz": arrays_path.name,
        "statistics_npz_sha256": _sha256_file(arrays_path),
        "policies": {
            "validity": "explicit_mask_and_finite_abs_lt_5e19",
            "population_variance": True,
            "output_minimum_scale": OUTPUT_MINIMUM_SCALE,
            "semantic_input_minimum_scale": SEMANTIC_INPUT_MINIMUM_SCALE,
            "pft_moments": "shared_across_active_slots",
            "bounded_loss_scale": "unit_range",
            "semantic_domain_centering": "zero",
        },
        "inputs": {
            name: {
                "feature_shape": list(value.mean.shape),
                "never_defined_columns": int(np.count_nonzero(value.count == 0)),
            }
            for name, value in statistics.inputs.items()
        },
        "outputs": {
            name: {
                "feature_shape": list(value.mean.shape),
                "label": statistics.field_labels[name],
                "constraint": fields_by_name[name].constraint,
                "pft_axis": fields_by_name[name].pft_axis,
                "full_pft_axis": fields_by_name[name].full_pft_axis,
                "never_defined_columns": int(np.count_nonzero(value.count == 0)),
                "loss_scale": (
                    "unit_range"
                    if fields_by_name[name].constraint in {"unit_interval", "signed_unit"}
                    else "train_standard_deviation"
                ),
            }
            for name, value in statistics.outputs.items()
        },
        "labels": list(statistics.schema.labels),
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def _load_moments(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    prefix: str,
) -> tuple[dict[str, MomentStatistics], set[str]]:
    result = {}
    consumed = set()
    for name, field in metadata.items():
        shape = tuple(int(value) for value in field["feature_shape"])
        members = {}
        for member in ("count", "mean", "m2", "variance", "scale"):
            key = f"{prefix}__{name}__{member}"
            if key not in arrays:
                raise ValueError(f"process statistics array is missing: {key}")
            value = np.asarray(arrays[key])
            if value.shape != shape:
                raise ValueError(f"process statistics shape drift: {key}")
            members[member] = value
            consumed.add(key)
        if members["count"].dtype != np.dtype(np.uint64):
            raise ValueError(f"process statistics count dtype drift: {name}")
        if any(members[key].dtype != np.dtype(np.float64) for key in members if key != "count"):
            raise ValueError(f"process statistics float dtype drift: {name}")
        expected_variance = np.zeros(shape, dtype=np.float64)
        np.divide(
            members["m2"],
            members["count"],
            out=expected_variance,
            where=members["count"] > 0,
        )
        if not np.allclose(
            expected_variance,
            members["variance"],
            rtol=1.0e-15,
            atol=0.0,
        ):
            raise ValueError(f"process statistics variance is inconsistent: {name}")
        if (
            np.any(~np.isfinite(members["mean"]))
            or np.any(~np.isfinite(members["m2"]))
            or np.any(members["m2"] < 0.0)
            or np.any(~np.isfinite(members["scale"]))
            or np.any(members["scale"] <= 0.0)
        ):
            raise ValueError(f"process statistics contain invalid moments: {name}")
        result[name] = MomentStatistics(**members)
    return result, consumed


def _validate_scale_policy(
    name: str,
    statistics: MomentStatistics,
    *,
    minimum_scale: float,
) -> None:
    standard_deviation = np.sqrt(np.maximum(statistics.variance, 0.0))
    expected = np.where(
        (statistics.count > 1)
        & np.isfinite(standard_deviation)
        & (standard_deviation > minimum_scale),
        standard_deviation,
        1.0,
    )
    if not np.array_equal(statistics.scale, expected):
        raise ValueError(f"process statistics scale policy drift: {name}")


def _validate_pft_axis_shared(
    name: str,
    statistics: MomentStatistics,
    *,
    axis: int,
    indices: Sequence[int],
) -> None:
    indices = tuple(int(value) for value in indices)
    if not indices:
        raise ValueError(f"PFT-shared field has no active slots: {name}")
    for member in ("count", "mean", "m2", "variance", "scale"):
        moved = np.moveaxis(np.asarray(getattr(statistics, member)), axis, 0)
        reference = moved[indices[0]]
        if any(not _same_array(reference, moved[index]) for index in indices[1:]):
            raise ValueError(f"PFT-shared process moments drift: {name}.{member}")


def _validate_semantic_pft_shared(
    name: str,
    statistics: MomentStatistics,
    element_pft_index: np.ndarray,
) -> None:
    pft = np.asarray(element_pft_index, dtype=np.int32).reshape(-1)
    slots = tuple(int(value) for value in np.unique(pft) if value >= 0)
    groups = tuple(np.flatnonzero(pft == slot) for slot in slots)
    if not groups:
        return
    if len({group.size for group in groups}) != 1:
        raise ValueError(f"semantic PFT layout drift: {name}")
    for coordinates in zip(*groups, strict=True):
        for member in ("count", "mean", "m2", "variance", "scale"):
            values = np.asarray(getattr(statistics, member)).reshape(-1)
            reference = values[coordinates[0]]
            if any(not _same_array(reference, values[index]) for index in coordinates[1:]):
                raise ValueError(f"semantic PFT-shared moments drift: {name}.{member}")


def load_process_training_statistics(
    path: str | Path,
    *,
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    candidate_contract: FirstCandidateContract,
    base_statistics: TrainingStatistics,
) -> ProcessTrainingStatistics:
    """Load the artifact and fail closed on hashes, identity, policy, and schema."""

    metadata_path = Path(path).resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported process training statistics schema")
    expected_policies = {
        "validity": "explicit_mask_and_finite_abs_lt_5e19",
        "population_variance": True,
        "output_minimum_scale": OUTPUT_MINIMUM_SCALE,
        "semantic_input_minimum_scale": SEMANTIC_INPUT_MINIMUM_SCALE,
        "pft_moments": "shared_across_active_slots",
        "bounded_loss_scale": "unit_range",
        "semantic_domain_centering": "zero",
    }
    if payload.get("policies") != expected_policies:
        raise ValueError("process training statistics policy drift")
    if payload.get("selection") != {"spatial_split": "train", "temporal_split": "train"}:
        raise ValueError("process training statistics are not train/train-only")
    dataset = payload.get("dataset", {})
    expected_dataset = {
        "dataset_id": index.parent_index.dataset_id,
        "teacher_git_head": index.parent_index.teacher_git_head,
        "dataset_manifest_sha256": index.parent_manifest_sha256,
        "markov_contract_sha256": index.parent_index.contract_sha256,
        "typed_contract_sha256": index.contract.sha256,
        "base_statistics_json_sha256": candidate_contract.payload["data"][
            "training_statistics_json_sha256"
        ],
        "base_statistics_npz_sha256": candidate_contract.payload["data"][
            "training_statistics_npz_sha256"
        ],
    }
    if dataset != expected_dataset:
        raise ValueError("process training statistics dataset identity drift")
    if payload.get("candidate_contract_sha256") != candidate_contract.sha256:
        raise ValueError("process training statistics candidate contract drift")
    partition = payload.get("partition", {})
    partition_index = int(partition.get("index", -1))
    partition_count = int(partition.get("count", 0))
    if partition_count < 1 or not 0 <= partition_index < partition_count:
        raise ValueError("process training statistics partition metadata is invalid")
    all_references = index.select(spatial_split="train", temporal_split="train")
    references = all_references[partition_index::partition_count]
    source_count, source_hash = _source_inventory(references)
    expected_samples = sum(int(reference.day_count) for reference in references)
    if (
        int(payload.get("sample_count", -1)) != expected_samples
        or int(payload.get("source_shard_count", -1)) != source_count
        or payload.get("source_inventory_sha256") != source_hash
    ):
        raise ValueError("process training statistics source partition drift")
    arrays_path = metadata_path.parent / str(payload["statistics_npz"])
    if _sha256_file(arrays_path) != payload.get("statistics_npz_sha256"):
        raise ValueError("process training statistics NPZ hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    inputs, consumed = _load_moments(arrays, payload.get("inputs", {}), prefix="input")
    outputs, output_consumed = _load_moments(
        arrays, payload.get("outputs", {}), prefix="output"
    )
    consumed |= output_consumed
    if tuple(inputs) != INPUT_NAMES:
        raise ValueError("process training input statistics registry drift")
    if set(outputs) != set(adapter.field_labels()):
        raise ValueError("process training output statistics registry drift")
    for name, statistics in inputs.items():
        _validate_scale_policy(
            f"input.{name}",
            statistics,
            minimum_scale=(
                SEMANTIC_INPUT_MINIMUM_SCALE
                if name in {"water", "carbon", "thermal"}
                else OUTPUT_MINIMUM_SCALE
            ),
        )
    for name, statistics in outputs.items():
        _validate_scale_policy(
            f"output.{name}",
            statistics,
            minimum_scale=OUTPUT_MINIMUM_SCALE,
        )
    expected_base_inputs = _base_input_statistics(base_statistics)
    for name in ("continuous", "forcing", "static", "annual"):
        if not all(
            _same_array(getattr(inputs[name], member), getattr(expected_base_inputs[name], member))
            for member in ("count", "mean", "m2", "variance", "scale")
        ):
            raise ValueError(f"process statistics do not reproduce accepted base input: {name}")

    fields = []
    loss_scales = {}
    for name, metadata in payload["outputs"].items():
        pft_axis = metadata["pft_axis"]
        fields.append(
            OutputFieldSpec(
                name=name,
                shape=tuple(metadata["feature_shape"]),
                label_index=tuple(payload["labels"]).index(metadata["label"]),
                constraint=metadata["constraint"],
                pft_axis=None if pft_axis is None else int(pft_axis),
                full_pft_axis=bool(metadata["full_pft_axis"]),
            )
        )
        key = f"loss_scale__{name}"
        if key not in arrays or arrays[key].shape != outputs[name].mean.shape:
            raise ValueError(f"process loss scale shape drift: {name}")
        loss_scales[name] = np.asarray(arrays[key], dtype=np.float64)
        consumed.add(key)
        if metadata["constraint"] in {"unit_interval", "signed_unit"}:
            if not np.array_equal(loss_scales[name], np.ones_like(loss_scales[name])):
                raise ValueError(f"bounded process loss scale drift: {name}")
        elif not np.array_equal(loss_scales[name], outputs[name].scale):
            raise ValueError(f"unbounded process loss scale drift: {name}")
    if set(arrays) != consumed:
        raise ValueError("process training statistics NPZ contains undeclared arrays")
    schema = NeuralOutputSchema(
        fields=tuple(fields),
        labels=tuple(payload["labels"]),
        active_pft_indices=adapter.active_pft_indices,
    )
    for field in schema.fields:
        if field.pft_axis is None:
            continue
        _validate_pft_axis_shared(
            field.name,
            outputs[field.name],
            axis=field.pft_axis,
            indices=(
                adapter.active_pft_indices
                if field.full_pft_axis
                else tuple(range(field.shape[field.pft_axis]))
            ),
        )
    axis_layout = build_process_axis_layout(adapter.input_assembler)
    for layout in axis_layout.domains:
        _validate_semantic_pft_shared(
            layout.name,
            inputs[layout.name],
            layout.element_pft_index,
        )
    return ProcessTrainingStatistics(
        candidate_contract_sha256=candidate_contract.sha256,
        dataset_id=dataset["dataset_id"],
        teacher_git_head=dataset["teacher_git_head"],
        markov_contract_sha256=dataset["markov_contract_sha256"],
        typed_contract_sha256=dataset["typed_contract_sha256"],
        dataset_manifest_sha256=dataset["dataset_manifest_sha256"],
        base_statistics_json_sha256=dataset["base_statistics_json_sha256"],
        base_statistics_npz_sha256=dataset["base_statistics_npz_sha256"],
        selection=payload["selection"],
        partition_index=partition_index,
        partition_count=partition_count,
        sample_count=int(payload["sample_count"]),
        source_shard_count=int(payload["source_shard_count"]),
        source_inventory_sha256=str(payload["source_inventory_sha256"]),
        schema=schema,
        field_labels=adapter.field_labels(),
        inputs=inputs,
        outputs=outputs,
        loss_scales=loss_scales,
    )


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)


def _merge_finalized_moments(
    values: Sequence[MomentStatistics],
    *,
    minimum_scale: float,
) -> MomentStatistics:
    if not values:
        raise ValueError("cannot merge an empty moment sequence")
    accumulator = MaskedMoments.from_statistics(values[0])
    for value in values[1:]:
        accumulator.merge(MaskedMoments.from_statistics(value))
    return accumulator.finalize(minimum_scale=minimum_scale)


def merge_process_training_statistics(
    partials: Sequence[ProcessTrainingStatistics],
    *,
    index: TypedSidecarIndex,
    adapter: DailyProcessTrainingAdapter,
    candidate_contract: FirstCandidateContract,
) -> ProcessTrainingStatistics:
    """Merge an exact, non-overlapping set of deterministic fit partitions."""

    if not partials:
        raise ValueError("cannot merge zero process-statistics partitions")
    partition_counts = {partial.partition_count for partial in partials}
    if len(partition_counts) != 1:
        raise ValueError("process-statistics partition counts do not agree")
    partition_count = partition_counts.pop()
    by_index = {partial.partition_index: partial for partial in partials}
    if len(by_index) != len(partials) or set(by_index) != set(range(partition_count)):
        raise ValueError("process-statistics partitions are duplicated or incomplete")
    ordered = tuple(by_index[index_value] for index_value in range(partition_count))
    first = ordered[0]
    expected_identity = (
        candidate_contract.sha256,
        index.parent_index.dataset_id,
        index.parent_index.teacher_git_head,
        index.parent_index.contract_sha256,
        index.contract.sha256,
        index.parent_manifest_sha256,
        candidate_contract.payload["data"]["training_statistics_json_sha256"],
        candidate_contract.payload["data"]["training_statistics_npz_sha256"],
    )
    for partial in ordered:
        identity = (
            partial.candidate_contract_sha256,
            partial.dataset_id,
            partial.teacher_git_head,
            partial.markov_contract_sha256,
            partial.typed_contract_sha256,
            partial.dataset_manifest_sha256,
            partial.base_statistics_json_sha256,
            partial.base_statistics_npz_sha256,
        )
        if identity != expected_identity:
            raise ValueError("process-statistics merge identity drift")
        if partial.selection != {"spatial_split": "train", "temporal_split": "train"}:
            raise ValueError("process-statistics merge includes a non-training split")
        if partial.schema != first.schema or partial.field_labels != first.field_labels:
            raise ValueError("process-statistics merge schema drift")

    all_references = index.select(spatial_split="train", temporal_split="train")
    source_count, source_hash = _source_inventory(all_references)
    expected_samples = sum(int(reference.day_count) for reference in all_references)
    if sum(partial.sample_count for partial in ordered) != expected_samples:
        raise ValueError("process-statistics merge sample count drift")
    if sum(partial.source_shard_count for partial in ordered) != source_count:
        raise ValueError("process-statistics merge shard count drift")

    inputs = {}
    for name in INPUT_NAMES:
        values = tuple(partial.inputs[name] for partial in ordered)
        if name in {"continuous", "forcing", "static", "annual"}:
            reference = values[0]
            if any(
                not all(
                    _same_array(getattr(reference, member), getattr(value, member))
                    for member in ("count", "mean", "m2", "variance", "scale")
                )
                for value in values[1:]
            ):
                raise ValueError(f"accepted base input moments drift across partitions: {name}")
            inputs[name] = reference
        else:
            inputs[name] = _merge_finalized_moments(
                values,
                minimum_scale=SEMANTIC_INPUT_MINIMUM_SCALE,
            )
    outputs = {
        name: _merge_finalized_moments(
            tuple(partial.outputs[name] for partial in ordered),
            minimum_scale=OUTPUT_MINIMUM_SCALE,
        )
        for name in first.outputs
    }
    constraints = {field.name: field.constraint for field in first.schema.fields}
    loss_scales = {
        name: (
            np.ones(value.mean.shape, dtype=np.float64)
            if constraints[name] in {"unit_interval", "signed_unit"}
            else value.scale.copy()
        )
        for name, value in outputs.items()
    }
    return ProcessTrainingStatistics(
        candidate_contract_sha256=candidate_contract.sha256,
        dataset_id=index.parent_index.dataset_id,
        teacher_git_head=index.parent_index.teacher_git_head,
        markov_contract_sha256=index.parent_index.contract_sha256,
        typed_contract_sha256=index.contract.sha256,
        dataset_manifest_sha256=index.parent_manifest_sha256,
        base_statistics_json_sha256=expected_identity[-2],
        base_statistics_npz_sha256=expected_identity[-1],
        selection={"spatial_split": "train", "temporal_split": "train"},
        partition_index=0,
        partition_count=1,
        sample_count=expected_samples,
        source_shard_count=source_count,
        source_inventory_sha256=source_hash,
        schema=first.schema,
        field_labels=adapter.field_labels(),
        inputs=inputs,
        outputs=outputs,
        loss_scales=loss_scales,
    )
