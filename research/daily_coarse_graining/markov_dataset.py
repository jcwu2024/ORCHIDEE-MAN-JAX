"""Verified dataset reader for Daily Markov Teacher shards."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

LEGACY_DATASET_SCHEMA_VERSION = "daily_teacher_dataset_manifest_v3"
DATASET_SCHEMA_VERSION = "daily_teacher_dataset_manifest_v4"
SUPPORTED_DATASET_SCHEMA_VERSIONS = frozenset(
    {LEGACY_DATASET_SCHEMA_VERSION, DATASET_SCHEMA_VERSION}
)
STATISTICS_SCHEMA_VERSION = "daily_teacher_training_statistics_v2"
ORCHIDEE_UNDEFINED_MAGNITUDE = 1.0e20
ORCHIDEE_UNDEFINED_THRESHOLD = 0.5 * ORCHIDEE_UNDEFINED_MAGNITUDE
SPLITS = frozenset({"train", "validation", "test"})
CONTINUOUS_ARRAY_NAMES = (
    "state",
    "fast_day_target",
    "forcing_native",
    "parameters",
    "landpoint_static",
    "annual_conditions",
    "next_state",
    "state_delta",
    "diagnostics",
)


def defined_numeric_mask(values: np.ndarray) -> np.ndarray:
    """Return values that are finite and are not ORCHIDEE ±1e20 sentinels."""

    values = np.asarray(values)
    return np.isfinite(values) & (np.abs(values) < ORCHIDEE_UNDEFINED_THRESHOLD)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MarkovShardRef:
    landpoint_id: str
    year: int
    spatial_split: str
    temporal_split: str
    path: Path
    sha256: str
    contract_sha256: str


@dataclass(frozen=True)
class MarkovDatasetIndex:
    dataset_id: str
    teacher_git_head: str
    contract_sha256: str
    shards: tuple[MarkovShardRef, ...]

    def select(
        self,
        *,
        spatial_split: str | None = None,
        temporal_split: str | None = None,
    ) -> tuple[MarkovShardRef, ...]:
        for name, value in (
            ("spatial_split", spatial_split),
            ("temporal_split", temporal_split),
        ):
            if value is not None and value not in SPLITS:
                raise ValueError(f"unknown {name} {value!r}")
        return tuple(
            shard
            for shard in self.shards
            if (spatial_split is None or shard.spatial_split == spatial_split)
            and (temporal_split is None or shard.temporal_split == temporal_split)
        )

    def samples(
        self,
        *,
        spatial_split: str | None = None,
        temporal_split: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        for reference in self.select(
            spatial_split=spatial_split,
            temporal_split=temporal_split,
        ):
            shard = load_markov_shard(reference.path)
            for index in range(shard.days):
                yield shard.sample(index) | {
                    "sample_landpoint_id": reference.landpoint_id,
                    "sample_spatial_split": reference.spatial_split,
                    "sample_temporal_split": reference.temporal_split,
                }

    def windows(
        self,
        horizon: int,
        *,
        stride: int = 1,
        spatial_split: str | None = None,
        temporal_split: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield contiguous windows without crossing a shard or restart boundary."""

        if horizon < 1 or stride < 1:
            raise ValueError("window horizon and stride must be positive")
        for reference in self.select(
            spatial_split=spatial_split,
            temporal_split=temporal_split,
        ):
            shard = load_markov_shard(reference.path)
            for start in range(0, shard.days - horizon + 1, stride):
                yield shard.window(start, horizon) | {
                    "sample_landpoint_id": reference.landpoint_id,
                    "sample_year": reference.year,
                    "sample_spatial_split": reference.spatial_split,
                    "sample_temporal_split": reference.temporal_split,
                }


@dataclass(frozen=True)
class FiniteColumnStatistics:
    count: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class TrainingStatistics:
    dataset_id: str
    teacher_git_head: str
    contract_sha256: str
    sample_count: int
    source_shards: tuple[str, ...]
    observations_per_column: Mapping[str, int]
    arrays: Mapping[str, FiniteColumnStatistics]


class _FiniteMoments:
    def __init__(self, feature_shape: tuple[int, ...]):
        self.feature_shape = feature_shape
        self.width = int(np.prod(feature_shape, dtype=np.int64))
        self.count = np.zeros(self.width, dtype=np.uint64)
        self.mean = np.zeros(self.width, dtype=np.float64)
        self.m2 = np.zeros(self.width, dtype=np.float64)

    def _merge(
        self,
        batch_count: np.ndarray,
        batch_mean: np.ndarray,
        batch_m2: np.ndarray,
    ) -> None:
        total = self.count + batch_count
        active = batch_count > 0
        delta = batch_mean - self.mean
        weight = np.zeros_like(self.mean)
        np.divide(batch_count, total, out=weight, where=active)
        cross_weight = np.zeros_like(self.mean)
        np.divide(
            self.count * batch_count,
            total,
            out=cross_weight,
            where=active,
        )
        self.mean = np.where(active, self.mean + delta * weight, self.mean)
        self.m2 = np.where(
            active,
            self.m2 + batch_m2 + delta * delta * cross_weight,
            self.m2,
        )
        self.count = total

    def update(self, values: np.ndarray, *, chunk_rows: int) -> None:
        rows = np.asarray(values, dtype=np.float64).reshape((-1, self.width))
        for start in range(0, rows.shape[0], chunk_rows):
            chunk = rows[start : start + chunk_rows]
            finite = defined_numeric_mask(chunk)
            safe = np.where(finite, chunk, 0.0)
            batch_count = finite.sum(axis=0, dtype=np.uint64)
            batch_mean = np.zeros(self.width, dtype=np.float64)
            np.divide(
                safe.sum(axis=0),
                batch_count,
                out=batch_mean,
                where=batch_count > 0,
            )
            centered = np.where(finite, chunk - batch_mean, 0.0)
            self._merge(batch_count, batch_mean, np.sum(centered * centered, axis=0))

    def update_constant(self, value: np.ndarray, *, repetitions: int) -> None:
        row = np.asarray(value, dtype=np.float64).reshape(self.width)
        finite = defined_numeric_mask(row)
        count = finite.astype(np.uint64) * np.uint64(repetitions)
        self._merge(count, np.where(finite, row, 0.0), np.zeros(self.width))

    def update_difference(
        self,
        minuend: np.ndarray,
        subtrahend: np.ndarray,
        *,
        chunk_rows: int,
    ) -> None:
        end_rows = np.asarray(minuend, dtype=np.float64).reshape((-1, self.width))
        start_rows = np.asarray(subtrahend, dtype=np.float64).reshape((-1, self.width))
        if end_rows.shape != start_rows.shape:
            raise ValueError("difference operands must have identical shapes")
        for start in range(0, end_rows.shape[0], chunk_rows):
            stop = start + chunk_rows
            end_chunk = end_rows[start:stop]
            start_chunk = start_rows[start:stop]
            valid = defined_numeric_mask(end_chunk) & defined_numeric_mask(
                start_chunk
            )
            difference = np.where(valid, end_chunk - start_chunk, np.nan)
            self.update(difference, chunk_rows=chunk_rows)

    def finalize(self, *, minimum_scale: float) -> FiniteColumnStatistics:
        variance = np.zeros(self.width, dtype=np.float64)
        np.divide(self.m2, self.count, out=variance, where=self.count > 0)
        variance = np.maximum(variance, 0.0)
        standard_deviation = np.sqrt(variance)
        scale = np.where(
            (self.count > 1) & (standard_deviation > minimum_scale),
            standard_deviation,
            1.0,
        )
        shape = self.feature_shape
        return FiniteColumnStatistics(
            count=self.count.reshape(shape),
            mean=self.mean.reshape(shape),
            variance=variance.reshape(shape),
            scale=scale.reshape(shape),
        )


def load_dataset_index(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
    verify_files: bool = True,
) -> MarkovDatasetIndex:
    """Load dataset identity and shard references.

    ``verify_files=False`` is reserved for post-acceptance execution phases
    that already bind a hash-verified manifest and reopen every selected shard.
    It avoids repeating tens of thousands of metadata-only filesystem probes
    before each matched training arm.
    """

    manifest_path = Path(manifest_path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version")
    if schema_version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
        raise ValueError(
            "dataset schema must be one of "
            f"{sorted(SUPPORTED_DATASET_SCHEMA_VERSIONS)!r}"
        )
    root = manifest_path.parent
    contract_sha256 = str(raw["markov_contract_sha256"])
    contract_metadata = raw.get("markov_contract")
    if contract_metadata is not None:
        encoded = json.dumps(
            contract_metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != contract_sha256:
            raise ValueError("dataset Markov contract metadata hash mismatch")
    references = []
    seen = set()
    for item in raw.get("shards", []):
        key = (str(item["landpoint_id"]), int(item["year"]))
        if key in seen:
            raise ValueError(f"duplicate dataset shard {key[0]}:{key[1]}")
        seen.add(key)
        spatial_split = str(item["spatial_split"])
        temporal_split = str(item["temporal_split"])
        if spatial_split not in SPLITS or temporal_split not in SPLITS:
            raise ValueError(f"invalid split for {key[0]}:{key[1]}")
        item_contract = item.get("markov_contract_sha256")
        if (
            schema_version == LEGACY_DATASET_SCHEMA_VERSION
            and item_contract != contract_sha256
        ):
            raise ValueError(f"contract drift for {key[0]}:{key[1]}")
        if item_contract not in {None, contract_sha256}:
            raise ValueError(f"contract drift for {key[0]}:{key[1]}")
        path = (root / item["shard"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"shard path escapes dataset root: {path}") from error
        if verify_files and not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = str(item["shard_sha256"])
        if verify_hashes and _sha256_file(path) != expected_hash:
            raise ValueError(f"shard hash mismatch for {key[0]}:{key[1]}")
        references.append(
            MarkovShardRef(
                landpoint_id=key[0],
                year=key[1],
                spatial_split=spatial_split,
                temporal_split=temporal_split,
                path=path,
                sha256=expected_hash,
                contract_sha256=contract_sha256,
            )
        )
    if not references:
        raise ValueError("dataset manifest contains no shards")
    return MarkovDatasetIndex(
        dataset_id=str(raw["dataset_id"]),
        teacher_git_head=str(raw["teacher_git_head"]),
        contract_sha256=contract_sha256,
        shards=tuple(references),
    )


def fit_training_statistics(
    index: MarkovDatasetIndex,
    *,
    chunk_rows: int = 64,
    minimum_scale: float = 1.0e-12,
) -> TrainingStatistics:
    """Fit finite-only column statistics from the frozen train/train split."""

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive")
    references = index.select(spatial_split="train", temporal_split="train")
    if not references:
        raise ValueError("dataset has no train/train shards")

    accumulators: dict[str, _FiniteMoments] | None = None
    observations = {name: 0 for name in CONTINUOUS_ARRAY_NAMES}
    sample_count = 0
    for reference in references:
        shard = load_markov_shard(reference.path)
        if accumulators is None:
            state_shape = (shard.state_trajectory.shape[1],)
            accumulators = {
                "state": _FiniteMoments(state_shape),
                "fast_day_target": _FiniteMoments(
                    (shard.fast_day_target.shape[-1],)
                ),
                "forcing_native": _FiniteMoments((shard.forcing_native.shape[-1],)),
                "parameters": _FiniteMoments(shard.parameters.shape),
                "landpoint_static": _FiniteMoments(shard.landpoint_static.shape),
                "annual_conditions": _FiniteMoments(shard.annual_conditions.shape),
                "next_state": _FiniteMoments(state_shape),
                "state_delta": _FiniteMoments(state_shape),
                "diagnostics": _FiniteMoments((shard.diagnostics.shape[-1],)),
            }
        observed_shapes = {
            "state": (shard.state_trajectory.shape[1],),
            "fast_day_target": (shard.fast_day_target.shape[-1],),
            "forcing_native": (shard.forcing_native.shape[-1],),
            "parameters": shard.parameters.shape,
            "landpoint_static": shard.landpoint_static.shape,
            "annual_conditions": shard.annual_conditions.shape,
            "next_state": (shard.state_trajectory.shape[1],),
            "state_delta": (shard.state_trajectory.shape[1],),
            "diagnostics": (shard.diagnostics.shape[-1],),
        }
        drift = {
            name: (shape, accumulators[name].feature_shape)
            for name, shape in observed_shapes.items()
            if shape != accumulators[name].feature_shape
        }
        if drift:
            raise ValueError(f"training shard feature-shape drift: {drift}")

        days = shard.days
        state = shard.state_trajectory[:-1]
        next_state = shard.state_trajectory[1:]
        accumulators["state"].update(state, chunk_rows=chunk_rows)
        accumulators["fast_day_target"].update(
            shard.fast_day_target, chunk_rows=chunk_rows
        )
        accumulators["next_state"].update(next_state, chunk_rows=chunk_rows)
        accumulators["state_delta"].update_difference(next_state, state, chunk_rows=chunk_rows)
        accumulators["forcing_native"].update(shard.forcing_native, chunk_rows=chunk_rows)
        accumulators["parameters"].update_constant(shard.parameters, repetitions=days)
        accumulators["landpoint_static"].update_constant(shard.landpoint_static, repetitions=days)
        accumulators["annual_conditions"].update_constant(shard.annual_conditions, repetitions=days)
        accumulators["diagnostics"].update(shard.diagnostics, chunk_rows=chunk_rows)
        for name in observations:
            observations[name] += days * (5 if name == "forcing_native" else 1)
        sample_count += days

    assert accumulators is not None
    return TrainingStatistics(
        dataset_id=index.dataset_id,
        teacher_git_head=index.teacher_git_head,
        contract_sha256=index.contract_sha256,
        sample_count=sample_count,
        source_shards=tuple(f"{ref.landpoint_id}:{ref.year}:{ref.sha256}" for ref in references),
        observations_per_column=observations,
        arrays={name: accumulator.finalize(minimum_scale=minimum_scale) for name, accumulator in accumulators.items()},
    )


def write_training_statistics(statistics: TrainingStatistics, path: str | Path) -> Path:
    """Write a hash-linked NPZ statistics asset and return its JSON metadata path."""

    path = Path(path)
    metadata_path = path if path.suffix == ".json" else path.with_suffix(".json")
    arrays_path = metadata_path.with_suffix(".npz")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"{name}__{field}": np.asarray(getattr(value, field))
        for name, value in statistics.arrays.items()
        for field in ("count", "mean", "variance", "scale")
    }
    np.savez_compressed(arrays_path, **arrays)
    summary = {
        name: {
            "feature_shape": list(value.mean.shape),
            "observations_per_column": int(statistics.observations_per_column[name]),
            "never_finite_columns": int(np.count_nonzero(value.count == 0)),
            "single_finite_columns": int(np.count_nonzero(value.count == 1)),
            "conditionally_finite_columns": int(
                np.count_nonzero((value.count > 0) & (value.count < statistics.observations_per_column[name]))
            ),
        }
        for name, value in statistics.arrays.items()
    }
    payload = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "dataset_id": statistics.dataset_id,
        "teacher_git_head": statistics.teacher_git_head,
        "markov_contract_sha256": statistics.contract_sha256,
        "selection": {"spatial_split": "train", "temporal_split": "train"},
        "sample_count": statistics.sample_count,
        "source_shards": list(statistics.source_shards),
        "statistics_npz": arrays_path.name,
        "statistics_npz_sha256": _sha256_file(arrays_path),
        "validity_policy": {
            "finite": True,
            "excluded_absolute_value_gte": ORCHIDEE_UNDEFINED_THRESHOLD,
            "source_sentinel_magnitude": ORCHIDEE_UNDEFINED_MAGNITUDE,
        },
        "arrays": summary,
        "zero_or_one_finite_policy": {"mean": "finite value or zero", "scale": 1.0},
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def load_training_statistics(
    path: str | Path,
    *,
    index: MarkovDatasetIndex | None = None,
) -> TrainingStatistics:
    metadata_path = Path(path).resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != STATISTICS_SCHEMA_VERSION:
        raise ValueError(f"statistics schema must be {STATISTICS_SCHEMA_VERSION!r}")
    expected_validity = {
        "finite": True,
        "excluded_absolute_value_gte": ORCHIDEE_UNDEFINED_THRESHOLD,
        "source_sentinel_magnitude": ORCHIDEE_UNDEFINED_MAGNITUDE,
    }
    if payload.get("validity_policy") != expected_validity:
        raise ValueError("training statistics validity policy mismatch")
    arrays_path = metadata_path.parent / payload["statistics_npz"]
    if _sha256_file(arrays_path) != payload["statistics_npz_sha256"]:
        raise ValueError("training statistics array hash mismatch")
    if index is not None:
        identity = (
            payload["dataset_id"],
            payload["teacher_git_head"],
            payload["markov_contract_sha256"],
        )
        expected = (index.dataset_id, index.teacher_git_head, index.contract_sha256)
        if identity != expected:
            raise ValueError("training statistics dataset identity mismatch")
    with np.load(arrays_path, allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    result = {}
    observations = {}
    for name, metadata in payload["arrays"].items():
        result[name] = FiniteColumnStatistics(
            **{field: values[f"{name}__{field}"] for field in ("count", "mean", "variance", "scale")}
        )
        observations[name] = int(metadata["observations_per_column"])
    return TrainingStatistics(
        dataset_id=str(payload["dataset_id"]),
        teacher_git_head=str(payload["teacher_git_head"]),
        contract_sha256=str(payload["markov_contract_sha256"]),
        sample_count=int(payload["sample_count"]),
        source_shards=tuple(payload["source_shards"]),
        observations_per_column=observations,
        arrays=result,
    )


def normalize_finite(values: np.ndarray, statistics: FiniteColumnStatistics) -> tuple[np.ndarray, np.ndarray]:
    """Normalize defined entries and map NaN/ORCHIDEE sentinels to zero."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[-statistics.mean.ndim :] != statistics.mean.shape:
        raise ValueError(f"normalization shape mismatch: {values.shape} versus {statistics.mean.shape}")
    finite = defined_numeric_mask(values)
    normalized = np.zeros_like(values, dtype=np.float64)
    np.subtract(values, statistics.mean, out=normalized, where=finite)
    np.divide(normalized, statistics.scale, out=normalized, where=finite)
    return normalized, finite


def denormalize(values: np.ndarray, statistics: FiniteColumnStatistics) -> np.ndarray:
    """Restore normalized finite model values to their physical scale."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape[-statistics.mean.ndim :] != statistics.mean.shape:
        raise ValueError(
            f"denormalization shape mismatch: {values.shape} "
            f"versus {statistics.mean.shape}"
        )
    return values * statistics.scale + statistics.mean


def collate_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack numerical model inputs/targets without exposing identity as a feature."""

    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    array_names = (
        "state",
        "fast_day_target",
        "forcing_native",
        "parameters",
        "landpoint_static",
        "annual_conditions",
        "next_state",
        "diagnostics",
        "year",
        "day_index",
    )
    result = {name: np.stack([np.asarray(sample[name]) for sample in samples]) for name in array_names}
    result["state_delta"] = result["next_state"] - result["state"]
    for group in ("discrete_state", "next_discrete_state"):
        keys = tuple(samples[0][group])
        if any(tuple(sample[group]) != keys for sample in samples[1:]):
            raise ValueError(f"{group} schema drift across samples")
        result[group] = {key: np.stack([np.asarray(sample[group][key]) for sample in samples]) for key in keys}
    for name in CONTINUOUS_ARRAY_NAMES:
        result[f"{name}_finite"] = np.isfinite(result[name])
    return result


def collate_windows(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack equal-horizon trajectories without crossing shard boundaries."""

    if not windows:
        raise ValueError("cannot collate an empty window sequence")
    horizon = int(np.asarray(windows[0]["day_index"]).shape[0])
    if horizon < 1 or any(
        np.asarray(window["day_index"]).shape != (horizon,) for window in windows
    ):
        raise ValueError("collated Markov windows must have one fixed horizon")
    array_names = (
        "initial_state",
        "state_trajectory",
        "teacher_next_state",
        "teacher_fast_day_target",
        "forcing_native",
        "parameters",
        "landpoint_static",
        "annual_conditions",
        "year",
        "day_index",
    )
    result = {
        name: np.stack([np.asarray(window[name]) for window in windows])
        for name in array_names
    }
    for group in (
        "initial_discrete_state",
        "discrete_trajectory",
        "teacher_next_discrete_state",
    ):
        keys = tuple(windows[0][group])
        if any(tuple(window[group]) != keys for window in windows[1:]):
            raise ValueError(f"{group} schema drift across windows")
        result[group] = {
            key: np.stack([np.asarray(window[group][key]) for window in windows])
            for key in keys
        }
    return result


def batched(samples: Iterable[Mapping[str, Any]], batch_size: int) -> Iterator[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pending = []
    for sample in samples:
        pending.append(sample)
        if len(pending) == batch_size:
            yield collate_samples(pending)
            pending.clear()
    if pending:
        yield collate_samples(pending)


@dataclass(frozen=True)
class _PrefetchFailure:
    error: BaseException


def prefetched_batches(
    samples: Iterable[Mapping[str, Any]],
    batch_size: int,
    *,
    prefetch: int = 2,
) -> Iterator[dict[str, Any]]:
    """Collate in one producer thread while bounding queued batches."""

    if prefetch < 1:
        raise ValueError("prefetch must be positive")
    pending: queue.Queue[Any] = queue.Queue(maxsize=prefetch)
    stopped = threading.Event()
    sentinel = object()

    def put(value: Any) -> bool:
        while not stopped.is_set():
            try:
                pending.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for batch in batched(samples, batch_size):
                if not put(batch):
                    return
        except BaseException as error:
            put(_PrefetchFailure(error))
        finally:
            put(sentinel)

    producer = threading.Thread(
        target=produce,
        name="markov-dataset-prefetch",
        daemon=True,
    )
    producer.start()
    try:
        while True:
            value = pending.get()
            if value is sentinel:
                break
            if isinstance(value, _PrefetchFailure):
                raise value.error
            yield value
    finally:
        stopped.set()
        producer.join(timeout=1.0)
