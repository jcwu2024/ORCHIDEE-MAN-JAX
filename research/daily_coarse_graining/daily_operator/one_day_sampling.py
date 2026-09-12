"""Deterministic host-only sampling over an already prepared one-day pool.

An epoch here covers the supplied prepared one-day rows. It is deliberately
not described as an epoch over the complete v6 Teacher dataset.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeAlias

import numpy as np

ArtifactId: TypeAlias = int | str
EPOCH_SCOPE = "prepared_one_day_pool"


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    result = int(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _artifact_id(value: Any) -> ArtifactId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("artifact_id must be an integer or string")
    if isinstance(value, str) and not value:
        raise ValueError("artifact_id must not be empty")
    return value


def _items(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error


def _optional_string(record: Mapping[str, Any], name: str) -> str | None:
    value = record.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string when supplied")
    return value


def _artifact_order_key(value: ArtifactId) -> tuple[int, int | str]:
    if isinstance(value, int):
        return (0, value)
    return (1, value)


@dataclass(frozen=True, slots=True)
class PreparedRowKey:
    """Exact occurrence of a row inside one compressed prepared artifact."""

    artifact_id: ArtifactId
    batch_row: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _artifact_id(self.artifact_id))
        object.__setattr__(self, "batch_row", _integer(self.batch_row, name="batch_row", minimum=0))


@dataclass(frozen=True, slots=True)
class DayKey:
    """Semantic Teacher-day identity used to prevent holdout leakage."""

    landpoint_id: str
    year: int
    start_row: int

    def __post_init__(self) -> None:
        if not isinstance(self.landpoint_id, str) or not self.landpoint_id:
            raise TypeError("landpoint_id must be a non-empty string")
        object.__setattr__(self, "year", _integer(self.year, name="year"))
        object.__setattr__(self, "start_row", _integer(self.start_row, name="start_row", minimum=0))


@dataclass(frozen=True, slots=True)
class OneDayPreparedRow:
    """Metadata needed to recover one row from ``artifact.batch.one_day``."""

    artifact_id: ArtifactId
    batch_row: int
    record_index: int
    row_index_within_record: int
    landpoint_id: str
    year: int
    start_row: int
    prepared_day_index: int | None = None
    parent_shard_sha256: str | None = None
    typed_shard_sha256: str | None = None

    @property
    def prepared_row_key(self) -> PreparedRowKey:
        return PreparedRowKey(self.artifact_id, self.batch_row)

    @property
    def day_key(self) -> DayKey:
        return DayKey(self.landpoint_id, self.year, self.start_row)

    def as_metadata(self) -> Mapping[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "batch_row": self.batch_row,
            "record_index": self.record_index,
            "row_index_within_record": self.row_index_within_record,
            "landpoint_id": self.landpoint_id,
            "year": self.year,
            "start_row": self.start_row,
            "prepared_day_index": self.prepared_day_index,
            "parent_shard_sha256": self.parent_shard_sha256,
            "typed_shard_sha256": self.typed_shard_sha256,
        }


@dataclass(frozen=True, slots=True)
class OneDayArtifact:
    """Flattened one-day row metadata for one prepared package."""

    artifact_id: ArtifactId
    rows: tuple[OneDayPreparedRow, ...]

    def __post_init__(self) -> None:
        artifact_id = _artifact_id(self.artifact_id)
        object.__setattr__(self, "artifact_id", artifact_id)
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("one-day artifact rows must be a non-empty tuple")
        for expected_batch_row, row in enumerate(self.rows):
            if not isinstance(row, OneDayPreparedRow):
                raise TypeError("one-day artifact rows must contain OneDayPreparedRow values")
            if row.artifact_id != artifact_id or row.batch_row != expected_batch_row:
                raise ValueError("one-day artifact row identity or flattened order drift")

    @classmethod
    def from_schedule_entry(
        cls,
        artifact_id: ArtifactId,
        schedule_entry: Mapping[str, Any],
        *,
        prepared_day_indices: Sequence[int] | None = None,
    ) -> OneDayArtifact:
        """Flatten ``components.one_day`` exactly as the v6 preparer stacks it."""

        artifact_id = _artifact_id(artifact_id)
        if not isinstance(schedule_entry, Mapping):
            raise TypeError("schedule_entry must be a mapping")
        components = schedule_entry.get("components")
        if not isinstance(components, Mapping):
            raise TypeError("schedule_entry components must be a mapping")
        records = _items(components.get("one_day"), name="components.one_day")
        if not records:
            raise ValueError("components.one_day must not be empty")

        flattened: list[tuple[int, int, str, int, int, str | None, str | None]] = []
        runtime_year = schedule_entry.get("runtime_year")
        declared_runtime_year = None if runtime_year is None else _integer(runtime_year, name="runtime_year")
        for record_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError("components.one_day records must be mappings")
            landpoint_id = record.get("landpoint_id")
            if not isinstance(landpoint_id, str) or not landpoint_id:
                raise TypeError("one-day record landpoint_id must be a non-empty string")
            year = _integer(record.get("year"), name="one-day record year")
            if declared_runtime_year is not None and year != declared_runtime_year:
                raise ValueError("one-day record year differs from artifact runtime_year")
            starts = _items(record.get("start_rows"), name="one-day record start_rows")
            if not starts:
                raise ValueError("one-day record start_rows must not be empty")
            parent_sha256 = _optional_string(record, "parent_shard_sha256")
            typed_sha256 = _optional_string(record, "typed_shard_sha256")
            for row_index, start_row in enumerate(starts):
                flattened.append(
                    (
                        record_index,
                        row_index,
                        landpoint_id,
                        year,
                        _integer(start_row, name="one-day start_row", minimum=0),
                        parent_sha256,
                        typed_sha256,
                    )
                )

        if prepared_day_indices is None:
            day_indices: tuple[int | None, ...] = (None,) * len(flattened)
        else:
            raw_day_indices = _items(prepared_day_indices, name="prepared_day_indices")
            if len(raw_day_indices) != len(flattened):
                raise ValueError("prepared_day_indices length differs from flattened one-day rows")
            day_indices = tuple(
                _integer(value, name="prepared_day_index", minimum=0) for value in raw_day_indices
            )

        rows = tuple(
            OneDayPreparedRow(
                artifact_id=artifact_id,
                batch_row=batch_row,
                record_index=value[0],
                row_index_within_record=value[1],
                landpoint_id=value[2],
                year=value[3],
                start_row=value[4],
                prepared_day_index=day_indices[batch_row],
                parent_shard_sha256=value[5],
                typed_shard_sha256=value[6],
            )
            for batch_row, value in enumerate(flattened)
        )
        return cls(artifact_id=artifact_id, rows=rows)


@dataclass(frozen=True, slots=True)
class OneDaySamplingConfig:
    seed: int
    batch_size: int = 32
    microbatch_size: int = 4
    max_artifacts_per_batch: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _integer(self.seed, name="seed", minimum=0))
        object.__setattr__(self, "batch_size", _integer(self.batch_size, name="batch_size", minimum=1))
        object.__setattr__(
            self,
            "microbatch_size",
            _integer(self.microbatch_size, name="microbatch_size", minimum=1),
        )
        object.__setattr__(
            self,
            "max_artifacts_per_batch",
            _integer(self.max_artifacts_per_batch, name="max_artifacts_per_batch", minimum=1),
        )
        if self.microbatch_size > self.batch_size:
            raise ValueError("microbatch_size must not exceed batch_size")


@dataclass(frozen=True, slots=True)
class OneDaySamplingCursor:
    epoch: int
    batch_offset: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", _integer(self.epoch, name="epoch", minimum=0))
        object.__setattr__(
            self,
            "batch_offset",
            _integer(self.batch_offset, name="batch_offset", minimum=0),
        )


@dataclass(frozen=True, slots=True)
class OneDayMicrobatch:
    index: int
    rows: tuple[OneDayPreparedRow, ...]

    @property
    def draw_count(self) -> int:
        return len(self.rows)

    @property
    def unique_day_count(self) -> int:
        return len({row.day_key for row in self.rows})

    @property
    def artifact_ids(self) -> tuple[ArtifactId, ...]:
        return tuple(dict.fromkeys(row.artifact_id for row in self.rows))


@dataclass(frozen=True, slots=True)
class OneDayBatch:
    """A metadata-only batch plan whose rows can be gathered after package load."""

    epoch: int
    seed: int
    batch_index: int
    draw_offset: int
    rows: tuple[OneDayPreparedRow, ...]
    requested_batch_size: int
    microbatch_size: int
    max_artifacts_per_batch: int
    epoch_draw_count: int
    epoch_unique_day_count: int
    epoch_unique_day_count_through_batch: int

    @property
    def cursor(self) -> OneDaySamplingCursor:
        return OneDaySamplingCursor(self.epoch, self.batch_index)

    @property
    def next_cursor(self) -> OneDaySamplingCursor:
        return OneDaySamplingCursor(self.epoch, self.batch_index + 1)

    @property
    def draw_count(self) -> int:
        return len(self.rows)

    @property
    def unique_day_count(self) -> int:
        return len({row.day_key for row in self.rows})

    @property
    def artifact_ids(self) -> tuple[ArtifactId, ...]:
        return tuple(dict.fromkeys(row.artifact_id for row in self.rows))

    @property
    def microbatches(self) -> tuple[OneDayMicrobatch, ...]:
        return tuple(
            OneDayMicrobatch(index=index, rows=self.rows[start : start + self.microbatch_size])
            for index, start in enumerate(range(0, len(self.rows), self.microbatch_size))
        )

    def as_metadata(self) -> Mapping[str, Any]:
        microbatches = []
        start = 0
        for microbatch in self.microbatches:
            stop = start + microbatch.draw_count
            microbatches.append(
                {
                    "index": microbatch.index,
                    "start": start,
                    "stop": stop,
                    "draw_count": microbatch.draw_count,
                    "unique_day_count": microbatch.unique_day_count,
                    "artifact_ids": list(microbatch.artifact_ids),
                }
            )
            start = stop
        return {
            "schema_version": "prepared_one_day_batch_plan_v1",
            "epoch_scope": EPOCH_SCOPE,
            "epoch": self.epoch,
            "seed": self.seed,
            "batch_index": self.batch_index,
            "draw_offset": self.draw_offset,
            "requested_batch_size": self.requested_batch_size,
            "actual_draw_count": self.draw_count,
            "unique_day_count": self.unique_day_count,
            "epoch_draw_count": self.epoch_draw_count,
            "epoch_unique_day_count": self.epoch_unique_day_count,
            "epoch_unique_day_count_through_batch": self.epoch_unique_day_count_through_batch,
            "artifact_ids": list(self.artifact_ids),
            "rows": [row.as_metadata() for row in self.rows],
            "microbatches": microbatches,
            "next_cursor": {
                "epoch": self.next_cursor.epoch,
                "batch_offset": self.next_cursor.batch_offset,
            },
        }


@dataclass(slots=True)
class _ArtifactCursor:
    artifact_id: ArtifactId
    rows: tuple[OneDayPreparedRow, ...]
    offset: int = 0

    @property
    def exhausted(self) -> bool:
        return self.offset >= len(self.rows)

    def take(self) -> OneDayPreparedRow:
        row = self.rows[self.offset]
        self.offset += 1
        return row


class OneDaySampler:
    """Re-derivable epoch iterator over prepared rows with bounded package mixing."""

    def __init__(
        self,
        artifacts: Sequence[OneDayArtifact],
        *,
        seed: int,
        batch_size: int = 32,
        microbatch_size: int = 4,
        max_artifacts_per_batch: int = 4,
        excluded_rows: Iterable[PreparedRowKey] = (),
        excluded_days: Iterable[DayKey] = (),
    ) -> None:
        self.config = OneDaySamplingConfig(
            seed=seed,
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            max_artifacts_per_batch=max_artifacts_per_batch,
        )
        supplied = tuple(artifacts)
        if not supplied:
            raise ValueError("one-day sampling requires at least one artifact")
        if any(not isinstance(artifact, OneDayArtifact) for artifact in supplied):
            raise TypeError("artifacts must contain OneDayArtifact values")
        ids = tuple(artifact.artifact_id for artifact in supplied)
        if len(set(ids)) != len(ids):
            raise ValueError("one-day artifact IDs must be unique")

        row_exclusions = frozenset(excluded_rows)
        day_exclusions = frozenset(excluded_days)
        if any(not isinstance(value, PreparedRowKey) for value in row_exclusions):
            raise TypeError("excluded_rows must contain PreparedRowKey values")
        if any(not isinstance(value, DayKey) for value in day_exclusions):
            raise TypeError("excluded_days must contain DayKey values")

        all_rows = tuple(row for artifact in supplied for row in artifact.rows)
        available_row_keys = {row.prepared_row_key for row in all_rows}
        available_day_keys = {row.day_key for row in all_rows}
        unknown_rows = row_exclusions - available_row_keys
        unknown_days = day_exclusions - available_day_keys
        if unknown_rows or unknown_days:
            raise ValueError("one-day exclusions contain identities absent from the prepared pool")

        filtered = []
        for artifact in supplied:
            rows = tuple(
                row
                for row in artifact.rows
                if row.prepared_row_key not in row_exclusions and row.day_key not in day_exclusions
            )
            if rows:
                filtered.append((artifact.artifact_id, rows))
        if not filtered:
            raise ValueError("one-day exclusions removed the complete prepared pool")

        self._rows_by_artifact = tuple(filtered)
        self._draw_count = sum(len(rows) for _artifact_id_value, rows in filtered)
        self._unique_day_count = len({row.day_key for _artifact_id_value, rows in filtered for row in rows})
        self._excluded_draw_count = len(all_rows) - self._draw_count
        self._source_artifact_count = len(supplied)

    @property
    def draw_count(self) -> int:
        """Prepared row occurrences consumed once in every epoch."""

        return self._draw_count

    @property
    def unique_day_count(self) -> int:
        """Distinct point/year/start-row identities in the included pool."""

        return self._unique_day_count

    @property
    def excluded_draw_count(self) -> int:
        return self._excluded_draw_count

    @property
    def artifact_count(self) -> int:
        return len(self._rows_by_artifact)

    def pool_metadata(self) -> Mapping[str, Any]:
        return {
            "epoch_scope": EPOCH_SCOPE,
            "seed": self.config.seed,
            "source_artifact_count": self._source_artifact_count,
            "included_artifact_count": self.artifact_count,
            "draw_count": self.draw_count,
            "unique_day_count": self.unique_day_count,
            "excluded_draw_count": self.excluded_draw_count,
            "batch_size": self.config.batch_size,
            "microbatch_size": self.config.microbatch_size,
            "max_artifacts_per_batch": self.config.max_artifacts_per_batch,
        }

    def _ordered_rows(self, epoch: int) -> tuple[tuple[ArtifactId, tuple[OneDayPreparedRow, ...]], ...]:
        canonical_artifacts = tuple(
            sorted(
                self._rows_by_artifact,
                key=lambda item: _artifact_order_key(item[0]),
            )
        )
        generator = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence((self.config.seed, epoch)))
        )
        artifact_permutation = generator.permutation(len(canonical_artifacts))
        shuffled_rows = []
        for artifact_id, rows in canonical_artifacts:
            canonical_rows = tuple(sorted(rows, key=lambda row: row.batch_row))
            row_permutation = generator.permutation(len(canonical_rows))
            shuffled_rows.append(
                (
                    artifact_id,
                    tuple(canonical_rows[int(index)] for index in row_permutation),
                )
            )
        return tuple(shuffled_rows[int(index)] for index in artifact_permutation)

    def _iter_complete_epoch(self, epoch: int) -> Iterator[OneDayBatch]:
        ordered = self._ordered_rows(epoch)
        pending = 0
        active: list[_ArtifactCursor] = []
        batch_index = 0
        draw_offset = 0
        seen_days: set[DayKey] = set()

        while active or pending < len(ordered):
            active = [cursor for cursor in active if not cursor.exhausted]
            while len(active) < self.config.max_artifacts_per_batch and pending < len(ordered):
                artifact_id, rows = ordered[pending]
                pending += 1
                active.append(_ArtifactCursor(artifact_id=artifact_id, rows=rows))
            if not active:
                break

            batch_rows: list[OneDayPreparedRow] = []
            while len(batch_rows) < self.config.batch_size:
                progressed = False
                for cursor in active:
                    if cursor.exhausted:
                        continue
                    batch_rows.append(cursor.take())
                    progressed = True
                    if len(batch_rows) == self.config.batch_size:
                        break
                if not progressed:
                    break
            if not batch_rows:
                raise RuntimeError("one-day sampler made no progress")

            seen_days.update(row.day_key for row in batch_rows)
            rows = tuple(batch_rows)
            yield OneDayBatch(
                epoch=epoch,
                seed=self.config.seed,
                batch_index=batch_index,
                draw_offset=draw_offset,
                rows=rows,
                requested_batch_size=self.config.batch_size,
                microbatch_size=self.config.microbatch_size,
                max_artifacts_per_batch=self.config.max_artifacts_per_batch,
                epoch_draw_count=self.draw_count,
                epoch_unique_day_count=self.unique_day_count,
                epoch_unique_day_count_through_batch=len(seen_days),
            )
            draw_offset += len(rows)
            batch_index += 1

        if draw_offset != self.draw_count or len(seen_days) != self.unique_day_count:
            raise RuntimeError("one-day sampler epoch coverage drift")

    def iter_epoch(self, epoch: int, *, start_batch: int = 0) -> Iterator[OneDayBatch]:
        """Yield one deterministic prepared-pool epoch, optionally resuming by batch offset."""

        epoch = _integer(epoch, name="epoch", minimum=0)
        start_batch = _integer(start_batch, name="start_batch", minimum=0)
        batch_count = 0
        for batch in self._iter_complete_epoch(epoch):
            batch_count += 1
            if batch.batch_index >= start_batch:
                yield batch
        if start_batch > batch_count:
            raise ValueError("start_batch lies beyond the prepared-pool epoch")

    def resume(self, cursor: OneDaySamplingCursor) -> Iterator[OneDayBatch]:
        if not isinstance(cursor, OneDaySamplingCursor):
            raise TypeError("cursor must be a OneDaySamplingCursor")
        return self.iter_epoch(cursor.epoch, start_batch=cursor.batch_offset)

    def batch_count(self, epoch: int) -> int:
        return sum(1 for _batch in self.iter_epoch(epoch))


__all__ = [
    "EPOCH_SCOPE",
    "DayKey",
    "OneDayArtifact",
    "OneDayBatch",
    "OneDayMicrobatch",
    "OneDayPreparedRow",
    "OneDaySampler",
    "OneDaySamplingConfig",
    "OneDaySamplingCursor",
    "PreparedRowKey",
]
