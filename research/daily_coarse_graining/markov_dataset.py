"""Verified dataset reader for Daily Markov Teacher shards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

DATASET_SCHEMA_VERSION = "daily_teacher_dataset_manifest_v2"
SPLITS = frozenset({"train", "validation", "test"})


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


def load_dataset_index(
    manifest_path: str | Path, *, verify_hashes: bool = True
) -> MarkovDatasetIndex:
    manifest_path = Path(manifest_path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"dataset schema must be {DATASET_SCHEMA_VERSION!r}")
    root = manifest_path.parent
    contract_sha256 = str(raw["markov_contract_sha256"])
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
        if item.get("markov_contract_sha256") != contract_sha256:
            raise ValueError(f"contract drift for {key[0]}:{key[1]}")
        path = (root / item["shard"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"shard path escapes dataset root: {path}") from error
        if not path.is_file():
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


def collate_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack numerical model inputs/targets without exposing identity as a feature."""

    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    array_names = (
        "state",
        "forcing_native",
        "parameters",
        "landpoint_static",
        "annual_conditions",
        "next_state",
        "diagnostics",
        "year",
        "day_index",
    )
    result = {
        name: np.stack([np.asarray(sample[name]) for sample in samples])
        for name in array_names
    }
    for group in ("discrete_state", "next_discrete_state"):
        keys = tuple(samples[0][group])
        if any(tuple(sample[group]) != keys for sample in samples[1:]):
            raise ValueError(f"{group} schema drift across samples")
        result[group] = {
            key: np.stack([np.asarray(sample[group][key]) for sample in samples])
            for key in keys
        }
    result["state_finite"] = np.isfinite(result["state"])
    result["forcing_finite"] = np.isfinite(result["forcing_native"])
    result["next_state_finite"] = np.isfinite(result["next_state"])
    result["diagnostics_finite"] = np.isfinite(result["diagnostics"])
    return result


def batched(
    samples: Iterable[Mapping[str, Any]], batch_size: int
) -> Iterator[dict[str, Any]]:
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
