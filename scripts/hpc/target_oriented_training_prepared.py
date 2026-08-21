"""Serialization for prepared target-oriented effective updates."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from scripts.hpc.successor_small_fit_prepared import atomic_json, sha256_file

PREPARED_ARTIFACT_SCHEMA = "target_oriented_prepared_update_v1"
PREPARED_MANIFEST_SCHEMA = "target_oriented_prepared_manifest_v1"


class PreparedTargetOrientedUpdateArtifact(NamedTuple):
    schema_version: str
    training_contract_sha256: str
    schedule_entry: Mapping[str, Any]
    trace_signature: str
    batch: Any


def write_prepared_target_artifact(
    path: Path,
    artifact: PreparedTargetOrientedUpdateArtifact,
) -> Mapping[str, Any]:
    if not path.name.endswith(".pkl.gz"):
        raise ValueError("target-oriented prepared artifact must end in .pkl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=1,
            fileobj=raw,
            mtime=0,
        ) as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_prepared_target_artifact(
    manifest_path: Path,
    record: Mapping[str, Any],
    *,
    training_contract_sha256: str,
    schedule_entry: Mapping[str, Any],
) -> PreparedTargetOrientedUpdateArtifact:
    root = manifest_path.parent.resolve()
    path = (root / str(record["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("prepared target artifact escaped its manifest directory") from error
    if not path.is_file() or sha256_file(path) != str(record["sha256"]):
        raise ValueError("prepared target artifact hash drift")
    with gzip.open(path, "rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, PreparedTargetOrientedUpdateArtifact):
        raise ValueError("prepared target artifact type drift")
    if (
        artifact.schema_version != PREPARED_ARTIFACT_SCHEMA
        or artifact.training_contract_sha256 != training_contract_sha256
        or dict(artifact.schedule_entry) != dict(schedule_entry)
        or artifact.trace_signature != str(record["trace_signature"])
    ):
        raise ValueError("prepared target artifact identity drift")
    return artifact


def load_prepared_target_manifest(
    path: Path,
    *,
    training_contract_sha256: str,
    schedule_sha256: str,
    updates: int,
    source_git_head: str,
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = payload.get("identity", {})
    entries = payload.get("entries", [])
    if (
        payload.get("schema_version") != PREPARED_MANIFEST_SCHEMA
        or payload.get("status") != "passed"
        or identity.get("training_contract_sha256") != training_contract_sha256
        or identity.get("schedule_sha256") != schedule_sha256
        or identity.get("source_git_head") != source_git_head
        or int(payload.get("updates", -1)) != updates
        or [int(record.get("update", -1)) for record in entries] != list(range(1, updates + 1))
    ):
        raise ValueError("prepared target manifest identity or coverage drift")
    return payload


__all__ = [
    "PREPARED_ARTIFACT_SCHEMA",
    "PREPARED_MANIFEST_SCHEMA",
    "PreparedTargetOrientedUpdateArtifact",
    "atomic_json",
    "load_prepared_target_artifact",
    "load_prepared_target_manifest",
    "sha256_file",
    "write_prepared_target_artifact",
]
