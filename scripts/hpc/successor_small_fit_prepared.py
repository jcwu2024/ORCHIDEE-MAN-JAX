"""Hash-bound serialization for preassembled successor small-fit updates."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, NamedTuple

PREPARED_ARTIFACT_SCHEMA = "successor_small_fit_prepared_update_v1"
PREPARED_MANIFEST_SCHEMA = "successor_small_fit_prepared_manifest_v1"


class PreparedUpdateArtifact(NamedTuple):
    schema_version: str
    schedule_entry: Mapping[str, Any]
    trace_signature: str
    batch: Any
    day_inputs: Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_prepared_artifact(
    path: Path,
    artifact: PreparedUpdateArtifact,
    *,
    compression: str = "none",
) -> Mapping[str, Any]:
    if compression not in {"none", "gzip"}:
        raise ValueError("unsupported successor prepared-artifact compression")
    if compression == "gzip" and not path.name.endswith(".pkl.gz"):
        raise ValueError("gzip successor artifact path must end in .pkl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw_handle:
        if compression == "gzip":
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=1,
                fileobj=raw_handle,
                mtime=0,
            ) as handle:
                pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(artifact, raw_handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "compression": compression,
    }


def load_prepared_manifest(
    path: Path,
    *,
    runner_contract_sha256: str,
    schedule_sha256: str,
    updates: int,
    source_git_head: str,
) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PREPARED_MANIFEST_SCHEMA:
        raise ValueError("unsupported successor prepared-batch manifest")
    if payload.get("status") != "passed":
        raise ValueError("successor prepared-batch manifest is not complete")
    identity = payload.get("identity", {})
    if (
        identity.get("runner_contract_sha256") != runner_contract_sha256
        or identity.get("schedule_sha256") != schedule_sha256
        or identity.get("source_git_head") != source_git_head
        or int(payload.get("updates", -1)) != updates
    ):
        raise ValueError("successor prepared-batch manifest identity drift")
    entries = payload.get("entries", [])
    if len(entries) != updates or [int(item.get("update", -1)) for item in entries] != list(range(1, updates + 1)):
        raise ValueError("successor prepared-batch manifest update coverage drift")
    return payload


def load_prepared_artifact(
    manifest_path: Path,
    record: Mapping[str, Any],
    *,
    schedule_entry: Mapping[str, Any],
) -> PreparedUpdateArtifact:
    path = (manifest_path.parent / str(record["path"])).resolve()
    try:
        path.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("prepared successor artifact escaped its manifest directory") from error
    if not path.is_file() or sha256_file(path) != str(record["sha256"]):
        raise ValueError("prepared successor artifact hash drift")
    compression = str(record.get("compression", "none"))
    if compression == "gzip":
        if not path.name.endswith(".pkl.gz"):
            raise ValueError("gzip successor artifact path must end in .pkl.gz")
        with gzip.open(path, "rb") as handle:
            artifact = pickle.load(handle)
    elif compression == "none":
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
    else:
        raise ValueError("unsupported successor prepared-artifact compression")
    if not isinstance(artifact, PreparedUpdateArtifact):
        raise ValueError("prepared successor artifact type drift")
    if artifact.schema_version != PREPARED_ARTIFACT_SCHEMA:
        raise ValueError("prepared successor artifact schema drift")
    if dict(artifact.schedule_entry) != dict(schedule_entry):
        raise ValueError("prepared successor artifact schedule drift")
    if artifact.trace_signature != str(record["trace_signature"]):
        raise ValueError("prepared successor artifact trace identity drift")
    return artifact
