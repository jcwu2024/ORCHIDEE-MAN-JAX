"""Hash-bound CPU-prepared artifacts for the successor 7/30-day screen."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file

PREPARED_ARTIFACT_SCHEMA = "successor_7_30_rollout_prepared_artifact_v1"
PREPARED_MANIFEST_SCHEMA = "successor_7_30_rollout_prepared_manifest_v1"


class PreparedRolloutScreenArtifact(NamedTuple):
    schema_version: str
    slice_id: str
    window_entry: Mapping[str, Any]
    trace_signature: str
    batch: Any
    day_inputs: Any


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_prepared_rollout_artifact(
    path: Path,
    artifact: PreparedRolloutScreenArtifact,
) -> Mapping[str, Any]:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_prepared_rollout_manifest(
    path: str | Path,
    *,
    screen_contract_sha256: str,
    window_plan_sha256: str,
    source_git_head: str,
) -> Mapping[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != PREPARED_MANIFEST_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("screen_contract_sha256") != screen_contract_sha256
        or payload.get("window_plan_sha256") != window_plan_sha256
        or payload.get("source_git_head") != source_git_head
        or not payload.get("entries")
    ):
        raise ValueError("successor rollout prepared-manifest identity drift")
    if int(payload.get("artifact_count", -1)) != len(payload["entries"]):
        raise ValueError("successor rollout prepared-manifest count drift")
    if int(payload.get("total_size_bytes", -1)) != sum(
        int(item["size_bytes"]) for item in payload["entries"]
    ):
        raise ValueError("successor rollout prepared-manifest size drift")
    return payload


def load_prepared_rollout_artifact(
    manifest_path: str | Path,
    record: Mapping[str, Any],
    *,
    slice_id: str,
    window_entry: Mapping[str, Any],
) -> PreparedRolloutScreenArtifact:
    root = Path(manifest_path).resolve().parent
    path = (root / str(record["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("successor rollout prepared artifact escaped its root") from error
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError("successor rollout prepared artifact hash drift")
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if (
        not isinstance(artifact, PreparedRolloutScreenArtifact)
        or artifact.schema_version != PREPARED_ARTIFACT_SCHEMA
        or artifact.slice_id != slice_id
        or dict(artifact.window_entry) != dict(window_entry)
        or artifact.trace_signature != record.get("trace_signature")
    ):
        raise ValueError("successor rollout prepared artifact payload drift")
    return artifact
