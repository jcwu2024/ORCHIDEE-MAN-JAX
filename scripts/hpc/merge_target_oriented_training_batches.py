"""Merge disjoint prepared-update chunks into one admitted 250-update manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from scripts.hpc.target_oriented_training_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedTargetOrientedUpdateArtifact,
    atomic_json,
    sha256_file,
)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "prepared_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    chunk_paths = tuple(path.resolve() for path in args.chunk_manifests)
    if not chunk_paths:
        raise ValueError("target-oriented merge requires prepared chunks")
    chunks = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in chunk_paths]
    first = chunks[0][1]
    identity = first.get("identity")
    updates = int(first.get("updates", -1))
    if updates != int(args.updates):
        raise ValueError("target-oriented merge update budget drift")
    merged = {}
    chunk_records = []
    for path, chunk in chunks:
        if (
            chunk.get("schema_version") != PREPARED_MANIFEST_SCHEMA
            or chunk.get("status") != "passed"
            or chunk.get("identity") != identity
            or int(chunk.get("updates", -1)) != updates
            or chunk.get("teacher_data_regenerated") is not False
            or chunk.get("historical_v5_used") is not False
            or chunk.get("sealed_test_read") is not False
        ):
            raise ValueError("target-oriented prepared chunk identity drift")
        start = int(chunk.get("start_update", -1))
        end = int(chunk.get("end_update", -1))
        expected = list(range(start, end + 1))
        if [int(record.get("update", -1)) for record in chunk.get("entries", ())] != expected:
            raise ValueError("target-oriented prepared chunk coverage drift")
        try:
            path.parent.relative_to(output_dir)
        except ValueError as error:
            raise ValueError("target-oriented prepared chunk escaped the merged root") from error
        for record in chunk["entries"]:
            update = int(record["update"])
            if update in merged:
                raise ValueError("target-oriented prepared chunks overlap")
            artifact_path = (path.parent / str(record["path"])).resolve()
            try:
                relative = artifact_path.relative_to(output_dir)
            except ValueError as error:
                raise ValueError("target-oriented prepared artifact escaped the merged root") from error
            adjusted = dict(record) | {"path": relative.as_posix()}
            if not artifact_path.is_file() or sha256_file(artifact_path) != str(record["sha256"]):
                raise ValueError("target-oriented prepared artifact hash drift during merge")
            with gzip.open(artifact_path, "rb") as handle:
                artifact = pickle.load(handle)
            if (
                not isinstance(artifact, PreparedTargetOrientedUpdateArtifact)
                or artifact.schema_version != PREPARED_ARTIFACT_SCHEMA
                or artifact.training_contract_sha256 != identity["training_contract_sha256"]
                or int(artifact.schedule_entry.get("update", -1)) != update
                or artifact.trace_signature != str(record["trace_signature"])
            ):
                raise ValueError("target-oriented prepared artifact payload drift during merge")
            merged[update] = adjusted
        chunk_records.append(
            {
                "manifest": str(path),
                "manifest_sha256": sha256_file(path),
                "start_update": start,
                "end_update": end,
                "artifact_count": len(chunk["entries"]),
            }
        )
    if sorted(merged) != list(range(1, updates + 1)):
        raise ValueError("target-oriented prepared chunks do not cover the full schedule")
    entries = [merged[update] for update in range(1, updates + 1)]
    payload = {
        "schema_version": PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "identity": identity,
        "updates": updates,
        "entries": entries,
        "trace_signatures": sorted({str(record["trace_signature"]) for record in entries}),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in entries),
        "chunks": chunk_records,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_read": False,
    }
    atomic_json(manifest_path, payload)
    return payload | {"manifest_sha256": sha256_file(manifest_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("chunk_manifests", nargs="+", type=Path)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
