"""Prepare one incremental successor staged-training interval on CPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.prepare_successor_small_fit_batches import _initialize_cpu_backend
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_small_fit import (
    _git_head,
    _prepare_update,
    _retained_tail_trace_signature,
)
from scripts.hpc.successor_small_fit_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PreparedUpdateArtifact,
    atomic_json,
    load_prepared_artifact,
    sha256_file,
    write_prepared_artifact,
)
from scripts.hpc.successor_staged_prepared import (
    STAGED_PREPARED_MANIFEST_SCHEMA,
    stage_update_range,
    staged_worker_updates,
)


def _identity(
    runtime: Mapping[str, Any],
    contract: Any,
    schedule: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "producer_git_head": _git_head(),
        "staged_training_contract_sha256": contract.sha256,
        "full_schedule_sha256": schedule["schedule_sha256"],
        "dataset_manifest_sha256": contract.payload["identity"][
            "dataset_manifest_sha256"
        ],
        "successor_candidate_contract_sha256": runtime[
            "successor_contract"
        ].sha256,
        "fast_owner_contract_sha256": runtime["owner_contract"].contract_sha256,
    }


def _load_progress(
    path: Path,
    *,
    identity: Mapping[str, Any],
    start_update: int,
    end_update: int,
    worker_index: int,
    worker_count: int,
) -> dict[str, Any]:
    worker = {"index": int(worker_index), "count": int(worker_count)}
    if not path.exists():
        return {
            "schema_version": STAGED_PREPARED_MANIFEST_SCHEMA,
            "status": "in_progress",
            "identity": dict(identity),
            "start_update": int(start_update),
            "end_update": int(end_update),
            "worker": worker,
            "entries": [],
            "artifact_compression": "gzip",
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_used": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != STAGED_PREPARED_MANIFEST_SCHEMA
        or payload.get("identity") != dict(identity)
        or int(payload.get("start_update", 0)) != int(start_update)
        or int(payload.get("end_update", 0)) != int(end_update)
        or payload.get("worker") != worker
        or payload.get("artifact_compression") != "gzip"
    ):
        raise ValueError("existing successor staged preparation identity drift")
    return payload


def _finalize(
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
    schedule: Mapping[str, Any],
    start_update: int,
    end_update: int,
    worker_count: int,
) -> Mapping[str, Any]:
    manifest_path = output_dir / "prepared_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    records: dict[int, Mapping[str, Any]] = {}
    for worker_index in range(int(worker_count)):
        partial_path = output_dir / "partials" / f"worker-{worker_index:04d}.json"
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        assigned = staged_worker_updates(
            start_update,
            end_update,
            worker_index,
            worker_count,
        )
        if (
            partial.get("schema_version") != STAGED_PREPARED_MANIFEST_SCHEMA
            or partial.get("status") != "passed"
            or partial.get("identity") != dict(identity)
            or partial.get("worker")
            != {"index": worker_index, "count": int(worker_count)}
            or tuple(int(item["update"]) for item in partial.get("entries", ()))
            != assigned
        ):
            raise ValueError(f"successor staged worker {worker_index} drift")
        for record in partial["entries"]:
            update = int(record["update"])
            if update in records:
                raise ValueError(f"duplicate successor staged update {update}")
            load_prepared_artifact(
                manifest_path,
                record,
                schedule_entry=schedule["entries"][update - 1],
            )
            records[update] = record
    expected = tuple(range(int(start_update), int(end_update) + 1))
    if tuple(sorted(records)) != expected:
        raise ValueError("successor staged preparation coverage is incomplete")
    entries = [records[update] for update in expected]
    payload = {
        "schema_version": STAGED_PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "identity": dict(identity),
        "start_update": int(start_update),
        "end_update": int(end_update),
        "artifact_count": len(entries),
        "entries": entries,
        "trace_signatures": sorted({record["trace_signature"] for record in entries}),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in entries),
        "worker_count": int(worker_count),
        "artifact_compression": "gzip",
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
    }
    atomic_json(manifest_path, payload)
    return payload | {"manifest_sha256": sha256_file(manifest_path)}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args, backend_initializer=_initialize_cpu_backend)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        STAGED_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        verify_successor_small_fit_schedule,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if contract.payload["schema_version"] != STAGED_SCHEMA_VERSION:
        raise ValueError("staged preparation requires the staged-v3 contract")
    start_update, end_update = stage_update_range(
        contract.sampling["stage_milestones"],
        args.milestone,
    )
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_successor_small_fit_schedule(
        references,
        contract,
        updates=int(contract.sampling["maximum_updates_per_arm"]),
    )
    verify_successor_small_fit_schedule(schedule, references, contract)
    identity = _identity(runtime, contract, schedule)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        if args.worker_index is not None:
            raise ValueError("staged finalizer forbids worker_index")
        return _finalize(
            output_dir,
            identity=identity,
            schedule=schedule,
            start_update=start_update,
            end_update=end_update,
            worker_count=args.worker_count,
        )
    if args.worker_index is None:
        raise ValueError("staged preparation worker requires worker_index")
    assigned = staged_worker_updates(
        start_update,
        end_update,
        args.worker_index,
        args.worker_count,
    )
    progress_path = (
        output_dir / "partials" / f"worker-{args.worker_index:04d}.json"
    )
    progress = _load_progress(
        progress_path,
        identity=identity,
        start_update=start_update,
        end_update=end_update,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
    )
    completed = {int(record["update"]): record for record in progress["entries"]}
    started = time.perf_counter()
    manifest_path = output_dir / "prepared_manifest.json"
    for update in assigned:
        entry = schedule["entries"][update - 1]
        existing = completed.get(update)
        if existing is not None:
            load_prepared_artifact(manifest_path, existing, schedule_entry=entry)
            continue
        entry_started = time.perf_counter()
        _reference, batch, adapter, preparation = _prepare_update(runtime, entry)
        artifact = PreparedUpdateArtifact(
            PREPARED_ARTIFACT_SCHEMA,
            dict(entry),
            _retained_tail_trace_signature(adapter),
            jax.tree_util.tree_map(np.asarray, batch),
            jax.tree_util.tree_map(np.asarray, adapter.day_inputs),
        )
        path = output_dir / f"update-{update:05d}.pkl.gz"
        record = {
            "update": update,
            "landpoint_id": str(entry["landpoint_id"]),
            "year": int(entry["year"]),
            "trace_signature": artifact.trace_signature,
            **write_prepared_artifact(path, artifact, compression="gzip"),
            "preparation": dict(preparation),
            "total_seconds": time.perf_counter() - entry_started,
        }
        completed[update] = record
        progress["entries"] = [completed[index] for index in sorted(completed)]
        progress["elapsed_seconds_this_invocation"] = time.perf_counter() - started
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "event": "successor_staged_update_prepared",
                    "update": update,
                    "size_bytes": record["size_bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    progress["status"] = "passed"
    progress["entries"] = [completed[update] for update in assigned]
    progress["total_size_bytes"] = sum(
        int(record["size_bytes"]) for record in progress["entries"]
    )
    progress["elapsed_seconds_this_invocation"] = time.perf_counter() - started
    atomic_json(progress_path, progress)
    return progress | {"manifest_sha256": sha256_file(progress_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--finalize", action="store_true")
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
