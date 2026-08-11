"""Preassemble frozen successor small-fit batches on a CPU host."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_small_fit import (
    _git_head,
    _prepare_update,
    _retained_tail_trace_signature,
    _validate_update_count,
)
from scripts.hpc.successor_small_fit_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedUpdateArtifact,
    atomic_json,
    load_prepared_artifact,
    sha256_file,
    write_prepared_artifact,
)


def _initialize_cpu_backend() -> tuple[Any, tuple[Any, ...]]:
    import jax

    jax.config.update("jax_platforms", "cpu")
    devices = tuple(jax.devices("cpu"))
    if jax.default_backend() != "cpu" or not devices:
        raise RuntimeError("successor batch preparation requires a CPU JAX backend")
    return jax, devices


def _identity(
    runtime: Mapping[str, Any],
    runner_contract: Any,
    schedule: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "source_git_head": _git_head(),
        "runner_contract_sha256": runner_contract.sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "dataset_manifest_sha256": runner_contract.payload["identity"]["dataset_manifest_sha256"],
        "successor_candidate_contract_sha256": runtime["successor_contract"].sha256,
        "fast_owner_contract_sha256": runtime["owner_contract"].contract_sha256,
    }


def _worker_updates(updates: int, worker_index: int, worker_count: int) -> tuple[int, ...]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index is outside worker_count")
    return tuple(range(worker_index + 1, updates + 1, worker_count))


def _load_progress(
    path: Path,
    identity: Mapping[str, Any],
    updates: int,
    *,
    worker_index: int | None = None,
    worker_count: int | None = None,
    artifact_compression: str = "none",
) -> dict[str, Any]:
    worker = None if worker_index is None else {"index": int(worker_index), "count": int(worker_count)}
    if not path.exists():
        payload = {
            "schema_version": PREPARED_MANIFEST_SCHEMA,
            "status": "in_progress",
            "identity": dict(identity),
            "updates": updates,
            "entries": [],
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "artifact_compression": artifact_compression,
        }
        if worker is not None:
            payload["worker"] = worker
        return payload
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != PREPARED_MANIFEST_SCHEMA
        or payload.get("identity") != dict(identity)
        or int(payload.get("updates", -1)) != updates
        or payload.get("worker") != worker
        or payload.get("artifact_compression", "none") != artifact_compression
    ):
        raise ValueError("existing successor preparation progress has different identity")
    return payload


def _finalize_worker_progress(
    output_dir: Path,
    identity: Mapping[str, Any],
    schedule: Mapping[str, Any],
    updates: int,
    worker_count: int,
    artifact_compression: str = "none",
) -> Mapping[str, Any]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    started = time.perf_counter()
    manifest_path = output_dir / "prepared_manifest.json"
    completed: dict[int, Mapping[str, Any]] = {}
    for worker_index in range(worker_count):
        partial_path = output_dir / "partials" / f"worker-{worker_index:04d}.json"
        if not partial_path.is_file():
            raise ValueError(f"missing successor preparation worker {worker_index}")
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        expected_updates = _worker_updates(updates, worker_index, worker_count)
        observed_updates = tuple(int(record.get("update", -1)) for record in partial.get("entries", ()))
        if (
            partial.get("schema_version") != PREPARED_MANIFEST_SCHEMA
            or partial.get("status") != "passed"
            or partial.get("identity") != dict(identity)
            or int(partial.get("updates", -1)) != updates
            or partial.get("worker") != {"index": worker_index, "count": worker_count}
            or partial.get("teacher_data_regenerated") is not False
            or partial.get("historical_v5_used") is not False
            or partial.get("artifact_compression", "none") != artifact_compression
            or observed_updates != expected_updates
        ):
            raise ValueError(f"successor preparation worker {worker_index} coverage or identity drift")
        for record in partial["entries"]:
            update = int(record["update"])
            if update in completed:
                raise ValueError(f"duplicate successor prepared update {update}")
            load_prepared_artifact(
                manifest_path,
                record,
                schedule_entry=schedule["entries"][update - 1],
            )
            completed[update] = record

    if tuple(sorted(completed)) != tuple(range(1, updates + 1)):
        raise ValueError("successor preparation worker merge has incomplete coverage")
    entries = [completed[index] for index in range(1, updates + 1)]
    progress = {
        "schema_version": PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "identity": dict(identity),
        "updates": updates,
        "entries": entries,
        "trace_signatures": sorted({record["trace_signature"] for record in entries}),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in entries),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "worker_count": worker_count,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "artifact_compression": artifact_compression,
    }
    atomic_json(manifest_path, progress)
    return progress | {"manifest_sha256": sha256_file(manifest_path)}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args, backend_initializer=_initialize_cpu_backend)
    jax = runtime["jax"]

    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        verify_successor_small_fit_schedule,
    )

    runner_contract = load_successor_small_fit_contract(args.runner_contract)
    updates = _validate_update_count(
        args.updates,
        int(runner_contract.sampling["maximum_updates_per_arm"]),
    )
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_successor_small_fit_schedule(
        references,
        runner_contract,
        updates=updates,
    )
    verify_successor_small_fit_schedule(schedule, references, runner_contract)
    identity = _identity(runtime, runner_contract, schedule)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = args.worker_count
    worker_index = args.worker_index
    if args.finalize:
        if worker_count is None or worker_index is not None:
            raise ValueError("finalize requires worker_count and forbids worker_index")
        return _finalize_worker_progress(
            output_dir,
            identity,
            schedule,
            updates,
            worker_count,
            args.artifact_compression,
        )
    if (worker_count is None) != (worker_index is None):
        raise ValueError("worker_index and worker_count must be supplied together")
    assigned_updates = (
        tuple(range(1, updates + 1)) if worker_index is None else _worker_updates(updates, worker_index, worker_count)
    )
    manifest_path = output_dir / "prepared_manifest.json"
    progress_path = (
        manifest_path if worker_index is None else output_dir / "partials" / f"worker-{worker_index:04d}.json"
    )
    progress = _load_progress(
        progress_path,
        identity,
        updates,
        worker_index=worker_index,
        worker_count=worker_count,
        artifact_compression=args.artifact_compression,
    )
    completed = {int(record["update"]): record for record in progress["entries"]}
    started = time.perf_counter()

    for update, entry in enumerate(schedule["entries"], start=1):
        if update not in assigned_updates:
            continue
        existing = completed.get(update)
        if existing is not None:
            load_prepared_artifact(
                manifest_path,
                existing,
                schedule_entry=entry,
            )
            print(f"successor_prepared_update_reused update={update}", flush=True)
            continue
        entry_started = time.perf_counter()
        _reference, batch, adapter, preparation = _prepare_update(runtime, entry)
        artifact = PreparedUpdateArtifact(
            schema_version=PREPARED_ARTIFACT_SCHEMA,
            schedule_entry=dict(entry),
            trace_signature=_retained_tail_trace_signature(adapter),
            batch=jax.tree_util.tree_map(np.asarray, batch),
            day_inputs=jax.tree_util.tree_map(np.asarray, adapter.day_inputs),
        )
        suffix = ".pkl.gz" if args.artifact_compression == "gzip" else ".pkl"
        artifact_path = output_dir / f"update-{update:04d}{suffix}"
        file_record = write_prepared_artifact(
            artifact_path,
            artifact,
            compression=args.artifact_compression,
        )
        record = {
            "update": update,
            "landpoint_id": str(entry["landpoint_id"]),
            "year": int(entry["year"]),
            "trace_signature": artifact.trace_signature,
            **file_record,
            "preparation": dict(preparation),
            "total_seconds": time.perf_counter() - entry_started,
        }
        completed[update] = record
        progress["entries"] = [completed[index] for index in sorted(completed)]
        progress["total_size_bytes"] = sum(int(item["size_bytes"]) for item in progress["entries"])
        progress["elapsed_seconds_this_invocation"] = time.perf_counter() - started
        atomic_json(progress_path, progress)
        print(
            "successor_prepared_update_written "
            f"update={update} size_bytes={record['size_bytes']} "
            f"seconds={record['total_seconds']:.3f}",
            flush=True,
        )

    progress["status"] = "passed"
    progress["entries"] = [completed[index] for index in assigned_updates]
    progress["trace_signatures"] = sorted({record["trace_signature"] for record in progress["entries"]})
    progress["total_size_bytes"] = sum(int(item["size_bytes"]) for item in progress["entries"])
    progress["elapsed_seconds_this_invocation"] = time.perf_counter() - started
    atomic_json(progress_path, progress)
    progress["manifest_sha256"] = sha256_file(progress_path)
    return progress


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--artifact-compression",
        choices=("none", "gzip"),
        default="none",
    )
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
