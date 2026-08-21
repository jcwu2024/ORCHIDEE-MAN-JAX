"""Prepare real-v6 target-oriented effective updates on a free CPU node."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _retained_adapter
from scripts.hpc.run_successor_small_fit import _retained_tail_trace_signature
from scripts.hpc.run_successor_small_fit_probe import _plan_context
from scripts.hpc.target_oriented_training_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedTargetOrientedUpdateArtifact,
    atomic_json,
    load_prepared_target_artifact,
    load_prepared_target_manifest,
    sha256_file,
    write_prepared_target_artifact,
)


def _initialize_cpu_backend() -> tuple[Any, tuple[Any, ...]]:
    import jax

    jax.config.update("jax_platforms", "cpu")
    devices = tuple(jax.devices("cpu"))
    if jax.default_backend() != "cpu" or not devices:
        raise RuntimeError("target-oriented preparation requires a CPU JAX backend")
    return jax, devices


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _identity(reference: Any) -> tuple[str, int, str, str]:
    return (
        str(reference.parent.landpoint_id),
        int(reference.parent.year),
        str(reference.parent.sha256),
        str(reference.sha256),
    )


def _record_identity(record: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(record["landpoint_id"]),
        int(record["year"]),
        str(record["parent_shard_sha256"]),
        str(record["typed_shard_sha256"]),
    )


def _records(entry: Mapping[str, Any]):
    components = entry["components"]
    return tuple(record for name in ("one_day", "seven_day", "thirty_day") for record in components[name])


def _retained_inputs_from_artifact(jax: Any, artifact: Any, entry: Mapping[str, Any]) -> Mapping[Any, Any]:
    result = {}
    batches = {
        "one_day": artifact.batch.one_day.day_inputs,
        "seven_day": artifact.batch.seven_day.day_inputs,
        "thirty_day": artifact.batch.thirty_day.day_inputs,
    }
    for name, batch in batches.items():
        offset = 0
        for record in entry["components"][name]:
            identity = _record_identity(record)
            if identity in result:
                raise ValueError("reused retained inputs contain a repeated point-year")
            result[identity] = jax.tree_util.tree_map(
                lambda value: np.asarray(value[offset]),
                batch,
            )
            offset += len(record["start_rows"])
        sizes = {int(np.asarray(value).shape[0]) for value in jax.tree_util.tree_leaves(batch)}
        if sizes != {offset}:
            raise ValueError(f"reused retained {name} batch coverage drift")
    if len(result) != 28:
        raise ValueError("reused retained inputs must cover 28 point-years")
    return result


def _prepare_retained_inputs(
    runtime: Mapping[str, Any],
    reference: Any,
    record: Mapping[str, Any],
) -> tuple[Any, str, float]:
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

    started = time.perf_counter()
    paths = _plan_context(
        runtime["manifest"],
        landpoint_id=str(record["landpoint_id"]),
        year=int(record["year"]),
    )
    context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    parent = load_markov_shard(reference.parent.path, contract=runtime["contract"])
    sample = parent.sample(int(record["start_rows"][0]))
    retained = _retained_adapter(
        runtime | {"context_paths": paths},
        context,
        sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )
    if int(retained.static_owner.runtime_year) != int(record["year"]):
        raise ValueError("prepared retained owner used the wrong runtime year")
    return (
        runtime["jax"].tree_util.tree_map(np.asarray, retained.day_inputs),
        _retained_tail_trace_signature(retained),
        time.perf_counter() - started,
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args, backend_initializer=_initialize_cpu_backend)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        load_target_oriented_training_contract,
        prepare_target_oriented_training_batch,
    )

    contract = load_target_oriented_training_contract(args.training_contract)
    maximum = int(contract.optimizer["maximum_updates"])
    if not 1 <= int(args.updates) <= maximum:
        raise ValueError("target-oriented preparation update count is outside its budget")
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_target_oriented_training_schedule(
        references,
        contract,
        updates=int(args.updates),
    )
    reuse_manifest_path = getattr(args, "reuse_retained_inputs_manifest", None)
    reuse_manifest = None
    if reuse_manifest_path is not None:
        reuse_manifest_path = reuse_manifest_path.resolve()
        raw_reuse = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
        reuse_manifest = load_prepared_target_manifest(
            reuse_manifest_path,
            training_contract_sha256=contract.sha256,
            schedule_sha256=schedule["schedule_sha256"],
            updates=int(args.updates),
            source_git_head=str(raw_reuse.get("identity", {}).get("source_git_head", "")),
        )
    lookup = {_identity(reference): reference for reference in references}
    identity = {
        "source_git_head": _git_head(),
        "training_contract_sha256": contract.sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "dataset_manifest_sha256": contract.payload["identity"]["dataset_manifest_sha256"],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "prepared_manifest.json"
    if manifest_path.exists():
        progress = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema_version") != PREPARED_MANIFEST_SCHEMA
            or progress.get("identity") != identity
            or int(progress.get("updates", -1)) != int(args.updates)
        ):
            raise ValueError("existing target-oriented preparation has different identity")
    else:
        progress = {
            "schema_version": PREPARED_MANIFEST_SCHEMA,
            "status": "in_progress",
            "identity": identity,
            "updates": int(args.updates),
            "entries": [],
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
        }
        if reuse_manifest_path is not None:
            progress["retained_inputs_reused_from"] = str(reuse_manifest_path)
    completed = {int(record["update"]): record for record in progress["entries"]}
    retained_cache: dict[tuple[str, int, str, str], tuple[Any, str, float]] = {}
    started = time.perf_counter()

    for update, entry in enumerate(schedule["entries"], start=1):
        existing = completed.get(update)
        if existing is not None:
            load_prepared_target_artifact(
                manifest_path,
                existing,
                training_contract_sha256=contract.sha256,
                schedule_entry=entry,
            )
            print(f"target_prepared_update_reused update={update}", flush=True)
            continue
        entry_started = time.perf_counter()
        if reuse_manifest is not None:
            reused_record = reuse_manifest["entries"][update - 1]
            reused_artifact = load_prepared_target_artifact(
                reuse_manifest_path,
                reused_record,
                training_contract_sha256=contract.sha256,
                schedule_entry=entry,
            )
            day_inputs = _retained_inputs_from_artifact(jax, reused_artifact, entry)
            signatures = {reused_artifact.trace_signature}
            retained_seconds = 0.0
        else:
            day_inputs = {}
            signatures = set()
            retained_seconds = 0.0
            for record in _records(entry):
                key = _record_identity(record)
                reference = lookup.get(key)
                if reference is None:
                    raise ValueError("target-oriented schedule record has no coherent reference")
                if key not in retained_cache:
                    retained_cache[key] = _prepare_retained_inputs(runtime, reference, record)
                    seconds = retained_cache[key][2]
                else:
                    seconds = 0.0
                dynamic, signature, _original_seconds = retained_cache[key]
                day_inputs[key] = dynamic
                signatures.add(signature)
                retained_seconds += seconds
        if len(signatures) != 1:
            raise ValueError("target-oriented update crosses retained executable structures")
        batch_started = time.perf_counter()
        batch = prepare_target_oriented_training_batch(
            references,
            entry,
            adapter=runtime["adapter"],
            schema=runtime["schema"],
            owner_contract=runtime["owner_contract"],
            day_inputs_by_reference=day_inputs,
        )
        batch_seconds = time.perf_counter() - batch_started
        artifact = PreparedTargetOrientedUpdateArtifact(
            schema_version=PREPARED_ARTIFACT_SCHEMA,
            training_contract_sha256=contract.sha256,
            schedule_entry=dict(entry),
            trace_signature=next(iter(signatures)),
            batch=jax.tree_util.tree_map(np.asarray, batch),
        )
        artifact_path = output_dir / f"update-{update:04d}.pkl.gz"
        file_record = write_prepared_target_artifact(artifact_path, artifact)
        record = {
            "update": update,
            "runtime_year": int(entry["runtime_year"]),
            "trace_signature": artifact.trace_signature,
            **file_record,
            "retained_input_preparation_seconds": retained_seconds,
            "batch_preparation_seconds": batch_seconds,
            "total_seconds": time.perf_counter() - entry_started,
        }
        completed[update] = record
        progress["entries"] = [completed[index] for index in sorted(completed)]
        progress["elapsed_seconds_this_invocation"] = time.perf_counter() - started
        progress["total_size_bytes"] = sum(int(value["size_bytes"]) for value in progress["entries"])
        atomic_json(manifest_path, progress)
        print(
            f"target_prepared_update_written update={update} "
            f"size_bytes={record['size_bytes']} seconds={record['total_seconds']:.3f}",
            flush=True,
        )

    progress["status"] = "passed"
    progress["entries"] = [completed[index] for index in range(1, int(args.updates) + 1)]
    progress["trace_signatures"] = sorted({record["trace_signature"] for record in progress["entries"]})
    progress["total_size_bytes"] = sum(int(record["size_bytes"]) for record in progress["entries"])
    progress["elapsed_seconds_this_invocation"] = time.perf_counter() - started
    atomic_json(manifest_path, progress)
    return progress | {"manifest_sha256": sha256_file(manifest_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-retained-inputs-manifest", type=Path)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
