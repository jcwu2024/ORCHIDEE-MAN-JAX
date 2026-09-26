"""Merge completed raw-v6 one-day chunks into one reader-compatible pool."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

MERGE_SCHEMA = "expanded_one_day_merge_v1"


def _resolved(path: str | Path, *, relative_to: Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = relative_to / value
    return value.resolve()


def _inside(path: Path, root: Path, *, name: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} lies outside the declared output root") from error
    return path


def _expected_schedule(entry: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        key: entry[key]
        for key in (
            "landpoint_id",
            "year",
            "parent_shard_sha256",
            "typed_shard_sha256",
            "start_rows",
        )
    }
    return {
        "update": int(entry["id"]),
        "runtime_year": int(entry["year"]),
        "components": {"one_day": [record]},
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _base_monitor_rows(catalogue: Mapping[str, Any]) -> dict[tuple[Any, int], Mapping[str, Any]]:
    entries = {entry["update"]: entry for entry in catalogue["one_day_entries"]}
    rows: dict[tuple[Any, int], Mapping[str, Any]] = {}
    for monitor in catalogue["monitor_rows"]:
        key = (monitor["artifact_id"], int(monitor["batch_row"]))
        if key in rows:
            raise ValueError("base catalogue monitor contains duplicate prepared rows")
        entry = entries.get(monitor["artifact_id"])
        if entry is None:
            raise ValueError("base catalogue monitor artifact is absent from one-day entries")
        flat = []
        for record in entry["components"]["one_day"]:
            flat.extend((record, int(start)) for start in record["start_rows"])
        if not 0 <= key[1] < len(flat):
            raise ValueError("base catalogue monitor batch row is outside its artifact")
        record, start = flat[key[1]]
        if (str(record["landpoint_id"]), int(record["year"]), start) != (
            str(monitor["landpoint_id"]), int(monitor["year"]), int(monitor["start_row"])
        ):
            raise ValueError("base catalogue monitor identity differs from its artifact row")
        rows[key] = {"monitor": monitor, "entry": entry, "record": record, "start_row": start}
    return rows


def _worker_records(plan: Mapping[str, Any], *, chunks_root: Path, output_root: Path) -> list[dict[str, Any]]:
    expected = {int(entry["id"]): entry for entry in plan["entries"]}
    if sorted(expected) != list(range(1, len(expected) + 1)):
        raise ValueError("expansion plan entry IDs must cover 1 through entry count")
    records: dict[int, dict[str, Any]] = {}
    manifests = sorted(chunks_root.rglob("prepared_manifest.json"))
    if not manifests:
        raise ValueError("expanded one-day chunks contain no worker manifests")
    for manifest_path in manifests:
        manifest_path = manifest_path.resolve()
        _inside(manifest_path, chunks_root, name="worker manifest")
        report = _read_json(manifest_path)
        if (
            report.get("schema_version") != "expanded_one_day_preparation_v1"
            or report.get("status") != "passed"
            or report.get("stage") != "completed"
            or report.get("persisted_reader_exact") is not True
        ):
            raise ValueError("expanded one-day worker report is not completed with exact readback")
        identity = report.get("identity", {})
        if (
            any(identity.get(key) != value for key, value in plan["source_identity"].items())
            or identity.get("expansion_seed") != plan["seed"]
            or identity.get("rows_per_point_year") != plan["rows_per_point_year"]
        ):
            raise ValueError("expanded one-day worker source identity differs from the plan")
        supplied = report.get("entries", [])
        reported_ids = [int(value) for value in report.get("entry_ids", [])]
        actual_ids = [int(record.get("update", -1)) for record in supplied]
        if reported_ids != actual_ids or len(set(actual_ids)) != len(actual_ids):
            raise ValueError("expanded one-day worker entry coverage drift")
        for record in supplied:
            identifier = int(record["update"])
            planned = expected.get(identifier)
            if planned is None or identifier in records:
                raise ValueError("expanded one-day worker entry is duplicate or absent from the plan")
            schedule = _expected_schedule(planned)
            if dict(record.get("schedule_entry", {})) != schedule:
                raise ValueError("expanded one-day worker schedule differs from the plan")
            if int(record.get("rows", -1)) != len(planned["start_rows"]) or int(
                record.get("new_rows", -1)
            ) != len(planned["new_start_rows"]):
                raise ValueError("expanded one-day worker row counts differ from the plan")
            payload = _resolved(str(record["path"]), relative_to=manifest_path.parent)
            _inside(payload, output_root, name="worker artifact")
            if not payload.is_file():
                raise ValueError("expanded one-day worker artifact is missing")
            records[identifier] = {
                "update": identifier,
                "path": payload.relative_to(output_root).as_posix(),
                "schedule_entry": schedule,
                "trace_signature": "one_day_only_no_rollout_trace",
                "source": "expanded_train",
            }
    if set(records) != set(expected):
        raise ValueError("expanded one-day worker coverage does not span the full plan")
    identities = {
        (
            value["landpoint_id"],
            int(value["year"]),
            value["parent_shard_sha256"],
            value["typed_shard_sha256"],
        )
        for value in expected.values()
    }
    if len(identities) != len(expected):
        raise ValueError("expanded one-day plan has overlapping source point-year identities")
    return [records[identifier] for identifier in sorted(records)]


def _write_monitor_artifacts(
    *,
    plan: Mapping[str, Any],
    catalogue: Mapping[str, Any],
    output_root: Path,
    source_manifest_path: Path,
    first_monitor_update: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        PreparedTargetOrientedTrainingBatch,
    )
    from scripts.hpc.target_oriented_training_prepared import (
        PREPARED_ARTIFACT_SCHEMA,
        PreparedTargetOrientedUpdateArtifact,
        write_prepared_target_artifact,
    )

    source = PreparedOneDayPool(source_manifest_path, cache_size=1)
    if source.identity != plan["source_identity"]:
        raise ValueError("base monitor source identity differs from the expansion plan")
    monitor_lookup = _base_monitor_rows(catalogue)
    groups: dict[Any, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for (artifact_id, batch_row), value in monitor_lookup.items():
        groups[artifact_id].append((batch_row, value))
    monitor_entries, remapped_rows = [], []
    for offset, source_id in enumerate(sorted(groups, key=lambda value: (str(type(value)), str(value)))):
        selected = sorted(groups[source_id], key=lambda item: item[0])
        update = first_monitor_update + offset
        source_rows = [(source_id, row) for row, _value in selected]
        one_day = source.batch(source_rows)
        source_batch, source_schedule = source.load(source_id)
        records = []
        for batch_row, value in selected:
            record = value["record"]
            records.append(
                {
                    "landpoint_id": record["landpoint_id"],
                    "year": int(record["year"]),
                    "parent_shard_sha256": record["parent_shard_sha256"],
                    "typed_shard_sha256": record["typed_shard_sha256"],
                    "start_rows": [value["start_row"]],
                }
            )
        runtime_year = int(source_schedule["runtime_year"])
        if any(int(record["year"]) != runtime_year for record in records):
            raise ValueError("base monitor source artifact crosses its runtime year")
        schedule = {"update": update, "runtime_year": runtime_year, "components": {"one_day": records}}
        artifact = PreparedTargetOrientedUpdateArtifact(
            PREPARED_ARTIFACT_SCHEMA,
            source.identity["training_contract_sha256"],
            schedule,
            "one_day_only_monitor_copy",
            PreparedTargetOrientedTrainingBatch(one_day, None, None),
        )
        directory = output_root / "monitor_chunks" / f"source-{source_id}"
        metadata = write_prepared_target_artifact(directory / f"monitor-{update:06d}.pkl.gz", artifact)
        monitor_entries.append(
            {
                "update": update,
                "path": (directory.relative_to(output_root) / metadata["path"]).as_posix(),
                "schedule_entry": schedule,
                "trace_signature": "one_day_only_monitor_copy",
                "source": "base_monitor_copy",
            }
        )
        for new_batch_row, (_old_batch_row, value) in enumerate(selected):
            monitor = value["monitor"]
            remapped_rows.append(
                {
                    "artifact_id": update,
                    "batch_row": new_batch_row,
                    "landpoint_id": str(monitor["landpoint_id"]),
                    "year": int(monitor["year"]),
                    "start_row": int(monitor["start_row"]),
                    "source_artifact_id": source_id,
                    "source_batch_row": int(monitor["batch_row"]),
                }
            )
    return monitor_entries, remapped_rows


def run(args):
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from scripts.hpc.run_fresh_start_one_day_pool import pool_plan
    from scripts.hpc.training_artifact_io import _atomic_json

    started = time.perf_counter()
    plan_path = args.plan.resolve()
    output_root = args.output_dir.resolve()
    chunks_root = (args.chunks_root or output_root / "chunks").resolve()
    if output_root in (plan_path.parent, plan_path):
        raise ValueError("output directory must be a new production root")
    _inside(chunks_root, output_root, name="chunks root")
    combined_manifest = output_root / "prepared_manifest.json"
    report_path = output_root / "merge_report.json"
    catalogue_path = output_root / "prepared_one_day_catalogue.json.gz"
    if any(path.exists() for path in (combined_manifest, report_path, catalogue_path)):
        raise FileExistsError("expanded one-day merge output already exists")
    plan = _read_json(plan_path)
    if (
        plan.get("schema_version") != "one_day_expansion_plan_v1"
        or plan.get("status") != "planned"
        or not all(plan.get("checks", {}).values())
    ):
        raise ValueError("expanded one-day plan is not valid")
    base_catalogue_path = _resolved(plan["catalogue"], relative_to=plan_path.parent)
    source_manifest_path = _resolved(plan["source_manifest"], relative_to=plan_path.parent)
    with gzip.open(base_catalogue_path, "rt", encoding="utf-8") as handle:
        base_catalogue = json.load(handle)
    if _resolved(base_catalogue.get("source_manifest", ""), relative_to=base_catalogue_path.parent) != source_manifest_path or base_catalogue.get(
        "source_identity"
    ) != plan["source_identity"]:
        raise ValueError("base catalogue differs from the declared expansion source")

    output_root.mkdir(parents=True, exist_ok=True)
    worker_entries = _worker_records(plan, chunks_root=chunks_root, output_root=output_root)
    monitor_entries, monitor_rows = _write_monitor_artifacts(
        plan=plan,
        catalogue=base_catalogue,
        output_root=output_root,
        source_manifest_path=source_manifest_path,
        first_monitor_update=len(plan["entries"]) + 1,
    )
    monitor_days = {(row["landpoint_id"], row["year"], row["start_row"]) for row in monitor_rows}
    if len(monitor_days) != len(monitor_rows):
        raise ValueError("base catalogue monitor days are not unique")
    if len(monitor_rows) != int(args.expected_monitor_days):
        raise ValueError("expanded one-day monitor count differs from the declared expectation")

    all_entries = worker_entries + monitor_entries
    identity = {
        **plan["source_identity"],
        "source_manifest": str(source_manifest_path),
        "expansion_plan": str(plan_path),
        "expansion_seed": plan["seed"],
        "rows_per_point_year": plan["rows_per_point_year"],
    }
    manifest = {
        "schema_version": "target_oriented_prepared_manifest_v1",
        "status": "passed",
        "updates": len(all_entries),
        "identity": identity,
        "entries": all_entries,
        "teacher_regenerated": False,
        "training_updates": 0,
        "merged_from_complete_expansion": True,
    }
    _atomic_json(combined_manifest, manifest)
    catalogue = {
        "source_manifest": str(combined_manifest),
        "source_identity": identity,
        "seed": int(args.sampling_seed),
        "batch_size": int(args.batch_size),
        "microbatch_size": int(args.microbatch_size),
        "max_artifacts_per_batch": int(args.max_artifacts_per_batch),
        "monitor_rows": monitor_rows,
        "one_day_entries": [
            {
                "update": entry["schedule_entry"]["update"],
                "runtime_year": entry["schedule_entry"]["runtime_year"],
                "components": entry["schedule_entry"]["components"],
            }
            for entry in all_entries
        ],
    }
    with gzip.open(catalogue_path, "wt", encoding="utf-8") as handle:
        json.dump(catalogue, handle, separators=(",", ":"))

    configuration = {
        "sampling": {
            "seed": int(args.sampling_seed),
            "batch_size": int(args.batch_size),
            "microbatch_size": int(args.microbatch_size),
            "max_artifacts_per_batch": int(args.max_artifacts_per_batch),
        }
    }
    sampler, panels = pool_plan(catalogue, configuration)
    batches = list(sampler.iter_epoch(0))
    training_days = {
        (row.landpoint_id, row.year, row.start_row) for batch in batches for row in batch.rows
    }
    if training_days & monitor_days or len(training_days) != sampler.unique_day_count:
        raise ValueError("expanded one-day train and monitor rows are not disjoint")
    if len(training_days) != int(plan["counts"]["combined_unique_train_days"]):
        raise ValueError("expanded one-day merged train days differ from the plan")
    if len(training_days) != int(args.expected_distinct_train_days):
        raise ValueError("expanded one-day distinct train-day count differs from the declared expectation")
    first = batches[0]
    if first.draw_count != int(args.batch_size):
        raise ValueError("expanded one-day first batch is not full")
    first_point_years = {(row.landpoint_id, row.year) for row in first.rows}
    if len(first_point_years) < 2:
        raise ValueError("expanded one-day first batch does not cross point-years")
    pool = PreparedOneDayPool(combined_manifest, cache_size=int(args.max_artifacts_per_batch))
    readback = pool.batch([(row.artifact_id, row.batch_row) for row in first.rows])
    if len(readback.inputs.continuous_state) != first.draw_count:
        raise ValueError("expanded one-day reader first-batch size drift")
    batch_histogram: dict[str, int] = defaultdict(int)
    microbatch_histogram: dict[str, int] = defaultdict(int)
    for batch in batches:
        batch_histogram[str(batch.draw_count)] += 1
        for microbatch in batch.microbatches:
            microbatch_histogram[str(microbatch.draw_count)] += 1
    counts = {
        "distinct_train_days": len(training_days),
        "train_row_draws": sampler.draw_count,
        "monitor_unique_days": len(monitor_days),
        "monitor_row_draws": len(monitor_rows),
        "worker_artifacts": len(worker_entries),
        "monitor_artifacts": len(monitor_entries),
        "epoch_batches": len(batches),
        "remainder_batches": sum(batch.draw_count != int(args.batch_size) for batch in batches),
        "batch_draw_count_histogram": dict(sorted(batch_histogram.items(), key=lambda item: int(item[0]))),
        "microbatch_draw_count_histogram": dict(
            sorted(microbatch_histogram.items(), key=lambda item: int(item[0]))
        ),
    }
    report = {
        "schema_version": MERGE_SCHEMA,
        "status": "passed",
        "stage": "completed",
        "plan": str(plan_path),
        "prepared_manifest": str(combined_manifest),
        "catalogue": str(catalogue_path),
        "source_identity": identity,
        "counts": counts,
        "rows_disjoint": True,
        "first_batch": first.as_metadata(),
        "panels": panels,
        "reader": pool.statistics(),
        "teacher_regenerated": False,
        "training_updates": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(report_path, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--chunks-root", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--sampling-seed", type=int, default=20260912)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--microbatch-size", type=int, default=4)
    result.add_argument("--max-artifacts-per-batch", type=int, default=4)
    result.add_argument("--expected-distinct-train-days", type=int, default=346539)
    result.add_argument("--expected-monitor-days", type=int, default=64)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
