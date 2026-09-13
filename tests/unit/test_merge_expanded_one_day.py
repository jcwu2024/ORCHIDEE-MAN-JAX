from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    PreparedTargetOrientedOneDayBatch,
    PreparedTargetOrientedTrainingBatch,
)
from scripts.hpc.merge_expanded_one_day import run
from scripts.hpc.target_oriented_training_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PreparedTargetOrientedUpdateArtifact,
    write_prepared_target_artifact,
)


class Inputs(NamedTuple):
    continuous_state: object


IDENTITY = {"training_contract_sha256": "contract-v6", "dataset": "immutable-v6"}


def _one_day(values):
    value = np.asarray(values, dtype=np.float64)
    return PreparedTargetOrientedOneDayBatch(
        inputs=Inputs(value[:, None]),
        process_targets={"process": value[:, None]},
        expected_next_state={"state": value[:, None]},
        expected_next_state_defined={"state": np.ones((len(value), 1), dtype=np.bool_)},
        expected_next_discrete_state={"flag": value.astype(np.int32)},
        day_inputs=np.asarray(value[:, None], dtype=np.float32),
    )


def _record(point, row, *, year=2000):
    return {
        "landpoint_id": point,
        "year": year,
        "parent_shard_sha256": f"parent-{point}",
        "typed_shard_sha256": f"typed-{point}",
        "start_rows": list(row) if isinstance(row, tuple) else [row],
    }


def _artifact(path: Path, update: int, schedule, values, *, trace="one_day_only_no_rollout_trace"):
    artifact = PreparedTargetOrientedUpdateArtifact(
        PREPARED_ARTIFACT_SCHEMA,
        IDENTITY["training_contract_sha256"],
        schedule,
        trace,
        PreparedTargetOrientedTrainingBatch(_one_day(values), None, None),
    )
    return write_prepared_target_artifact(path, artifact)


def _inputs(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_entries = []
    for update, point, rows, values in ((10, "a", (1, 2), (101, 102)), (11, "b", (3, 4), (203, 204))):
        schedule = {"update": update, "runtime_year": 2000, "components": {"one_day": [_record(point, rows)]}}
        metadata = _artifact(source_root / f"source-{update}.pkl.gz", update, schedule, values)
        source_entries.append({"update": update, **metadata})
    source_manifest = source_root / "prepared_manifest.json"
    source_manifest.write_text(json.dumps({"status": "passed", "identity": IDENTITY, "entries": source_entries}))

    catalogue = tmp_path / "base.catalogue.json.gz"
    with gzip.open(catalogue, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "source_manifest": str(source_manifest),
                "source_identity": IDENTITY,
                "monitor_rows": [
                    {"artifact_id": 10, "batch_row": 1, "landpoint_id": "a", "year": 2000, "start_row": 2},
                    {"artifact_id": 11, "batch_row": 1, "landpoint_id": "b", "year": 2000, "start_row": 4},
                ],
                "one_day_entries": [
                    {"update": 10, "runtime_year": 2000, "components": {"one_day": [_record("a", (1, 2))]}},
                    {"update": 11, "runtime_year": 2000, "components": {"one_day": [_record("b", (3, 4))]}},
                ],
            },
            handle,
        )
    entries = []
    for identifier, point, rows in ((1, "a", [1, 5]), (2, "b", [3, 6])):
        entries.append(
            {
                "id": identifier,
                "landpoint_id": point,
                "year": 2000,
                "parent_shard_sha256": f"parent-{point}",
                "typed_shard_sha256": f"typed-{point}",
                "start_rows": rows,
                "new_start_rows": [rows[-1]],
                "day_count": 8,
                "retained_source": {"artifact_id": 9 + identifier, "batch_row": 0, "reference_row": rows[0]},
            }
        )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": "one_day_expansion_plan_v1",
                "status": "planned",
                "seed": 20260913,
                "rows_per_point_year": 1,
                "catalogue": str(catalogue),
                "source_manifest": str(source_manifest),
                "source_identity": IDENTITY,
                "entries": entries,
                "counts": {"combined_unique_train_days": 4},
                "checks": {"new_disjoint_from_base_and_panels": True, "unique_new_rows": True},
            }
        )
    )
    output = tmp_path / "production"
    chunks = output / "chunks"
    worker_manifests = []
    for entry, values in zip(entries, ((101, 105), (203, 206)), strict=True):
        directory = chunks / f"worker-{entry['id']}"
        schedule = {
            "update": entry["id"],
            "runtime_year": entry["year"],
            "components": {"one_day": [_record(entry["landpoint_id"], tuple(entry["start_rows"]))]},
        }
        metadata = _artifact(directory / f"worker-{entry['id']}.pkl.gz", entry["id"], schedule, values)
        manifest = directory / "prepared_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "expanded_one_day_preparation_v1",
                    "status": "passed",
                    "stage": "completed",
                    "persisted_reader_exact": True,
                    "identity": {**IDENTITY, "expansion_seed": 20260913, "rows_per_point_year": 1},
                    "entry_ids": [entry["id"]],
                    "entries": [
                        {
                            "update": entry["id"],
                            "path": metadata["path"],
                            "rows": 2,
                            "new_rows": 1,
                            "schedule_entry": schedule,
                        }
                    ],
                }
            )
        )
        worker_manifests.append(manifest)
    return plan, chunks, output, worker_manifests


def _args(plan, chunks, output):
    return type(
        "Args",
        (),
        {
            "plan": plan,
            "chunks_root": chunks,
            "output_dir": output,
            "sampling_seed": 20260912,
            "batch_size": 4,
            "microbatch_size": 2,
            "max_artifacts_per_batch": 4,
            "expected_distinct_train_days": 4,
            "expected_monitor_days": 2,
        },
    )()


def test_merge_rewrites_worker_paths_and_copies_monitor_rows_for_existing_sampler(tmp_path):
    plan, chunks, output, _worker_manifests = _inputs(tmp_path)

    report = run(_args(plan, chunks, output))

    assert report["status"] == "passed"
    assert report["counts"] == {
        "distinct_train_days": 4,
        "train_row_draws": 4,
        "monitor_unique_days": 2,
        "monitor_row_draws": 2,
        "worker_artifacts": 2,
        "monitor_artifacts": 2,
        "epoch_batches": 1,
        "remainder_batches": 0,
        "batch_draw_count_histogram": {"4": 1},
        "microbatch_draw_count_histogram": {"2": 2},
    }
    assert report["rows_disjoint"] is True
    assert len({(row["landpoint_id"], row["year"]) for row in report["first_batch"]["rows"]}) == 2
    merged = json.loads(Path(report["prepared_manifest"]).read_text())
    assert [entry["update"] for entry in merged["entries"]] == [1, 2, 3, 4]
    assert all(not Path(entry["path"]).is_absolute() for entry in merged["entries"])
    with gzip.open(report["catalogue"], "rt", encoding="utf-8") as handle:
        catalogue = json.load(handle)
    assert {row["artifact_id"] for row in catalogue["monitor_rows"]} == {3, 4}
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool

    monitor = PreparedOneDayPool(Path(report["prepared_manifest"])).batch(
        [(row["artifact_id"], row["batch_row"]) for row in catalogue["monitor_rows"]]
    )
    np.testing.assert_array_equal(monitor.inputs.continuous_state[:, 0], [102, 204])
    assert monitor.day_inputs.dtype == np.float32
    assert monitor.expected_next_state_defined["state"].all()
    train_days = {
        (record["landpoint_id"], record["year"], row)
        for entry in catalogue["one_day_entries"][:2]
        for record in entry["components"]["one_day"]
        for row in record["start_rows"]
    }
    assert not train_days & {(row["landpoint_id"], row["year"], row["start_row"]) for row in catalogue["monitor_rows"]}


def test_merge_rejects_incomplete_worker_coverage(tmp_path):
    plan, chunks, output, workers = _inputs(tmp_path)
    workers[-1].unlink()

    with pytest.raises(ValueError, match="coverage"):
        run(_args(plan, chunks, output))
