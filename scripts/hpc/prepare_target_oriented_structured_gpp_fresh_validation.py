"""Prepare the fresh structured-GPP 7/30-day panel on a free CPU node."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.successor_rollout_screen import (
    prepare_successor_rollout_screen_batch,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_screen import (
    build_fresh_validation_panel,
    load_structured_gpp_fresh_screen_contract,
    verify_fresh_validation_panel,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    ordered_target_oriented_validation_entries,
)
from scripts.hpc.prepare_successor_rollout_screen import _prepare_adapter
from scripts.hpc.prepare_successor_small_fit_batches import _initialize_cpu_backend
from scripts.hpc.prepare_target_oriented_validation import _reference
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_small_fit import _retained_tail_trace_signature
from scripts.hpc.successor_rollout_screen_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PreparedRolloutScreenArtifact,
    atomic_json,
    load_prepared_rollout_artifact,
    write_prepared_rollout_artifact,
)
from scripts.hpc.target_oriented_validation_prepared import PREPARED_SCHEMA


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args, backend_initializer=_initialize_cpu_backend)
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        load_target_oriented_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        load_target_oriented_training_contract,
    )

    training_contract = load_target_oriented_training_contract(args.training_contract)
    target_contract = load_target_oriented_contract()
    screen_contract = load_structured_gpp_fresh_screen_contract(args.screen_contract)
    consumed_panel = json.loads(
        args.consumed_validation_panel.read_text(encoding="utf-8")
    )
    panel = build_fresh_validation_panel(
        runtime["index"],
        screen_contract,
        target_contract,
        consumed_panel,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "validation_panel.json"
    manifest_path = output_dir / "prepared_manifest.json"
    if panel_path.exists():
        existing_panel = json.loads(panel_path.read_text(encoding="utf-8"))
        verify_fresh_validation_panel(
            existing_panel,
            runtime["index"],
            screen_contract,
            target_contract,
            consumed_panel,
        )
        if existing_panel != panel:
            raise ValueError("fresh validation panel changed during resume")
    else:
        atomic_json(panel_path, panel)

    identity = {
        "source_git_head": _git_head(),
        "screen_contract_sha256": screen_contract.sha256,
        "training_contract_sha256": training_contract.sha256,
        "target_contract_sha256": target_contract.sha256,
        "validation_panel_sha256": panel["panel_sha256"],
        "consumed_validation_panel_sha256": consumed_panel["panel_sha256"],
    }
    if manifest_path.exists():
        progress = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            progress.get("identity") != identity
            or progress.get("schema_version") != PREPARED_SCHEMA
        ):
            raise ValueError("existing fresh validation preparation has different identity")
    else:
        progress = {
            "schema_version": PREPARED_SCHEMA,
            "status": "in_progress",
            "identity": identity,
            "source_git_head": identity["source_git_head"],
            "screen_contract_sha256": screen_contract.sha256,
            "training_contract_sha256": training_contract.sha256,
            "target_contract_sha256": target_contract.sha256,
            "validation_panel": panel_path.name,
            "validation_panel_sha256": panel["panel_sha256"],
            "validation_panel_file_sha256": sha256_file(panel_path),
            "entries": [],
            "target_values_read_during_selection": False,
            "sealed_test_used": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
        }

    completed = {int(record["ordinal"]): record for record in progress["entries"]}
    ordered = ordered_target_oriented_validation_entries(panel)
    started = time.perf_counter()
    for ordinal, (slice_id, horizon, entry) in enumerate(ordered, start=1):
        existing = completed.get(ordinal)
        if existing is not None:
            load_prepared_rollout_artifact(
                manifest_path,
                existing,
                slice_id=slice_id,
                window_entry=entry,
            )
            print(f"fresh_validation_prepared_reused ordinal={ordinal}", flush=True)
            continue
        entry_started = time.perf_counter()
        reference = _reference(runtime["index"], slice_id, entry)
        batch = prepare_successor_rollout_screen_batch(
            reference,
            entry,
            adapter=runtime["adapter"],
            rollout_days=horizon,
        )
        retained = _prepare_adapter(runtime, reference, int(entry["start_rows"][0]))
        artifact = PreparedRolloutScreenArtifact(
            PREPARED_ARTIFACT_SCHEMA,
            slice_id,
            dict(entry),
            _retained_tail_trace_signature(retained),
            runtime["jax"].tree_util.tree_map(np.asarray, batch),
            runtime["jax"].tree_util.tree_map(np.asarray, retained.day_inputs),
        )
        path = output_dir / f"validation-{ordinal:03d}.pkl"
        if path.exists():
            file_record = {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            provisional = {
                "ordinal": ordinal,
                "slice_id": slice_id,
                "horizon_days": horizon,
                "window_entry": dict(entry),
                "trace_signature": artifact.trace_signature,
                **file_record,
            }
            load_prepared_rollout_artifact(
                manifest_path,
                provisional,
                slice_id=slice_id,
                window_entry=entry,
            )
        else:
            file_record = write_prepared_rollout_artifact(path, artifact)
        completed[ordinal] = {
            "ordinal": ordinal,
            "slice_id": slice_id,
            "horizon_days": horizon,
            "window_entry": dict(entry),
            "trace_signature": artifact.trace_signature,
            "preparation_seconds": time.perf_counter() - entry_started,
            **file_record,
        }
        progress["entries"] = [completed[index] for index in sorted(completed)]
        progress["artifact_count"] = len(progress["entries"])
        progress["total_size_bytes"] = sum(
            int(record["size_bytes"]) for record in progress["entries"]
        )
        progress["elapsed_seconds_this_invocation"] = time.perf_counter() - started
        atomic_json(manifest_path, progress)
        print(
            f"fresh_validation_prepared_written ordinal={ordinal} "
            f"slice={slice_id} horizon={horizon} "
            f"seconds={completed[ordinal]['preparation_seconds']:.3f}",
            flush=True,
        )

    progress["status"] = "passed"
    progress["entries"] = [completed[index] for index in range(1, len(ordered) + 1)]
    progress["artifact_count"] = len(ordered)
    progress["window_count"] = len(ordered)
    progress["total_size_bytes"] = sum(
        int(record["size_bytes"]) for record in progress["entries"]
    )
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
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--consumed-validation-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
