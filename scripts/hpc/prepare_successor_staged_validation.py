"""Prepare the frozen successor milestone-validation panel on CPU."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_rollout_screen import (
    prepare_successor_rollout_screen_batch,
)
from scripts.hpc.prepare_successor_rollout_screen import _prepare_adapter
from scripts.hpc.prepare_successor_small_fit_batches import _initialize_cpu_backend
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_small_fit import _git_head, _retained_tail_trace_signature
from scripts.hpc.successor_rollout_screen_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PreparedRolloutScreenArtifact,
    atomic_json,
    write_prepared_rollout_artifact,
)
from scripts.hpc.successor_staged_validation_prepared import (
    STAGED_VALIDATION_PREPARED_SCHEMA,
    ordered_validation_entries,
)


def _reference(index: Any, slice_id: str, entry: Mapping[str, Any]) -> Any:
    split = {
        "temporal": ("train", "validation"),
        "spatial": ("validation", "train"),
        "joint": ("validation", "validation"),
    }[slice_id]
    matches = tuple(
        reference
        for reference in index.select(
            spatial_split=split[0],
            temporal_split=split[1],
        )
        if str(reference.parent.landpoint_id) == str(entry["landpoint_id"])
        and int(reference.parent.year) == int(entry["year"])
    )
    if len(matches) != 1:
        raise ValueError("successor staged validation shard identity is ambiguous")
    reference = matches[0]
    if (
        str(reference.parent.sha256) != entry["parent_shard_sha256"]
        or str(reference.sha256) != entry["typed_shard_sha256"]
        or int(reference.day_count) != int(entry["day_count"])
    ):
        raise ValueError("successor staged validation shard identity drift")
    return reference


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "prepared_manifest.json"
    panel_copy = output_dir / "validation_panel.json"
    if manifest_path.exists() or panel_copy.exists():
        raise FileExistsError("successor staged validation output already exists")
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        STAGED_SCHEMA_VERSION,
        load_successor_small_fit_contract,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        _canonical_sha256,
        _sha256_file,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if contract.payload["schema_version"] != STAGED_SCHEMA_VERSION:
        raise ValueError("staged validation requires the staged-v3 contract")
    panel_path = args.validation_panel.resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if (
        panel.get("panel_sha256") != _canonical_sha256(panel)
        or panel.get("runner_contract_sha256") != contract.sha256
        or panel.get("sealed_test_used") is not False
    ):
        raise ValueError("successor staged validation panel drift")
    runtime = _load_extended_runtime(args, backend_initializer=_initialize_cpu_backend)
    jax = runtime["jax"]
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(panel_path, panel_copy)
    records = []
    for ordinal, (slice_id, entry) in enumerate(
        ordered_validation_entries(panel),
        start=1,
    ):
        reference = _reference(runtime["index"], slice_id, entry)
        adapter = _prepare_adapter(runtime, reference, int(entry["start_rows"][0]))
        batch = prepare_successor_rollout_screen_batch(
            reference,
            entry,
            adapter=runtime["adapter"],
            rollout_days=int(panel["rollout_days"]),
        )
        artifact = PreparedRolloutScreenArtifact(
            PREPARED_ARTIFACT_SCHEMA,
            slice_id,
            dict(entry),
            _retained_tail_trace_signature(adapter),
            jax.tree_util.tree_map(np.asarray, batch),
            jax.tree_util.tree_map(np.asarray, adapter.day_inputs),
        )
        path = output_dir / f"validation-{ordinal:03d}.pkl"
        records.append(
            {
                "ordinal": ordinal,
                "slice_id": slice_id,
                "window_entry": dict(entry),
                "trace_signature": artifact.trace_signature,
                **write_prepared_rollout_artifact(path, artifact),
            }
        )
    payload = {
        "schema_version": STAGED_VALIDATION_PREPARED_SCHEMA,
        "status": "passed",
        "source_git_head": _git_head(),
        "staged_training_contract_sha256": contract.sha256,
        "validation_panel": panel_copy.name,
        "validation_panel_sha256": panel["panel_sha256"],
        "validation_panel_file_sha256": _sha256_file(panel_copy),
        "artifact_count": len(records),
        "window_count": sum(
            len(entry["start_rows"])
            for _slice_id, entry in ordered_validation_entries(panel)
        ),
        "entries": records,
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "elapsed_seconds": time.perf_counter() - started,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
    }
    atomic_json(manifest_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
