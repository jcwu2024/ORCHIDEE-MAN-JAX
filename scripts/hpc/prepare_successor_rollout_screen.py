"""Prepare deterministic successor 7/30-day validation windows on a free CPU node."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_rollout_screen import (
    build_successor_screen_window_plan,
    load_successor_rollout_screen_contract,
    prepare_successor_rollout_screen_batch,
    reference_for_screen_entry,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _retained_adapter
from scripts.hpc.run_successor_small_fit import _retained_tail_trace_signature
from scripts.hpc.run_successor_small_fit_probe import _plan_context
from scripts.hpc.successor_rollout_screen_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedRolloutScreenArtifact,
    atomic_json,
    write_prepared_rollout_artifact,
)


def _cpu_backend() -> tuple[Any, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    devices = jax.devices("cpu")
    if not devices:
        raise ValueError("successor rollout preparation found no CPU device")
    return jax, devices


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _prepare_adapter(runtime: Mapping[str, Any], reference: Any, start: int) -> Any:
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

    parent = load_markov_shard(reference.parent.path, contract=runtime["contract"])
    sample = parent.sample(start)
    paths = _plan_context(
        runtime["manifest"],
        landpoint_id=str(reference.parent.landpoint_id),
        year=int(reference.parent.year),
    )
    context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    return _retained_adapter(
        runtime | {"context_paths": paths},
        context,
        sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "prepared_manifest.json"
    plan_path = output_dir / "window_plan.json"
    if manifest_path.exists() or plan_path.exists():
        raise FileExistsError("successor rollout prepared output already exists")
    started = time.perf_counter()
    contract = load_successor_rollout_screen_contract(args.screen_contract)
    runtime = _load_extended_runtime(args, backend_initializer=_cpu_backend)
    plan = build_successor_screen_window_plan(
        runtime["index"],
        contract,
        mode=args.mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(plan_path, plan)
    records = []
    ordinal = 0
    for slice_id, slice_plan in plan["slices"].items():
        for entry in slice_plan["entries"]:
            ordinal += 1
            entry_started = time.perf_counter()
            reference = reference_for_screen_entry(
                runtime["index"],
                contract,
                slice_id=slice_id,
                entry=entry,
            )
            batch = prepare_successor_rollout_screen_batch(
                reference,
                entry,
                adapter=runtime["adapter"],
                rollout_days=int(plan["rollout_days"]),
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
            path = output_dir / f"screen-{ordinal:03d}.pkl"
            file_record = write_prepared_rollout_artifact(path, artifact)
            records.append(
                {
                    "ordinal": ordinal,
                    "slice_id": slice_id,
                    "window_entry": dict(entry),
                    "trace_signature": artifact.trace_signature,
                    "preparation_seconds": time.perf_counter() - entry_started,
                    **file_record,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "successor_rollout_screen_prepared",
                        "ordinal": ordinal,
                        "slice_id": slice_id,
                        "landpoint_id": entry["landpoint_id"],
                        "year": entry["year"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    report = {
        "schema_version": PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "source_git_head": _git_head(),
        "screen_contract_sha256": contract.sha256,
        "window_plan": str(plan_path),
        "window_plan_sha256": plan["plan_sha256"],
        "mode": args.mode,
        "artifact_count": len(records),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in records),
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "backend": runtime["jax"].default_backend(),
        "elapsed_seconds": time.perf_counter() - started,
        "entries": records,
    }
    atomic_json(manifest_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("free_smoke", "formal_bounded_screen"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
