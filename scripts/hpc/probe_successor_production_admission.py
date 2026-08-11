"""Fail-fast CPU admission of every immutable successor production-fit asset."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    admit_prepared_manifest_for_checkpoint_fit,
    load_successor_checkpoint_handoff,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.successor_small_fit_prepared import (
    load_prepared_artifact,
    load_prepared_manifest,
    sha256_file,
)


def _cpu_backend() -> tuple[Any, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    devices = jax.devices("cpu")
    if not devices:
        raise ValueError("successor production admission found no CPU device")
    return jax, devices


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"successor production-admission output already exists: {args.output}")
    started = time.perf_counter()
    runtime = _load_extended_runtime(args, backend_initializer=_cpu_backend)

    from research.daily_coarse_graining.daily_operator.successor_frozen_coefficients import (
        load_frozen_successor_coefficients,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
    )

    runner_contract = load_successor_small_fit_contract(args.runner_contract)
    handoff = load_successor_checkpoint_handoff(args.checkpoint_handoff)
    updates = int(runner_contract.sampling["maximum_updates_per_arm"])
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
    producer = handoff.producer
    prepared = load_prepared_manifest(
        args.prepared_manifest,
        runner_contract_sha256=str(producer["producer_runner_contract_sha256"]),
        schedule_sha256=schedule["schedule_sha256"],
        updates=updates,
        source_git_head=str(producer["producer_git_head"]),
    )
    prepared_sha256 = sha256_file(args.prepared_manifest)
    artifact_admission = admit_prepared_manifest_for_checkpoint_fit(
        handoff,
        args.prepared_manifest,
        prepared,
        runner_contract_sha256=runner_contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        consumer_git_head=_git_head(),
    )
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        prepared["entries"][0],
        schedule_entry=schedule["entries"][0],
    )
    objective = runner_contract.payload["objective"]
    coefficients = load_frozen_successor_coefficients(
        args.coefficient_contract,
        source_report_path=args.coefficient_source_report,
        runner_contract_sha256=runner_contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        successor_contract_sha256=runtime["successor_contract"].sha256,
        fast_owner_contract_sha256=runtime["owner_contract"].contract_sha256,
        first_prepared_artifact_sha256=str(prepared["entries"][0]["sha256"]),
        components=LONG_HORIZON_COMPONENTS,
        gradient_ratio_targets=objective["gradient_ratio_targets"],
        coefficient_minimum=float(objective["coefficient_minimum"]),
        coefficient_maximum=float(objective["coefficient_maximum"]),
    )
    report = {
        "schema_version": "successor_production_fit_asset_admission_v1",
        "status": "passed",
        "git_head": _git_head(),
        "backend": runtime["jax"].default_backend(),
        "updates_per_arm": updates,
        "runner_contract_sha256": runner_contract.sha256,
        "checkpoint_handoff_contract_sha256": handoff.sha256,
        "prepared_manifest": str(args.prepared_manifest.resolve()),
        "prepared_manifest_sha256": prepared_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "first_prepared_artifact_sha256": prepared["entries"][0]["sha256"],
        "first_artifact_trace_signature": first_artifact.trace_signature,
        "frozen_coefficient_contract_sha256": coefficients.sha256,
        "frozen_coefficient_source_report_sha256": coefficients.payload["source_evidence"][
            "runner_report_sha256"
        ],
        "frozen_coefficient_source_schedule_sha256": coefficients.payload["identity"][
            "schedule_sha256"
        ],
        "resolved_coefficients": dict(coefficients.coefficients),
        "artifact_admission": dict(artifact_admission),
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "training_updates_executed": 0,
        "checkpoint_written": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--checkpoint-handoff", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

