"""Freeze successor loss coefficients and milestone-validation windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.successor_small_fit import (
    build_successor_validation_panel,
    load_successor_small_fit_contract,
)
from research.daily_coarse_graining.daily_operator.successor_training import (
    LONG_HORIZON_COMPONENTS,
)
from scripts.hpc.probe_successor_multibatch_coefficients import (
    _aggregate_coefficient_records,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.successor_small_fit_prepared import atomic_json


def _cpu_backend() -> tuple[Any, tuple[Any, ...]]:
    import jax

    jax.config.update("jax_platforms", "cpu")
    devices = tuple(jax.devices("cpu"))
    if not devices or jax.default_backend() != "cpu":
        raise RuntimeError("successor asset freeze requires a CPU backend")
    return jax, devices


def _freeze_coefficients(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    contract: Any,
    prepared_manifest_path: Path,
) -> Mapping[str, Any]:
    objective = contract.payload["objective"]
    probe_contract_sha256 = contract.payload["identity"][
        "coefficient_probe_runner_contract_sha256"
    ]
    measurements = report.get("measurements", ())
    if (
        report.get("status") != "passed"
        or report.get("training_updates") != 0
        or report.get("checkpoint_written") is not False
        or report.get("sealed_test_used") is not False
        or report.get("historical_v5_used") is not False
        or report.get("teacher_data_regenerated") is not False
        or report.get("hard_failure_count") != 0
        or report.get("runner_contract_sha256") != probe_contract_sha256
        or int(report.get("measurement_record_count", 0)) != 32
        or len(measurements) != 32
        or report.get("prepared_manifest_sha256")
        != sha256_file(prepared_manifest_path)
    ):
        raise ValueError("successor coefficient report is not freeze eligible")
    aggregation = _aggregate_coefficient_records(
        measurements,
        component_names=LONG_HORIZON_COMPONENTS,
        targets=objective["gradient_ratio_targets"],
        coefficient_minimum=float(objective["coefficient_minimum"]),
        coefficient_maximum=float(objective["coefficient_maximum"]),
        coefficient_aggregation=objective["coefficient_aggregation"],
        coefficient_quantile=float(objective["coefficient_quantile"]),
    )
    prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    coefficients = {
        name: float(aggregation["resolved_coefficients"][name])
        for name in LONG_HORIZON_COMPONENTS
    }
    if set(aggregation["resolved_coefficients"]) != set(LONG_HORIZON_COMPONENTS):
        raise ValueError("successor coefficient registry drift")
    payload = {
        "schema_version": "successor_training_frozen_coefficients_v2",
        "status": "accepted",
        "identity": {
            "staged_training_contract_sha256": contract.sha256,
            "coefficient_probe_runner_contract_sha256": probe_contract_sha256,
            "successor_candidate_contract_sha256": contract.payload["identity"][
                "successor_candidate_contract_sha256"
            ],
            "fast_owner_contract_sha256": contract.payload["identity"][
                "fast_owner_contract_sha256"
            ],
            "prepared_manifest_sha256": sha256_file(prepared_manifest_path),
            "prepared_schedule_sha256": prepared["identity"]["schedule_sha256"],
        },
        "source_evidence": {
            "git_head": report["git_head"],
            "probe_report_sha256": sha256_file(report_path),
            "record_count": aggregation["record_count"],
            "hard_failure_count": report["hard_failure_count"],
            "reported_coefficient_aggregation": report[
                "coefficient_aggregation"
            ],
        },
        "objective": {
            "components": list(LONG_HORIZON_COMPONENTS),
            "split_label": "train/train",
            "gradient_ratio_targets": objective["gradient_ratio_targets"],
            "coefficient_minimum": objective["coefficient_minimum"],
            "coefficient_maximum": objective["coefficient_maximum"],
            "coefficient_aggregation": objective["coefficient_aggregation"],
            "coefficient_quantile": objective["coefficient_quantile"],
            "quantile_method": "lower",
            "resolved_coefficients": coefficients,
            "same_coefficients_for_both_arms": True,
            "recalibrate_during_fit": False,
            "validation_weight_tuning_allowed": False,
        },
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    import hashlib

    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    contract = load_successor_small_fit_contract(args.runner_contract)
    report_path = args.coefficient_report.resolve()
    prepared_manifest_path = args.prepared_manifest.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    coefficients = _freeze_coefficients(
        report,
        report_path=report_path,
        contract=contract,
        prepared_manifest_path=prepared_manifest_path,
    )
    runtime = _load_extended_runtime(args, backend_initializer=_cpu_backend)
    excluded_path = args.excluded_window_plan.resolve()
    excluded = json.loads(excluded_path.read_text(encoding="utf-8"))
    if (
        sha256_file(excluded_path)
        != contract.payload["validation"][
            "excluded_formal_window_plan_file_sha256"
        ]
    ):
        raise ValueError("excluded formal window-plan file hash drift")
    validation = build_successor_validation_panel(
        runtime["index"].shards,
        contract,
        excluded_window_plan=excluded,
    )
    atomic_json(args.coefficient_output, coefficients)
    atomic_json(args.validation_output, validation)
    return {
        "status": "passed",
        "coefficient_contract_sha256": coefficients["contract_sha256"],
        "validation_panel_sha256": validation["panel_sha256"],
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--coefficient-report", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--excluded-window-plan", type=Path, required=True)
    parser.add_argument("--coefficient-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
