"""Audit bounded direct common-direction scales at coupled-GPP update 5."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_update5_scale_audit import (
    DIRECT_RECALIBRATION_RESULT_SHA256,
    load_coupled_gpp_update5_scale_audit_contract,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _parser as _smoke_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    run as _run_smoke,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import _evaluate_scales

SCHEMA_VERSION = "target_oriented_coupled_gpp_update5_scale_audit_result_v1"


def _load_smoke_result(path: Path, contract: Any) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report.get("smoke", {}).get("records", ())
    if (
        sha256_file(path) != DIRECT_RECALIBRATION_RESULT_SHA256
        or sha256_file(path) != contract.payload["identity"]["direct_recalibration_result_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_direct_recalibration_smoke_result_v1"
        or report.get("status") != "failed_scientific_gate"
        or report.get("classification") != "common_direction_finite_step_rejection"
        or report.get("identity", {}).get("recalibration_contract_sha256")
        != contract.payload["identity"]["direct_recalibration_contract_sha256"]
        or report.get("smoke", {}).get("updates_accepted") != [3, 4]
        or report.get("smoke", {}).get("updates_evaluated") != [3, 4, 5]
        or len(records) != 3
        or records[-1].get("common_first_order_direction") is not True
        or records[-1].get("finite_step_accepted") is not False
        or report.get("policy", {}).get("checkpoint_written") is not False
        or report.get("integrity", {}).get("parent_recovery_unchanged") is not True
    ):
        raise ValueError("coupled GPP direct recalibration result drift")
    return report


def run(args: Any) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_coupled_gpp_update5_scale_audit_contract(args.scale_audit_contract)
    predecessor = _load_smoke_result(args.smoke_result, contract)
    recovery_before = args.parent_recovery.stat()

    reconstructed, internal = _run_smoke(args, write_output=False, return_internal=True)
    if (
        reconstructed["status"] != "failed_scientific_gate"
        or reconstructed["classification"] != "common_direction_finite_step_rejection"
        or reconstructed["smoke"]["updates_accepted"] != [3, 4]
        or reconstructed["smoke"]["updates_evaluated"] != [3, 4, 5]
        or int(internal["macro_update"]) != 5
    ):
        raise RuntimeError("coupled GPP update-5 scale reconstruction drift")

    scales = _evaluate_scales(
        args,
        internal["context"],
        initialized_parameters=internal["initialized_parameters"],
        before_trainable=internal["before_trainable"],
        proposed_trainable=internal["proposed_trainable"],
        before_losses=internal["before_losses"],
        scales=contract.scales,
        scale_one_evaluation=internal["full_scale_evaluation"],
        steps=internal["steps"],
        direct_steps=internal["direct_steps"],
        state_steps=internal["state_steps"],
        layout=internal["layout"],
        group=internal["group"],
        flux_basis=internal["flux_basis"],
        coefficients=internal["coefficients"],
        relative_tolerance=internal["relative_tolerance"],
        label="update5_common_direct",
        direction_norm=internal["direction_norm"],
    )
    accepted_scales = [float(record["scale"]) for record in scales if bool(record["finite_step_accepted"])]
    passed = bool(accepted_scales)
    classification = "bounded_direct_finite_step" if passed else "no_bounded_direct_finite_step"

    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    smoke_result_unchanged = sha256_file(args.smoke_result) == DIRECT_RECALIBRATION_RESULT_SHA256
    if not recovery_unchanged or not smoke_result_unchanged:
        raise RuntimeError("coupled GPP update-5 scale retention boundary failed")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed_scientific_gate",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "scale_audit_contract_sha256": contract.sha256,
            "direct_recalibration_contract_sha256": reconstructed["identity"]["recalibration_contract_sha256"],
            "direct_recalibration_result_sha256": sha256_file(args.smoke_result),
            "schedule_sha256": reconstructed["identity"]["schedule_sha256"],
        },
        "policy": {
            "optimizer_updates_persisted": 0,
            "parameter_proposals_persisted": 0,
            "checkpoint_written": False,
            "parent_recovery_modified": False,
            "fresh_validation_values_evaluated": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "reconstruction": {
            "predecessor_classification": predecessor["classification"],
            "current_classification": reconstructed["classification"],
            "updates_accepted": reconstructed["smoke"]["updates_accepted"],
            "updates_evaluated": reconstructed["smoke"]["updates_evaluated"],
            "numeric_differences_reporting_only": True,
        },
        "update_5": {
            "accepted_scales": accepted_scales,
            "scales": scales,
        },
        "integrity": {
            "parent_recovery_unchanged": recovery_unchanged,
            "direct_recalibration_result_unchanged": smoke_result_unchanged,
            "reconstructed_integrity": reconstructed["integrity"],
        },
        "memory_after": _memory_report(internal["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"]["on_bounded_finite_step" if passed else "on_no_bounded_finite_step"],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _smoke_parser()
    parser.add_argument("--scale-audit-contract", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
