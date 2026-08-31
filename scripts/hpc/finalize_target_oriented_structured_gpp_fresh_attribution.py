"""Finalize the hash-bound fresh GPP attribution without numerical re-execution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_attribution import (
    StructuredGPPFreshAttributionContract,
    all_numeric_values_finite,
    classify_fresh_attribution,
    formal_metric_difference,
    load_structured_gpp_fresh_attribution_contract,
)

RAW_SCHEMA_VERSION = "target_oriented_structured_gpp_fresh_attribution_result_v1"
SCHEMA_VERSION = "target_oriented_structured_gpp_fresh_attribution_finalized_result_v1"
EXPECTED_RAW_REPORT_SHA256 = "07031dbf8b61553dadc631214d1a2eba1a3a329487d10517580a897fee615d4e"
EXPECTED_NUMERICAL_EXECUTION_HEAD = "5dad8999aa2444bb6eac4bc71fee1993822015de"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def finalize_report(
    raw_report: Mapping[str, Any],
    formal_report: Mapping[str, Any],
    contract: StructuredGPPFreshAttributionContract,
    *,
    raw_report_sha256: str,
    formal_report_sha256: str,
    finalizer_git_head: str,
    expected_raw_report_sha256: str = EXPECTED_RAW_REPORT_SHA256,
) -> Mapping[str, Any]:
    identity = contract.payload["identity"]
    raw_identity = raw_report.get("identity", {})
    if (
        raw_report_sha256 != expected_raw_report_sha256
        or formal_report_sha256 != identity["fresh_formal_report_sha256"]
        or raw_report.get("schema_version") != RAW_SCHEMA_VERSION
        or raw_report.get("mode") != "formal"
        or int(raw_report.get("attribution", {}).get("evaluated_windows", -1)) != 96
        or raw_identity.get("git_head") != EXPECTED_NUMERICAL_EXECUTION_HEAD
        or raw_identity.get("attribution_contract_sha256") != contract.sha256
        or raw_identity.get("fresh_formal_report_sha256") != formal_report_sha256
    ):
        raise ValueError("fresh-attribution raw report identity drift")

    result = copy.deepcopy(raw_report)
    attribution = result["attribution"]
    metric_difference = formal_metric_difference(attribution["ordinary_free"], formal_report["panel"])
    classification = classify_fresh_attribution(
        attribution["teacher_day_start_direct_gpp"],
        attribution["gpp_residual_decomposition"]["cumulative"],
        attribution["ordinary_free"]["slices"],
        attribution["teacher_gpp_bridge"]["slices"],
        contract,
    )
    gates = contract.payload["integrity_gates"]
    integrity = attribution["integrity"]
    integrity["ordinary_free_report_maximum_absolute_metric_difference"] = metric_difference
    integrity["ordinary_free_report_reproduced"] = metric_difference <= float(
        gates["ordinary_free_report_maximum_absolute_metric_difference"]
    )
    integrity["gpp_residual_identity_passed"] = float(
        integrity["gpp_residual_identity_maximum_absolute_error"]
    ) <= float(gates["gpp_residual_identity_maximum_absolute_error"])
    integrity["gpp_squared_decomposition_passed"] = float(
        integrity["gpp_squared_decomposition_relative_error_maximum"]
    ) <= float(gates["gpp_squared_decomposition_relative_error_maximum"])
    integrity["carbon_residual_identity_passed"] = float(
        integrity["carbon_residual_identity_maximum_absolute_error"]
    ) <= float(gates["carbon_residual_identity_maximum_absolute_error"])
    integrity["hard_gates_passed"] = (
        int(integrity["ordinary_free_hard_failure_count"]) == 0
        and int(integrity["teacher_gpp_bridge_hard_failure_count"]) == 0
    )
    attribution["classification"] = classification
    integrity["all_reported_statistics_finite_and_nonempty"] = all_numeric_values_finite(
        {
            "ordinary_free": attribution["ordinary_free"],
            "teacher_day_start_direct_gpp": attribution["teacher_day_start_direct_gpp"],
            "gpp_residual_decomposition": attribution["gpp_residual_decomposition"],
            "teacher_gpp_bridge": attribution["teacher_gpp_bridge"],
            "signed_carbon_residual_propagation": attribution["signed_carbon_residual_propagation"],
            "classification": classification,
            "integrity_metric_difference": metric_difference,
        }
    )
    integrity["passed"] = all(
        bool(integrity[name])
        for name in (
            "ordinary_free_report_reproduced",
            "gpp_residual_identity_passed",
            "gpp_squared_decomposition_passed",
            "carbon_residual_identity_passed",
            "hard_gates_passed",
            "all_reported_statistics_finite_and_nonempty",
        )
    )

    result["schema_version"] = SCHEMA_VERSION
    result["status"] = "passed" if integrity["passed"] else "failed"
    result["identity"] = {
        **raw_identity,
        "git_head": finalizer_git_head,
        "numerical_execution_git_head": raw_identity["git_head"],
        "finalizer_git_head": finalizer_git_head,
        "raw_numerical_report_sha256": raw_report_sha256,
    }
    result["finalization"] = {
        "method": "cpu_summary_only_from_hash_bound_raw_statistics",
        "corrected_issue": "formal mode was shadowed by the carbon-summary loop variable",
        "numerical_reexecution": False,
        "gpu_execution": False,
        "teacher_data_read": False,
        "parameter_or_checkpoint_change": False,
    }
    result["next_gate"] = (
        contract.payload["next_gate"]["on_integrity_pass"]
        if integrity["passed"]
        else contract.payload["next_gate"]["on_integrity_failure"]
    )
    return result


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    contract = load_structured_gpp_fresh_attribution_contract(args.attribution_contract)
    raw_sha256 = _sha256_file(args.raw_attribution_report)
    formal_sha256 = _sha256_file(args.fresh_formal_report)
    report = finalize_report(
        json.loads(args.raw_attribution_report.read_text(encoding="utf-8")),
        json.loads(args.fresh_formal_report.read_text(encoding="utf-8")),
        contract,
        raw_report_sha256=raw_sha256,
        formal_report_sha256=formal_sha256,
        finalizer_git_head=_git_head(),
    )
    report["finalization"]["elapsed_seconds"] = time.perf_counter() - started
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-attribution-report", type=Path, required=True)
    parser.add_argument("--fresh-formal-report", type=Path, required=True)
    parser.add_argument("--attribution-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
