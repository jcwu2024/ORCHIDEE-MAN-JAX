"""Audit the selected coupled-GPP state on independent non-test 7/30-day windows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_scale_selected_transfer import (
    PRIMARY_SOURCES,
    SCALE_SELECTING_RESULT_SHA256,
    load_coupled_gpp_scale_selected_transfer_contract,
    transfer_accepted,
    transfer_changes,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import (
    _evaluate_artifacts,
    _fixed_parameter_report,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selecting_recalibration import (
    _git_head,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selecting_recalibration import (
    _parser as _scale_selecting_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_scale_selecting_recalibration import (
    run as _run_scale_selecting,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_scale_selected_transfer_audit_result_v1"


def _load_scale_selecting_result(path: Path, contract: Any) -> Mapping[str, Any]:
    if sha256_file(path) != SCALE_SELECTING_RESULT_SHA256:
        raise ValueError("coupled GPP scale-selecting result identity drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "passed"
        or report.get("classification") != "three_scale_selected_recalibrations_accepted"
        or report.get("identity", {}).get("scale_selecting_contract_sha256")
        != contract.payload["identity"]["scale_selecting_contract_sha256"]
        or tuple(report.get("smoke", {}).get("updates_accepted", ())) != (5, 6, 7)
        or tuple(float(value) for value in report.get("smoke", {}).get("selected_scales", ())) != (0.25, 1.0, 0.5)
        or report.get("policy", {}).get("checkpoint_written") is not False
    ):
        raise ValueError("coupled GPP scale-selecting result is not transfer eligible")
    return report


def _per_artifact_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    before_records = before["records"]
    after_records = after["records"]
    if len(before_records) != len(after_records):
        raise ValueError("coupled GPP transfer artifact count drift")
    records = []
    for first, second in zip(before_records, after_records, strict=True):
        if int(first["artifact_update"]) != int(second["artifact_update"]):
            raise ValueError("coupled GPP transfer artifact order drift")
        changes = transfer_changes(first["source_losses"], second["source_losses"])
        records.append(
            {
                "artifact_update": int(first["artifact_update"]),
                "runtime_year": int(first["runtime_year"]),
                "source_changes": changes,
                "primary_sources_improved": {
                    name: float(changes[name]["relative_change"]) < 0.0 for name in PRIMARY_SOURCES
                },
            }
        )
    return records


def run(args: Any) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    contract = load_coupled_gpp_scale_selected_transfer_contract(args.transfer_contract)
    source_sha_before = sha256_file(args.scale_selecting_result)
    predecessor = _load_scale_selecting_result(args.scale_selecting_result, contract)
    recovery_before = args.parent_recovery.stat()

    reconstructed, internal = _run_scale_selecting(
        args,
        write_output=False,
        return_internal=True,
    )
    if (
        reconstructed["status"] != predecessor["status"]
        or reconstructed["classification"] != predecessor["classification"]
        or reconstructed["smoke"]["updates_accepted"] != predecessor["smoke"]["updates_accepted"]
        or reconstructed["smoke"]["selected_scales"] != predecessor["smoke"]["selected_scales"]
    ):
        raise RuntimeError("coupled GPP scale-selected state reconstruction drift")

    context = internal["context"]
    missing = sorted(set(contract.monitor_updates) - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"coupled GPP transfer prepared coverage drift: {missing}")
    initialized_parameters = internal["initialized_parameters"]
    baseline_trainable = internal["baseline_trainable"]
    final_trainable = internal["final_trainable"]
    baseline_parameters = replace_coupled_gpp_trainable(initialized_parameters, baseline_trainable)
    final_parameters = replace_coupled_gpp_trainable(initialized_parameters, final_trainable)

    baseline = _evaluate_artifacts(
        args,
        context,
        parameters=baseline_parameters,
        trainable=baseline_trainable,
        steps=internal["steps"],
        direct_steps=internal["direct_steps"],
        state_steps=internal["state_steps"],
        layout=internal["layout"],
        updates=contract.monitor_updates,
        flux_basis=internal["flux_basis"],
        coefficients=internal["coefficients"],
        label="scale_selected_transfer_before_update_5",
    )
    selected = _evaluate_artifacts(
        args,
        context,
        parameters=final_parameters,
        trainable=final_trainable,
        steps=internal["steps"],
        direct_steps=internal["direct_steps"],
        state_steps=internal["state_steps"],
        layout=internal["layout"],
        updates=contract.monitor_updates,
        flux_basis=internal["flux_basis"],
        coefficients=internal["coefficients"],
        label="scale_selected_transfer_after_update_7",
    )
    changes = transfer_changes(baseline["source_mean_losses"], selected["source_mean_losses"])
    accepted = transfer_accepted(changes)
    artifact_changes = _per_artifact_changes(baseline, selected)
    improved_counts = {
        name: sum(bool(record["primary_sources_improved"][name]) for record in artifact_changes)
        for name in PRIMARY_SOURCES
    }

    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    fixed_report = _fixed_parameter_report(
        context["runtime"]["jax"],
        initialized_parameters,
        final_parameters,
    )
    source_unchanged = sha256_file(args.scale_selecting_result) == source_sha_before
    if not recovery_unchanged or not fixed_report["exactly_unchanged"] or not source_unchanged:
        raise RuntimeError("coupled GPP transfer retention boundary failed")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if accepted else "failed_scientific_gate",
        "classification": (
            "non_test_seven_thirty_day_transfer_accepted" if accepted else "non_test_seven_thirty_day_transfer_rejected"
        ),
        "identity": {
            "git_head": _git_head(),
            "transfer_contract_sha256": contract.sha256,
            "scale_selecting_result_sha256": source_sha_before,
            "scale_selecting_contract_sha256": reconstructed["identity"]["scale_selecting_contract_sha256"],
            "training_schedule_sha256": reconstructed["identity"]["schedule_sha256"],
        },
        "policy": {
            "optimizer_updates_persisted": 0,
            "reconstructed_parameters_persisted": False,
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
            "updates_accepted": reconstructed["smoke"]["updates_accepted"],
            "selected_scales": reconstructed["smoke"]["selected_scales"],
            "baseline": "state_before_update_5",
            "selected": "state_after_update_7",
        },
        "transfer": {
            "artifact_updates": list(contract.monitor_updates),
            "seven_day_window_count": int(contract.payload["data"]["seven_day_window_count"]),
            "thirty_day_window_count": int(contract.payload["data"]["thirty_day_window_count"]),
            "source_changes": changes,
            "primary_improved_artifact_counts": improved_counts,
            "artifact_count": len(artifact_changes),
            "per_artifact_changes": artifact_changes,
            "baseline_evaluation": baseline,
            "selected_evaluation": selected,
            "physical_hard_gate_passed": bool(baseline["hard_gate_passed"] and selected["hard_gate_passed"]),
            "accepted": accepted,
        },
        "integrity": {
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "scale_selecting_result_unchanged": source_unchanged,
        },
        "memory_after": _memory_report(internal["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"]["on_transfer_pass" if accepted else "on_transfer_failure"],
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _scale_selecting_parser()
    parser.add_argument("--transfer-contract", type=Path, required=True)
    parser.add_argument("--scale-selecting-result", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
