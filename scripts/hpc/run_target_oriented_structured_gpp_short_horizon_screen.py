"""Evaluate the selected structured-GPP checkpoint on the frozen 7/30-day panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_checkpoint import (
    load_structured_gpp_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    derive_target_oriented_acceptance_scales,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _replace_adapter
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
)
from scripts.hpc.run_target_oriented_training import (
    _evaluate_panel,
    _make_validation_evaluator,
)
from scripts.hpc.target_oriented_validation_prepared import (
    load_target_oriented_validation_prepared,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_short_horizon_screen_result_v1"
CONTRACT_SCHEMA_VERSION = "target_oriented_structured_gpp_short_horizon_screen_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_screen_contract(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = _contract_sha256(payload)
    evaluation = payload.get("evaluation", {})
    execution = payload.get("execution", {})
    if (
        payload.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or payload.get("contract_sha256") != observed
        or evaluation.get("slices") != ["temporal", "spatial", "joint"]
        or evaluation.get("horizons_days") != [7, 30]
        or int(evaluation.get("windows_per_slice_per_horizon", -1)) != 16
        or int(evaluation.get("formal_window_count", -1)) != 96
        or int(evaluation.get("admission_window_count", -1)) != 6
        or int(execution.get("optimizer_updates", -1)) != 0
        or int(execution.get("checkpoint_writes", -1)) != 0
        or execution.get("teacher_data_regeneration_allowed") is not False
        or execution.get("historical_v5_allowed") is not False
        or execution.get("sealed_test_allowed") is not False
    ):
        raise ValueError("structured GPP short-horizon screen contract drift")
    return payload | {"source_path": str(source), "sha256": observed}


def _load_training_report(path: Path, contract: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = contract["identity"]
    if sha256_file(path) != identity["joint_training_report_sha256"]:
        raise ValueError("structured GPP joint-training report hash drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = report.get("selected_checkpoint", {})
    if (
        report.get("schema_version") != "target_oriented_structured_gpp_joint_training_result_v1"
        or report.get("status") != "passed"
        or report.get("next_gate") != "run_frozen_non_test_96_window_seven_thirty_day_screen"
        or report.get("policy", {}).get("validation_values_evaluated") is not False
        or int(selected.get("completed_updates", -1))
        != int(identity["selected_checkpoint_optimizer_step"])
        or selected.get("sha256") != identity["selected_checkpoint_sha256"]
        or int(selected.get("parameter_count", -1))
        != int(identity["trainable_specialist_parameter_count"])
    ):
        raise ValueError("structured GPP joint-training result is not screen eligible")
    return report


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    screen = _load_screen_contract(args.screen_contract)
    screen_identity = screen["identity"]
    training_report = _load_training_report(args.joint_training_report, screen)
    if sha256_file(args.selected_checkpoint) != screen_identity["selected_checkpoint_sha256"]:
        raise ValueError("structured GPP selected checkpoint hash drift")
    if sha256_file(args.distribution_training_report) != screen_identity["parent_distribution_report_sha256"]:
        raise ValueError("structured GPP parent training report hash drift")
    if args.stage_checkpoint_sha256 != screen_identity["base_attention_checkpoint_sha256"]:
        raise ValueError("structured GPP base checkpoint identity drift")
    if args.distribution_checkpoint_sha256 != screen_identity["parent_distribution_checkpoint_sha256"]:
        raise ValueError("structured GPP distribution checkpoint identity drift")

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
        load_distribution_aware_checkpoint,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
        build_target_oriented_structured_gpp_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        initialize_structured_gpp_parameters,
        load_structured_gpp_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_training import (
        load_structured_gpp_joint_training_contract,
    )

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=True,
        )
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    target_contract = context["target_contract"]
    if (
        amendment.sha256 != screen_identity["flux_basis_amendment_sha256"]
        or context["training_contract"].sha256 != screen_identity["base_training_contract_sha256"]
        or target_contract.sha256 != screen_identity["target_contract_sha256"]
    ):
        raise ValueError("structured GPP short-horizon parent contract drift")

    structured_contract = load_structured_gpp_contract(args.structured_contract)
    joint_contract = load_structured_gpp_joint_training_contract(args.joint_training_contract)
    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    if (
        structured_contract.sha256 != screen_identity["candidate_contract_sha256"]
        or joint_contract.sha256 != screen_identity["joint_training_contract_sha256"]
        or training_report["identity"]["joint_training_contract_sha256"] != joint_contract.sha256
    ):
        raise ValueError("structured GPP short-horizon candidate contract drift")

    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    distribution_checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=args.distribution_checkpoint_sha256,
        identity=distribution_report["identity"],
        completed_updates=250,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(distribution_report["training"]["clip_threshold"]),
    )
    initialized_parent = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=distribution_contract,
    )
    parent_parameters = _replace_adapter(
        initialized_parent,
        jax.device_put(distribution_checkpoint["adapter"]),
    )
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("structured GPP short-horizon parent parameter count drift")

    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    initialized = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    selected_checkpoint = load_structured_gpp_checkpoint(
        args.selected_checkpoint,
        expected_sha256=screen_identity["selected_checkpoint_sha256"],
        identity=training_report["identity"],
        completed_updates=int(screen_identity["selected_checkpoint_optimizer_step"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(training_report["training"]["clip_threshold"]),
    )
    specialist = jax.device_put(selected_checkpoint["specialist"])
    parameters = replace_specialist(initialized, specialist)
    if (
        specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
        or _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
    ):
        raise ValueError("structured GPP short-horizon candidate parameter count drift")

    validation_prepared, panel = load_target_oriented_validation_prepared(
        args.validation_prepared_manifest,
        training_contract_sha256=context["training_contract"].sha256,
        target_contract_sha256=target_contract.sha256,
        verify_artifacts=False,
    )
    if (
        validation_prepared["manifest_sha256"]
        != screen_identity["validation_prepared_manifest_sha256"]
        or panel["panel_sha256"] != screen_identity["validation_panel_sha256"]
    ):
        raise ValueError("structured GPP short-horizon validation identity drift")

    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    evaluator = _make_validation_evaluator(
        runtime,
        head,
        context["template"],
        acceptance_scales,
        context["scales"],
    )
    panel_report = _evaluate_panel(
        runtime,
        parameters=parameters,
        evaluator=evaluator,
        prepared_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=panel,
        target_contract=target_contract,
        mode=args.mode,
        label=f"structured_gpp_step_498_{args.mode}",
    )
    admission_passed = args.mode == "admission" and panel_report["hard_failure_count"] == 0
    formal_passed = args.mode == "formal" and panel_report["absolute_short_horizon_passed"]
    next_gate = screen["next_gate"][
        "on_admission_pass"
        if admission_passed
        else "on_admission_failure"
        if args.mode == "admission"
        else "on_formal_absolute_pass"
        if formal_passed
        else "on_formal_absolute_failure"
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "mode": args.mode,
        "identity": {
            "git_head": _git_head(),
            "screen_contract_sha256": screen["sha256"],
            "joint_training_report_sha256": sha256_file(args.joint_training_report),
            "selected_checkpoint_sha256": sha256_file(args.selected_checkpoint),
            "selected_checkpoint_optimizer_step": int(screen_identity["selected_checkpoint_optimizer_step"]),
            "validation_prepared_manifest_sha256": validation_prepared["manifest_sha256"],
            "validation_panel_sha256": panel["panel_sha256"],
        },
        "policy": {
            "optimizer_updates_executed": 0,
            "checkpoint_writes": 0,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "model": {
            "candidate_parameter_count": CANDIDATE_PARAMETER_COUNT,
            "trainable_specialist_parameter_count": SPECIALIST_PARAMETER_COUNT,
            "retained_parent_parameter_count": int(screen_identity["retained_parent_parameter_count"]),
        },
        "panel": panel_report,
        "acceptance": {
            "admission_execution_and_hard_gates_passed": admission_passed,
            "formal_absolute_short_horizon_passed": formal_passed,
            "scientific_promotion_decided": args.mode == "formal",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
    }
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--flux-basis-amendment", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--distribution-contract", type=Path, required=True)
    parser.add_argument("--distribution-training-report", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint-sha256", required=True)
    parser.add_argument("--structured-contract", type=Path, required=True)
    parser.add_argument("--joint-training-contract", type=Path, required=True)
    parser.add_argument("--joint-training-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--mode", choices=("admission", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
