"""Run two discarded coupled-GPP updates with fresh AdamW moments."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_bounded_training import (
    build_coupled_gpp_training_schedule,
    load_coupled_gpp_bounded_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_checkpoint import (
    load_coupled_gpp_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_fresh_moment import (
    GEOMETRY_REPORT_SHA256,
    fresh_moment_admission_passed,
    load_coupled_gpp_fresh_moment_admission_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    RETAINED_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    build_reachable_state_anchor_layout,
    coupled_gpp_trainable_tree,
    load_coupled_gpp_objective_calibration_contract,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
    target_oriented_adamw_update,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.calibrate_target_oriented_coupled_gpp_objective import (
    _make_direct_steps,
    _make_state_steps,
    _validate_reachability_report,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import (
    _fixed_parameter_report,
    _training_proposal,
    _validate_backoff_report,
)
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import _make_proposal_check

SCHEMA_VERSION = "target_oriented_coupled_gpp_fresh_moment_admission_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _load_geometry_report(path: Path, contract: Any) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    reset = report.get("directions", {}).get("reset_adamw", {})
    inherited = report.get("directions", {}).get("inherited_adamw", {})
    if (
        sha256_file(path) != GEOMETRY_REPORT_SHA256
        or sha256_file(path) != contract.payload["identity"]["geometry_report_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_update2_geometry_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "moment_reset_admissible"
        or reset.get("accepted_scales") != [1.0, 0.0625]
        or inherited.get("accepted_scales") != []
        or report.get("policy", {}).get("optimizer_updates_persisted") != 0
        or report.get("parent", {}).get("recovery_unchanged") is not True
    ):
        raise ValueError("coupled GPP fresh-moment geometry evidence drift")
    return report


def _load_parent_result(path: Path, contract: Any, bounded_contract: Any) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        sha256_file(path) != contract.payload["identity"]["parent_result_report_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_bounded_training_result_v1"
        or report.get("status") != "stopped_scientific_gate"
        or report.get("training", {}).get("completed_updates") != 1
        or report.get("training", {}).get("stopped_reason") != "finite_step_rejected"
        or report.get("selected_checkpoint") is not None
        or report.get("identity", {}).get("bounded_training_contract_sha256") != bounded_contract.sha256
    ):
        raise ValueError("coupled GPP fresh-moment parent result drift")
    return report


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    admission_contract = load_coupled_gpp_fresh_moment_admission_contract(args.fresh_moment_contract)
    bounded_contract = load_coupled_gpp_bounded_training_contract(args.bounded_training_contract)
    objective_contract = load_coupled_gpp_objective_calibration_contract(args.objective_calibration_contract)
    identity_contract = bounded_contract.payload["identity"]
    if (
        admission_contract.payload["identity"]["bounded_training_contract_sha256"] != bounded_contract.sha256
        or objective_contract.sha256 != identity_contract["objective_calibration_contract_sha256"]
    ):
        raise ValueError("coupled GPP fresh-moment contract chain drift")
    _validate_backoff_report(args.optimizer_backoff_report, bounded_contract)
    layout = build_reachable_state_anchor_layout(objective_contract)
    _validate_reachability_report(args.reachability_report, objective_contract, layout)
    geometry_report = _load_geometry_report(args.geometry_report, admission_contract)
    parent_result = _load_parent_result(args.parent_result_report, admission_contract, bounded_contract)

    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator_dynamic_retained_inputs
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )
    from scripts.hpc.audit_target_oriented_gpp_state_reachability import _build_candidate

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    if (
        amendment.sha256 != identity_contract["flux_basis_amendment_sha256"]
        or sha256_file(args.prepared_manifest) != identity_contract["prepared_manifest_sha256"]
        or context["training_contract"].sha256 != identity_contract["base_training_contract_sha256"]
        or context["target_contract"].sha256 != identity_contract["target_contract_sha256"]
    ):
        raise ValueError("coupled GPP fresh-moment data identity drift")
    schedule = build_coupled_gpp_training_schedule()
    if schedule.sha256 != admission_contract.payload["identity"]["schedule_sha256"] or admission_contract.groups != (
        schedule.groups[1],
        schedule.groups[2],
    ):
        raise ValueError("coupled GPP fresh-moment schedule drift")
    missing = sorted(
        {value for group in admission_contract.groups for value in group} - set(context["prepared_records"])
    )
    if missing:
        raise ValueError(f"coupled GPP fresh-moment prepared coverage drift: {missing}")

    candidate = _build_candidate(runtime, args, context["template"])
    initialized_parameters = candidate["parameters"]
    initialized_trainable = coupled_gpp_trainable_tree(initialized_parameters)
    fixed_parameters = replace_coupled_gpp_trainable(initialized_parameters, {})
    if (
        candidate["coupled_contract"].sha256 != identity_contract["coupled_candidate_contract_sha256"]
        or sha256_file(args.selected_checkpoint) != identity_contract["structured_checkpoint_sha256"]
        or _tree_size(jax, initialized_parameters) != int(identity_contract["candidate_parameter_count"])
        or _tree_size(jax, initialized_trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, fixed_parameters) != RETAINED_PARAMETER_COUNT
    ):
        raise ValueError("coupled GPP fresh-moment parameter boundary drift")

    steps = _make_steps(
        runtime,
        candidate["head"],
        context["template"],
        context["scales"],
        context["target_contract"],
        science_flux_basis=amendment.flux_basis,
    )
    gradient_steps = steps["make_trainable_gradient_steps"](fixed_parameters, replace_coupled_gpp_trainable)
    direct_steps = _make_direct_steps(runtime, candidate["head"], initialized_parameters)
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=candidate["head"],
        retained_tail_template=context["template"],
    )
    state_steps = _make_state_steps(
        runtime,
        transition,
        initialized_parameters,
        context["scales"].state.other,
        layout,
    )
    context["coupled_learning_rate"] = bounded_contract.learning_rate
    optimizer_step = jax.jit(
        lambda values, derivatives, state: target_oriented_adamw_update(
            values,
            derivatives,
            state,
            learning_rate=jnp.asarray(bounded_contract.learning_rate, dtype=jnp.float32),
            clip_norm=jnp.asarray(bounded_contract.clip_threshold, dtype=jnp.float32),
            contract=context["training_contract"],
        )
    )
    proposal_check = _make_proposal_check(jax, jnp)
    memory_before = _memory_report(runtime["devices"][0])

    recovery_before = args.parent_recovery.stat()
    recovered = load_coupled_gpp_recovery(
        args.parent_recovery,
        identity=parent_result["identity"],
        parameter_count=TRAINABLE_PARAMETER_COUNT,
        clip_threshold=bounded_contract.clip_threshold,
        maximum_updates=8,
    )
    if (
        recovered is None
        or int(recovered["completed_updates"]) != 1
        or int(np.asarray(recovered["optimizer"].step)) != 1
    ):
        raise ValueError("coupled GPP fresh-moment recovery state drift")
    trainable = jax.device_put(recovered["trainable"])
    records = []
    relative_tolerance = float(
        admission_contract.payload["admission"]["all_four_source_losses_must_not_increase_relative_tolerance"]
    )
    for optimizer_update, group in zip(admission_contract.updates, admission_contract.groups, strict=True):
        parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
        proposal = _training_proposal(
            args,
            context,
            parameters=parameters,
            trainable=trainable,
            optimizer=initialize_target_oriented_adamw(trainable),
            optimizer_step=optimizer_step,
            proposal_check=proposal_check,
            steps=steps,
            gradient_steps=gradient_steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            group=group,
            flux_basis=amendment.flux_basis,
            coefficients=bounded_contract.coefficients,
            relative_tolerance=relative_tolerance,
            optimizer_update=optimizer_update,
            expected_optimizer_state_step=1,
        )
        records.append(proposal["record"])
        print(
            f"coupled_gpp fresh_moment_admission update={optimizer_update} accepted={proposal['accepted']}",
            flush=True,
        )
        if not proposal["accepted"]:
            break
        trainable = proposal["updated_trainable"]

    passed = fresh_moment_admission_passed(records)
    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    final_parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    fixed_report = _fixed_parameter_report(jax, initialized_parameters, final_parameters)
    source_checkpoint_unchanged = (
        sha256_file(args.selected_checkpoint) == identity_contract["structured_checkpoint_sha256"]
    )
    if not recovery_unchanged or not fixed_report["exactly_unchanged"] or not source_checkpoint_unchanged:
        raise RuntimeError("coupled GPP fresh-moment retention boundary failed")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed_scientific_gate",
        "identity": {
            "git_head": _git_head(),
            "fresh_moment_contract_sha256": admission_contract.sha256,
            "geometry_report_sha256": sha256_file(args.geometry_report),
            "parent_result_report_sha256": sha256_file(args.parent_result_report),
            "parent_git_head": parent_result["identity"]["git_head"],
            "bounded_training_contract_sha256": bounded_contract.sha256,
            "coupled_candidate_contract_sha256": candidate["coupled_contract"].sha256,
            "schedule_sha256": schedule.sha256,
        },
        "policy": {
            "fresh_optimizer_moments_per_macro_update": True,
            "optimizer_bias_correction_step": 1,
            "optimizer_updates_proposed": len(records),
            "optimizer_updates_persisted": 0,
            "parameter_updates_persisted": 0,
            "checkpoint_written": False,
            "parent_recovery_modified": False,
            "fresh_validation_values_evaluated": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "parent": {
            "completed_updates": int(recovered["completed_updates"]),
            "recovery_unchanged": recovery_unchanged,
            "geometry_classification": geometry_report["classification"],
        },
        "admission": {
            "updates_required": list(admission_contract.updates),
            "updates_evaluated": [int(record["optimizer_update"]) for record in records],
            "both_consecutive_proposals_passed": passed,
            "records": records,
        },
        "model": {
            "candidate_parameter_count": _tree_size(jax, final_parameters),
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "fixed_parameters": fixed_report,
            "source_checkpoint_unchanged": source_checkpoint_unchanged,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": admission_contract.payload["next_gate"]["on_pass" if passed else "on_scientific_rejection"],
    }
    _atomic_json(output, report)
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
    parser.add_argument("--broad-training-report", type=Path, required=True)
    parser.add_argument("--coupled-contract", type=Path, required=True)
    parser.add_argument("--reachability-contract", type=Path, required=True)
    parser.add_argument("--gpu-admission-report", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--objective-calibration-contract", type=Path, required=True)
    parser.add_argument("--objective-calibration-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--optimizer-backoff-report", type=Path, required=True)
    parser.add_argument("--bounded-training-contract", type=Path, required=True)
    parser.add_argument("--fresh-moment-contract", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--parent-result-report", type=Path, required=True)
    parser.add_argument("--parent-recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
