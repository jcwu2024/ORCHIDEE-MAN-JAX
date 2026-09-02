"""Audit a four-source common direction at coupled-GPP update 3."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_bounded_training import (
    build_coupled_gpp_training_schedule,
    load_coupled_gpp_bounded_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_checkpoint import (
    load_coupled_gpp_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_fresh_moment import (
    load_coupled_gpp_fresh_moment_admission_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    GRADIENT_SOURCES,
    RETAINED_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    build_reachable_state_anchor_layout,
    coupled_gpp_trainable_tree,
    load_coupled_gpp_objective_calibration_contract,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_update3_common_direction import (
    FRESH_MOMENT_RESULT_SHA256,
    UPDATE_THREE_ARTIFACTS,
    UPDATE_TWO_ARTIFACTS,
    classify_update3_common_direction,
    load_coupled_gpp_update3_common_direction_contract,
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
from scripts.hpc.probe_successor_direct_trust_region import (
    _make_combined_direction,
    _make_direct_virtual_step,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import (
    _fixed_parameter_report,
    _training_proposal,
    _validate_backoff_report,
)
from scripts.hpc.run_target_oriented_coupled_gpp_fresh_moment_admission import _load_parent_result
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import (
    _direction_record,
    _evaluate_scales,
)
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import _make_proposal_check

SCHEMA_VERSION = "target_oriented_coupled_gpp_update3_common_direction_result_v1"
PASSING_CLASSIFICATIONS = (
    "direct_common_finite_step",
    "fresh_adamw_common_finite_step",
)


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _load_fresh_moment_result(path: Path, audit_contract: Any) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    records = report.get("admission", {}).get("records", ())
    if (
        sha256_file(path) != FRESH_MOMENT_RESULT_SHA256
        or sha256_file(path) != audit_contract.payload["identity"]["fresh_moment_result_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_fresh_moment_admission_result_v1"
        or report.get("status") != "failed_scientific_gate"
        or len(records) != 2
        or int(records[0].get("optimizer_update", -1)) != 2
        or records[0].get("finite_step_accepted") is not True
        or int(records[1].get("optimizer_update", -1)) != 3
        or records[1].get("finite_step_accepted") is not False
        or report.get("policy", {}).get("optimizer_updates_persisted") != 0
        or report.get("policy", {}).get("parameter_updates_persisted") != 0
        or report.get("parent", {}).get("recovery_unchanged") is not True
    ):
        raise ValueError("coupled GPP update-3 predecessor evidence drift")
    return report


def _change_differences(current: Mapping[str, Any], reference: Mapping[str, Any]) -> Mapping[str, float]:
    return {
        name: float(current["source_changes"][name]["relative_change"])
        - float(reference["source_changes"][name]["relative_change"])
        for name in GRADIENT_SOURCES
    }


def _accepted_scales(records: list[Mapping[str, Any]]) -> list[float]:
    return [float(record["scale"]) for record in records if bool(record["finite_step_accepted"])]


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    audit_contract = load_coupled_gpp_update3_common_direction_contract(args.audit_contract)
    fresh_contract = load_coupled_gpp_fresh_moment_admission_contract(args.fresh_moment_contract)
    bounded_contract = load_coupled_gpp_bounded_training_contract(args.bounded_training_contract)
    objective_contract = load_coupled_gpp_objective_calibration_contract(args.objective_calibration_contract)
    identity_contract = bounded_contract.payload["identity"]
    if (
        audit_contract.payload["identity"]["fresh_moment_contract_sha256"] != fresh_contract.sha256
        or audit_contract.payload["identity"]["bounded_training_contract_sha256"] != bounded_contract.sha256
        or objective_contract.sha256 != identity_contract["objective_calibration_contract_sha256"]
    ):
        raise ValueError("coupled GPP update-3 common-direction contract chain drift")
    _validate_backoff_report(args.optimizer_backoff_report, bounded_contract)
    layout = build_reachable_state_anchor_layout(objective_contract)
    _validate_reachability_report(args.reachability_report, objective_contract, layout)
    parent_result = _load_parent_result(args.parent_result_report, fresh_contract, bounded_contract)
    predecessor = _load_fresh_moment_result(args.fresh_moment_result, audit_contract)

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
        raise ValueError("coupled GPP update-3 common-direction data identity drift")
    schedule = build_coupled_gpp_training_schedule()
    if (
        schedule.sha256 != audit_contract.payload["identity"]["schedule_sha256"]
        or schedule.groups[1] != UPDATE_TWO_ARTIFACTS
        or schedule.groups[2] != UPDATE_THREE_ARTIFACTS
    ):
        raise ValueError("coupled GPP update-3 common-direction schedule drift")
    required = set(UPDATE_TWO_ARTIFACTS) | set(UPDATE_THREE_ARTIFACTS)
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"coupled GPP update-3 common-direction prepared coverage drift: {missing}")

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
        raise ValueError("coupled GPP update-3 common-direction parameter boundary drift")

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
    gram_measurement = _make_gradient_gram(jax)
    combine_direction = _make_combined_direction(jax)
    direct_virtual_step = _make_direct_virtual_step(jax)
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
        raise ValueError("coupled GPP update-3 common-direction recovery state drift")
    trainable = jax.device_put(recovered["trainable"])
    relative_tolerance = float(audit_contract.payload["evaluation"]["relative_source_tolerance"])

    parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    update_two = _training_proposal(
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
        group=UPDATE_TWO_ARTIFACTS,
        flux_basis=amendment.flux_basis,
        coefficients=bounded_contract.coefficients,
        relative_tolerance=relative_tolerance,
        optimizer_update=2,
        expected_optimizer_state_step=1,
    )
    if not update_two["accepted"]:
        raise RuntimeError("coupled GPP update-3 reconstruction no longer accepts fresh update 2")
    update_two_trainable = update_two["updated_trainable"]

    update_three_parameters = replace_coupled_gpp_trainable(initialized_parameters, update_two_trainable)
    update_three = _training_proposal(
        args,
        context,
        parameters=update_three_parameters,
        trainable=update_two_trainable,
        optimizer=initialize_target_oriented_adamw(update_two_trainable),
        optimizer_step=optimizer_step,
        proposal_check=proposal_check,
        steps=steps,
        gradient_steps=gradient_steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
        layout=layout,
        group=UPDATE_THREE_ARTIFACTS,
        flux_basis=amendment.flux_basis,
        coefficients=bounded_contract.coefficients,
        relative_tolerance=relative_tolerance,
        optimizer_update=3,
        return_internal=True,
        expected_optimizer_state_step=1,
    )
    if update_three["accepted"]:
        raise RuntimeError("coupled GPP update-3 frozen fresh-moment rejection drift")

    source_gradients = update_three["internal"]["source_gradients"]
    ordered_gradients = tuple(source_gradients[name] for name in GRADIENT_SOURCES)
    gram, source_nonfinite = gram_measurement(ordered_gradients)
    jax.block_until_ready((gram, source_nonfinite))
    gram_host = np.asarray(gram, dtype=np.float64)
    nonfinite_host = np.asarray(source_nonfinite, dtype=np.int64)
    diagnostics = gradient_direction_diagnostics(gram_host, labels=GRADIENT_SOURCES)
    selected = diagnostics["directions"]["minimum_norm_unit"]
    weights = np.asarray([selected["weights"][name] for name in GRADIENT_SOURCES], dtype=np.float64)
    common_gradient, common_norm, common_nonfinite = combine_direction(ordered_gradients, weights)
    jax.block_until_ready((common_gradient, common_norm, common_nonfinite))
    if (
        np.any(nonfinite_host != 0)
        or int(np.asarray(common_nonfinite)) != 0
        or not np.isfinite(float(np.asarray(common_norm)))
        or float(np.asarray(common_norm)) <= 0.0
    ):
        raise RuntimeError("coupled GPP update-3 common gradient is invalid")

    reference_delta = float(audit_contract.payload["direction"]["reference_update3_fresh_adamw_parameter_delta_l2"])
    direct_proposed, direct_nonfinite = direct_virtual_step(
        update_two_trainable,
        common_gradient,
        common_norm,
        np.asarray(reference_delta, dtype=np.float64),
    )
    jax.block_until_ready((direct_proposed, direct_nonfinite))
    if int(np.asarray(direct_nonfinite)) != 0:
        raise RuntimeError("coupled GPP update-3 direct proposal is nonfinite")
    direct_direction = _direction_record(
        jax,
        jnp,
        source_gradients,
        update_two_trainable,
        direct_proposed,
    )

    adamw_proposed, adamw_state, adamw_raw_norm, adamw_clip_scale = optimizer_step(
        update_two_trainable,
        common_gradient,
        initialize_target_oriented_adamw(update_two_trainable),
    )
    jax.block_until_ready((adamw_proposed, adamw_state, adamw_raw_norm, adamw_clip_scale))
    parameter_nonfinite, optimizer_nonfinite, adamw_delta_squared = proposal_check(
        update_two_trainable,
        adamw_proposed,
        adamw_state,
    )
    jax.block_until_ready((parameter_nonfinite, optimizer_nonfinite, adamw_delta_squared))
    if (
        int(np.asarray(parameter_nonfinite)) != 0
        or int(np.asarray(optimizer_nonfinite)) != 0
        or int(np.asarray(adamw_state.step)) != 1
        or float(np.asarray(adamw_delta_squared)) <= 0.0
    ):
        raise RuntimeError("coupled GPP update-3 common AdamW proposal is invalid")
    adamw_direction = _direction_record(
        jax,
        jnp,
        source_gradients,
        update_two_trainable,
        adamw_proposed,
    )

    before_losses = update_three["record"]["source_mean_losses_before"]
    scales = audit_contract.scales
    direct_scales = _evaluate_scales(
        args,
        context,
        initialized_parameters=initialized_parameters,
        before_trainable=update_two_trainable,
        proposed_trainable=direct_proposed,
        before_losses=before_losses,
        scales=scales,
        scale_one_evaluation=None,
        steps=steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
        layout=layout,
        group=UPDATE_THREE_ARTIFACTS,
        flux_basis=amendment.flux_basis,
        coefficients=bounded_contract.coefficients,
        relative_tolerance=relative_tolerance,
        label="update3_common_direct",
        direction_norm=reference_delta,
    )
    adamw_delta = float(np.sqrt(float(np.asarray(adamw_delta_squared))))
    adamw_scales = _evaluate_scales(
        args,
        context,
        initialized_parameters=initialized_parameters,
        before_trainable=update_two_trainable,
        proposed_trainable=adamw_proposed,
        before_losses=before_losses,
        scales=scales,
        scale_one_evaluation=None,
        steps=steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
        layout=layout,
        group=UPDATE_THREE_ARTIFACTS,
        flux_basis=amendment.flux_basis,
        coefficients=bounded_contract.coefficients,
        relative_tolerance=relative_tolerance,
        label="update3_common_fresh_adamw",
        direction_norm=adamw_delta,
    )
    direct_accepted = _accepted_scales(direct_scales)
    adamw_accepted = _accepted_scales(adamw_scales)
    classification = classify_update3_common_direction(
        raw_source_cosines=selected["normalized_directional_derivatives"],
        direct_accepted_scales=direct_accepted,
        adamw_source_cosines=adamw_direction["source_gradient_cosines"],
        adamw_accepted_scales=adamw_accepted,
        minimum_source_cosine=float(audit_contract.payload["direction"]["minimum_source_cosine"]),
    )
    passed = classification in PASSING_CLASSIFICATIONS

    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    final_parameters = replace_coupled_gpp_trainable(initialized_parameters, update_two_trainable)
    fixed_report = _fixed_parameter_report(jax, initialized_parameters, final_parameters)
    source_checkpoint_unchanged = (
        sha256_file(args.selected_checkpoint) == identity_contract["structured_checkpoint_sha256"]
    )
    if not recovery_unchanged or not fixed_report["exactly_unchanged"] or not source_checkpoint_unchanged:
        raise RuntimeError("coupled GPP update-3 common-direction retention boundary failed")

    predecessor_records = predecessor["admission"]["records"]
    selected_realization = None
    if classification == "direct_common_finite_step":
        selected_realization = {"name": "direct", "largest_accepted_scale": direct_accepted[0]}
    elif classification == "fresh_adamw_common_finite_step":
        selected_realization = {"name": "fresh_adamw", "largest_accepted_scale": adamw_accepted[0]}
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed_scientific_gate",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "audit_contract_sha256": audit_contract.sha256,
            "fresh_moment_contract_sha256": fresh_contract.sha256,
            "fresh_moment_result_sha256": sha256_file(args.fresh_moment_result),
            "parent_result_report_sha256": sha256_file(args.parent_result_report),
            "bounded_training_contract_sha256": bounded_contract.sha256,
            "coupled_candidate_contract_sha256": candidate["coupled_contract"].sha256,
            "schedule_sha256": schedule.sha256,
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
            "update_2_accepted": update_two["accepted"],
            "update_3_fixed_weight_accepted": update_three["accepted"],
            "numeric_differences_reporting_only": {
                "update_2_source_relative_changes": _change_differences(update_two["record"], predecessor_records[0]),
                "update_3_source_relative_changes": _change_differences(update_three["record"], predecessor_records[1]),
            },
            "update_2_record": update_two["record"],
            "update_3_fixed_weight_record": update_three["record"],
        },
        "common_direction": {
            "gradient_gram": gram_host.tolist(),
            "source_nonfinite_counts": {
                name: int(nonfinite_host[index]) for index, name in enumerate(GRADIENT_SOURCES)
            },
            "diagnostics": diagnostics,
            "selected_minimum_norm_direction": selected,
            "combined_gradient_norm": float(np.asarray(common_norm)),
            "selected_realization": selected_realization,
        },
        "realizations": {
            "direct": {
                "direction": direct_direction,
                "reference_parameter_delta_l2": reference_delta,
                "accepted_scales": direct_accepted,
                "scales": direct_scales,
            },
            "fresh_adamw": {
                "direction": adamw_direction,
                "optimizer_state_step": int(np.asarray(adamw_state.step)),
                "raw_gradient_norm": float(np.asarray(adamw_raw_norm)),
                "clip_scale": float(np.asarray(adamw_clip_scale)),
                "parameter_delta_l2": adamw_delta,
                "accepted_scales": adamw_accepted,
                "scales": adamw_scales,
            },
        },
        "integrity": {
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "source_checkpoint_unchanged": source_checkpoint_unchanged,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": audit_contract.payload["next_gate"][
            "on_common_finite_step"
            if passed
            else (
                "on_no_common_first_order_direction"
                if classification == "no_common_first_order_direction"
                else "on_no_common_finite_step"
            )
        ],
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
    parser.add_argument("--audit-contract", type=Path, required=True)
    parser.add_argument("--fresh-moment-result", type=Path, required=True)
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
