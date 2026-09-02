"""Audit inherited versus reset AdamW geometry at the rejected update 2."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_bounded_training import (
    build_coupled_gpp_training_schedule,
    finite_step_accepted,
    load_coupled_gpp_bounded_training_contract,
    source_changes,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_checkpoint import (
    load_coupled_gpp_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    GRADIENT_SOURCES,
    RETAINED_PARAMETER_COUNT,
    TRAINABLE_FAMILIES,
    TRAINABLE_PARAMETER_COUNT,
    build_reachable_state_anchor_layout,
    coupled_gpp_trainable_tree,
    load_coupled_gpp_objective_calibration_contract,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_optimizer_preflight import (
    four_source_macro_objectives,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_update2_geometry import (
    PARENT_REJECTED_SHA256,
    PARENT_RESULT_SHA256,
    UPDATE_TWO_ARTIFACTS,
    classify_update2_geometry,
    load_coupled_gpp_update2_geometry_contract,
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
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import _tree_cosine
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import (
    _evaluate_artifacts,
    _fixed_parameter_report,
    _training_proposal,
    _validate_backoff_report,
)
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import (
    _make_proposal_check,
    _tree_scalar_report,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_update2_geometry_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _parent_evidence(
    args: argparse.Namespace,
    audit_contract: Any,
    bounded_contract: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    result = json.loads(args.parent_result_report.read_text(encoding="utf-8"))
    rejected = json.loads(args.parent_rejected_proposal.read_text(encoding="utf-8"))
    identity = audit_contract.payload["identity"]
    proposal = rejected.get("proposal", {})
    if (
        sha256_file(args.parent_result_report) != PARENT_RESULT_SHA256
        or sha256_file(args.parent_result_report) != identity["parent_result_report_sha256"]
        or result.get("schema_version") != "target_oriented_coupled_gpp_bounded_training_result_v1"
        or result.get("status") != "stopped_scientific_gate"
        or result.get("training", {}).get("completed_updates") != 1
        or result.get("training", {}).get("stopped_reason") != "finite_step_rejected"
        or result.get("selected_checkpoint") is not None
        or result.get("identity", {}).get("bounded_training_contract_sha256") != bounded_contract.sha256
        or sha256_file(args.parent_rejected_proposal) != PARENT_REJECTED_SHA256
        or sha256_file(args.parent_rejected_proposal) != identity["parent_rejected_proposal_sha256"]
        or rejected.get("schema_version") != "target_oriented_coupled_gpp_rejected_training_proposal_v1"
        or rejected.get("identity") != result.get("identity")
        or int(rejected.get("updates_retained_before_rejection", -1)) != 1
        or int(proposal.get("optimizer_update", -1)) != 2
        or tuple(int(value) for value in proposal.get("artifact_updates", ())) != UPDATE_TWO_ARTIFACTS
        or proposal.get("finite_step_accepted") is not False
    ):
        raise ValueError("coupled GPP update-2 parent evidence drift")
    return result, rejected


def _direction_record(
    jax: Any,
    jnp: Any,
    source_gradients: Mapping[str, Any],
    before: Any,
    proposed: Any,
) -> Mapping[str, Any]:
    direction = jax.tree_util.tree_map(lambda first, second: first - second, before, proposed)
    report = _tree_scalar_report(jax, jnp, direction)
    cosines = {name: _tree_cosine(jax, source_gradients[name], direction) for name in GRADIENT_SOURCES}
    if not report["all_finite"] or report["element_count"] != TRAINABLE_PARAMETER_COUNT or report["l2_norm"] <= 0.0:
        raise RuntimeError("coupled GPP update-2 optimizer direction is invalid")
    return {
        "parameter_descent_direction": report,
        "source_gradient_cosines": cosines,
        "common_source_descent_direction": all(float(cosines[name]) > 0.0 for name in GRADIENT_SOURCES),
    }


def _scaled_parameters(jax: Any, before: Any, proposed: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(
        lambda first, second: first + float(scale) * (second - first),
        before,
        proposed,
    )


def _scale_record(
    *,
    scale: float,
    before_losses: Mapping[str, float],
    evaluation: Mapping[str, Any],
    coefficients: Mapping[str, float],
    relative_tolerance: float,
    direction_norm: float,
) -> Mapping[str, Any]:
    after = evaluation["source_mean_losses"]
    accepted = finite_step_accepted(
        before_losses,
        after,
        coefficients=coefficients,
        relative_tolerance=relative_tolerance,
    )
    return {
        "scale": float(scale),
        "parameter_delta_l2": float(scale) * float(direction_norm),
        "source_losses_after": after,
        "source_changes": source_changes(before_losses, after),
        "weighted_objective_after": four_source_macro_objectives(after, coefficients),
        "finite_step_accepted": accepted,
        "hard_gate_passed": bool(evaluation["hard_gate_passed"]),
        "evaluation": evaluation,
    }


def _evaluate_scales(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    *,
    initialized_parameters: Any,
    before_trainable: Any,
    proposed_trainable: Any,
    before_losses: Mapping[str, float],
    scales: Sequence[float],
    scale_one_evaluation: Mapping[str, Any] | None,
    steps: Mapping[str, Any],
    direct_steps: Mapping[str, Any],
    state_steps: Mapping[str, Any],
    layout: Any,
    group: Sequence[int],
    flux_basis: Any,
    coefficients: Mapping[str, float],
    relative_tolerance: float,
    label: str,
    direction_norm: float,
) -> list[Mapping[str, Any]]:
    jax = context["runtime"]["jax"]
    records = []
    for scale in scales:
        if float(scale) == 1.0 and scale_one_evaluation is not None:
            evaluation = scale_one_evaluation
        else:
            scaled = _scaled_parameters(jax, before_trainable, proposed_trainable, float(scale))
            parameters = replace_coupled_gpp_trainable(initialized_parameters, scaled)
            evaluation = _evaluate_artifacts(
                args,
                context,
                parameters=parameters,
                trainable=scaled,
                steps=steps,
                direct_steps=direct_steps,
                state_steps=state_steps,
                layout=layout,
                updates=group,
                flux_basis=flux_basis,
                coefficients=coefficients,
                label=f"{label}_scale_{float(scale):g}",
            )
            del scaled, parameters
        records.append(
            _scale_record(
                scale=float(scale),
                before_losses=before_losses,
                evaluation=evaluation,
                coefficients=coefficients,
                relative_tolerance=relative_tolerance,
                direction_norm=direction_norm,
            )
        )
        print(
            f"coupled_gpp update2_geometry direction={label} scale={float(scale):g} "
            f"accepted={records[-1]['finite_step_accepted']}",
            flush=True,
        )
    return records


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    audit_contract = load_coupled_gpp_update2_geometry_contract(args.audit_contract)
    bounded_contract = load_coupled_gpp_bounded_training_contract(args.bounded_training_contract)
    objective_contract = load_coupled_gpp_objective_calibration_contract(args.objective_calibration_contract)
    identity_contract = bounded_contract.payload["identity"]
    if (
        audit_contract.payload["identity"]["bounded_training_contract_sha256"] != bounded_contract.sha256
        or objective_contract.sha256 != identity_contract["objective_calibration_contract_sha256"]
    ):
        raise ValueError("coupled GPP update-2 contract chain drift")
    _validate_backoff_report(args.optimizer_backoff_report, bounded_contract)
    layout = build_reachable_state_anchor_layout(objective_contract)
    _validate_reachability_report(args.reachability_report, objective_contract, layout)
    parent_result, parent_rejected = _parent_evidence(args, audit_contract, bounded_contract)

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
        raise ValueError("coupled GPP update-2 data identity drift")
    schedule = build_coupled_gpp_training_schedule()
    if (
        schedule.sha256 != audit_contract.payload["identity"]["schedule_sha256"]
        or schedule.groups[1] != UPDATE_TWO_ARTIFACTS
    ):
        raise ValueError("coupled GPP update-2 schedule drift")
    missing = sorted(set(UPDATE_TWO_ARTIFACTS) - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"coupled GPP update-2 prepared coverage drift: {missing}")

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
        raise ValueError("coupled GPP update-2 parameter boundary drift")

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
        raise ValueError("coupled GPP update-2 recovery state drift")
    trainable = jax.device_put(recovered["trainable"])
    optimizer = jax.device_put(recovered["optimizer"])
    parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    relative_tolerance = float(audit_contract.payload["evaluation"]["relative_source_tolerance"])

    inherited = _training_proposal(
        args,
        context,
        parameters=parameters,
        trainable=trainable,
        optimizer=optimizer,
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
        return_internal=True,
    )
    internal = inherited["internal"]
    source_gradients = internal["source_gradients"]
    parent_changes = parent_rejected["proposal"]["source_changes"]
    tolerance = float(
        audit_contract.payload["evaluation"]["reproduce_parent_rejected_scale_one_changes_absolute_tolerance"]
    )
    reproduced = all(
        abs(
            float(inherited["record"]["source_changes"][name]["relative_change"])
            - float(parent_changes[name]["relative_change"])
        )
        <= tolerance
        for name in GRADIENT_SOURCES
    )
    if inherited["accepted"] or not reproduced:
        raise RuntimeError("coupled GPP update-2 rejected proposal was not reproduced")

    inherited_direction = _direction_record(
        jax,
        jnp,
        source_gradients,
        trainable,
        inherited["updated_trainable"],
    )
    reset_optimizer = initialize_target_oriented_adamw(trainable)
    reset_trainable, reset_state, reset_raw_norm, reset_clip_scale = optimizer_step(
        trainable,
        internal["combined_gradient"],
        reset_optimizer,
    )
    jax.block_until_ready((reset_trainable, reset_state, reset_raw_norm, reset_clip_scale))
    parameter_nonfinite, optimizer_nonfinite, reset_delta_squared = proposal_check(
        trainable,
        reset_trainable,
        reset_state,
    )
    jax.block_until_ready((parameter_nonfinite, optimizer_nonfinite, reset_delta_squared))
    if (
        int(np.asarray(parameter_nonfinite)) != 0
        or int(np.asarray(optimizer_nonfinite)) != 0
        or int(np.asarray(reset_state.step)) != 1
        or float(np.asarray(reset_delta_squared)) <= 0.0
    ):
        raise RuntimeError("coupled GPP update-2 reset proposal is invalid")
    reset_direction = _direction_record(jax, jnp, source_gradients, trainable, reset_trainable)

    before_losses = inherited["record"]["source_mean_losses_before"]
    scales = audit_contract.scales
    inherited_scales = _evaluate_scales(
        args,
        context,
        initialized_parameters=initialized_parameters,
        before_trainable=trainable,
        proposed_trainable=inherited["updated_trainable"],
        before_losses=before_losses,
        scales=scales,
        scale_one_evaluation=inherited["record"]["post_evaluation"],
        steps=steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
        layout=layout,
        group=UPDATE_TWO_ARTIFACTS,
        flux_basis=amendment.flux_basis,
        coefficients=bounded_contract.coefficients,
        relative_tolerance=relative_tolerance,
        label="inherited_adamw",
        direction_norm=float(inherited_direction["parameter_descent_direction"]["l2_norm"]),
    )
    reset_scales = _evaluate_scales(
        args,
        context,
        initialized_parameters=initialized_parameters,
        before_trainable=trainable,
        proposed_trainable=reset_trainable,
        before_losses=before_losses,
        scales=scales,
        scale_one_evaluation=None,
        steps=steps,
        direct_steps=direct_steps,
        state_steps=state_steps,
        layout=layout,
        group=UPDATE_TWO_ARTIFACTS,
        flux_basis=amendment.flux_basis,
        coefficients=bounded_contract.coefficients,
        relative_tolerance=relative_tolerance,
        label="reset_adamw",
        direction_norm=float(reset_direction["parameter_descent_direction"]["l2_norm"]),
    )
    inherited_accepted = [float(record["scale"]) for record in inherited_scales if record["finite_step_accepted"]]
    reset_accepted = [float(record["scale"]) for record in reset_scales if record["finite_step_accepted"]]
    classification = classify_update2_geometry(
        inherited_cosines=inherited_direction["source_gradient_cosines"],
        reset_cosines=reset_direction["source_gradient_cosines"],
        inherited_accepted_scales=inherited_accepted,
        reset_accepted_scales=reset_accepted,
    )
    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    fixed_report = _fixed_parameter_report(jax, initialized_parameters, parameters)
    if not recovery_unchanged or not fixed_report["exactly_unchanged"]:
        raise RuntimeError("coupled GPP update-2 retention boundary failed")

    next_key = (
        "on_inherited_backtracking_admissible"
        if classification == "inherited_backtracking_admissible"
        else "on_moment_reset_admissible"
        if classification == "moment_reset_admissible"
        else "on_no_admissible_scale"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": {
            "git_head": _git_head(),
            "audit_contract_sha256": audit_contract.sha256,
            "parent_result_report_sha256": sha256_file(args.parent_result_report),
            "parent_rejected_proposal_sha256": sha256_file(args.parent_rejected_proposal),
            "parent_git_head": parent_result["identity"]["git_head"],
            "bounded_training_contract_sha256": bounded_contract.sha256,
            "coupled_candidate_contract_sha256": candidate["coupled_contract"].sha256,
            "schedule_sha256": schedule.sha256,
        },
        "policy": {
            "optimizer_updates_proposed": 2,
            "optimizer_updates_persisted": 0,
            "parameter_proposals_persisted": 0,
            "checkpoint_written": False,
            "recovery_modified": False,
            "fresh_validation_values_evaluated": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "parent": {
            "completed_updates": int(recovered["completed_updates"]),
            "optimizer_step": int(np.asarray(optimizer.step)),
            "source_losses_before_update2": before_losses,
            "weighted_objective_before_update2": four_source_macro_objectives(
                before_losses,
                bounded_contract.coefficients,
            ),
            "rejected_scale_one_reproduced": reproduced,
            "recovery_unchanged": recovery_unchanged,
        },
        "gradient": {
            "sources": inherited["record"]["gradients"]["sources"],
            "combined": inherited["record"]["gradients"]["combined"],
            "source_to_combined_cosines": inherited["record"]["gradients"]["source_to_combined_cosines"],
        },
        "directions": {
            "inherited_adamw": inherited_direction
            | {
                "optimizer_step_after_proposal": int(np.asarray(inherited["updated_optimizer"].step)),
                "raw_gradient_norm": inherited["record"]["optimizer"]["raw_gradient_norm"],
                "clip_scale": inherited["record"]["optimizer"]["clip_scale"],
                "scales": inherited_scales,
                "accepted_scales": inherited_accepted,
            },
            "reset_adamw": reset_direction
            | {
                "optimizer_step_after_proposal": int(np.asarray(reset_state.step)),
                "raw_gradient_norm": float(np.asarray(reset_raw_norm)),
                "clip_scale": float(np.asarray(reset_clip_scale)),
                "family_parameter_steps": {
                    name: _tree_scalar_report(
                        jax,
                        jnp,
                        jax.tree_util.tree_map(
                            lambda first, second: second - first,
                            trainable[name],
                            reset_trainable[name],
                        ),
                    )
                    for name in TRAINABLE_FAMILIES
                },
                "scales": reset_scales,
                "accepted_scales": reset_accepted,
            },
        },
        "classification": classification,
        "model": {
            "candidate_parameter_count": _tree_size(jax, parameters),
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "fixed_parameters": fixed_report,
            "source_checkpoint_unchanged": sha256_file(args.selected_checkpoint)
            == identity_contract["structured_checkpoint_sha256"],
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": audit_contract.payload["next_gate"][next_key],
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
    parser.add_argument("--audit-contract", type=Path, required=True)
    parser.add_argument("--parent-result-report", type=Path, required=True)
    parser.add_argument("--parent-rejected-proposal", type=Path, required=True)
    parser.add_argument("--parent-recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
