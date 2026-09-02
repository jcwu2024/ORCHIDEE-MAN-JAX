"""Run three discarded direct common-direction coupled-GPP updates."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    gradient_direction_diagnostics,
)
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
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_recalibration import (
    COMMON_DIRECTION_RESULT_SHA256,
    DIRECT_RECALIBRATION_GROUPS,
    DIRECT_RECALIBRATION_UPDATES,
    REQUIRED_CONSECUTIVE_UPDATES,
    load_coupled_gpp_direct_recalibration_contract,
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
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_optimizer_preflight import (
    four_source_macro_objectives,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_update3_common_direction import (
    load_coupled_gpp_update3_common_direction_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
    target_oriented_adamw_update,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    _add_gradient,
    _load_context,
    _tree_size,
    _zero_gradient,
)
from scripts.hpc.audit_target_oriented_gpp_state_reachability import _build_candidate
from scripts.hpc.calibrate_target_oriented_coupled_gpp_objective import (
    _make_direct_steps,
    _make_state_steps,
    _state_gradient,
    _validate_reachability_report,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _direct_gradient,
    _horizon_gradient,
    _one_day_hard,
)
from scripts.hpc.probe_successor_direct_trust_region import (
    _make_combined_direction,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_bounded_training import (
    _all_hard_zero,
    _evaluate_artifacts,
    _fixed_parameter_report,
    _hard_host,
    _mean_gradient,
    _mean_losses,
    _source_losses,
    _training_proposal,
    _tree_scalar_report,
    _validate_backoff_report,
)
from scripts.hpc.run_target_oriented_coupled_gpp_fresh_moment_admission import _load_parent_result
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import _direction_record
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import _make_proposal_check

SCHEMA_VERSION = "target_oriented_coupled_gpp_direct_recalibration_smoke_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_dtype_preserving_direct_virtual_step(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def step(parameters: Any, direction: Any, direction_norm: Any, step_norm: Any) -> Any:
        proposed = jax.tree_util.tree_map(
            lambda value, gradient: (
                value
                - jnp.asarray(step_norm, dtype=value.dtype)
                * jnp.asarray(gradient, dtype=value.dtype)
                / jnp.maximum(
                    jnp.asarray(direction_norm, dtype=value.dtype),
                    jnp.asarray(1.0e-30, dtype=value.dtype),
                )
            ),
            parameters,
            direction,
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int64)
                for leaf in jax.tree_util.tree_leaves(proposed)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        return proposed, nonfinite

    return step


def _load_common_direction_result(path: Path, contract: Any, audit_contract: Any) -> Mapping[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    direct = report.get("realizations", {}).get("direct", {})
    if (
        sha256_file(path) != COMMON_DIRECTION_RESULT_SHA256
        or sha256_file(path) != contract.payload["identity"]["common_direction_result_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_update3_common_direction_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "direct_common_finite_step"
        or report.get("identity", {}).get("audit_contract_sha256") != audit_contract.sha256
        or direct.get("accepted_scales") != [1.0, 0.5, 0.25, 0.125, 0.0625]
        or report.get("policy", {}).get("checkpoint_written") is not False
        or report.get("integrity", {}).get("parent_recovery_unchanged") is not True
    ):
        raise ValueError("coupled GPP common-direction predecessor evidence drift")
    return report


def _measure_source_gradients(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    *,
    parameters: Any,
    trainable: Any,
    steps: Mapping[str, Any],
    gradient_steps: Mapping[str, Any],
    direct_steps: Mapping[str, Any],
    state_steps: Mapping[str, Any],
    layout: Any,
    group: Sequence[int],
    flux_basis: Any,
    macro_update: int,
) -> Mapping[str, Any]:
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    started = time.perf_counter()
    gradient_sums = {name: _zero_gradient(jax, jnp, trainable) for name in GRADIENT_SOURCES}
    loss_records = []
    hard_records = []
    artifact_records = []
    for ordinal, update in enumerate(group, start=1):
        artifact_started = time.perf_counter()
        artifact = _load_artifact(args, context, int(update))
        direct, direct_gradient, direct_forward, direct_backward = _direct_gradient(
            jax, jnp, trainable, artifact.batch, direct_steps
        )
        state, state_gradient, state_forward, state_backward = _state_gradient(
            jax, jnp, trainable, artifact.batch, state_steps, layout
        )
        one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
        seven, seven_hard, seven_gradient, seven_timing = _horizon_gradient(
            jax,
            jnp,
            parameters,
            trainable,
            artifact.batch,
            steps,
            gradient_steps,
            horizon=7,
            flux_basis=flux_basis,
        )
        thirty, thirty_hard, thirty_gradient, thirty_timing = _horizon_gradient(
            jax,
            jnp,
            parameters,
            trainable,
            artifact.batch,
            steps,
            gradient_steps,
            horizon=30,
            flux_basis=flux_basis,
        )
        current = {
            GRADIENT_SOURCES[0]: direct_gradient,
            GRADIENT_SOURCES[1]: state_gradient,
            GRADIENT_SOURCES[2]: seven_gradient,
            GRADIENT_SOURCES[3]: thirty_gradient,
        }
        gradient_sums = {name: _add_gradient(jax, jnp, gradient_sums[name], current[name]) for name in GRADIENT_SOURCES}
        jax.block_until_ready(gradient_sums)
        losses = _source_losses(direct, state, seven, thirty)
        if not all(np.isfinite(value) and value >= 0.0 for value in losses.values()):
            raise RuntimeError(f"coupled GPP direct recalibration update {macro_update} has invalid loss")
        hard = {
            "artifact_update": int(update),
            "one_day": _hard_host(jax.device_get(one_hard)),
            "seven_day": _hard_host(jax.device_get(seven_hard)),
            "thirty_day": _hard_host(jax.device_get(thirty_hard)),
        }
        loss_records.append(losses)
        hard_records.append(hard)
        artifact_records.append(
            {
                "artifact_update": int(update),
                "runtime_year": int(context["schedule"]["entries"][int(update) - 1]["runtime_year"]),
                "source_losses_before": losses,
                "hard_counts_before": hard,
                "timing_seconds": {
                    "direct_forward": direct_forward,
                    "direct_gradient": direct_backward,
                    "state_forward": state_forward,
                    "state_gradient": state_backward,
                    "seven_day": seven_timing,
                    "thirty_day": thirty_timing,
                    "total": time.perf_counter() - artifact_started,
                },
            }
        )
        print(
            f"coupled_gpp direct_recalibration update={macro_update} gradient={ordinal}/{len(group)} "
            f"artifact={update} seconds={artifact_records[-1]['timing_seconds']['total']:.3f}",
            flush=True,
        )
        del current, direct_gradient, state_gradient, seven_gradient, thirty_gradient

    if not _all_hard_zero(hard_records):
        raise RuntimeError(f"coupled GPP direct recalibration update {macro_update} pre-step hard gate failed")
    source_gradients = {name: _mean_gradient(jax, gradient_sums[name], len(group)) for name in GRADIENT_SOURCES}
    source_reports = {name: _tree_scalar_report(jax, jnp, source_gradients[name]) for name in GRADIENT_SOURCES}
    if not all(
        report["all_finite"]
        and report["element_count"] == TRAINABLE_PARAMETER_COUNT
        and report["l2_norm"] > 0.0
        and report["nonzero_leaf_count"] == report["leaf_count"]
        for report in source_reports.values()
    ):
        raise RuntimeError(f"coupled GPP direct recalibration update {macro_update} gradient engineering gate failed")
    return {
        "source_gradients": source_gradients,
        "source_reports": source_reports,
        "source_mean_losses": _mean_losses(loss_records),
        "artifact_records": artifact_records,
        "timing_seconds": {"total": time.perf_counter() - started},
    }


def run(
    args: argparse.Namespace,
    *,
    write_output: bool = True,
    return_internal: bool = False,
) -> Mapping[str, Any] | tuple[Mapping[str, Any], Mapping[str, Any]]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_coupled_gpp_direct_recalibration_contract(args.recalibration_contract)
    audit_contract = load_coupled_gpp_update3_common_direction_contract(args.audit_contract)
    fresh_contract = load_coupled_gpp_fresh_moment_admission_contract(args.fresh_moment_contract)
    bounded_contract = load_coupled_gpp_bounded_training_contract(args.bounded_training_contract)
    objective_contract = load_coupled_gpp_objective_calibration_contract(args.objective_calibration_contract)
    identity_contract = bounded_contract.payload["identity"]
    if (
        contract.payload["identity"]["common_direction_audit_contract_sha256"] != audit_contract.sha256
        or contract.payload["identity"]["fresh_moment_contract_sha256"] != fresh_contract.sha256
        or contract.payload["identity"]["bounded_training_contract_sha256"] != bounded_contract.sha256
        or objective_contract.sha256 != identity_contract["objective_calibration_contract_sha256"]
    ):
        raise ValueError("coupled GPP direct recalibration contract chain drift")
    predecessor = _load_common_direction_result(args.audit_result, contract, audit_contract)
    _validate_backoff_report(args.optimizer_backoff_report, bounded_contract)
    layout = build_reachable_state_anchor_layout(objective_contract)
    _validate_reachability_report(args.reachability_report, objective_contract, layout)
    parent_result = _load_parent_result(args.parent_result_report, fresh_contract, bounded_contract)

    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator_dynamic_retained_inputs
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )

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
        raise ValueError("coupled GPP direct recalibration data identity drift")
    schedule = build_coupled_gpp_training_schedule()
    if (
        schedule.sha256 != contract.payload["identity"]["schedule_sha256"]
        or schedule.groups[1] != tuple(contract.payload["data"]["reconstruction_artifact_updates"])
        or tuple(schedule.groups[index - 1] for index in DIRECT_RECALIBRATION_UPDATES) != DIRECT_RECALIBRATION_GROUPS
    ):
        raise ValueError("coupled GPP direct recalibration schedule drift")
    required = set(schedule.groups[1]) | {value for group in DIRECT_RECALIBRATION_GROUPS for value in group}
    missing = sorted(required - set(context["prepared_records"]))
    if missing:
        raise ValueError(f"coupled GPP direct recalibration prepared coverage drift: {missing}")

    candidate = _build_candidate(runtime, args, context["template"])
    initialized_parameters = candidate["parameters"]
    initialized_trainable = coupled_gpp_trainable_tree(initialized_parameters)
    fixed_parameters = replace_coupled_gpp_trainable(initialized_parameters, {})
    if (
        candidate["coupled_contract"].sha256 != identity_contract["coupled_candidate_contract_sha256"]
        or candidate["coupled_contract"].sha256 != contract.payload["identity"]["coupled_candidate_contract_sha256"]
        or sha256_file(args.selected_checkpoint) != identity_contract["structured_checkpoint_sha256"]
        or _tree_size(jax, initialized_parameters) != int(identity_contract["candidate_parameter_count"])
        or _tree_size(jax, initialized_trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, fixed_parameters) != RETAINED_PARAMETER_COUNT
    ):
        raise ValueError("coupled GPP direct recalibration parameter boundary drift")

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
    direct_virtual_step = _make_dtype_preserving_direct_virtual_step(jax)
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
        raise ValueError("coupled GPP direct recalibration recovery state drift")
    trainable = jax.device_put(recovered["trainable"])
    relative_tolerance = float(contract.payload["evaluation"]["relative_source_tolerance"])

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
        group=schedule.groups[1],
        flux_basis=amendment.flux_basis,
        coefficients=bounded_contract.coefficients,
        relative_tolerance=relative_tolerance,
        optimizer_update=2,
        expected_optimizer_state_step=1,
    )
    if not update_two["accepted"]:
        raise RuntimeError("coupled GPP direct recalibration no longer accepts fresh update 2")
    trainable = update_two["updated_trainable"]

    records = []
    accepted_updates = []
    stop_classification = None
    last_internal = None
    minimum_cosine = float(contract.payload["direction"]["minimum_source_cosine"])
    parameter_step_l2 = float(contract.payload["direction"]["parameter_step_l2"])
    for macro_update, group in zip(DIRECT_RECALIBRATION_UPDATES, DIRECT_RECALIBRATION_GROUPS, strict=True):
        update_started = time.perf_counter()
        parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
        measurement = _measure_source_gradients(
            args,
            context,
            parameters=parameters,
            trainable=trainable,
            steps=steps,
            gradient_steps=gradient_steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            group=group,
            flux_basis=amendment.flux_basis,
            macro_update=macro_update,
        )
        source_gradients = measurement["source_gradients"]
        ordered_gradients = tuple(source_gradients[name] for name in GRADIENT_SOURCES)
        gram, source_nonfinite = gram_measurement(ordered_gradients)
        jax.block_until_ready((gram, source_nonfinite))
        gram_host = np.asarray(gram, dtype=np.float64)
        nonfinite_host = np.asarray(source_nonfinite, dtype=np.int64)
        diagnostics = gradient_direction_diagnostics(gram_host, labels=GRADIENT_SOURCES)
        selected = diagnostics["directions"]["minimum_norm_unit"]
        common_first_order = bool(
            np.all(nonfinite_host == 0)
            and all(
                float(selected["normalized_directional_derivatives"][name]) >= minimum_cosine
                for name in GRADIENT_SOURCES
            )
        )
        base_record = {
            "macro_update": macro_update,
            "artifact_updates": list(group),
            "source_mean_losses_before": measurement["source_mean_losses"],
            "weighted_objective_before": four_source_macro_objectives(
                measurement["source_mean_losses"], bounded_contract.coefficients
            ),
            "source_gradients": measurement["source_reports"],
            "gradient_gram": gram_host.tolist(),
            "source_nonfinite_counts": {
                name: int(nonfinite_host[index]) for index, name in enumerate(GRADIENT_SOURCES)
            },
            "direction_diagnostics": diagnostics,
            "selected_minimum_norm_direction": selected,
            "common_first_order_direction": common_first_order,
            "artifact_records": measurement["artifact_records"],
        }
        if not common_first_order:
            records.append(
                base_record
                | {"finite_step_accepted": False, "timing_seconds": {"total": time.perf_counter() - update_started}}
            )
            stop_classification = "no_common_first_order_direction"
            break

        weights = np.asarray([selected["weights"][name] for name in GRADIENT_SOURCES], dtype=np.float64)
        common_gradient, common_norm, common_nonfinite = combine_direction(ordered_gradients, weights)
        jax.block_until_ready((common_gradient, common_norm, common_nonfinite))
        if (
            int(np.asarray(common_nonfinite)) != 0
            or not np.isfinite(float(np.asarray(common_norm)))
            or float(np.asarray(common_norm)) <= 0.0
        ):
            raise RuntimeError(f"coupled GPP direct recalibration update {macro_update} common gradient is invalid")
        proposed, proposal_nonfinite = direct_virtual_step(
            trainable,
            common_gradient,
            common_norm,
            np.asarray(parameter_step_l2, dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposal_nonfinite))
        if int(np.asarray(proposal_nonfinite)) != 0:
            raise RuntimeError(f"coupled GPP direct recalibration update {macro_update} proposal is nonfinite")
        direction = _direction_record(jax, jnp, source_gradients, trainable, proposed)
        if direction["common_source_descent_direction"] is not True:
            raise RuntimeError(f"coupled GPP direct recalibration update {macro_update} realization drift")
        proposed_parameters = replace_coupled_gpp_trainable(initialized_parameters, proposed)
        after = _evaluate_artifacts(
            args,
            context,
            parameters=proposed_parameters,
            trainable=proposed,
            steps=steps,
            direct_steps=direct_steps,
            state_steps=state_steps,
            layout=layout,
            updates=group,
            flux_basis=amendment.flux_basis,
            coefficients=bounded_contract.coefficients,
            label=f"direct_recalibration_update_{macro_update}_post",
        )
        before_losses = measurement["source_mean_losses"]
        after_losses = after["source_mean_losses"]
        accepted = finite_step_accepted(
            before_losses,
            after_losses,
            coefficients=bounded_contract.coefficients,
            relative_tolerance=relative_tolerance,
        )
        record = base_record | {
            "combined_gradient_norm": float(np.asarray(common_norm)),
            "direction": direction,
            "parameter_delta_l2": parameter_step_l2,
            "source_mean_losses_after": after_losses,
            "source_changes": source_changes(before_losses, after_losses),
            "weighted_objective_after": four_source_macro_objectives(after_losses, bounded_contract.coefficients),
            "post_evaluation": after,
            "finite_step_accepted": accepted,
            "timing_seconds": {"total": time.perf_counter() - update_started},
        }
        records.append(record)
        last_internal = {
            "macro_update": macro_update,
            "group": group,
            "before_trainable": trainable,
            "proposed_trainable": proposed,
            "before_losses": before_losses,
            "full_scale_evaluation": after,
            "direction_norm": float(direction["parameter_descent_direction"]["l2_norm"]),
        }
        print(f"coupled_gpp direct_recalibration update={macro_update} accepted={accepted}", flush=True)
        if not accepted:
            stop_classification = "common_direction_finite_step_rejection"
            break
        trainable = proposed
        accepted_updates.append(macro_update)

    passed = len(accepted_updates) == REQUIRED_CONSECUTIVE_UPDATES
    classification = "three_direct_recalibrations_accepted" if passed else stop_classification
    recovery_after = args.parent_recovery.stat()
    recovery_unchanged = (
        recovery_before.st_size == recovery_after.st_size and recovery_before.st_mtime_ns == recovery_after.st_mtime_ns
    )
    final_parameters = replace_coupled_gpp_trainable(initialized_parameters, trainable)
    fixed_report = _fixed_parameter_report(jax, initialized_parameters, final_parameters)
    source_checkpoint_unchanged = (
        sha256_file(args.selected_checkpoint) == identity_contract["structured_checkpoint_sha256"]
    )
    audit_result_unchanged = sha256_file(args.audit_result) == COMMON_DIRECTION_RESULT_SHA256
    if (
        not recovery_unchanged
        or not fixed_report["exactly_unchanged"]
        or not source_checkpoint_unchanged
        or not audit_result_unchanged
    ):
        raise RuntimeError("coupled GPP direct recalibration retention boundary failed")

    next_key = (
        "on_three_updates_accepted"
        if passed
        else (
            "on_no_common_first_order_direction"
            if classification == "no_common_first_order_direction"
            else "on_common_direction_finite_step_rejection"
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed_scientific_gate",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "recalibration_contract_sha256": contract.sha256,
            "common_direction_audit_contract_sha256": audit_contract.sha256,
            "common_direction_result_sha256": sha256_file(args.audit_result),
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
            "fresh_update_2_accepted": update_two["accepted"],
            "fresh_update_2_record": update_two["record"],
        },
        "smoke": {
            "required_consecutive_updates": REQUIRED_CONSECUTIVE_UPDATES,
            "updates_evaluated": [int(record["macro_update"]) for record in records],
            "updates_accepted": accepted_updates,
            "records": records,
        },
        "integrity": {
            "parent_recovery_unchanged": recovery_unchanged,
            "fixed_parameters": fixed_report,
            "source_checkpoint_unchanged": source_checkpoint_unchanged,
            "common_direction_result_unchanged": audit_result_unchanged,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": contract.payload["next_gate"][next_key],
        "predecessor_classification": predecessor["classification"],
    }
    if write_output:
        _atomic_json(output, report)
    if return_internal:
        if last_internal is None or int(last_internal["macro_update"]) != DIRECT_RECALIBRATION_UPDATES[-1]:
            raise RuntimeError("coupled GPP direct recalibration did not expose the terminal update state")
        return report, last_internal | {
            "context": context,
            "head": candidate["head"],
            "initialized_parameters": initialized_parameters,
            "steps": steps,
            "gradient_steps": gradient_steps,
            "direct_steps": direct_steps,
            "state_steps": state_steps,
            "layout": layout,
            "flux_basis": amendment.flux_basis,
            "coefficients": bounded_contract.coefficients,
            "relative_tolerance": relative_tolerance,
            "device": runtime["devices"][0],
        }
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
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--recalibration-contract", type=Path, required=True)
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
