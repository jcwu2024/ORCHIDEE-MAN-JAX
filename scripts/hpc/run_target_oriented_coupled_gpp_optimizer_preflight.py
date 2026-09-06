"""Run one discarded macro-batch AdamW proposal for coupled GPP."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    RETAINED_PARAMETER_COUNT,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
    TRAINABLE_FAMILIES,
    TRAINABLE_PARAMETER_COUNT,
    add_reachable_state_anchor_statistics,
    build_reachable_state_anchor_layout,
    coupled_gpp_trainable_tree,
    field_grouped_state_anchor_from_sufficient_statistics,
    load_coupled_gpp_objective_calibration_contract,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_optimizer_preflight import (
    BACKOFF_SCHEMA_VERSION,
    four_source_macro_objectives,
    load_coupled_gpp_optimizer_preflight_contract,
    macro_source_changes,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    structured_gpp_direct_loss,
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
from scripts.hpc.audit_target_oriented_gpp_state_reachability import (
    _build_candidate,
)
from scripts.hpc.calibrate_target_oriented_coupled_gpp_objective import (
    _make_direct_steps,
    _make_state_steps,
    _state_gradient,
    _validate_reachability_report,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _load_artifact,
)
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _direct_gradient,
    _horizon_gradient,
    _tree_cosine,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _hard_failure_count,
    _make_steps,
    _slice,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _direct_forward,
    _hard_host,
    _horizon_forward,
    _one_day_hard,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_optimizer_preflight_result_v1"
BACKOFF_RESULT_SCHEMA_VERSION = "target_oriented_coupled_gpp_optimizer_backoff_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _mean_gradient(jax: Any, gradient: Any, count: int) -> Any:
    denominator = np.float32(count)
    return jax.tree_util.tree_map(lambda value: value / denominator, gradient)


def _weighted_gradient(
    jax: Any,
    gradients: Mapping[str, Any],
    coefficients: Mapping[str, float],
) -> Any:
    return jax.tree_util.tree_map(
        lambda *values: sum(
            (np.float32(coefficients[name]) * value for name, value in zip(GRADIENT_SOURCES, values, strict=True)),
            np.float32(0.0),
        ),
        *(gradients[name] for name in GRADIENT_SOURCES),
    )


def _state_forward(
    jax: Any,
    trainable: Any,
    batch: Any,
    state_steps: Mapping[str, Any],
    layout: Any,
) -> tuple[Any, float]:
    microbatches = tuple(_slice(batch.one_day, start, start + 4) for start in range(0, 32, 4))
    started = time.perf_counter()
    values = [state_steps["forward"](trainable, item) for item in microbatches]
    jax.block_until_ready(values)
    total = values[0]
    for value in values[1:]:
        total = add_reachable_state_anchor_statistics(total, value)
    result = field_grouped_state_anchor_from_sufficient_statistics(
        total,
        field_names=layout.field_names,
    )
    jax.block_until_ready(result)
    return result, time.perf_counter() - started


def _source_losses(direct: Any, state: Any, seven: Any, thirty: Any) -> Mapping[str, float]:
    return {
        DIRECT_SOURCE: float(np.asarray(structured_gpp_direct_loss(direct))),
        STATE_SOURCE: float(np.asarray(state.total)),
        SEVEN_SOURCE: float(np.asarray(seven.total)),
        THIRTY_SOURCE: float(np.asarray(thirty.total)),
    }


def _fixed_parameter_report(
    jax: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    replace_trainable: Any = replace_coupled_gpp_trainable,
) -> Mapping[str, Any]:
    first = replace_trainable(before, {})
    second = replace_trainable(after, {})
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    same_tree = first_tree == second_tree
    same_objects = same_tree and all(left is right for left, right in zip(first_leaves, second_leaves, strict=True))
    return {
        "same_tree_structure": same_tree,
        "same_array_objects": same_objects,
        "leaf_count": len(first_leaves),
        "element_count": sum(int(np.prod(value.shape, dtype=np.int64)) for value in first_leaves),
        "exactly_unchanged": same_objects,
    }


def _validate_calibration_report(
    path: Path,
    contract: Any,
    objective_contract: Any,
) -> Mapping[str, Any]:
    identity = contract.payload["identity"]
    report = json.loads(path.read_text(encoding="utf-8"))
    calibration = report.get("calibration", {})
    if (
        sha256_file(path) != identity["objective_calibration_report_sha256"]
        or report.get("schema_version") != "target_oriented_coupled_gpp_objective_calibration_result_v1"
        or report.get("status") != "passed"
        or report.get("identity", {}).get("git_head") != identity["objective_calibration_execution_git_head"]
        or report.get("identity", {}).get("objective_calibration_contract_sha256") != objective_contract.sha256
        or tuple(int(record["artifact_update"]) for record in report.get("records", ())) != contract.artifact_updates
        or calibration.get("resolved_coefficients") != dict(contract.coefficients)
        or float(calibration.get("combined_gradient_p90_clip_threshold", 0.0)) != contract.clip_threshold
        or not all(bool(value) for value in report.get("acceptance", {}).values())
        or report.get("next_gate") != "freeze_discarded_four_source_optimizer_proposal_preflight"
    ):
        raise ValueError("coupled GPP optimizer-preflight calibration result is not accepted")
    return report


def _validate_rejected_optimizer_preflight_report(
    path: Path,
    contract: Any,
) -> Mapping[str, Any]:
    identity = contract.payload["identity"]
    report = json.loads(path.read_text(encoding="utf-8"))
    acceptance = report.get("acceptance", {})
    if (
        sha256_file(path) != identity["rejected_optimizer_preflight_report_sha256"]
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "failed"
        or report.get("identity", {}).get("git_head") != identity["rejected_optimizer_preflight_execution_git_head"]
        or report.get("identity", {}).get("optimizer_preflight_contract_sha256")
        != identity["rejected_optimizer_preflight_contract_sha256"]
        or acceptance.get("every_macro_source_loss_decreased") is not False
        or acceptance.get("weighted_macro_objective_decreased") is not True
        or acceptance.get("optimizer_realizes_all_source_descent_directions") is not True
        or acceptance.get("physical_hard_gates_passed") is not True
        or report.get("next_gate") != "stop_before_training_and_review_optimizer_realization_or_objective_conflict"
    ):
        raise ValueError("coupled GPP optimizer-backoff predecessor is not accepted")
    return report


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_coupled_gpp_optimizer_preflight_contract(args.optimizer_preflight_contract)
    backoff = contract.payload["schema_version"] == BACKOFF_SCHEMA_VERSION
    objective_contract = load_coupled_gpp_objective_calibration_contract(args.objective_calibration_contract)
    identity = contract.payload["identity"]
    if objective_contract.sha256 != identity["objective_calibration_contract_sha256"]:
        raise ValueError("coupled GPP optimizer-preflight objective contract drift")
    calibration_report = _validate_calibration_report(
        args.objective_calibration_report,
        contract,
        objective_contract,
    )
    rejected_preflight_report = None
    if backoff:
        if args.rejected_optimizer_preflight_report is None:
            raise ValueError("coupled GPP optimizer backoff requires the rejected preflight report")
        rejected_preflight_report = _validate_rejected_optimizer_preflight_report(
            args.rejected_optimizer_preflight_report,
            contract,
        )
    elif args.rejected_optimizer_preflight_report is not None:
        raise ValueError("base coupled GPP optimizer preflight received a predecessor report")
    layout = build_reachable_state_anchor_layout(objective_contract)
    _validate_reachability_report(
        args.reachability_report,
        objective_contract,
        layout,
    )

    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )
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
        amendment.sha256 != identity["flux_basis_amendment_sha256"]
        or sha256_file(args.prepared_manifest) != identity["prepared_manifest_sha256"]
        or context["training_contract"].sha256 != identity["base_training_contract_sha256"]
        or context["target_contract"].sha256 != identity["target_contract_sha256"]
    ):
        raise ValueError("coupled GPP optimizer-preflight data identity drift")

    candidate = _build_candidate(runtime, args, context["template"])
    parameters = candidate["parameters"]
    head = candidate["head"]
    if (
        candidate["coupled_contract"].sha256 != identity["coupled_candidate_contract_sha256"]
        or sha256_file(args.selected_checkpoint) != identity["structured_checkpoint_sha256"]
        or _tree_size(jax, parameters) != int(identity["candidate_parameter_count"])
    ):
        raise ValueError("coupled GPP optimizer-preflight candidate identity drift")
    trainable = coupled_gpp_trainable_tree(parameters)
    fixed_parameters = replace_coupled_gpp_trainable(parameters, {})
    if (
        _tree_size(jax, trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, fixed_parameters) != RETAINED_PARAMETER_COUNT
    ):
        raise ValueError("coupled GPP optimizer-preflight parameter boundary drift")

    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        context["target_contract"],
        science_flux_basis=amendment.flux_basis,
    )
    gradient_steps = steps["make_trainable_gradient_steps"](
        fixed_parameters,
        replace_coupled_gpp_trainable,
    )
    direct_steps = _make_direct_steps(runtime, head, parameters)
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail_template=context["template"],
    )
    state_steps = _make_state_steps(
        runtime,
        transition,
        parameters,
        context["scales"].state.other,
        layout,
    )
    memory_before = _memory_report(runtime["devices"][0])

    gradient_sums = {name: _zero_gradient(jax, jnp, trainable) for name in GRADIENT_SOURCES}
    before_records = []
    pre_hard_records = []
    gradient_timings = []
    for ordinal, update in enumerate(contract.artifact_updates, start=1):
        artifact = _load_artifact(args, context, update)
        record_started = time.perf_counter()
        direct, direct_gradient, direct_forward, direct_backward = _direct_gradient(
            jax,
            jnp,
            trainable,
            artifact.batch,
            direct_steps,
        )
        state, state_gradient, state_forward, state_backward = _state_gradient(
            jax,
            jnp,
            trainable,
            artifact.batch,
            state_steps,
            layout,
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
            flux_basis=amendment.flux_basis,
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
            flux_basis=amendment.flux_basis,
        )
        current = {
            DIRECT_SOURCE: direct_gradient,
            STATE_SOURCE: state_gradient,
            SEVEN_SOURCE: seven_gradient,
            THIRTY_SOURCE: thirty_gradient,
        }
        gradient_sums = {name: _add_gradient(jax, jnp, gradient_sums[name], current[name]) for name in GRADIENT_SOURCES}
        jax.block_until_ready(gradient_sums)
        before_records.append(_source_losses(direct, state, seven, thirty))
        pre_hard_records.append(
            {
                "artifact_update": update,
                "one_day": _hard_host(jax.device_get(one_hard)),
                "seven_day": _hard_host(jax.device_get(seven_hard)),
                "thirty_day": _hard_host(jax.device_get(thirty_hard)),
            }
        )
        gradient_timings.append(
            {
                "artifact_update": update,
                "direct_forward": direct_forward,
                "direct_gradient": direct_backward,
                "state_forward": state_forward,
                "state_gradient": state_backward,
                "seven_day": seven_timing,
                "thirty_day": thirty_timing,
                "total": time.perf_counter() - record_started,
            }
        )
        print(
            f"coupled_gpp optimizer_preflight gradient={ordinal}/8 "
            f"artifact={update} seconds={gradient_timings[-1]['total']:.3f}",
            flush=True,
        )
        del current, direct_gradient, state_gradient, seven_gradient, thirty_gradient

    source_gradients = {
        name: _mean_gradient(jax, gradient_sums[name], len(contract.artifact_updates)) for name in GRADIENT_SOURCES
    }
    combined_gradient = _weighted_gradient(jax, source_gradients, contract.coefficients)
    jax.block_until_ready((source_gradients, combined_gradient))
    source_gradient_reports = {
        name: _tree_numeric_report(jax, jax.device_get(source_gradients[name])) for name in GRADIENT_SOURCES
    }
    combined_gradient_report = _tree_numeric_report(
        jax,
        jax.device_get(combined_gradient),
    )
    source_gradient_cosines = {
        f"{left}_vs_{right}": _tree_cosine(
            jax,
            source_gradients[left],
            source_gradients[right],
        )
        for index, left in enumerate(GRADIENT_SOURCES)
        for right in GRADIENT_SOURCES[index + 1 :]
    }

    optimizer = initialize_target_oriented_adamw(trainable)
    learning_rate = contract.learning_rate
    optimizer_step = jax.jit(
        lambda values, derivatives, state: target_oriented_adamw_update(
            values,
            derivatives,
            state,
            learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
            clip_norm=jnp.asarray(contract.clip_threshold, dtype=jnp.float32),
            contract=context["training_contract"],
        )
    )
    proposal_started = time.perf_counter()
    updated_trainable, optimizer_after, raw_norm, clip_scale = optimizer_step(
        trainable,
        combined_gradient,
        optimizer,
    )
    jax.block_until_ready((updated_trainable, optimizer_after, raw_norm, clip_scale))
    proposal_seconds = time.perf_counter() - proposal_started
    proposed_parameters = replace_coupled_gpp_trainable(parameters, updated_trainable)
    parameter_step = jax.tree_util.tree_map(
        lambda after, before: after - before,
        updated_trainable,
        trainable,
    )
    negative_step = jax.tree_util.tree_map(lambda value: -value, parameter_step)
    jax.block_until_ready((parameter_step, negative_step))
    source_to_negative_step = {
        name: _tree_cosine(jax, source_gradients[name], negative_step) for name in GRADIENT_SOURCES
    }
    step_report = _tree_numeric_report(jax, jax.device_get(parameter_step))
    family_step_reports = {
        name: _tree_numeric_report(jax, jax.device_get(parameter_step[name])) for name in TRAINABLE_FAMILIES
    }
    optimizer_report = _tree_numeric_report(jax, jax.device_get(optimizer_after))
    fixed_report = _fixed_parameter_report(jax, parameters, proposed_parameters)

    after_records = []
    post_hard_records = []
    forward_timings = []
    for ordinal, update in enumerate(contract.artifact_updates, start=1):
        artifact = _load_artifact(args, context, update)
        record_started = time.perf_counter()
        direct, direct_seconds = _direct_forward(
            jax,
            updated_trainable,
            artifact.batch,
            direct_steps,
        )
        state, state_seconds = _state_forward(
            jax,
            updated_trainable,
            artifact.batch,
            state_steps,
            layout,
        )
        one_hard = _one_day_hard(jax, jnp, proposed_parameters, artifact.batch, steps)
        seven, seven_hard, seven_seconds = _horizon_forward(
            jax,
            jnp,
            proposed_parameters,
            artifact.batch,
            steps,
            horizon=7,
            flux_basis=amendment.flux_basis,
        )
        thirty, thirty_hard, thirty_seconds = _horizon_forward(
            jax,
            jnp,
            proposed_parameters,
            artifact.batch,
            steps,
            horizon=30,
            flux_basis=amendment.flux_basis,
        )
        after_records.append(_source_losses(direct, state, seven, thirty))
        post_hard_records.append(
            {
                "artifact_update": update,
                "one_day": _hard_host(jax.device_get(one_hard)),
                "seven_day": _hard_host(jax.device_get(seven_hard)),
                "thirty_day": _hard_host(jax.device_get(thirty_hard)),
            }
        )
        forward_timings.append(
            {
                "artifact_update": update,
                "direct": direct_seconds,
                "state": state_seconds,
                "seven_day": seven_seconds,
                "thirty_day": thirty_seconds,
                "total": time.perf_counter() - record_started,
            }
        )
        print(
            f"coupled_gpp optimizer_preflight post={ordinal}/8 "
            f"artifact={update} seconds={forward_timings[-1]['total']:.3f}",
            flush=True,
        )

    changes = macro_source_changes(before_records, after_records)
    before_macro = {name: changes[name]["before"] for name in GRADIENT_SOURCES}
    after_macro = {name: changes[name]["after"] for name in GRADIENT_SOURCES}
    objective_before = four_source_macro_objectives(before_macro, contract.coefficients)
    objective_after = four_source_macro_objectives(after_macro, contract.coefficients)
    artifact_changes = [
        {
            "artifact_update": update,
            "source_changes": macro_source_changes((before,), (after,)),
        }
        for update, before, after in zip(
            contract.artifact_updates,
            before_records,
            after_records,
            strict=True,
        )
    ]

    source_gradients_passed = all(
        report["all_finite"]
        and report["element_count"] == TRAINABLE_PARAMETER_COUNT
        and report["l2_norm"] > 0.0
        and report["nonzero_leaf_count"] == report["leaf_count"]
        for report in source_gradient_reports.values()
    )
    combined_gradient_passed = bool(
        combined_gradient_report["all_finite"]
        and combined_gradient_report["element_count"] == TRAINABLE_PARAMETER_COUNT
        and combined_gradient_report["l2_norm"] > 0.0
        and combined_gradient_report["nonzero_leaf_count"] == combined_gradient_report["leaf_count"]
    )
    proposal_passed = bool(
        step_report["all_finite"]
        and step_report["element_count"] == TRAINABLE_PARAMETER_COUNT
        and step_report["l2_norm"] > 0.0
        and step_report["nonzero_leaf_count"] == step_report["leaf_count"]
        and all(
            value["all_finite"] and value["l2_norm"] > 0.0 and value["nonzero_leaf_count"] == value["leaf_count"]
            for value in family_step_reports.values()
        )
        and optimizer_report["all_finite"]
        and int(np.asarray(optimizer_after.step)) == 1
        and np.isfinite(float(np.asarray(raw_norm)))
        and np.isfinite(float(np.asarray(clip_scale)))
    )
    direction_passed = all(float(value) > 0.0 for value in source_to_negative_step.values())
    source_loss_passed = all(float(value["after"]) < float(value["before"]) for value in changes.values())
    weighted_loss_passed = objective_after["joint"] < objective_before["joint"]
    physical_passed = all(
        _hard_failure_count(value) == 0
        for records in (pre_hard_records, post_hard_records)
        for record in records
        for name, value in record.items()
        if name != "artifact_update"
    )
    checkpoint_unchanged = sha256_file(args.selected_checkpoint) == identity["structured_checkpoint_sha256"]
    passed = bool(
        source_gradients_passed
        and combined_gradient_passed
        and proposal_passed
        and direction_passed
        and source_loss_passed
        and weighted_loss_passed
        and physical_passed
        and fixed_report["exactly_unchanged"]
        and fixed_report["element_count"] == RETAINED_PARAMETER_COUNT
        and checkpoint_unchanged
    )

    report = {
        "schema_version": BACKOFF_RESULT_SCHEMA_VERSION if backoff else SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "optimizer_preflight_contract_sha256": contract.sha256,
            "objective_calibration_contract_sha256": objective_contract.sha256,
            "objective_calibration_report_sha256": sha256_file(args.objective_calibration_report),
            "structured_checkpoint_sha256": sha256_file(args.selected_checkpoint),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "coupled_candidate_contract_sha256": candidate["coupled_contract"].sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_created": True,
            "optimizer_updates_proposed": 1,
            "optimizer_updates_persisted": 0,
            "checkpoint_written": False,
            "recovery_written": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "macro_batch": {
            "artifact_updates": list(contract.artifact_updates),
            "artifact_count": len(contract.artifact_updates),
            "source_gradient_aggregation": "arithmetic_mean_across_eight_artifacts",
            "source_loss_aggregation": "arithmetic_mean_across_eight_artifact_losses",
        },
        "gradients": {
            "sources": source_gradient_reports,
            "combined": combined_gradient_report,
            "source_pairwise_cosines": source_gradient_cosines,
            "source_to_negative_parameter_step_cosines": source_to_negative_step,
        },
        "proposal": {
            "optimizer": "adamw",
            "fresh_optimizer_state": True,
            "learning_rate": learning_rate,
            "gradient_clip_threshold": contract.clip_threshold,
            "raw_gradient_norm": float(np.asarray(raw_norm)),
            "clip_scale": float(np.asarray(clip_scale)),
            "optimizer_step": int(np.asarray(optimizer_after.step)),
            "parameter_step": step_report,
            "family_parameter_steps": family_step_reports,
            "optimizer_state": optimizer_report,
            "fixed_parameters": fixed_report,
            "checkpoint_unchanged": checkpoint_unchanged,
            "seconds": proposal_seconds,
        },
        "objectives": {
            "coefficients": dict(contract.coefficients),
            "macro_source_changes": changes,
            "weighted_before": objective_before,
            "weighted_after": objective_after,
            "artifact_changes": artifact_changes,
        },
        "physical_hard_counts": {
            "before": pre_hard_records,
            "after": post_hard_records,
        },
        "timing_seconds": {
            "gradient_records": gradient_timings,
            "post_proposal_forward_records": forward_timings,
            "elapsed": time.perf_counter() - started,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "acceptance": {
            "mean_source_gradients_passed": source_gradients_passed,
            "combined_gradient_passed": combined_gradient_passed,
            "proposal_finite_and_complete": proposal_passed,
            "optimizer_realizes_all_source_descent_directions": direction_passed,
            "every_macro_source_loss_decreased": source_loss_passed,
            "weighted_macro_objective_decreased": weighted_loss_passed,
            "physical_hard_gates_passed": physical_passed,
            "fixed_parameters_passed": fixed_report["exactly_unchanged"],
            "structured_checkpoint_unchanged": checkpoint_unchanged,
        },
        "calibration_summary": {
            "status": calibration_report["status"],
            "resolved_coefficients": calibration_report["calibration"]["resolved_coefficients"],
            "combined_gradient_p90_clip_threshold": calibration_report["calibration"][
                "combined_gradient_p90_clip_threshold"
            ],
        },
        "rejected_preflight_summary": (
            {
                "status": rejected_preflight_report["status"],
                "report_sha256": sha256_file(args.rejected_optimizer_preflight_report),
                "learning_rate": rejected_preflight_report["proposal"]["learning_rate"],
                "macro_source_changes": rejected_preflight_report["objectives"]["macro_source_changes"],
            }
            if rejected_preflight_report is not None
            else None
        ),
        "next_gate": contract.payload["next_gate"]["on_pass" if passed else "on_failure"],
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
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--coupled-contract", type=Path, required=True)
    parser.add_argument("--reachability-contract", type=Path, required=True)
    parser.add_argument("--gpu-admission-report", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--objective-calibration-contract", type=Path, required=True)
    parser.add_argument("--objective-calibration-report", type=Path, required=True)
    parser.add_argument("--optimizer-preflight-contract", type=Path, required=True)
    parser.add_argument("--rejected-optimizer-preflight-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
