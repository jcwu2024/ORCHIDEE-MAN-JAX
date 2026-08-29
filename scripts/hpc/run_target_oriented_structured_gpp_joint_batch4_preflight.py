"""Run two independent discarded batch-four structured-GPP AdamW proposals."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_checkpoint import (
    load_structured_gpp_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_batch4 import (
    RUNTIME_YEAR_GROUPS,
    load_structured_gpp_joint_batch4_contract,
    mean_source_losses,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
    load_structured_gpp_joint_gradient_calibration_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_heterogeneity import (
    load_structured_gpp_joint_heterogeneity_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_update import (
    structured_gpp_joint_objectives,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
    structured_gpp_direct_loss,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    target_oriented_adamw_update,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.audit_target_oriented_structured_gpp_joint_gradient_heterogeneity import (
    _mean_host_gradients,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _load_artifact,
    _replace_adapter,
)
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _combine_host_gradients,
    _direct_gradient,
    _horizon_gradient,
    _make_direct_steps,
    _one_day_hard,
    _retained_parameter_report,
    _tree_cosine,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED, _tree_numeric_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _make_steps
from scripts.hpc.run_target_oriented_structured_gpp_gpu_admission import (
    _component_gradient_reports,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
    _direct_forward,
    _hard_host,
    _horizon_forward,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_batch4_preflight_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _source_losses(direct: Any, seven: Any, thirty: Any) -> Mapping[str, float]:
    return {
        DIRECT_SOURCE: float(np.asarray(structured_gpp_direct_loss(direct))),
        SEVEN_SOURCE: float(np.asarray(seven.total)),
        THIRTY_SOURCE: float(np.asarray(thirty.total)),
    }


def _relative_change(before: float, after: float) -> float:
    if not np.isfinite(before) or before <= 0.0 or not np.isfinite(after):
        raise ValueError("structured GPP batch4 source loss is invalid")
    return (after - before) / before


def _loss_change(before: Mapping[str, float], after: Mapping[str, float]) -> Mapping[str, Any]:
    return {
        name: {
            "before": float(before[name]),
            "after": float(after[name]),
            "relative_change": _relative_change(float(before[name]), float(after[name])),
        }
        for name in GRADIENT_SOURCES
    }


def _accepted_heterogeneity_result(args: argparse.Namespace, contract: Any) -> Mapping[str, Any]:
    identity = contract.payload["identity"]
    if sha256_file(args.heterogeneity_report) != identity["heterogeneity_report_sha256"]:
        raise ValueError("structured GPP batch4 heterogeneity report hash drift")
    heterogeneity_contract = load_structured_gpp_joint_heterogeneity_contract(
        args.heterogeneity_contract
    )
    if heterogeneity_contract.sha256 != identity["heterogeneity_contract_sha256"]:
        raise ValueError("structured GPP batch4 heterogeneity contract identity drift")
    report = json.loads(args.heterogeneity_report.read_text(encoding="utf-8"))
    groups = report.get("effective_batch_analysis", {}).get("grouped_reports", {}).get("4", [])
    if (
        report.get("status") != "passed"
        or report.get("identity", {}).get("git_head")
        != identity["heterogeneity_execution_git_head"]
        or report.get("identity", {}).get("heterogeneity_contract_sha256")
        != heterogeneity_contract.sha256
        or not all(bool(value) for value in report.get("acceptance", {}).values())
        or report.get("effective_batch_analysis", {}).get(
            "minimum_consistent_effective_batch_size"
        )
        != 4
        or tuple(tuple(group["artifact_updates"]) for group in groups)
        != contract.artifact_groups
        or not all(bool(group["common_descent_passed"]) for group in groups)
        or report.get("next_gate")
        != "freeze_discarded_accumulated_adamw_preflight_at_minimum_consistent_size"
    ):
        raise ValueError("structured GPP batch4 heterogeneity result is not accepted")
    return report


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_structured_gpp_joint_batch4_contract(args.batch4_contract)
    identity = contract.payload["identity"]
    _accepted_heterogeneity_result(args, contract)
    calibration_contract = load_structured_gpp_joint_gradient_calibration_contract(
        args.calibration_contract
    )
    if (
        calibration_contract.sha256 != identity["calibration_contract_sha256"]
        or sha256_file(args.calibration_report) != identity["calibration_report_sha256"]
    ):
        raise ValueError("structured GPP batch4 calibration identity drift")
    calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    if (
        calibration_report.get("status") != "passed"
        or not all(bool(value) for value in calibration_report["acceptance"].values())
        or calibration_report["coefficient_calibration"]["resolved_coefficients"]
        != {
            SEVEN_SOURCE: contract.coefficients[SEVEN_SOURCE],
            THIRTY_SOURCE: contract.coefficients[THIRTY_SOURCE],
        }
        or float(
            calibration_report["coefficient_calibration"][
                "combined_gradient_p90_clip_threshold"
            ]
        )
        != contract.clip_threshold
    ):
        raise ValueError("structured GPP batch4 calibration result is not accepted")
    if sha256_file(args.prepared_manifest) != identity["prepared_manifest_sha256"]:
        raise ValueError("structured GPP batch4 prepared-manifest identity drift")
    if sha256_file(args.warm_start_report) != identity["warm_start_report_sha256"]:
        raise ValueError("structured GPP batch4 warm-start report hash drift")
    warm_report = json.loads(args.warm_start_report.read_text(encoding="utf-8"))
    if (
        warm_report.get("status") != "passed"
        or warm_report["checkpoint"]["sha256"] != args.warm_start_checkpoint_sha256
        or args.warm_start_checkpoint_sha256 != identity["warm_start_checkpoint_sha256"]
        or warm_report["checkpoint"]["promotion_eligible"] is not False
    ):
        raise ValueError("structured GPP batch4 warm-start result is not accepted")

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

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if amendment.sha256 != identity["flux_basis_amendment_sha256"]:
        raise ValueError("structured GPP batch4 flux-basis identity drift")
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
    target_contract = context["target_contract"]
    if (
        context["training_contract"].sha256 != identity["base_training_contract_sha256"]
        or target_contract.sha256 != identity["target_contract_sha256"]
    ):
        raise ValueError("structured GPP batch4 parent contract identity drift")

    distribution_contract = load_distribution_aware_maintenance_contract(
        args.distribution_contract
    )
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    if structured_contract.sha256 != identity["candidate_contract_sha256"]:
        raise ValueError("structured GPP batch4 candidate contract identity drift")
    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        distribution_report.get("status") != "passed"
        or distribution_report["checkpoint"]["sha256"]
        != args.distribution_checkpoint_sha256
        or args.distribution_checkpoint_sha256
        != identity["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("structured GPP batch4 distribution checkpoint identity drift")
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
        raise ValueError("structured GPP batch4 parent parameter count drift")
    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    initialized_parameters = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    warm_checkpoint = load_structured_gpp_checkpoint(
        args.warm_start_checkpoint,
        expected_sha256=args.warm_start_checkpoint_sha256,
        identity=warm_report["identity"],
        completed_updates=int(identity["warm_start_checkpoint_update"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(identity["warm_start_clip_threshold"]),
    )
    specialist, optimizer = jax.device_put(
        (warm_checkpoint["specialist"], warm_checkpoint["optimizer"])
    )
    del warm_checkpoint
    parameters = replace_specialist(initialized_parameters, specialist)
    if (
        _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
        or specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
        or int(np.asarray(optimizer.step)) != 250
    ):
        raise ValueError("structured GPP batch4 endpoint identity drift")

    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        science_flux_basis=amendment.flux_basis,
    )
    fixed_parameters = replace_specialist(parameters, {})
    gradient_steps = steps["make_trainable_gradient_steps"](
        fixed_parameters, replace_specialist
    )
    direct_steps = _make_direct_steps(runtime, head, parameters)
    learning_rate = float(contract.payload["optimizer_proposals"]["learning_rate"])
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
    memory_before = _memory_report(runtime["devices"][0])
    group_reports = []

    for group_index, artifact_updates in enumerate(contract.artifact_groups):
        group_started = time.perf_counter()
        records = []
        source_gradients = {name: [] for name in GRADIENT_SOURCES}
        for artifact_index, update in enumerate(artifact_updates):
            artifact = _load_artifact(args, context, update)
            record_started = time.perf_counter()
            direct, direct_gradient, direct_forward_seconds, direct_gradient_seconds = (
                _direct_gradient(jax, jnp, specialist, artifact.batch, direct_steps)
            )
            one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
            seven, seven_hard, seven_gradient, seven_timing = _horizon_gradient(
                jax,
                jnp,
                parameters,
                specialist,
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
                specialist,
                artifact.batch,
                steps,
                gradient_steps,
                horizon=30,
                flux_basis=amendment.flux_basis,
            )
            host_gradients = jax.device_get(
                {
                    DIRECT_SOURCE: direct_gradient,
                    SEVEN_SOURCE: seven_gradient,
                    THIRTY_SOURCE: thirty_gradient,
                }
            )
            for name in GRADIENT_SOURCES:
                source_gradients[name].append(host_gradients[name])
            runtime_year = int(context["schedule"]["entries"][update - 1]["runtime_year"])
            if runtime_year != RUNTIME_YEAR_GROUPS[group_index][artifact_index]:
                raise ValueError("structured GPP batch4 runtime-year identity drift")
            records.append(
                {
                    "artifact_update": update,
                    "runtime_year": runtime_year,
                    "before": _source_losses(direct, seven, thirty),
                    "hard_counts_before": {
                        "one_day": _hard_host(jax.device_get(one_hard)),
                        "seven_day": _hard_host(jax.device_get(seven_hard)),
                        "thirty_day": _hard_host(jax.device_get(thirty_hard)),
                    },
                    "timing_seconds": {
                        "direct_forward": direct_forward_seconds,
                        "direct_gradient": direct_gradient_seconds,
                        "seven_day": seven_timing,
                        "thirty_day": thirty_timing,
                        "pre_proposal_total": time.perf_counter() - record_started,
                    },
                }
            )
            del direct_gradient, seven_gradient, thirty_gradient

        mean_gradients_host = {
            name: _mean_host_gradients(jax, tuple(values))
            for name, values in source_gradients.items()
        }
        combined_gradient_host = _combine_host_gradients(
            jax, mean_gradients_host, contract.coefficients
        )
        combined_gradient = jax.device_put(combined_gradient_host)
        update_started = time.perf_counter()
        updated_specialist, optimizer_after, raw_norm, clip_scale = optimizer_step(
            specialist, combined_gradient, optimizer
        )
        jax.block_until_ready((updated_specialist, optimizer_after, raw_norm, clip_scale))
        optimizer_seconds = time.perf_counter() - update_started
        proposed_parameters = replace_specialist(parameters, updated_specialist)

        for record in records:
            post_started = time.perf_counter()
            batch = _load_artifact(
                args, context, int(record["artifact_update"])
            ).batch
            direct_after, direct_seconds = _direct_forward(
                jax, updated_specialist, batch, direct_steps
            )
            one_hard_after = _one_day_hard(jax, jnp, proposed_parameters, batch, steps)
            seven_after, seven_hard_after, seven_seconds = _horizon_forward(
                jax,
                jnp,
                proposed_parameters,
                batch,
                steps,
                horizon=7,
                flux_basis=amendment.flux_basis,
            )
            thirty_after, thirty_hard_after, thirty_seconds = _horizon_forward(
                jax,
                jnp,
                proposed_parameters,
                batch,
                steps,
                horizon=30,
                flux_basis=amendment.flux_basis,
            )
            after = _source_losses(direct_after, seven_after, thirty_after)
            record["after"] = after
            record["source_changes"] = _loss_change(record["before"], after)
            record["hard_counts_after"] = {
                "one_day": _hard_host(jax.device_get(one_hard_after)),
                "seven_day": _hard_host(jax.device_get(seven_hard_after)),
                "thirty_day": _hard_host(jax.device_get(thirty_hard_after)),
            }
            record["timing_seconds"]["post_update_direct_forward"] = direct_seconds
            record["timing_seconds"]["post_update_seven_day_forward"] = seven_seconds
            record["timing_seconds"]["post_update_thirty_day_forward"] = thirty_seconds
            record["timing_seconds"]["post_proposal_total"] = time.perf_counter() - post_started

        mean_before = mean_source_losses(tuple(record["before"] for record in records))
        mean_after = mean_source_losses(tuple(record["after"] for record in records))
        mean_changes = _loss_change(mean_before, mean_after)
        objective_before = structured_gpp_joint_objectives(mean_before, contract.coefficients)
        objective_after = structured_gpp_joint_objectives(mean_after, contract.coefficients)
        step = jax.tree_util.tree_map(
            lambda after, before: after - before, updated_specialist, specialist
        )
        negative_step = jax.tree_util.tree_map(lambda value: -value, step)
        host_step, host_negative_step = jax.device_get((step, negative_step))
        source_gradient_reports = {
            name: _tree_numeric_report(jax, gradient)
            for name, gradient in mean_gradients_host.items()
        }
        combined_report = _tree_numeric_report(jax, combined_gradient_host)
        source_to_negative_step = {
            name: _tree_cosine(jax, gradient, host_negative_step)
            for name, gradient in mean_gradients_host.items()
        }
        step_report = _tree_numeric_report(jax, host_step)
        step_components = _component_gradient_reports(jax, host_step)
        optimizer_report = _tree_numeric_report(jax, jax.device_get(optimizer_after))
        retained = _retained_parameter_report(jax, parameters, proposed_parameters)
        gradients_passed = all(
            value["all_finite"]
            and value["element_count"] == SPECIALIST_PARAMETER_COUNT
            and value["l2_norm"] > 0.0
            and value["nonzero_leaf_count"] == value["leaf_count"]
            for value in source_gradient_reports.values()
        )
        combined_passed = (
            combined_report["all_finite"]
            and combined_report["element_count"] == SPECIALIST_PARAMETER_COUNT
            and combined_report["l2_norm"] > 0.0
            and combined_report["nonzero_leaf_count"] == combined_report["leaf_count"]
        )
        realization_passed = (
            all(np.isfinite(value) and value > 0.0 for value in source_to_negative_step.values())
            and all(value["after"] < value["before"] for value in mean_changes.values())
            and objective_after["weighted_science"] < objective_before["weighted_science"]
            and objective_after["joint"] < objective_before["joint"]
            and step_report["all_finite"]
            and step_report["l2_norm"] > 0.0
            and all(
                value["all_finite"] and value["l2_norm"] > 0.0
                for value in step_components.values()
            )
            and optimizer_report["all_finite"]
            and int(np.asarray(optimizer_after.step)) == 251
            and float(np.asarray(raw_norm)) > 0.0
            and 0.0 < float(np.asarray(clip_scale)) <= 1.0
        )
        physical_passed = all(
            _hard_failure_count(value) == 0
            for record in records
            for value in (
                *record["hard_counts_before"].values(),
                *record["hard_counts_after"].values(),
            )
        )
        group_passed = (
            gradients_passed
            and combined_passed
            and realization_passed
            and physical_passed
            and retained["exactly_unchanged"]
        )
        group_reports.append(
            {
                "group_index": group_index,
                "artifact_updates": list(artifact_updates),
                "status": "passed" if group_passed else "failed",
                "proposal_initial_optimizer_step": int(np.asarray(optimizer.step)),
                "proposal_uses_original_warm_start_state": True,
                "per_artifact_changes_reported_not_thresholded": True,
                "artifact_records": records,
                "objective": {
                    "coefficients": dict(contract.coefficients),
                    "mean_source_changes": mean_changes,
                    "before": objective_before,
                    "after": objective_after,
                    "joint_relative_change": _relative_change(
                        objective_before["joint"], objective_after["joint"]
                    ),
                    "weighted_science_relative_change": _relative_change(
                        objective_before["weighted_science"],
                        objective_after["weighted_science"],
                    ),
                },
                "gradient": {
                    "aggregate_sources": source_gradient_reports,
                    "aggregate_combined": combined_report,
                    "aggregate_source_pairwise_cosines": {
                        "direct_vs_seven": _tree_cosine(
                            jax,
                            mean_gradients_host[DIRECT_SOURCE],
                            mean_gradients_host[SEVEN_SOURCE],
                        ),
                        "direct_vs_thirty": _tree_cosine(
                            jax,
                            mean_gradients_host[DIRECT_SOURCE],
                            mean_gradients_host[THIRTY_SOURCE],
                        ),
                        "seven_vs_thirty": _tree_cosine(
                            jax,
                            mean_gradients_host[SEVEN_SOURCE],
                            mean_gradients_host[THIRTY_SOURCE],
                        ),
                    },
                    "aggregate_source_to_negative_parameter_step_cosines": (
                        source_to_negative_step
                    ),
                },
                "optimizer_proposal": {
                    "initial_step": int(np.asarray(optimizer.step)),
                    "proposed_step": int(np.asarray(optimizer_after.step)),
                    "learning_rate": learning_rate,
                    "clip_threshold": contract.clip_threshold,
                    "raw_gradient_norm": float(np.asarray(raw_norm)),
                    "clip_scale": float(np.asarray(clip_scale)),
                    "parameter_step": step_report | {"components": step_components},
                    "proposed_optimizer_state": optimizer_report,
                },
                "retained_parent_parameters": retained,
                "acceptance": {
                    "aggregate_source_gradients_passed": gradients_passed,
                    "aggregate_combined_gradient_passed": combined_passed,
                    "optimizer_realization_passed": realization_passed,
                    "physical_hard_gates_passed": physical_passed,
                    "retained_parent_parameters_passed": retained["exactly_unchanged"],
                },
                "timing_seconds": {
                    "optimizer_compile_and_step": optimizer_seconds,
                    "group_total": time.perf_counter() - group_started,
                },
            }
        )
        print(
            f"structured_gpp batch4 group={group_index} "
            f"passed={group_passed} raw_norm={float(np.asarray(raw_norm)):.9g} "
            f"clip_scale={float(np.asarray(clip_scale)):.9g} "
            f"seconds={group_reports[-1]['timing_seconds']['group_total']:.3f}",
            flush=True,
        )
        source_gradients.clear()
        del combined_gradient, updated_specialist, optimizer_after, proposed_parameters

    checkpoint_unchanged = (
        sha256_file(args.warm_start_checkpoint) == args.warm_start_checkpoint_sha256
    )
    proposals_independent = all(
        report["proposal_initial_optimizer_step"] == 250
        and report["proposal_uses_original_warm_start_state"]
        for report in group_reports
    )
    passed = (
        len(group_reports) == 2
        and all(report["status"] == "passed" for report in group_reports)
        and proposals_independent
        and checkpoint_unchanged
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "batch4_contract_sha256": contract.sha256,
            "heterogeneity_contract_sha256": identity["heterogeneity_contract_sha256"],
            "heterogeneity_report_sha256": sha256_file(args.heterogeneity_report),
            "calibration_contract_sha256": calibration_contract.sha256,
            "calibration_report_sha256": sha256_file(args.calibration_report),
            "warm_start_report_sha256": sha256_file(args.warm_start_report),
            "warm_start_checkpoint_sha256": args.warm_start_checkpoint_sha256,
            "candidate_contract_sha256": structured_contract.sha256,
            "distribution_checkpoint_sha256": args.distribution_checkpoint_sha256,
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "base_training_contract_sha256": context["training_contract"].sha256,
            "target_contract_sha256": target_contract.sha256,
            "flux_basis_amendment_sha256": amendment.sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_state_inherited_from_warm_start": True,
            "independent_optimizer_proposals_executed": 2,
            "optimizer_updates_persisted": 0,
            "checkpoint_written": False,
            "recovery_written": False,
            "proposal_retained_after_process_exit": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "candidate_parameter_count": _tree_size(jax, parameters),
            "trainable_specialist_parameter_count": specialist_element_count(specialist),
            "retained_parent_parameter_count": PARENT_PARAMETER_COUNT,
        },
        "effective_batch_size": 4,
        "group_reports": group_reports,
        "acceptance": {
            "required_group_count_passed": len(group_reports) == 2,
            "both_group_proposals_passed": all(
                report["status"] == "passed" for report in group_reports
            ),
            "proposals_independent_from_same_warm_start_passed": proposals_independent,
            "warm_start_checkpoint_unchanged": checkpoint_unchanged,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
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
    parser.add_argument("--warm-start-report", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint-sha256", required=True)
    parser.add_argument("--calibration-contract", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--heterogeneity-contract", type=Path, required=True)
    parser.add_argument("--heterogeneity-report", type=Path, required=True)
    parser.add_argument("--batch4-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
