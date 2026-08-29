"""Execute one discarded structured-GPP joint optimizer proposal."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
    load_structured_gpp_joint_gradient_calibration_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_update import (
    load_structured_gpp_joint_update_preflight_contract,
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
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact, _replace_adapter
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
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
from scripts.hpc.run_target_oriented_structured_gpp_gpu_admission import _component_gradient_reports
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _direct_forward,
    _hard_host,
    _horizon_forward,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_update_preflight_result_v1"
PARENT_PARAMETER_COUNT = 93_504_164
ADAPTER_PARAMETER_COUNT = 9_750_529


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _combine_gradients(
    jax: Any,
    gradients: Mapping[str, Any],
    coefficients: Mapping[str, float],
) -> Any:
    return jax.tree_util.tree_map(
        lambda direct, seven, thirty: (
            direct
            + np.float32(coefficients[SEVEN_SOURCE]) * seven
            + np.float32(coefficients[THIRTY_SOURCE]) * thirty
        ),
        gradients[DIRECT_SOURCE],
        gradients[SEVEN_SOURCE],
        gradients[THIRTY_SOURCE],
    )


def _source_losses(direct: Any, seven: Any, thirty: Any) -> Mapping[str, Any]:
    return {
        DIRECT_SOURCE: structured_gpp_direct_loss(direct),
        SEVEN_SOURCE: seven.total,
        THIRTY_SOURCE: thirty.total,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_structured_gpp_joint_update_preflight_contract(args.joint_preflight_contract)
    identity = contract.payload["identity"]
    calibration_contract = load_structured_gpp_joint_gradient_calibration_contract(args.calibration_contract)
    if calibration_contract.sha256 != identity["calibration_contract_sha256"]:
        raise ValueError("structured GPP joint-update calibration contract identity drift")
    if sha256_file(args.calibration_report) != identity["calibration_report_sha256"]:
        raise ValueError("structured GPP joint-update calibration report hash drift")
    calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    horizon_coefficients = calibration_report.get("coefficient_calibration", {}).get(
        "resolved_coefficients", {}
    )
    if (
        calibration_report.get("status") != "passed"
        or calibration_report.get("identity", {}).get("git_head")
        != identity["calibration_execution_git_head"]
        or calibration_report.get("identity", {}).get("calibration_contract_sha256")
        != calibration_contract.sha256
        or horizon_coefficients
        != {
            SEVEN_SOURCE: contract.coefficients[SEVEN_SOURCE],
            THIRTY_SOURCE: contract.coefficients[THIRTY_SOURCE],
        }
    ):
        raise ValueError("structured GPP joint-update calibration result is not accepted")
    if (
        float(calibration_report["coefficient_calibration"]["combined_gradient_p90_clip_threshold"])
        != contract.clip_threshold
        or not all(bool(value) for value in calibration_report["acceptance"].values())
        or calibration_report["next_gate"]
        != "freeze_discarded_joint_update_preflight_using_calibrated_coefficients"
    ):
        raise ValueError("structured GPP joint-update calibrated policy drift")
    if sha256_file(args.prepared_manifest) != identity["prepared_manifest_sha256"]:
        raise ValueError("structured GPP joint-update prepared-manifest identity drift")
    if sha256_file(args.warm_start_report) != identity["warm_start_report_sha256"]:
        raise ValueError("structured GPP joint-update warm-start report hash drift")
    warm_report = json.loads(args.warm_start_report.read_text(encoding="utf-8"))
    if (
        warm_report.get("status") != "passed"
        or warm_report["checkpoint"]["sha256"] != args.warm_start_checkpoint_sha256
        or args.warm_start_checkpoint_sha256 != identity["warm_start_checkpoint_sha256"]
        or warm_report["checkpoint"]["promotion_eligible"] is not False
    ):
        raise ValueError("structured GPP joint-update warm-start result is not accepted")

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
        raise ValueError("structured GPP joint-update flux-basis identity drift")
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
        raise ValueError("structured GPP joint-update parent contract identity drift")

    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    if structured_contract.sha256 != identity["candidate_contract_sha256"]:
        raise ValueError("structured GPP joint-update candidate contract identity drift")
    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        distribution_report.get("status") != "passed"
        or distribution_report["checkpoint"]["sha256"] != args.distribution_checkpoint_sha256
        or args.distribution_checkpoint_sha256 != identity["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("structured GPP joint-update distribution checkpoint identity drift")
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
        raise ValueError("structured GPP joint-update parent parameter count drift")
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
        raise ValueError("structured GPP joint-update endpoint identity drift")

    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        science_flux_basis=amendment.flux_basis,
    )
    fixed_parameters = replace_specialist(parameters, {})
    gradient_steps = steps["make_trainable_gradient_steps"](fixed_parameters, replace_specialist)
    direct_steps = _make_direct_steps(runtime, head, parameters)
    memory_before = _memory_report(runtime["devices"][0])

    artifact = _load_artifact(args, context, contract.artifact_update)
    proposal_started = time.perf_counter()
    direct_before, direct_gradient, direct_forward_seconds, direct_gradient_seconds = _direct_gradient(
        jax, jnp, specialist, artifact.batch, direct_steps
    )
    one_hard_before = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
    seven_before, seven_hard_before, seven_gradient, seven_timing = _horizon_gradient(
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
    thirty_before, thirty_hard_before, thirty_gradient, thirty_timing = _horizon_gradient(
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
    gradients = {
        DIRECT_SOURCE: direct_gradient,
        SEVEN_SOURCE: seven_gradient,
        THIRTY_SOURCE: thirty_gradient,
    }
    combined_gradient = _combine_gradients(jax, gradients, contract.coefficients)
    jax.block_until_ready(combined_gradient)

    learning_rate = float(contract.payload["optimizer_proposal"]["learning_rate"])
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
    update_started = time.perf_counter()
    updated_specialist, optimizer_after, raw_norm, clip_scale = optimizer_step(
        specialist, combined_gradient, optimizer
    )
    jax.block_until_ready((updated_specialist, optimizer_after, raw_norm, clip_scale))
    optimizer_seconds = time.perf_counter() - update_started
    proposed_parameters = replace_specialist(parameters, updated_specialist)

    direct_after, post_direct_seconds = _direct_forward(
        jax, updated_specialist, artifact.batch, direct_steps
    )
    one_hard_after = _one_day_hard(jax, jnp, proposed_parameters, artifact.batch, steps)
    seven_after, seven_hard_after, post_seven_seconds = _horizon_forward(
        jax,
        jnp,
        proposed_parameters,
        artifact.batch,
        steps,
        horizon=7,
        flux_basis=amendment.flux_basis,
    )
    thirty_after, thirty_hard_after, post_thirty_seconds = _horizon_forward(
        jax,
        jnp,
        proposed_parameters,
        artifact.batch,
        steps,
        horizon=30,
        flux_basis=amendment.flux_basis,
    )

    source_before = _source_losses(direct_before, seven_before, thirty_before)
    source_after = _source_losses(direct_after, seven_after, thirty_after)
    objective_before = structured_gpp_joint_objectives(source_before, contract.coefficients)
    objective_after = structured_gpp_joint_objectives(source_after, contract.coefficients)
    source_changes = {
        name: {
            "before": float(np.asarray(source_before[name])),
            "after": float(np.asarray(source_after[name])),
            "relative_change": (
                float(np.asarray(source_after[name])) - float(np.asarray(source_before[name]))
            )
            / float(np.asarray(source_before[name])),
        }
        for name in GRADIENT_SOURCES
    }

    step = jax.tree_util.tree_map(lambda after, before: after - before, updated_specialist, specialist)
    negative_step = jax.tree_util.tree_map(lambda value: -value, step)
    host_gradients = jax.device_get(gradients | {"combined": combined_gradient})
    host_step, host_negative_step = jax.device_get((step, negative_step))
    source_gradient_reports = {
        name: _tree_numeric_report(jax, host_gradients[name]) for name in GRADIENT_SOURCES
    }
    combined_report = _tree_numeric_report(jax, host_gradients["combined"])
    source_to_negative_step = {
        name: _tree_cosine(jax, host_gradients[name], host_negative_step)
        for name in GRADIENT_SOURCES
    }
    step_report = _tree_numeric_report(jax, host_step)
    step_components = _component_gradient_reports(jax, host_step)
    optimizer_report = _tree_numeric_report(jax, jax.device_get(optimizer_after))
    retained = _retained_parameter_report(jax, parameters, proposed_parameters)
    checkpoint_unchanged = sha256_file(args.warm_start_checkpoint) == args.warm_start_checkpoint_sha256
    hard_before = {
        "one_day": _hard_host(jax.device_get(one_hard_before)),
        "seven_day": _hard_host(jax.device_get(seven_hard_before)),
        "thirty_day": _hard_host(jax.device_get(thirty_hard_before)),
    }
    hard_after = {
        "one_day": _hard_host(jax.device_get(one_hard_after)),
        "seven_day": _hard_host(jax.device_get(seven_hard_after)),
        "thirty_day": _hard_host(jax.device_get(thirty_hard_after)),
    }

    source_gradients_passed = all(
        value["all_finite"]
        and value["element_count"] == SPECIALIST_PARAMETER_COUNT
        and value["l2_norm"] > 0.0
        and value["nonzero_leaf_count"] == value["leaf_count"]
        for value in source_gradient_reports.values()
    )
    combined_gradient_passed = (
        combined_report["all_finite"]
        and combined_report["element_count"] == SPECIALIST_PARAMETER_COUNT
        and combined_report["l2_norm"] > 0.0
        and combined_report["nonzero_leaf_count"] == combined_report["leaf_count"]
    )
    optimizer_realization_passed = (
        all(np.isfinite(value) and value > 0.0 for value in source_to_negative_step.values())
        and all(value["after"] < value["before"] for value in source_changes.values())
        and objective_after["weighted_science"] < objective_before["weighted_science"]
        and objective_after["joint"] < objective_before["joint"]
        and step_report["all_finite"]
        and step_report["l2_norm"] > 0.0
        and all(value["all_finite"] and value["l2_norm"] > 0.0 for value in step_components.values())
        and optimizer_report["all_finite"]
        and int(np.asarray(optimizer_after.step)) == 251
        and float(np.asarray(raw_norm)) > 0.0
        and 0.0 < float(np.asarray(clip_scale)) <= 1.0
    )
    physical_passed = all(
        _hard_failure_count(value) == 0 for value in (*hard_before.values(), *hard_after.values())
    )
    passed = (
        source_gradients_passed
        and combined_gradient_passed
        and optimizer_realization_passed
        and physical_passed
        and retained["exactly_unchanged"]
        and checkpoint_unchanged
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "joint_preflight_contract_sha256": contract.sha256,
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
            "optimizer_updates_executed": 1,
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
            "retained_parent_parameters": retained,
        },
        "artifact": {
            "prepared_update": contract.artifact_update,
            "runtime_year": int(context["schedule"]["entries"][contract.artifact_update - 1]["runtime_year"]),
            "direct_gpp_teacher_days": 208,
            "seven_day_windows": 8,
            "thirty_day_windows": 4,
        },
        "objective": {
            "coefficients": dict(contract.coefficients),
            "source_changes": source_changes,
            "before": objective_before,
            "after": objective_after,
            "joint_relative_change": (objective_after["joint"] - objective_before["joint"])
            / objective_before["joint"],
            "weighted_science_relative_change": (
                objective_after["weighted_science"] - objective_before["weighted_science"]
            )
            / objective_before["weighted_science"],
        },
        "gradient": {
            "sources": source_gradient_reports,
            "combined": combined_report,
            "source_pairwise_cosines": {
                "direct_vs_seven": _tree_cosine(
                    jax, host_gradients[DIRECT_SOURCE], host_gradients[SEVEN_SOURCE]
                ),
                "direct_vs_thirty": _tree_cosine(
                    jax, host_gradients[DIRECT_SOURCE], host_gradients[THIRTY_SOURCE]
                ),
                "seven_vs_thirty": _tree_cosine(
                    jax, host_gradients[SEVEN_SOURCE], host_gradients[THIRTY_SOURCE]
                ),
            },
            "source_to_negative_parameter_step_cosines": source_to_negative_step,
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
        "hard_counts_before": hard_before,
        "hard_counts_after": hard_after,
        "acceptance": {
            "source_gradients_passed": source_gradients_passed,
            "combined_gradient_passed": combined_gradient_passed,
            "optimizer_realization_passed": optimizer_realization_passed,
            "physical_hard_gates_passed": physical_passed,
            "retained_parent_parameters_passed": retained["exactly_unchanged"],
            "warm_start_checkpoint_unchanged": checkpoint_unchanged,
        },
        "timing_seconds": {
            "direct_forward": direct_forward_seconds,
            "direct_gradient": direct_gradient_seconds,
            "seven_day": seven_timing,
            "thirty_day": thirty_timing,
            "optimizer_compile_and_step": optimizer_seconds,
            "post_update_direct_forward": post_direct_seconds,
            "post_update_seven_day_forward": post_seven_seconds,
            "post_update_thirty_day_forward": post_thirty_seconds,
            "proposal_total": time.perf_counter() - proposal_started,
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
    parser.add_argument("--joint-preflight-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
