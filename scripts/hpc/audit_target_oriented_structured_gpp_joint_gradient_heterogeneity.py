"""Audit structured-GPP joint gradients across effective batch sizes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_heterogeneity import (
    grouped_artifact_updates,
    load_structured_gpp_joint_heterogeneity_contract,
    minimum_consistent_effective_batch_size,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
    structured_gpp_direct_loss,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact, _replace_adapter
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _combine_host_gradients,
    _direct_gradient,
    _horizon_gradient,
    _make_direct_steps,
    _one_day_hard,
    _retained_parameter_report,
    _science_record,
    _tree_cosine,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED, _tree_numeric_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import _hard_failure_count, _make_steps
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
    _hard_host,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_gradient_heterogeneity_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _mean_host_gradients(jax: Any, gradients: Sequence[Any]) -> Any:
    values = tuple(gradients)
    if not values:
        raise ValueError("structured GPP heterogeneity mean requires gradients")
    scale = np.float32(1.0 / len(values))
    return jax.tree_util.tree_map(
        lambda *leaves: sum(
            (np.asarray(leaf, dtype=np.float32) for leaf in leaves),
            np.float32(0.0),
        )
        * scale,
        *values,
    )


def _group_report(
    jax: Any,
    updates: tuple[int, ...],
    gradients_by_update: Mapping[int, Mapping[str, Any]],
    coefficients: Mapping[str, float],
) -> Mapping[str, Any]:
    sources = {
        name: _mean_host_gradients(
            jax,
            [gradients_by_update[update][name] for update in updates],
        )
        for name in GRADIENT_SOURCES
    }
    combined = _combine_host_gradients(jax, sources, coefficients)
    source_to_combined = {
        name: _tree_cosine(jax, sources[name], combined) for name in GRADIENT_SOURCES
    }
    return {
        "artifact_updates": list(updates),
        "source_gradients": {
            name: _tree_numeric_report(jax, sources[name]) for name in GRADIENT_SOURCES
        },
        "combined_gradient": _tree_numeric_report(jax, combined),
        "pairwise_gradient_cosines": {
            "direct_vs_seven": _tree_cosine(jax, sources[DIRECT_SOURCE], sources[SEVEN_SOURCE]),
            "direct_vs_thirty": _tree_cosine(jax, sources[DIRECT_SOURCE], sources[THIRTY_SOURCE]),
            "seven_vs_thirty": _tree_cosine(jax, sources[SEVEN_SOURCE], sources[THIRTY_SOURCE]),
        },
        "source_to_combined_cosines": source_to_combined,
        "common_descent_passed": all(value > 0.0 for value in source_to_combined.values()),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_structured_gpp_joint_heterogeneity_contract(args.heterogeneity_contract)
    identity = contract.payload["identity"]
    if sha256_file(args.joint_preflight_report) != identity["joint_preflight_report_sha256"]:
        raise ValueError("structured GPP heterogeneity preflight report hash drift")
    preflight_report = json.loads(args.joint_preflight_report.read_text(encoding="utf-8"))
    if (
        preflight_report.get("status") != "failed"
        or preflight_report["identity"]["joint_preflight_contract_sha256"]
        != identity["joint_preflight_contract_sha256"]
        or preflight_report["acceptance"]["optimizer_realization_passed"] is not False
        or not all(
            bool(preflight_report["acceptance"][name])
            for name in (
                "source_gradients_passed",
                "combined_gradient_passed",
                "physical_hard_gates_passed",
                "retained_parent_parameters_passed",
                "warm_start_checkpoint_unchanged",
            )
        )
        or preflight_report["next_gate"]
        != "stop_before_paid_joint_training_and_review_optimizer_realization"
    ):
        raise ValueError("structured GPP heterogeneity preflight result is not accepted")
    calibration_contract = load_structured_gpp_joint_gradient_calibration_contract(args.calibration_contract)
    if calibration_contract.sha256 != identity["calibration_contract_sha256"]:
        raise ValueError("structured GPP heterogeneity calibration contract identity drift")
    if sha256_file(args.calibration_report) != identity["calibration_report_sha256"]:
        raise ValueError("structured GPP heterogeneity calibration report hash drift")
    calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    if (
        calibration_report.get("status") != "passed"
        or not all(bool(value) for value in calibration_report["acceptance"].values())
        or calibration_report["coefficient_calibration"]["resolved_coefficients"]
        != {
            SEVEN_SOURCE: contract.coefficients[SEVEN_SOURCE],
            THIRTY_SOURCE: contract.coefficients[THIRTY_SOURCE],
        }
    ):
        raise ValueError("structured GPP heterogeneity calibration result is not accepted")
    if sha256_file(args.prepared_manifest) != identity["prepared_manifest_sha256"]:
        raise ValueError("structured GPP heterogeneity prepared-manifest identity drift")
    if sha256_file(args.warm_start_report) != identity["warm_start_report_sha256"]:
        raise ValueError("structured GPP heterogeneity warm-start report hash drift")
    warm_report = json.loads(args.warm_start_report.read_text(encoding="utf-8"))
    if (
        warm_report.get("status") != "passed"
        or warm_report["checkpoint"]["sha256"] != args.warm_start_checkpoint_sha256
        or args.warm_start_checkpoint_sha256 != identity["warm_start_checkpoint_sha256"]
        or warm_report["checkpoint"]["promotion_eligible"] is not False
    ):
        raise ValueError("structured GPP heterogeneity warm-start result is not accepted")

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
        raise ValueError("structured GPP heterogeneity flux-basis identity drift")
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
        raise ValueError("structured GPP heterogeneity parent contract identity drift")

    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    if structured_contract.sha256 != identity["candidate_contract_sha256"]:
        raise ValueError("structured GPP heterogeneity candidate contract identity drift")
    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        distribution_report.get("status") != "passed"
        or distribution_report["checkpoint"]["sha256"] != args.distribution_checkpoint_sha256
        or args.distribution_checkpoint_sha256 != identity["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("structured GPP heterogeneity distribution checkpoint identity drift")
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
        raise ValueError("structured GPP heterogeneity parent parameter count drift")
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
    checkpoint = load_structured_gpp_checkpoint(
        args.warm_start_checkpoint,
        expected_sha256=args.warm_start_checkpoint_sha256,
        identity=warm_report["identity"],
        completed_updates=int(identity["warm_start_checkpoint_update"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(identity["warm_start_clip_threshold"]),
    )
    specialist = jax.device_put(checkpoint["specialist"])
    del checkpoint
    parameters = replace_specialist(initialized_parameters, specialist)
    if (
        _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
        or specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
    ):
        raise ValueError("structured GPP heterogeneity endpoint identity drift")

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

    records: list[dict[str, Any]] = []
    gradients_by_update: dict[int, Mapping[str, Any]] = {}
    expected_years = tuple(int(value) for value in contract.payload["data"]["artifact_runtime_years"])
    for update, expected_year in zip(contract.artifact_updates, expected_years, strict=True):
        artifact = _load_artifact(args, context, update)
        record_started = time.perf_counter()
        direct, direct_gradient, direct_forward_seconds, direct_gradient_seconds = _direct_gradient(
            jax, jnp, specialist, artifact.batch, direct_steps
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
        combined = _combine_host_gradients(jax, host_gradients, contract.coefficients)
        runtime_year = int(context["schedule"]["entries"][update - 1]["runtime_year"])
        if runtime_year != expected_year:
            raise ValueError("structured GPP heterogeneity runtime-year identity drift")
        source_reports = {
            name: _tree_numeric_report(jax, host_gradients[name]) for name in GRADIENT_SOURCES
        }
        records.append(
            {
                "artifact_update": update,
                "runtime_year": runtime_year,
                "objectives": {
                    DIRECT_SOURCE: {
                        "total": float(np.asarray(structured_gpp_direct_loss(direct))),
                        "evaluated_values": int(np.asarray(direct.count)),
                    },
                    SEVEN_SOURCE: _science_record(seven),
                    THIRTY_SOURCE: _science_record(thirty),
                },
                "source_gradients": source_reports,
                "combined_gradient": _tree_numeric_report(jax, combined),
                "pairwise_gradient_cosines": {
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
                "source_to_combined_cosines": {
                    name: _tree_cosine(jax, host_gradients[name], combined)
                    for name in GRADIENT_SOURCES
                },
                "hard_counts": {
                    "one_day": _hard_host(jax.device_get(one_hard)),
                    "seven_day": _hard_host(jax.device_get(seven_hard)),
                    "thirty_day": _hard_host(jax.device_get(thirty_hard)),
                },
                "timing_seconds": {
                    "direct_forward": direct_forward_seconds,
                    "direct_gradient": direct_gradient_seconds,
                    "seven_day": seven_timing,
                    "thirty_day": thirty_timing,
                    "total": time.perf_counter() - record_started,
                },
            }
        )
        gradients_by_update[update] = host_gradients
        print(
            f"structured_gpp heterogeneity artifact={update} "
            f"direct={source_reports[DIRECT_SOURCE]['l2_norm']:.9g} "
            f"seven={source_reports[SEVEN_SOURCE]['l2_norm']:.9g} "
            f"thirty={source_reports[THIRTY_SOURCE]['l2_norm']:.9g} "
            f"seconds={records[-1]['timing_seconds']['total']:.3f}",
            flush=True,
        )
        del direct_gradient, seven_gradient, thirty_gradient, combined

    grouped_reports: dict[int, list[Mapping[str, Any]]] = {}
    for size in contract.effective_batch_sizes:
        grouped_reports[size] = [
            _group_report(jax, updates, gradients_by_update, contract.coefficients)
            for updates in grouped_artifact_updates(contract.artifact_updates, size)
        ]
    minimum_size = minimum_consistent_effective_batch_size(
        grouped_reports, contract.effective_batch_sizes
    )
    gradients_by_update.clear()

    source_passed = all(
        report["all_finite"]
        and report["element_count"] == SPECIALIST_PARAMETER_COUNT
        and report["l2_norm"] > 0.0
        and report["nonzero_leaf_count"] == report["leaf_count"]
        for record in records
        for report in record["source_gradients"].values()
    )
    grouped_passed = all(
        report["all_finite"]
        and report["element_count"] == SPECIALIST_PARAMETER_COUNT
        and report["l2_norm"] > 0.0
        for groups in grouped_reports.values()
        for group in groups
        for report in (*group["source_gradients"].values(), group["combined_gradient"])
    )
    objectives_passed = all(
        np.isfinite(float(objective["total"])) and float(objective["total"]) > 0.0
        for record in records
        for objective in record["objectives"].values()
    ) and all(record["objectives"][DIRECT_SOURCE]["evaluated_values"] == 208 for record in records)
    physical_passed = all(
        _hard_failure_count(value) == 0
        for record in records
        for value in record["hard_counts"].values()
    )
    retained = _retained_parameter_report(jax, parameters, parameters)
    checkpoint_unchanged = sha256_file(args.warm_start_checkpoint) == args.warm_start_checkpoint_sha256
    passed = (
        source_passed
        and grouped_passed
        and objectives_passed
        and physical_passed
        and retained["exactly_unchanged"]
        and checkpoint_unchanged
    )
    if not passed:
        next_gate = contract.payload["next_gate"]["on_execution_failure"]
    elif minimum_size is None:
        next_gate = contract.payload["next_gate"]["on_no_consistent_size"]
    else:
        next_gate = contract.payload["next_gate"]["on_consistent_size_found"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "heterogeneity_contract_sha256": contract.sha256,
            "joint_preflight_report_sha256": sha256_file(args.joint_preflight_report),
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
            "optimizer_created": False,
            "optimizer_updates_executed": 0,
            "checkpoint_written": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "candidate_parameter_count": CANDIDATE_PARAMETER_COUNT,
            "differentiated_specialist_parameter_count": SPECIALIST_PARAMETER_COUNT,
            "retained_parent_parameters": retained,
        },
        "coefficients": dict(contract.coefficients),
        "records": records,
        "effective_batch_analysis": {
            "candidate_sizes": list(contract.effective_batch_sizes),
            "grouped_reports": {str(size): values for size, values in grouped_reports.items()},
            "minimum_consistent_effective_batch_size": minimum_size,
            "classification": (
                "consistent_effective_batch_size_found"
                if minimum_size is not None
                else "no_consistent_effective_batch_size_in_frozen_grid"
            ),
        },
        "acceptance": {
            "source_gradients_passed": source_passed,
            "grouped_gradients_passed": grouped_passed,
            "objectives_passed": objectives_passed,
            "physical_hard_gates_passed": physical_passed,
            "retained_parent_parameters_passed": retained["exactly_unchanged"],
            "warm_start_checkpoint_unchanged": checkpoint_unchanged,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
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
    parser.add_argument("--joint-preflight-report", type=Path, required=True)
    parser.add_argument("--heterogeneity-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
