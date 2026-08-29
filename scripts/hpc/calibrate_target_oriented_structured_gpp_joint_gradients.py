"""Calibrate train-only 7/30-day gradients at the structured-GPP warm endpoint."""

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
    combined_gradient_p90_clip,
    load_structured_gpp_joint_gradient_calibration_contract,
    resolve_structured_gpp_joint_gradient_coefficients,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    load_structured_gpp_warm_start_contract,
    replace_specialist,
    specialist_element_count,
    structured_gpp_direct_loss,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_science_window_sufficient_statistics,
    science_window_from_sufficient_statistics,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    _add_gradient,
    _load_context,
    _tree_size,
    _zero_gradient,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact, _replace_adapter
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED, _tree_numeric_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _hard_failure_count,
    _make_steps,
    _merge_statistics,
    _slice,
    _zero_hard,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
    _direct_gradient,
    _hard_host,
    _make_direct_steps,
    _one_day_hard,
    _retained_parameter_report,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_gradient_calibration_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _tree_cosine(jax: Any, left: Any, right: Any) -> float:
    dot = 0.0
    left_squared = 0.0
    right_squared = 0.0
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree:
        raise ValueError("structured GPP gradient cosine tree drift")
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf, dtype=np.float64)
        right_array = np.asarray(right_leaf, dtype=np.float64)
        dot += float(np.sum(left_array * right_array, dtype=np.float64))
        left_squared += float(np.sum(left_array * left_array, dtype=np.float64))
        right_squared += float(np.sum(right_array * right_array, dtype=np.float64))
    denominator = np.sqrt(left_squared * right_squared)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("structured GPP gradient cosine denominator is invalid")
    return float(np.clip(dot / denominator, -1.0, 1.0))


def _combine_host_gradients(jax: Any, gradients: Mapping[str, Any], coefficients: Mapping[str, float]) -> Any:
    return jax.tree_util.tree_map(
        lambda direct, seven, thirty: (
            np.asarray(direct, dtype=np.float32)
            + np.float32(coefficients[SEVEN_SOURCE]) * np.asarray(seven, dtype=np.float32)
            + np.float32(coefficients[THIRTY_SOURCE]) * np.asarray(thirty, dtype=np.float32)
        ),
        gradients[DIRECT_SOURCE],
        gradients[SEVEN_SOURCE],
        gradients[THIRTY_SOURCE],
    )


def _science_record(value: Any) -> Mapping[str, float]:
    return {
        "total": float(np.asarray(value.total)),
        "flux_mse": float(np.asarray(value.flux_mse)),
        "flux_bias_squared": float(np.asarray(value.flux_bias_squared)),
        "stock_mse": float(np.asarray(value.stock_mse)),
        "stock_bias_squared": float(np.asarray(value.stock_bias_squared)),
    }


def _horizon_gradient(
    jax: Any,
    jnp: Any,
    parameters: Mapping[str, Any],
    specialist: Any,
    batch: Any,
    steps: Mapping[str, Any],
    gradient_steps: Mapping[str, Any],
    *,
    horizon: int,
    flux_basis: tuple[str, ...],
) -> tuple[Any, Any, Any, Mapping[str, float]]:
    source = batch.seven_day if horizon == 7 else batch.thirty_day
    count = 8 if horizon == 7 else 4
    name = "seven" if horizon == 7 else "thirty"
    microbatches = tuple(_slice(source, index, index + 1) for index in range(count))

    forward_started = time.perf_counter()
    forward = [steps[f"{name}_forward"](parameters, value) for value in microbatches]
    jax.block_until_ready(forward)
    total = _merge_statistics(add_science_window_sufficient_statistics, [value[0] for value in forward])
    hard = _zero_hard(jnp)
    for value in forward:
        hard = _add_hard(jax, hard, value[1])
    result = science_window_from_sufficient_statistics(total, flux_basis=flux_basis)
    jax.block_until_ready((result, hard))
    forward_seconds = time.perf_counter() - forward_started

    gradient_started = time.perf_counter()
    gradient = _zero_gradient(jax, jnp, specialist)
    for microbatch in microbatches:
        _value, current = gradient_steps[f"{name}_gradient"](specialist, microbatch, total)
        gradient = _add_gradient(jax, jnp, gradient, current)
    jax.block_until_ready(gradient)
    gradient_seconds = time.perf_counter() - gradient_started
    print(
        f"structured_gpp joint_calibration horizon={horizon} "
        f"forward={forward_seconds:.3f} gradient={gradient_seconds:.3f}",
        flush=True,
    )
    return result, hard, gradient, {
        "forward": forward_seconds,
        "gradient": gradient_seconds,
        "total": forward_seconds + gradient_seconds,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_structured_gpp_joint_gradient_calibration_contract(args.calibration_contract)
    identity_contract = contract.payload["identity"]
    warm = load_structured_gpp_warm_start_contract(args.warm_start_contract)
    if warm.sha256 != identity_contract["warm_start_contract_sha256"]:
        raise ValueError("structured GPP joint calibration warm-start contract identity drift")
    if sha256_file(args.warm_start_report) != identity_contract["warm_start_report_sha256"]:
        raise ValueError("structured GPP joint calibration warm-start report hash drift")
    warm_report = json.loads(args.warm_start_report.read_text(encoding="utf-8"))
    if (
        warm_report.get("status") != "passed"
        or warm_report["endpoint_comparison"]["capability_gate_passed"] is not True
        or warm_report["checkpoint"]["sha256"] != args.warm_start_checkpoint_sha256
        or args.warm_start_checkpoint_sha256 != identity_contract["warm_start_checkpoint_sha256"]
        or warm_report["checkpoint"]["promotion_eligible"] is not False
    ):
        raise ValueError("structured GPP joint calibration warm-start result is not accepted")
    if sha256_file(args.prepared_manifest) != identity_contract["prepared_manifest_sha256"]:
        raise ValueError("structured GPP joint calibration prepared-manifest identity drift")

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
    if amendment.sha256 != identity_contract["flux_basis_amendment_sha256"]:
        raise ValueError("structured GPP joint calibration flux-basis identity drift")
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
        context["training_contract"].sha256 != identity_contract["base_training_contract_sha256"]
        or target_contract.sha256 != identity_contract["target_contract_sha256"]
    ):
        raise ValueError("structured GPP joint calibration parent contract identity drift")

    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    if structured_contract.sha256 != identity_contract["candidate_contract_sha256"]:
        raise ValueError("structured GPP joint calibration candidate contract identity drift")
    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        distribution_report.get("status") != "passed"
        or distribution_report["checkpoint"]["sha256"] != args.distribution_checkpoint_sha256
        or args.distribution_checkpoint_sha256 != identity_contract["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("structured GPP joint calibration distribution checkpoint identity drift")
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
        raise ValueError("structured GPP joint calibration parent parameter count drift")
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
        completed_updates=int(identity_contract["warm_start_checkpoint_update"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(identity_contract["warm_start_clip_threshold"]),
    )
    specialist = jax.device_put(checkpoint["specialist"])
    del checkpoint
    parameters = replace_specialist(initialized_parameters, specialist)
    if (
        _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
        or specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
    ):
        raise ValueError("structured GPP joint calibration candidate parameter count drift")

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
    stored_gradients: list[Mapping[str, Any]] = []
    for update in contract.calibration_updates:
        artifact = _load_artifact(args, context, update)
        record_started = time.perf_counter()
        direct_total, direct_gradient, direct_forward_seconds, direct_gradient_seconds = _direct_gradient(
            jax,
            jnp,
            specialist,
            artifact.batch,
            direct_steps,
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
        source_reports = {
            name: _tree_numeric_report(jax, host_gradients[name])
            for name in GRADIENT_SOURCES
        }
        records.append(
            {
                "artifact_update": update,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "objectives": {
                    DIRECT_SOURCE: {
                        "total": float(np.asarray(structured_gpp_direct_loss(direct_total))),
                        "evaluated_values": int(np.asarray(direct_total.count)),
                    },
                    SEVEN_SOURCE: _science_record(seven),
                    THIRTY_SOURCE: _science_record(thirty),
                },
                "source_gradients": source_reports,
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
        stored_gradients.append(host_gradients)
        print(
            f"structured_gpp joint_calibration artifact={update} "
            f"direct={source_reports[DIRECT_SOURCE]['l2_norm']:.9g} "
            f"seven={source_reports[SEVEN_SOURCE]['l2_norm']:.9g} "
            f"thirty={source_reports[THIRTY_SOURCE]['l2_norm']:.9g} "
            f"seconds={records[-1]['timing_seconds']['total']:.3f}",
            flush=True,
        )
        del direct_gradient, seven_gradient, thirty_gradient

    coefficients = resolve_structured_gpp_joint_gradient_coefficients(records, contract)
    combined_norms = []
    for record, gradients in zip(records, stored_gradients, strict=True):
        combined = _combine_host_gradients(jax, gradients, coefficients.resolved_coefficients)
        combined_report = _tree_numeric_report(jax, combined)
        combined_norms.append(float(combined_report["l2_norm"]))
        record["combined_gradient"] = combined_report
        record["source_to_combined_cosines"] = {
            name: _tree_cosine(jax, gradients[name], combined)
            for name in GRADIENT_SOURCES
        }
        del combined
    stored_gradients.clear()

    calibration_policy = contract.payload["coefficient_calibration"]
    clip_threshold = combined_gradient_p90_clip(
        combined_norms,
        minimum=float(calibration_policy["combined_gradient_clip_minimum"]),
    )
    source_gradients_passed = all(
        report["all_finite"]
        and report["element_count"] == SPECIALIST_PARAMETER_COUNT
        and report["l2_norm"] > 0.0
        and report["nonzero_leaf_count"] == report["leaf_count"]
        for record in records
        for report in record["source_gradients"].values()
    )
    combined_gradients_passed = all(
        record["combined_gradient"]["all_finite"]
        and record["combined_gradient"]["element_count"] == SPECIALIST_PARAMETER_COUNT
        and record["combined_gradient"]["l2_norm"] > 0.0
        and record["combined_gradient"]["nonzero_leaf_count"] == record["combined_gradient"]["leaf_count"]
        for record in records
    )
    objectives_passed = all(
        np.isfinite(float(objective["total"])) and float(objective["total"]) > 0.0
        for record in records
        for objective in record["objectives"].values()
    ) and all(record["objectives"][DIRECT_SOURCE]["evaluated_values"] == 208 for record in records)
    hard_passed = all(
        _hard_failure_count(counts) == 0
        for record in records
        for counts in record["hard_counts"].values()
    )
    retained = _retained_parameter_report(jax, parameters, parameters)
    checkpoint_unchanged = sha256_file(args.warm_start_checkpoint) == args.warm_start_checkpoint_sha256
    passed = (
        source_gradients_passed
        and combined_gradients_passed
        and objectives_passed
        and hard_passed
        and coefficients.coefficients_inside_interval
        and np.isfinite(clip_threshold)
        and clip_threshold > 0.0
        and retained["exactly_unchanged"]
        and checkpoint_unchanged
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "calibration_contract_sha256": contract.sha256,
            "warm_start_contract_sha256": warm.sha256,
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
            "warm_start_checkpoint_modified": not checkpoint_unchanged,
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
        "coefficient_calibration": {
            "reference_source": DIRECT_SOURCE,
            "target_gradient_norm_ratios": dict(contract.target_gradient_norm_ratios),
            "resolved_coefficients": dict(coefficients.resolved_coefficients),
            "source_distributions": dict(coefficients.source_distributions),
            "coefficients_inside_interval": coefficients.coefficients_inside_interval,
            "combined_gradient_norms": combined_norms,
            "combined_gradient_p90_clip_threshold": clip_threshold,
        },
        "acceptance": {
            "source_gradients_passed": source_gradients_passed,
            "combined_gradients_passed": combined_gradients_passed,
            "objectives_passed": objectives_passed,
            "physical_hard_gates_passed": hard_passed,
            "retained_parent_parameters_passed": retained["exactly_unchanged"],
            "warm_start_checkpoint_unchanged": checkpoint_unchanged,
        },
        "records": records,
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
    parser.add_argument("--warm-start-contract", type=Path, required=True)
    parser.add_argument("--warm-start-report", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint-sha256", required=True)
    parser.add_argument("--calibration-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
