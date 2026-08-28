"""Calibrate the Teacher-day-start guard without creating an optimizer."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _load_context,
    _tree_size,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    PROBE_SEED,
    _tree_numeric_report,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _effective_gradient,
    _hard_failure_count,
    _make_steps,
    _merge_statistics,
    _slice,
    _zero_hard,
)
from scripts.hpc.run_target_oriented_source_shaped_one_update_admission import (
    _objective_record,
)
from scripts.hpc.target_oriented_training_prepared import (
    load_prepared_target_artifact,
)

SCHEMA_VERSION = "target_oriented_distribution_aware_guard_calibration_result_v1"
CANDIDATE_PARAMETER_COUNT = 93_504_933
ADAPTER_PARAMETER_COUNT = 9_750_529
ADAPTER_KEY = "distribution_aware_maintenance"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _adapter_tree(parameters: Mapping[str, Any]) -> Any:
    return parameters["target_oriented"][ADAPTER_KEY]


def _replace_adapter(
    parameters: Mapping[str, Any],
    adapter: Any,
) -> Mapping[str, Any]:
    target = dict(parameters["target_oriented"])
    target[ADAPTER_KEY] = adapter
    return dict(parameters, target_oriented=target)


def _load_artifact(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    update: int,
) -> Any:
    artifact = load_prepared_target_artifact(
        args.prepared_manifest,
        context["prepared_records"][update],
        training_contract_sha256=context["training_contract"].sha256,
        schedule_entry=context["schedule"]["entries"][update - 1],
    )
    if artifact.trace_signature != context["trace_signature"]:
        raise ValueError(f"distribution-aware calibration trace drift at update {update}")
    return artifact


def _hard_host(value: Any) -> Mapping[str, int]:
    return {name: int(np.asarray(item)) for name, item in value._asdict().items()}


def _teacher_guard_gradient(
    jax: Any,
    jnp: Any,
    parameters: Mapping[str, Any],
    adapter: Any,
    batch: Any,
    steps: Mapping[str, Any],
    gradient_steps: Mapping[str, Any],
    target_contract: Any,
    weights: Any,
) -> tuple[Any, Mapping[str, Any], Any, float, float]:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        add_target_oriented_acceptance_statistics,
        finalize_target_oriented_acceptance_statistics,
    )

    started = time.perf_counter()
    batches = {
        7: [_slice(batch.seven_day, index, index + 1) for index in range(8)],
        30: [_slice(batch.thirty_day, index, index + 1) for index in range(4)],
    }
    forwards = {
        7: [steps["seven_teacher_state_acceptance_forward"](parameters, item) for item in batches[7]],
        30: [steps["thirty_teacher_state_acceptance_forward"](parameters, item) for item in batches[30]],
    }
    jax.block_until_ready(forwards)
    forward_seconds = time.perf_counter() - started
    totals = {
        horizon: _merge_statistics(
            add_target_oriented_acceptance_statistics,
            [value[0] for value in forwards[horizon]],
        )
        for horizon in (7, 30)
    }
    hard = _zero_hard(jnp)
    for horizon in (7, 30):
        for value in forwards[horizon]:
            hard = _add_hard(jax, hard, value[1])

    def zero_gradient() -> Any:
        return jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(value, dtype=jnp.float32),
            adapter,
        )

    def add_gradient(left: Any, right: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda first, second: first + jnp.asarray(second, dtype=jnp.float32),
            left,
            right,
        )

    gradient_started = time.perf_counter()
    gradients = {}
    for horizon, name in ((7, "seven_gradient"), (30, "thirty_gradient")):
        accumulated = zero_gradient()
        for item in batches[horizon]:
            _value, gradient = gradient_steps[name](
                adapter,
                item,
                totals[horizon],
                weights,
            )
            accumulated = add_gradient(accumulated, gradient)
        gradients[str(horizon)] = accumulated
    gradient = add_gradient(gradients["7"], gradients["30"])
    jax.block_until_ready((gradient, gradients, hard))
    gradient_seconds = time.perf_counter() - gradient_started

    acceptance = {
        str(horizon): finalize_target_oriented_acceptance_statistics(
            totals[horizon],
            horizon=horizon,
            target_contract=target_contract,
        )
        for horizon in (7, 30)
    }
    selected = ("maintenance_respiration", "growth_respiration", "npp")
    cell_scores = {
        f"{horizon}.{name}": 0.5
        * (
            float(acceptance[str(horizon)]["keys"][name]["normalized_rmse"]) ** 2
            + float(acceptance[str(horizon)]["keys"][name]["normalized_signed_bias"]) ** 2
        )
        for horizon in (7, 30)
        for name in selected
    }
    objective = float(np.mean(tuple(cell_scores.values())))
    return (
        gradient,
        {
            "total": objective,
            "cell_scores": cell_scores,
            "acceptance": acceptance,
            "gradient_sources": {
                horizon: _tree_numeric_report(jax, jax.device_get(value)) for horizon, value in gradients.items()
            },
        },
        hard,
        forward_seconds,
        gradient_seconds,
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_training import (
        calibrate_teacher_guard_coefficient,
        load_distribution_aware_guard_calibration_contract,
    )

    calibration_contract = load_distribution_aware_guard_calibration_contract(args.calibration_contract)
    identity = calibration_contract.payload["identity"]
    if sha256_file(args.gpu_admission_report) != identity["gpu_admission_report_sha256"]:
        raise ValueError("distribution-aware GPU admission report identity drift")
    admission = json.loads(args.gpu_admission_report.read_text(encoding="utf-8"))
    if (
        admission.get("status") != "passed"
        or admission["policy"]["training_updates"] != 0
        or admission["policy"]["optimizer_created"]
    ):
        raise ValueError("distribution-aware GPU admission is not accepted")
    if sha256_file(args.prepared_manifest) != identity["prepared_manifest_sha256"]:
        raise ValueError("distribution-aware prepared-manifest identity drift")

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if amendment.sha256 != identity["flux_basis_amendment_sha256"]:
        raise ValueError("distribution-aware flux-basis identity drift")
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
    if context["training_contract"].sha256 != identity["training_contract_sha256"]:
        raise ValueError("distribution-aware training-contract identity drift")
    if target_contract.sha256 != identity["target_contract_sha256"]:
        raise ValueError("distribution-aware target-contract identity drift")

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        derive_target_oriented_acceptance_scales,
    )

    candidate_contract = load_distribution_aware_maintenance_contract(args.candidate_contract)
    if candidate_contract.sha256 != identity["candidate_contract_sha256"]:
        raise ValueError("distribution-aware candidate-contract identity drift")
    parameters = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=candidate_contract,
    )
    adapter = _adapter_tree(parameters)
    if _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT:
        raise ValueError("distribution-aware calibration candidate parameter count drift")
    if _tree_size(jax, adapter) != ADAPTER_PARAMETER_COUNT:
        raise ValueError("distribution-aware calibration adapter parameter count drift")

    head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=candidate_contract,
    )
    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        acceptance_scales=acceptance_scales,
        acceptance_flux_names=AUDIT_FLUX_KEYS,
        science_flux_basis=amendment.flux_basis,
    )
    free_gradient_steps = steps["make_trainable_gradient_steps"](
        parameters,
        _replace_adapter,
    )
    guard_gradient_steps = steps["make_trainable_teacher_state_acceptance_gradient_steps"](
        parameters,
        _replace_adapter,
    )
    weights = jnp.asarray(
        [
            1.0 / 6.0 if name in {"maintenance_respiration", "growth_respiration", "npp"} else 0.0
            for name in AUDIT_FLUX_KEYS
        ],
        dtype=jnp.float64,
    )

    memory_before = _memory_report(runtime["devices"][0])
    records = []
    free_norms = []
    guard_norms = []
    for update in calibration_contract.calibration_updates:
        artifact = _load_artifact(args, context, update)
        batch_started = time.perf_counter()
        (
            free_gradient,
            _free_sources,
            free_components,
            _free_details,
            free_hard,
            _free_hard_sources,
            free_forward_seconds,
            free_gradient_seconds,
        ) = _effective_gradient(
            jax,
            jnp,
            parameters,
            artifact.batch,
            steps,
            target_contract,
            science_flux_basis=amendment.flux_basis,
            gradient_parameters=adapter,
            gradient_steps=free_gradient_steps,
        )
        guard_gradient, guard_objective, guard_hard, guard_forward_seconds, guard_gradient_seconds = (
            _teacher_guard_gradient(
                jax,
                jnp,
                parameters,
                adapter,
                artifact.batch,
                steps,
                guard_gradient_steps,
                target_contract,
                weights,
            )
        )
        free_report = _tree_numeric_report(jax, jax.device_get(free_gradient))
        guard_report = _tree_numeric_report(jax, jax.device_get(guard_gradient))
        free_norms.append(float(free_report["l2_norm"]))
        guard_norms.append(float(guard_report["l2_norm"]))
        free_hard_host = _hard_host(jax.device_get(free_hard))
        guard_hard_host = _hard_host(jax.device_get(guard_hard))
        records.append(
            {
                "artifact_update": update,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "free_objective": _objective_record(free_components, target_contract),
                "teacher_guard_objective": guard_objective,
                "free_gradient": free_report,
                "teacher_guard_gradient": guard_report,
                "free_hard_counts": free_hard_host,
                "teacher_guard_hard_counts": guard_hard_host,
                "free_statistics_forward_seconds": free_forward_seconds,
                "free_microbatch_gradient_seconds": free_gradient_seconds,
                "teacher_guard_forward_seconds": guard_forward_seconds,
                "teacher_guard_gradient_seconds": guard_gradient_seconds,
                "total_seconds": time.perf_counter() - batch_started,
            }
        )
        print(
            f"distribution_aware guard_calibration={update} "
            f"free_norm={free_norms[-1]:.9g} guard_norm={guard_norms[-1]:.9g} "
            f"seconds={records[-1]['total_seconds']:.3f}",
            flush=True,
        )
        del free_gradient, guard_gradient

    bounds = calibration_contract.payload["coefficient"]
    coefficient = calibrate_teacher_guard_coefficient(
        free_norms,
        guard_norms,
        minimum=float(bounds["minimum"]),
        maximum=float(bounds["maximum"]),
    )
    required_count = int(calibration_contract.payload["acceptance"]["required_gradient_element_count_per_objective"])
    passed = (
        len(records) == 8
        and all(
            record[name]["all_finite"]
            and record[name]["element_count"] == required_count
            and record[name]["l2_norm"] > 0.0
            for record in records
            for name in ("free_gradient", "teacher_guard_gradient")
        )
        and all(
            _hard_failure_count(record[name]) == 0
            for record in records
            for name in ("free_hard_counts", "teacher_guard_hard_counts")
        )
        and np.isfinite(coefficient.coefficient)
        and float(bounds["minimum"]) <= coefficient.coefficient <= float(bounds["maximum"])
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "calibration_contract_sha256": calibration_contract.sha256,
            "candidate_contract_sha256": candidate_contract.sha256,
            "gpu_admission_report_sha256": sha256_file(args.gpu_admission_report),
            "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "training_contract_sha256": context["training_contract"].sha256,
            "target_contract_sha256": target_contract.sha256,
            "flux_basis_amendment_sha256": amendment.sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_created": False,
            "training_updates": 0,
            "checkpoint_written": False,
            "recovery_written": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "candidate_parameter_count": _tree_size(jax, parameters),
            "trainable_adapter_parameter_count": _tree_size(jax, adapter),
            "parent_parameters_trainable": False,
        },
        "coefficient": {
            "free_gradient_norms": list(coefficient.free_gradient_norms),
            "teacher_guard_gradient_norms": list(coefficient.teacher_guard_gradient_norms),
            "median_free_gradient_norm": coefficient.median_free_gradient_norm,
            "median_teacher_guard_gradient_norm": coefficient.median_teacher_guard_gradient_norm,
            "unclipped": coefficient.unclipped_coefficient,
            "minimum": float(bounds["minimum"]),
            "maximum": float(bounds["maximum"]),
            "value": coefficient.coefficient,
            "clipped": coefficient.clipped,
        },
        "records": records,
        "device_memory": {
            "before_calibration": memory_before,
            "after_calibration": _memory_report(runtime["devices"][0]),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": ("freeze_separate_bounded_training_contract" if passed else "stop_and_repair_guard_calibration"),
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
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--calibration-contract", type=Path, required=True)
    parser.add_argument("--gpu-admission-report", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
