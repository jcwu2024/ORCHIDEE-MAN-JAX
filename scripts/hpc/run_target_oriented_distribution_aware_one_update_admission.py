"""Admit one ephemeral adapter-only update with the frozen Teacher guard."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_training import (
    combine_adapter_gradients,
    load_distribution_aware_bounded_training_contract,
    load_distribution_aware_guard_calibration_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _load_context,
    _tree_size,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _adapter_tree,
    _load_artifact,
    _replace_adapter,
    _teacher_guard_gradient,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    PROBE_SEED,
    _tree_numeric_report,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _effective_gradient,
    _hard_failure_count,
    _make_steps,
)
from scripts.hpc.run_target_oriented_source_shaped_branch_training import (
    _make_optimizer_step,
)
from scripts.hpc.run_target_oriented_source_shaped_one_update_admission import (
    _all_finite,
    _effective_forward,
    _hard_host,
    _objective_record,
)

SCHEMA_VERSION = "target_oriented_distribution_aware_one_update_admission_v1"
CANDIDATE_PARAMETER_COUNT = 93_504_933
ADAPTER_PARAMETER_COUNT = 9_750_529
ADAPTER_KEY = "distribution_aware_maintenance"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _retained_parameter_report(
    jax: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Mapping[str, Any]:
    def retained(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        target = {
            name: value
            for name, value in parameters["target_oriented"].items()
            if name != ADAPTER_KEY
        }
        return dict(parameters, target_oriented=target)

    before_leaves, before_structure = jax.tree_util.tree_flatten(retained(before))
    after_leaves, after_structure = jax.tree_util.tree_flatten(retained(after))
    same_structure = before_structure == after_structure
    same_objects = same_structure and all(
        left is right for left, right in zip(before_leaves, after_leaves, strict=True)
    )
    return {
        "leaf_count": len(before_leaves),
        "element_count": sum(int(np.prod(value.shape, dtype=np.int64)) for value in before_leaves),
        "same_tree_structure": same_structure,
        "same_array_objects": same_objects,
        "exactly_unchanged": same_objects,
    }


def _guard_objective_finite(value: Mapping[str, Any]) -> bool:
    return np.isfinite(float(value["total"])) and all(
        np.isfinite(float(item)) for item in value["cell_scores"].values()
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    bounded = load_distribution_aware_bounded_training_contract(args.bounded_contract)
    calibration = load_distribution_aware_guard_calibration_contract(args.calibration_contract)
    identity = bounded.payload["identity"]
    if calibration.sha256 != identity["guard_calibration_contract_sha256"]:
        raise ValueError("distribution-aware calibration contract identity drift")
    if sha256_file(args.calibration_report) != identity["guard_calibration_report_sha256"]:
        raise ValueError("distribution-aware calibration report identity drift")
    calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    coefficient = float(bounded.payload["objective"]["teacher_guard_coefficient"])
    if calibration_report.get("status") != "passed" or float(calibration_report["coefficient"]["value"]) != coefficient:
        raise ValueError("distribution-aware accepted Teacher-guard coefficient drift")
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
    if (
        context["training_contract"].sha256 != identity["base_training_contract_sha256"]
        or target_contract.sha256 != identity["target_contract_sha256"]
        or args.stage_checkpoint_sha256 != identity["parent_checkpoint_sha256"]
    ):
        raise ValueError("distribution-aware parent or objective identity drift")

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        derive_target_oriented_acceptance_scales,
        initialize_target_oriented_adamw,
        target_oriented_learning_rate,
    )

    candidate = load_distribution_aware_maintenance_contract(args.candidate_contract)
    if candidate.sha256 != identity["candidate_contract_sha256"]:
        raise ValueError("distribution-aware candidate contract identity drift")
    parameters = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=candidate,
    )
    adapter = _adapter_tree(parameters)
    if (
        _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
        or _tree_size(jax, adapter) != ADAPTER_PARAMETER_COUNT
    ):
        raise ValueError("distribution-aware one-update parameter count drift")

    head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=candidate,
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
    free_gradient_steps = steps["make_trainable_gradient_steps"](parameters, _replace_adapter)
    guard_gradient_steps = steps["make_trainable_teacher_state_acceptance_gradient_steps"](
        parameters,
        _replace_adapter,
    )
    guard_weights = jnp.asarray(
        [
            1.0 / 6.0 if name in {"maintenance_respiration", "growth_respiration", "npp"} else 0.0
            for name in AUDIT_FLUX_KEYS
        ],
        dtype=jnp.float64,
    )
    artifact_update = int(bounded.payload["one_update_admission"]["artifact_update"])
    artifact = _load_artifact(args, context, artifact_update)
    memory_before = _memory_report(runtime["devices"][0])
    update_started = time.perf_counter()

    (
        free_gradient,
        _free_sources,
        free_components,
        _free_details,
        free_hard,
        free_hard_sources,
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
    (
        guard_gradient,
        guard_objective,
        guard_hard,
        guard_forward_seconds,
        guard_gradient_seconds,
    ) = _teacher_guard_gradient(
        jax,
        jnp,
        parameters,
        adapter,
        artifact.batch,
        steps,
        guard_gradient_steps,
        target_contract,
        guard_weights,
    )
    combined_gradient = combine_adapter_gradients(
        jax.tree_util,
        free_gradient,
        guard_gradient,
        coefficient,
    )
    jax.block_until_ready((combined_gradient, free_hard, guard_hard))
    free_objective = _objective_record(free_components, target_contract)
    combined_objective_before = float(free_objective["total"] + coefficient * guard_objective["total"])
    gradient_reports = {
        "free": _tree_numeric_report(jax, jax.device_get(free_gradient)),
        "teacher_guard": _tree_numeric_report(jax, jax.device_get(guard_gradient)),
        "combined": _tree_numeric_report(jax, jax.device_get(combined_gradient)),
    }
    hard_before = {
        "free": _hard_host(jax.device_get(free_hard)),
        "teacher_guard": _hard_host(jax.device_get(guard_hard)),
    }

    clip_threshold = float(bounded.optimizer["global_gradient_norm_clip"]["threshold"])
    optimizer = initialize_target_oriented_adamw(adapter)
    learning_rate = target_oriented_learning_rate(bounded, 1)
    optimizer_step = _make_optimizer_step(jax, jnp, bounded, clip_threshold)
    proposal_started = time.perf_counter()
    proposal = optimizer_step(
        adapter,
        combined_gradient,
        optimizer,
        jnp.asarray(learning_rate, dtype=jnp.float32),
    )
    jax.block_until_ready(proposal)
    proposal_seconds = time.perf_counter() - proposal_started
    updated_adapter, optimizer_after = proposal[0], proposal[1]
    proposed_parameters = _replace_adapter(parameters, updated_adapter)
    retained = _retained_parameter_report(jax, parameters, proposed_parameters)
    adapter_step = jax.tree_util.tree_map(
        lambda after, before: after - before,
        updated_adapter,
        adapter,
    )
    adapter_step_report = _tree_numeric_report(jax, jax.device_get(adapter_step))

    free_after, free_hard_after, _free_sources_after, free_after_seconds = _effective_forward(
        jax,
        jnp,
        proposed_parameters,
        artifact.batch,
        steps,
        target_contract,
        amendment.flux_basis,
    )
    (
        _guard_gradient_after,
        guard_after,
        guard_hard_after,
        guard_forward_after_seconds,
        guard_gradient_after_seconds,
    ) = _teacher_guard_gradient(
        jax,
        jnp,
        proposed_parameters,
        updated_adapter,
        artifact.batch,
        steps,
        guard_gradient_steps,
        target_contract,
        guard_weights,
    )
    combined_objective_after = float(free_after["total"] + coefficient * guard_after["total"])
    hard_after = {
        "free": _hard_host(jax.device_get(free_hard_after)),
        "teacher_guard": _hard_host(jax.device_get(guard_hard_after)),
    }
    optimizer_update = int(np.asarray(jax.device_get(optimizer_after.step)))
    raw_norm = float(np.asarray(proposal[2]))
    clip_scale = float(np.asarray(proposal[3]))
    parameter_step_norm = float(np.asarray(proposal[5]))
    passed = (
        _all_finite(free_objective)
        and _all_finite(free_after)
        and _guard_objective_finite(guard_objective)
        and _guard_objective_finite(guard_after)
        and np.isfinite(combined_objective_before)
        and np.isfinite(combined_objective_after)
        and all(
            report["all_finite"]
            and report["element_count"] == ADAPTER_PARAMETER_COUNT
            and report["l2_norm"] > 0.0
            and report["nonzero_leaf_count"] == report["leaf_count"]
            for report in gradient_reports.values()
        )
        and all(
            _hard_failure_count(value) == 0
            for phase in (hard_before, hard_after)
            for value in phase.values()
        )
        and retained["exactly_unchanged"]
        and adapter_step_report["all_finite"]
        and adapter_step_report["l2_norm"] > 0.0
        and int(np.asarray(proposal[4])) == 0
        and optimizer_update == 1
        and raw_norm > 0.0
        and 0.0 < clip_scale <= 1.0
        and parameter_step_norm > 0.0
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "bounded_training_contract_sha256": bounded.sha256,
            "candidate_contract_sha256": candidate.sha256,
            "guard_calibration_contract_sha256": calibration.sha256,
            "guard_calibration_report_sha256": sha256_file(args.calibration_report),
            "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
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
            "optimizer_created": True,
            "training_updates_executed": 1,
            "training_updates_persisted": 0,
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
            "trainable_adapter_parameter_count": _tree_size(jax, adapter),
            "retained_parameters": retained,
        },
        "ephemeral_update": {
            "artifact_update": artifact_update,
            "runtime_year": int(context["schedule"]["entries"][artifact_update - 1]["runtime_year"]),
            "teacher_guard_coefficient": coefficient,
            "learning_rate": learning_rate,
            "clip_threshold": clip_threshold,
            "free_objective_before": free_objective,
            "teacher_guard_objective_before": guard_objective,
            "combined_objective_before": combined_objective_before,
            "free_objective_after": free_after,
            "teacher_guard_objective_after": guard_after,
            "combined_objective_after": combined_objective_after,
            "gradients": gradient_reports,
            "raw_combined_gradient_norm": raw_norm,
            "clip_scale": clip_scale,
            "clipped": clip_scale < 1.0,
            "optimizer_step": optimizer_update,
            "optimizer_parameter_or_state_nonfinite_count": int(np.asarray(proposal[4])),
            "parameter_step_norm": parameter_step_norm,
            "adapter_parameter_step": adapter_step_report,
            "hard_counts_before": hard_before,
            "hard_counts_after": hard_after,
            "timing_seconds": {
                "free_forward": free_forward_seconds,
                "free_gradient": free_gradient_seconds,
                "teacher_guard_forward": guard_forward_seconds,
                "teacher_guard_gradient": guard_gradient_seconds,
                "optimizer_compile_and_step": proposal_seconds,
                "post_update_free_forward": free_after_seconds,
                "post_update_teacher_guard_forward": guard_forward_after_seconds,
                "post_update_teacher_guard_gradient": guard_gradient_after_seconds,
                "total": time.perf_counter() - update_started,
            },
        },
        "device_memory": {
            "before_update": memory_before,
            "after_update": _memory_report(runtime["devices"][0]),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "request_paid_250_update_adapter_training"
            if passed
            else "stop_and_repair_distribution_aware_one_update_admission"
        ),
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
    parser.add_argument("--bounded-contract", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
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

