"""Train the distribution-aware adapter and run the frozen train/train screen."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
    load_distribution_aware_recovery,
    write_distribution_aware_checkpoint,
    write_distribution_aware_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_training import (
    combine_adapter_gradients,
    compare_distribution_aware_train_train_groups,
    load_distribution_aware_bounded_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _load_context,
    _tree_size,
)
from scripts.hpc.audit_target_oriented_flux_protected_virtual_step import _evaluate_group
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
from scripts.hpc.run_target_oriented_distribution_aware_one_update_admission import (
    _guard_objective_finite,
    _retained_parameter_report,
)
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
    _hard_host,
    _objective_record,
)
from scripts.hpc.run_target_oriented_source_shaped_train_train_zero_panel import _groups

SCHEMA_VERSION = "target_oriented_distribution_aware_training_v1"
CANDIDATE_PARAMETER_COUNT = 93_504_933
ADAPTER_PARAMETER_COUNT = 9_750_529
ONE_UPDATE_REPORT_SHA256 = "a32b2c175b153d8ac7c732b70a1bf9b2124fbe49a122ab7414f16a7842c7734e"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _training_artifact_updates(contract: Any) -> tuple[int, ...]:
    training = contract.payload["data"]["training_updates"]
    excluded = {int(value) for value in training["excluded_artifact_updates"]}
    updates = tuple(
        value
        for value in range(
            int(training["artifact_candidate_start"]),
            int(training["artifact_candidate_end"]) + 1,
        )
        if value not in excluded
    )
    if len(updates) != int(training["optimizer_update_count"]):
        raise ValueError("distribution-aware training artifact count drift")
    return updates


def _screen_groups(
    context: Mapping[str, Any],
    parameters: Any,
    steps: Mapping[str, Any],
    target_contract: Any,
    flux_basis: Any,
    updates: tuple[int, ...],
    *,
    label: str,
) -> list[Mapping[str, Any]]:
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    records = []
    for group in _groups(context, updates):
        started = time.perf_counter()
        record, _totals = _evaluate_group(
            jax,
            jnp,
            parameters,
            steps,
            group,
            target_contract,
            flux_basis=flux_basis,
        )
        records.append(
            {
                "update": int(group["update"]),
                "runtime_year": int(group["runtime_year"]),
                "elapsed_seconds": time.perf_counter() - started,
            }
            | record
        )
        print(f"distribution_aware training {label}_screen={group['update']} complete", flush=True)
    return records


def _screen(
    context: Mapping[str, Any],
    parameters: Any,
    steps: Mapping[str, Any],
    target_contract: Any,
    flux_basis: Any,
    parent_groups: list[Mapping[str, Any]],
    contract: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    candidate_groups = _screen_groups(
        context,
        parameters,
        steps,
        target_contract,
        flux_basis,
        contract.evaluation_updates,
        label=label,
    )
    return {
        "candidate_groups": candidate_groups,
        "comparison_to_fixed_parent": compare_distribution_aware_train_train_groups(
            parent_groups,
            candidate_groups,
            contract,
        ),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()

    bounded = load_distribution_aware_bounded_training_contract(args.bounded_contract)
    contract_identity = bounded.payload["identity"]
    if sha256_file(args.one_update_admission_report) != ONE_UPDATE_REPORT_SHA256:
        raise ValueError("distribution-aware one-update report identity drift")
    one_update_report = json.loads(args.one_update_admission_report.read_text(encoding="utf-8"))
    if (
        one_update_report.get("status") != "passed"
        or one_update_report["next_gate"] != "request_paid_250_update_adapter_training"
        or one_update_report["policy"]["training_updates_persisted"] != 0
    ):
        raise ValueError("distribution-aware one-update admission is not accepted")
    if sha256_file(args.parent_baseline_report) != contract_identity["parent_train_train_report_sha256"]:
        raise ValueError("distribution-aware parent train/train report identity drift")
    parent_groups = json.loads(args.parent_baseline_report.read_text(encoding="utf-8"))["baseline_groups"]
    if sha256_file(args.prepared_manifest) != contract_identity["prepared_manifest_sha256"]:
        raise ValueError("distribution-aware prepared-manifest identity drift")

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if amendment.sha256 != contract_identity["flux_basis_amendment_sha256"]:
        raise ValueError("distribution-aware flux-basis identity drift")
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    context["prepared_manifest"] = args.prepared_manifest
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    target_contract = context["target_contract"]
    if (
        context["training_contract"].sha256 != contract_identity["base_training_contract_sha256"]
        or target_contract.sha256 != contract_identity["target_contract_sha256"]
        or args.stage_checkpoint_sha256 != contract_identity["parent_checkpoint_sha256"]
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
    if candidate.sha256 != contract_identity["candidate_contract_sha256"]:
        raise ValueError("distribution-aware candidate contract identity drift")
    parent_parameters = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=candidate,
    )
    initial_adapter = _adapter_tree(parent_parameters)
    if (
        _tree_size(jax, parent_parameters) != CANDIDATE_PARAMETER_COUNT
        or _tree_size(jax, initial_adapter) != ADAPTER_PARAMETER_COUNT
    ):
        raise ValueError("distribution-aware training parameter count drift")

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
    free_gradient_steps = steps["make_trainable_gradient_steps"](parent_parameters, _replace_adapter)
    guard_gradient_steps = steps["make_trainable_teacher_state_acceptance_gradient_steps"](
        parent_parameters,
        _replace_adapter,
    )
    guard_weights = jnp.asarray(
        [
            1.0 / 6.0 if name in {"maintenance_respiration", "growth_respiration", "npp"} else 0.0
            for name in AUDIT_FLUX_KEYS
        ],
        dtype=jnp.float64,
    )
    coefficient = float(bounded.payload["objective"]["teacher_guard_coefficient"])
    clip_threshold = float(bounded.optimizer["global_gradient_norm_clip"]["threshold"])
    optimizer_step = _make_optimizer_step(jax, jnp, bounded, clip_threshold)
    artifact_updates = _training_artifact_updates(bounded)
    maximum_updates = len(artifact_updates)
    recovery_interval = int(bounded.payload["bounded_stage"]["recovery_interval_updates"])
    memory_before = _memory_report(runtime["devices"][0])

    identity = {
        "git_head": _git_head(),
        "bounded_training_contract_sha256": bounded.sha256,
        "candidate_contract_sha256": candidate.sha256,
        "one_update_admission_report_sha256": ONE_UPDATE_REPORT_SHA256,
        "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
        "parent_train_train_report_sha256": sha256_file(args.parent_baseline_report),
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "base_training_contract_sha256": context["training_contract"].sha256,
        "target_contract_sha256": target_contract.sha256,
        "flux_basis_amendment_sha256": amendment.sha256,
    }
    recovery_path = output_dir / "recovery.pkl"
    recovery = load_distribution_aware_recovery(
        recovery_path,
        identity=identity,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=clip_threshold,
        maximum_updates=maximum_updates,
    )
    if recovery is None:
        initial_screen = _screen(
            context,
            parent_parameters,
            steps,
            target_contract,
            amendment.flux_basis,
            parent_groups,
            bounded,
            label="initial",
        )
        if (
            not initial_screen["comparison_to_fixed_parent"]["all_scores_finite"]
            or not initial_screen["comparison_to_fixed_parent"]["hard_gate_passed"]
        ):
            raise RuntimeError("distribution-aware initial screen failed its hard gate")
        adapter = initial_adapter
        optimizer = initialize_target_oriented_adamw(adapter)
        completed = 0
        records: list[Mapping[str, Any]] = []
    else:
        initial_screen = recovery["initial_screen"]
        adapter, optimizer = jax.device_put((recovery["adapter"], recovery["optimizer"]))
        completed = int(recovery["completed_updates"])
        records = list(recovery["update_records"])
        print(f"distribution_aware training resumed update={completed}", flush=True)
    jax.block_until_ready((adapter, optimizer))

    progress_path = output_dir / "progress.json"
    latest_recovery = None
    for optimizer_update in range(completed + 1, maximum_updates + 1):
        artifact_update = artifact_updates[optimizer_update - 1]
        artifact = _load_artifact(args, context, artifact_update)
        parameters = _replace_adapter(parent_parameters, adapter)
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
        combined_objective = float(free_objective["total"] + coefficient * guard_objective["total"])
        gradient_reports = {
            "free": _tree_numeric_report(jax, jax.device_get(free_gradient)),
            "teacher_guard": _tree_numeric_report(jax, jax.device_get(guard_gradient)),
            "combined": _tree_numeric_report(jax, jax.device_get(combined_gradient)),
        }
        hard_host = {
            "free": _hard_host(jax.device_get(free_hard)),
            "teacher_guard": _hard_host(jax.device_get(guard_hard)),
        }
        if (
            not _all_finite(free_objective)
            or not _guard_objective_finite(guard_objective)
            or not np.isfinite(combined_objective)
            or any(_hard_failure_count(value) != 0 for value in hard_host.values())
            or any(
                not report["all_finite"]
                or report["element_count"] != ADAPTER_PARAMETER_COUNT
                or report["l2_norm"] <= 0.0
                or report["nonzero_leaf_count"] != report["leaf_count"]
                for report in gradient_reports.values()
            )
        ):
            raise RuntimeError(f"distribution-aware training update {optimizer_update} failed before proposal")

        learning_rate = target_oriented_learning_rate(bounded, optimizer_update)
        proposal = optimizer_step(
            adapter,
            combined_gradient,
            optimizer,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        jax.block_until_ready(proposal)
        adapter, optimizer = proposal[0], proposal[1]
        raw_norm = float(np.asarray(proposal[2]))
        clip_scale = float(np.asarray(proposal[3]))
        step_norm = float(np.asarray(proposal[5]))
        if (
            int(np.asarray(proposal[4])) != 0
            or not all(np.isfinite(value) for value in (raw_norm, clip_scale, step_norm))
            or raw_norm <= 0.0
            or not 0.0 < clip_scale <= 1.0
            or step_norm <= 0.0
        ):
            raise RuntimeError(f"distribution-aware training update {optimizer_update} proposal is invalid")
        record = {
            "optimizer_update": optimizer_update,
            "artifact_update": artifact_update,
            "runtime_year": int(context["schedule"]["entries"][artifact_update - 1]["runtime_year"]),
            "learning_rate": learning_rate,
            "teacher_guard_coefficient": coefficient,
            "free_objective": free_objective,
            "teacher_guard_objective": {
                "total": float(guard_objective["total"]),
                "cell_scores": {
                    name: float(value) for name, value in guard_objective["cell_scores"].items()
                },
            },
            "combined_objective": combined_objective,
            "gradients": gradient_reports,
            "raw_combined_gradient_norm": raw_norm,
            "clip_threshold": clip_threshold,
            "clip_scale": clip_scale,
            "clipped": clip_scale < 1.0,
            "parameter_step_norm": step_norm,
            "hard_counts": hard_host,
            "free_hard_count_sources": {
                name: _hard_host(jax.device_get(value)) for name, value in free_hard_sources.items()
            },
            "timing_seconds": {
                "free_forward": free_forward_seconds,
                "free_gradient": free_gradient_seconds,
                "teacher_guard_forward": guard_forward_seconds,
                "teacher_guard_gradient": guard_gradient_seconds,
                "total": time.perf_counter() - update_started,
            },
        }
        records.append(record)
        _atomic_json(
            progress_path,
            {
                "schema_version": "target_oriented_distribution_aware_training_progress_v1",
                "identity": identity,
                "completed_updates": optimizer_update,
                "updates": records,
            },
        )
        print(
            f"distribution_aware training update={optimizer_update}/{maximum_updates} "
            f"artifact={artifact_update} objective={combined_objective:.9g} "
            f"norm={raw_norm:.9g} clip={clip_scale:.6g} "
            f"seconds={record['timing_seconds']['total']:.3f}",
            flush=True,
        )
        if optimizer_update % recovery_interval == 0:
            host_adapter, host_optimizer = jax.device_get((adapter, optimizer))
            latest_recovery = write_distribution_aware_recovery(
                recovery_path,
                identity=identity,
                adapter=host_adapter,
                optimizer=host_optimizer,
                completed_updates=optimizer_update,
                parameter_count=ADAPTER_PARAMETER_COUNT,
                clip_threshold=clip_threshold,
                update_records=records,
                initial_screen=initial_screen,
            )
            print(f"distribution_aware training recovery={optimizer_update}", flush=True)
        del free_gradient, guard_gradient, combined_gradient, parameters

    final_parameters = _replace_adapter(parent_parameters, adapter)
    retained = _retained_parameter_report(jax, parent_parameters, final_parameters)
    final_screen = _screen(
        context,
        final_parameters,
        steps,
        target_contract,
        amendment.flux_basis,
        parent_groups,
        bounded,
        label="final",
    )
    comparison = final_screen["comparison_to_fixed_parent"]
    execution_passed = (
        len(records) == maximum_updates
        and int(np.asarray(optimizer.step)) == maximum_updates
        and retained["exactly_unchanged"]
        and comparison["all_scores_finite"]
        and comparison["hard_gate_passed"]
        and all(
            all(gradient["all_finite"] for gradient in record["gradients"].values())
            and all(_hard_failure_count(value) == 0 for value in record["hard_counts"].values())
            for record in records
        )
    )
    if not execution_passed:
        raise RuntimeError("distribution-aware training failed its execution or hard gate")

    host_adapter, host_optimizer = jax.device_get((adapter, optimizer))
    checkpoint = write_distribution_aware_checkpoint(
        output_dir / "stage-update-0250.pkl",
        identity=identity,
        adapter=host_adapter,
        optimizer=host_optimizer,
        completed_updates=maximum_updates,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=clip_threshold,
        update_records=records,
        initial_screen=initial_screen,
        final_screen=final_screen,
    )
    norms = np.asarray([record["raw_combined_gradient_norm"] for record in records], dtype=np.float64)
    timings = np.asarray([record["timing_seconds"]["total"] for record in records], dtype=np.float64)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": identity,
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "training_updates_executed": maximum_updates,
            "training_updates_persisted": maximum_updates,
            "retained_parent_parameters_trainable": False,
            "recovery_written": latest_recovery is not None,
            "final_checkpoint_written": True,
            "checkpoint_promotion_eligible": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "candidate_parameter_count": CANDIDATE_PARAMETER_COUNT,
            "trainable_adapter_parameter_count": ADAPTER_PARAMETER_COUNT,
            "retained_parameters": retained,
        },
        "training": {
            "artifact_updates": list(artifact_updates),
            "optimizer_update_count": maximum_updates,
            "teacher_guard_coefficient": coefficient,
            "clip_threshold": clip_threshold,
            "clipped_update_count": sum(bool(record["clipped"]) for record in records),
            "raw_combined_gradient_norm": {
                "minimum": float(np.min(norms)),
                "median": float(np.median(norms)),
                "maximum": float(np.max(norms)),
            },
            "update_seconds": {
                "minimum": float(np.min(timings)),
                "median": float(np.median(timings)),
                "maximum": float(np.max(timings)),
                "total": float(np.sum(timings)),
            },
            "updates": records,
        },
        "initial_screen": initial_screen,
        "final_screen": final_screen,
        "checkpoint": checkpoint,
        "latest_recovery": latest_recovery,
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "review_before_validation_and_365_day_rollout"
            if comparison["scientific_promotion_gate_passed"]
            else (
                "review_bounded_continuation_without_promotion"
                if comparison["relative_mechanism_gate_passed"]
                else "stop_unchanged_adapter_only_continuation"
            )
        ),
    }
    _atomic_json(report_path, report)
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
    parser.add_argument("--bounded-contract", type=Path, required=True)
    parser.add_argument("--one-update-admission-report", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--parent-baseline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

