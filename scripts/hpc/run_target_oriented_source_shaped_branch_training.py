"""Train only the source-shaped branches and run the frozen train/train screen."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_checkpoint import (
    load_source_shaped_recovery,
    write_source_shaped_checkpoint,
    write_source_shaped_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_screen import (
    compare_source_shaped_train_train_groups,
    load_source_shaped_train_train_screen_contract,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _load_context,
    _tree_size,
)
from scripts.hpc.audit_target_oriented_flux_protected_virtual_step import _evaluate_group
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED, _tree_numeric_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _effective_gradient,
    _hard_failure_count,
    _make_steps,
)
from scripts.hpc.run_target_oriented_source_shaped_one_update_admission import (
    BRANCH_PARAMETER_COUNT,
    CANDIDATE_PARAMETER_COUNT,
    _all_finite,
    _branch_tree,
    _hard_host,
    _load_artifact,
    _objective_record,
    _replace_source_shaped_branches,
    _retained_parameter_report,
)
from scripts.hpc.run_target_oriented_source_shaped_train_train_zero_panel import _groups

SCHEMA_VERSION = "target_oriented_source_shaped_branch_training_v1"
ONE_UPDATE_REPORT_SHA256 = "c2743cbc3cd563b682176531e7ec2310c292f5fd1f63bbf625c846f79434ea59"
BRANCH_AUTODIFF_REPORT_SHA256 = "5852d996f2baa8afa5e68d0446d30d09bff64b5ac7996836f448a27535e58538"
ZERO_PANEL_REPORT_SHA256 = "6e7302fe4131d408ca469a62e3c31d9259fe9efa4a2e713e4242bc549e094291"
RECOVERY_INTERVAL = 10


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _load_accepted_report(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} report identity drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError(f"{label} report is not accepted")
    return report


def _make_optimizer_step(jax: Any, jnp: Any, screen: Any, clip_threshold: float) -> Any:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        target_oriented_adamw_update,
    )

    @jax.jit
    def step(branches: Any, gradient: Any, optimizer: Any, learning_rate: Any) -> Any:
        updated, next_optimizer, raw_norm, clip_scale = target_oriented_adamw_update(
            branches,
            gradient,
            optimizer,
            learning_rate=learning_rate,
            clip_norm=clip_threshold,
            contract=screen,
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int32)
                for leaf in jax.tree_util.tree_leaves((updated, next_optimizer))
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )
        step_squared = sum(
            (
                jnp.sum(
                    jnp.square(jnp.asarray(after) - jnp.asarray(before)),
                    dtype=jnp.float32,
                )
                for before, after in zip(
                    jax.tree_util.tree_leaves(branches),
                    jax.tree_util.tree_leaves(updated),
                    strict=True,
                )
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return updated, next_optimizer, raw_norm, clip_scale, nonfinite, jnp.sqrt(step_squared)

    return step


def _screen_groups(
    context: Mapping[str, Any],
    parameters: Any,
    steps: Mapping[str, Any],
    target_contract: Any,
    flux_basis: Any,
    updates: tuple[int, ...],
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
        print(f"source_shaped branch_training screen_update={group['update']} complete", flush=True)
    return records


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()

    screen = load_source_shaped_train_train_screen_contract(args.screen_contract)
    screen_identity = screen.payload["identity"]
    one_update_report = _load_accepted_report(
        args.one_update_admission_report,
        ONE_UPDATE_REPORT_SHA256,
        "source-shaped one-update admission",
    )
    direct_report = _load_accepted_report(
        args.branch_autodiff_admission_report,
        BRANCH_AUTODIFF_REPORT_SHA256,
        "source-shaped branch-autodiff admission",
    )
    zero_report = _load_accepted_report(
        args.zero_panel_report,
        ZERO_PANEL_REPORT_SHA256,
        "source-shaped zero-panel",
    )
    if sha256_file(args.parent_baseline_report) != screen_identity["parent_train_train_report_sha256"]:
        raise ValueError("source-shaped parent train/train report identity drift")
    parent_report = json.loads(args.parent_baseline_report.read_text(encoding="utf-8"))
    parent_groups = parent_report["baseline_groups"]

    clip_threshold = float(one_update_report["calibration"]["clip_threshold"])
    if (
        direct_report["parameter_counts"]["directly_differentiated"] != BRANCH_PARAMETER_COUNT
        or float(direct_report["ephemeral_proposal"]["clip_threshold"]) != clip_threshold
        or zero_report["identity"]["screen_contract_sha256"] != screen.sha256
    ):
        raise ValueError("source-shaped accepted pre-training evidence drift")

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
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
    device = runtime["devices"][0]
    prepared_manifest_sha256 = sha256_file(args.prepared_manifest)
    if (
        args.stage_checkpoint_sha256 != screen_identity["parent_checkpoint_sha256"]
        or prepared_manifest_sha256 != screen_identity["prepared_manifest_sha256"]
        or context["training_contract"].sha256 != screen_identity["training_contract_sha256"]
        or target_contract.sha256 != screen_identity["target_contract_sha256"]
    ):
        raise ValueError("source-shaped branch-training parent or data identity drift")

    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_source_shaped_maintenance_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_maintenance import (
        initialize_source_shaped_maintenance_parameters,
        load_source_shaped_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        derive_target_oriented_acceptance_scales,
        initialize_target_oriented_adamw,
        target_oriented_learning_rate,
    )

    source = load_source_shaped_maintenance_contract(args.source_contract)
    if source.sha256 != screen_identity["source_shaped_contract_sha256"]:
        raise ValueError("source-shaped branch-training mechanism contract drift")
    parent_parameters = initialize_source_shaped_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260827),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        source_contract=source,
    )
    initial_branches = _branch_tree(parent_parameters)
    if (
        _tree_size(jax, parent_parameters) != CANDIDATE_PARAMETER_COUNT
        or _tree_size(jax, initial_branches) != BRANCH_PARAMETER_COUNT
    ):
        raise ValueError("source-shaped branch-training parameter count drift")

    head = build_target_oriented_source_shaped_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        source_contract=source,
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
    direct_steps = steps["make_trainable_gradient_steps"](
        parent_parameters,
        _replace_source_shaped_branches,
    )
    optimizer_step = _make_optimizer_step(jax, jnp, screen, clip_threshold)
    memory_before = _memory_report(device)

    training = screen.payload["data"]["branch_training_updates"]
    artifact_start = int(training["artifact_update_start"])
    artifact_end = int(training["artifact_update_end"])
    maximum_updates = int(training["optimizer_update_count"])
    if (artifact_start, artifact_end, maximum_updates) != (252, 351, 100):
        raise ValueError("source-shaped branch-training schedule drift")
    identity = {
        "git_head": _git_head(),
        "screen_contract_sha256": screen.sha256,
        "source_shaped_contract_sha256": source.sha256,
        "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "training_contract_sha256": context["training_contract"].sha256,
        "target_contract_sha256": target_contract.sha256,
        "one_update_admission_report_sha256": ONE_UPDATE_REPORT_SHA256,
        "branch_autodiff_admission_report_sha256": BRANCH_AUTODIFF_REPORT_SHA256,
        "zero_panel_report_sha256": ZERO_PANEL_REPORT_SHA256,
    }
    recovery_path = output_dir / "recovery.pkl"
    recovery = load_source_shaped_recovery(
        recovery_path,
        identity=identity,
        parameter_count=BRANCH_PARAMETER_COUNT,
        clip_threshold=clip_threshold,
        maximum_updates=maximum_updates,
    )
    if recovery is None:
        completed = 0
        branches = initial_branches
        optimizer = initialize_target_oriented_adamw(branches)
        records: list[Mapping[str, Any]] = []
    else:
        completed = int(recovery["completed_updates"])
        branches, optimizer = jax.device_put((recovery["branches"], recovery["optimizer"]))
        records = list(recovery["update_records"])
        print(f"source_shaped branch_training resumed update={completed}", flush=True)
    jax.block_until_ready((branches, optimizer))

    progress_path = output_dir / "progress.json"
    latest_recovery = None
    for optimizer_update in range(completed + 1, maximum_updates + 1):
        artifact_update = artifact_start + optimizer_update - 1
        artifact = _load_artifact(args, context, artifact_update)
        parameters = _replace_source_shaped_branches(parent_parameters, branches)
        update_started = time.perf_counter()
        (
            gradient,
            gradient_sources,
            components,
            _component_details,
            hard,
            hard_sources,
            forward_seconds,
            gradient_seconds,
        ) = _effective_gradient(
            jax,
            jnp,
            parameters,
            artifact.batch,
            steps,
            target_contract,
            science_flux_basis=amendment.flux_basis,
            gradient_parameters=branches,
            gradient_steps=direct_steps,
        )
        jax.block_until_ready((gradient, hard))
        hard_host = _hard_host(jax.device_get(hard))
        objective = _objective_record(components, target_contract)
        gradient_report = _tree_numeric_report(jax, jax.device_get(gradient))
        if (
            _hard_failure_count(hard_host) != 0
            or not _all_finite(objective)
            or not gradient_report["all_finite"]
            or gradient_report["l2_norm"] <= 0.0
        ):
            raise RuntimeError(f"source-shaped branch training update {optimizer_update} failed before proposal")

        learning_rate = target_oriented_learning_rate(screen, optimizer_update)
        proposal = optimizer_step(
            branches,
            gradient,
            optimizer,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        jax.block_until_ready(proposal)
        branches, optimizer = proposal[0], proposal[1]
        raw_norm = float(np.asarray(proposal[2]))
        clip_scale = float(np.asarray(proposal[3]))
        nonfinite = int(np.asarray(proposal[4]))
        step_norm = float(np.asarray(proposal[5]))
        proposal_valid = (
            all(np.isfinite(value) for value in (raw_norm, clip_scale, step_norm))
            and raw_norm > 0.0
            and 0.0 < clip_scale <= 1.0
            and step_norm > 0.0
        )
        if nonfinite != 0 or not proposal_valid:
            raise RuntimeError(f"source-shaped branch training update {optimizer_update} proposal is invalid")
        record = {
            "optimizer_update": optimizer_update,
            "artifact_update": artifact_update,
            "runtime_year": int(context["schedule"]["entries"][artifact_update - 1]["runtime_year"]),
            "learning_rate": learning_rate,
            "objective": objective,
            "branch_gradient": gradient_report,
            "raw_branch_gradient_norm": raw_norm,
            "clip_threshold": clip_threshold,
            "clip_scale": clip_scale,
            "clipped": clip_scale < 1.0,
            "parameter_step_norm": step_norm,
            "hard_counts": hard_host,
            "hard_count_sources": {name: _hard_host(jax.device_get(value)) for name, value in hard_sources.items()},
            "statistics_forward_seconds": forward_seconds,
            "microbatch_gradient_seconds": gradient_seconds,
            "total_seconds": time.perf_counter() - update_started,
        }
        records.append(record)
        _atomic_json(
            progress_path,
            {
                "schema_version": "target_oriented_source_shaped_branch_training_progress_v1",
                "identity": identity,
                "completed_updates": optimizer_update,
                "updates": records,
            },
        )
        print(
            f"source_shaped branch_training update={optimizer_update}/{maximum_updates} "
            f"artifact={artifact_update} loss={objective['total']:.9g} "
            f"norm={raw_norm:.9g} clip={clip_scale:.6g} seconds={record['total_seconds']:.3f}",
            flush=True,
        )
        if optimizer_update % RECOVERY_INTERVAL == 0:
            host_branches, host_optimizer = jax.device_get((branches, optimizer))
            latest_recovery = write_source_shaped_recovery(
                recovery_path,
                identity=identity,
                branches=host_branches,
                optimizer=host_optimizer,
                completed_updates=optimizer_update,
                parameter_count=BRANCH_PARAMETER_COUNT,
                clip_threshold=clip_threshold,
                update_records=records,
            )
            print(f"source_shaped branch_training recovery={optimizer_update}", flush=True)
        del gradient, gradient_sources, parameters

    final_parameters = _replace_source_shaped_branches(parent_parameters, branches)
    retained_report = _retained_parameter_report(jax, parent_parameters, final_parameters)
    candidate_groups = _screen_groups(
        context,
        final_parameters,
        steps,
        target_contract,
        amendment.flux_basis,
        screen.evaluation_updates,
    )
    comparison = compare_source_shaped_train_train_groups(
        parent_groups,
        candidate_groups,
        screen,
    )
    execution_passed = (
        len(records) == maximum_updates
        and int(np.asarray(optimizer.step)) == maximum_updates
        and retained_report["exactly_unchanged"]
        and comparison["all_scores_finite"]
        and comparison["hard_gate_passed"]
        and all(record["branch_gradient"]["all_finite"] for record in records)
        and all(_hard_failure_count(record["hard_counts"]) == 0 for record in records)
    )
    if not execution_passed:
        raise RuntimeError("source-shaped branch training failed its execution or hard gate")

    host_branches, host_optimizer = jax.device_get((branches, optimizer))
    checkpoint = write_source_shaped_checkpoint(
        output_dir / "stage-update-0100.pkl",
        identity=identity,
        branches=host_branches,
        optimizer=host_optimizer,
        completed_updates=maximum_updates,
        parameter_count=BRANCH_PARAMETER_COUNT,
        clip_threshold=clip_threshold,
        update_records=records,
        candidate_groups=candidate_groups,
        screen_comparison=comparison,
    )
    raw_norms = np.asarray(
        [record["raw_branch_gradient_norm"] for record in records],
        dtype=np.float64,
    )
    timings = np.asarray([record["total_seconds"] for record in records], dtype=np.float64)
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
            "trainable_branch_parameter_count": BRANCH_PARAMETER_COUNT,
            "retained_parameters": retained_report,
        },
        "training": {
            "artifact_update_start": artifact_start,
            "artifact_update_end": artifact_end,
            "optimizer_update_count": maximum_updates,
            "clip_threshold": clip_threshold,
            "clipped_update_count": sum(bool(record["clipped"]) for record in records),
            "raw_gradient_norm": {
                "minimum": float(np.min(raw_norms)),
                "median": float(np.median(raw_norms)),
                "maximum": float(np.max(raw_norms)),
            },
            "update_seconds": {
                "minimum": float(np.min(timings)),
                "median": float(np.median(timings)),
                "maximum": float(np.max(timings)),
                "total": float(np.sum(timings)),
            },
            "updates": records,
        },
        "initial_screen": {
            "report_sha256": ZERO_PANEL_REPORT_SHA256,
            "comparison_to_fixed_parent": zero_report["comparison_to_fixed_parent"],
        },
        "final_screen": {
            "candidate_groups": candidate_groups,
            "comparison_to_fixed_parent": comparison,
        },
        "checkpoint": checkpoint,
        "latest_recovery": latest_recovery,
        "memory_before": memory_before,
        "memory_after": _memory_report(device),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "review_before_any_validation_or_broader_training"
            if comparison["relative_source_gate_passed"]
            else "stop_and_attribute_branch_training_screen_failure"
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
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--parent-baseline-report", type=Path, required=True)
    parser.add_argument("--zero-panel-report", type=Path, required=True)
    parser.add_argument("--one-update-admission-report", type=Path, required=True)
    parser.add_argument("--branch-autodiff-admission-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
