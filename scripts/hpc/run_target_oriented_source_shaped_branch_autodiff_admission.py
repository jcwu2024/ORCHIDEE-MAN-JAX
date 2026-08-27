"""Compare direct branch autodiff with the admitted extracted branch gradient."""

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
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_screen import (
    load_source_shaped_train_train_screen_contract,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _load_context,
    _tree_size,
)
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
    _effective_forward,
    _hard_host,
    _load_artifact,
    _objective_record,
    _replace_source_shaped_branches,
)

SCHEMA_VERSION = "target_oriented_source_shaped_branch_autodiff_admission_v1"
ADMITTED_REPORT_SHA256 = "c2743cbc3cd563b682176531e7ec2310c292f5fd1f63bbf625c846f79434ea59"
GRADIENT_ATOL = 1.0e-8
GRADIENT_RTOL = 1.0e-6
OBJECTIVE_ATOL = 1.0e-12
OBJECTIVE_RTOL = 1.0e-10


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _tree_comparison(jax: Any, left: Any, right: Any) -> Mapping[str, Any]:
    left_leaves, left_structure = jax.tree_util.tree_flatten(jax.device_get(left))
    right_leaves, right_structure = jax.tree_util.tree_flatten(jax.device_get(right))
    if left_structure != right_structure or len(left_leaves) != len(right_leaves):
        return {
            "passed": False,
            "same_tree_structure": False,
            "leaf_count": len(left_leaves),
        }
    maximum_absolute = 0.0
    difference_squared = 0.0
    reference_squared = 0.0
    failing_elements = 0
    element_count = 0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        first = np.asarray(left_leaf, dtype=np.float64)
        second = np.asarray(right_leaf, dtype=np.float64)
        difference = first - second
        tolerance = GRADIENT_ATOL + GRADIENT_RTOL * np.abs(second)
        maximum_absolute = max(
            maximum_absolute,
            float(np.max(np.abs(difference))) if difference.size else 0.0,
        )
        difference_squared += float(np.sum(difference * difference, dtype=np.float64))
        reference_squared += float(np.sum(second * second, dtype=np.float64))
        failing_elements += int(np.count_nonzero(np.abs(difference) > tolerance))
        element_count += int(difference.size)
    difference_norm = float(np.sqrt(difference_squared))
    reference_norm = float(np.sqrt(reference_squared))
    return {
        "passed": failing_elements == 0,
        "same_tree_structure": True,
        "leaf_count": len(left_leaves),
        "element_count": element_count,
        "failing_element_count": failing_elements,
        "maximum_absolute_difference": maximum_absolute,
        "difference_l2_norm": difference_norm,
        "reference_l2_norm": reference_norm,
        "relative_l2_difference": difference_norm / max(reference_norm, 1.0e-30),
        "atol": GRADIENT_ATOL,
        "rtol": GRADIENT_RTOL,
    }


def _objective_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Mapping[str, Any]:
    names = ("total",)
    pairs = [(name, float(left[name]), float(right[name])) for name in names]
    for group in ("components", "weighted_components"):
        if set(left[group]) != set(right[group]):
            return {"passed": False, "registry_match": False}
        pairs.extend(
            (f"{group}.{name}", float(left[group][name]), float(right[group][name]))
            for name in sorted(left[group])
        )
    differences = {name: first - second for name, first, second in pairs}
    passed = all(
        np.isclose(first, second, atol=OBJECTIVE_ATOL, rtol=OBJECTIVE_RTOL)
        for _name, first, second in pairs
    )
    return {
        "passed": bool(passed),
        "registry_match": True,
        "maximum_absolute_difference": max(abs(value) for value in differences.values()),
        "differences": differences,
        "atol": OBJECTIVE_ATOL,
        "rtol": OBJECTIVE_RTOL,
    }


def _hard_comparison(left: Any, right: Any) -> Mapping[str, Any]:
    first = _hard_host(left)
    second = _hard_host(right)
    return {
        "passed": first == second,
        "direct_branch": first,
        "complete_gradient": second,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    if sha256_file(args.one_update_admission_report) != ADMITTED_REPORT_SHA256:
        raise ValueError("source-shaped one-update admission report identity drift")
    admitted = json.loads(args.one_update_admission_report.read_text(encoding="utf-8"))
    if admitted["status"] != "passed" or admitted["policy"]["training_updates_persisted"] != 0:
        raise ValueError("source-shaped one-update admission report is not an accepted ephemeral run")

    screen = load_source_shaped_train_train_screen_contract(args.screen_contract)
    screen_identity = screen.payload["identity"]
    prepared_manifest_sha256 = sha256_file(args.prepared_manifest)
    if args.stage_checkpoint_sha256 != screen_identity["parent_checkpoint_sha256"]:
        raise ValueError("source-shaped branch-autodiff parent checkpoint identity drift")
    if prepared_manifest_sha256 != screen_identity["prepared_manifest_sha256"]:
        raise ValueError("source-shaped branch-autodiff prepared manifest identity drift")
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
    target_contract = context["target_contract"]
    device = runtime["devices"][0]
    if context["training_contract"].sha256 != screen_identity["training_contract_sha256"]:
        raise ValueError("source-shaped branch-autodiff training contract identity drift")
    if target_contract.sha256 != screen_identity["target_contract_sha256"]:
        raise ValueError("source-shaped branch-autodiff target contract identity drift")

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
        target_oriented_adamw_update,
        target_oriented_learning_rate,
    )

    source = load_source_shaped_maintenance_contract(args.source_contract)
    if source.sha256 != screen_identity["source_shaped_contract_sha256"]:
        raise ValueError("source-shaped branch-autodiff mechanism contract drift")
    parameters = initialize_source_shaped_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260827),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        source_contract=source,
    )
    branches = _branch_tree(parameters)
    if _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT:
        raise ValueError("source-shaped branch-autodiff candidate parameter count drift")
    if _tree_size(jax, branches) != BRANCH_PARAMETER_COUNT:
        raise ValueError("source-shaped branch-autodiff trainable parameter count drift")

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
        parameters,
        _replace_source_shaped_branches,
    )
    artifact_update = int(admitted["ephemeral_update"]["artifact_update"])
    artifact = _load_artifact(args, context, artifact_update)
    memory_before = _memory_report(device)

    direct_started = time.perf_counter()
    (
        direct_gradient,
        direct_sources,
        direct_components,
        _direct_details,
        direct_hard,
        direct_hard_sources,
        direct_forward_seconds,
        direct_gradient_seconds,
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
    jax.block_until_ready((direct_gradient, direct_hard))
    direct_cold_seconds = time.perf_counter() - direct_started
    memory_after_direct_cold = _memory_report(device)

    hot_started = time.perf_counter()
    (
        hot_gradient,
        hot_sources,
        hot_components,
        _hot_details,
        hot_hard,
        _hot_hard_sources,
        hot_forward_seconds,
        hot_gradient_seconds,
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
    jax.block_until_ready((hot_gradient, hot_hard))
    direct_hot_seconds = time.perf_counter() - hot_started
    memory_after_direct_hot = _memory_report(device)

    complete_started = time.perf_counter()
    (
        complete_gradient,
        complete_sources,
        complete_components,
        _complete_details,
        complete_hard,
        complete_hard_sources,
        complete_forward_seconds,
        complete_gradient_seconds,
    ) = _effective_gradient(
        jax,
        jnp,
        parameters,
        artifact.batch,
        steps,
        target_contract,
        science_flux_basis=amendment.flux_basis,
    )
    extracted_gradient = _branch_tree(complete_gradient)
    extracted_sources = {
        name: _branch_tree(value) for name, value in complete_sources.items()
    }
    jax.block_until_ready((extracted_gradient, complete_hard))
    complete_seconds = time.perf_counter() - complete_started
    memory_after_complete = _memory_report(device)

    gradient_comparison = _tree_comparison(jax, direct_gradient, extracted_gradient)
    hot_replay_comparison = _tree_comparison(jax, direct_gradient, hot_gradient)
    source_comparisons = {
        name: _tree_comparison(jax, direct_sources[name], extracted_sources[name])
        for name in sorted(direct_sources)
    }
    direct_objective = _objective_record(direct_components, target_contract)
    hot_objective = _objective_record(hot_components, target_contract)
    complete_objective = _objective_record(complete_components, target_contract)
    objective_comparison = _objective_comparison(direct_objective, complete_objective)
    hot_objective_comparison = _objective_comparison(direct_objective, hot_objective)
    hard_comparison = _hard_comparison(direct_hard, complete_hard)
    hot_hard_comparison = _hard_comparison(direct_hard, hot_hard)
    hard_source_comparisons = {
        name: _hard_comparison(direct_hard_sources[name], complete_hard_sources[name])
        for name in sorted(direct_hard_sources)
    }

    clip_threshold = float(admitted["calibration"]["clip_threshold"])
    learning_rate = target_oriented_learning_rate(screen, 1)
    optimizer = initialize_target_oriented_adamw(branches)

    @jax.jit
    def optimizer_proposal(values: Any, derivatives: Any, state: Any) -> Any:
        updated, next_state, raw_norm, clip_scale = target_oriented_adamw_update(
            values,
            derivatives,
            state,
            learning_rate=learning_rate,
            clip_norm=clip_threshold,
            contract=screen,
        )
        step_squared = sum(
            (
                jnp.sum(
                    jnp.square(jnp.asarray(after) - jnp.asarray(before)),
                    dtype=jnp.float32,
                )
                for before, after in zip(
                    jax.tree_util.tree_leaves(values),
                    jax.tree_util.tree_leaves(updated),
                    strict=True,
                )
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return updated, next_state, raw_norm, clip_scale, jnp.sqrt(step_squared)

    direct_proposal = optimizer_proposal(branches, direct_gradient, optimizer)
    complete_proposal = optimizer_proposal(branches, extracted_gradient, optimizer)
    jax.block_until_ready((direct_proposal, complete_proposal))
    proposal_parameter_comparison = _tree_comparison(
        jax,
        direct_proposal[0],
        complete_proposal[0],
    )
    proposal_state_comparison = _tree_comparison(
        jax,
        direct_proposal[1],
        complete_proposal[1],
    )
    proposal_scalar_differences = {
        "raw_norm": float(np.asarray(direct_proposal[2] - complete_proposal[2])),
        "clip_scale": float(np.asarray(direct_proposal[3] - complete_proposal[3])),
        "step_norm": float(np.asarray(direct_proposal[4] - complete_proposal[4])),
    }
    proposed_parameters = _replace_source_shaped_branches(
        parameters,
        direct_proposal[0],
    )
    objective_after, hard_after, hard_sources_after, post_update_forward_seconds = (
        _effective_forward(
            jax,
            jnp,
            proposed_parameters,
            artifact.batch,
            steps,
            target_contract,
            amendment.flux_basis,
        )
    )
    reference_before = admitted["ephemeral_update"]["objective_before"]
    reference_after = admitted["ephemeral_update"]["objective_after"]
    admitted_before_comparison = _objective_comparison(direct_objective, reference_before)
    admitted_after_comparison = _objective_comparison(objective_after, reference_after)

    direct_report = _tree_numeric_report(jax, jax.device_get(direct_gradient))
    branch_reports = {
        name: _tree_numeric_report(jax, jax.device_get(direct_gradient[name]))
        for name in ("aboveground", "belowground")
    }
    admitted_hot_seconds = float(
        np.median(
            np.asarray(
                [record["total_seconds"] for record in admitted["calibration"]["records"][1:]],
                dtype=np.float64,
            )
        )
    )
    admitted_peak_bytes = int(
        admitted["device_memory"]["after_admission"]["peak_bytes_in_use"]
    )
    direct_peak_bytes = int(memory_after_direct_hot.get("peak_bytes_in_use", 0))
    resource_improved = (
        0 < direct_peak_bytes < admitted_peak_bytes
        and direct_hot_seconds < admitted_hot_seconds
    )
    numerical_passed = (
        gradient_comparison["passed"]
        and hot_replay_comparison["passed"]
        and all(value["passed"] for value in source_comparisons.values())
        and objective_comparison["passed"]
        and hot_objective_comparison["passed"]
        and hard_comparison["passed"]
        and hot_hard_comparison["passed"]
        and all(value["passed"] for value in hard_source_comparisons.values())
        and proposal_parameter_comparison["passed"]
        and proposal_state_comparison["passed"]
        and all(abs(value) <= 1.0e-12 for value in proposal_scalar_differences.values())
        and admitted_before_comparison["passed"]
        and admitted_after_comparison["passed"]
    )
    physical_passed = (
        direct_report["all_finite"]
        and direct_report["l2_norm"] > 0.0
        and all(
            branch_reports[name]["all_finite"] and branch_reports[name]["l2_norm"] > 0.0
            for name in ("aboveground", "belowground")
        )
        and _all_finite(direct_objective)
        and _all_finite(objective_after)
        and _hard_failure_count(_hard_host(direct_hard)) == 0
        and _hard_failure_count(_hard_host(hard_after)) == 0
        and all(
            _tree_numeric_report(jax, jax.device_get(value))["all_finite"]
            for value in direct_proposal[:2]
        )
        and int(np.asarray(direct_proposal[1].step)) == 1
    )
    passed = numerical_passed and physical_passed and resource_improved
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "one_update_admission_report_sha256": ADMITTED_REPORT_SHA256,
            "screen_contract_sha256": screen.sha256,
            "source_shaped_contract_sha256": source.sha256,
            "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
            "prepared_manifest_sha256": prepared_manifest_sha256,
            "training_contract_sha256": context["training_contract"].sha256,
            "target_contract_sha256": target_contract.sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_proposals_evaluated": 1,
            "training_updates_executed": 0,
            "training_updates_persisted": 0,
            "checkpoint_written": False,
            "recovery_written": False,
            "proposal_retained_after_process_exit": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "artifact_update": artifact_update,
        "parameter_counts": {
            "complete": _tree_size(jax, parameters),
            "directly_differentiated": _tree_size(jax, branches),
        },
        "direct_branch_gradient": direct_report
        | {
            "branches": branch_reports,
            "cold_total_seconds": direct_cold_seconds,
            "cold_forward_seconds": direct_forward_seconds,
            "cold_gradient_seconds": direct_gradient_seconds,
            "hot_total_seconds": direct_hot_seconds,
            "hot_forward_seconds": hot_forward_seconds,
            "hot_gradient_seconds": hot_gradient_seconds,
        },
        "complete_gradient_reference": {
            "total_seconds": complete_seconds,
            "forward_seconds": complete_forward_seconds,
            "gradient_seconds": complete_gradient_seconds,
        },
        "equivalence": {
            "numerical_passed": numerical_passed,
            "gradient": gradient_comparison,
            "gradient_sources": source_comparisons,
            "hot_replay_gradient": hot_replay_comparison,
            "objective": objective_comparison,
            "hot_replay_objective": hot_objective_comparison,
            "hard_counts": hard_comparison,
            "hot_replay_hard_counts": hot_hard_comparison,
            "hard_count_sources": hard_source_comparisons,
            "proposal_parameters": proposal_parameter_comparison,
            "proposal_optimizer_state": proposal_state_comparison,
            "proposal_scalar_differences": proposal_scalar_differences,
            "admitted_objective_before": admitted_before_comparison,
            "admitted_objective_after": admitted_after_comparison,
        },
        "ephemeral_proposal": {
            "learning_rate": learning_rate,
            "clip_threshold": clip_threshold,
            "raw_gradient_norm": float(np.asarray(direct_proposal[2])),
            "clip_scale": float(np.asarray(direct_proposal[3])),
            "parameter_step_norm": float(np.asarray(direct_proposal[4])),
            "optimizer_step": int(np.asarray(direct_proposal[1].step)),
            "objective_before": direct_objective,
            "objective_after": objective_after,
            "hard_counts_after": _hard_host(hard_after),
            "hard_count_sources_after": {
                name: _hard_host(value) for name, value in hard_sources_after.items()
            },
            "post_update_forward_seconds": post_update_forward_seconds,
        },
        "resource": {
            "improved": resource_improved,
            "admitted_extracted_path_hot_seconds": admitted_hot_seconds,
            "direct_branch_hot_seconds": direct_hot_seconds,
            "hot_speedup": admitted_hot_seconds / direct_hot_seconds,
            "admitted_extracted_path_peak_bytes": admitted_peak_bytes,
            "direct_branch_peak_bytes": direct_peak_bytes,
            "peak_memory_reduction_fraction": 1.0 - direct_peak_bytes / admitted_peak_bytes,
            "memory_before": memory_before,
            "memory_after_direct_cold": memory_after_direct_cold,
            "memory_after_direct_hot": memory_after_direct_hot,
            "memory_after_complete_reference": memory_after_complete,
        },
        "physical_gate_passed": physical_passed,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "request_paid_100_update_branch_only_training"
            if passed
            else "stop_and_repair_branch_autodiff_or_resource_path"
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
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--one-update-admission-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
