"""Admit one ephemeral branch-only update for the source-shaped candidate."""

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
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_source_shaped_one_update_admission_v1"
CANDIDATE_PARAMETER_COUNT = 84_148_646
BRANCH_PARAMETER_COUNT = 394_242
BRANCH_KEY = "source_shaped_maintenance_heads"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _branch_tree(parameters: Mapping[str, Any]) -> Any:
    branches = parameters["target_oriented"][BRANCH_KEY]
    if tuple(sorted(branches)) != ("aboveground", "belowground"):
        raise ValueError("source-shaped branch registry drift")
    return branches


def _replace_source_shaped_branches(
    parameters: Mapping[str, Any],
    branches: Any,
) -> Mapping[str, Any]:
    target = dict(parameters["target_oriented"])
    target[BRANCH_KEY] = branches
    return dict(parameters, target_oriented=target)


def _without_source_shaped_branches(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    target = {
        name: value
        for name, value in parameters["target_oriented"].items()
        if name != BRANCH_KEY
    }
    return dict(parameters, target_oriented=target)


def _retained_parameter_report(
    jax: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Mapping[str, Any]:
    before_leaves, before_structure = jax.tree_util.tree_flatten(
        _without_source_shaped_branches(before)
    )
    after_leaves, after_structure = jax.tree_util.tree_flatten(
        _without_source_shaped_branches(after)
    )
    same_structure = before_structure == after_structure
    same_objects = same_structure and all(
        left is right for left, right in zip(before_leaves, after_leaves, strict=True)
    )
    return {
        "leaf_count": len(before_leaves),
        "element_count": sum(
            int(np.prod(value.shape, dtype=np.int64)) for value in before_leaves
        ),
        "same_tree_structure": same_structure,
        "same_array_objects": same_objects,
        "exactly_unchanged": same_objects,
    }


def _hard_host(value: Any) -> Mapping[str, int]:
    return {name: int(np.asarray(item)) for name, item in value._asdict().items()}


def _objective_record(
    components: Mapping[str, Any],
    target_contract: Any,
) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        compose_target_oriented_loss,
    )

    scalar_components = {
        name: value.total if hasattr(value, "total") else value
        for name, value in components.items()
    }
    result = compose_target_oriented_loss(
        one_day_state_anchor_value=scalar_components["one_day_state_anchor"],
        seven_day_science=scalar_components["seven_day_science"],
        thirty_day_science=scalar_components["thirty_day_science"],
        necessary_process_supervision=scalar_components["necessary_process_supervision"],
        contract=target_contract,
    )
    return {
        "total": float(np.asarray(result.total)),
        "components": {
            name: float(np.asarray(value))
            for name, value in result.components.items()
        },
        "weighted_components": {
            name: float(np.asarray(value))
            for name, value in result.weighted_components.items()
        },
    }


def _effective_forward(
    jax: Any,
    jnp: Any,
    parameters: Any,
    batch: Any,
    steps: Mapping[str, Any],
    target_contract: Any,
) -> tuple[Mapping[str, Any], Any, Mapping[str, Any], float]:
    """Evaluate the exact effective-batch objective and hard gates without gradients."""

    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        add_science_window_sufficient_statistics,
        merge_error_sufficient_statistics,
        necessary_process_supervision_from_sufficient_statistics,
        one_day_state_anchor_from_sufficient_statistics,
        science_window_from_sufficient_statistics,
    )

    started = time.perf_counter()
    one_values = [
        steps["one_forward"](parameters, _slice(batch.one_day, start, start + 4))
        for start in range(0, 32, 4)
    ]
    seven_values = [
        steps["seven_forward"](parameters, _slice(batch.seven_day, index, index + 1))
        for index in range(8)
    ]
    thirty_values = [
        steps["thirty_forward"](parameters, _slice(batch.thirty_day, index, index + 1))
        for index in range(4)
    ]
    jax.block_until_ready((one_values, seven_values, thirty_values))

    total_state = merge_error_sufficient_statistics([value[0] for value in one_values])
    total_process = merge_error_sufficient_statistics([value[1] for value in one_values])
    total_seven = _merge_statistics(
        add_science_window_sufficient_statistics,
        [value[0] for value in seven_values],
    )
    total_thirty = _merge_statistics(
        add_science_window_sufficient_statistics,
        [value[0] for value in thirty_values],
    )
    state = one_day_state_anchor_from_sufficient_statistics(total_state)
    process = necessary_process_supervision_from_sufficient_statistics(
        total_process,
        target_contract,
    )
    seven = science_window_from_sufficient_statistics(total_seven)
    thirty = science_window_from_sufficient_statistics(total_thirty)
    objective = _objective_record(
        {
            "one_day_state_anchor": state.total,
            "seven_day_science": seven.total,
            "thirty_day_science": thirty.total,
            "necessary_process_supervision": process,
        },
        target_contract,
    )

    hard_sources = {}
    for name, values in (
        ("one_day", one_values),
        ("seven_day", seven_values),
        ("thirty_day", thirty_values),
    ):
        counts = _zero_hard(jnp)
        for value in values:
            counts = _add_hard(jax, counts, value[-1])
        hard_sources[name] = counts
    hard = _zero_hard(jnp)
    for counts in hard_sources.values():
        hard = _add_hard(jax, hard, counts)
    jax.block_until_ready((hard, objective))
    return objective, hard, hard_sources, time.perf_counter() - started


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
        raise ValueError(f"source-shaped admission trace drift at artifact update {update}")
    return artifact


def _all_finite(values: Mapping[str, Any]) -> bool:
    return np.isfinite(values["total"]) and all(
        np.isfinite(value) for value in values["components"].values()
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    screen = load_source_shaped_train_train_screen_contract(args.screen_contract)
    screen_identity = screen.payload["identity"]
    prepared_manifest_sha256 = sha256_file(args.prepared_manifest)
    if args.stage_checkpoint_sha256 != screen_identity["parent_checkpoint_sha256"]:
        raise ValueError("source-shaped admission parent checkpoint identity drift")
    if prepared_manifest_sha256 != screen_identity["prepared_manifest_sha256"]:
        raise ValueError("source-shaped admission prepared manifest identity drift")
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
        raise ValueError("source-shaped admission training contract identity drift")
    if target_contract.sha256 != screen_identity["target_contract_sha256"]:
        raise ValueError("source-shaped admission target contract identity drift")

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
    expected_source_sha256 = screen_identity["source_shaped_contract_sha256"]
    if source.sha256 != expected_source_sha256:
        raise ValueError("source-shaped admission mechanism contract drift")
    parameters = initialize_source_shaped_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260827),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        source_contract=source,
    )
    if _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT:
        raise ValueError("source-shaped admission candidate parameter count drift")
    branches = _branch_tree(parameters)
    if _tree_size(jax, branches) != BRANCH_PARAMETER_COUNT:
        raise ValueError("source-shaped admission branch parameter count drift")

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

    memory_before = _memory_report(device)
    calibration_updates = tuple(
        int(value) for value in screen.payload["data"]["calibration_updates"]
    )
    calibration_records = []
    branch_norms = []
    for update in calibration_updates:
        artifact = _load_artifact(args, context, update)
        batch_started = time.perf_counter()
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
        )
        branch_gradient = _branch_tree(gradient)
        jax.block_until_ready((branch_gradient, hard))
        branch_report = _tree_numeric_report(jax, jax.device_get(branch_gradient))
        branch_reports = {
            name: _tree_numeric_report(
                jax,
                jax.device_get(branch_gradient[name]),
            )
            for name in ("aboveground", "belowground")
        }
        hard_host = _hard_host(jax.device_get(hard))
        objective = _objective_record(components, target_contract)
        norm = float(branch_report["l2_norm"])
        branch_norms.append(norm)
        calibration_records.append(
            {
                "artifact_update": update,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "objective": objective,
                "branch_gradient": branch_report,
                "branches": branch_reports,
                "hard_counts": hard_host,
                "hard_count_sources": {
                    name: _hard_host(jax.device_get(value))
                    for name, value in hard_sources.items()
                },
                "statistics_forward_seconds": forward_seconds,
                "microbatch_gradient_seconds": gradient_seconds,
                "total_seconds": time.perf_counter() - batch_started,
            }
        )
        print(
            f"source_shaped one_update_admission calibration={update} "
            f"branch_norm={norm:.9g} seconds={calibration_records[-1]['total_seconds']:.3f}",
            flush=True,
        )
        del gradient, gradient_sources, branch_gradient

    clip_threshold = float(
        np.percentile(np.asarray(branch_norms, dtype=np.float64), 90.0)
    )
    training = screen.payload["data"]["branch_training_updates"]
    training_artifact_update = int(training["artifact_update_start"])
    artifact = _load_artifact(args, context, training_artifact_update)
    update_started = time.perf_counter()
    (
        full_gradient,
        gradient_sources,
        components,
        _component_details,
        hard_before,
        hard_sources_before,
        forward_seconds,
        gradient_seconds,
    ) = _effective_gradient(
        jax,
        jnp,
        parameters,
        artifact.batch,
        steps,
        target_contract,
    )
    branch_gradient = _branch_tree(full_gradient)
    jax.block_until_ready((branch_gradient, hard_before))
    branch_gradient_report = _tree_numeric_report(
        jax,
        jax.device_get(branch_gradient),
    )
    branch_gradient_reports = {
        name: _tree_numeric_report(jax, jax.device_get(branch_gradient[name]))
        for name in ("aboveground", "belowground")
    }
    objective_before = _objective_record(components, target_contract)
    hard_before_host = _hard_host(jax.device_get(hard_before))

    optimizer = initialize_target_oriented_adamw(branches)
    jax.block_until_ready(optimizer)
    learning_rate = target_oriented_learning_rate(screen, 1)

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
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int32)
                for leaf in jax.tree_util.tree_leaves((updated, next_state))
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
                    jax.tree_util.tree_leaves(values),
                    jax.tree_util.tree_leaves(updated),
                    strict=True,
                )
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return updated, next_state, raw_norm, clip_scale, nonfinite, jnp.sqrt(step_squared)

    optimizer_started = time.perf_counter()
    proposal = optimizer_proposal(branches, branch_gradient, optimizer)
    jax.block_until_ready(proposal)
    optimizer_seconds = time.perf_counter() - optimizer_started
    updated_branches, optimizer_after = proposal[0], proposal[1]
    proposed_parameters = _replace_source_shaped_branches(parameters, updated_branches)

    retained_report = _retained_parameter_report(jax, parameters, proposed_parameters)
    branch_steps = {
        name: _tree_numeric_report(
            jax,
            jax.device_get(
                jax.tree_util.tree_map(
                    lambda after, before: after - before,
                    updated_branches[name],
                    branches[name],
                )
            ),
        )
        for name in ("aboveground", "belowground")
    }

    @jax.jit
    def nonfinite_count(tree: Any) -> Any:
        return sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int32)
                for leaf in jax.tree_util.tree_leaves(tree)
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )

    parameter_nonfinite_count = int(
        np.asarray(jax.device_get(nonfinite_count(proposed_parameters)))
    )
    objective_after, hard_after, hard_sources_after, after_forward_seconds = (
        _effective_forward(
            jax,
            jnp,
            proposed_parameters,
            artifact.batch,
            steps,
            target_contract,
        )
    )
    hard_after_host = _hard_host(jax.device_get(hard_after))

    calibration_passed = (
        len(calibration_records) == 8
        and all(record["branch_gradient"]["all_finite"] for record in calibration_records)
        and all(record["branch_gradient"]["l2_norm"] > 0.0 for record in calibration_records)
        and all(
            record["branches"][name]["all_finite"]
            and record["branches"][name]["l2_norm"] > 0.0
            for record in calibration_records
            for name in ("aboveground", "belowground")
        )
        and all(_all_finite(record["objective"]) for record in calibration_records)
        and all(
            _hard_failure_count(record["hard_counts"]) == 0
            for record in calibration_records
        )
        and np.isfinite(clip_threshold)
        and clip_threshold > 0.0
    )
    optimizer_step = int(np.asarray(jax.device_get(optimizer_after.step)))
    passed = (
        calibration_passed
        and _all_finite(objective_before)
        and _all_finite(objective_after)
        and branch_gradient_report["all_finite"]
        and branch_gradient_report["l2_norm"] > 0.0
        and all(
            branch_gradient_reports[name]["all_finite"]
            and branch_gradient_reports[name]["l2_norm"] > 0.0
            for name in ("aboveground", "belowground")
        )
        and _hard_failure_count(hard_before_host) == 0
        and _hard_failure_count(hard_after_host) == 0
        and retained_report["exactly_unchanged"]
        and all(
            branch_steps[name]["all_finite"] and branch_steps[name]["l2_norm"] > 0.0
            for name in ("aboveground", "belowground")
        )
        and int(np.asarray(proposal[4])) == 0
        and parameter_nonfinite_count == 0
        and optimizer_step == 1
        and all(
            np.isfinite(value)
            for value in (
                learning_rate,
                clip_threshold,
                float(np.asarray(proposal[2])),
                float(np.asarray(proposal[3])),
                float(np.asarray(proposal[5])),
            )
        )
        and float(np.asarray(proposal[5])) > 0.0
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
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
            "trainable_branch_parameter_count": _tree_size(jax, branches),
            "retained_parameter_count": retained_report["element_count"],
        },
        "calibration": {
            "artifact_updates": list(calibration_updates),
            "branch_gradient_norms": branch_norms,
            "percentile": 90.0,
            "clip_threshold": clip_threshold,
            "passed": calibration_passed,
            "records": calibration_records,
        },
        "ephemeral_update": {
            "artifact_update": training_artifact_update,
            "runtime_year": int(
                context["schedule"]["entries"][training_artifact_update - 1]["runtime_year"]
            ),
            "learning_rate": learning_rate,
            "objective_before": objective_before,
            "objective_after": objective_after,
            "branch_gradient": branch_gradient_report,
            "branch_gradients": branch_gradient_reports,
            "raw_branch_gradient_norm": float(np.asarray(proposal[2])),
            "clip_threshold": clip_threshold,
            "clip_scale": float(np.asarray(proposal[3])),
            "clipped": float(np.asarray(proposal[3])) < 1.0,
            "optimizer_step": optimizer_step,
            "optimizer_parameter_or_state_nonfinite_count": int(np.asarray(proposal[4])),
            "parameter_step_norm": float(np.asarray(proposal[5])),
            "branch_parameter_steps": branch_steps,
            "retained_parameters": retained_report,
            "full_parameter_nonfinite_count": parameter_nonfinite_count,
            "hard_counts_before": hard_before_host,
            "hard_count_sources_before": {
                name: _hard_host(jax.device_get(value))
                for name, value in hard_sources_before.items()
            },
            "hard_counts_after": hard_after_host,
            "hard_count_sources_after": {
                name: _hard_host(jax.device_get(value))
                for name, value in hard_sources_after.items()
            },
            "statistics_forward_seconds": forward_seconds,
            "microbatch_gradient_seconds": gradient_seconds,
            "optimizer_compile_and_step_seconds": optimizer_seconds,
            "post_update_forward_seconds": after_forward_seconds,
            "total_seconds": time.perf_counter() - update_started,
        },
        "device_memory": {
            "before_calibration": memory_before,
            "after_admission": _memory_report(device),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "review_then_request_paid_100_update_branch_only_training"
            if passed
            else "stop_and_repair_one_update_admission"
        ),
    }
    _atomic_json(output, report)
    del full_gradient, gradient_sources, proposed_parameters, optimizer_after
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
