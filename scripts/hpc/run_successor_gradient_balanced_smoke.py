"""Run the frozen 10-update gradient-balanced Attention smoke."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    GRADIENT_FEASIBILITY_LABELS,
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
)
from research.daily_coarse_graining.daily_operator.successor_rollout_screen import (
    add_rollout_screen_statistics,
    empty_rollout_screen_statistics,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from research.daily_coarse_graining.daily_operator.successor_staged_training import (
    REQUIRED_VALIDATION_SLICES,
    finalize_staged_validation_statistics,
)
from scripts.hpc.prepare_successor_rollout_screen import (
    _prepare_adapter as _prepare_validation_adapter,
)
from scripts.hpc.prepare_successor_staged_validation import (
    _reference as _validation_reference,
)
from scripts.hpc.probe_successor_batch_diversity import (
    POINTS_PER_MIXED_UPDATE,
    _join_context_anchors,
    _slice_context_batch,
    _stack_contexts,
)
from scripts.hpc.probe_successor_gradient_feasibility import (
    PARENT_UPDATE,
    _load_contract,
    _make_measurement,
    _sha256_file,
)
from scripts.hpc.run_successor_batch_diversity_short_fit import (
    _make_mixed_train_step,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_named_biomass_objective_ab import (
    classify_objective_ab,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_rollout_screen import _make_evaluator
from scripts.hpc.run_successor_small_fit import (
    _device_memory_report,
    _hard_counts_host,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.run_successor_staged_training import _training_identity
from scripts.hpc.successor_rollout_screen_prepared import (
    load_prepared_rollout_artifact,
)
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)
from scripts.hpc.successor_staged_validation_prepared import (
    load_staged_validation_prepared,
    ordered_validation_entries,
)

SMOKE_UPDATES = 10
CONTROL = "current_objective_control"
CANDIDATE = "gradient_balanced_candidate"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_gradient_gram(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def measure(gradients: Sequence[Any]) -> Any:
        def dot(left: Any, right: Any) -> Any:
            return sum(
                (
                    jnp.vdot(jnp.asarray(a, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64))
                    for a, b in zip(
                        jax.tree_util.tree_leaves(left),
                        jax.tree_util.tree_leaves(right),
                        strict=True,
                    )
                ),
                jnp.asarray(0.0, dtype=jnp.float64),
            )

        gram = jnp.stack(
            tuple(
                jnp.stack(tuple(dot(left, right) for right in gradients))
                for left in gradients
            )
        )
        nonfinite = jnp.stack(
            tuple(
                sum(
                    (
                        jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int64)
                        for leaf in jax.tree_util.tree_leaves(gradient)
                    ),
                    jnp.asarray(0, dtype=jnp.int64),
                )
                for gradient in gradients
            )
        )
        return gram, nonfinite

    return measure


def _make_candidate_apply_step(jax: Any, runner_contract: Any) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_adamw_update,
    )

    @jax.jit
    def apply_step(
        parameters: Any,
        optimizer: Any,
        gradients: Sequence[Any],
        weights: Any,
        learning_rate: Any,
    ) -> Any:
        combined = jax.tree_util.tree_map(
            lambda *leaves: sum(
                (
                    weights[index] * jnp.asarray(leaf)
                    for index, leaf in enumerate(leaves)
                ),
                jnp.asarray(0.0, dtype=jnp.float64),
            ),
            *gradients,
        )
        next_parameters, next_optimizer, gradient_norm = successor_adamw_update(
            parameters,
            combined,
            optimizer,
            learning_rate,
            runner_contract,
        )
        return next_parameters, next_optimizer, gradient_norm

    return apply_step


def _candidate_update(
    jax: Any,
    measurement: Any,
    gram_measurement: Any,
    apply_step: Any,
    parameters: Any,
    optimizer: Any,
    anchor_batch: Any,
    context_batch: Any,
    context_inputs: Any,
    learning_rate: Any,
) -> tuple[Any, Any, Mapping[str, Any]]:
    gradients = []
    losses_host = None
    hard_host = None
    for seed in np.eye(len(GRADIENT_FEASIBILITY_LABELS), dtype=np.float64):
        losses, gradient, hard_counts = measurement(
            parameters,
            seed,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        jax.block_until_ready((losses, gradient, hard_counts))
        current_losses = np.asarray(losses, dtype=np.float64)
        current_hard = _hard_counts_host(hard_counts)
        if losses_host is None:
            losses_host = current_losses
            hard_host = current_hard
        elif not np.array_equal(losses_host, current_losses) or hard_host != current_hard:
            raise ValueError("gradient-balanced seed changed objective outputs")
        gradients.append(gradient)
    gram, nonfinite = gram_measurement(tuple(gradients))
    jax.block_until_ready((gram, nonfinite))
    gram_host = np.asarray(gram, dtype=np.float64)
    nonfinite_host = np.asarray(nonfinite, dtype=np.int64)
    diagnostics = gradient_direction_diagnostics(gram_host)
    direction = diagnostics["directions"]["minimum_norm_unit"]
    weights = np.asarray(
        [direction["weights"][label] for label in GRADIENT_FEASIBILITY_LABELS],
        dtype=np.float64,
    )
    if (
        losses_host is None
        or hard_host is None
        or np.any(~np.isfinite(losses_host))
        or np.any(nonfinite_host != 0)
        or sum(hard_host.values()) != 0
    ):
        raise RuntimeError("gradient-balanced candidate update failed hard gates")
    next_parameters, next_optimizer, gradient_norm = apply_step(
        parameters,
        optimizer,
        tuple(gradients),
        weights,
        learning_rate,
    )
    jax.block_until_ready((next_parameters, next_optimizer, gradient_norm))
    record = {
        "losses": {
            label: float(losses_host[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "gradient_norms": diagnostics["gradient_norms"],
        "minimum_norm_direction": direction,
        "minimum_component_cosine": min(
            direction["normalized_directional_derivatives"].values()
        ),
        "combined_gradient_norm": float(np.asarray(gradient_norm)),
        "nonfinite_gradient_values": {
            label: int(nonfinite_host[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "hard_gate_counts": hard_host,
    }
    del gradients, gram, nonfinite
    return next_parameters, next_optimizer, record


def selected_validation_subset(
    prepared: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], tuple[str, Mapping[str, Any]]], ...]:
    paired = tuple(
        zip(prepared["entries"], ordered_validation_entries(panel), strict=True)
    )
    selected = []
    for slice_id in REQUIRED_VALIDATION_SLICES:
        current = tuple(item for item in paired if item[1][0] == slice_id)
        if len(current) < 2:
            raise ValueError("gradient-balanced validation slice is too short")
        selected.extend((current[0], current[-1]))
    return tuple(selected)


def _evaluate_subset(
    runtime: Mapping[str, Any],
    *,
    contract: Any,
    parameters: Mapping[str, Any],
    long_scales: Any,
    adapter_template: Any,
    prepared_manifest_path: Path,
    prepared: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    evaluator = _make_evaluator(
        runtime,
        SOURCE_ROUTED_ATTENTION,
        adapter_template,
        long_scales,
        contract,
    )
    accumulators = {
        objective: {
            slice_id: empty_rollout_screen_statistics()
            for slice_id in REQUIRED_VALIDATION_SLICES
        }
        for objective in (CONTROL, CANDIDATE)
    }
    selected = selected_validation_subset(prepared, panel)
    for ordinal, (record, (slice_id, entry)) in enumerate(selected, start=1):
        artifact = load_prepared_rollout_artifact(
            prepared_manifest_path,
            record,
            slice_id=slice_id,
            window_entry=entry,
        )
        if artifact.trace_signature != _retained_tail_trace_signature(adapter_template):
            raise ValueError("gradient-balanced validation retained-tail trace drift")
        for objective in (CONTROL, CANDIDATE):
            statistics = evaluator(
                parameters[objective],
                artifact.batch,
                artifact.day_inputs,
            )
            jax.block_until_ready(statistics)
            accumulators[objective][slice_id] = add_rollout_screen_statistics(
                accumulators[objective][slice_id],
                jax.device_get(statistics),
            )
        print(
            json.dumps(
                {
                    "event": "gradient_balanced_smoke_validation",
                    "ordinal": ordinal,
                    "total": len(selected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        objective: {
            "slices": {
                slice_id: finalize_staged_validation_statistics(
                    accumulators[objective][slice_id],
                    primary_outputs=contract.payload["validation"]["primary_outputs"],
                )
                for slice_id in REQUIRED_VALIDATION_SLICES
            }
        }
        for objective in (CONTROL, CANDIDATE)
    }


def classify_gradient_balanced_smoke(
    validation: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    gradient_clip: float,
) -> Mapping[str, Any]:
    objective_validation = {
        "current_objective_control": validation[CONTROL],
        "named_biomass_candidate": validation[CANDIDATE],
    }
    validation_result = classify_objective_ab(
        objective_validation,
        gates=gates,
    )
    feasible_updates = sum(
        float(update["candidate"]["minimum_component_cosine"])
        >= float(gates["minimum_candidate_direction_component_cosine"])
        for update in updates
    )
    candidate_clip_updates = sum(
        float(update["candidate"]["combined_gradient_norm"]) > gradient_clip
        for update in updates
    )
    training_checks = {
        "direction_feasible_updates": feasible_updates
        >= int(gates["minimum_direction_feasible_updates"]),
        "candidate_gradient_clip_within_limit": candidate_clip_updates
        <= int(gates["maximum_candidate_gradient_clip_updates"]),
    }
    passed = validation_result["passed"] and all(training_checks.values())
    return {
        "status": (
            "gradient_balanced_smoke_admitted"
            if passed
            else "gradient_balanced_smoke_rejected"
        ),
        "passed": passed,
        "training_checks": training_checks,
        "direction_feasible_updates": feasible_updates,
        "candidate_gradient_clip_updates": candidate_clip_updates,
        "validation": validation_result,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    smoke = _load_contract(
        args.smoke_contract,
        schema_version="successor_gradient_balanced_smoke_v1",
        status="frozen_10_update_free_gpu_no_checkpoint",
    )
    acceptance = _load_contract(
        args.feasibility_acceptance,
        schema_version="successor_gradient_feasibility_acceptance_v1",
        status="accepted_zero_update_common_descent_supported",
    )
    candidate = _load_contract(
        args.candidate_coefficient_contract,
        schema_version="successor_named_biomass_candidate_coefficients_v1",
        status="accepted_zero_update_preflight_no_training_authorized",
    )
    if (
        smoke["identity"]["gradient_feasibility_acceptance_sha256"]
        != acceptance["contract_sha256"]
        or _sha256_file(args.feasibility_report)
        != acceptance["identity"]["report_file_sha256"]
    ):
        raise ValueError("gradient-balanced feasibility identity drift")

    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        STAGED_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        successor_learning_rate,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        derive_long_horizon_scales,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        load_named_biomass_tendency_scale_asset,
        load_successor_training_assets,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if (
        contract.payload["schema_version"] != STAGED_SCHEMA_VERSION
        or smoke["identity"]["staged_training_contract_sha256"] != contract.sha256
        or smoke["identity"]["candidate_coefficient_contract_sha256"]
        != candidate["contract_sha256"]
    ):
        raise ValueError("gradient-balanced training identity drift")
    arms = tuple(contract.payload["matched_arms"])
    references = runtime["index"].select(spatial_split="train", temporal_split="train")
    schedule = build_successor_small_fit_schedule(
        references,
        contract,
        updates=int(contract.sampling["maximum_updates_per_arm"]),
    )
    verify_successor_small_fit_schedule(schedule, references, contract)
    start_update, end_update = stage_update_range(
        contract.sampling["stage_milestones"],
        2000,
    )
    manifest = load_staged_prepared_manifest(
        args.prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        full_schedule_sha256=schedule["schedule_sha256"],
        start_update=start_update,
        end_update=end_update,
    )
    if len(manifest["entries"]) < SMOKE_UPDATES * POINTS_PER_MIXED_UPDATE:
        raise ValueError("gradient-balanced prepared manifest is too short")
    validation_prepared, prepared_panel = load_staged_validation_prepared(
        args.validation_prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        validation_panel_sha256=args.validation_panel_sha256,
    )
    assets = load_successor_training_assets(
        args.coefficient_contract,
        args.validation_panel,
        coefficient_report_path=args.coefficient_report,
        prepared_manifest_path=args.coefficient_prepared_manifest,
        runner_contract=contract,
        components=LONG_HORIZON_COMPONENTS,
    )
    if prepared_panel != assets.validation_panel:
        raise ValueError("gradient-balanced validation panel drift")
    scale_asset = load_named_biomass_tendency_scale_asset(
        args.tendency_scale_asset,
        args.tendency_scale_acceptance,
    )
    scale_acceptance = json.loads(
        args.tendency_scale_acceptance.read_text(encoding="utf-8")
    )
    if scale_acceptance["contract_sha256"] != smoke["identity"][
        "scale_acceptance_contract_sha256"
    ]:
        raise ValueError("gradient-balanced tendency-scale identity drift")

    identity = _training_identity(runtime, contract, schedule, assets, validation_prepared)
    first_record = manifest["entries"][0]
    first_entry = schedule["entries"][int(first_record["update"]) - 1]
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        first_record,
        schedule_entry=first_entry,
    )
    _reference, adapter_template, adapter_timing = _prepare_adapter(runtime, first_entry)
    signature = _retained_tail_trace_signature(adapter_template)
    if first_artifact.trace_signature != signature:
        raise ValueError("gradient-balanced retained-tail trace drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(lambda value: value[0], first_artifact.batch.anchor_inputs)
    )
    initialized = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(int(contract.sampling["seed"])),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=assembled,
            arm=arm,
        )
        for arm in arms
    }
    counts = {arm: successor_parameter_count(initialized[arm]) for arm in arms}
    if counts.get(SOURCE_ROUTED_ATTENTION) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("gradient-balanced Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("gradient-balanced smoke requires update-1000 recovery")
    parent_parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
    parent_optimizer = recovery["optimizers"][SOURCE_ROUTED_ATTENTION]
    parameters = {CONTROL: parent_parameters, CANDIDATE: parent_parameters}
    optimizers = {CONTROL: parent_optimizer, CANDIDATE: parent_optimizer}
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    control_step = _make_mixed_train_step(
        runtime,
        SOURCE_ROUTED_ATTENTION,
        adapter_template,
        assets.coefficients,
        long_scales,
        contract,
    )
    tendency_coefficient = float(
        candidate["resolved_coefficients"]["tendency_mse"]
    )
    candidate_measurement = _make_measurement(
        runtime,
        adapter_template,
        runner_contract=contract,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        flux_coefficients=assets.coefficients,
        tendency_coefficient=tendency_coefficient,
    )
    gram_measurement = _make_gradient_gram(jax)
    candidate_apply = _make_candidate_apply_step(jax, contract)

    updates = []
    memory_before = _device_memory_report(runtime["devices"])
    for short_update in range(1, SMOKE_UPDATES + 1):
        offset = (short_update - 1) * POINTS_PER_MIXED_UPDATE
        group = []
        for group_index in range(POINTS_PER_MIXED_UPDATE):
            record = manifest["entries"][offset + group_index]
            entry = schedule["entries"][int(record["update"]) - 1]
            artifact = load_prepared_artifact(
                args.prepared_manifest,
                record,
                schedule_entry=entry,
            )
            if artifact.trace_signature != signature:
                raise ValueError("gradient-balanced prepared trace drift")
            group.append((entry, artifact))
        point_years = {
            (str(entry["landpoint_id"]), int(entry["year"]))
            for entry, _artifact in group
        }
        if len(point_years) != POINTS_PER_MIXED_UPDATE:
            raise ValueError("gradient-balanced group repeated a point-year")
        contexts = tuple(
            _slice_context_batch(jax, artifact.batch, context_index=index)
            for index, (_entry, artifact) in enumerate(group)
        )
        anchor_batch = _join_context_anchors(jax, contexts)
        context_batch = _stack_contexts(jax, contexts)
        context_inputs = _stack_contexts(
            jax,
            tuple(artifact.day_inputs for _entry, artifact in group),
        )
        absolute_update = PARENT_UPDATE + short_update
        learning_rate = np.asarray(
            successor_learning_rate(contract, absolute_update),
            dtype=np.float64,
        )
        control_result = control_step(
            parameters[CONTROL],
            optimizers[CONTROL],
            anchor_batch,
            context_batch,
            context_inputs,
            learning_rate,
        )
        jax.block_until_ready(control_result)
        control_hard = _hard_counts_host(control_result.hard_counts)
        if not bool(np.asarray(control_result.update_applied)):
            raise RuntimeError("gradient-balanced control update failed")
        parameters[CONTROL] = control_result.parameters
        optimizers[CONTROL] = control_result.optimizer
        next_parameters, next_optimizer, candidate_record = _candidate_update(
            jax,
            candidate_measurement,
            gram_measurement,
            candidate_apply,
            parameters[CANDIDATE],
            optimizers[CANDIDATE],
            anchor_batch,
            context_batch,
            context_inputs,
            learning_rate,
        )
        parameters[CANDIDATE] = next_parameters
        optimizers[CANDIDATE] = next_optimizer
        updates.append(
            {
                "short_update": short_update,
                "optimizer_update": absolute_update,
                "learning_rate": float(learning_rate),
                "point_years": [
                    {
                        "update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                "control": {
                    "loss": float(np.asarray(control_result.loss)),
                    "one_day_anchor": float(np.asarray(control_result.anchor)),
                    "gradient_norm": float(np.asarray(control_result.gradient_norm)),
                    "hard_gate_counts": control_hard,
                },
                "candidate": candidate_record,
            }
        )
        del group, contexts, anchor_batch, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "gradient_balanced_smoke_update",
                    "update": short_update,
                    "total": SMOKE_UPDATES,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    first_slice, first_validation_entry = ordered_validation_entries(
        assets.validation_panel
    )[0]
    validation_reference = _validation_reference(
        runtime["index"],
        first_slice,
        first_validation_entry,
    )
    validation_adapter = _prepare_validation_adapter(
        runtime,
        validation_reference,
        int(first_validation_entry["start_rows"][0]),
    )
    validation = _evaluate_subset(
        runtime,
        contract=contract,
        parameters=parameters,
        long_scales=long_scales,
        adapter_template=validation_adapter,
        prepared_manifest_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=assets.validation_panel,
    )
    classification = classify_gradient_balanced_smoke(
        validation,
        updates,
        gates=smoke["gates"],
        gradient_clip=float(contract.optimizer["global_gradient_norm_clip"]),
    )
    report = {
        "schema_version": "successor_gradient_balanced_smoke_report_v1",
        "status": "completed",
        "git_head": _git_head(),
        "identity": {
            "smoke_contract_sha256": smoke["contract_sha256"],
            "feasibility_acceptance_sha256": acceptance["contract_sha256"],
            "feasibility_report_sha256": _sha256_file(args.feasibility_report),
            "staged_training_contract_sha256": contract.sha256,
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "updates_per_objective": SMOKE_UPDATES,
        "same_parent_parameters_and_optimizer": True,
        "same_training_groups_and_learning_rates": True,
        "classification": classification,
        "updates": updates,
        "validation": validation,
        "validation_subset_artifact_count": 6,
        "validation_subset_window_count": 12,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_written": False,
        "recovery_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "next_gate": (
            smoke["pass_action"]
            if classification["passed"]
            else smoke["fail_action"]
        ),
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--smoke-contract", type=Path, required=True)
    parser.add_argument("--feasibility-acceptance", type=Path, required=True)
    parser.add_argument("--feasibility-report", type=Path, required=True)
    parser.add_argument("--candidate-coefficient-contract", type=Path, required=True)
    parser.add_argument("--tendency-scale-asset", type=Path, required=True)
    parser.add_argument("--tendency-scale-acceptance", type=Path, required=True)
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-report", type=Path, required=True)
    parser.add_argument("--coefficient-prepared-manifest", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-panel-sha256", required=True)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--recovery-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
