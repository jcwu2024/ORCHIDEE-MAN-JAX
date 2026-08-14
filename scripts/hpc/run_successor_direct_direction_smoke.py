"""Run the frozen 10-update direct-direction Attention smoke."""

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
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
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
from scripts.hpc.probe_successor_direct_trust_region import (
    _make_combined_direction,
    _make_direct_virtual_step,
)
from scripts.hpc.probe_successor_gradient_feasibility import (
    PARENT_UPDATE,
    _load_contract,
    _make_loss_evaluator,
    _make_measurement,
    _sha256_file,
)
from scripts.hpc.probe_successor_realized_step import _relative_changes
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_gradient_balanced_smoke import (
    CANDIDATE as VALIDATION_CANDIDATE,
)
from scripts.hpc.run_successor_gradient_balanced_smoke import (
    CONTROL as VALIDATION_BASELINE,
)
from scripts.hpc.run_successor_gradient_balanced_smoke import (
    _evaluate_subset,
    _make_gradient_gram,
)
from scripts.hpc.run_successor_named_biomass_objective_ab import (
    classify_objective_ab,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import (
    _device_memory_report,
    _hard_counts_host,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.run_successor_staged_training import _training_identity
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


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _direct_update(
    jax: Any,
    measurement: Any,
    loss_evaluator: Any,
    gram_measurement: Any,
    combine: Any,
    direct_step: Any,
    parameters: Any,
    anchor_batch: Any,
    context_batch: Any,
    context_inputs: Any,
    step_norm: float,
) -> tuple[Any, Mapping[str, Any]]:
    gradients = []
    before = None
    parent_hard = None
    for seed in np.eye(len(GRADIENT_FEASIBILITY_LABELS), dtype=np.float64):
        losses, gradient, hard_counts = measurement(
            parameters,
            seed,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        jax.block_until_ready((losses, gradient, hard_counts))
        current = np.asarray(losses, dtype=np.float64)
        current_hard = _hard_counts_host(hard_counts)
        if before is None:
            before = current
            parent_hard = current_hard
        elif not np.array_equal(before, current) or parent_hard != current_hard:
            raise ValueError("direct-direction seed changed objective outputs")
        gradients.append(gradient)
    if before is None or parent_hard is None:
        raise ValueError("direct-direction update produced no gradients")

    gram, component_nonfinite = gram_measurement(tuple(gradients))
    jax.block_until_ready((gram, component_nonfinite))
    diagnostics = gradient_direction_diagnostics(np.asarray(gram, dtype=np.float64))
    selected = diagnostics["directions"]["minimum_norm_unit"]
    weights = np.asarray(
        [selected["weights"][label] for label in GRADIENT_FEASIBILITY_LABELS],
        dtype=np.float64,
    )
    direction, direction_norm, direction_nonfinite = combine(tuple(gradients), weights)
    proposed, proposed_nonfinite = direct_step(
        parameters,
        direction,
        direction_norm,
        np.asarray(step_norm, dtype=np.float64),
    )
    losses_after, hard_counts_after = loss_evaluator(
        proposed,
        anchor_batch,
        context_batch,
        context_inputs,
    )
    jax.block_until_ready(
        (
            proposed,
            direction_norm,
            direction_nonfinite,
            proposed_nonfinite,
            losses_after,
            hard_counts_after,
        )
    )
    after = np.asarray(losses_after, dtype=np.float64)
    after_hard = _hard_counts_host(hard_counts_after)
    relative = _relative_changes(before, after)
    nonfinite = {
        "component_gradients": int(
            np.sum(np.asarray(component_nonfinite, dtype=np.int64))
        ),
        "raw_direction": int(np.asarray(direction_nonfinite)),
        "proposed_parameters": int(np.asarray(proposed_nonfinite)),
        "losses": int(np.count_nonzero(~np.isfinite(before)))
        + int(np.count_nonzero(~np.isfinite(after))),
        "relative_changes": int(
            np.count_nonzero(~np.isfinite(tuple(relative.values())))
        ),
    }
    update_applied = (
        sum(parent_hard.values()) == 0
        and sum(after_hard.values()) == 0
        and sum(nonfinite.values()) == 0
    )
    record = {
        "update_applied": update_applied,
        "step_norm": step_norm,
        "losses_before": {
            label: float(before[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "losses_after": {
            label: float(after[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "loss_relative_changes": relative,
        "parent_hard_gate_counts": parent_hard,
        "proposed_hard_gate_counts": after_hard,
        "gradient_norms": diagnostics["gradient_norms"],
        "raw_direction": {
            "direction_norm": float(np.asarray(direction_norm)),
            "minimum_component_cosine": min(
                selected["normalized_directional_derivatives"].values()
            ),
            "component_cosines": selected["normalized_directional_derivatives"],
            "normalized_simplex_weights": selected["normalized_simplex_weights"],
            "raw_gradient_coefficients": selected["weights"],
        },
        "nonfinite_values": nonfinite,
    }
    next_parameters = proposed if update_applied else parameters
    del gradients, gram, component_nonfinite, direction, losses_after
    return next_parameters, record


def classify_direct_direction_smoke(
    validation: Mapping[str, Any],
    updates: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
) -> Mapping[str, Any]:
    total_updates = int(gates["total_updates"])
    relative_limit = float(
        gates["maximum_nonincrease_component_relative_increase"]
    )
    global_limit = float(gates["maximum_any_update_component_relative_increase"])
    median_decrease = float(gates["minimum_each_component_median_relative_decrease"])
    complete = len(updates) == total_updates and all(
        bool(update["update_applied"]) for update in updates
    )
    maximum_changes = [
        max(float(value) for value in update["loss_relative_changes"].values())
        for update in updates
    ]
    nonincrease_updates = sum(value <= relative_limit for value in maximum_changes)
    component_medians = {
        label: (
            float(
                np.median(
                    [
                        float(update["loss_relative_changes"][label])
                        for update in updates
                    ]
                )
            )
            if updates
            else None
        )
        for label in GRADIENT_FEASIBILITY_LABELS
    }
    hard_failure_count = sum(
        sum(update["parent_hard_gate_counts"].values())
        + sum(update["proposed_hard_gate_counts"].values())
        for update in updates
    )
    nonfinite_count = sum(
        sum(update["nonfinite_values"].values()) for update in updates
    )
    training_checks = {
        "all_updates_completed": complete,
        "hard_failures_zero": hard_failure_count
        == int(gates["hard_failure_count"]),
        "nonfinite_values_zero": nonfinite_count
        == int(gates["nonfinite_value_count"]),
        "enough_all_component_nonincrease_updates": nonincrease_updates
        >= int(gates["minimum_all_component_nonincrease_updates"]),
        "every_update_within_global_regression_bound": bool(maximum_changes)
        and max(maximum_changes) <= global_limit,
        "each_component_median_decreased": all(
            value is not None and value <= -median_decrease
            for value in component_medians.values()
        ),
    }
    validation_result = classify_objective_ab(
        {
            "current_objective_control": validation[VALIDATION_BASELINE],
            "named_biomass_candidate": validation[VALIDATION_CANDIDATE],
        },
        gates=gates,
    )
    passed = all(training_checks.values()) and validation_result["passed"]
    return {
        "status": (
            "direct_direction_smoke_admitted"
            if passed
            else "direct_direction_smoke_rejected"
        ),
        "passed": passed,
        "training_checks": training_checks,
        "all_component_nonincrease_updates": nonincrease_updates,
        "maximum_update_component_relative_increase": (
            max(maximum_changes) if maximum_changes else None
        ),
        "component_median_relative_changes": component_medians,
        "hard_failure_count": hard_failure_count,
        "nonfinite_value_count": nonfinite_count,
        "validation": validation_result,
        "next_action": (
            "review_bounded_direct_direction_training_design"
            if passed
            else "close_direct_direction_training_and_audit_batch_to_rollout_generalization"
        ),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    smoke = _load_contract(
        args.smoke_contract,
        schema_version="successor_direct_direction_smoke_v1",
        status="frozen_10_update_free_gpu_no_checkpoint",
    )
    trust = _load_contract(
        args.trust_region_contract,
        schema_version="successor_direct_trust_region_audit_v1",
        status="frozen_zero_update_direct_direction_grid",
    )
    candidate = _load_contract(
        args.candidate_coefficient_contract,
        schema_version="successor_named_biomass_candidate_coefficients_v1",
        status="accepted_zero_update_preflight_no_training_authorized",
    )
    if (
        smoke["identity"]["direct_trust_region_contract_sha256"]
        != trust["contract_sha256"]
        or _sha256_file(args.trust_region_report)
        != smoke["identity"]["direct_trust_region_report_sha256"]
    ):
        raise ValueError("direct-direction trust-region identity drift")
    trust_report = json.loads(args.trust_region_report.read_text(encoding="utf-8"))
    if (
        trust_report.get("schema_version")
        != "successor_direct_trust_region_audit_report_v1"
        or trust_report.get("status") != "completed"
        or not trust_report.get("evidence_gates", {}).get("evidence_valid", False)
        or trust_report.get("classification", {}).get("status")
        != "direct_trust_region_supported"
        or float(trust_report["classification"]["selected_step_norm"])
        != float(smoke["update"]["global_step_norm"])
    ):
        raise ValueError("direct-direction selected step identity drift")

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
        raise ValueError("direct-direction training identity drift")
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
        raise ValueError("direct-direction prepared manifest is too short")
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
        raise ValueError("direct-direction validation panel drift")
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
        raise ValueError("direct-direction tendency-scale identity drift")

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
        raise ValueError("direct-direction retained-tail trace drift")
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
        raise ValueError("direct-direction Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("direct-direction smoke requires update-1000 recovery")
    parent_parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
    parameters = parent_parameters
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    tendency_coefficient = float(candidate["resolved_coefficients"]["tendency_mse"])
    objective_arguments = {
        "runner_contract": contract,
        "long_scales": long_scales,
        "tendency_scales": scale_asset.scales,
        "flux_coefficients": assets.coefficients,
        "tendency_coefficient": tendency_coefficient,
    }
    measurement = _make_measurement(runtime, adapter_template, **objective_arguments)
    loss_evaluator = _make_loss_evaluator(
        runtime,
        adapter_template,
        **objective_arguments,
    )
    gram_measurement = _make_gradient_gram(jax)
    combine = _make_combined_direction(jax)
    direct_step = _make_direct_virtual_step(jax)
    step_norm = float(smoke["update"]["global_step_norm"])

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
                raise ValueError("direct-direction prepared trace drift")
            group.append((entry, artifact))
        point_years = {
            (str(entry["landpoint_id"]), int(entry["year"]))
            for entry, _artifact in group
        }
        if len(point_years) != POINTS_PER_MIXED_UPDATE:
            raise ValueError("direct-direction group repeated a point-year")
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
        parameters, update_record = _direct_update(
            jax,
            measurement,
            loss_evaluator,
            gram_measurement,
            combine,
            direct_step,
            parameters,
            anchor_batch,
            context_batch,
            context_inputs,
            step_norm,
        )
        updates.append(
            {
                "short_update": short_update,
                "scheduled_update": PARENT_UPDATE + short_update,
                "point_years": [
                    {
                        "update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                **update_record,
            }
        )
        del group, contexts, anchor_batch, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "direct_direction_smoke_update",
                    "update": short_update,
                    "total": SMOKE_UPDATES,
                    "applied": update_record["update_applied"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not update_record["update_applied"]:
            break

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
        parameters={
            VALIDATION_BASELINE: parent_parameters,
            VALIDATION_CANDIDATE: parameters,
        },
        long_scales=long_scales,
        adapter_template=validation_adapter,
        prepared_manifest_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=assets.validation_panel,
    )
    classification = classify_direct_direction_smoke(
        validation,
        updates,
        gates=smoke["gates"],
    )
    report = {
        "schema_version": "successor_direct_direction_smoke_report_v1",
        "status": "completed",
        "git_head": _git_head(),
        "identity": {
            "smoke_contract_sha256": smoke["contract_sha256"],
            "trust_region_report_sha256": _sha256_file(args.trust_region_report),
            "staged_training_contract_sha256": contract.sha256,
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "step_norm": step_norm,
        "updates": updates,
        "validation": validation,
        "classification": classification,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_written": False,
        "recovery_written": False,
        "final_parameters_retained": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "next_gate": classification["next_action"],
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
    parser.add_argument("--trust-region-contract", type=Path, required=True)
    parser.add_argument("--trust-region-report", type=Path, required=True)
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
