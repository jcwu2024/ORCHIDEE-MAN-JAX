"""Attribute the stopped direct-step failure to owned parameter blocks."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.parameter_geometry import (
    SUCCESSOR_PARAMETER_BLOCKS,
    exclude_successor_parameter_block,
    successor_parameter_geometry_report,
)
from research.daily_coarse_graining.daily_operator.successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from scripts.hpc.probe_successor_batch_diversity import POINTS_PER_MIXED_UPDATE
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
from scripts.hpc.probe_successor_multibatch_direction import (
    AUDIT_GROUPS,
    GRADIENT_FEASIBILITY_LABELS,
    _change_matrix_summary,
    _evaluate_active_set_step,
    _frozen_scalarization_seeds,
    _load_group,
    _make_loss_directional_evaluator,
    _make_nonfinite_counter,
    _recalibrate_active_set_direction,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
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


def classify_parameter_geometry_attribution(
    geometry: Mapping[str, Any],
    reference: Mapping[str, Any],
    projections: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
    pass_action: str,
    fail_action: str,
) -> Mapping[str, Any]:
    """Require a faithful failed reference and at least one safe exclusion."""

    geometry_checks = {
        "block_count_exact": int(geometry["block_count"])
        == int(gates["geometry"]["expected_block_count"]),
        "top_level_key_count_exact": int(geometry["top_level_key_count"])
        == int(gates["geometry"]["expected_top_level_key_count"]),
        "direction_fraction_closes": abs(
            float(geometry["direction_squared_fraction_sum"]) - 1.0
        )
        <= float(
            gates["geometry"]["maximum_direction_squared_fraction_sum_error"]
        ),
        "geometry_finite": int(geometry["nonfinite_value_count"])
        == int(gates["geometry"]["nonfinite_value_count"]),
    }
    reference_checks = {
        "reference_reproduced": bool(reference["reference_reproduced"]),
        "direction_passed": bool(reference["direction_passed"]),
        "finite_step_failed": not bool(reference["finite_step_safe"]),
        "hard_failures_zero": int(reference["hard_failure_count"])
        == int(gates["projection"]["hard_failure_count"]),
        "nonfinite_values_zero": int(reference["nonfinite_value_count"])
        == int(gates["projection"]["nonfinite_value_count"]),
    }
    safe = tuple(
        record["excluded_block"]
        for record in projections
        if record["direction_passed"] and record["finite_step_safe"]
    )
    projection_checks = {
        "candidate_count_exact": len(projections)
        == int(gates["geometry"]["expected_block_count"]),
        "each_block_tested_once": set(record["excluded_block"] for record in projections)
        == set(geometry["blocks"]),
        "hard_failures_zero": sum(
            int(record["hard_failure_count"]) for record in projections
        )
        == int(gates["projection"]["hard_failure_count"]),
        "nonfinite_values_zero": sum(
            int(record["nonfinite_value_count"]) for record in projections
        )
        == int(gates["projection"]["nonfinite_value_count"]),
        "safe_projection_found": len(safe)
        >= int(gates["attribution"]["minimum_safe_leave_one_out_projections"]),
    }
    passed = bool(
        all(geometry_checks.values())
        and all(reference_checks.values())
        and all(projection_checks.values())
    )
    return {
        "status": (
            "parameter_geometry_block_attributed"
            if passed
            else "parameter_geometry_not_attributed"
        ),
        "passed": passed,
        "geometry_checks": geometry_checks,
        "reference_checks": reference_checks,
        "projection_checks": projection_checks,
        "safe_excluded_blocks": list(safe),
        "direction_preserving_excluded_blocks": [
            record["excluded_block"]
            for record in projections
            if record["direction_passed"]
        ],
        "next_action": pass_action if passed else fail_action,
    }


def _summary_error(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> float:
    values = [
        abs(
            float(observed["maximum_group_component_relative_increase"])
            - float(expected["maximum_group_component_relative_increase"])
        )
    ]
    values.extend(
        abs(
            float(observed["component_median_relative_changes"][label])
            - float(expected["component_median_relative_changes"][label])
        )
        for label in GRADIENT_FEASIBILITY_LABELS
    )
    return max(values)


def _evaluate_projected_direction(
    *,
    jax: Any,
    parameters: Any,
    direction: Any,
    directional_evaluator: Any,
    loss_evaluator: Any,
    direct_step: Any,
    step_norm: float,
    gates: Mapping[str, Any],
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    prepared_manifest_path: Path,
    signature: Any,
) -> Mapping[str, Any]:
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    flat_direction, _unravel = ravel_pytree(direction)
    direction_norm = jnp.linalg.norm(flat_direction)
    unit_direction = jax.tree_util.tree_map(
        lambda value: value / jnp.maximum(direction_norm, 1.0e-30),
        direction,
    )
    direction_nonfinite = jnp.sum(~jnp.isfinite(flat_direction), dtype=jnp.int64)
    jax.block_until_ready((direction_norm, direction_nonfinite, unit_direction))
    derivatives = {}
    before_losses = []
    hard_failure_count = 0
    nonfinite_value_count = int(np.asarray(direction_nonfinite)) + int(
        not np.isfinite(float(np.asarray(direction_norm)))
        or float(np.asarray(direction_norm)) <= 0.0
    )
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            prepared_manifest_path,
            signature,
            group_index,
        )
        losses, directional, hard_counts = directional_evaluator(
            parameters,
            unit_direction,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        jax.block_until_ready((losses, directional, hard_counts))
        losses_host = np.asarray(losses, dtype=np.float64)
        directional_host = np.asarray(directional, dtype=np.float64)
        nonfinite_value_count += int(np.sum(~np.isfinite(losses_host)))
        nonfinite_value_count += int(np.sum(~np.isfinite(directional_host)))
        hard_host = _hard_counts_host(hard_counts)
        hard_failure_count += sum(hard_host.values())
        before_losses.append(losses_host)
        for component_index, label in enumerate(GRADIENT_FEASIBILITY_LABELS):
            constraint = f"group_{group_index + 1:02d}.{label}"
            derivatives[constraint] = float(
                directional_host[component_index]
                / max(abs(losses_host[component_index]), 1.0e-30)
            )
        del anchor_batch, context_batch, context_inputs, group
        del losses, directional, hard_counts
        gc.collect()
    minimum_derivative = min(derivatives.values())
    direction_passed = bool(
        minimum_derivative
        > float(gates["projection"]["minimum_relative_directional_derivative"])
        and hard_failure_count == 0
        and nonfinite_value_count == 0
    )
    finite_step = None
    if direction_passed:
        proposed, proposed_nonfinite = direct_step(
            parameters,
            direction,
            direction_norm,
            np.asarray(step_norm, dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposed_nonfinite))
        evaluated = _evaluate_active_set_step(
            jax=jax,
            loss_evaluator=loss_evaluator,
            parameters=proposed,
            proposal_nonfinite=int(np.asarray(proposed_nonfinite)),
            before_losses=before_losses,
            macro_before_losses=before_losses,
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=prepared_manifest_path,
            signature=signature,
        )
        finite_step = _change_matrix_summary(
            evaluated["current_change_matrix"],
            maximum_increase=float(
                gates["projection"]["maximum_group_component_relative_increase"]
            ),
        )
        hard_failure_count += int(evaluated["hard_failure_count"])
        nonfinite_value_count += int(evaluated["nonfinite_value_count"])
        finite_step["hard_failure_count"] = int(evaluated["hard_failure_count"])
        finite_step["nonfinite_value_count"] = int(
            evaluated["nonfinite_value_count"]
        )
        del proposed, evaluated
        gc.collect()
    return {
        "direction_norm": float(np.asarray(direction_norm)),
        "minimum_relative_directional_derivative": minimum_derivative,
        "direction_passed": direction_passed,
        "finite_step": finite_step,
        "finite_step_safe": bool(
            finite_step
            and finite_step["all_group_components_nonincreasing"]
            and finite_step["hard_failure_count"] == 0
            and finite_step["nonfinite_value_count"] == 0
        ),
        "hard_failure_count": hard_failure_count,
        "nonfinite_value_count": nonfinite_value_count,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    audit = _load_contract(
        args.audit_contract,
        schema_version="successor_parameter_geometry_attribution_v1",
        status="frozen_zero_retention_seven_block_leave_one_out",
    )
    predecessor_contract = _load_contract(
        args.predecessor_contract,
        schema_version="successor_group_aware_step_extension_v1",
        status="frozen_zero_update_forty_objective_extended_grid",
    )
    candidate = _load_contract(
        args.candidate_coefficient_contract,
        schema_version="successor_named_biomass_candidate_coefficients_v1",
        status="accepted_zero_update_preflight_no_training_authorized",
    )
    predecessor = json.loads(args.predecessor_report.read_text(encoding="utf-8"))
    cumulative = json.loads(args.cumulative_report.read_text(encoding="utf-8"))
    if (
        predecessor_contract["contract_sha256"]
        != audit["identity"]["group_aware_step_extension_contract_sha256"]
        or _sha256_file(args.predecessor_report)
        != audit["identity"]["group_aware_step_extension_report_sha256"]
        or _sha256_file(args.cumulative_report)
        != audit["identity"]["cumulative_microstep_report_sha256"]
        or cumulative.get("identity", {}).get("smoke_contract_sha256")
        != audit["identity"]["cumulative_microstep_contract_sha256"]
        or cumulative.get("classification", {}).get("status")
        != "cumulative_microstep_smoke_not_supported"
        or not cumulative.get("evidence_gates", {}).get("evidence_valid", False)
    ):
        raise ValueError("parameter-geometry predecessor identity drift")
    contract_blocks = {
        name: tuple(keys) for name, keys in audit["parameter_blocks"].items()
    }
    if contract_blocks != dict(SUCCESSOR_PARAMETER_BLOCKS):
        raise ValueError("parameter-geometry block contract drift")

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
        or audit["identity"]["staged_training_contract_sha256"] != contract.sha256
        or audit["identity"]["candidate_coefficient_contract_sha256"]
        != candidate["contract_sha256"]
    ):
        raise ValueError("parameter-geometry training identity drift")
    arms = tuple(contract.payload["matched_arms"])
    references = runtime["index"].select(
        spatial_split="train", temporal_split="train"
    )
    schedule = build_successor_small_fit_schedule(
        references,
        contract,
        updates=int(contract.sampling["maximum_updates_per_arm"]),
    )
    verify_successor_small_fit_schedule(schedule, references, contract)
    start_update, end_update = stage_update_range(
        contract.sampling["stage_milestones"], 2000
    )
    manifest = load_staged_prepared_manifest(
        args.prepared_manifest,
        staged_training_contract_sha256=contract.sha256,
        full_schedule_sha256=schedule["schedule_sha256"],
        start_update=start_update,
        end_update=end_update,
    )
    if len(manifest["entries"]) < AUDIT_GROUPS * POINTS_PER_MIXED_UPDATE:
        raise ValueError("parameter-geometry prepared manifest is too short")
    assets = load_successor_training_assets(
        args.coefficient_contract,
        args.validation_panel,
        coefficient_report_path=args.coefficient_report,
        prepared_manifest_path=args.coefficient_prepared_manifest,
        runner_contract=contract,
        components=LONG_HORIZON_COMPONENTS,
    )
    scale_asset = load_named_biomass_tendency_scale_asset(
        args.tendency_scale_asset,
        args.tendency_scale_acceptance,
    )
    scale_acceptance = json.loads(
        args.tendency_scale_acceptance.read_text(encoding="utf-8")
    )
    if (
        scale_acceptance["contract_sha256"]
        != audit["identity"]["scale_acceptance_contract_sha256"]
    ):
        raise ValueError("parameter-geometry tendency-scale identity drift")
    validation_identity = {
        "manifest_sha256": _sha256_file(args.validation_prepared_manifest)
    }
    identity = _training_identity(
        runtime, contract, schedule, assets, validation_identity
    )
    first_record = manifest["entries"][0]
    first_entry = schedule["entries"][int(first_record["update"]) - 1]
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        first_record,
        schedule_entry=first_entry,
    )
    _reference, adapter_template, adapter_timing = _prepare_adapter(
        runtime, first_entry
    )
    signature = _retained_tail_trace_signature(adapter_template)
    if first_artifact.trace_signature != signature:
        raise ValueError("parameter-geometry initial trace drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(
            lambda value: value[0], first_artifact.batch.anchor_inputs
        )
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
        raise ValueError("parameter-geometry Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("parameter-geometry requires update-1000 recovery")
    parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    objective_arguments = {
        "runner_contract": contract,
        "long_scales": long_scales,
        "tendency_scales": scale_asset.scales,
        "flux_coefficients": assets.coefficients,
        "tendency_coefficient": float(
            candidate["resolved_coefficients"]["tendency_mse"]
        ),
    }
    measurement = _make_measurement(runtime, adapter_template, **objective_arguments)
    loss_evaluator = _make_loss_evaluator(
        runtime, adapter_template, **objective_arguments
    )
    directional_evaluator = _make_loss_directional_evaluator(
        runtime, adapter_template, **objective_arguments
    )
    memory_before = _device_memory_report(runtime["devices"])

    direct_norms = tuple(
        float(value)
        for value in audit["stopped_state_reconstruction"]["direct_step_norms"]
    )
    microstep_norm = float(
        audit["stopped_state_reconstruction"]["safe_microstep_norm"]
    )
    seeds, parent_scalarization = _frozen_scalarization_seeds(
        predecessor, expected_step_norm=direct_norms[0]
    )
    active_parent_groups = tuple(
        int(index) for index in np.flatnonzero(np.any(seeds > 0.0, axis=1))
    )
    combine = _make_combined_direction(jax)
    direct_step = _make_direct_virtual_step(jax)
    count_nonfinite = _make_nonfinite_counter(jax)
    first_gradients = []
    reconstruction_hard = 0
    reconstruction_nonfinite = 0
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        if group_index in active_parent_groups:
            losses, gradient, hard_counts = measurement(
                parameters,
                seeds[group_index],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            gradient_nonfinite = count_nonfinite(gradient)
            jax.block_until_ready(
                (losses, gradient, hard_counts, gradient_nonfinite)
            )
            first_gradients.append(gradient)
            reconstruction_nonfinite += int(np.asarray(gradient_nonfinite))
        else:
            losses, hard_counts = loss_evaluator(
                parameters,
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses, hard_counts))
        reconstruction_hard += sum(_hard_counts_host(hard_counts).values())
        reconstruction_nonfinite += int(
            np.sum(~np.isfinite(np.asarray(losses, dtype=np.float64)))
        )
        del anchor_batch, context_batch, context_inputs, group, losses, hard_counts
        gc.collect()
    first_direction, first_direction_norm, first_direction_nonfinite = combine(
        tuple(first_gradients), np.ones(len(first_gradients), dtype=np.float64)
    )
    parameters, proposed_nonfinite = direct_step(
        parameters,
        first_direction,
        first_direction_norm,
        np.asarray(direct_norms[0], dtype=np.float64),
    )
    jax.block_until_ready(
        (
            parameters,
            first_direction_norm,
            first_direction_nonfinite,
            proposed_nonfinite,
        )
    )
    first_direction_error = abs(
        float(np.asarray(first_direction_norm))
        - float(parent_scalarization["expected_direction_norm"])
    ) / max(float(parent_scalarization["expected_direction_norm"]), 1.0e-30)
    reconstruction_nonfinite += int(np.asarray(first_direction_nonfinite))
    reconstruction_nonfinite += int(np.asarray(proposed_nonfinite))
    del first_gradients, first_direction
    gc.collect()

    constraint_labels = tuple(
        f"group_{group_index + 1:02d}.{label}"
        for group_index in range(AUDIT_GROUPS)
        for label in GRADIENT_FEASIBILITY_LABELS
    )
    parent_weights = predecessor["aggregate_direction"][
        "normalized_simplex_weights"
    ]
    support = tuple(
        label
        for label in constraint_labels
        if float(parent_weights[label])
        > float(audit["active_set"]["weight_tolerance"])
    )

    for step_index, step_norm in enumerate((direct_norms[1], microstep_norm)):
        recalibrated = _recalibrate_active_set_direction(
            jax=jax,
            measurement=measurement,
            directional_evaluator=directional_evaluator,
            parameters=parameters,
            initial_support=support,
            active_set=audit["active_set"],
            gates=audit["gates"]["reference"],
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=args.prepared_manifest,
            signature=signature,
        )
        proposed, proposed_nonfinite = direct_step(
            parameters,
            recalibrated["direction"],
            recalibrated["direction_norm"],
            np.asarray(step_norm, dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposed_nonfinite))
        evaluated = _evaluate_active_set_step(
            jax=jax,
            loss_evaluator=loss_evaluator,
            parameters=proposed,
            proposal_nonfinite=int(np.asarray(proposed_nonfinite)),
            before_losses=recalibrated["losses"],
            macro_before_losses=recalibrated["losses"],
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=args.prepared_manifest,
            signature=signature,
        )
        summary = _change_matrix_summary(
            evaluated["current_change_matrix"],
            maximum_increase=float(
                audit["gates"]["projection"][
                    "maximum_group_component_relative_increase"
                ]
            ),
        )
        reconstruction_hard += int(recalibrated["hard_failure_count"])
        reconstruction_hard += int(evaluated["hard_failure_count"])
        reconstruction_nonfinite += int(recalibrated["nonfinite_value_count"])
        reconstruction_nonfinite += int(evaluated["nonfinite_value_count"])
        if step_index == 1:
            expected = cumulative["microsteps"][0]
            safe_reference_error = max(
                _summary_error(summary, expected["current"]),
                abs(
                    float(recalibrated["minimum_relative_directional_derivative"])
                    - float(expected["minimum_relative_directional_derivative"])
                ),
            )
            safe_reference_support = tuple(recalibrated["support"]) == tuple(
                expected["final_support"]
            )
        parameters = proposed
        support = tuple(recalibrated["support"])
        del evaluated, recalibrated
        gc.collect()

    stopped = _recalibrate_active_set_direction(
        jax=jax,
        measurement=measurement,
        directional_evaluator=directional_evaluator,
        parameters=parameters,
        initial_support=support,
        active_set=audit["active_set"],
        gates=audit["gates"]["reference"],
        manifest=manifest,
        schedule=schedule,
        prepared_manifest_path=args.prepared_manifest,
        signature=signature,
    )
    full_proposed, full_proposed_nonfinite = direct_step(
        parameters,
        stopped["direction"],
        stopped["direction_norm"],
        np.asarray(microstep_norm, dtype=np.float64),
    )
    jax.block_until_ready((full_proposed, full_proposed_nonfinite))
    full_evaluated = _evaluate_active_set_step(
        jax=jax,
        loss_evaluator=loss_evaluator,
        parameters=full_proposed,
        proposal_nonfinite=int(np.asarray(full_proposed_nonfinite)),
        before_losses=stopped["losses"],
        macro_before_losses=stopped["losses"],
        manifest=manifest,
        schedule=schedule,
        prepared_manifest_path=args.prepared_manifest,
        signature=signature,
    )
    full_summary = _change_matrix_summary(
        full_evaluated["current_change_matrix"],
        maximum_increase=float(
            audit["gates"]["projection"][
                "maximum_group_component_relative_increase"
            ]
        ),
    )
    failed_reference = cumulative["microsteps"][1]
    tolerance = float(
        audit["stopped_state_reconstruction"]["maximum_reference_value_error"]
    )
    failed_reference_error = max(
        _summary_error(full_summary, failed_reference["current"]),
        abs(
            float(stopped["minimum_relative_directional_derivative"])
            - float(failed_reference["minimum_relative_directional_derivative"])
        ),
    )
    failed_reference_support = tuple(stopped["support"]) == tuple(
        failed_reference["final_support"]
    )
    reconstruction_hard += int(stopped["hard_failure_count"])
    reconstruction_hard += int(full_evaluated["hard_failure_count"])
    reconstruction_nonfinite += int(stopped["nonfinite_value_count"])
    reconstruction_nonfinite += int(full_evaluated["nonfinite_value_count"])
    reference_reproduced = bool(
        first_direction_error <= tolerance
        and safe_reference_error <= tolerance
        and safe_reference_support
        and failed_reference_error <= tolerance
        and failed_reference_support
    )
    geometry = successor_parameter_geometry_report(
        parameters,
        stopped["direction"],
        step_norm=microstep_norm,
    )
    reference = {
        "reference_reproduced": reference_reproduced,
        "direction_passed": bool(stopped["direction_passed"]),
        "minimum_relative_directional_derivative": float(
            stopped["minimum_relative_directional_derivative"]
        ),
        "finite_step": full_summary,
        "finite_step_safe": bool(
            full_summary["all_group_components_nonincreasing"]
            and full_evaluated["hard_failure_count"] == 0
            and full_evaluated["nonfinite_value_count"] == 0
        ),
        "hard_failure_count": int(stopped["hard_failure_count"])
        + int(full_evaluated["hard_failure_count"]),
        "nonfinite_value_count": int(stopped["nonfinite_value_count"])
        + int(full_evaluated["nonfinite_value_count"]),
    }
    del full_proposed, full_evaluated
    gc.collect()

    projections = []
    for block in SUCCESSOR_PARAMETER_BLOCKS:
        projection_started = time.perf_counter()
        projected = exclude_successor_parameter_block(stopped["direction"], block)
        result = _evaluate_projected_direction(
            jax=jax,
            parameters=parameters,
            direction=projected,
            directional_evaluator=directional_evaluator,
            loss_evaluator=loss_evaluator,
            direct_step=direct_step,
            step_norm=microstep_norm,
            gates=audit["gates"],
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=args.prepared_manifest,
            signature=signature,
        )
        record = {
            "excluded_block": block,
            **result,
            "elapsed_seconds": time.perf_counter() - projection_started,
        }
        projections.append(record)
        print(
            json.dumps(
                {
                    "event": "parameter_geometry_projection",
                    "excluded_block": block,
                    "direction_passed": record["direction_passed"],
                    "finite_step_safe": record["finite_step_safe"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del projected
        gc.collect()

    classification = classify_parameter_geometry_attribution(
        geometry,
        reference,
        projections,
        gates=audit["gates"],
        pass_action=audit["pass_action"],
        fail_action=audit["fail_action"],
    )
    evidence_valid = bool(
        reference_reproduced
        and reconstruction_hard
        == int(audit["stopped_state_reconstruction"]["hard_failure_count"])
        and reconstruction_nonfinite
        == int(audit["stopped_state_reconstruction"]["nonfinite_value_count"])
    )
    report = {
        "schema_version": "successor_parameter_geometry_attribution_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": subprocess_git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.predecessor_report),
            "cumulative_report_sha256": _sha256_file(args.cumulative_report),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "reconstruction": {
            "first_direction_norm_relative_error": first_direction_error,
            "safe_reference_error": safe_reference_error,
            "safe_reference_support_reproduced": safe_reference_support,
            "failed_reference_error": failed_reference_error,
            "failed_reference_support_reproduced": failed_reference_support,
            "hard_failure_count": reconstruction_hard,
            "nonfinite_value_count": reconstruction_nonfinite,
        },
        "geometry": geometry,
        "reference": reference,
        "projections": projections,
        "classification": classification,
        "evidence_gates": {"evidence_valid": evidence_valid},
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "reconstructed_direct_updates": 3,
        "optimizer_updates": 0,
        "checkpoint_written": False,
        "recovery_written": False,
        "final_parameters_retained": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "next_gate": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_parameter_geometry_evidence"
        ),
    }
    _atomic_json(args.output, report)
    return report


def subprocess_git_head() -> str:
    import subprocess

    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--audit-contract", type=Path, required=True)
    parser.add_argument("--predecessor-contract", type=Path, required=True)
    parser.add_argument("--predecessor-report", type=Path, required=True)
    parser.add_argument("--cumulative-report", type=Path, required=True)
    parser.add_argument("--candidate-coefficient-contract", type=Path, required=True)
    parser.add_argument("--tendency-scale-asset", type=Path, required=True)
    parser.add_argument("--tendency-scale-acceptance", type=Path, required=True)
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-report", type=Path, required=True)
    parser.add_argument("--coefficient-prepared-manifest", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
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
