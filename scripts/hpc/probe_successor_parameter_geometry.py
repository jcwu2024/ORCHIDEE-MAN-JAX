"""Attribute the stopped direct-step failure to owned parameter blocks."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.output_locality import (
    aggregate_normalized_output_changes,
    aggregate_output_locality,
    classify_maintenance_output_projection,
    classify_output_locality,
    summarize_normalized_output_change,
    summarize_output_locality_group,
)
from research.daily_coarse_graining.daily_operator.parameter_geometry import (
    SUCCESSOR_DECODER_LEAVES,
    SUCCESSOR_PARAMETER_BLOCKS,
    exclude_successor_decoder_leaf,
    exclude_successor_parameter_block,
    successor_decoder_leaf_geometry_report,
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
    _FrozenPredictionHead,
    _hard_counts_host,
    _head,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.run_successor_staged_training import _training_identity
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)

OUTPUT_LOCALITY_LOSS_LABELS = GRADIENT_FEASIBILITY_LABELS[1:]
MAINTENANCE_OUTPUT_FIELD = "daily.fast_maintenance_respiration"


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

    expected_direction_fraction = float(
        gates["geometry"].get("expected_direction_squared_fraction_sum", 1.0)
    )
    geometry_checks = {
        "block_count_exact": int(geometry["block_count"])
        == int(gates["geometry"]["expected_block_count"]),
        "top_level_key_count_exact": int(geometry["top_level_key_count"])
        == int(gates["geometry"]["expected_top_level_key_count"]),
        "direction_fraction_closes": abs(
            float(geometry["direction_squared_fraction_sum"])
            - expected_direction_fraction
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


def _make_output_locality_functions(
    runtime: Mapping[str, Any],
    adapter_template: Any,
    *,
    runner_contract: Any,
    long_scales: Any,
    tendency_scales: Mapping[int, Mapping[str, float]],
    flux_coefficients: Mapping[str, float],
    tendency_coefficient: float,
) -> Mapping[str, Any]:
    """Compile parameter-to-output tapes and output-tape loss derivatives."""

    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
        physical_outputs_from_training_representation,
        training_representation_outputs,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        DailyOperatorRolloutTrace,
        extract_biomass_productivity_trajectories,
        rollout_input_from_template,
        trajectories_from_rollout,
    )
    from scripts.hpc.probe_successor_named_biomass_gradients import (
        _calibration_loss_vector,
    )

    jax = runtime["jax"]
    head = _head(runtime, SOURCE_ROUTED_ATTENTION)
    schema = runtime["schema"]
    input_assembler = runtime["adapter"].input_assembler

    def _frozen_transition(day_inputs: Any) -> Any:
        retained_tail = replace(adapter_template, day_inputs=day_inputs)
        return bind_daily_operator(
            input_assembler=input_assembler,
            process_head=_FrozenPredictionHead(),
            retained_tail=retained_tail,
        )

    def parameter_tape_one(parameters: Any, batch: Any, day_inputs: Any) -> Any:
        templates = jax.tree_util.tree_map(lambda value: value[0], batch.rollout_inputs)
        continuous = jnp.asarray(templates.continuous_state, dtype=jnp.float64)
        initial_discrete = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value)[0], templates.discrete_state
        )
        transition = _frozen_transition(day_inputs)

        def step(carry: Any, template: Any) -> tuple[Any, Any]:
            current = rollout_input_from_template(
                template,
                carry[0],
                carry[1],
                runtime["contract"],
            )
            assembled = input_assembler.assemble(current)
            context = head.assemble_context(parameters, assembled)
            physical = head.named_outputs(parameters, context, assembled)
            represented = training_representation_outputs(physical, assembled, schema)
            prediction = head.prediction_from_named_outputs(physical, assembled)
            result = transition(prediction, current)
            return (result.next_state.other, result.next_discrete_state), represented

        _, tape = jax.lax.scan(
            step,
            (continuous[0], initial_discrete),
            templates,
        )
        return tape

    def tape_trajectory_one(tape: Any, batch: Any, day_inputs: Any) -> Any:
        templates = jax.tree_util.tree_map(lambda value: value[0], batch.rollout_inputs)
        continuous = jnp.asarray(templates.continuous_state, dtype=jnp.float64)
        initial_discrete = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value)[0], templates.discrete_state
        )
        transition = _frozen_transition(day_inputs)

        def step(carry: Any, values: Any) -> tuple[Any, Any]:
            template, represented = values
            current = rollout_input_from_template(
                template,
                carry[0],
                carry[1],
                runtime["contract"],
            )
            assembled = input_assembler.assemble(current)
            physical = physical_outputs_from_training_representation(
                represented,
                assembled,
                schema,
            )
            prediction = head.prediction_from_named_outputs(physical, assembled)
            result = transition(prediction, current)
            trace = DailyOperatorRolloutTrace(
                result=result,
                pft_fraction=current.pft.pft_fraction,
                active_pft_mask=current.pft.active_pft_mask,
            )
            return (result.next_state.other, result.next_discrete_state), trace

        _, trace = jax.lax.scan(
            step,
            (continuous[0], initial_discrete),
            (templates, tape),
        )
        predicted = trajectories_from_rollout(trace, runtime["contract"])
        expected = extract_biomass_productivity_trajectories(
            batch.expected_next_state[0],
            batch.expected_next_state_defined[0],
            daily_fast_gpp=batch.expected_fast_gpp[0],
            daily_fast_gpp_defined=batch.expected_fast_gpp_defined[0],
            pft_fraction=templates.pft.pft_fraction,
            active_pft_mask=templates.pft.active_pft_mask,
            contract=runtime["contract"],
        )
        batched_trace = jax.tree_util.tree_map(lambda value: value[None], trace)
        hard_counts = successor_rollout_hard_gate_counts(
            batch,
            batched_trace,
            markov_contract=runtime["contract"],
            budget_atol=float(
                runner_contract.payload["gates"]["budget_absolute_tolerance"]
            ),
            stock_atol=float(
                runner_contract.payload["gates"][
                    "stock_negative_absolute_tolerance"
                ]
            ),
        )
        return predicted, expected, hard_counts

    group_parameter_tape = jax.vmap(parameter_tape_one, in_axes=(None, 0, 0))

    def group_tape_objective(tape: Any, batches: Any, day_inputs: Any) -> Any:
        predicted, expected, counts = jax.vmap(tape_trajectory_one)(
            tape,
            batches,
            day_inputs,
        )
        hard_counts = jax.tree_util.tree_map(
            lambda values: jnp.sum(values, axis=0), counts
        )
        long_components = _calibration_loss_vector(
            jax,
            predicted,
            expected,
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            tendency_scales=tendency_scales,
            flux_coefficients=flux_coefficients,
        )
        losses = jnp.stack(
            (
                long_components[0],
                tendency_coefficient * long_components[1],
                tendency_coefficient * long_components[2],
            )
        )
        return losses, hard_counts

    @jax.jit
    def parameter_tape_jvp(
        parameters: Any,
        descent_direction: Any,
        batches: Any,
        day_inputs: Any,
    ) -> Any:
        return jax.jvp(
            lambda current: group_parameter_tape(current, batches, day_inputs),
            (parameters,),
            (descent_direction,),
        )

    parameter_tape = jax.jit(group_parameter_tape)

    def group_maintenance_tape(
        parameters: Any,
        batches: Any,
        day_inputs: Any,
    ) -> Any:
        return group_parameter_tape(parameters, batches, day_inputs)[
            MAINTENANCE_OUTPUT_FIELD
        ]

    maintenance_tape = jax.jit(group_maintenance_tape)

    @jax.jit
    def maintenance_tape_jvp(
        parameters: Any,
        descent_direction: Any,
        batches: Any,
        day_inputs: Any,
    ) -> Any:
        return jax.jvp(
            lambda current: group_maintenance_tape(
                current,
                batches,
                day_inputs,
            ),
            (parameters,),
            (descent_direction,),
        )

    maintenance_scale = jnp.asarray(
        runtime["output_normalization"].scale[MAINTENANCE_OUTPUT_FIELD],
        dtype=jnp.float64,
    )

    def group_maintenance_penalty(
        parameters: Any,
        base_maintenance: Any,
        batches: Any,
        day_inputs: Any,
    ) -> tuple[Any, Any]:
        proposed_maintenance = group_maintenance_tape(
            parameters,
            batches,
            day_inputs,
        )
        normalized_change = (
            proposed_maintenance - base_maintenance
        ) / maintenance_scale
        return jnp.mean(jnp.square(normalized_change)), proposed_maintenance

    maintenance_penalty_value_and_grad = jax.jit(
        jax.value_and_grad(group_maintenance_penalty, has_aux=True)
    )

    @jax.jit
    def tape_measurement(tape: Any, batches: Any, day_inputs: Any) -> Any:
        losses, pullback, hard_counts = jax.vjp(
            lambda current: group_tape_objective(current, batches, day_inputs),
            tape,
            has_aux=True,
        )
        basis = jnp.eye(len(OUTPUT_LOCALITY_LOSS_LABELS), dtype=jnp.float64)
        gradients = jax.lax.map(lambda seed: pullback(seed)[0], basis)
        return losses, gradients, hard_counts

    tape_evaluator = jax.jit(group_tape_objective)
    return {
        "parameter_tape_jvp": parameter_tape_jvp,
        "parameter_tape": parameter_tape,
        "maintenance_tape_jvp": maintenance_tape_jvp,
        "maintenance_tape": maintenance_tape,
        "maintenance_penalty_value_and_grad": maintenance_penalty_value_and_grad,
        "tape_measurement": tape_measurement,
        "tape_evaluator": tape_evaluator,
    }


def _run_output_locality_audit(
    *,
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    runtime: Mapping[str, Any],
    contract: Any,
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    adapter_template: Any,
    adapter_timing: Mapping[str, Any],
    signature: Any,
    parameters: Any,
    proposed_parameters: Any,
    direction: Any,
    direction_norm: Any,
    stopped_losses: Sequence[np.ndarray],
    proposed_losses: Sequence[np.ndarray],
    relative_derivatives: Mapping[str, float],
    long_scales: Any,
    scale_asset: Any,
    assets: Any,
    tendency_coefficient: float,
    reference: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    memory_before: Mapping[str, Any],
    started: float,
) -> Mapping[str, Any]:
    """Measure the failed finite step in normalized decoded-output coordinates."""

    jax = runtime["jax"]
    import jax.numpy as jnp

    step_norm = float(audit["finite_step"]["global_parameter_step_norm"])
    descent_direction = jax.tree_util.tree_map(
        lambda value: -value / jnp.maximum(direction_norm, 1.0e-30),
        direction,
    )
    functions = _make_output_locality_functions(
        runtime,
        adapter_template,
        runner_contract=contract,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        flux_coefficients=assets.coefficients,
        tendency_coefficient=tendency_coefficient,
    )
    field_labels = {
        field.name: runtime["schema"].labels[field.label_index]
        for field in runtime["schema"].fields
    }
    group_records = []
    for group_index in range(AUDIT_GROUPS):
        group_started = time.perf_counter()
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        del anchor_batch
        base_tape, output_tangent = functions["parameter_tape_jvp"](
            parameters,
            descent_direction,
            context_batch,
            context_inputs,
        )
        proposed_tape = functions["parameter_tape"](
            proposed_parameters,
            context_batch,
            context_inputs,
        )
        tape_base_losses, output_gradients, base_hard = functions[
            "tape_measurement"
        ](base_tape, context_batch, context_inputs)
        tape_proposed_losses, proposed_hard = functions["tape_evaluator"](
            proposed_tape,
            context_batch,
            context_inputs,
        )
        jax.block_until_ready(
            (
                base_tape,
                output_tangent,
                proposed_tape,
                tape_base_losses,
                output_gradients,
                base_hard,
                tape_proposed_losses,
                proposed_hard,
            )
        )
        base_host, tangent_host, proposed_host, gradients_host = jax.device_get(
            (base_tape, output_tangent, proposed_tape, output_gradients)
        )
        end_base = np.asarray(stopped_losses[group_index], dtype=np.float64)[1:]
        end_proposed = np.asarray(proposed_losses[group_index], dtype=np.float64)[
            1:
        ]
        first_order = np.asarray(
            [
                -step_norm
                * float(
                    relative_derivatives[
                        f"group_{group_index + 1:02d}.{label}"
                    ]
                )
                * max(abs(end_base[index]), 1.0e-30)
                for index, label in enumerate(OUTPUT_LOCALITY_LOSS_LABELS)
            ],
            dtype=np.float64,
        )
        group_record = summarize_output_locality_group(
            fields=runtime["schema"].fields,
            field_labels=field_labels,
            scales=runtime["output_normalization"].scale,
            base_outputs=base_host,
            output_tangents=tangent_host,
            proposed_outputs=proposed_host,
            output_gradients=gradients_host,
            loss_labels=OUTPUT_LOCALITY_LOSS_LABELS,
            base_losses=np.asarray(tape_base_losses, dtype=np.float64),
            proposed_losses=np.asarray(tape_proposed_losses, dtype=np.float64),
            end_to_end_base_losses=end_base,
            end_to_end_proposed_losses=end_proposed,
            end_to_end_first_order_changes=first_order,
            step_norm=step_norm,
        )
        base_hard_host = _hard_counts_host(base_hard)
        proposed_hard_host = _hard_counts_host(proposed_hard)
        group_record = {
            "group": group_index,
            "point_years": [
                {
                    "landpoint_id": str(entry["landpoint_id"]),
                    "year": int(entry["year"]),
                }
                for entry, _artifact in group
            ],
            **group_record,
            "base_hard_gate_counts": base_hard_host,
            "proposed_hard_gate_counts": proposed_hard_host,
            "hard_failure_count": sum(base_hard_host.values())
            + sum(proposed_hard_host.values()),
            "elapsed_seconds": time.perf_counter() - group_started,
        }
        group_records.append(group_record)
        print(
            json.dumps(
                {
                    "event": "successor_output_locality_group",
                    "group": group_index + 1,
                    "total": AUDIT_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del context_batch, context_inputs, group
        del base_tape, output_tangent, proposed_tape, output_gradients
        del tape_base_losses, tape_proposed_losses, base_hard, proposed_hard
        del base_host, tangent_host, proposed_host, gradients_host
        gc.collect()

    aggregate = aggregate_output_locality(
        group_records,
        loss_labels=OUTPUT_LOCALITY_LOSS_LABELS,
        top_k=int(audit["localization"]["top_k_fields"]),
    )
    classification = classify_output_locality(
        aggregate,
        group_records,
        gates=audit["gates"],
        localized_action=audit["localized_action"],
        distributed_action=audit["distributed_action"],
        downstream_action=audit["downstream_action"],
    )
    reconstruction_checks = {
        "stopped_reference_reproduced": bool(reference["reference_reproduced"]),
        "common_descent_direction_reproduced": bool(reference["direction_passed"]),
        "failed_finite_step_reproduced": not bool(reference["finite_step_safe"]),
        "reconstruction_hard_failures_zero": int(
            reconstruction["hard_failure_count"]
        )
        == int(audit["stopped_state_reconstruction"]["hard_failure_count"]),
        "reconstruction_nonfinite_values_zero": int(
            reconstruction["nonfinite_value_count"]
        )
        == int(audit["stopped_state_reconstruction"]["nonfinite_value_count"]),
    }
    evidence_valid = bool(
        classification["evidence_valid"] and all(reconstruction_checks.values())
    )
    classification = {
        **classification,
        "status": (
            classification["status"]
            if evidence_valid
            else "invalid_output_locality_evidence"
        ),
        "evidence_valid": evidence_valid,
        "reconstruction_checks": reconstruction_checks,
        "next_action": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_output_locality_evidence"
        ),
    }
    public_groups = [
        {name: value for name, value in group.items() if name != "field_records"}
        for group in group_records
    ]
    report = {
        "schema_version": "successor_output_locality_audit_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": subprocess_git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "decoder_leaf_report_sha256": _sha256_file(args.decoder_leaf_report),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": audit["model"]["parameter_count"],
        "parent_update": PARENT_UPDATE,
        "representation": audit["output_coordinates"],
        "reference": reference,
        "reconstruction": reconstruction,
        "groups": public_groups,
        "aggregate": aggregate,
        "classification": classification,
        "evidence_gates": {
            "evidence_valid": evidence_valid,
            "hard_failure_count": sum(
                int(group["hard_failure_count"]) for group in group_records
            ),
            "nonfinite_value_count": int(aggregate["nonfinite_value_count"]),
        },
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
        "next_gate": classification["next_action"],
    }
    _atomic_json(args.output, report)
    return report


def _exact_maintenance_change(
    aggregate: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "group_count": int(aggregate["group_count"]),
        "scalar_count": int(aggregate["scalar_count"]),
        "normalized_exact_rms": float(aggregate["normalized_exact_rms"]),
        "maximum_absolute_exact_change": float(
            aggregate["maximum_absolute_exact_change"]
        ),
        "nonfinite_value_count": int(aggregate["nonfinite_value_count"]),
    }


def _run_maintenance_output_projection(
    *,
    args: argparse.Namespace,
    audit: Mapping[str, Any],
    runtime: Mapping[str, Any],
    contract: Any,
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    adapter_template: Any,
    adapter_timing: Mapping[str, Any],
    signature: Any,
    parameters: Any,
    raw_parameters: Any,
    direction: Any,
    direction_norm: Any,
    stopped_losses: Sequence[np.ndarray],
    safe_snapshot: Mapping[str, Any],
    raw_loss_summary: Mapping[str, Any],
    long_scales: Any,
    scale_asset: Any,
    assets: Any,
    tendency_coefficient: float,
    reference: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    output_locality_reference: Mapping[str, Any],
    loss_evaluator: Any,
    direct_step: Any,
    memory_before: Mapping[str, Any],
    started: float,
) -> Mapping[str, Any]:
    """Project the failed proposal along one maintenance-local gradient."""

    jax = runtime["jax"]
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    functions = _make_output_locality_functions(
        runtime,
        adapter_template,
        runner_contract=contract,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        flux_coefficients=assets.coefficients,
        tendency_coefficient=tendency_coefficient,
    )
    scale = np.asarray(
        runtime["output_normalization"].scale[MAINTENANCE_OUTPUT_FIELD],
        dtype=np.float64,
    )
    safe_descent = jax.tree_util.tree_map(
        lambda value: -value / jnp.maximum(safe_snapshot["direction_norm"], 1.0e-30),
        safe_snapshot["direction"],
    )
    raw_descent = jax.tree_util.tree_map(
        lambda value: -value / jnp.maximum(direction_norm, 1.0e-30),
        direction,
    )
    safe_step_norm = float(audit["stopped_state_reconstruction"]["safe_microstep_norm"])
    raw_step_norm = float(audit["matched_reference"]["raw_control_step_norm"])
    safe_records = []
    raw_records = []
    raw_base_by_group = []
    penalty_values = []
    penalty_gradient = None
    penalty_nonfinite_count = 0
    for group_index in range(AUDIT_GROUPS):
        group_started = time.perf_counter()
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        del anchor_batch
        safe_base, safe_tangent = functions["maintenance_tape_jvp"](
            safe_snapshot["parameters_before"],
            safe_descent,
            context_batch,
            context_inputs,
        )
        raw_base, raw_tangent = functions["maintenance_tape_jvp"](
            parameters,
            raw_descent,
            context_batch,
            context_inputs,
        )
        (penalty_result, group_gradient) = functions[
            "maintenance_penalty_value_and_grad"
        ](
            raw_parameters,
            raw_base,
            context_batch,
            context_inputs,
        )
        penalty_value, raw_proposed = penalty_result
        jax.block_until_ready(
            (
                safe_base,
                safe_tangent,
                raw_base,
                raw_tangent,
                penalty_value,
                raw_proposed,
                group_gradient,
            )
        )
        (
            safe_base_host,
            safe_tangent_host,
            raw_base_host,
            raw_tangent_host,
            raw_proposed_host,
        ) = jax.device_get(
            (
                safe_base,
                safe_tangent,
                raw_base,
                raw_tangent,
                raw_proposed,
            )
        )
        safe_records.append(
            summarize_normalized_output_change(
                base=safe_base_host,
                tangent=safe_tangent_host,
                proposed=raw_base_host,
                scale=scale,
                step_norm=safe_step_norm,
            )
        )
        raw_records.append(
            summarize_normalized_output_change(
                base=raw_base_host,
                tangent=raw_tangent_host,
                proposed=raw_proposed_host,
                scale=scale,
                step_norm=raw_step_norm,
            )
        )
        raw_base_by_group.append(raw_base_host)
        penalty_value_host = float(np.asarray(penalty_value))
        penalty_values.append(penalty_value_host)
        penalty_nonfinite_count += int(not np.isfinite(penalty_value_host))
        penalty_gradient = (
            jax.tree_util.tree_map(
                lambda value: value / AUDIT_GROUPS,
                group_gradient,
            )
            if penalty_gradient is None
            else jax.tree_util.tree_map(
                lambda total, value: total + value / AUDIT_GROUPS,
                penalty_gradient,
                group_gradient,
            )
        )
        jax.block_until_ready(penalty_gradient)
        print(
            json.dumps(
                {
                    "event": "successor_maintenance_penalty_group",
                    "group": group_index + 1,
                    "total": AUDIT_GROUPS,
                    "elapsed_seconds": time.perf_counter() - group_started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del context_batch, context_inputs, group
        del safe_base, safe_tangent, raw_base, raw_tangent
        del penalty_result, penalty_value, raw_proposed, group_gradient
        del safe_base_host, safe_tangent_host, raw_tangent_host, raw_proposed_host
        gc.collect()
    if penalty_gradient is None:
        raise ValueError("maintenance projection did not produce a gradient")

    safe_change = aggregate_normalized_output_changes(safe_records)
    raw_change = aggregate_normalized_output_changes(raw_records)
    flat_penalty_gradient, _unravel = ravel_pytree(penalty_gradient)
    penalty_gradient_norm = jnp.linalg.norm(flat_penalty_gradient)
    penalty_gradient_nonfinite = jnp.sum(
        ~jnp.isfinite(flat_penalty_gradient), dtype=jnp.int64
    )
    flat_raw_displacement, _unravel = ravel_pytree(
        jax.tree_util.tree_map(
            lambda proposed, base: proposed - base,
            raw_parameters,
            parameters,
        )
    )
    raw_displacement_norm = jnp.linalg.norm(flat_raw_displacement)
    raw_displacement_nonfinite = jnp.sum(
        ~jnp.isfinite(flat_raw_displacement), dtype=jnp.int64
    )
    jax.block_until_ready(
        (
            penalty_gradient,
            penalty_gradient_norm,
            penalty_gradient_nonfinite,
            raw_displacement_norm,
            raw_displacement_nonfinite,
        )
    )
    penalty_gradient_norm_host = float(np.asarray(penalty_gradient_norm))
    penalty_nonfinite_count += int(np.asarray(penalty_gradient_nonfinite))
    penalty_nonfinite_count += int(np.asarray(raw_displacement_nonfinite))
    penalty_nonfinite_count += int(
        not np.isfinite(penalty_gradient_norm_host)
        or penalty_gradient_norm_host <= 0.0
    )
    raw_displacement_norm_host = float(np.asarray(raw_displacement_norm))
    penalty_mean = float(np.mean(np.asarray(penalty_values, dtype=np.float64)))
    penalty_rms = float(np.sqrt(max(penalty_mean, 0.0)))
    penalty_rms_error = abs(
        penalty_rms - float(raw_change["normalized_exact_rms"])
    )
    cap_multiplier = float(audit["maintenance_coordinate"]["cap_multiplier"])
    maintenance_cap = {
        "source": audit["maintenance_coordinate"]["cap_source"],
        "multiplier": cap_multiplier,
        "normalized_exact_rms": cap_multiplier
        * float(safe_change["normalized_exact_rms"]),
        "maximum_absolute_exact_change": cap_multiplier
        * float(safe_change["maximum_absolute_exact_change"]),
    }
    safe_reference = {
        "step_norm": safe_step_norm,
        "loss_summary": safe_snapshot["loss_summary"],
        "maintenance_change": safe_change,
        "hard_failure_count": int(safe_snapshot["hard_failure_count"]),
        "nonfinite_value_count": int(safe_snapshot["nonfinite_value_count"])
        + int(safe_change["nonfinite_value_count"]),
    }
    raw_control = {
        "step_norm": raw_step_norm,
        "loss_summary": raw_loss_summary,
        "maintenance_change": raw_change,
        "maintenance_cap": maintenance_cap,
        "parameter_displacement_norm": raw_displacement_norm_host,
        "hard_failure_count": int(reference["hard_failure_count"]),
        "nonfinite_value_count": int(reference["nonfinite_value_count"])
        + int(raw_change["nonfinite_value_count"]),
    }

    candidate_records = []
    correction_norms = tuple(
        float(value) for value in audit["correction"]["candidate_correction_norms"]
    )
    for correction_norm in correction_norms:
        candidate_started = time.perf_counter()
        candidate_parameters, proposal_nonfinite = direct_step(
            raw_parameters,
            penalty_gradient,
            penalty_gradient_norm,
            np.asarray(correction_norm, dtype=np.float64),
        )
        flat_candidate_displacement, _unravel = ravel_pytree(
            jax.tree_util.tree_map(
                lambda candidate_value, base: candidate_value - base,
                candidate_parameters,
                parameters,
            )
        )
        candidate_displacement_norm = jnp.linalg.norm(flat_candidate_displacement)
        candidate_displacement_nonfinite = jnp.sum(
            ~jnp.isfinite(flat_candidate_displacement), dtype=jnp.int64
        )
        jax.block_until_ready(
            (
                candidate_parameters,
                proposal_nonfinite,
                candidate_displacement_norm,
                candidate_displacement_nonfinite,
            )
        )
        evaluated = _evaluate_active_set_step(
            jax=jax,
            loss_evaluator=loss_evaluator,
            parameters=candidate_parameters,
            proposal_nonfinite=int(np.asarray(proposal_nonfinite)),
            before_losses=stopped_losses,
            macro_before_losses=stopped_losses,
            manifest=manifest,
            schedule=schedule,
            prepared_manifest_path=args.prepared_manifest,
            signature=signature,
        )
        loss_summary = _change_matrix_summary(
            evaluated["current_change_matrix"],
            maximum_increase=float(
                audit["gates"]["projection"][
                    "maximum_group_component_relative_increase"
                ]
            ),
        )
        candidate_change_records = []
        for group_index in range(AUDIT_GROUPS):
            anchor_batch, context_batch, context_inputs, group = _load_group(
                jax,
                manifest,
                schedule,
                args.prepared_manifest,
                signature,
                group_index,
            )
            del anchor_batch
            candidate_maintenance = functions["maintenance_tape"](
                candidate_parameters,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready(candidate_maintenance)
            candidate_host = np.asarray(
                jax.device_get(candidate_maintenance), dtype=np.float64
            )
            candidate_change_records.append(
                summarize_normalized_output_change(
                    base=raw_base_by_group[group_index],
                    tangent=np.zeros_like(candidate_host),
                    proposed=candidate_host,
                    scale=scale,
                    step_norm=1.0,
                )
            )
            del context_batch, context_inputs, group
            del candidate_maintenance, candidate_host
            gc.collect()
        candidate_change = _exact_maintenance_change(
            aggregate_normalized_output_changes(candidate_change_records)
        )
        displacement_norm_host = float(np.asarray(candidate_displacement_norm))
        candidate_nonfinite = int(evaluated["nonfinite_value_count"])
        candidate_nonfinite += int(np.asarray(candidate_displacement_nonfinite))
        candidate_nonfinite += int(candidate_change["nonfinite_value_count"])
        record = {
            "correction_norm": correction_norm,
            "parameter_displacement_norm": displacement_norm_host,
            "displacement_fraction": displacement_norm_host
            / max(raw_displacement_norm_host, 1.0e-30),
            "loss_summary": loss_summary,
            "maintenance_change": candidate_change,
            "maintenance_cap": maintenance_cap,
            "hard_failure_count": int(evaluated["hard_failure_count"]),
            "nonfinite_value_count": candidate_nonfinite,
            "elapsed_seconds": time.perf_counter() - candidate_started,
        }
        candidate_records.append(record)
        print(
            json.dumps(
                {
                    "event": "successor_maintenance_projection_candidate",
                    "correction_norm": correction_norm,
                    "all_group_components_nonincreasing": loss_summary[
                        "all_group_components_nonincreasing"
                    ],
                    "maintenance_rms": candidate_change[
                        "normalized_exact_rms"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del candidate_parameters, proposal_nonfinite
        del flat_candidate_displacement, candidate_displacement_norm
        del candidate_displacement_nonfinite, evaluated, candidate_change_records
        gc.collect()

    classification = classify_maintenance_output_projection(
        safe_reference=safe_reference,
        raw_control=raw_control,
        candidates=candidate_records,
        gates={
            **audit["gates"],
            "candidate_correction_norms": list(correction_norms),
        },
        pass_action=audit["pass_action"],
        fail_action=audit["fail_action"],
    )
    tolerance = float(
        audit["stopped_state_reconstruction"]["maximum_reference_value_error"]
    )
    evidence_checks = {
        "output_locality_evidence_valid": bool(
            output_locality_reference.get("evidence_gates", {}).get(
                "evidence_valid", False
            )
        ),
        "output_locality_classification_reproduced": output_locality_reference.get(
            "classification", {}
        ).get("status")
        == "decoded_output_nonlinearity_localized",
        "stopped_reference_reproduced": bool(reference["reference_reproduced"]),
        "safe_reference_group_count_exact": int(safe_change["group_count"])
        == int(audit["gates"]["expected_group_count"]),
        "raw_control_group_count_exact": int(raw_change["group_count"])
        == int(audit["gates"]["expected_group_count"]),
        "raw_displacement_norm_reproduced": abs(
            raw_displacement_norm_host - raw_step_norm
        )
        <= tolerance,
        "penalty_rms_consistent": penalty_rms_error
        <= float(audit["gates"]["maximum_penalty_rms_consistency_error"]),
        "penalty_gradient_finite_nonzero": penalty_nonfinite_count
        == int(audit["gates"]["nonfinite_value_count"]),
        "reconstruction_hard_failures_zero": int(
            reconstruction["hard_failure_count"]
        )
        == int(audit["stopped_state_reconstruction"]["hard_failure_count"]),
        "reconstruction_nonfinite_values_zero": int(
            reconstruction["nonfinite_value_count"]
        )
        == int(audit["stopped_state_reconstruction"]["nonfinite_value_count"]),
    }
    evidence_valid = bool(
        classification["evidence_valid"] and all(evidence_checks.values())
    )
    classification = {
        **classification,
        "status": (
            classification["status"]
            if evidence_valid
            else "invalid_maintenance_output_projection_evidence"
        ),
        "passed": bool(classification["passed"] and evidence_valid),
        "evidence_valid": evidence_valid,
        "evidence_checks": evidence_checks,
        "next_action": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_maintenance_projection_evidence"
        ),
    }
    report = {
        "schema_version": "successor_maintenance_output_projection_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": subprocess_git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "output_locality_report_sha256": _sha256_file(
                args.output_locality_report
            ),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": audit["model"]["parameter_count"],
        "parent_update": PARENT_UPDATE,
        "maintenance_coordinate": audit["maintenance_coordinate"],
        "correction": {
            **audit["correction"],
            "gradient_norm": penalty_gradient_norm_host,
            "penalty_mean_squared_normalized_change": penalty_mean,
            "penalty_rms": penalty_rms,
            "penalty_rms_consistency_error": penalty_rms_error,
            "nonfinite_value_count": penalty_nonfinite_count,
        },
        "reference": reference,
        "reconstruction": reconstruction,
        "safe_reference": safe_reference,
        "raw_control": raw_control,
        "candidates": candidate_records,
        "classification": classification,
        "evidence_gates": {
            "evidence_valid": evidence_valid,
            "hard_failure_count": int(safe_reference["hard_failure_count"])
            + int(raw_control["hard_failure_count"])
            + sum(int(record["hard_failure_count"]) for record in candidate_records),
            "nonfinite_value_count": int(safe_reference["nonfinite_value_count"])
            + int(raw_control["nonfinite_value_count"])
            + int(penalty_nonfinite_count)
            + sum(
                int(record["nonfinite_value_count"])
                for record in candidate_records
            ),
        },
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
        "next_gate": classification["next_action"],
    }
    _atomic_json(args.output, report)
    return report


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    decoder_leaf = bool(args.decoder_leaf_attribution)
    output_locality = bool(args.output_locality)
    maintenance_projection = bool(args.maintenance_projection)
    if sum((decoder_leaf, output_locality, maintenance_projection)) > 1:
        raise ValueError(
            "choose decoder-leaf, output-locality, or maintenance projection"
        )
    audit = _load_contract(
        args.audit_contract,
        schema_version=(
            "successor_maintenance_output_projection_v1"
            if maintenance_projection
            else (
                "successor_output_locality_audit_v1"
                if output_locality
                else (
                    "successor_decoder_leaf_attribution_v1"
                    if decoder_leaf
                    else "successor_parameter_geometry_attribution_v1"
                )
            )
        ),
        status=(
            "frozen_zero_retention_maintenance_output_trust_region"
            if maintenance_projection
            else (
                "frozen_zero_retention_normalized_output_decomposition"
                if output_locality
                else (
                    "frozen_zero_retention_four_decoder_leaf_leave_one_out"
                    if decoder_leaf
                    else "frozen_zero_retention_seven_block_leave_one_out"
                )
            )
        ),
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
    parameter_geometry_reference = None
    decoder_leaf_reference = None
    output_locality_reference = None
    if decoder_leaf or output_locality or maintenance_projection:
        if args.parameter_geometry_report is None:
            raise ValueError("audit requires parameter-geometry report")
        parameter_geometry_reference = json.loads(
            args.parameter_geometry_report.read_text(encoding="utf-8")
        )
        if (
            _sha256_file(args.parameter_geometry_report)
            != audit["identity"]["parameter_geometry_report_sha256"]
            or parameter_geometry_reference.get("identity", {}).get(
                "audit_contract_sha256"
            )
            != audit["identity"]["parameter_geometry_contract_sha256"]
            or parameter_geometry_reference.get("classification", {}).get("status")
            != "parameter_geometry_not_attributed"
            or not parameter_geometry_reference.get("evidence_gates", {}).get(
                "evidence_valid", False
            )
        ):
            raise ValueError("parameter-geometry reference drift")
    if decoder_leaf:
        contract_leaves = {
            name: tuple(path) for name, path in audit["decoder_leaves"].items()
        }
        if contract_leaves != dict(SUCCESSOR_DECODER_LEAVES):
            raise ValueError("decoder-leaf contract drift")
    elif output_locality or maintenance_projection:
        if args.decoder_leaf_report is None:
            raise ValueError("output audit requires decoder-leaf report")
        decoder_leaf_reference = json.loads(
            args.decoder_leaf_report.read_text(encoding="utf-8")
        )
        if (
            _sha256_file(args.decoder_leaf_report)
            != audit["identity"]["decoder_leaf_report_sha256"]
            or decoder_leaf_reference.get("identity", {}).get(
                "audit_contract_sha256"
            )
            != audit["identity"]["decoder_leaf_contract_sha256"]
            or decoder_leaf_reference.get("classification", {}).get("status")
            != "parameter_geometry_not_attributed"
            or not decoder_leaf_reference.get("evidence_gates", {}).get(
                "evidence_valid", False
            )
        ):
            raise ValueError("decoder-leaf reference drift")
        if maintenance_projection:
            if args.output_locality_report is None:
                raise ValueError(
                    "maintenance projection requires output-locality report"
                )
            output_locality_reference = json.loads(
                args.output_locality_report.read_text(encoding="utf-8")
            )
            if (
                _sha256_file(args.output_locality_report)
                != audit["identity"]["output_locality_report_sha256"]
                or output_locality_reference.get("identity", {}).get(
                    "audit_contract_sha256"
                )
                != audit["identity"]["output_locality_contract_sha256"]
                or output_locality_reference.get("classification", {}).get(
                    "status"
                )
                != "decoded_output_nonlinearity_localized"
                or not output_locality_reference.get("evidence_gates", {}).get(
                    "evidence_valid", False
                )
            ):
                raise ValueError("output-locality reference drift")
    else:
        contract_blocks = {
            name: tuple(keys) for name, keys in audit["parameter_blocks"].items()
        }
        if contract_blocks != dict(SUCCESSOR_PARAMETER_BLOCKS):
            raise ValueError("parameter-geometry block contract drift")

    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    if (output_locality or maintenance_projection) and (
        len(runtime["schema"].fields)
        != int(audit["output_coordinates"]["expected_field_count"])
        or len(runtime["schema"].labels)
        != int(audit["output_coordinates"]["expected_label_count"])
    ):
        raise ValueError("output-locality schema registry drift")
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
    if (output_locality or maintenance_projection) and counts[
        SOURCE_ROUTED_ATTENTION
    ] != int(
        audit["model"]["parameter_count"]
    ):
        raise ValueError("output audit parameter count drift")
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

    safe_snapshot = None
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
            safe_snapshot = {
                "parameters_before": parameters,
                "direction": recalibrated["direction"],
                "direction_norm": recalibrated["direction_norm"],
                "losses_before": recalibrated["losses"],
                "losses_after": evaluated["losses"],
                "loss_summary": summary,
                "hard_failure_count": int(recalibrated["hard_failure_count"])
                + int(evaluated["hard_failure_count"]),
                "nonfinite_value_count": int(
                    recalibrated["nonfinite_value_count"]
                )
                + int(evaluated["nonfinite_value_count"]),
            }
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
    geometry = None
    if not maintenance_projection:
        geometry = (
            successor_decoder_leaf_geometry_report(
                parameters,
                stopped["direction"],
                step_norm=microstep_norm,
            )
            if decoder_leaf
            else successor_parameter_geometry_report(
                parameters,
                stopped["direction"],
                step_norm=microstep_norm,
            )
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
    reconstruction = {
        "first_direction_norm_relative_error": first_direction_error,
        "safe_reference_error": safe_reference_error,
        "safe_reference_support_reproduced": safe_reference_support,
        "failed_reference_error": failed_reference_error,
        "failed_reference_support_reproduced": failed_reference_support,
        "hard_failure_count": reconstruction_hard,
        "nonfinite_value_count": reconstruction_nonfinite,
    }
    if maintenance_projection:
        if output_locality_reference is None:
            raise ValueError("maintenance output-locality evidence was not loaded")
        if safe_snapshot is None:
            raise ValueError("maintenance safe reference was not reconstructed")
        return _run_maintenance_output_projection(
            args=args,
            audit=audit,
            runtime=runtime,
            contract=contract,
            schedule=schedule,
            manifest=manifest,
            adapter_template=adapter_template,
            adapter_timing=adapter_timing,
            signature=signature,
            parameters=parameters,
            raw_parameters=full_proposed,
            direction=stopped["direction"],
            direction_norm=stopped["direction_norm"],
            stopped_losses=stopped["losses"],
            safe_snapshot=safe_snapshot,
            raw_loss_summary=full_summary,
            long_scales=long_scales,
            scale_asset=scale_asset,
            assets=assets,
            tendency_coefficient=float(
                candidate["resolved_coefficients"]["tendency_mse"]
            ),
            reference=reference,
            reconstruction=reconstruction,
            output_locality_reference=output_locality_reference,
            loss_evaluator=loss_evaluator,
            direct_step=direct_step,
            memory_before=memory_before,
            started=started,
        )
    if output_locality:
        if decoder_leaf_reference is None:
            raise ValueError("output-locality decoder-leaf evidence was not loaded")
        return _run_output_locality_audit(
            args=args,
            audit=audit,
            runtime=runtime,
            contract=contract,
            schedule=schedule,
            manifest=manifest,
            adapter_template=adapter_template,
            adapter_timing=adapter_timing,
            signature=signature,
            parameters=parameters,
            proposed_parameters=full_proposed,
            direction=stopped["direction"],
            direction_norm=stopped["direction_norm"],
            stopped_losses=stopped["losses"],
            proposed_losses=full_evaluated["losses"],
            relative_derivatives=stopped["relative_directional_derivatives"],
            long_scales=long_scales,
            scale_asset=scale_asset,
            assets=assets,
            tendency_coefficient=float(
                candidate["resolved_coefficients"]["tendency_mse"]
            ),
            reference=reference,
            reconstruction=reconstruction,
            memory_before=memory_before,
            started=started,
        )
    del full_proposed, full_evaluated
    gc.collect()

    projections = []
    projection_names = (
        tuple(SUCCESSOR_DECODER_LEAVES)
        if decoder_leaf
        else tuple(SUCCESSOR_PARAMETER_BLOCKS)
    )
    for block in projection_names:
        projection_started = time.perf_counter()
        projected = (
            exclude_successor_decoder_leaf(stopped["direction"], block)
            if decoder_leaf
            else exclude_successor_parameter_block(stopped["direction"], block)
        )
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
        "schema_version": (
            "successor_decoder_leaf_attribution_report_v1"
            if decoder_leaf
            else "successor_parameter_geometry_attribution_report_v1"
        ),
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": subprocess_git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "predecessor_report_sha256": _sha256_file(args.predecessor_report),
            "cumulative_report_sha256": _sha256_file(args.cumulative_report),
            "parameter_geometry_report_sha256": (
                _sha256_file(args.parameter_geometry_report)
                if decoder_leaf
                else None
            ),
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "reconstruction": reconstruction,
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
    parser.add_argument("--parameter-geometry-report", type=Path)
    parser.add_argument("--decoder-leaf-report", type=Path)
    parser.add_argument("--output-locality-report", type=Path)
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
    parser.add_argument("--decoder-leaf-attribution", action="store_true")
    parser.add_argument("--output-locality", action="store_true")
    parser.add_argument("--maintenance-projection", action="store_true")
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
