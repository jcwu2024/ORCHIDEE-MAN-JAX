"""Audit how AdamW realizes the accepted raw common-descent direction."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    GRADIENT_FEASIBILITY_LABELS,
    classify_realized_step_audit,
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from scripts.hpc.probe_successor_batch_diversity import (
    DIAGNOSTIC_GROUPS,
    POINTS_PER_MIXED_UPDATE,
    _join_context_anchors,
    _selected_positions,
    _slice_context_batch,
    _stack_contexts,
)
from scripts.hpc.probe_successor_gradient_feasibility import (
    PARENT_UPDATE,
    _load_contract,
    _make_measurement,
    _sha256_file,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_gradient_balanced_smoke import _make_gradient_gram
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


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_proposed_directions(jax: Any, contract: Any) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_adamw_update,
    )

    @jax.jit
    def propose(
        parameters: Any,
        inherited_optimizer: Any,
        reset_optimizer: Any,
        gradients: tuple[Any, ...],
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
        inherited_parameters, _inherited_state, raw_norm = successor_adamw_update(
            parameters,
            combined,
            inherited_optimizer,
            learning_rate,
            contract,
        )
        reset_parameters, _reset_state, _reset_norm = successor_adamw_update(
            parameters,
            combined,
            reset_optimizer,
            learning_rate,
            contract,
        )
        inherited_direction = jax.tree_util.tree_map(
            lambda current, proposed: current - proposed,
            parameters,
            inherited_parameters,
        )
        reset_direction = jax.tree_util.tree_map(
            lambda current, proposed: current - proposed,
            parameters,
            reset_parameters,
        )
        return (
            inherited_parameters,
            reset_parameters,
            combined,
            inherited_direction,
            reset_direction,
            raw_norm,
        )

    return propose


def _direction_record(
    gram: np.ndarray,
    direction_index: int,
) -> Mapping[str, Any]:
    component_count = len(GRADIENT_FEASIBILITY_LABELS)
    component_norms = np.sqrt(np.maximum(np.diag(gram)[:component_count], 0.0))
    direction_norm = float(np.sqrt(max(float(gram[direction_index, direction_index]), 0.0)))
    cosines = np.divide(
        gram[:component_count, direction_index],
        component_norms * direction_norm,
        out=np.zeros(component_count, dtype=np.float64),
        where=(component_norms > 0.0) & (direction_norm > 0.0),
    )
    return {
        "direction_norm": direction_norm,
        "component_cosines": {
            label: float(cosines[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "minimum_component_cosine": float(np.min(cosines)),
    }


def _relative_changes(before: np.ndarray, after: np.ndarray) -> Mapping[str, float]:
    values = np.divide(
        after,
        before,
        out=np.full_like(after, np.inf, dtype=np.float64),
        where=before != 0.0,
    ) - 1.0
    values = np.where((before == 0.0) & (after == 0.0), 0.0, values)
    return {
        label: float(values[index])
        for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
    }


def _loss_record(
    jax: Any,
    measurement: Any,
    parameters: Any,
    anchor_batch: Any,
    context_batch: Any,
    context_inputs: Any,
) -> tuple[np.ndarray, Mapping[str, int]]:
    losses, zero_gradient, hard_counts = measurement(
        parameters,
        np.zeros(len(GRADIENT_FEASIBILITY_LABELS), dtype=np.float64),
        anchor_batch,
        context_batch,
        context_inputs,
    )
    jax.block_until_ready((losses, zero_gradient, hard_counts))
    result = np.asarray(losses, dtype=np.float64)
    hard = _hard_counts_host(hard_counts)
    del losses, zero_gradient, hard_counts
    return result, hard


def _group_record(
    jax: Any,
    measurement: Any,
    gram_measurement: Any,
    propose: Any,
    parameters: Any,
    inherited_optimizer: Any,
    reset_optimizer: Any,
    anchor_batch: Any,
    context_batch: Any,
    context_inputs: Any,
    learning_rate: Any,
) -> Mapping[str, Any]:
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
            raise ValueError("realized-step seed changed objective outputs")
        gradients.append(gradient)

    component_gram, component_nonfinite = gram_measurement(tuple(gradients))
    jax.block_until_ready((component_gram, component_nonfinite))
    component_gram_host = np.asarray(component_gram, dtype=np.float64)
    diagnostics = gradient_direction_diagnostics(component_gram_host)
    selected = diagnostics["directions"]["minimum_norm_unit"]
    weights = np.asarray(
        [selected["weights"][label] for label in GRADIENT_FEASIBILITY_LABELS],
        dtype=np.float64,
    )
    proposed = propose(
        parameters,
        inherited_optimizer,
        reset_optimizer,
        tuple(gradients),
        weights,
        learning_rate,
    )
    jax.block_until_ready(proposed)
    (
        inherited_parameters,
        reset_parameters,
        combined,
        inherited_direction,
        reset_direction,
        raw_norm,
    ) = proposed
    augmented_gram, augmented_nonfinite = gram_measurement(
        tuple(gradients)
        + (combined, inherited_direction, reset_direction)
    )
    jax.block_until_ready((augmented_gram, augmented_nonfinite))
    augmented_host = np.asarray(augmented_gram, dtype=np.float64)
    inherited_after, inherited_hard = _loss_record(
        jax,
        measurement,
        inherited_parameters,
        anchor_batch,
        context_batch,
        context_inputs,
    )
    reset_after, reset_hard = _loss_record(
        jax,
        measurement,
        reset_parameters,
        anchor_batch,
        context_batch,
        context_inputs,
    )
    if before is None or parent_hard is None:
        raise ValueError("realized-step produced no component gradients")
    raw_direction = _direction_record(augmented_host, 4)
    inherited_record = _direction_record(augmented_host, 5)
    reset_record = _direction_record(augmented_host, 6)
    inherited_record |= {
        "virtual_losses_after": {
            label: float(inherited_after[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "virtual_loss_relative_changes": _relative_changes(before, inherited_after),
        "hard_gate_counts": inherited_hard,
    }
    reset_record |= {
        "virtual_losses_after": {
            label: float(reset_after[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "virtual_loss_relative_changes": _relative_changes(before, reset_after),
        "hard_gate_counts": reset_hard,
    }
    nonfinite_host = np.asarray(augmented_nonfinite, dtype=np.int64)
    record = {
        "losses_before": {
            label: float(before[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "parent_hard_gate_counts": parent_hard,
        "gradient_norms": diagnostics["gradient_norms"],
        "raw_direction": raw_direction
        | {
            "normalized_simplex_weights": selected["normalized_simplex_weights"],
            "raw_gradient_coefficients": selected["weights"],
            "optimizer_reported_gradient_norm": float(np.asarray(raw_norm)),
        },
        "optimizers": {
            "inherited_adamw": inherited_record,
            "reset_adamw": reset_record,
        },
        "nonfinite_values": {
            "component_gradients": int(np.sum(nonfinite_host[:4])),
            "raw_direction": int(nonfinite_host[4]),
            "inherited_direction": int(nonfinite_host[5]),
            "reset_direction": int(nonfinite_host[6]),
        },
    }
    del (
        gradients,
        component_gram,
        component_nonfinite,
        proposed,
        inherited_parameters,
        reset_parameters,
        combined,
        inherited_direction,
        reset_direction,
        augmented_gram,
        augmented_nonfinite,
    )
    return record


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    audit = _load_contract(
        args.audit_contract,
        schema_version="successor_realized_step_audit_v1",
        status="frozen_zero_update_optimizer_handoff_audit",
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
        audit["identity"]["gradient_feasibility_acceptance_sha256"]
        != acceptance["contract_sha256"]
        or _sha256_file(args.rejected_smoke_report)
        != audit["identity"]["rejected_gradient_balanced_smoke_report_sha256"]
    ):
        raise ValueError("realized-step predecessor identity drift")

    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        STAGED_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        initialize_successor_adamw,
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
        or audit["identity"]["staged_training_contract_sha256"] != contract.sha256
        or audit["identity"]["candidate_coefficient_contract_sha256"]
        != candidate["contract_sha256"]
    ):
        raise ValueError("realized-step training identity drift")
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
    assets = load_successor_training_assets(
        args.coefficient_contract,
        args.validation_panel,
        coefficient_report_path=args.coefficient_report,
        prepared_manifest_path=args.coefficient_prepared_manifest,
        runner_contract=contract,
        components=LONG_HORIZON_COMPONENTS,
    )
    validation_identity = {
        "manifest_sha256": _sha256_file(args.validation_prepared_manifest)
    }
    scale_asset = load_named_biomass_tendency_scale_asset(
        args.tendency_scale_asset,
        args.tendency_scale_acceptance,
    )
    scale_acceptance = json.loads(
        args.tendency_scale_acceptance.read_text(encoding="utf-8")
    )
    if scale_acceptance["contract_sha256"] != audit["identity"][
        "scale_acceptance_contract_sha256"
    ]:
        raise ValueError("realized-step tendency-scale identity drift")

    identity = _training_identity(runtime, contract, schedule, assets, validation_identity)
    selected = []
    for position in _selected_positions(len(manifest["entries"])):
        record = manifest["entries"][position]
        entry = schedule["entries"][int(record["update"]) - 1]
        artifact = load_prepared_artifact(
            args.prepared_manifest,
            record,
            schedule_entry=entry,
        )
        selected.append((entry, artifact))
    if len(selected) != DIAGNOSTIC_GROUPS * POINTS_PER_MIXED_UPDATE:
        raise ValueError("realized-step selection size drift")
    first_entry, first_artifact = selected[0]
    _reference, adapter_template, adapter_timing = _prepare_adapter(runtime, first_entry)
    signature = _retained_tail_trace_signature(adapter_template)
    if any(artifact.trace_signature != signature for _entry, artifact in selected):
        raise ValueError("realized-step retained-tail trace drift")
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
        raise ValueError("realized-step Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("realized-step audit requires update-1000 recovery")
    parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
    inherited_optimizer = recovery["optimizers"][SOURCE_ROUTED_ATTENTION]
    reset_optimizer = initialize_successor_adamw(parameters)
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    measurement = _make_measurement(
        runtime,
        adapter_template,
        runner_contract=contract,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        flux_coefficients=assets.coefficients,
        tendency_coefficient=float(candidate["resolved_coefficients"]["tendency_mse"]),
    )
    gram_measurement = _make_gradient_gram(jax)
    propose = _make_proposed_directions(jax, contract)
    learning_rate = np.asarray(
        successor_learning_rate(contract, PARENT_UPDATE + 1),
        dtype=np.float64,
    )

    records = []
    memory_before = _device_memory_report(runtime["devices"])
    for group_index in range(DIAGNOSTIC_GROUPS):
        group = selected[
            group_index * POINTS_PER_MIXED_UPDATE :
            (group_index + 1) * POINTS_PER_MIXED_UPDATE
        ]
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
        record = _group_record(
            jax,
            measurement,
            gram_measurement,
            propose,
            parameters,
            inherited_optimizer,
            reset_optimizer,
            anchor_batch,
            context_batch,
            context_inputs,
            learning_rate,
        )
        records.append(
            {
                "group": group_index,
                "point_years": [
                    {
                        "update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                **record,
            }
        )
        del group, contexts, anchor_batch, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "successor_realized_step_group",
                    "group": group_index + 1,
                    "total": DIAGNOSTIC_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    hard_failure_count = sum(
        sum(record["parent_hard_gate_counts"].values())
        + sum(record["optimizers"]["inherited_adamw"]["hard_gate_counts"].values())
        + sum(record["optimizers"]["reset_adamw"]["hard_gate_counts"].values())
        for record in records
    )
    nonfinite_count = sum(
        sum(record["nonfinite_values"].values()) for record in records
    )
    all_losses_finite = all(
        all(
            np.isfinite(value)
            for value in (
                *record["losses_before"].values(),
                *record["optimizers"]["inherited_adamw"][
                    "virtual_losses_after"
                ].values(),
                *record["optimizers"]["reset_adamw"][
                    "virtual_losses_after"
                ].values(),
            )
        )
        for record in records
    )
    classification = classify_realized_step_audit(records, gates=audit["gates"])
    evidence_valid = (
        hard_failure_count == int(audit["gates"]["hard_failure_count"])
        and nonfinite_count == int(audit["gates"]["nonfinite_value_count"])
        and all_losses_finite
    )
    report = {
        "schema_version": "successor_realized_step_audit_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": _git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "gradient_feasibility_acceptance_sha256": acceptance["contract_sha256"],
            "rejected_gradient_balanced_smoke_report_sha256": _sha256_file(
                args.rejected_smoke_report
            ),
            "staged_training_contract_sha256": contract.sha256,
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "virtual_update": PARENT_UPDATE + 1,
        "learning_rate": float(learning_rate),
        "classification": classification,
        "evidence_gates": {
            "evidence_valid": evidence_valid,
            "all_losses_finite": all_losses_finite,
            "hard_failure_count": hard_failure_count,
            "nonfinite_value_count": nonfinite_count,
        },
        "records": records,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_updates": 0,
        "proposed_parameters_retained": False,
        "optimizer_state_retained": False,
        "checkpoint_written": False,
        "recovery_written": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "next_gate": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_realized_step_evidence"
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
    parser.add_argument("--audit-contract", type=Path, required=True)
    parser.add_argument("--feasibility-acceptance", type=Path, required=True)
    parser.add_argument("--rejected-smoke-report", type=Path, required=True)
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
