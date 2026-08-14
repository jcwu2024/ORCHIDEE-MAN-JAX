"""Audit shared direct-direction trust-region step norms without training."""

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
    classify_direct_trust_region,
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
    _make_loss_evaluator,
    _make_measurement,
    _sha256_file,
)
from scripts.hpc.probe_successor_realized_step import _relative_changes
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


def _make_combined_direction(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def combine(gradients: tuple[Any, ...], weights: Any) -> Any:
        direction = jax.tree_util.tree_map(
            lambda *leaves: sum(
                (
                    weights[index] * jnp.asarray(leaf)
                    for index, leaf in enumerate(leaves)
                ),
                jnp.asarray(0.0, dtype=jnp.float64),
            ),
            *gradients,
        )
        squared = sum(
            (
                jnp.sum(jnp.asarray(leaf, dtype=jnp.float64) ** 2)
                for leaf in jax.tree_util.tree_leaves(direction)
            ),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int64)
                for leaf in jax.tree_util.tree_leaves(direction)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        return direction, jnp.sqrt(squared), nonfinite

    return combine


def _make_direct_virtual_step(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def step(parameters: Any, direction: Any, direction_norm: Any, step_norm: Any) -> Any:
        proposed = jax.tree_util.tree_map(
            lambda value, gradient: value
            - step_norm * gradient / jnp.maximum(direction_norm, 1.0e-30),
            parameters,
            direction,
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int64)
                for leaf in jax.tree_util.tree_leaves(proposed)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        return proposed, nonfinite

    return step


def _group_record(
    jax: Any,
    measurement: Any,
    loss_evaluator: Any,
    gram_measurement: Any,
    combine: Any,
    virtual_step: Any,
    parameters: Any,
    anchor_batch: Any,
    context_batch: Any,
    context_inputs: Any,
    step_norms: tuple[float, ...],
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
            raise ValueError("direct trust-region seed changed objective outputs")
        gradients.append(gradient)
    if before is None or parent_hard is None:
        raise ValueError("direct trust-region produced no gradients")

    gram, component_nonfinite = gram_measurement(tuple(gradients))
    jax.block_until_ready((gram, component_nonfinite))
    diagnostics = gradient_direction_diagnostics(np.asarray(gram, dtype=np.float64))
    selected = diagnostics["directions"]["minimum_norm_unit"]
    weights = np.asarray(
        [selected["weights"][label] for label in GRADIENT_FEASIBILITY_LABELS],
        dtype=np.float64,
    )
    direction, direction_norm, direction_nonfinite = combine(tuple(gradients), weights)
    jax.block_until_ready((direction, direction_norm, direction_nonfinite))

    steps = {}
    proposed_nonfinite_count = 0
    for step_norm in step_norms:
        proposed, proposed_nonfinite = virtual_step(
            parameters,
            direction,
            direction_norm,
            np.asarray(step_norm, dtype=np.float64),
        )
        jax.block_until_ready((proposed, proposed_nonfinite))
        losses_after, hard_counts = loss_evaluator(
            proposed,
            anchor_batch,
            context_batch,
            context_inputs,
        )
        jax.block_until_ready((losses_after, hard_counts))
        after = np.asarray(losses_after, dtype=np.float64)
        hard = _hard_counts_host(hard_counts)
        key = format(step_norm, ".12g")
        steps[key] = {
            "step_norm": step_norm,
            "virtual_losses_after": {
                label: float(after[index])
                for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
            },
            "virtual_loss_relative_changes": _relative_changes(before, after),
            "hard_gate_counts": hard,
            "nonfinite_proposed_parameter_values": int(
                np.asarray(proposed_nonfinite)
            ),
        }
        proposed_nonfinite_count += int(np.asarray(proposed_nonfinite))
        del proposed, proposed_nonfinite, losses_after, hard_counts

    record = {
        "losses_before": {
            label: float(before[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "parent_hard_gate_counts": parent_hard,
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
        "steps": steps,
        "nonfinite_values": {
            "component_gradients": int(
                np.sum(np.asarray(component_nonfinite, dtype=np.int64))
            ),
            "raw_direction": int(np.asarray(direction_nonfinite)),
            "proposed_parameters": proposed_nonfinite_count,
        },
    }
    del gradients, gram, component_nonfinite, direction, direction_nonfinite
    return record


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    audit = _load_contract(
        args.audit_contract,
        schema_version="successor_direct_trust_region_audit_v1",
        status="frozen_zero_update_direct_direction_grid",
    )
    candidate = _load_contract(
        args.candidate_coefficient_contract,
        schema_version="successor_named_biomass_candidate_coefficients_v1",
        status="accepted_zero_update_preflight_no_training_authorized",
    )
    if _sha256_file(args.realized_step_report) != audit["identity"][
        "realized_step_report_sha256"
    ]:
        raise ValueError("direct trust-region predecessor identity drift")

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
        raise ValueError("direct trust-region training identity drift")
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
        raise ValueError("direct trust-region tendency-scale identity drift")

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
        raise ValueError("direct trust-region selection size drift")
    first_entry, first_artifact = selected[0]
    _reference, adapter_template, adapter_timing = _prepare_adapter(runtime, first_entry)
    signature = _retained_tail_trace_signature(adapter_template)
    if any(artifact.trace_signature != signature for _entry, artifact in selected):
        raise ValueError("direct trust-region retained-tail trace drift")
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
        raise ValueError("direct trust-region Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("direct trust-region requires update-1000 recovery")
    parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
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
    loss_evaluator = _make_loss_evaluator(
        runtime,
        adapter_template,
        runner_contract=contract,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        flux_coefficients=assets.coefficients,
        tendency_coefficient=float(candidate["resolved_coefficients"]["tendency_mse"]),
    )
    gram_measurement = _make_gradient_gram(jax)
    combine = _make_combined_direction(jax)
    virtual_step = _make_direct_virtual_step(jax)
    step_norms = tuple(float(value) for value in audit["step_norm_grid"])

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
            loss_evaluator,
            gram_measurement,
            combine,
            virtual_step,
            parameters,
            anchor_batch,
            context_batch,
            context_inputs,
            step_norms,
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
                    "event": "successor_direct_trust_region_group",
                    "group": group_index + 1,
                    "total": DIAGNOSTIC_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    hard_failure_count = sum(
        sum(record["parent_hard_gate_counts"].values())
        + sum(
            sum(step["hard_gate_counts"].values())
            for step in record["steps"].values()
        )
        for record in records
    )
    nonfinite_count = sum(
        sum(record["nonfinite_values"].values()) for record in records
    )
    all_losses_finite = all(
        all(np.isfinite(value) for value in record["losses_before"].values())
        and all(
            all(np.isfinite(value) for value in step["virtual_losses_after"].values())
            for step in record["steps"].values()
        )
        for record in records
    )
    classification = classify_direct_trust_region(
        records,
        step_norms=step_norms,
        gates=audit["gates"],
    )
    evidence_valid = (
        hard_failure_count == int(audit["gates"]["hard_failure_count"])
        and nonfinite_count == int(audit["gates"]["nonfinite_value_count"])
        and all_losses_finite
    )
    report = {
        "schema_version": "successor_direct_trust_region_audit_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": _git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "realized_step_report_sha256": _sha256_file(args.realized_step_report),
            "staged_training_contract_sha256": contract.sha256,
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "step_norm_grid": list(step_norms),
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
        "checkpoint_written": False,
        "recovery_written": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "next_gate": (
            classification["next_action"]
            if evidence_valid
            else "stop_and_repair_direct_trust_region_evidence"
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
    parser.add_argument("--realized-step-report", type=Path, required=True)
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
