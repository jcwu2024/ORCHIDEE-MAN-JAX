"""Audit one accumulated-gradient direction across ten mixed batches."""

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

AUDIT_GROUPS = 10


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_gradient_accumulator(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def accumulate(current: Any, gradient: Any, scale: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda total, value: total + scale * jnp.asarray(value),
            current,
            gradient,
        )

    return accumulate


def _make_nonfinite_counter(jax: Any) -> Any:
    import jax.numpy as jnp

    @jax.jit
    def count(values: Any) -> Any:
        return sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int64)
                for leaf in jax.tree_util.tree_leaves(values)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )

    return count


def _empty_accumulators(jax: Any, parameters: Any) -> list[Any]:
    import jax.numpy as jnp

    return [
        jax.tree_util.tree_map(lambda value: jnp.zeros_like(value), parameters)
        for _label in GRADIENT_FEASIBILITY_LABELS
    ]


def _load_group(
    jax: Any,
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    prepared_manifest_path: Path,
    signature: Mapping[str, Any],
    group_index: int,
) -> tuple[Any, Any, Any, list[tuple[Mapping[str, Any], Any]]]:
    offset = group_index * POINTS_PER_MIXED_UPDATE
    group = []
    for point_index in range(POINTS_PER_MIXED_UPDATE):
        record = manifest["entries"][offset + point_index]
        entry = schedule["entries"][int(record["update"]) - 1]
        artifact = load_prepared_artifact(
            prepared_manifest_path,
            record,
            schedule_entry=entry,
        )
        if artifact.trace_signature != signature:
            raise ValueError("multibatch direction retained-tail trace drift")
        group.append((entry, artifact))
    point_years = {
        (str(entry["landpoint_id"]), int(entry["year"]))
        for entry, _artifact in group
    }
    if len(point_years) != POINTS_PER_MIXED_UPDATE:
        raise ValueError("multibatch direction group repeated a point-year")
    contexts = tuple(
        _slice_context_batch(jax, artifact.batch, context_index=index)
        for index, (_entry, artifact) in enumerate(group)
    )
    return (
        _join_context_anchors(jax, contexts),
        _stack_contexts(jax, contexts),
        _stack_contexts(
            jax,
            tuple(artifact.day_inputs for _entry, artifact in group),
        ),
        group,
    )


def _accumulate_group_gradients(
    jax: Any,
    measurement: Any,
    accumulate: Any,
    count_nonfinite: Any,
    accumulators: list[Any],
    parameters: Any,
    anchor_batch: Any,
    context_batch: Any,
    context_inputs: Any,
) -> Mapping[str, Any]:
    before = None
    parent_hard = None
    nonfinite = {}
    scale = np.asarray(1.0 / AUDIT_GROUPS, dtype=np.float64)
    for component_index, seed in enumerate(
        np.eye(len(GRADIENT_FEASIBILITY_LABELS), dtype=np.float64)
    ):
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
            raise ValueError("multibatch direction seed changed objective outputs")
        gradient_nonfinite = count_nonfinite(gradient)
        jax.block_until_ready(gradient_nonfinite)
        label = GRADIENT_FEASIBILITY_LABELS[component_index]
        nonfinite[label] = int(np.asarray(gradient_nonfinite))
        accumulators[component_index] = accumulate(
            accumulators[component_index],
            gradient,
            scale,
        )
        jax.block_until_ready(accumulators[component_index])
        del losses, gradient, hard_counts
    if before is None or parent_hard is None:
        raise ValueError("multibatch direction produced no gradients")
    return {
        "losses_before": {
            label: float(before[index])
            for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
        },
        "parent_hard_gate_counts": parent_hard,
        "nonfinite_component_gradient_values": nonfinite,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    audit = _load_contract(
        args.audit_contract,
        schema_version="successor_multibatch_direction_audit_v1",
        status="frozen_zero_update_accumulated_gradient_grid",
    )
    smoke_contract = _load_contract(
        args.smoke_contract,
        schema_version="successor_direct_direction_smoke_v1",
        status="frozen_10_update_free_gpu_no_checkpoint",
    )
    candidate = _load_contract(
        args.candidate_coefficient_contract,
        schema_version="successor_named_biomass_candidate_coefficients_v1",
        status="accepted_zero_update_preflight_no_training_authorized",
    )
    if (
        smoke_contract["contract_sha256"]
        != audit["identity"]["direct_direction_smoke_contract_sha256"]
        or _sha256_file(args.smoke_report)
        != audit["identity"]["direct_direction_smoke_report_sha256"]
    ):
        raise ValueError("multibatch direction predecessor identity drift")

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
        raise ValueError("multibatch direction training identity drift")
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
    if len(manifest["entries"]) < AUDIT_GROUPS * POINTS_PER_MIXED_UPDATE:
        raise ValueError("multibatch direction prepared manifest is too short")
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
        raise ValueError("multibatch direction tendency-scale identity drift")

    identity = _training_identity(runtime, contract, schedule, assets, validation_identity)
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
        raise ValueError("multibatch direction initial trace drift")
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
        raise ValueError("multibatch direction Attention parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("multibatch direction requires update-1000 recovery")
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
        runtime,
        adapter_template,
        **objective_arguments,
    )
    accumulate = _make_gradient_accumulator(jax)
    count_nonfinite = _make_nonfinite_counter(jax)
    accumulators = _empty_accumulators(jax, parameters)
    memory_before = _device_memory_report(runtime["devices"])

    records = []
    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        record = _accumulate_group_gradients(
            jax,
            measurement,
            accumulate,
            count_nonfinite,
            accumulators,
            parameters,
            anchor_batch,
            context_batch,
            context_inputs,
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
                "steps": {},
            }
        )
        del anchor_batch, context_batch, context_inputs, group
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "multibatch_direction_gradient_group",
                    "group": group_index + 1,
                    "total": AUDIT_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    gram_measurement = _make_gradient_gram(jax)
    gram, aggregate_nonfinite = gram_measurement(tuple(accumulators))
    jax.block_until_ready((gram, aggregate_nonfinite))
    diagnostics = gradient_direction_diagnostics(np.asarray(gram, dtype=np.float64))
    selected = diagnostics["directions"]["minimum_norm_unit"]
    weights = np.asarray(
        [selected["weights"][label] for label in GRADIENT_FEASIBILITY_LABELS],
        dtype=np.float64,
    )
    combine = _make_combined_direction(jax)
    direction, direction_norm, direction_nonfinite = combine(
        tuple(accumulators),
        weights,
    )
    jax.block_until_ready((direction, direction_norm, direction_nonfinite))
    direct_step = _make_direct_virtual_step(jax)
    step_norms = tuple(float(value) for value in audit["step_norm_grid"])
    proposals = {}
    proposal_nonfinite = {}
    for step_norm in step_norms:
        proposed, nonfinite = direct_step(
            parameters,
            direction,
            direction_norm,
            np.asarray(step_norm, dtype=np.float64),
        )
        jax.block_until_ready((proposed, nonfinite))
        key = format(step_norm, ".12g")
        proposals[key] = proposed
        proposal_nonfinite[key] = int(np.asarray(nonfinite))

    for group_index in range(AUDIT_GROUPS):
        anchor_batch, context_batch, context_inputs, group = _load_group(
            jax,
            manifest,
            schedule,
            args.prepared_manifest,
            signature,
            group_index,
        )
        before = np.asarray(
            [
                records[group_index]["losses_before"][label]
                for label in GRADIENT_FEASIBILITY_LABELS
            ],
            dtype=np.float64,
        )
        for step_norm in step_norms:
            key = format(step_norm, ".12g")
            losses_after, hard_counts = loss_evaluator(
                proposals[key],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((losses_after, hard_counts))
            after = np.asarray(losses_after, dtype=np.float64)
            records[group_index]["steps"][key] = {
                "step_norm": step_norm,
                "virtual_losses_after": {
                    label: float(after[index])
                    for index, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
                },
                "virtual_loss_relative_changes": _relative_changes(before, after),
                "hard_gate_counts": _hard_counts_host(hard_counts),
                "nonfinite_proposed_parameter_values": proposal_nonfinite[key],
            }
            del losses_after, hard_counts
        del anchor_batch, context_batch, context_inputs, group
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "multibatch_direction_virtual_group",
                    "group": group_index + 1,
                    "total": AUDIT_GROUPS,
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
    component_nonfinite_count = sum(
        sum(record["nonfinite_component_gradient_values"].values())
        for record in records
    )
    nonfinite_count = (
        component_nonfinite_count
        + int(np.sum(np.asarray(aggregate_nonfinite, dtype=np.int64)))
        + int(np.asarray(direction_nonfinite))
        + sum(proposal_nonfinite.values())
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
    classification = dict(classification)
    classification["next_action"] = (
        audit["pass_action"] if classification["passed"] else audit["fail_action"]
    )
    evidence_valid = (
        hard_failure_count == int(audit["gates"]["hard_failure_count"])
        and nonfinite_count == int(audit["gates"]["nonfinite_value_count"])
        and all_losses_finite
    )
    report = {
        "schema_version": "successor_multibatch_direction_audit_report_v1",
        "status": "completed" if evidence_valid else "invalid_evidence",
        "git_head": _git_head(),
        "identity": {
            "audit_contract_sha256": audit["contract_sha256"],
            "direct_direction_smoke_report_sha256": _sha256_file(args.smoke_report),
            "staged_training_contract_sha256": contract.sha256,
        },
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": counts[SOURCE_ROUTED_ATTENTION],
        "parent_update": PARENT_UPDATE,
        "gradient_aggregation": audit["direction"],
        "step_norm_grid": list(step_norms),
        "aggregate_direction": {
            "gradient_norms": diagnostics["gradient_norms"],
            "direction_norm": float(np.asarray(direction_norm)),
            "minimum_component_cosine": min(
                selected["normalized_directional_derivatives"].values()
            ),
            "component_cosines": selected["normalized_directional_derivatives"],
            "normalized_simplex_weights": selected["normalized_simplex_weights"],
            "raw_gradient_coefficients": selected["weights"],
        },
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
            else "stop_and_repair_multibatch_direction_evidence"
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
    parser.add_argument("--smoke-contract", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
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
