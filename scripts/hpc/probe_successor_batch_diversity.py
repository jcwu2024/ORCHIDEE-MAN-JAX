"""Compare equal-sample one- and four-point-year gradients at update 1000."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import (
    _device_memory_report,
    _hard_counts_host,
    _head,
    _make_arm_objectives,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.run_successor_staged_training import _training_identity
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact, sha256_file
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)
from scripts.hpc.successor_staged_validation_prepared import (
    load_staged_validation_prepared,
)

POINTS_PER_MIXED_UPDATE = 4
ANCHORS_PER_CONTEXT = 8
ROLLOUTS_PER_CONTEXT = 1
DIAGNOSTIC_GROUPS = 4
EQUIVALENCE_RELATIVE_TOLERANCE = 1.0e-8


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _slice_context_batch(
    jax: Any,
    batch: Any,
    *,
    context_index: int,
) -> Any:
    """Select one disjoint quarter of a canonical 32-anchor/four-window batch."""

    index = int(context_index)
    if not 0 <= index < POINTS_PER_MIXED_UPDATE:
        raise ValueError("successor diversity context index is outside [0, 4)")
    anchor_start = index * ANCHORS_PER_CONTEXT
    anchor_stop = anchor_start + ANCHORS_PER_CONTEXT
    rollout = slice(index, index + ROLLOUTS_PER_CONTEXT)

    def anchors(tree: Any) -> Any:
        return jax.tree_util.tree_map(lambda value: value[anchor_start:anchor_stop], tree)

    def rollouts(tree: Any) -> Any:
        return jax.tree_util.tree_map(lambda value: value[rollout], tree)

    return batch._replace(
        anchor_inputs=anchors(batch.anchor_inputs),
        anchor_targets=anchors(batch.anchor_targets),
        rollout_inputs=rollouts(batch.rollout_inputs),
        rollout_targets=rollouts(batch.rollout_targets),
        expected_next_state=rollouts(batch.expected_next_state),
        expected_next_state_defined=rollouts(batch.expected_next_state_defined),
        expected_fast_gpp=rollouts(batch.expected_fast_gpp),
        expected_fast_gpp_defined=rollouts(batch.expected_fast_gpp_defined),
        expected_fast_day_target=rollouts(batch.expected_fast_day_target),
        expected_next_discrete_state=rollouts(batch.expected_next_discrete_state),
    )


def _join_context_anchors(jax: Any, batches: Sequence[Any]) -> Any:
    if len(batches) != POINTS_PER_MIXED_UPDATE:
        raise ValueError("successor diversity update requires four context batches")

    def concatenate(*values: Any) -> Any:
        return np.concatenate(tuple(np.asarray(value) for value in values), axis=0)

    first = batches[0]
    return first._replace(
        anchor_inputs=jax.tree_util.tree_map(
            concatenate,
            *(batch.anchor_inputs for batch in batches),
        ),
        anchor_targets=jax.tree_util.tree_map(
            concatenate,
            *(batch.anchor_targets for batch in batches),
        ),
    )


def _stack_contexts(jax: Any, values: Sequence[Any]) -> Any:
    if len(values) != POINTS_PER_MIXED_UPDATE:
        raise ValueError("successor diversity update requires four contexts")
    return jax.tree_util.tree_map(
        lambda *leaves: jax.numpy.stack(leaves),
        *values,
    )


def _flatten_gradient(jax: Any, gradient: Any) -> np.ndarray:
    leaves = tuple(
        np.ravel(np.asarray(value, dtype=np.float64))
        for value in jax.device_get(jax.tree_util.tree_leaves(gradient))
    )
    if not leaves:
        raise ValueError("successor diversity gradient tree is empty")
    result = np.concatenate(leaves)
    if not np.all(np.isfinite(result)):
        raise ValueError("successor diversity gradient contains nonfinite values")
    return result


def _gradient_summary(vectors: Sequence[np.ndarray], *, clip_norm: float) -> Mapping[str, Any]:
    if len(vectors) < 2:
        raise ValueError("successor diversity summary requires at least two gradients")
    matrix = np.stack(tuple(np.asarray(value, dtype=np.float64) for value in vectors))
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("successor diversity gradient norms must be positive and finite")
    normalized = matrix / norms[:, None]
    cosines = normalized @ normalized.T
    upper = cosines[np.triu_indices(len(vectors), k=1)]
    mean = np.mean(matrix, axis=0)
    mean_norm = float(np.linalg.norm(mean))
    mean_cosines = normalized @ (mean / max(mean_norm, 1.0e-300))
    return {
        "count": len(vectors),
        "norms": [float(value) for value in norms],
        "norm_mean": float(np.mean(norms)),
        "norm_coefficient_of_variation": float(np.std(norms) / np.mean(norms)),
        "clip_count": int(np.count_nonzero(norms > float(clip_norm))),
        "pairwise_cosine_minimum": float(np.min(upper)),
        "pairwise_cosine_median": float(np.median(upper)),
        "pairwise_cosine_maximum": float(np.max(upper)),
        "pairwise_negative_fraction": float(np.mean(upper < 0.0)),
        "mean_gradient_norm": mean_norm,
        "cancellation_ratio": float(mean_norm / np.mean(norms)),
        "cosine_to_mean_minimum": float(np.min(mean_cosines)),
        "cosine_to_mean_median": float(np.median(mean_cosines)),
    }


def _compiled_memory_report(compiled: Any) -> Mapping[str, Any]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {"available": False}
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
        "host_generated_code_size_in_bytes",
    )
    result = {
        name: int(getattr(stats, name))
        for name in names
        if getattr(stats, name, None) is not None
    }
    result["available"] = True
    result["estimated_device_resident_bytes"] = max(
        0,
        result.get("argument_size_in_bytes", 0)
        + result.get("output_size_in_bytes", 0)
        - result.get("alias_size_in_bytes", 0)
        + result.get("temp_size_in_bytes", 0),
    )
    return result


def _make_measurement(jax: Any, objective: Any) -> Any:
    import jax.numpy as jnp

    def measurement(parameters: Any, *arguments: Any) -> Any:
        (loss, auxiliary), gradient = jax.value_and_grad(
            lambda values: objective(values, *arguments),
            has_aux=True,
        )(parameters)
        squared = sum(
            (
                jnp.sum(jnp.asarray(value, dtype=jnp.float64) ** 2)
                for value in jax.tree_util.tree_leaves(gradient)
            ),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(value)))
                for value in jax.tree_util.tree_leaves(gradient)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        anchor, components, hard_counts = auxiliary
        return loss, anchor, components, hard_counts, gradient, jnp.sqrt(squared), nonfinite

    return jax.jit(measurement)


def _make_full_objective(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    coefficients: Mapping[str, float],
    long_scales: Any,
    contract: Any,
) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
    )

    anchor_objective, rollout_objective = _make_arm_objectives(
        runtime,
        arm,
        adapter_template,
        long_scales,
    )
    coefficient_vector = jnp.asarray(
        tuple(coefficients[name] for name in LONG_HORIZON_COMPONENTS),
        dtype=jnp.float64,
    )

    def objective(parameters: Any, batch: Any, day_inputs: Any) -> Any:
        anchor = anchor_objective(parameters, batch)
        components, trace = rollout_objective(
            parameters,
            batch,
            day_inputs,
            rematerialize=True,
        )
        hard_counts = successor_rollout_hard_gate_counts(
            batch,
            trace,
            markov_contract=runtime["contract"],
            budget_atol=float(contract.payload["gates"]["budget_absolute_tolerance"]),
            stock_atol=float(contract.payload["gates"]["stock_negative_absolute_tolerance"]),
        )
        return anchor + jnp.sum(coefficient_vector * components), (
            anchor,
            components,
            hard_counts,
        )

    return objective


def _make_mixed_objective(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    coefficients: Mapping[str, float],
    long_scales: Any,
    contract: Any,
) -> Any:
    import jax
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        _batched_rollout_objective_components,
        successor_rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        extract_biomass_productivity_trajectories,
        rollout_daily_operator,
        trajectories_from_rollout,
    )

    anchor_objective, _ = _make_arm_objectives(
        runtime,
        arm,
        adapter_template,
        long_scales,
    )
    process_head = _head(runtime, arm)
    coefficient_vector = jnp.asarray(
        tuple(coefficients[name] for name in LONG_HORIZON_COMPONENTS),
        dtype=jnp.float64,
    )

    def context_trajectory(parameters: Any, batch: Any, day_inputs: Any) -> Any:
        retained_tail = replace(adapter_template, day_inputs=day_inputs)
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=process_head,
            retained_tail=retained_tail,
        )
        templates = jax.tree_util.tree_map(lambda value: value[0], batch.rollout_inputs)
        trace = rollout_daily_operator(
            parameters,
            templates,
            transition=transition,
            contract=runtime["contract"],
            rematerialize=True,
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
            budget_atol=float(contract.payload["gates"]["budget_absolute_tolerance"]),
            stock_atol=float(contract.payload["gates"]["stock_negative_absolute_tolerance"]),
        )
        return predicted, expected, hard_counts

    def objective(
        parameters: Any,
        anchor_batch: Any,
        context_batches: Any,
        day_inputs: Any,
    ) -> Any:
        anchor = anchor_objective(parameters, anchor_batch)
        predicted, expected, context_hard_counts = jax.vmap(
            context_trajectory,
            in_axes=(None, 0, 0),
        )(
            parameters,
            context_batches,
            day_inputs,
        )
        hard_counts = jax.tree_util.tree_map(
            lambda values: jnp.sum(values, axis=0),
            context_hard_counts,
        )
        components = _batched_rollout_objective_components(
            predicted,
            expected,
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            horizons=(1, 7, 30),
        )
        return anchor + jnp.sum(coefficient_vector * components), (
            anchor,
            components,
            hard_counts,
        )

    return objective


def _measurement_record(jax: Any, result: Any) -> tuple[Mapping[str, Any], np.ndarray]:
    loss, anchor, components, hard_counts, gradient, norm, nonfinite = result
    vector = _flatten_gradient(jax, gradient)
    record = {
        "loss": float(np.asarray(loss)),
        "one_day_anchor": float(np.asarray(anchor)),
        "long_horizon_components": [float(value) for value in np.asarray(components)],
        "gradient_norm": float(np.asarray(norm)),
        "nonfinite_gradient_values": int(np.asarray(nonfinite)),
        "hard_gate_counts": _hard_counts_host(hard_counts),
    }
    return record, vector


def _selected_positions(count: int) -> tuple[int, ...]:
    wanted = POINTS_PER_MIXED_UPDATE * DIAGNOSTIC_GROUPS
    if count < wanted:
        raise ValueError("successor diversity manifest has too few artifacts")
    return tuple(int(value) for value in np.linspace(0, count - 1, wanted, dtype=np.int64))


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
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
        load_successor_training_assets,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if contract.payload["schema_version"] != STAGED_SCHEMA_VERSION:
        raise ValueError("successor diversity probe requires staged-v3")
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
        raise ValueError("successor diversity validation panel drift")
    identity = _training_identity(
        runtime,
        contract,
        schedule,
        assets,
        validation_prepared,
    )

    positions = _selected_positions(len(manifest["entries"]))
    selected = []
    for position in positions:
        record = manifest["entries"][position]
        update = int(record["update"])
        entry = schedule["entries"][update - 1]
        artifact = load_prepared_artifact(
            args.prepared_manifest,
            record,
            schedule_entry=entry,
        )
        selected.append((entry, artifact))
    point_years = {
        (str(entry["landpoint_id"]), int(entry["year"])) for entry, _artifact in selected
    }
    if len(point_years) != len(selected):
        raise ValueError("successor diversity selection repeated a point-year")
    first_entry, first_artifact = selected[0]
    _reference, adapter_template, adapter_timing = _prepare_adapter(runtime, first_entry)
    signature = _retained_tail_trace_signature(adapter_template)
    if any(artifact.trace_signature != signature for _entry, artifact in selected):
        raise ValueError("successor diversity retained-tail trace drift")

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
    if set(counts.values()) != {EXPECTED_PARAMETER_COUNT}:
        raise ValueError("successor diversity parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=1000,
        maximum_completed_updates=1000,
    )
    if recovery is None or recovery["validation_state"]["completed_milestones"] != [500, 1000]:
        raise ValueError("successor diversity requires the accepted update-1000 recovery")
    parameters = dict(recovery["parameters"])
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    clip_norm = float(contract.optimizer["global_gradient_norm_clip"])
    device_memory_before = _device_memory_report(runtime["devices"])
    arm_reports = {}

    for arm in arms:
        full = _make_measurement(
            jax,
            _make_full_objective(
                runtime,
                arm,
                adapter_template,
                assets.coefficients,
                long_scales,
                contract,
            ),
        )
        mixed = _make_measurement(
            jax,
            _make_mixed_objective(
                runtime,
                arm,
                adapter_template,
                assets.coefficients,
                long_scales,
                contract,
            ),
        )

        first_batch = first_artifact.batch
        partitioned = tuple(
            _slice_context_batch(jax, first_batch, context_index=index)
            for index in range(POINTS_PER_MIXED_UPDATE)
        )
        partitioned_anchor = _join_context_anchors(jax, partitioned)
        partitioned_batch = _stack_contexts(jax, partitioned)
        repeated_inputs = _stack_contexts(
            jax,
            tuple(first_artifact.day_inputs for _ in partitioned),
        )
        compiled_full = full.lower(
            parameters[arm],
            first_batch,
            first_artifact.day_inputs,
        ).compile()
        compiled_mixed = mixed.lower(
            parameters[arm],
            partitioned_anchor,
            partitioned_batch,
            repeated_inputs,
        ).compile()

        full_equivalence_result = compiled_full(
            parameters[arm],
            first_batch,
            first_artifact.day_inputs,
        )
        mixed_equivalence_result = compiled_mixed(
            parameters[arm],
            partitioned_anchor,
            partitioned_batch,
            repeated_inputs,
        )
        jax.block_until_ready((full_equivalence_result, mixed_equivalence_result))
        full_equivalence, full_vector = _measurement_record(jax, full_equivalence_result)
        mixed_equivalence, mixed_vector = _measurement_record(jax, mixed_equivalence_result)
        relative_gradient_difference = float(
            np.linalg.norm(full_vector - mixed_vector)
            / max(np.linalg.norm(full_vector), 1.0e-300)
        )
        relative_loss_difference = float(
            abs(full_equivalence["loss"] - mixed_equivalence["loss"])
            / max(abs(full_equivalence["loss"]), 1.0e-300)
        )
        component_difference = float(
            np.max(
                np.abs(
                    np.asarray(full_equivalence["long_horizon_components"])
                    - np.asarray(mixed_equivalence["long_horizon_components"])
                )
            )
        )
        equivalence_passed = (
            relative_gradient_difference <= EQUIVALENCE_RELATIVE_TOLERANCE
            and relative_loss_difference <= EQUIVALENCE_RELATIVE_TOLERANCE
            and component_difference <= EQUIVALENCE_RELATIVE_TOLERANCE
            and full_equivalence["hard_gate_counts"] == mixed_equivalence["hard_gate_counts"]
        )
        if not equivalence_passed:
            raise ValueError(f"successor diversity mixed objective drift for {arm}")

        single_vectors = []
        mixed_vectors = []
        groups = []
        for group_index in range(DIAGNOSTIC_GROUPS):
            group = selected[
                group_index * POINTS_PER_MIXED_UPDATE :
                (group_index + 1) * POINTS_PER_MIXED_UPDATE
            ]
            designated = group[group_index % POINTS_PER_MIXED_UPDATE]
            single_result = compiled_full(
                parameters[arm],
                designated[1].batch,
                designated[1].day_inputs,
            )
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
            mixed_result = compiled_mixed(
                parameters[arm],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((single_result, mixed_result))
            single_record, single_vector = _measurement_record(jax, single_result)
            mixed_record, current_mixed_vector = _measurement_record(jax, mixed_result)
            if (
                single_record["nonfinite_gradient_values"] != 0
                or mixed_record["nonfinite_gradient_values"] != 0
                or sum(single_record["hard_gate_counts"].values()) != 0
                or sum(mixed_record["hard_gate_counts"].values()) != 0
            ):
                raise ValueError(f"successor diversity hard failure for {arm}/group {group_index}")
            single_vectors.append(single_vector)
            mixed_vectors.append(current_mixed_vector)
            groups.append(
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
                    "single_point_year_update": int(designated[0]["update"]),
                    "single": single_record,
                    "mixed": mixed_record,
                    "single_to_mixed_gradient_cosine": float(
                        np.dot(single_vector, current_mixed_vector)
                        / max(
                            np.linalg.norm(single_vector)
                            * np.linalg.norm(current_mixed_vector),
                            1.0e-300,
                        )
                    ),
                }
            )
            del single_result, mixed_result, single_vector, current_mixed_vector
            gc.collect()
            print(
                json.dumps(
                    {
                        "event": "successor_batch_diversity_group",
                        "arm": arm,
                        "group": group_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        arm_reports[arm] = {
            "equivalence": {
                "passed": equivalence_passed,
                "relative_loss_difference": relative_loss_difference,
                "maximum_component_absolute_difference": component_difference,
                "relative_gradient_l2_difference": relative_gradient_difference,
            },
            "compiled_memory": {
                "single_point_year": _compiled_memory_report(compiled_full),
                "four_point_year": _compiled_memory_report(compiled_mixed),
            },
            "single_point_year_gradients": _gradient_summary(
                single_vectors,
                clip_norm=clip_norm,
            ),
            "four_point_year_gradients": _gradient_summary(
                mixed_vectors,
                clip_norm=clip_norm,
            ),
            "groups": groups,
        }
        del compiled_full, compiled_mixed, full, mixed, single_vectors, mixed_vectors
        gc.collect()

    report = {
        "schema_version": "successor_batch_diversity_gradient_probe_v1",
        "status": "passed",
        "git_head": _git_head(),
        "staged_training_contract_sha256": contract.sha256,
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "recovery_sha256": sha256_file(args.recovery_path),
        "recovery_completed_updates": int(recovery["completed_updates"]),
        "parameter_count_per_arm": counts,
        "batch_comparison": {
            "single_point_year": {"point_years": 1, "one_day_anchors": 32, "rollout_windows": 4},
            "four_point_year": {"point_years": 4, "one_day_anchors": 32, "rollout_windows": 4},
            "rollout_days": 30,
            "joint_bias_across_four_windows_preserved": True,
            "same_loss_coefficients": True,
        },
        "selected_manifest_positions": list(positions),
        "arms": arm_reports,
        "adapter_preparation": adapter_timing,
        "device_memory_before": device_memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "training_updates": 0,
        "checkpoint_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "next_gate": "review_then_run_short_free_empirical_training",
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
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
