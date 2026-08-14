"""Audit per-output train/train gradients at the accepted update-1000 state."""

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
from scripts.hpc.probe_successor_batch_diversity import (
    DIAGNOSTIC_GROUPS,
    POINTS_PER_MIXED_UPDATE,
    _join_context_anchors,
    _selected_positions,
    _slice_context_batch,
    _stack_contexts,
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
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)
from scripts.hpc.successor_staged_validation_prepared import (
    load_staged_validation_prepared,
)

PARENT_UPDATE = 1000
HORIZONS = (1, 7, 30)
AUDITED_OUTPUTS = (
    "gpp",
    "npp",
    "maintenance_respiration",
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "total_live_biomass_carbon",
)
GRADIENT_LABELS = ("total", "one_day_anchor", *AUDITED_OUTPUTS)
CONFLICT_PAIRS = tuple(
    ("maintenance_respiration", name)
    for name in (
        "npp",
        "aboveground_biomass_carbon",
        "belowground_biomass_carbon",
        "total_live_biomass_carbon",
    )
)
CONFLICT_COSINE_THRESHOLD = -0.1
CONFLICT_GROUP_FRACTION = 0.75
MITIGATION_COSINE_DELTA = 0.1


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _field_contribution(
    jnp: Any,
    result: Any,
    name: str,
    coefficients: Mapping[str, float],
) -> Any:
    values = []
    for horizon in HORIZONS:
        current = result.by_horizon[horizon]
        if name in current.cumulative_flux:
            error = current.cumulative_flux[name]
            errors = current.cumulative_flux
            weighted = (
                float(coefficients["cumulative_flux_mse"]) * error.mse
                + float(coefficients["cumulative_flux_bias_squared"])
                * error.bias_squared
            )
        elif name in current.terminal_stock:
            error = current.terminal_stock[name]
            errors = current.terminal_stock
            weighted = (
                float(coefficients["terminal_stock_mse"]) * error.mse
                + float(coefficients["terminal_stock_bias_squared"])
                * error.bias_squared
            )
        else:
            raise ValueError(f"per-output gradient field is absent: {name}")
        active_count = jnp.sum(
            jnp.stack(tuple(value.count > 0.0 for value in errors.values())),
            dtype=jnp.float64,
        )
        values.append(
            jnp.where(
                (error.count > 0.0) & (active_count > 0.0),
                weighted / jnp.maximum(active_count, 1.0),
                0.0,
            )
        )
    return jnp.mean(jnp.stack(tuple(values)))


def _loss_vector(
    jax: Any,
    anchor: Any,
    predicted: Any,
    expected: Any,
    *,
    coefficients: Mapping[str, float],
    long_scales: Any,
) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
        long_horizon_biomass_productivity_objective,
    )
    actual = jax.tree_util.tree_map(lambda value: jnp.swapaxes(value, 0, 1), predicted)
    wanted = jax.tree_util.tree_map(lambda value: jnp.swapaxes(value, 0, 1), expected)
    long = long_horizon_biomass_productivity_objective(
        actual.fluxes,
        wanted.fluxes,
        actual.stocks,
        wanted.stocks,
        flux_scales=long_scales.fluxes,
        stock_scales=long_scales.stocks,
        coefficients=coefficients,
        horizons=HORIZONS,
    )
    selected = tuple(
        _field_contribution(
            jnp,
            long,
            name,
            coefficients,
        )
        for name in AUDITED_OUTPUTS
    )
    return jnp.stack((anchor + long.total, anchor, *selected))


def _make_full_objective(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    coefficients: Mapping[str, float],
    long_scales: Any,
    contract: Any,
) -> Any:
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
        rollout_daily_operator,
        trajectories_from_rollout,
    )

    jax = runtime["jax"]
    anchor_objective, _ = _make_arm_objectives(
        runtime,
        arm,
        adapter_template,
        long_scales,
    )
    process_head = _head(runtime, arm)

    def objective(parameters: Any, batch: Any, day_inputs: Any) -> Any:
        anchor = anchor_objective(parameters, batch)
        retained_tail = replace(adapter_template, day_inputs=day_inputs)
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=process_head,
            retained_tail=retained_tail,
        )

        def rollout_one(templates: Any) -> Any:
            return rollout_daily_operator(
                parameters,
                templates,
                transition=transition,
                contract=runtime["contract"],
                rematerialize=True,
            )

        trace = jax.vmap(rollout_one)(batch.rollout_inputs)
        predicted = jax.vmap(
            lambda value: trajectories_from_rollout(value, runtime["contract"])
        )(trace)

        def expected_one(state, defined, gpp, gpp_defined, templates):
            return extract_biomass_productivity_trajectories(
                state,
                defined,
                daily_fast_gpp=gpp,
                daily_fast_gpp_defined=gpp_defined,
                pft_fraction=templates.pft.pft_fraction,
                active_pft_mask=templates.pft.active_pft_mask,
                contract=runtime["contract"],
            )

        expected = jax.vmap(expected_one)(
            batch.expected_next_state,
            batch.expected_next_state_defined,
            batch.expected_fast_gpp,
            batch.expected_fast_gpp_defined,
            batch.rollout_inputs,
        )
        hard_counts = successor_rollout_hard_gate_counts(
            batch,
            trace,
            markov_contract=runtime["contract"],
            budget_atol=float(contract.payload["gates"]["budget_absolute_tolerance"]),
            stock_atol=float(contract.payload["gates"]["stock_negative_absolute_tolerance"]),
        )
        return _loss_vector(
            jax,
            anchor,
            predicted,
            expected,
            coefficients=coefficients,
            long_scales=long_scales,
        ), hard_counts

    return objective


def _make_mixed_objective(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    coefficients: Mapping[str, float],
    long_scales: Any,
    contract: Any,
) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
        rollout_daily_operator,
        trajectories_from_rollout,
    )

    jax = runtime["jax"]
    anchor_objective, _ = _make_arm_objectives(
        runtime,
        arm,
        adapter_template,
        long_scales,
    )
    process_head = _head(runtime, arm)

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
        )(parameters, context_batches, day_inputs)
        hard_counts = jax.tree_util.tree_map(
            lambda values: jnp.sum(values, axis=0),
            context_hard_counts,
        )
        return _loss_vector(
            jax,
            anchor,
            predicted,
            expected,
            coefficients=coefficients,
            long_scales=long_scales,
        ), hard_counts

    return objective


def _make_measurement(jax: Any, objective: Any) -> Any:
    import jax.numpy as jnp

    def measurement(parameters: Any, *arguments: Any) -> Any:
        values, pullback, hard_counts = jax.vjp(
            lambda current: objective(current, *arguments),
            parameters,
            has_aux=True,
        )
        gradients = jax.lax.map(
            lambda seed: pullback(seed)[0],
            jnp.eye(len(GRADIENT_LABELS), dtype=jnp.float64),
        )
        gram = sum(
            (
                jnp.einsum("i...,j...->ij", leaf, leaf)
                for leaf in jax.tree_util.tree_leaves(gradients)
            ),
            jnp.zeros(
                (len(GRADIENT_LABELS), len(GRADIENT_LABELS)),
                dtype=jnp.float64,
            ),
        )
        norms = jnp.sqrt(jnp.maximum(jnp.diag(gram), 0.0))
        denominator = norms[:, None] * norms[None, :]
        cosines = jnp.where(denominator > 0.0, gram / denominator, 0.0)
        nonfinite = sum(
            (
                jnp.sum(
                    ~jnp.isfinite(leaf),
                    axis=tuple(range(1, leaf.ndim)),
                )
                for leaf in jax.tree_util.tree_leaves(gradients)
            ),
            jnp.zeros((len(GRADIENT_LABELS),), dtype=jnp.int64),
        )
        return values, norms, cosines, nonfinite, hard_counts

    return jax.jit(measurement)


def _measurement_record(result: Any) -> Mapping[str, Any]:
    values, norms, cosines, nonfinite, hard_counts = result
    host_values = np.asarray(values, dtype=np.float64)
    host_norms = np.asarray(norms, dtype=np.float64)
    host_cosines = np.asarray(cosines, dtype=np.float64)
    host_nonfinite = np.asarray(nonfinite, dtype=np.int64)
    return {
        "losses": {
            label: float(host_values[index])
            for index, label in enumerate(GRADIENT_LABELS)
        },
        "gradient_norms": {
            label: float(host_norms[index])
            for index, label in enumerate(GRADIENT_LABELS)
        },
        "cosines": {
            left: {
                right: float(host_cosines[left_index, right_index])
                for right_index, right in enumerate(GRADIENT_LABELS)
            }
            for left_index, left in enumerate(GRADIENT_LABELS)
        },
        "nonfinite_gradient_values": {
            label: int(host_nonfinite[index])
            for index, label in enumerate(GRADIENT_LABELS)
        },
        "hard_gate_counts": _hard_counts_host(hard_counts),
    }


def _pair_key(left: str, right: str) -> str:
    positions = {label: index for index, label in enumerate(GRADIENT_LABELS)}
    if left not in positions or right not in positions or left == right:
        raise ValueError(f"invalid gradient pair: {left}, {right}")
    first, second = sorted((left, right), key=positions.__getitem__)
    return f"{first}__vs__{second}"


def _summarize(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(records) != DIAGNOSTIC_GROUPS:
        raise ValueError("per-output gradient summary requires four groups")
    norms = {
        label: np.asarray(
            [record["gradient_norms"][label] for record in records],
            dtype=np.float64,
        )
        for label in GRADIENT_LABELS
    }
    pairs = {}
    for left_index, left in enumerate(GRADIENT_LABELS):
        for right in GRADIENT_LABELS[left_index + 1 :]:
            values = np.asarray(
                [record["cosines"][left][right] for record in records],
                dtype=np.float64,
            )
            pairs[_pair_key(left, right)] = {
                "cosines": [float(value) for value in values],
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
                "negative_fraction": float(np.mean(values < 0.0)),
            }
    return {
        "gradient_norms": {
            label: {
                "values": [float(value) for value in values],
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            }
            for label, values in norms.items()
        },
        "pairs": pairs,
    }


def _conflict_classification(
    single: Mapping[str, Any],
    mixed: Mapping[str, Any],
) -> Mapping[str, Any]:
    records = {}
    confirmed = 0
    mitigated = 0
    for left, right in CONFLICT_PAIRS:
        key = _pair_key(left, right)
        single_pair = single["pairs"][key]
        mixed_pair = mixed["pairs"][key]
        stable_conflict = (
            float(single_pair["median"]) <= CONFLICT_COSINE_THRESHOLD
            and float(single_pair["negative_fraction"]) >= CONFLICT_GROUP_FRACTION
        )
        delta = float(mixed_pair["median"]) - float(single_pair["median"])
        current_mitigated = stable_conflict and delta >= MITIGATION_COSINE_DELTA
        confirmed += int(stable_conflict)
        mitigated += int(current_mitigated)
        records[key] = {
            "single_median_cosine": float(single_pair["median"]),
            "single_negative_fraction": float(single_pair["negative_fraction"]),
            "mixed_median_cosine": float(mixed_pair["median"]),
            "mixed_negative_fraction": float(mixed_pair["negative_fraction"]),
            "mixed_minus_single_median_cosine": delta,
            "stable_single_conflict": stable_conflict,
            "mixed_mitigated": current_mitigated,
        }
    supported = confirmed >= 2 and mitigated >= 2
    return {
        "status": (
            "attention_mixed_conflict_mitigation_supported"
            if supported
            else "attention_mixed_conflict_mitigation_not_supported"
        ),
        "confirmed_conflict_pairs": confirmed,
        "mitigated_conflict_pairs": mitigated,
        "required_pairs": 2,
        "pairs": records,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
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
        raise ValueError("per-output gradient audit requires staged-v3")
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
        raise ValueError("per-output gradient audit validation identity drift")
    identity = _training_identity(runtime, contract, schedule, assets, validation_prepared)
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
    first_entry, first_artifact = selected[0]
    _reference, adapter_template, adapter_timing = _prepare_adapter(runtime, first_entry)
    signature = _retained_tail_trace_signature(adapter_template)
    if any(artifact.trace_signature != signature for _entry, artifact in selected):
        raise ValueError("per-output gradient audit retained-tail trace drift")
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
        raise ValueError("per-output gradient audit parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("per-output gradient audit requires update-1000 recovery")
    parameters = dict(recovery["parameters"])
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    full_measurements = {
        arm: _make_measurement(
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
        for arm in arms
    }
    mixed_measurements = {
        arm: _make_measurement(
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
        for arm in arms
    }
    records = {arm: [] for arm in arms}
    memory_before = _device_memory_report(runtime["devices"])
    for group_index in range(DIAGNOSTIC_GROUPS):
        group = selected[
            group_index * POINTS_PER_MIXED_UPDATE :
            (group_index + 1) * POINTS_PER_MIXED_UPDATE
        ]
        designated = group[group_index % POINTS_PER_MIXED_UPDATE]
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
        for arm in arms:
            single_result = full_measurements[arm](
                parameters[arm],
                designated[1].batch,
                designated[1].day_inputs,
            )
            mixed_result = mixed_measurements[arm](
                parameters[arm],
                anchor_batch,
                context_batch,
                context_inputs,
            )
            jax.block_until_ready((single_result, mixed_result))
            single_record = _measurement_record(single_result)
            mixed_record = _measurement_record(mixed_result)
            if (
                sum(single_record["hard_gate_counts"].values()) != 0
                or sum(mixed_record["hard_gate_counts"].values()) != 0
                or sum(single_record["nonfinite_gradient_values"].values()) != 0
                or sum(mixed_record["nonfinite_gradient_values"].values()) != 0
            ):
                raise ValueError(f"per-output gradient hard failure for {arm}/group {group_index}")
            records[arm].append(
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
                }
            )
        del group, contexts, anchor_batch, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "successor_per_output_gradient_group",
                    "group": group_index,
                    "total": DIAGNOSTIC_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summaries = {
        arm: {
            "single": _summarize(tuple(record["single"] for record in records[arm])),
            "mixed": _summarize(tuple(record["mixed"] for record in records[arm])),
        }
        for arm in arms
    }
    classification = _conflict_classification(
        summaries[SOURCE_ROUTED_ATTENTION]["single"],
        summaries[SOURCE_ROUTED_ATTENTION]["mixed"],
    )
    report = {
        "schema_version": "successor_per_output_gradient_audit_v1",
        "status": "passed",
        "git_head": _git_head(),
        "parent_update": PARENT_UPDATE,
        "parameter_count_per_arm": counts,
        "gradient_labels": list(GRADIENT_LABELS),
        "audited_outputs": list(AUDITED_OUTPUTS),
        "loss_contribution_policy": {
            "normalization": "frozen_train_train_scales",
            "coefficients": dict(assets.coefficients),
            "field_reduction": "exact_equal_field_contribution_to_frozen_macro_loss",
            "horizons": list(HORIZONS),
        },
        "classification": classification,
        "summaries": summaries,
        "records": records,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - started,
        "training_updates": 0,
        "checkpoint_written": False,
        "recovery_written": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "next_gate": (
            "independent_attention_mixed_sampling_replication_design"
            if classification["status"]
            == "attention_mixed_conflict_mitigation_supported"
            else "review_loss_conflict_without_more_training"
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
