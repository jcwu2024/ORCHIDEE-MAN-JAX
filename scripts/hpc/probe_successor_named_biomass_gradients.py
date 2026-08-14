"""Zero-update real-v6 gradient preflight for named AGB/BGB tendencies."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from scripts.hpc.probe_successor_batch_diversity import (
    DIAGNOSTIC_GROUPS,
    POINTS_PER_MIXED_UPDATE,
    _selected_positions,
    _slice_context_batch,
    _stack_contexts,
)
from scripts.hpc.probe_successor_per_output_gradients import PARENT_UPDATE
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import (
    _device_memory_report,
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
from scripts.hpc.successor_staged_validation_prepared import (
    load_staged_validation_prepared,
)

HORIZONS = (1, 7, 30)
GRADIENT_LABELS = (
    "npp",
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
)


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("contract_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_preflight_contract(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "successor_named_biomass_gradient_preflight_v1"
        or payload.get("status")
        != "frozen_zero_update_preflight_no_training_authorized"
        or payload.get("contract_sha256") != _canonical_sha256(payload)
    ):
        raise ValueError("named biomass gradient preflight contract drift")
    return payload


def _field_macro_contribution(
    jnp: Any,
    by_horizon: Mapping[int, Any],
    *,
    field: str,
    fields: tuple[str, ...],
    mse_coefficient: float,
    bias_coefficient: float,
    registry: str,
) -> Any:
    values = []
    for horizon in HORIZONS:
        errors = getattr(by_horizon[horizon], registry)
        error = errors[field]
        active_count = jnp.sum(
            jnp.stack(tuple(errors[name].count > 0.0 for name in fields)),
            dtype=jnp.float64,
        )
        weighted = (
            mse_coefficient * error.mse
            + bias_coefficient * error.bias_squared
        )
        values.append(
            jnp.where(
                (error.count > 0.0) & (active_count > 0.0),
                weighted / jnp.maximum(active_count, 1.0),
                0.0,
            )
        )
    return jnp.mean(jnp.stack(tuple(values)))


def _calibration_loss_vector(
    jax: Any,
    predicted: Any,
    expected: Any,
    *,
    flux_scales: Mapping[str, float],
    stock_scales: Mapping[str, float],
    tendency_scales: Mapping[int, Mapping[str, float]],
    flux_coefficients: Mapping[str, float],
) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
        long_horizon_biomass_productivity_objective,
    )
    from research.daily_coarse_graining.daily_operator.named_biomass_tendency import (
        NAMED_BIOMASS_TENDENCY_FIELDS,
        named_biomass_tendency_objective,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        CANDIDATE_OPTIMIZED_FLUX_FIELDS,
        LONG_HORIZON_COMPONENTS,
    )

    actual = jax.tree_util.tree_map(lambda value: jnp.swapaxes(value, 0, 1), predicted)
    wanted = jax.tree_util.tree_map(lambda value: jnp.swapaxes(value, 0, 1), expected)
    long_horizon = long_horizon_biomass_productivity_objective(
        actual.fluxes,
        wanted.fluxes,
        actual.stocks,
        wanted.stocks,
        flux_scales=flux_scales,
        stock_scales=stock_scales,
        coefficients={name: 0.0 for name in LONG_HORIZON_COMPONENTS},
        horizons=HORIZONS,
    )
    tendency = named_biomass_tendency_objective(
        actual.stocks,
        wanted.stocks,
        scales=tendency_scales,
        coefficients={"tendency_mse": 1.0, "tendency_bias_squared": 1.0},
        horizons=HORIZONS,
    )
    npp = _field_macro_contribution(
        jnp,
        long_horizon.by_horizon,
        field="npp",
        fields=CANDIDATE_OPTIMIZED_FLUX_FIELDS,
        mse_coefficient=float(flux_coefficients["cumulative_flux_mse"]),
        bias_coefficient=float(
            flux_coefficients["cumulative_flux_bias_squared"]
        ),
        registry="cumulative_flux",
    )
    stocks = tuple(
        _field_macro_contribution(
            jnp,
            tendency.by_horizon,
            field=name,
            fields=NAMED_BIOMASS_TENDENCY_FIELDS,
            mse_coefficient=1.0,
            bias_coefficient=1.0,
            registry="errors",
        )
        for name in NAMED_BIOMASS_TENDENCY_FIELDS
    )
    return jnp.stack((npp, *stocks))


def _make_context_trajectory(
    runtime: Mapping[str, Any],
    adapter_template: Any,
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
    process_head = _head(runtime, "source_routed_process_attention_v2")

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
            stock_atol=float(
                contract.payload["gates"]["stock_negative_absolute_tolerance"]
            ),
        )
        return predicted, expected, hard_counts

    return context_trajectory


def _make_calibration_measurement(
    runtime: Mapping[str, Any],
    context_trajectory: Any,
    *,
    long_scales: Any,
    tendency_scales: Mapping[int, Mapping[str, float]],
    flux_coefficients: Mapping[str, float],
) -> Any:
    import jax.numpy as jnp

    jax = runtime["jax"]

    def objective(parameters: Any, batches: Any, day_inputs: Any) -> Any:
        predicted, expected, counts = jax.vmap(
            context_trajectory,
            in_axes=(None, 0, 0),
        )(parameters, batches, day_inputs)
        hard_counts = jax.tree_util.tree_map(
            lambda values: jnp.sum(values, axis=0),
            counts,
        )
        losses = _calibration_loss_vector(
            jax,
            predicted,
            expected,
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            tendency_scales=tendency_scales,
            flux_coefficients=flux_coefficients,
        )
        return losses, hard_counts

    def measurement(parameters: Any, seed: Any, batches: Any, day_inputs: Any) -> Any:
        values, pullback, hard_counts = jax.vjp(
            lambda current: objective(current, batches, day_inputs),
            parameters,
            has_aux=True,
        )
        return values, pullback(seed)[0], hard_counts

    return jax.jit(measurement)


def _make_candidate_measurement(
    runtime: Mapping[str, Any],
    context_trajectory: Any,
    adapter_template: Any,
    *,
    long_scales: Any,
    tendency_scales: Mapping[int, Mapping[str, float]],
    flux_coefficients: Mapping[str, float],
) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_anchor_batch_loss,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        candidate_successor_training_objective,
    )

    jax = runtime["jax"]
    head = _head(runtime, "source_routed_process_attention_v2")

    def objective(
        parameters: Any,
        coefficient: Any,
        anchor_batch: Any,
        batches: Any,
        day_inputs: Any,
    ) -> Any:
        anchor = successor_anchor_batch_loss(
            parameters,
            anchor_batch.anchor_inputs,
            anchor_batch.anchor_targets,
            head=head,
            schema=runtime["schema"],
            scales=runtime["loss_scales"],
        )
        predicted, expected, counts = jax.vmap(
            context_trajectory,
            in_axes=(None, 0, 0),
        )(parameters, batches, day_inputs)
        hard_counts = jax.tree_util.tree_map(
            lambda values: jnp.sum(values, axis=0),
            counts,
        )
        actual = jax.tree_util.tree_map(
            lambda value: jnp.swapaxes(value, 0, 1),
            predicted,
        )
        wanted = jax.tree_util.tree_map(
            lambda value: jnp.swapaxes(value, 0, 1),
            expected,
        )
        result = candidate_successor_training_objective(
            anchor,
            actual,
            wanted,
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            tendency_scales=tendency_scales,
            coefficients={
                "cumulative_flux_mse": flux_coefficients[
                    "cumulative_flux_mse"
                ],
                "cumulative_flux_bias_squared": flux_coefficients[
                    "cumulative_flux_bias_squared"
                ],
                "tendency_mse": coefficient,
                "tendency_bias_squared": coefficient,
            },
            horizons=HORIZONS,
        )
        components = jnp.stack(
            (
                result.one_day_anchor,
                result.cumulative_flux_mse,
                result.cumulative_flux_bias_squared,
                result.tendency_mse,
                result.tendency_bias_squared,
            )
        )
        return result.total, (components, hard_counts)

    def measurement(*arguments: Any) -> Any:
        (loss, auxiliary), gradient = jax.value_and_grad(
            objective,
            argnums=0,
            has_aux=True,
        )(*arguments)
        squared = sum(
            (
                jnp.sum(jnp.asarray(value, dtype=jnp.float64) ** 2)
                for value in jax.tree_util.tree_leaves(gradient)
            ),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(value)), dtype=jnp.int64)
                for value in jax.tree_util.tree_leaves(gradient)
            ),
            jnp.asarray(0, dtype=jnp.int64),
        )
        components, hard_counts = auxiliary
        return loss, components, jnp.sqrt(squared), nonfinite, hard_counts

    return jax.jit(measurement)


def _gradient_record(
    jax: Any,
    measurement: Any,
    parameters: Any,
    batches: Any,
    day_inputs: Any,
) -> Mapping[str, Any]:
    values_host = None
    hard_host = None
    norms = {}
    nonfinite = {}
    for index, label in enumerate(GRADIENT_LABELS):
        seed = np.eye(len(GRADIENT_LABELS), dtype=np.float64)[index]
        values, gradient, hard_counts = measurement(
            parameters,
            seed,
            batches,
            day_inputs,
        )
        jax.block_until_ready((values, gradient, hard_counts))
        current_values = np.asarray(values, dtype=np.float64)
        current_hard = _hard_counts_host(hard_counts)
        if values_host is None:
            values_host = current_values
            hard_host = current_hard
        elif not np.array_equal(values_host, current_values) or hard_host != current_hard:
            raise ValueError("named biomass gradient seed changed objective outputs")
        leaves = tuple(
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in jax.device_get(jax.tree_util.tree_leaves(gradient))
        )
        vector = np.concatenate(leaves)
        norms[label] = float(np.linalg.norm(vector))
        nonfinite[label] = int(np.count_nonzero(~np.isfinite(vector)))
        del values, gradient, hard_counts, leaves, vector
    return {
        "losses": {
            label: float(values_host[index])
            for index, label in enumerate(GRADIENT_LABELS)
        },
        "gradient_norms": norms,
        "nonfinite_gradient_values": nonfinite,
        "hard_gate_counts": hard_host,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    preflight = _load_preflight_contract(args.preflight_contract)
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
        calibrate_named_biomass_gradient_coefficient,
        derive_long_horizon_scales,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        load_named_biomass_tendency_scale_asset,
        load_successor_training_assets,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if (
        contract.payload["schema_version"] != STAGED_SCHEMA_VERSION
        or contract.sha256 != preflight["staged_training_contract_sha256"]
    ):
        raise ValueError("named biomass preflight staged contract drift")
    arms = tuple(contract.payload["matched_arms"])
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
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
        raise ValueError("named biomass preflight validation identity drift")
    scale_asset = load_named_biomass_tendency_scale_asset(
        args.tendency_scale_asset,
        args.tendency_scale_acceptance,
    )
    scale_acceptance = json.loads(
        args.tendency_scale_acceptance.read_text(encoding="utf-8")
    )
    if (
        scale_acceptance["contract_sha256"]
        != preflight["scale_acceptance_contract_sha256"]
    ):
        raise ValueError("named biomass preflight scale acceptance drift")

    identity = _training_identity(runtime, contract, schedule, assets, validation_prepared)
    selected = []
    for position in _selected_positions(len(manifest["entries"])):
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
        raise ValueError("named biomass preflight retained-tail trace drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(
            lambda value: value[0],
            first_artifact.batch.anchor_inputs,
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
    if set(counts.values()) != {EXPECTED_PARAMETER_COUNT}:
        raise ValueError("named biomass preflight parameter count drift")
    recovery = load_staged_recovery(
        args.recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if recovery is None:
        raise ValueError("named biomass preflight requires update-1000 recovery")
    parameters = recovery["parameters"][SOURCE_ROUTED_ATTENTION]
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    context_trajectory = _make_context_trajectory(runtime, adapter_template, contract)
    calibration_measurement = _make_calibration_measurement(
        runtime,
        context_trajectory,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        flux_coefficients=assets.coefficients,
    )
    groups = []
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
        context_batch = _stack_contexts(jax, contexts)
        context_inputs = _stack_contexts(
            jax,
            tuple(artifact.day_inputs for _entry, artifact in group),
        )
        record = _gradient_record(
            jax,
            calibration_measurement,
            parameters,
            context_batch,
            context_inputs,
        )
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
                "calibration": record,
            }
        )
        del contexts, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "named_biomass_gradient_calibration_group",
                    "group": group_index,
                    "total": DIAGNOSTIC_GROUPS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rule = preflight["coefficient_rule"]
    calibration = calibrate_named_biomass_gradient_coefficient(
        [item["calibration"]["gradient_norms"]["npp"] for item in groups],
        [
            item["calibration"]["gradient_norms"][
                "aboveground_biomass_carbon"
            ]
            for item in groups
        ],
        [
            item["calibration"]["gradient_norms"][
                "belowground_biomass_carbon"
            ]
            for item in groups
        ],
        target_ratio_interval=tuple(
            rule["each_stock_to_npp_median_gradient_ratio_interval"]
        ),
        coefficient_interval=tuple(rule["coefficient_interval"]),
    )
    candidate_measurement = _make_candidate_measurement(
        runtime,
        context_trajectory,
        adapter_template,
        long_scales=long_scales,
        tendency_scales=scale_asset.scales,
        flux_coefficients=assets.coefficients,
    )
    full_records = []
    for group_index, item in enumerate(groups):
        group = selected[
            group_index * POINTS_PER_MIXED_UPDATE :
            (group_index + 1) * POINTS_PER_MIXED_UPDATE
        ]
        designated = group[group_index % POINTS_PER_MIXED_UPDATE]
        contexts = tuple(
            _slice_context_batch(jax, artifact.batch, context_index=index)
            for index, (_entry, artifact) in enumerate(group)
        )
        context_batch = _stack_contexts(jax, contexts)
        context_inputs = _stack_contexts(
            jax,
            tuple(artifact.day_inputs for _entry, artifact in group),
        )
        result = candidate_measurement(
            parameters,
            np.asarray(calibration.coefficient, dtype=np.float64),
            designated[1].batch,
            context_batch,
            context_inputs,
        )
        jax.block_until_ready(result)
        loss, components, gradient_norm, nonfinite, hard_counts = result
        full_record = {
            "loss": float(np.asarray(loss)),
            "components": {
                name: float(value)
                for name, value in zip(
                    (
                        "one_day_anchor",
                        "cumulative_flux_mse",
                        "cumulative_flux_bias_squared",
                        "tendency_mse",
                        "tendency_bias_squared",
                    ),
                    np.asarray(components),
                    strict=True,
                )
            },
            "gradient_norm": float(np.asarray(gradient_norm)),
            "nonfinite_gradient_values": int(np.asarray(nonfinite)),
            "hard_gate_counts": _hard_counts_host(hard_counts),
        }
        item["full_candidate"] = full_record
        full_records.append(full_record)
        del result, contexts, context_batch, context_inputs
        gc.collect()

    hard_failure_count = sum(
        sum(item["calibration"]["hard_gate_counts"].values())
        + sum(item["full_candidate"]["hard_gate_counts"].values())
        for item in groups
    )
    nonfinite_gradient_count = sum(
        sum(item["calibration"]["nonfinite_gradient_values"].values())
        + int(item["full_candidate"]["nonfinite_gradient_values"])
        for item in groups
    )
    all_losses_finite = all(
        all(np.isfinite(tuple(item["calibration"]["losses"].values())))
        and np.isfinite(item["full_candidate"]["loss"])
        and all(np.isfinite(tuple(item["full_candidate"]["components"].values())))
        for item in groups
    )
    passed = (
        calibration.passed
        and hard_failure_count == 0
        and nonfinite_gradient_count == 0
        and all_losses_finite
        and all(item["full_candidate"]["gradient_norm"] > 0.0 for item in groups)
    )
    report = {
        "schema_version": "successor_named_biomass_gradient_preflight_report_v1",
        "status": "passed" if passed else "failed",
        "git_head": _git_head(),
        "identity": {
            "preflight_contract_sha256": preflight["contract_sha256"],
            "scale_acceptance_contract_sha256": scale_acceptance[
                "contract_sha256"
            ],
            "scale_asset_file_sha256": scale_asset.file_sha256,
            "staged_training_contract_sha256": contract.sha256,
        },
        "parent_update": PARENT_UPDATE,
        "audited_arm": SOURCE_ROUTED_ATTENTION,
        "parameter_count_per_arm": counts,
        "gradient_labels": list(GRADIENT_LABELS),
        "calibration": {
            "coefficient": calibration.coefficient,
            "median_gradient_norms": dict(calibration.median_gradient_norms),
            "achieved_stock_to_npp_ratios": dict(
                calibration.achieved_stock_to_npp_ratios
            ),
            "coefficient_within_bounds": calibration.coefficient_within_bounds,
            "ratios_within_interval": calibration.ratios_within_interval,
            "passed": calibration.passed,
            "rule": dict(rule),
        },
        "gates": {
            "all_losses_finite": all_losses_finite,
            "nonfinite_gradient_count": nonfinite_gradient_count,
            "hard_failure_count": hard_failure_count,
            "all_full_candidate_gradient_norms_positive": all(
                item["full_candidate"]["gradient_norm"] > 0.0 for item in groups
            ),
        },
        "groups": groups,
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
            "freeze_resolved_candidate_coefficients_and_review_training_gates"
            if passed
            else "stop_named_biomass_tendency_mechanism_without_training"
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
    parser.add_argument("--preflight-contract", type=Path, required=True)
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
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
