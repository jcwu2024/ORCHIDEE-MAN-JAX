"""Attribute fresh-panel GPP error without training or parameter changes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_checkpoint import (
    load_broad_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_attribution import (
    add_residual_decomposition_statistics,
    classify_fresh_attribution,
    finalize_residual_decomposition_statistics,
    load_structured_gpp_fresh_attribution_contract,
    residual_decomposition_sufficient_statistics,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_screen import (
    load_structured_gpp_fresh_screen_contract,
    verify_fresh_validation_panel,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_error_sufficient_statistics,
    add_target_oriented_acceptance_statistics,
    derive_target_oriented_acceptance_scales,
    finalize_target_oriented_acceptance_statistics,
    normalized_error_from_sufficient_statistics,
    normalized_error_sufficient_statistics,
    target_oriented_acceptance_sufficient_statistics,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    _hard_host,
    _load_context,
    _tree_size,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _replace_adapter,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import HardCounts
from scripts.hpc.run_target_oriented_structured_gpp_fresh_screen import (
    _load_broad_training_report,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
)
from scripts.hpc.run_target_oriented_training import (
    _add_hard_host,
    _hard_failure_count,
    _validation_records,
)
from scripts.hpc.successor_rollout_screen_prepared import (
    load_prepared_rollout_artifact,
)
from scripts.hpc.target_oriented_validation_prepared import (
    load_target_oriented_validation_prepared,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_fresh_attribution_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_attribution_evaluator(
    runtime: Mapping[str, Any],
    head: Any,
    template: Any,
    acceptance_scales: Any,
    state_scales: Any,
) -> Any:
    import jax
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
        rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_training import (
        TERMINAL_STOCK_FIELDS,
        extract_biomass_productivity_trajectories,
        rollout_daily_operator,
        trajectories_from_rollout,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_gpp_bridge import (
        rollout_with_teacher_fast_gpp,
        teacher_gpp_bridge_transition,
    )

    learned_names = tuple(field.name for field in runtime["learned_schema"].fields)

    @partial(jax.jit, static_argnames=("horizon",))
    def evaluate(parameters: Any, batch: Any, day_inputs: Any, *, horizon: int) -> Any:
        retained = replace(template, day_inputs=day_inputs)
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=head,
            retained_tail=retained,
        )
        bridge_transition = partial(
            teacher_gpp_bridge_transition,
            input_assembler=runtime["adapter"].input_assembler,
            process_head=head,
            retained_tail=retained,
        )

        trace = jax.vmap(
            lambda inputs: rollout_daily_operator(
                parameters,
                inputs,
                transition=transition,
                contract=runtime["contract"],
                rematerialize=False,
            )
        )(batch.rollout_inputs)
        bridge_trace = jax.vmap(
            lambda inputs, teacher_gpp: rollout_with_teacher_fast_gpp(
                parameters,
                inputs,
                teacher_gpp,
                transition=bridge_transition,
                contract=runtime["contract"],
                rematerialize=False,
            )
        )(batch.rollout_inputs, batch.expected_fast_gpp)
        predicted = jax.vmap(lambda value: trajectories_from_rollout(value, runtime["contract"]))(trace)
        bridge_predicted = jax.vmap(lambda value: trajectories_from_rollout(value, runtime["contract"]))(bridge_trace)

        def expected_one(state, defined, gpp, gpp_defined, inputs):
            return extract_biomass_productivity_trajectories(
                state,
                defined,
                daily_fast_gpp=gpp,
                daily_fast_gpp_defined=gpp_defined,
                pft_fraction=inputs.pft.pft_fraction,
                active_pft_mask=inputs.pft.active_pft_mask,
                contract=runtime["contract"],
            )

        expected = jax.vmap(expected_one)(
            batch.expected_next_state,
            batch.expected_next_state_defined,
            batch.expected_fast_gpp,
            batch.expected_fast_gpp_defined,
            batch.rollout_inputs,
        )
        one_predicted = jax.tree_util.tree_map(lambda value: value[0], predicted)
        one_bridge = jax.tree_util.tree_map(lambda value: value[0], bridge_predicted)
        one_expected = jax.tree_util.tree_map(lambda value: value[0], expected)
        free_statistics = target_oriented_acceptance_sufficient_statistics(
            one_predicted.fluxes,
            one_expected.fluxes,
            one_predicted.stocks,
            one_expected.stocks,
            scales=acceptance_scales,
            horizon=horizon,
        )
        bridge_statistics = target_oriented_acceptance_sufficient_statistics(
            one_bridge.fluxes,
            one_expected.fluxes,
            one_bridge.stocks,
            one_expected.stocks,
            scales=acceptance_scales,
            horizon=horizon,
        )

        teacher_inputs = jax.tree_util.tree_map(lambda value: value[0], batch.rollout_inputs)

        def direct_gpp_step(carry: Any, inputs: Any) -> tuple[Any, Any]:
            assembled = runtime["adapter"].input_assembler.assemble(inputs)
            context = head.assemble_context(parameters, assembled)
            prediction = head(parameters, context, assembled)
            return carry, prediction.named_outputs["daily.fast_gpp"]

        _, direct_values = jax.lax.scan(direct_gpp_step, None, teacher_inputs)
        direct_predicted = jnp.asarray(direct_values, dtype=jnp.float64)
        direct_expected = jnp.asarray(batch.expected_fast_gpp[0], dtype=jnp.float64)
        direct_defined = jnp.asarray(batch.expected_fast_gpp_defined[0], dtype=jnp.bool_)
        fraction = jnp.asarray(teacher_inputs.pft.pft_fraction, dtype=jnp.float64)
        active = jnp.asarray(teacher_inputs.pft.active_pft_mask, dtype=jnp.bool_)
        if fraction.ndim == 2:
            fraction = fraction[:, None, :]
        if active.ndim == 2:
            active = active[:, None, :]
        admitted = active & direct_defined
        direct_complete = jnp.all((~active) | admitted, axis=-1)
        direct_area = jnp.sum(jnp.where(admitted, direct_predicted * fraction, 0.0), axis=-1)
        expected_area = jnp.sum(jnp.where(admitted, direct_expected * fraction, 0.0), axis=-1)
        free_gpp = one_predicted.fluxes["gpp"]
        common_daily = (
            direct_complete
            & jnp.asarray(free_gpp.defined, dtype=jnp.bool_)
            & jnp.asarray(one_expected.fluxes["gpp"].defined, dtype=jnp.bool_)
        )
        daily_decomposition = residual_decomposition_sufficient_statistics(
            expected_area,
            direct_area,
            free_gpp.value,
            common_daily,
            acceptance_scales.fluxes["gpp"],
        )
        complete_horizon = jnp.all(common_daily[:horizon], axis=0)
        direct_sum = jnp.sum(jnp.where(common_daily[:horizon], direct_area[:horizon], 0.0), axis=0)
        free_sum = jnp.sum(jnp.where(common_daily[:horizon], free_gpp.value[:horizon], 0.0), axis=0)
        expected_sum = jnp.sum(jnp.where(common_daily[:horizon], expected_area[:horizon], 0.0), axis=0)
        horizon_scale = horizon * jnp.asarray(acceptance_scales.fluxes["gpp"], dtype=jnp.float64)
        horizon_decomposition = residual_decomposition_sufficient_statistics(
            expected_sum,
            direct_sum,
            free_sum,
            complete_horizon,
            horizon_scale,
        )
        direct_cumulative = normalized_error_sufficient_statistics(
            direct_sum,
            expected_sum,
            complete_horizon,
            horizon_scale,
        )

        def hard_counts(local_trace, local_predicted):
            rollout_hard = rollout_hard_gate_counts(
                local_predicted.stocks,
                one_expected.stocks,
                nonnegative_stock_fields=TERMINAL_STOCK_FIELDS,
                budget_residuals={
                    "water": local_trace.result.diagnostics["water_budget_residual"][0],
                    "carbon": local_trace.result.diagnostics["carbon_budget_residual"][0],
                },
                predicted_discrete_state=jax.tree_util.tree_map(
                    lambda value: value[0], local_trace.result.next_discrete_state
                ),
                expected_discrete_state=jax.tree_util.tree_map(
                    lambda value: value[0], batch.expected_next_discrete_state
                ),
                budget_atol=2.0e-12,
                stock_atol=2.0e-12,
            )
            actual_state = jnp.asarray(local_trace.result.next_state.other[0])
            actual_defined = jnp.asarray(local_trace.result.next_state_defined.other[0], dtype=jnp.bool_)
            wanted_defined = jnp.asarray(batch.expected_next_state_defined[0], dtype=jnp.bool_)
            dynamic = jnp.broadcast_to(
                jnp.asarray(state_scales.state_declared_dynamic.other, dtype=jnp.bool_),
                actual_defined.shape,
            )
            mismatch = actual_defined != wanted_defined
            prediction_nonfinite = sum(
                (
                    jnp.sum(~jnp.isfinite(jnp.asarray(value)), dtype=jnp.int32)
                    for name, value in local_trace.result.prediction.named_outputs.items()
                    if name in learned_names
                ),
                jnp.asarray(0, dtype=jnp.int32),
            )
            return HardCounts(
                prediction_nonfinite + rollout_hard.nonfinite_defined_prediction,
                jnp.sum(wanted_defined & ~jnp.isfinite(actual_state), dtype=jnp.int32),
                jnp.sum(mismatch & ~dynamic, dtype=jnp.int32) + rollout_hard.defined_mask_mismatch,
                jnp.sum(mismatch & dynamic, dtype=jnp.int32),
                rollout_hard.discrete_state_mismatch,
                rollout_hard.budget_residual_failure,
                rollout_hard.negative_declared_stock,
            )

        def carbon_identity_error(local_predicted):
            fields = ("gpp", "maintenance_respiration", "growth_respiration")
            defined = jnp.ones_like(jnp.asarray(local_predicted.fluxes["gpp"].defined), dtype=jnp.bool_)
            for name in fields:
                defined = (
                    defined
                    & jnp.asarray(local_predicted.fluxes[name].defined, dtype=jnp.bool_)
                    & jnp.asarray(one_expected.fluxes[name].defined, dtype=jnp.bool_)
                )
            complete = jnp.all(defined[:horizon], axis=0)

            def residual(name):
                return jnp.sum(
                    jnp.where(
                        defined[:horizon],
                        local_predicted.fluxes[name].value[:horizon] - one_expected.fluxes[name].value[:horizon],
                        0.0,
                    ),
                    axis=0,
                )

            gpp_residual = residual("gpp")
            maintenance_residual = residual("maintenance_respiration")
            growth_residual = residual("growth_respiration")
            npp_residual = gpp_residual - maintenance_residual - growth_residual
            closure = npp_residual - (gpp_residual - maintenance_residual - growth_residual)
            return {
                "gpp": jnp.where(complete, gpp_residual, 0.0),
                "maintenance_respiration": jnp.where(complete, maintenance_residual, 0.0),
                "growth_respiration": jnp.where(complete, growth_residual, 0.0),
                "npp": jnp.where(complete, npp_residual, 0.0),
                "maximum_absolute_identity_error": jnp.max(jnp.where(complete, jnp.abs(closure), 0.0)),
            }

        return {
            "free_statistics": free_statistics,
            "bridge_statistics": bridge_statistics,
            "daily_decomposition": daily_decomposition,
            "horizon_decomposition": horizon_decomposition,
            "direct_cumulative": direct_cumulative,
            "free_carbon_residuals": carbon_identity_error(one_predicted),
            "bridge_carbon_residuals": carbon_identity_error(one_bridge),
            "free_hard": hard_counts(trace, one_predicted),
            "bridge_hard": hard_counts(bridge_trace, one_bridge),
        }

    return evaluate


def _direct_cumulative_report(statistics: Any, *, horizon: int, target_contract: Any):
    error = normalized_error_from_sufficient_statistics(statistics)
    mse = float(np.asarray(error.mse))
    bias = float(np.asarray(error.mean_bias))
    count = float(np.asarray(error.count))
    rmse = float(np.sqrt(max(mse, 0.0)))
    gate = target_contract.payload["acceptance"]["short_horizon"][str(horizon)]
    return {
        "normalized_rmse": rmse,
        "normalized_signed_bias": bias,
        "evaluated_values": count,
        "rmse_maximum": float(gate["normalized_rmse_maximum"]),
        "absolute_bias_maximum": float(gate["absolute_normalized_signed_bias_maximum"]),
        "passed": bool(
            count > 0.0
            and np.isfinite(rmse)
            and np.isfinite(bias)
            and rmse <= float(gate["normalized_rmse_maximum"])
            and abs(bias) <= float(gate["absolute_normalized_signed_bias_maximum"])
        ),
    }


def _formal_metric_difference(observed: Mapping[str, Any], formal: Mapping[str, Any]) -> float:
    differences = [
        abs(float(observed["science_macro_normalized_rmse"]) - float(formal["science_macro_normalized_rmse"]))
    ]
    if observed["hard_counts"] != formal["hard_counts"]:
        return float("inf")
    for slice_id in ("temporal", "spatial", "joint"):
        for horizon in ("7", "30"):
            left = observed["slices"][slice_id][horizon]
            right = formal["slices"][slice_id][horizon]
            if bool(left["all_key_gates_passed"]) != bool(right["all_key_gates_passed"]):
                return float("inf")
            differences.append(
                abs(float(left["science_macro_normalized_rmse"]) - float(right["science_macro_normalized_rmse"]))
            )
            for name in left["keys"]:
                if bool(left["keys"][name]["passed"]) != bool(right["keys"][name]["passed"]):
                    return float("inf")
                for metric in ("normalized_rmse", "normalized_signed_bias"):
                    differences.append(abs(float(left["keys"][name][metric]) - float(right["keys"][name][metric])))
    return max(differences)


def _all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_numeric_values_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        return bool(np.isfinite(value))
    return True


def _summarize_carbon_residuals(
    records: Mapping[str, Mapping[str, list[float]]],
) -> Mapping[str, Any]:
    result = {}
    for mode in ("ordinary_free_rollout", "teacher_gpp_bridge"):
        arrays = {name: np.asarray(values, dtype=np.float64) for name, values in records[mode].items()}
        if any(value.size < 1 or not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("fresh-attribution carbon residuals are empty or nonfinite")

        def correlation(left: str, right: str) -> float:
            if arrays[left].size < 2:
                return 0.0
            left_std = float(np.std(arrays[left]))
            right_std = float(np.std(arrays[right]))
            if left_std <= 0.0 or right_std <= 0.0:
                return 0.0
            return float(np.corrcoef(arrays[left], arrays[right])[0, 1])

        means = {name: float(np.mean(value)) for name, value in arrays.items()}
        result[mode] = {
            "evaluated_values": int(arrays["gpp"].size),
            "fields": {
                name: {
                    "mean_cumulative_residual": means[name],
                    "median_cumulative_residual": float(np.median(value)),
                    "rms_cumulative_residual": float(np.sqrt(np.mean(value * value))),
                }
                for name, value in arrays.items()
            },
            "npp_mean_residual_components": {
                "gpp": means["gpp"],
                "negative_maintenance_respiration": -means["maintenance_respiration"],
                "negative_growth_respiration": -means["growth_respiration"],
                "reconstructed_npp": means["gpp"] - means["maintenance_respiration"] - means["growth_respiration"],
                "reported_npp": means["npp"],
            },
            "window_residual_correlations": {
                "gpp_vs_growth_respiration": correlation("gpp", "growth_respiration"),
                "gpp_vs_npp": correlation("gpp", "npp"),
            },
        }
    return result


def _evaluate_panel(
    runtime: Mapping[str, Any],
    *,
    parameters: Any,
    evaluator: Any,
    prepared_path: Path,
    prepared: Mapping[str, Any],
    panel: Mapping[str, Any],
    target_contract: Any,
    formal_report: Mapping[str, Any],
    attribution_contract: Any,
    mode: str,
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    free_hard = {name: 0 for name in HardCounts._fields}
    bridge_hard = {name: 0 for name in HardCounts._fields}
    maximum_carbon_identity_error = 0.0
    carbon_records: dict[tuple[str, int], dict[str, dict[str, list[float]]]] = {}
    timings = []
    selected = _validation_records(panel, prepared, mode=mode)
    for ordinal, (record, (slice_id, horizon, entry)) in enumerate(selected, start=1):
        artifact = load_prepared_rollout_artifact(
            prepared_path,
            record,
            slice_id=slice_id,
            window_entry=entry,
        )
        started = time.perf_counter()
        local = evaluator(
            parameters,
            artifact.batch,
            artifact.day_inputs,
            horizon=horizon,
        )
        jax.block_until_ready(local)
        timings.append(time.perf_counter() - started)
        local = jax.device_get(local)
        key = (slice_id, horizon)
        if key not in cells:
            cells[key] = {
                "free": local["free_statistics"],
                "bridge": local["bridge_statistics"],
                "daily_decomposition": local["daily_decomposition"],
                "horizon_decomposition": local["horizon_decomposition"],
                "direct_cumulative": local["direct_cumulative"],
            }
        else:
            cell = cells[key]
            cell["free"] = add_target_oriented_acceptance_statistics(cell["free"], local["free_statistics"])
            cell["bridge"] = add_target_oriented_acceptance_statistics(cell["bridge"], local["bridge_statistics"])
            for name in ("daily_decomposition", "horizon_decomposition"):
                cell[name] = add_residual_decomposition_statistics(cell[name], local[name])
            cell["direct_cumulative"] = add_error_sufficient_statistics(
                cell["direct_cumulative"], local["direct_cumulative"]
            )
        free_hard = _add_hard_host(free_hard, _hard_host(local["free_hard"]))
        bridge_hard = _add_hard_host(bridge_hard, _hard_host(local["bridge_hard"]))
        maximum_carbon_identity_error = max(
            maximum_carbon_identity_error,
            float(local["free_carbon_residuals"]["maximum_absolute_identity_error"]),
            float(local["bridge_carbon_residuals"]["maximum_absolute_identity_error"]),
        )
        carbon_cell = carbon_records.setdefault(
            key,
            {
                mode: {
                    name: []
                    for name in (
                        "gpp",
                        "maintenance_respiration",
                        "growth_respiration",
                        "npp",
                    )
                }
                for mode in ("ordinary_free_rollout", "teacher_gpp_bridge")
            },
        )
        for mode, source_name in (
            ("ordinary_free_rollout", "free_carbon_residuals"),
            ("teacher_gpp_bridge", "bridge_carbon_residuals"),
        ):
            for name in carbon_cell[mode]:
                carbon_cell[mode][name].extend(
                    np.asarray(local[source_name][name], dtype=np.float64).reshape(-1).tolist()
                )
        print(
            f"fresh_attribution ordinal={ordinal}/{len(selected)} slice={slice_id} horizon={horizon}",
            flush=True,
        )

    free_reports = {}
    bridge_reports = {}
    direct_reports = {}
    daily_decomposition = {}
    horizon_decomposition = {}
    carbon_propagation = {}
    free_macros = []
    maximum_residual_closure = 0.0
    maximum_squared_closure = 0.0
    for slice_id in ("temporal", "spatial", "joint"):
        free_reports[slice_id] = {}
        bridge_reports[slice_id] = {}
        direct_reports[slice_id] = {}
        daily_decomposition[slice_id] = {}
        horizon_decomposition[slice_id] = {}
        carbon_propagation[slice_id] = {}
        for horizon in (7, 30):
            cell = cells[(slice_id, horizon)]
            free = finalize_target_oriented_acceptance_statistics(
                cell["free"], horizon=horizon, target_contract=target_contract
            )
            bridge = finalize_target_oriented_acceptance_statistics(
                cell["bridge"], horizon=horizon, target_contract=target_contract
            )
            daily = finalize_residual_decomposition_statistics(cell["daily_decomposition"])
            cumulative = finalize_residual_decomposition_statistics(cell["horizon_decomposition"])
            free_reports[slice_id][str(horizon)] = free
            bridge_reports[slice_id][str(horizon)] = bridge
            direct_reports[slice_id][str(horizon)] = _direct_cumulative_report(
                cell["direct_cumulative"],
                horizon=horizon,
                target_contract=target_contract,
            )
            daily_decomposition[slice_id][str(horizon)] = daily
            horizon_decomposition[slice_id][str(horizon)] = cumulative
            carbon_propagation[slice_id][str(horizon)] = _summarize_carbon_residuals(
                carbon_records[(slice_id, horizon)]
            )
            free_macros.append(free["science_macro_normalized_rmse"])
            maximum_residual_closure = max(
                maximum_residual_closure,
                daily["maximum_absolute_residual_closure_error"],
                cumulative["maximum_absolute_residual_closure_error"],
            )
            maximum_squared_closure = max(
                maximum_squared_closure,
                daily["squared_decomposition_relative_error"],
                cumulative["squared_decomposition_relative_error"],
            )

    observed_free = {
        "science_macro_normalized_rmse": float(np.mean(free_macros)),
        "slices": free_reports,
        "hard_counts": free_hard,
    }
    metric_difference = _formal_metric_difference(observed_free, formal_report["panel"]) if mode == "formal" else None
    classification = (
        classify_fresh_attribution(
            direct_reports,
            horizon_decomposition,
            free_reports,
            bridge_reports,
            attribution_contract,
        )
        if mode == "formal"
        else {"classification": "not_run_in_admission"}
    )
    gates = attribution_contract.payload["integrity_gates"]
    free_hard_count = _hard_failure_count(free_hard)
    bridge_hard_count = _hard_failure_count(bridge_hard)
    all_finite = _all_numeric_values_finite(
        {
            "ordinary_free": observed_free,
            "teacher_day_start_direct_gpp": direct_reports,
            "gpp_residual_decomposition_daily": daily_decomposition,
            "gpp_residual_decomposition_cumulative": horizon_decomposition,
            "teacher_gpp_bridge": bridge_reports,
            "signed_carbon_residual_propagation": carbon_propagation,
            "classification": classification,
        }
    )
    integrity = {
        "ordinary_free_report_maximum_absolute_metric_difference": metric_difference,
        "ordinary_free_report_reproduced": mode != "formal"
        or metric_difference <= float(gates["ordinary_free_report_maximum_absolute_metric_difference"]),
        "gpp_residual_identity_maximum_absolute_error": maximum_residual_closure,
        "gpp_residual_identity_passed": maximum_residual_closure
        <= float(gates["gpp_residual_identity_maximum_absolute_error"]),
        "gpp_squared_decomposition_relative_error_maximum": maximum_squared_closure,
        "gpp_squared_decomposition_passed": maximum_squared_closure
        <= float(gates["gpp_squared_decomposition_relative_error_maximum"]),
        "carbon_residual_identity_maximum_absolute_error": maximum_carbon_identity_error,
        "carbon_residual_identity_passed": maximum_carbon_identity_error
        <= float(gates["carbon_residual_identity_maximum_absolute_error"]),
        "ordinary_free_hard_failure_count": free_hard_count,
        "teacher_gpp_bridge_hard_failure_count": bridge_hard_count,
        "hard_gates_passed": free_hard_count == 0 and bridge_hard_count == 0,
        "all_reported_statistics_finite_and_nonempty": all_finite,
    }
    integrity["passed"] = all(
        bool(integrity[name])
        for name in (
            "ordinary_free_report_reproduced",
            "gpp_residual_identity_passed",
            "gpp_squared_decomposition_passed",
            "carbon_residual_identity_passed",
            "hard_gates_passed",
            "all_reported_statistics_finite_and_nonempty",
        )
    )
    return {
        "evaluated_windows": len(selected),
        "ordinary_free": observed_free,
        "teacher_day_start_direct_gpp": direct_reports,
        "gpp_residual_decomposition": {
            "daily": daily_decomposition,
            "cumulative": horizon_decomposition,
        },
        "teacher_gpp_bridge": {"slices": bridge_reports, "hard_counts": bridge_hard},
        "signed_carbon_residual_propagation": carbon_propagation,
        "classification": classification,
        "integrity": integrity,
        "timing_seconds": {
            "total": float(sum(timings)),
            "minimum": float(min(timings)),
            "median": float(np.median(timings)),
            "maximum": float(max(timings)),
        },
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    attribution = load_structured_gpp_fresh_attribution_contract(args.attribution_contract)
    identity = attribution.payload["identity"]
    screen = load_structured_gpp_fresh_screen_contract(args.screen_contract)
    if screen.sha256 != identity["fresh_screen_contract_sha256"]:
        raise ValueError("fresh-attribution screen identity drift")
    formal_sha256 = sha256_file(args.fresh_formal_report)
    if formal_sha256 != identity["fresh_formal_report_sha256"]:
        raise ValueError("fresh-attribution formal report hash drift")
    formal_report = json.loads(args.fresh_formal_report.read_text(encoding="utf-8"))
    if (
        formal_report.get("status") != "completed"
        or formal_report.get("mode") != "formal"
        or formal_report.get("identity", {}).get("validation_panel_sha256") != identity["validation_panel_sha256"]
    ):
        raise ValueError("fresh-attribution formal report is ineligible")

    screen_identity = screen.payload["identity"]
    broad_report = _load_broad_training_report(args.broad_training_report, screen)
    if (
        sha256_file(args.selected_checkpoint) != identity["selected_checkpoint_sha256"]
        or sha256_file(args.distribution_training_report) != screen_identity["parent_distribution_report_sha256"]
        or args.stage_checkpoint_sha256 != screen_identity["base_attention_checkpoint_sha256"]
        or args.distribution_checkpoint_sha256 != screen_identity["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("fresh-attribution checkpoint identity drift")

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
        load_distribution_aware_checkpoint,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
        build_target_oriented_structured_gpp_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        initialize_structured_gpp_parameters,
        load_structured_gpp_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_training import (
        load_structured_gpp_broad_training_contract,
    )

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    target_contract = context["target_contract"]
    broad_contract = load_structured_gpp_broad_training_contract(args.broad_training_contract)
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    if (
        amendment.sha256 != screen_identity["flux_basis_amendment_sha256"]
        or context["training_contract"].sha256 != screen_identity["base_training_contract_sha256"]
        or target_contract.sha256 != screen_identity["target_contract_sha256"]
        or broad_contract.sha256 != screen_identity["broad_training_contract_sha256"]
        or structured_contract.sha256 != screen_identity["candidate_contract_sha256"]
    ):
        raise ValueError("fresh-attribution parent contract drift")

    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    distribution_checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=args.distribution_checkpoint_sha256,
        identity=distribution_report["identity"],
        completed_updates=250,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(distribution_report["training"]["clip_threshold"]),
    )
    initialized_parent = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=distribution_contract,
    )
    parent_parameters = _replace_adapter(initialized_parent, jax.device_put(distribution_checkpoint["adapter"]))
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("fresh-attribution parent parameter count drift")

    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(distribution_head, structured_contract=structured_contract)
    initialized = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    selected_checkpoint = load_broad_checkpoint(
        args.selected_checkpoint,
        expected_sha256=identity["selected_checkpoint_sha256"],
        identity=broad_report["identity"],
        completed_updates=int(identity["selected_checkpoint_update"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(broad_report["training"]["clip_threshold"]),
    )
    specialist = jax.device_put(selected_checkpoint["specialist"])
    parameters = replace_specialist(initialized, specialist)
    if (
        specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
        or _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
        or CANDIDATE_PARAMETER_COUNT != int(identity["candidate_parameter_count"])
    ):
        raise ValueError("fresh-attribution candidate parameter count drift")

    prepared, panel = load_target_oriented_validation_prepared(
        args.fresh_validation_prepared_manifest,
        training_contract_sha256=context["training_contract"].sha256,
        target_contract_sha256=target_contract.sha256,
        verify_artifacts=False,
    )
    consumed_panel = json.loads(args.consumed_validation_panel.read_text(encoding="utf-8"))
    verify_fresh_validation_panel(panel, runtime["index"], screen, target_contract, consumed_panel)
    if (
        prepared["manifest_sha256"] != identity["validation_prepared_manifest_sha256"]
        or panel["panel_sha256"] != identity["validation_panel_sha256"]
    ):
        raise ValueError("fresh-attribution prepared-panel identity drift")

    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    evaluator = _make_attribution_evaluator(
        runtime,
        head,
        context["template"],
        acceptance_scales,
        context["scales"],
    )
    panel_report = _evaluate_panel(
        runtime,
        parameters=parameters,
        evaluator=evaluator,
        prepared_path=args.fresh_validation_prepared_manifest,
        prepared=prepared,
        panel=panel,
        target_contract=target_contract,
        formal_report=formal_report,
        attribution_contract=attribution,
        mode=args.mode,
    )
    integrity_passed = bool(panel_report["integrity"]["passed"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if integrity_passed else "failed",
        "mode": args.mode,
        "identity": {
            "git_head": _git_head(),
            "attribution_contract_sha256": attribution.sha256,
            "fresh_screen_contract_sha256": screen.sha256,
            "fresh_formal_report_sha256": formal_sha256,
            "selected_checkpoint_sha256": sha256_file(args.selected_checkpoint),
            "selected_checkpoint_optimizer_step": int(identity["selected_checkpoint_update"]),
            "validation_prepared_manifest_sha256": prepared["manifest_sha256"],
            "validation_panel_sha256": panel["panel_sha256"],
        },
        "policy": {
            "optimizer_updates_executed": 0,
            "parameter_steps_executed": 0,
            "checkpoint_writes": 0,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "model": {
            "candidate_parameter_count": CANDIDATE_PARAMETER_COUNT,
            "trainable_specialist_parameter_count": SPECIALIST_PARAMETER_COUNT,
            "retained_parent_parameter_count": int(identity["retained_parent_parameter_count"]),
        },
        "attribution": panel_report,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": attribution.payload["next_gate"][
            "on_admission_pass"
            if integrity_passed and args.mode == "admission"
            else "on_integrity_pass"
            if integrity_passed
            else "on_integrity_failure"
        ],
    }
    _atomic_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--flux-basis-amendment", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--fresh-validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--consumed-validation-panel", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--distribution-contract", type=Path, required=True)
    parser.add_argument("--distribution-training-report", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint-sha256", required=True)
    parser.add_argument("--structured-contract", type=Path, required=True)
    parser.add_argument("--broad-training-contract", type=Path, required=True)
    parser.add_argument("--broad-training-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--attribution-contract", type=Path, required=True)
    parser.add_argument("--fresh-formal-report", type=Path, required=True)
    parser.add_argument("--mode", choices=("admission", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
