"""Run eight zero-update target-oriented full-objective gradients on one GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    PROBE_SEED,
    _load_extended_runtime,
    _tree_numeric_report,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    _atomic_json,
    _retained_adapter,
)
from scripts.hpc.run_successor_small_fit import _retained_tail_trace_signature
from scripts.hpc.run_successor_small_fit_probe import _plan_context
from scripts.hpc.run_target_oriented_attention_gpu_admission import (
    PARAMETER_COUNT,
    _gradient_family_report,
    _head,
)
from scripts.hpc.target_oriented_training_prepared import (
    load_prepared_target_artifact,
    load_prepared_target_manifest,
)


class HardCounts(NamedTuple):
    prediction_nonfinite: Any
    state_nonfinite: Any
    unexpected_state_mask_mismatch: Any
    declared_dynamic_state_mask_mismatch: Any
    discrete_mismatch: Any
    budget_failure: Any
    stock_negative: Any


def _window_hard_counts(
    prediction_nonfinite: Any,
    state_nonfinite: Any,
    unexpected_state_mismatch: Any,
    declared_dynamic_state_mismatch: Any,
    rollout: Any,
) -> HardCounts:
    return HardCounts(
        prediction_nonfinite,
        state_nonfinite,
        unexpected_state_mismatch + rollout.defined_mask_mismatch,
        declared_dynamic_state_mismatch,
        rollout.discrete_state_mismatch,
        rollout.budget_residual_failure,
        rollout.negative_declared_stock,
    )


def _hard_failure_count(counts: Mapping[str, int]) -> int:
    return sum(
        int(value)
        for name, value in counts.items()
        if name != "declared_dynamic_state_mask_mismatch"
    )


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _slice(value: Any, start: int, stop: int) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda leaf: leaf[start:stop], value)


def _validate_prepared_objective_shapes(
    artifact: Any,
    *,
    complete_schema: Any,
    scales: Any,
) -> None:
    expected_targets = {field.name for field in complete_schema.fields}
    targets = artifact.batch.one_day.process_targets
    if set(targets) != expected_targets:
        raise ValueError("prepared one-day batch does not contain the complete process target boundary")
    for name in scales.state._fields:
        expected = np.asarray(getattr(artifact.batch.one_day.expected_next_state, name))
        scale = np.asarray(getattr(scales.state, name))
        if expected.shape != (32, *scale.shape):
            raise ValueError(
                f"prepared one-day state scale shape drift for {name}: expected={expected.shape}, scale={scale.shape}"
            )
        for label, masks in (
            ("loss eligibility", scales.state_loss_eligible),
            ("declared dynamic status", scales.state_declared_dynamic),
        ):
            mask = np.asarray(getattr(masks, name))
            if mask.shape != scale.shape:
                raise ValueError(f"prepared one-day state {label} shape drift for {name}")


def _zero_hard(jnp: Any) -> HardCounts:
    zero = jnp.asarray(0, dtype=jnp.int32)
    return HardCounts(zero, zero, zero, zero, zero, zero, zero)


def _add_hard(jax: Any, left: HardCounts, right: HardCounts) -> HardCounts:
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _sum_tree(jax: Any, values: Any) -> Any:
    return sum(
        (value for value in jax.tree_util.tree_leaves(values)),
        0,
    )


def _prepare_template(runtime: Mapping[str, Any], entry: Mapping[str, Any]) -> Any:
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

    first = entry["components"]["one_day"][0]
    matches = tuple(
        reference
        for reference in runtime["index"].select(
            spatial_split="train",
            temporal_split="train",
        )
        if str(reference.parent.landpoint_id) == str(first["landpoint_id"])
        and int(reference.parent.year) == int(first["year"])
        and str(reference.parent.sha256) == str(first["parent_shard_sha256"])
        and str(reference.sha256) == str(first["typed_shard_sha256"])
    )
    if len(matches) != 1:
        raise ValueError("preflight retained template reference is ambiguous")
    paths = _plan_context(
        runtime["manifest"],
        landpoint_id=str(first["landpoint_id"]),
        year=int(first["year"]),
    )
    context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    parent = load_markov_shard(matches[0].parent.path, contract=runtime["contract"])
    sample = parent.sample(int(first["start_rows"][0]))
    return _retained_adapter(
        runtime | {"context_paths": paths},
        context,
        sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )


def _scaffold_year_invariance(
    jax: Any,
    parameters: Any,
    inputs: Any,
    day_inputs: Any,
    *,
    transition_builder: Any,
    template: Any,
    alternate_year: int,
) -> Mapping[str, Any]:
    first = transition_builder(template)(parameters, inputs, day_inputs)
    alternate = replace(
        template,
        static_owner=replace(template.static_owner, runtime_year=int(alternate_year)),
    )
    second = transition_builder(alternate)(parameters, inputs, day_inputs)
    jax.block_until_ready((first, second))
    first_leaves, first_tree = jax.tree_util.tree_flatten(
        (first.next_state, first.next_state_defined, first.next_discrete_state, first.diagnostics)
    )
    second_leaves, second_tree = jax.tree_util.tree_flatten(
        (second.next_state, second.next_state_defined, second.next_discrete_state, second.diagnostics)
    )
    if first_tree != second_tree:
        raise ValueError("retained scaffold-year comparison trees differ")
    maximum = 0.0
    exact = True
    for left, right in zip(first_leaves, second_leaves, strict=True):
        left = np.asarray(left)
        right = np.asarray(right)
        if left.dtype.kind in "fc":
            maximum = max(maximum, float(np.max(np.abs(left - right))) if left.size else 0.0)
            exact = exact and bool(np.array_equal(left, right, equal_nan=True))
        else:
            exact = exact and bool(np.array_equal(left, right))
    return {
        "passed": exact,
        "template_year": int(template.static_owner.runtime_year),
        "alternate_year": int(alternate_year),
        "maximum_absolute_difference": maximum,
    }


def _make_steps(
    runtime: Mapping[str, Any],
    head: Any,
    template: Any,
    scales: Any,
    target_contract: Any,
    *,
    acceptance_scales: Any | None = None,
    acceptance_flux_names: tuple[str, ...] = (),
    science_flux_basis: tuple[str, ...] | None = None,
    diagnostic_teacher_carry_state_key: str | None = None,
    diagnostic_teacher_carry_state_keys: tuple[str, ...] | None = None,
):
    import jax
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
        rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_output_loss import (
        masked_mechanistic_output_loss,
    )
    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        TERMINAL_STOCK_FIELDS,
        extract_biomass_productivity_trajectories,
        rollout_daily_operator,
        trajectories_from_rollout,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_maintenance_head_conditioning import (
        replace_maintenance_head,
        resolve_maintenance_head_index,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_snowheat_recurrence import (
        resolve_continuous_state_leaf,
        resolve_continuous_state_leaves,
        rollout_with_teacher_state_leaves,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        _linearized_error_loss,
        linearized_one_day_state_loss,
        linearized_process_supervision_loss,
        linearized_science_flux_loss,
        linearized_science_stock_loss,
        linearized_science_window_loss,
        one_day_state_sufficient_statistics,
        process_supervision_sufficient_statistics,
        science_window_from_sufficient_statistics,
        science_window_sufficient_statistics,
        target_oriented_acceptance_sufficient_statistics,
    )
    from research.daily_coarse_graining.daily_operator.training_loss import (
        ProcessLossResult,
    )
    from research.daily_coarse_graining.daily_operator.types import StateFeatureGroups

    dynamic = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail_template=template,
    )
    learned_output_names = tuple(field.name for field in runtime["learned_schema"].fields)
    maintenance_head_index = resolve_maintenance_head_index(runtime["learned_schema"])
    resolved_flux_basis = tuple(science_flux_basis or tuple(scales.fluxes))
    if set(resolved_flux_basis) != set(scales.fluxes):
        raise ValueError("target-oriented step flux basis and scale registry differ")
    diagnostic_flux_basis = (
        "gpp",
        "maintenance_respiration",
        "growth_respiration",
    )
    if not set(diagnostic_flux_basis) <= set(resolved_flux_basis):
        raise ValueError("target-oriented diagnostic flux basis is unavailable")
    if (
        diagnostic_teacher_carry_state_key is not None
        and diagnostic_teacher_carry_state_keys is not None
    ):
        raise ValueError("diagnostic Teacher-carry state key boundaries conflict")
    diagnostic_teacher_carry_keys = diagnostic_teacher_carry_state_keys
    if diagnostic_teacher_carry_state_key is not None:
        resolve_continuous_state_leaf(
            runtime["contract"],
            diagnostic_teacher_carry_state_key,
        )
        diagnostic_teacher_carry_keys = (diagnostic_teacher_carry_state_key,)
    if diagnostic_teacher_carry_keys is not None:
        diagnostic_teacher_carry_keys = tuple(diagnostic_teacher_carry_keys)
        resolve_continuous_state_leaves(
            runtime["contract"],
            diagnostic_teacher_carry_keys,
        )

    def state_hard(predicted, predicted_defined, expected_defined):
        nonfinite = jnp.asarray(0, dtype=jnp.int32)
        unexpected_mismatch = jnp.asarray(0, dtype=jnp.int32)
        declared_dynamic_mismatch = jnp.asarray(0, dtype=jnp.int32)
        for name in StateFeatureGroups._fields:
            value = jnp.asarray(getattr(predicted, name))
            actual = jnp.asarray(getattr(predicted_defined, name), dtype=jnp.bool_)
            wanted = jnp.asarray(getattr(expected_defined, name), dtype=jnp.bool_)
            dynamic = jnp.asarray(getattr(scales.state_declared_dynamic, name), dtype=jnp.bool_)
            local_mismatch = actual != wanted
            nonfinite = nonfinite + jnp.sum(wanted & ~jnp.isfinite(value), dtype=jnp.int32)
            unexpected_mismatch = unexpected_mismatch + jnp.sum(local_mismatch & ~dynamic, dtype=jnp.int32)
            declared_dynamic_mismatch = declared_dynamic_mismatch + jnp.sum(
                local_mismatch & dynamic,
                dtype=jnp.int32,
            )
        return nonfinite, unexpected_mismatch, declared_dynamic_mismatch

    def discrete_hard(predicted, expected):
        left, left_tree = jax.tree_util.tree_flatten(predicted)
        right, right_tree = jax.tree_util.tree_flatten(expected)
        if left_tree != right_tree:
            raise ValueError("preflight discrete-state trees differ")
        return sum(
            (jnp.sum(jnp.asarray(a) != jnp.asarray(b), dtype=jnp.int32) for a, b in zip(left, right, strict=True)),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def one_sample(parameters, inputs, targets, expected_state, expected_defined, expected_discrete, day_inputs):
        result = dynamic(parameters, inputs, day_inputs)
        assembled = runtime["adapter"].input_assembler.assemble(inputs)
        process = masked_mechanistic_output_loss(
            result.prediction.named_outputs,
            targets,
            inputs=assembled,
            complete_schema=runtime["schema"],
            learned_schema=runtime["learned_schema"],
            scales=runtime["loss_scales"],
            architecture_contract=runtime["architecture_contract"],
            mechanism_contract=runtime["mechanism_contract"],
            auxiliary_outputs=result.prediction.auxiliary_outputs,
        )
        prediction_nonfinite = sum(
            (
                jnp.sum(
                    jnp.asarray(targets[name].defined, dtype=jnp.bool_)
                    & ~jnp.isfinite(jnp.asarray(result.prediction.named_outputs[name])),
                    dtype=jnp.int32,
                )
                for name in learned_output_names
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )
        state_nonfinite, unexpected_state_mismatch, declared_dynamic_state_mismatch = state_hard(
            result.next_state,
            result.next_state_defined,
            expected_defined,
        )
        budget = jnp.sum(
            ~jnp.isfinite(
                jnp.stack(
                    (
                        result.diagnostics["water_budget_residual"],
                        result.diagnostics["carbon_budget_residual"],
                    )
                )
            )
            | (
                jnp.abs(
                    jnp.stack(
                        (
                            result.diagnostics["water_budget_residual"],
                            result.diagnostics["carbon_budget_residual"],
                        )
                    )
                )
                > 2.0e-12
            ),
            dtype=jnp.int32,
        )
        hard = HardCounts(
            prediction_nonfinite,
            state_nonfinite,
            unexpected_state_mismatch,
            declared_dynamic_state_mismatch,
            discrete_hard(result.next_discrete_state, expected_discrete),
            budget,
            jnp.asarray(0, dtype=jnp.int32),
        )
        del expected_state
        return result.next_state, result.next_state_defined, process.label_mse, process.label_count, hard

    def one_statistics(parameters, microbatch):
        predicted, predicted_defined, label_mse, label_count, hard = jax.vmap(
            one_sample,
            in_axes=(None, 0, 0, 0, 0, 0, 0),
        )(
            parameters,
            microbatch.inputs,
            microbatch.process_targets,
            microbatch.expected_next_state,
            microbatch.expected_next_state_defined,
            microbatch.expected_next_discrete_state,
            microbatch.day_inputs,
        )
        state = one_day_state_sufficient_statistics(
            predicted,
            microbatch.expected_next_state,
            predicted_defined,
            microbatch.expected_next_state_defined,
            scales.state,
            eligible=scales.state_loss_eligible,
        )
        process = process_supervision_sufficient_statistics(
            ProcessLossResult(
                total=jnp.asarray(0.0),
                label_mse=label_mse,
                label_count=label_count,
                field_mse={},
                field_count={},
            )
        )
        return state, process, jax.tree_util.tree_map(lambda value: jnp.sum(value, axis=0), hard)

    def expected_window_trajectories(microbatch, templates):
        expected_state = jax.tree_util.tree_map(lambda value: value[0], microbatch.expected_next_state)
        expected_defined = jax.tree_util.tree_map(lambda value: value[0], microbatch.expected_next_state_defined)
        expected_gpp = microbatch.expected_fast_gpp[0]
        expected_gpp_defined = microbatch.expected_fast_gpp_defined[0]
        return extract_biomass_productivity_trajectories(
            expected_state.other,
            expected_defined.other,
            daily_fast_gpp=expected_gpp,
            daily_fast_gpp_defined=expected_gpp_defined,
            pft_fraction=templates.pft.pft_fraction,
            active_pft_mask=templates.pft.active_pft_mask,
            contract=runtime["contract"],
        )

    def window_trajectories(parameters, microbatch, *, rematerialize: bool):
        templates = jax.tree_util.tree_map(lambda value: value[0], microbatch.inputs)
        day_inputs = jax.tree_util.tree_map(lambda value: value[0], microbatch.day_inputs)

        def transition(values, inputs):
            return dynamic(values, inputs, day_inputs)

        trace = rollout_daily_operator(
            parameters,
            templates,
            transition=transition,
            contract=runtime["contract"],
            rematerialize=rematerialize,
        )
        predicted = trajectories_from_rollout(trace, runtime["contract"])
        expected = expected_window_trajectories(microbatch, templates)
        expected_defined = jax.tree_util.tree_map(lambda value: value[0], microbatch.expected_next_state_defined)
        expected_discrete = jax.tree_util.tree_map(lambda value: value[0], microbatch.expected_next_discrete_state)
        stock_hard = rollout_hard_gate_counts(
            predicted.stocks,
            expected.stocks,
            nonnegative_stock_fields=TERMINAL_STOCK_FIELDS,
            budget_residuals={
                "water": trace.result.diagnostics["water_budget_residual"],
                "carbon": trace.result.diagnostics["carbon_budget_residual"],
            },
            predicted_discrete_state=trace.result.next_discrete_state,
            expected_discrete_state=expected_discrete,
            budget_atol=2.0e-12,
            stock_atol=2.0e-12,
        )
        state_nonfinite, unexpected_state_mismatch, declared_dynamic_state_mismatch = state_hard(
            trace.result.next_state,
            trace.result.next_state_defined,
            expected_defined,
        )
        prediction_nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(value)), dtype=jnp.int32)
                for name, value in trace.result.prediction.named_outputs.items()
                if name in learned_output_names
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )
        hard = _window_hard_counts(
            prediction_nonfinite,
            state_nonfinite,
            unexpected_state_mismatch,
            declared_dynamic_state_mismatch,
            stock_hard,
        )
        return predicted, expected, hard

    def teacher_state_window_trajectories(parameters, microbatch, *, rematerialize: bool):
        templates = jax.tree_util.tree_map(lambda value: value[0], microbatch.inputs)
        day_inputs = jax.tree_util.tree_map(lambda value: value[0], microbatch.day_inputs)

        def independent_day(_carry, daily_template):
            result = dynamic(parameters, daily_template, day_inputs)
            return _carry, result

        scan_day = jax.checkpoint(independent_day) if rematerialize else independent_day
        _, result = jax.lax.scan(scan_day, None, templates)
        gpp = result.prediction.named_outputs["daily.fast_gpp"]
        gpp_defined = jnp.broadcast_to(
            jnp.asarray(templates.pft.active_pft_mask, dtype=jnp.bool_)[:, None, :],
            jnp.asarray(gpp).shape,
        )
        predicted = extract_biomass_productivity_trajectories(
            result.next_state.other,
            result.next_state_defined.other,
            daily_fast_gpp=gpp,
            daily_fast_gpp_defined=gpp_defined,
            pft_fraction=templates.pft.pft_fraction,
            active_pft_mask=templates.pft.active_pft_mask,
            contract=runtime["contract"],
        )
        return predicted, expected_window_trajectories(microbatch, templates)

    def teacher_carry_window_trajectories(parameters, microbatch, *, rematerialize: bool):
        if diagnostic_teacher_carry_keys is None:
            raise ValueError("diagnostic Teacher-carry state keys are unavailable")
        templates = jax.tree_util.tree_map(lambda value: value[0], microbatch.inputs)
        day_inputs = jax.tree_util.tree_map(lambda value: value[0], microbatch.day_inputs)

        def transition(values, inputs):
            return dynamic(values, inputs, day_inputs)

        trace = rollout_with_teacher_state_leaves(
            parameters,
            templates,
            transition=transition,
            contract=runtime["contract"],
            state_keys=diagnostic_teacher_carry_keys,
            rematerialize=rematerialize,
        )
        predicted = trajectories_from_rollout(trace, runtime["contract"])
        expected = expected_window_trajectories(microbatch, templates)
        expected_defined = jax.tree_util.tree_map(
            lambda value: value[0],
            microbatch.expected_next_state_defined,
        )
        expected_discrete = jax.tree_util.tree_map(
            lambda value: value[0],
            microbatch.expected_next_discrete_state,
        )
        stock_hard = rollout_hard_gate_counts(
            predicted.stocks,
            expected.stocks,
            nonnegative_stock_fields=TERMINAL_STOCK_FIELDS,
            budget_residuals={
                "water": trace.result.diagnostics["water_budget_residual"],
                "carbon": trace.result.diagnostics["carbon_budget_residual"],
            },
            predicted_discrete_state=trace.result.next_discrete_state,
            expected_discrete_state=expected_discrete,
            budget_atol=2.0e-12,
            stock_atol=2.0e-12,
        )
        state_nonfinite, unexpected_state_mismatch, declared_dynamic_state_mismatch = state_hard(
            trace.result.next_state,
            trace.result.next_state_defined,
            expected_defined,
        )
        prediction_nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(value)), dtype=jnp.int32)
                for name, value in trace.result.prediction.named_outputs.items()
                if name in learned_output_names
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )
        hard = _window_hard_counts(
            prediction_nonfinite,
            state_nonfinite,
            unexpected_state_mismatch,
            declared_dynamic_state_mismatch,
            stock_hard,
        )
        return predicted, expected, hard

    def window_statistics(parameters, microbatch, *, horizon: int, rematerialize: bool):
        predicted, expected, hard = window_trajectories(
            parameters,
            microbatch,
            rematerialize=rematerialize,
        )
        statistics = science_window_sufficient_statistics(
            predicted.fluxes,
            expected.fluxes,
            predicted.stocks,
            expected.stocks,
            flux_scales=scales.fluxes,
            stock_scales=scales.stocks,
            horizon=horizon,
            flux_basis=resolved_flux_basis,
        )
        return statistics, hard

    def window_acceptance_statistics(
        parameters,
        microbatch,
        *,
        horizon: int,
        rematerialize: bool,
    ):
        if acceptance_scales is None:
            raise ValueError("acceptance scales are required for flux-gradient diagnostics")
        predicted, expected, hard = window_trajectories(
            parameters,
            microbatch,
            rematerialize=rematerialize,
        )
        statistics = target_oriented_acceptance_sufficient_statistics(
            predicted.fluxes,
            expected.fluxes,
            predicted.stocks,
            expected.stocks,
            scales=acceptance_scales,
            horizon=horizon,
        )
        return statistics, hard

    def teacher_state_window_statistics(
        parameters,
        microbatch,
        *,
        horizon: int,
        rematerialize: bool,
    ):
        predicted, expected = teacher_state_window_trajectories(
            parameters,
            microbatch,
            rematerialize=rematerialize,
        )
        return science_window_sufficient_statistics(
            predicted.fluxes,
            expected.fluxes,
            predicted.stocks,
            expected.stocks,
            flux_scales=scales.fluxes,
            stock_scales=scales.stocks,
            horizon=horizon,
            flux_basis=resolved_flux_basis,
        )

    def teacher_carry_window_statistics(
        parameters,
        microbatch,
        *,
        horizon: int,
        rematerialize: bool,
    ):
        predicted, expected, hard = teacher_carry_window_trajectories(
            parameters,
            microbatch,
            rematerialize=rematerialize,
        )
        statistics = science_window_sufficient_statistics(
            predicted.fluxes,
            expected.fluxes,
            predicted.stocks,
            expected.stocks,
            flux_scales=scales.fluxes,
            stock_scales=scales.stocks,
            horizon=horizon,
            flux_basis=resolved_flux_basis,
        )
        return statistics, hard

    one_forward = jax.jit(one_statistics)
    seven_forward = jax.jit(
        lambda parameters, batch: window_statistics(parameters, batch, horizon=7, rematerialize=False)
    )
    thirty_forward = jax.jit(
        lambda parameters, batch: window_statistics(parameters, batch, horizon=30, rematerialize=False)
    )

    def one_linear(parameters, batch, total_state, total_process):
        state, process, _hard = one_statistics(parameters, batch)
        return linearized_one_day_state_loss(state, total_state) + target_contract.loss_coefficients[
            "necessary_process_supervision"
        ] * linearized_process_supervision_loss(
            process,
            total_process,
            target_contract,
        )

    def window_linear(parameters, batch, total, *, horizon: int):
        current, _hard = window_statistics(
            parameters,
            batch,
            horizon=horizon,
            rematerialize=True,
        )
        return linearized_science_window_loss(
            current,
            total,
            flux_basis=resolved_flux_basis,
        )

    def window_flux_linear(parameters, batch, total, *, horizon: int):
        current, _hard = window_statistics(
            parameters,
            batch,
            horizon=horizon,
            rematerialize=True,
        )
        return linearized_science_flux_loss(
            current,
            total,
            flux_basis=resolved_flux_basis,
        )

    def window_stock_linear(parameters, batch, total, *, horizon: int):
        current, _hard = window_statistics(
            parameters,
            batch,
            horizon=horizon,
            rematerialize=True,
        )
        return linearized_science_stock_loss(current, total)

    def maintenance_head_linear(
        head_parameters,
        parameters,
        batch,
        total,
        *,
        horizon: int,
    ):
        proposed = replace_maintenance_head(
            parameters,
            maintenance_head_index,
            head_parameters,
        )
        current, _hard = window_statistics(
            proposed,
            batch,
            horizon=horizon,
            rematerialize=True,
        )
        return _linearized_error_loss(
            current.fluxes["maintenance_respiration"],
            total.fluxes["maintenance_respiration"],
            mse_weight=0.5,
            bias_squared_weight=0.5,
        )

    def standalone_window_flux_score(parameters, batch, *, horizon: int):
        current, _hard = window_statistics(
            parameters,
            batch,
            horizon=horizon,
            rematerialize=True,
        )
        result = science_window_from_sufficient_statistics(
            current,
            flux_basis=resolved_flux_basis,
        )
        return 0.25 * (result.flux_mse + result.flux_bias_squared)

    def field_score_vector(statistics):
        scores = []
        for name in diagnostic_flux_basis:
            current = statistics.fluxes[name]
            count = jnp.asarray(current.count, dtype=jnp.float64)
            denominator = jnp.maximum(count, 1.0)
            mse = jnp.asarray(current.sum_squared, dtype=jnp.float64) / denominator
            bias = jnp.asarray(current.sum_signed, dtype=jnp.float64) / denominator
            scores.append(jnp.where(count > 0.0, 0.5 * (mse + bias * bias), 0.0))
        return jnp.stack(tuple(scores))

    def standalone_window_flux_field_score(parameters, batch, weights, *, horizon: int):
        current, _hard = window_statistics(
            parameters,
            batch,
            horizon=horizon,
            rematerialize=True,
        )
        return jnp.vdot(
            jnp.asarray(weights, dtype=jnp.float64),
            field_score_vector(current),
        )

    def teacher_state_window_flux_field_score(parameters, batch, weights, *, horizon: int):
        current = teacher_state_window_statistics(
            parameters,
            batch,
            horizon=horizon,
            rematerialize=True,
        )
        return jnp.vdot(
            jnp.asarray(weights, dtype=jnp.float64),
            field_score_vector(current),
        )

    def teacher_carry_window_flux_field_score(parameters, batch, weights, *, horizon: int):
        current, hard = teacher_carry_window_statistics(
            parameters,
            batch,
            horizon=horizon,
            rematerialize=True,
        )
        score = jnp.vdot(
            jnp.asarray(weights, dtype=jnp.float64),
            field_score_vector(current),
        )
        return score, hard

    one_gradient = jax.jit(jax.value_and_grad(one_linear))
    seven_gradient = jax.jit(
        jax.value_and_grad(lambda parameters, batch, total: window_linear(parameters, batch, total, horizon=7))
    )
    thirty_gradient = jax.jit(
        jax.value_and_grad(lambda parameters, batch, total: window_linear(parameters, batch, total, horizon=30))
    )
    result = {
        "transition_builder": lambda retained: (
            lambda values, inputs, day_inputs: bind_daily_operator_dynamic_retained_inputs(
                input_assembler=runtime["adapter"].input_assembler,
                process_head=head,
                retained_tail_template=retained,
            )(values, inputs, day_inputs)
        ),
        "one_forward": one_forward,
        "seven_forward": seven_forward,
        "thirty_forward": thirty_forward,
        "one_gradient": one_gradient,
        "seven_gradient": seven_gradient,
        "thirty_gradient": thirty_gradient,
        "seven_flux_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch, total: window_flux_linear(
                    parameters,
                    batch,
                    total,
                    horizon=7,
                )
            )
        ),
        "seven_stock_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch, total: window_stock_linear(
                    parameters,
                    batch,
                    total,
                    horizon=7,
                )
            )
        ),
        "thirty_flux_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch, total: window_flux_linear(
                    parameters,
                    batch,
                    total,
                    horizon=30,
                )
            )
        ),
        "thirty_stock_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch, total: window_stock_linear(
                    parameters,
                    batch,
                    total,
                    horizon=30,
                )
            )
        ),
        "seven_maintenance_head_gradient": jax.jit(
            jax.value_and_grad(
                lambda head_parameters, parameters, batch, total: maintenance_head_linear(
                    head_parameters,
                    parameters,
                    batch,
                    total,
                    horizon=7,
                )
            )
        ),
        "thirty_maintenance_head_gradient": jax.jit(
            jax.value_and_grad(
                lambda head_parameters, parameters, batch, total: maintenance_head_linear(
                    head_parameters,
                    parameters,
                    batch,
                    total,
                    horizon=30,
                )
            )
        ),
        "seven_flux_window_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch: standalone_window_flux_score(
                    parameters,
                    batch,
                    horizon=7,
                )
            )
        ),
        "thirty_flux_window_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch: standalone_window_flux_score(
                    parameters,
                    batch,
                    horizon=30,
                )
            )
        ),
        "seven_flux_field_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch, weights: standalone_window_flux_field_score(
                    parameters,
                    batch,
                    weights,
                    horizon=7,
                )
            )
        ),
        "thirty_flux_field_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch, weights: standalone_window_flux_field_score(
                    parameters,
                    batch,
                    weights,
                    horizon=30,
                )
            )
        ),
        "seven_teacher_state_flux_field_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch, weights: teacher_state_window_flux_field_score(
                    parameters,
                    batch,
                    weights,
                    horizon=7,
                )
            )
        ),
        "thirty_teacher_state_flux_field_gradient": jax.jit(
            jax.value_and_grad(
                lambda parameters, batch, weights: teacher_state_window_flux_field_score(
                    parameters,
                    batch,
                    weights,
                    horizon=30,
                )
            )
        ),
    }
    if diagnostic_teacher_carry_keys is not None:
        result.update(
            {
                "seven_teacher_carry_flux_field_gradient": jax.jit(
                    jax.value_and_grad(
                        lambda parameters, batch, weights: teacher_carry_window_flux_field_score(
                            parameters,
                            batch,
                            weights,
                            horizon=7,
                        ),
                        has_aux=True,
                    )
                ),
                "thirty_teacher_carry_flux_field_gradient": jax.jit(
                    jax.value_and_grad(
                        lambda parameters, batch, weights: teacher_carry_window_flux_field_score(
                            parameters,
                            batch,
                            weights,
                            horizon=30,
                        ),
                        has_aux=True,
                    )
                ),
            }
        )
    if acceptance_scales is not None:
        if not acceptance_flux_names or not set(acceptance_flux_names) <= set(acceptance_scales.fluxes):
            raise ValueError("acceptance flux-gradient registry is invalid")

        def acceptance_flux_linear(parameters, batch, total, weights, *, horizon: int):
            current, _hard = window_acceptance_statistics(
                parameters,
                batch,
                horizon=horizon,
                rematerialize=True,
            )
            if jnp.asarray(weights).shape != (len(acceptance_flux_names),):
                raise ValueError("acceptance flux-gradient weight shape drift")
            value = jnp.asarray(0.0, dtype=jnp.float64)
            for index, name in enumerate(acceptance_flux_names):
                value = value + jnp.asarray(weights[index], dtype=jnp.float64) * _linearized_error_loss(
                    current.fluxes[name],
                    total.fluxes[name],
                    mse_weight=0.5,
                    bias_squared_weight=0.5,
                )
            return value

        result.update(
            {
                "seven_acceptance_forward": jax.jit(
                    lambda parameters, batch: window_acceptance_statistics(
                        parameters,
                        batch,
                        horizon=7,
                        rematerialize=False,
                    )
                ),
                "thirty_acceptance_forward": jax.jit(
                    lambda parameters, batch: window_acceptance_statistics(
                        parameters,
                        batch,
                        horizon=30,
                        rematerialize=False,
                    )
                ),
                "thirty_acceptance_gradient": jax.jit(
                    jax.value_and_grad(
                        lambda parameters, batch, total, weights: acceptance_flux_linear(
                            parameters,
                            batch,
                            total,
                            weights,
                            horizon=30,
                        )
                    )
                ),
            }
        )
    return result


def _merge_statistics(add: Any, values: list[Any]) -> Any:
    if not values:
        raise ValueError("cannot merge empty target-oriented statistics")
    result = values[0]
    for value in values[1:]:
        result = add(result, value)
    return result


def _effective_gradient(
    jax: Any,
    jnp: Any,
    parameters: Any,
    batch: Any,
    steps: Mapping[str, Any],
    target_contract: Any,
    *,
    science_flux_basis: Any | None = None,
) -> tuple[
    Any,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    HardCounts,
    Mapping[str, HardCounts],
    float,
    float,
]:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        add_science_window_sufficient_statistics,
        merge_error_sufficient_statistics,
        necessary_process_supervision_from_sufficient_statistics,
        one_day_state_anchor_from_sufficient_statistics,
        science_window_from_sufficient_statistics,
    )

    started = time.perf_counter()
    one_batches = [_slice(batch.one_day, start, start + 4) for start in range(0, 32, 4)]
    seven_batches = [_slice(batch.seven_day, index, index + 1) for index in range(8)]
    thirty_batches = [_slice(batch.thirty_day, index, index + 1) for index in range(4)]
    one_values = [steps["one_forward"](parameters, value) for value in one_batches]
    seven_values = [steps["seven_forward"](parameters, value) for value in seven_batches]
    thirty_values = [steps["thirty_forward"](parameters, value) for value in thirty_batches]
    jax.block_until_ready((one_values, seven_values, thirty_values))
    forward_seconds = time.perf_counter() - started
    total_state = merge_error_sufficient_statistics([value[0] for value in one_values])
    total_process = merge_error_sufficient_statistics([value[1] for value in one_values])
    total_seven = _merge_statistics(
        add_science_window_sufficient_statistics,
        [value[0] for value in seven_values],
    )
    total_thirty = _merge_statistics(
        add_science_window_sufficient_statistics,
        [value[0] for value in thirty_values],
    )
    hard_sources = {}
    for name, values in (
        ("one_day", one_values),
        ("seven_day", seven_values),
        ("thirty_day", thirty_values),
    ):
        source_hard = _zero_hard(jnp)
        for value in values:
            source_hard = _add_hard(jax, source_hard, value[-1])
        hard_sources[name] = source_hard
    hard = _zero_hard(jnp)
    for source_hard in hard_sources.values():
        hard = _add_hard(jax, hard, source_hard)
    state_result = one_day_state_anchor_from_sufficient_statistics(total_state)
    process_result = necessary_process_supervision_from_sufficient_statistics(
        total_process,
        target_contract,
    )
    seven_result = science_window_from_sufficient_statistics(
        total_seven,
        flux_basis=science_flux_basis,
    )
    thirty_result = science_window_from_sufficient_statistics(
        total_thirty,
        flux_basis=science_flux_basis,
    )
    objective = {
        "one_day_state_anchor": state_result.total,
        "seven_day_science": seven_result.total,
        "thirty_day_science": thirty_result.total,
        "necessary_process_supervision": process_result,
    }
    objective_details = {
        "one_day_state_anchor": {
            "group_loss": state_result.group_loss,
            "group_mse": state_result.group_mse,
            "group_bias_squared": state_result.group_bias_squared,
            "group_count": state_result.group_count,
        },
        "seven_day_science": {
            "flux_mse": seven_result.flux_mse,
            "flux_bias_squared": seven_result.flux_bias_squared,
            "stock_mse": seven_result.stock_mse,
            "stock_bias_squared": seven_result.stock_bias_squared,
        },
        "thirty_day_science": {
            "flux_mse": thirty_result.flux_mse,
            "flux_bias_squared": thirty_result.flux_bias_squared,
            "stock_mse": thirty_result.stock_mse,
            "stock_bias_squared": thirty_result.stock_bias_squared,
        },
    }
    gradient_started = time.perf_counter()

    def zero_gradient():
        return jax.tree_util.tree_map(lambda value: jnp.zeros_like(value, dtype=jnp.float32), parameters)

    def add_gradient(current, gradient):
        return jax.tree_util.tree_map(
            lambda left, right: left + jnp.asarray(right, dtype=jnp.float32),
            current,
            gradient,
        )

    one_accumulated = zero_gradient()
    source_started = time.perf_counter()
    for microbatch in one_batches:
        _value, gradient = steps["one_gradient"](
            parameters,
            microbatch,
            total_state,
            total_process,
        )
        one_accumulated = add_gradient(one_accumulated, gradient)
    jax.block_until_ready(one_accumulated)
    print(
        "target_full_step_preflight gradient_source=one_day_and_process "
        f"seconds={time.perf_counter() - source_started:.3f}",
        flush=True,
    )
    seven_accumulated = zero_gradient()
    source_started = time.perf_counter()
    for microbatch in seven_batches:
        _value, gradient = steps["seven_gradient"](
            parameters,
            microbatch,
            total_seven,
        )
        seven_accumulated = add_gradient(seven_accumulated, gradient)
    jax.block_until_ready(seven_accumulated)
    print(
        "target_full_step_preflight gradient_source=seven_day_science "
        f"seconds={time.perf_counter() - source_started:.3f}",
        flush=True,
    )
    thirty_accumulated = zero_gradient()
    source_started = time.perf_counter()
    for microbatch in thirty_batches:
        _value, gradient = steps["thirty_gradient"](
            parameters,
            microbatch,
            total_thirty,
        )
        thirty_accumulated = add_gradient(thirty_accumulated, gradient)
    jax.block_until_ready(thirty_accumulated)
    print(
        "target_full_step_preflight gradient_source=thirty_day_science "
        f"seconds={time.perf_counter() - source_started:.3f}",
        flush=True,
    )
    gradient_sources = {
        "one_day_and_process": one_accumulated,
        "seven_day_science": seven_accumulated,
        "thirty_day_science": thirty_accumulated,
    }
    accumulated = jax.tree_util.tree_map(
        lambda one, seven, thirty: one + seven + thirty,
        one_accumulated,
        seven_accumulated,
        thirty_accumulated,
    )
    jax.block_until_ready((accumulated, gradient_sources))
    gradient_seconds = time.perf_counter() - gradient_started
    return (
        accumulated,
        gradient_sources,
        objective,
        objective_details,
        hard,
        hard_sources,
        forward_seconds,
        gradient_seconds,
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_attention import (
        initialize_target_oriented_attention_parameters,
        target_oriented_attention_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        compose_target_oriented_loss,
        load_target_oriented_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        derive_target_oriented_scales,
        initialize_target_oriented_adamw,
        load_target_oriented_state_anchor_policy,
        load_target_oriented_training_contract,
        target_oriented_adamw_update,
        target_oriented_learning_rate,
    )

    if int(args.updates) != 8:
        raise ValueError("target-oriented clipping preflight requires exactly eight batches")
    batch_count = int(args.batches)
    if batch_count not in (1, 8):
        raise ValueError("target-oriented full-step execution requires one diagnostic or eight calibration batches")
    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    target_contract = load_target_oriented_contract()
    training_contract = load_target_oriented_training_contract(args.training_contract)
    state_anchor_policy = load_target_oriented_state_anchor_policy()
    learned_schema = build_mechanistic_learned_schema(runtime["schema"], mechanism)
    layout = build_mechanistic_output_layout(
        adapter=runtime["adapter"],
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        owner_contract=runtime["owner_contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        inputs=runtime["probe_assembled"],
    )
    runtime = dict(runtime) | {
        "architecture_contract": architecture,
        "mechanism_contract": mechanism,
        "learned_schema": learned_schema,
        "target_layout": layout,
    }
    head = _head(runtime, target_contract)
    parameters = initialize_target_oriented_attention_parameters(
        jax.random.PRNGKey(PROBE_SEED),
        schema=learned_schema,
        layout=layout,
        inputs=runtime["probe_assembled"],
        architecture_contract=architecture,
        target_contract=target_contract,
    )
    if target_oriented_attention_parameter_count(parameters) != PARAMETER_COUNT:
        raise ValueError("target-oriented preflight parameter count drift")
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_target_oriented_training_schedule(
        references,
        training_contract,
        updates=8,
    )
    source_head = _git_head()
    prepared_identity = json.loads(args.prepared_manifest.read_text(encoding="utf-8")).get("identity", {})
    prepared_source_head = str(prepared_identity.get("source_git_head", ""))
    prepared_manifest = load_prepared_target_manifest(
        args.prepared_manifest,
        training_contract_sha256=training_contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        updates=8,
        source_git_head=prepared_source_head,
    )
    template = _prepare_template(runtime, schedule["entries"][0])
    if template.static_owner.include_modelout_diagnostics:
        template = replace(
            template,
            static_owner=replace(
                template.static_owner,
                include_modelout_diagnostics=False,
            ),
        )
    scales = derive_target_oriented_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
        state_anchor_policy=state_anchor_policy,
    )
    first_artifact = load_prepared_target_artifact(
        args.prepared_manifest,
        prepared_manifest["entries"][0],
        training_contract_sha256=training_contract.sha256,
        schedule_entry=schedule["entries"][0],
    )
    _validate_prepared_objective_shapes(
        first_artifact,
        complete_schema=runtime["schema"],
        scales=scales,
    )
    steps = _make_steps(runtime, head, template, scales, target_contract)
    if first_artifact.trace_signature != _retained_tail_trace_signature(template):
        raise ValueError("prepared and live retained executable structures differ")
    first_input = jax.tree_util.tree_map(lambda value: value[0], first_artifact.batch.one_day.inputs)
    first_day_inputs = jax.tree_util.tree_map(lambda value: value[0], first_artifact.batch.one_day.day_inputs)
    alternate_year = int(schedule["entries"][1]["runtime_year"])
    scaffold = _scaffold_year_invariance(
        jax,
        parameters,
        first_input,
        first_day_inputs,
        transition_builder=steps["transition_builder"],
        template=template,
        alternate_year=alternate_year,
    )
    if not scaffold["passed"]:
        raise ValueError("retained scaffold year changes the scientific transition")

    device = runtime["devices"][0]
    memory_before = _memory_report(device)
    records = []
    norms = []
    for index, (entry, prepared_record) in enumerate(
        zip(
            schedule["entries"][:batch_count],
            prepared_manifest["entries"][:batch_count],
            strict=True,
        ),
        start=1,
    ):
        artifact = (
            first_artifact
            if index == 1
            else load_prepared_target_artifact(
                args.prepared_manifest,
                prepared_record,
                training_contract_sha256=training_contract.sha256,
                schedule_entry=entry,
            )
        )
        started = time.perf_counter()
        (
            gradient,
            gradient_sources,
            components,
            component_details,
            hard,
            hard_sources,
            forward_seconds,
            gradient_seconds,
        ) = _effective_gradient(jax, jnp, parameters, artifact.batch, steps, target_contract)
        host_gradient = jax.device_get(gradient)
        gradient_report = _tree_numeric_report(jax, host_gradient)
        gradient_families = _gradient_family_report(jax, host_gradient)
        gradient_source_reports = {}
        for name, source_gradient in gradient_sources.items():
            host_source = jax.device_get(source_gradient)
            gradient_source_reports[name] = {
                "gradient": _tree_numeric_report(jax, host_source),
                "gradient_families": _gradient_family_report(jax, host_source),
            }
        norm = float(gradient_report["l2_norm"])
        norms.append(norm)
        hard_host = {name: int(np.asarray(value)) for name, value in hard._asdict().items()}
        hard_source_host = {
            source: {name: int(np.asarray(value)) for name, value in counts._asdict().items()}
            for source, counts in hard_sources.items()
        }
        component_values = {
            name: value.total if hasattr(value, "total") else value for name, value in components.items()
        }
        component_detail_values = jax.tree_util.tree_map(
            lambda value: float(np.asarray(value)),
            jax.device_get(component_details),
        )
        total = compose_target_oriented_loss(
            one_day_state_anchor_value=component_values["one_day_state_anchor"],
            seven_day_science=component_values["seven_day_science"],
            thirty_day_science=component_values["thirty_day_science"],
            necessary_process_supervision=component_values["necessary_process_supervision"],
            contract=target_contract,
        )
        records.append(
            {
                "update": index,
                "runtime_year": int(entry["runtime_year"]),
                "loss": float(np.asarray(total.total)),
                "components": {name: float(np.asarray(value)) for name, value in component_values.items()},
                "unclipped_gradient_norm": norm,
                "gradient": gradient_report,
                "gradient_families": gradient_families,
                "gradient_sources": gradient_source_reports,
                "hard_counts": hard_host,
                "hard_count_sources": hard_source_host,
                "component_details": component_detail_values,
                "statistics_forward_seconds": forward_seconds,
                "microbatch_gradient_seconds": gradient_seconds,
                "effective_gradient_seconds": forward_seconds + gradient_seconds,
                "total_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"target_full_step_preflight batch={index} norm={norm:.9g} seconds={records[-1]['total_seconds']:.3f}",
            flush=True,
        )
    threshold = float(np.percentile(np.asarray(norms, dtype=np.float64), 90.0))
    optimizer = initialize_target_oriented_adamw(parameters)
    jax.block_until_ready(optimizer)
    learning_rate = target_oriented_learning_rate(training_contract, 1)

    @jax.jit
    def optimizer_proposal(values, derivatives, state):
        updated, next_state, raw_norm, clip_scale = target_oriented_adamw_update(
            values,
            derivatives,
            state,
            learning_rate=learning_rate,
            clip_norm=threshold,
            contract=training_contract,
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int32)
                for leaf in jax.tree_util.tree_leaves((updated, next_state))
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )
        step_squared = sum(
            (
                jnp.sum(
                    jnp.square(jnp.asarray(after) - jnp.asarray(before)),
                    dtype=jnp.float32,
                )
                for before, after in zip(
                    jax.tree_util.tree_leaves(values),
                    jax.tree_util.tree_leaves(updated),
                    strict=True,
                )
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return (
            updated,
            next_state,
            raw_norm,
            clip_scale,
            nonfinite,
            jnp.sqrt(step_squared),
        )

    optimizer_cold_started = time.perf_counter()
    proposal = optimizer_proposal(parameters, gradient, optimizer)
    jax.block_until_ready(proposal)
    optimizer_cold_seconds = time.perf_counter() - optimizer_cold_started
    optimizer_hot_started = time.perf_counter()
    proposal = optimizer_proposal(parameters, gradient, optimizer)
    jax.block_until_ready(proposal)
    optimizer_hot_seconds = time.perf_counter() - optimizer_hot_started
    optimizer_report = {
        "learning_rate": learning_rate,
        "raw_gradient_norm": float(np.asarray(proposal[2])),
        "clip_threshold": threshold,
        "clip_scale": float(np.asarray(proposal[3])),
        "nonfinite_parameter_or_state_count": int(np.asarray(proposal[4])),
        "parameter_step_norm": float(np.asarray(proposal[5])),
        "cold_compile_and_proposal_seconds": optimizer_cold_seconds,
        "hot_proposal_seconds": optimizer_hot_seconds,
        "proposal_retained": False,
    }
    memory_after = _memory_report(device)
    maximum_live = max(
        int(memory_before.get("peak_bytes_in_use", 0)),
        int(memory_after.get("peak_bytes_in_use", 0)),
    )
    required_gradient_families = (
        "base_source_encoder",
        "domain_refinement",
        "science_cross_attention",
        "science_token_mixer",
        "primary_heads",
        "state_heads",
        "auxiliary_heads",
    )
    hot_gradient_seconds = [record["effective_gradient_seconds"] for record in records[1:]]
    if not hot_gradient_seconds:
        hot_gradient_seconds = [record["effective_gradient_seconds"] for record in records]
    hot_full_step_seconds = [value + optimizer_hot_seconds for value in hot_gradient_seconds]
    passed = (
        scaffold["passed"]
        and all(record["gradient"]["all_finite"] for record in records)
        and all(record["unclipped_gradient_norm"] > 0.0 for record in records)
        and all(
            np.isfinite(record["loss"]) and all(np.isfinite(value) for value in record["components"].values())
            for record in records
        )
        and all(
            record["gradient_families"][name]["all_finite"] and record["gradient_families"][name]["l2_norm"] > 0.0
            for record in records
            for name in required_gradient_families
        )
        and all(_hard_failure_count(record["hard_counts"]) == 0 for record in records)
        and np.isfinite(threshold)
        and threshold > 0.0
        and optimizer_report["nonfinite_parameter_or_state_count"] == 0
        and optimizer_report["parameter_step_norm"] > 0.0
        and maximum_live > 0
        and maximum_live <= int(training_contract.payload["preflight"]["actual_peak_bytes_in_use_maximum"])
    )
    report = {
        "schema_version": (
            "target_oriented_full_step_preflight_v1"
            if batch_count == 8
            else "target_oriented_representative_gradient_probe_v1"
        ),
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": source_head,
            "prepared_source_git_head": prepared_source_head,
            "training_contract_sha256": training_contract.sha256,
            "state_anchor_policy_sha256": state_anchor_policy.sha256,
            "schedule_sha256": schedule["schedule_sha256"],
        },
        "policy": {
            "training_updates_retained": 0,
            "checkpoint_written": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "evaluated_batch_count": batch_count,
            "gradient_clip_calibration_complete": batch_count == 8,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "scaffold_year_invariance": scaffold,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "maximum_observed_peak_bytes_in_use": maximum_live,
        "optimizer_proposal": optimizer_report,
        "hot_full_effective_step_seconds": {
            "sample_count": len(hot_full_step_seconds),
            "minimum": float(np.min(hot_full_step_seconds)),
            "median": float(np.median(hot_full_step_seconds)),
            "maximum": float(np.max(hot_full_step_seconds)),
        },
        "gradient_norms": norms,
        "gradient_clip_p90": threshold,
        "batches": records,
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
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
