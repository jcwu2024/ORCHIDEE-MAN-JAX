"""Test whether the localized Day-5 GPP jump is carried by snowheat."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
    rollout_hard_gate_counts,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.successor_training import (
    TERMINAL_STOCK_FIELDS,
    DailyOperatorRolloutTrace,
    extract_biomass_productivity_trajectories,
    rollout_input_from_template,
    trajectories_from_rollout,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_localized_jump import (
    MATERIAL_NORMALIZED_STATE_CHANGE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_snowheat_causal import (
    SCHEMA_VERSION,
    TARGET_ARTIFACT_UPDATE,
    TARGET_DAY_INDEX,
    TARGET_WINDOW_INDEX,
    classify_snowheat_day5_intervention,
    factor_log_attribution,
    resolve_coupled_gpp_snowheat_sources,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import GPP_FIELD
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    science_window_from_sufficient_statistics,
    science_window_sufficient_statistics,
)
from research.daily_coarse_graining.daily_operator.types import StateFeatureGroups
from scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump import (
    FORWARD_REPRODUCTION_ATOL,
    FORWARD_REPRODUCTION_RTOL,
    _make_trace_evaluator,
    reconstruct_localized_jump_parameters,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import _science_record
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _parser as _persistent_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import _git_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _hard_failure_count,
    _slice,
    _window_hard_counts,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import _hard_host

LOCALIZED_RESULT_SHA256 = "cdca61cdcdf88ff0fd1fc8a07cfe07ead7180438844b7618c3f81f94281f618e"
FACTORIZATION_RTOL = 1.0e-7
FACTORIZATION_ATOL = 1.0e-10


def _load_localized_result(path: Path) -> Mapping[str, Any]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        digest != LOCALIZED_RESULT_SHA256
        or report.get("schema_version") != "target_oriented_coupled_gpp_localized_jump_trace_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "continuous_state_change_first"
        or int(report.get("daily_trace", {}).get("selected_window_index", -1)) != TARGET_WINDOW_INDEX
        or int(report.get("daily_trace", {}).get("first_material_day_index", -1)) != 1
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
    ):
        raise ValueError("coupled-GPP localized-jump parent result drift")
    return report


def _make_matched_snowheat_evaluator(
    reconstruction: Any,
    leaves: tuple[Any, ...],
) -> Any:
    runtime = reconstruction.runtime
    jax = reconstruction.jax
    jnp = reconstruction.jnp

    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )

    dynamic = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=reconstruction.bootstrap["head"],
        retained_tail_template=reconstruction.context["template"],
    )
    state_mask = jnp.zeros((runtime["contract"].continuous_state_width,), dtype=jnp.bool_)
    for leaf in leaves:
        state_mask = state_mask.at[int(leaf.start) : int(leaf.stop)].set(True)

    @jax.jit
    def evaluate(parameters: Any, microbatch: Any, control_day_starts: Any) -> Any:
        templates = jax.tree_util.tree_map(lambda value: value[0], microbatch.inputs)
        day_inputs = jax.tree_util.tree_map(lambda value: value[0], microbatch.day_inputs)
        initial_discrete = jax.tree_util.tree_map(
            lambda value: jnp.asarray(value)[0],
            templates.discrete_state,
        )

        def step(carry: Any, values: Any) -> Any:
            template, control_state = values
            intervened = jnp.where(state_mask, control_state, carry[0])
            current = rollout_input_from_template(
                template,
                intervened,
                carry[1],
                runtime["contract"],
            )
            result = dynamic(parameters, current, day_inputs)
            trace = DailyOperatorRolloutTrace(
                result=result,
                pft_fraction=current.pft.pft_fraction,
                active_pft_mask=current.pft.active_pft_mask,
            )
            return (result.next_state.other, result.next_discrete_state), trace

        _, trace = jax.lax.scan(
            step,
            (jnp.asarray(templates.continuous_state)[0], initial_discrete),
            (templates, control_day_starts),
        )
        return trace, trajectories_from_rollout(trace, runtime["contract"])

    return evaluate


def _scenario_day_starts(
    reconstruction: Any,
    microbatch: Any,
    trace: Any,
    *,
    state_mask: Any | None = None,
    reference: Any | None = None,
) -> tuple[Any, Any, Any]:
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    templates = jax.tree_util.tree_map(lambda value: value[0], microbatch.inputs)
    continuous = jnp.concatenate(
        (
            jnp.asarray(templates.continuous_state)[:1],
            jnp.asarray(trace.result.next_state.other)[:-1],
        ),
        axis=0,
    )
    if state_mask is not None:
        if reference is None:
            raise ValueError("matched snowheat day starts require a reference")
        continuous = jnp.where(jnp.asarray(state_mask)[None, :], reference, continuous)
    initial_discrete = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)[0],
        templates.discrete_state,
    )
    discrete = jax.tree_util.tree_map(
        lambda initial, following: jnp.concatenate((initial[None], jnp.asarray(following)[:-1]), axis=0),
        initial_discrete,
        trace.result.next_discrete_state,
    )
    return templates, continuous, discrete


def _make_factor_evaluator(reconstruction: Any) -> Any:
    jax = reconstruction.jax
    head = reconstruction.bootstrap["head"]
    assembler = reconstruction.runtime["adapter"].input_assembler
    contract = reconstruction.runtime["contract"]

    @jax.jit
    def evaluate(parameters: Any, templates: Any, continuous: Any, discrete: Any) -> Any:
        def one(template: Any, state: Any, state_discrete: Any) -> Any:
            current = rollout_input_from_template(template, state, state_discrete, contract)
            assembled = assembler.assemble(current)
            context = head.assemble_context(parameters, assembled)
            return head.gpp_factorization(parameters, context, assembled)

        return jax.vmap(one)(templates, continuous, discrete)

    return evaluate


def _make_score_evaluator(reconstruction: Any) -> Any:
    runtime = reconstruction.runtime
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    scales = reconstruction.context["scales"]
    flux_basis = reconstruction.bootstrap["flux_basis"]
    learned_output_names = frozenset(field.name for field in runtime["learned_schema"].fields)

    @jax.jit
    def evaluate(trace: Any, microbatch: Any) -> Any:
        templates = jax.tree_util.tree_map(lambda value: value[0], microbatch.inputs)
        predicted = trajectories_from_rollout(trace, runtime["contract"])
        expected_state = jax.tree_util.tree_map(lambda value: value[0], microbatch.expected_next_state)
        expected_defined = jax.tree_util.tree_map(
            lambda value: value[0],
            microbatch.expected_next_state_defined,
        )
        expected = extract_biomass_productivity_trajectories(
            expected_state.other,
            expected_defined.other,
            daily_fast_gpp=microbatch.expected_fast_gpp[0],
            daily_fast_gpp_defined=microbatch.expected_fast_gpp_defined[0],
            pft_fraction=templates.pft.pft_fraction,
            active_pft_mask=templates.pft.active_pft_mask,
            contract=runtime["contract"],
        )
        statistics = science_window_sufficient_statistics(
            predicted.fluxes,
            expected.fluxes,
            predicted.stocks,
            expected.stocks,
            flux_scales=scales.fluxes,
            stock_scales=scales.stocks,
            horizon=7,
            flux_basis=flux_basis,
        )
        score = science_window_from_sufficient_statistics(statistics, flux_basis=flux_basis)

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
        state_nonfinite = jnp.asarray(0, dtype=jnp.int32)
        unexpected_state_mismatch = jnp.asarray(0, dtype=jnp.int32)
        declared_dynamic_state_mismatch = jnp.asarray(0, dtype=jnp.int32)
        for name in StateFeatureGroups._fields:
            state = jnp.asarray(getattr(trace.result.next_state, name))
            actual = jnp.asarray(getattr(trace.result.next_state_defined, name), dtype=jnp.bool_)
            wanted = jnp.asarray(getattr(expected_defined, name), dtype=jnp.bool_)
            dynamic_mask = jnp.asarray(getattr(scales.state_declared_dynamic, name), dtype=jnp.bool_)
            mismatch = actual != wanted
            state_nonfinite += jnp.sum(wanted & ~jnp.isfinite(state), dtype=jnp.int32)
            unexpected_state_mismatch += jnp.sum(mismatch & ~dynamic_mask, dtype=jnp.int32)
            declared_dynamic_state_mismatch += jnp.sum(mismatch & dynamic_mask, dtype=jnp.int32)
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
        return score, hard

    return evaluate


def _day_scalar(trajectory: Any, day: int) -> float:
    value = np.asarray(trajectory.value, dtype=np.float64)[int(day)]
    if value.size != 1:
        raise ValueError("localized GPP trajectory is not scalar")
    return float(value.reshape(()))


def _factor_day(factors: Mapping[str, Any], day: int) -> Mapping[str, Any]:
    return {name: np.asarray(value)[int(day)] for name, value in factors.items()}


def _factor_records(factors: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    days = int(np.asarray(factors["potential"]).shape[0])
    return [
        {
            "day_index": day,
            "day_ordinal": day + 1,
            **{name: np.asarray(value)[day].tolist() for name, value in factors.items()},
        }
        for day in range(days)
    ]


def _snowheat_records(states: Any, leaves: tuple[Any, ...]) -> Mapping[str, Any]:
    values = np.asarray(states, dtype=np.float64)
    return {
        leaf.key: values[:, int(leaf.start) : int(leaf.stop)].reshape((values.shape[0], *leaf.shape)).tolist()
        for leaf in leaves
    }


def _factorization_comparison(trace: Any, factors: Mapping[str, Any]) -> Mapping[str, Any]:
    observed = np.asarray(trace.result.prediction.named_outputs[GPP_FIELD], dtype=np.float64)
    reconstructed = np.asarray(factors["routed_gpp"], dtype=np.float64)
    if observed.shape != reconstructed.shape:
        raise ValueError("factorized and process-head GPP shapes differ")
    return {
        "maximum_absolute_error": float(np.max(np.abs(observed - reconstructed))),
        "within_tolerance": bool(
            np.allclose(
                observed,
                reconstructed,
                rtol=FACTORIZATION_RTOL,
                atol=FACTORIZATION_ATOL,
            )
        ),
    }


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    parent = _load_localized_result(args.localized_jump_report.resolve())

    reconstruction = reconstruct_localized_jump_parameters(args)
    runtime = reconstruction.runtime
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    leaves, source_routes = resolve_coupled_gpp_snowheat_sources(
        runtime["contract"],
        runtime["owner_contract"],
        reconstruction.bootstrap["layout"],
    )
    artifact = _load_artifact(args, reconstruction.context, TARGET_ARTIFACT_UPDATE)
    microbatch = _slice(artifact.batch.seven_day, TARGET_WINDOW_INDEX, TARGET_WINDOW_INDEX + 1)

    free_evaluator = _make_trace_evaluator(reconstruction.bootstrap)
    control_trace, control_trajectories = free_evaluator(reconstruction.control_parameters, microbatch)
    treatment_trace, treatment_trajectories = free_evaluator(reconstruction.treatment_parameters, microbatch)
    jax.block_until_ready((control_trace, control_trajectories, treatment_trace, treatment_trajectories))

    templates, control_starts, control_discrete = _scenario_day_starts(
        reconstruction,
        microbatch,
        control_trace,
    )
    state_mask = jnp.zeros((runtime["contract"].continuous_state_width,), dtype=jnp.bool_)
    for leaf in leaves:
        state_mask = state_mask.at[int(leaf.start) : int(leaf.stop)].set(True)
    matched_evaluator = _make_matched_snowheat_evaluator(reconstruction, leaves)
    matched_trace, matched_trajectories = matched_evaluator(
        reconstruction.treatment_parameters,
        microbatch,
        control_starts,
    )
    jax.block_until_ready((matched_trace, matched_trajectories))

    _, treatment_starts, treatment_discrete = _scenario_day_starts(
        reconstruction,
        microbatch,
        treatment_trace,
    )
    _, matched_starts, matched_discrete = _scenario_day_starts(
        reconstruction,
        microbatch,
        matched_trace,
        state_mask=state_mask,
        reference=control_starts,
    )
    factor_evaluator = _make_factor_evaluator(reconstruction)
    control_factors = factor_evaluator(
        reconstruction.control_parameters,
        templates,
        control_starts,
        control_discrete,
    )
    treatment_factors = factor_evaluator(
        reconstruction.treatment_parameters,
        templates,
        treatment_starts,
        treatment_discrete,
    )
    matched_factors = factor_evaluator(
        reconstruction.treatment_parameters,
        templates,
        matched_starts,
        matched_discrete,
    )
    jax.block_until_ready((control_factors, treatment_factors, matched_factors))

    score_evaluator = _make_score_evaluator(reconstruction)
    score_records = {}
    hard_records = {}
    for name, trace in (
        ("control_free", control_trace),
        ("treatment_free", treatment_trace),
        ("treatment_matched_snowheat", matched_trace),
    ):
        score, hard = score_evaluator(trace, microbatch)
        jax.block_until_ready((score, hard))
        score_records[name] = _science_record(score)
        hard_records[name] = _hard_host(jax.device_get(hard))

    control_gpp = _day_scalar(control_trajectories.fluxes["gpp"], TARGET_DAY_INDEX)
    treatment_gpp = _day_scalar(treatment_trajectories.fluxes["gpp"], TARGET_DAY_INDEX)
    matched_gpp = _day_scalar(matched_trajectories.fluxes["gpp"], TARGET_DAY_INDEX)
    classification = classify_snowheat_day5_intervention(
        control_gpp=control_gpp,
        treatment_gpp=treatment_gpp,
        matched_snowheat_gpp=matched_gpp,
    )

    day5_routed = np.asarray(control_factors["routed_gpp"])[TARGET_DAY_INDEX].reshape(-1)
    vegetation_pft = int(np.argmax(day5_routed))
    free_attribution = factor_log_attribution(
        _factor_day(control_factors, TARGET_DAY_INDEX),
        _factor_day(treatment_factors, TARGET_DAY_INDEX),
        pft_index=vegetation_pft,
    )
    cut_attribution = factor_log_attribution(
        _factor_day(control_factors, TARGET_DAY_INDEX),
        _factor_day(matched_factors, TARGET_DAY_INDEX),
        pft_index=vegetation_pft,
    )

    factor_comparisons = {
        "control_free": _factorization_comparison(control_trace, control_factors),
        "treatment_free": _factorization_comparison(treatment_trace, treatment_factors),
        "treatment_matched_snowheat": _factorization_comparison(matched_trace, matched_factors),
    }
    factors_finite = all(
        np.all(np.isfinite(np.asarray(value)))
        for factors in (control_factors, treatment_factors, matched_factors)
        for value in factors.values()
    )
    factorization_passed = (
        factors_finite
        and all(bool(value["within_tolerance"]) for value in factor_comparisons.values())
        and all(
            abs(float(value["factor_log_ratio_residual"])) <= 1.0e-6 for value in (free_attribution, cut_attribution)
        )
    )

    parent_days = parent["daily_trace"]["days"]
    parent_control_gpp = np.asarray(
        [value["science_fluxes"]["gpp"]["control"] for value in parent_days],
        dtype=np.float64,
    )
    parent_treatment_gpp = np.asarray(
        [value["science_fluxes"]["gpp"]["treatment"] for value in parent_days],
        dtype=np.float64,
    )
    observed_control_gpp = np.asarray(control_trajectories.fluxes["gpp"].value, dtype=np.float64).reshape(-1)
    observed_treatment_gpp = np.asarray(treatment_trajectories.fluxes["gpp"].value, dtype=np.float64).reshape(-1)
    free_trace_reproduced = bool(
        np.allclose(
            observed_control_gpp,
            parent_control_gpp,
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
        and np.allclose(
            observed_treatment_gpp,
            parent_treatment_gpp,
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )
    parent_window = parent["window_localization"]["selection"]["records"][TARGET_WINDOW_INDEX]
    free_scores_reproduced = bool(
        np.isclose(
            score_records["control_free"]["total"],
            parent_window["control_score"],
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
        and np.isclose(
            score_records["treatment_free"]["total"],
            parent_window["treatment_score"],
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )
    hard_failure_counts = {name: _hard_failure_count(value) for name, value in hard_records.items()}
    engineering_passed = bool(
        free_trace_reproduced
        and free_scores_reproduced
        and factorization_passed
        and all(value == 0 for value in hard_failure_counts.values())
    )
    if not engineering_passed:
        raise RuntimeError("coupled-GPP matched snowheat intervention failed its engineering gate")

    trajectory_records = {}
    for name, trace, trajectories, starts, factors in (
        ("control_free", control_trace, control_trajectories, control_starts, control_factors),
        ("treatment_free", treatment_trace, treatment_trajectories, treatment_starts, treatment_factors),
        (
            "treatment_matched_snowheat",
            matched_trace,
            matched_trajectories,
            matched_starts,
            matched_factors,
        ),
    ):
        trajectory_records[name] = {
            "score": score_records[name],
            "hard_counts": hard_records[name],
            "gpp": np.asarray(trajectories.fluxes["gpp"].value, dtype=np.float64).reshape(-1).tolist(),
            "pft_fraction": np.asarray(trace.pft_fraction, dtype=np.float64).tolist(),
            "snowheat_day_start": _snowheat_records(starts, leaves),
            "gpp_factorization": _factor_records(factors),
            "factorized_routed_gpp_maximum_absolute_error": factor_comparisons[name]["maximum_absolute_error"],
            "factorized_routed_gpp_within_tolerance": factor_comparisons[name]["within_tolerance"],
        }

    control_score = float(score_records["control_free"]["total"])
    treatment_score = float(score_records["treatment_free"]["total"])
    matched_score = float(score_records["treatment_matched_snowheat"]["total"])
    score_jump = abs(treatment_score - control_score)
    score_retained = abs(matched_score - control_score) / max(score_jump, 1.0e-30)
    action = {
        "snowheat_path_causal_for_day5_jump": "pause_for_user_discussion_before_any_gpp_routing_or_architecture_change",
        "snowheat_path_partial_for_day5_jump": "separate_direct_day5_snowheat_routing_from_prior_mediated_state",
        "snowheat_path_not_explanatory_for_day5_jump": "trace_other_day5_gpp_inputs_without_architecture_change",
    }[classification["classification"]]

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": classification["classification"],
        "identity": {
            "git_head": _git_head(),
            "localized_jump_report_sha256": sha256_file(args.localized_jump_report),
            "parent_persistent_training_report_sha256": sha256_file(args.parent_persistent_training_report),
            "parent_rejected_proposal_sha256": sha256_file(args.parent_rejected_proposal),
            "parent_recovery_sha256": reconstruction.recovery_digest,
        },
        "policy": {
            "logical_updates_applied": 0,
            "historical_parameter_displacements_reconstructed_in_memory": 1,
            "optimizer_invoked": False,
            "trainable_parameters_written": False,
            "checkpoint_written": False,
            "recovery_written": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "future_teacher_state_used_by_intervention": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
            "architecture_changed": False,
        },
        "target": {
            "artifact_update": TARGET_ARTIFACT_UPDATE,
            "runtime_year": int(
                reconstruction.context["schedule"]["entries"][TARGET_ARTIFACT_UPDATE - 1]["runtime_year"]
            ),
            "window_index": TARGET_WINDOW_INDEX,
            "window_ordinal": TARGET_WINDOW_INDEX + 1,
            "day5_index": TARGET_DAY_INDEX,
            "parent_first_material_normalized_state_threshold": MATERIAL_NORMALIZED_STATE_CHANGE,
        },
        "source_routes": source_routes,
        "direction_reconstruction": {
            "nonzero_source_count": len(reconstruction.plan["nonzero_sources"]),
            "historical_combined_gradient_norm": reconstruction.plan["historical_combined_gradient_norm"],
            "reconstructed_combined_gradient_norm": reconstruction.combined_norm,
            "treatment_scale": reconstruction.treatment_scale,
            "historical_parameter_delta_l2": reconstruction.historical_delta,
            "reconstructed_parameter_delta_l2": reconstruction.realized_delta,
        },
        "trajectories": trajectory_records,
        "day5_causal_test": classification
        | {
            "control_gpp": control_gpp,
            "treatment_free_gpp": treatment_gpp,
            "treatment_matched_snowheat_gpp": matched_gpp,
            "vegetation_pft_index": vegetation_pft,
            "free_treatment_factor_attribution": free_attribution,
            "matched_snowheat_factor_attribution": cut_attribution,
        },
        "selected_window_score_effect": {
            "control": control_score,
            "treatment_free": treatment_score,
            "treatment_matched_snowheat": matched_score,
            "free_absolute_displacement": score_jump,
            "matched_snowheat_retained_displacement_fraction": score_retained,
        },
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "free_gpp_trace_reproduced": free_trace_reproduced,
            "free_window_scores_reproduced": free_scores_reproduced,
            "factorization_passed": factorization_passed,
            "factorized_routed_gpp_within_tolerance": {
                name: bool(value["within_tolerance"]) for name, value in factor_comparisons.items()
            },
            "factors_finite": factors_finite,
            "hard_failure_counts": hard_failure_counts,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "memory_after": _memory_report(reconstruction.bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": action,
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _persistent_parser()
    parser.add_argument("--parent-persistent-training-report", type=Path, required=True)
    parser.add_argument("--parent-rejected-proposal", type=Path, required=True)
    parser.add_argument("--parent-artifact-aware-recovery", type=Path, required=True)
    parser.add_argument("--localized-jump-report", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
