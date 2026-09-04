"""Separate Day-5 coupled-GPP parameter and carried-state effects."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    strong_baseline_parameter_count,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.successor_training import (
    rollout_input_from_template,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_day5_crossed_attribution import (
    INTERPOLATION_FRACTIONS,
    PARAMETER_FAMILY_COUNTS,
    SCHEMA_VERSION,
    STATE_COMPONENT_COORDINATE_COUNTS,
    classify_parameter_state_cross,
    classify_state_owner_controls,
    parameter_family_controls,
    resolve_crossed_gpp_state_components,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    TRAINABLE_FAMILIES,
    coupled_gpp_trainable_tree,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_snowheat_causal import (
    TARGET_ARTIFACT_UPDATE,
    TARGET_DAY_INDEX,
    TARGET_WINDOW_INDEX,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import GPP_FIELD
from research.daily_coarse_graining.daily_operator.types import StateFeatureGroups
from scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump import (
    FORWARD_REPRODUCTION_ATOL,
    FORWARD_REPRODUCTION_RTOL,
    _make_trace_evaluator,
    reconstruct_localized_jump_parameters,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal import (
    FACTORIZATION_ATOL,
    FACTORIZATION_RTOL,
    _factor_day,
    _make_factor_evaluator,
    _scenario_day_starts,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal import (
    _parser as _snowheat_parser,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _initialization_only,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import _git_head
from scripts.hpc.run_target_oriented_full_step_preflight import _slice

SNOWHEAT_RESULT_SHA256 = "a19c6230e14d50b204d3d5ebf0eb70c721c3d0e7683d06d03083313da15a2e23"
BUDGET_ATOL = 2.0e-12
ADDITIVITY_ATOL = 1.0e-12


def _load_snowheat_result(path: Path) -> Mapping[str, Any]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        digest != SNOWHEAT_RESULT_SHA256
        or report.get("schema_version")
        != "target_oriented_coupled_gpp_snowheat_causal_intervention_result_v1"
        or report.get("status") != "passed"
        or report.get("classification") != "snowheat_path_not_explanatory_for_day5_jump"
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
        or float(report.get("day5_causal_test", {}).get("retained_jump_fraction", -1.0)) != 1.0
    ):
        raise ValueError("coupled-GPP snowheat result drift")
    return report


def _make_one_day_evaluator(reconstruction: Any) -> Any:
    runtime = reconstruction.runtime
    jax = reconstruction.jax

    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )

    dynamic = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=reconstruction.bootstrap["head"],
        retained_tail_template=reconstruction.context["template"],
    )

    @jax.jit
    def evaluate(
        parameters: Any,
        template: Any,
        continuous: Any,
        discrete: Any,
        day_inputs: Any,
    ) -> Any:
        current = rollout_input_from_template(
            template,
            continuous,
            discrete,
            runtime["contract"],
        )
        return dynamic(parameters, current, day_inputs)

    return evaluate


def _tree_day(jax: Any, tree: Any, day: int) -> Any:
    return jax.tree_util.tree_map(lambda value: value[int(day)], tree)


def _tree_day_batch(jax: Any, tree: Any, day: int) -> Any:
    return jax.tree_util.tree_map(lambda value: value[int(day) : int(day) + 1], tree)


def _tree_mismatch_count(jax: Any, first: Any, second: Any) -> int:
    first_leaves, first_tree = jax.tree_util.tree_flatten(first)
    second_leaves, second_tree = jax.tree_util.tree_flatten(second)
    if first_tree != second_tree:
        raise ValueError("crossed attribution discrete tree structure drift")
    return int(
        sum(
            np.sum(np.asarray(left) != np.asarray(right), dtype=np.int64)
            for left, right in zip(first_leaves, second_leaves, strict=True)
        )
    )


def _runtime_year_for_update(schedule: Mapping[str, Any], update: int) -> int:
    return int(schedule["entries"][int(update) - 1]["runtime_year"])


def _factor_record(factors: Mapping[str, Any], pft_index: int) -> Mapping[str, Any]:
    record = {name: np.asarray(value).tolist() for name, value in factors.items()}
    record["gpp"] = float(np.asarray(factors["routed_gpp"]).reshape(-1)[int(pft_index)])
    return record


def _evaluate_factors(
    evaluator: Any,
    reconstruction: Any,
    parameters: Any,
    template_batch: Any,
    state: Any,
    discrete_batch: Any,
) -> Mapping[str, Any]:
    jax = reconstruction.jax
    factors = evaluator(
        parameters,
        template_batch,
        state[None, :],
        discrete_batch,
    )
    jax.block_until_ready(factors)
    return _factor_day(jax.device_get(factors), 0)


def _one_day_integrity(jax: Any, result: Any, pft_index: int) -> Mapping[str, Any]:
    host = jax.device_get(result)
    nonfinite = 0
    for name in StateFeatureGroups._fields:
        state = np.asarray(getattr(host.next_state, name))
        defined = np.asarray(getattr(host.next_state_defined, name), dtype=np.bool_)
        nonfinite += int(np.sum(defined & ~np.isfinite(state), dtype=np.int64))
    water = float(np.max(np.abs(np.asarray(host.diagnostics["water_budget_residual"]))))
    carbon = float(np.max(np.abs(np.asarray(host.diagnostics["carbon_budget_residual"]))))
    output = np.asarray(host.prediction.named_outputs[GPP_FIELD], dtype=np.float64)
    gpp = float(output.reshape(-1)[int(pft_index)])
    passed = bool(
        nonfinite == 0
        and np.isfinite(gpp)
        and gpp >= 0.0
        and water <= BUDGET_ATOL
        and carbon <= BUDGET_ATOL
    )
    return {
        "gpp": gpp,
        "gpp_output": output.tolist(),
        "defined_state_nonfinite_count": nonfinite,
        "maximum_absolute_water_budget_residual": water,
        "maximum_absolute_carbon_budget_residual": carbon,
        "passed": passed,
    }


def _factorization_comparison(
    result: Mapping[str, Any],
    factors: Mapping[str, Any],
) -> Mapping[str, Any]:
    observed = np.asarray(result["gpp_output"], dtype=np.float64)
    reconstructed = np.asarray(factors["routed_gpp"], dtype=np.float64)
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


def _interpolate_tree(jax: Any, jnp: Any, first: Any, second: Any, fraction: float) -> Any:
    if fraction == 0.0:
        return first
    if fraction == 1.0:
        return second
    return jax.tree_util.tree_map(
        lambda left, right: left
        + jnp.asarray(fraction, dtype=jnp.asarray(left).dtype) * (right - left),
        first,
        second,
    )


def _state_delta_record(
    control: np.ndarray,
    treatment: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
) -> Mapping[str, Any]:
    selected = np.asarray(mask, dtype=np.bool_)
    delta = np.abs(treatment[selected] - control[selected])
    selected_scale = scale[selected]
    normalized = delta / np.where(
        np.isfinite(selected_scale) & (selected_scale > 0.0),
        selected_scale,
        1.0,
    )
    finite = np.isfinite(delta) & np.isfinite(normalized)
    return {
        "coordinate_count": int(np.sum(selected)),
        "finite_coordinate_count": int(np.sum(finite)),
        "nonzero_coordinate_count": int(np.sum(delta[finite] != 0.0)),
        "maximum_absolute_change": float(np.max(delta[finite])) if np.any(finite) else None,
        "maximum_normalized_change": (
            float(np.max(normalized[finite])) if np.any(finite) else None
        ),
        "normalized_l2": float(np.linalg.norm(normalized[finite])) if np.any(finite) else None,
    }


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    parent = _load_snowheat_result(args.snowheat_report.resolve())

    bootstrap = _initialization_only(args)
    runtime = bootstrap["context"]["runtime"]
    source_components = resolve_crossed_gpp_state_components(
        runtime["contract"],
        bootstrap["head"].layout,
    )
    source_routes = {
        name: {key: value for key, value in record.items() if key != "mask"}
        for name, record in source_components.items()
    }
    initialized_trainable = coupled_gpp_trainable_tree(
        bootstrap["initialized_parameters"]
    )
    parameter_counts = {
        name: strong_baseline_parameter_count(initialized_trainable[name])
        for name in TRAINABLE_FAMILIES
    }
    if parameter_counts != PARAMETER_FAMILY_COUNTS:
        raise ValueError("crossed attribution parameter-family count drift")
    runtime_year = _runtime_year_for_update(
        bootstrap["context"]["schedule"],
        TARGET_ARTIFACT_UPDATE,
    )

    reconstruction = reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)
    runtime = reconstruction.runtime
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    control_trainable = coupled_gpp_trainable_tree(reconstruction.control_parameters)
    treatment_trainable = coupled_gpp_trainable_tree(reconstruction.treatment_parameters)
    parameter_counts_passed = bool(
        all(
            strong_baseline_parameter_count(values[name])
            == PARAMETER_FAMILY_COUNTS[name]
            for values in (control_trainable, treatment_trainable)
            for name in TRAINABLE_FAMILIES
        )
    )

    artifact = _load_artifact(args, reconstruction.context, TARGET_ARTIFACT_UPDATE)
    microbatch = _slice(
        artifact.batch.seven_day,
        TARGET_WINDOW_INDEX,
        TARGET_WINDOW_INDEX + 1,
    )
    trace_evaluator = _make_trace_evaluator(bootstrap)
    control_trace, control_trajectories = trace_evaluator(
        reconstruction.control_parameters,
        microbatch,
    )
    treatment_trace, treatment_trajectories = trace_evaluator(
        reconstruction.treatment_parameters,
        microbatch,
    )
    jax.block_until_ready(
        (control_trace, control_trajectories, treatment_trace, treatment_trajectories)
    )
    observed_control_trace = np.asarray(
        control_trajectories.fluxes["gpp"].value,
        dtype=np.float64,
    ).reshape(-1)
    observed_treatment_trace = np.asarray(
        treatment_trajectories.fluxes["gpp"].value,
        dtype=np.float64,
    ).reshape(-1)
    expected_control_trace = np.asarray(
        parent["trajectories"]["control_free"]["gpp"],
        dtype=np.float64,
    )
    expected_treatment_trace = np.asarray(
        parent["trajectories"]["treatment_free"]["gpp"],
        dtype=np.float64,
    )
    free_trace_reproduced = bool(
        np.allclose(
            observed_control_trace,
            expected_control_trace,
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
        and np.allclose(
            observed_treatment_trace,
            expected_treatment_trace,
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )

    templates, control_starts, control_discrete = _scenario_day_starts(
        reconstruction,
        microbatch,
        control_trace,
    )
    _, treatment_starts, treatment_discrete = _scenario_day_starts(
        reconstruction,
        microbatch,
        treatment_trace,
    )
    template = _tree_day(jax, templates, TARGET_DAY_INDEX)
    template_batch = _tree_day_batch(jax, templates, TARGET_DAY_INDEX)
    control_discrete_day = _tree_day(jax, control_discrete, TARGET_DAY_INDEX)
    treatment_discrete_day = _tree_day(jax, treatment_discrete, TARGET_DAY_INDEX)
    control_discrete_batch = _tree_day_batch(jax, control_discrete, TARGET_DAY_INDEX)
    treatment_discrete_batch = _tree_day_batch(jax, treatment_discrete, TARGET_DAY_INDEX)
    discrete_mismatch_count = _tree_mismatch_count(
        jax,
        control_discrete_day,
        treatment_discrete_day,
    )
    control_state = jnp.asarray(control_starts[TARGET_DAY_INDEX])
    treatment_state = jnp.asarray(treatment_starts[TARGET_DAY_INDEX])
    state_scale = np.asarray(reconstruction.context["scales"].state.other, dtype=np.float64)
    state_delta = {
        name: _state_delta_record(
            np.asarray(control_state, dtype=np.float64),
            np.asarray(treatment_state, dtype=np.float64),
            state_scale,
            np.asarray(record["mask"], dtype=np.bool_),
        )
        for name, record in source_components.items()
    }

    factor_evaluator = _make_factor_evaluator(reconstruction)
    one_day_evaluator = _make_one_day_evaluator(reconstruction)
    day_inputs = jax.tree_util.tree_map(lambda value: value[0], microbatch.day_inputs)
    cell_specs = {
        "CC": (
            reconstruction.control_parameters,
            control_state,
            control_discrete_day,
            control_discrete_batch,
        ),
        "TC": (
            reconstruction.treatment_parameters,
            control_state,
            control_discrete_day,
            control_discrete_batch,
        ),
        "CT": (
            reconstruction.control_parameters,
            treatment_state,
            treatment_discrete_day,
            treatment_discrete_batch,
        ),
        "TT": (
            reconstruction.treatment_parameters,
            treatment_state,
            treatment_discrete_day,
            treatment_discrete_batch,
        ),
    }
    raw_factors = {}
    raw_results = {}
    for name, (parameters, state, discrete, discrete_batch) in cell_specs.items():
        factors = _evaluate_factors(
            factor_evaluator,
            reconstruction,
            parameters,
            template_batch,
            state,
            discrete_batch,
        )
        result = one_day_evaluator(parameters, template, state, discrete, day_inputs)
        jax.block_until_ready(result)
        raw_factors[name] = factors
        raw_results[name] = result
        print(f"coupled_gpp day5_cross cell={name}/4", flush=True)

    routed = np.asarray(raw_factors["CC"]["routed_gpp"], dtype=np.float64).reshape(-1)
    vegetation_pft_index = int(np.argmax(routed))
    if routed[vegetation_pft_index] <= 0.0:
        raise RuntimeError("crossed attribution has no positive routed PFT GPP")
    cell_records = {}
    cell_factorization_passed = True
    crossed_day_integrity_passed = True
    for name in cell_specs:
        integrity = _one_day_integrity(jax, raw_results[name], vegetation_pft_index)
        comparison = _factorization_comparison(
            integrity,
            raw_factors[name],
        )
        factor_record = _factor_record(raw_factors[name], vegetation_pft_index)
        cell_factorization_passed &= bool(comparison["within_tolerance"])
        crossed_day_integrity_passed &= bool(integrity["passed"])
        cell_records[name] = {
            "parameters": "control" if name[0] == "C" else "treatment",
            "day5_start_state": "control" if name[1] == "C" else "treatment",
            "complete_operator": integrity,
            "factorization_comparison": comparison,
            "factors": factor_record,
        }
    cell_gpp = {
        name: float(record["complete_operator"]["gpp"])
        for name, record in cell_records.items()
    }
    crossed_free_cells_reproduced = bool(
        np.isclose(
            cell_gpp["CC"],
            expected_control_trace[TARGET_DAY_INDEX],
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
        and np.isclose(
            cell_gpp["TT"],
            expected_treatment_trace[TARGET_DAY_INDEX],
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )
    crossed = classify_parameter_state_cross(
        gpp_cc=cell_gpp["CC"],
        gpp_tc=cell_gpp["TC"],
        gpp_ct=cell_gpp["CT"],
        gpp_tt=cell_gpp["TT"],
    )

    owner_records = {}
    matched_owner_gpp = {}
    transplanted_owner_gpp = {}
    all_factors_finite = all(
        np.all(np.isfinite(np.asarray(value)))
        for factors in raw_factors.values()
        for value in factors.values()
    )
    all_routed_gpp_nonnegative = all(
        np.all(np.asarray(factors["routed_gpp"]) >= 0.0)
        for factors in raw_factors.values()
    )
    for name, component in source_components.items():
        mask = jnp.asarray(component["mask"], dtype=jnp.bool_)
        matched_state = jnp.where(mask, control_state, treatment_state)
        transplanted_state = jnp.where(mask, treatment_state, control_state)
        matched = _evaluate_factors(
            factor_evaluator,
            reconstruction,
            reconstruction.treatment_parameters,
            template_batch,
            matched_state,
            treatment_discrete_batch,
        )
        transplanted = _evaluate_factors(
            factor_evaluator,
            reconstruction,
            reconstruction.treatment_parameters,
            template_batch,
            transplanted_state,
            control_discrete_batch,
        )
        matched_record = _factor_record(matched, vegetation_pft_index)
        transplanted_record = _factor_record(transplanted, vegetation_pft_index)
        matched_owner_gpp[name] = float(matched_record["gpp"])
        transplanted_owner_gpp[name] = float(transplanted_record["gpp"])
        owner_records[name] = {
            "matched_from_treatment_state": matched_record,
            "transplanted_into_control_state": transplanted_record,
        }
        all_factors_finite &= all(
            np.all(np.isfinite(np.asarray(value)))
            for factors in (matched, transplanted)
            for value in factors.values()
        )
        all_routed_gpp_nonnegative &= bool(
            np.all(np.asarray(matched["routed_gpp"]) >= 0.0)
            and np.all(np.asarray(transplanted["routed_gpp"]) >= 0.0)
        )
        print(f"coupled_gpp day5_cross state_owner={name}", flush=True)
    owner_classification = classify_state_owner_controls(
        gpp_treatment_control_state=float(
            cell_records["TC"]["factors"]["gpp"]
        ),
        gpp_treatment_treatment_state=float(
            cell_records["TT"]["factors"]["gpp"]
        ),
        matched_owner_gpp=matched_owner_gpp,
        transplanted_owner_gpp=transplanted_owner_gpp,
        state_path_classification=str(crossed["classification"]),
    )
    for name, record in owner_classification["components"].items():
        owner_records[name] = {**owner_records[name], **record}

    family_parameters = {}
    for name in TRAINABLE_FAMILIES:
        mixed_trainable = dict(control_trainable)
        mixed_trainable[name] = treatment_trainable[name]
        family_parameters[name] = replace_coupled_gpp_trainable(
            reconstruction.control_parameters,
            mixed_trainable,
        )
    family_state_records = {}
    for state_name, state, discrete_batch, control_cell, treatment_cell in (
        ("control_state", control_state, control_discrete_batch, "CC", "TC"),
        ("treatment_state", treatment_state, treatment_discrete_batch, "CT", "TT"),
    ):
        family_gpp = {}
        factor_records = {}
        for name in TRAINABLE_FAMILIES:
            factors = _evaluate_factors(
                factor_evaluator,
                reconstruction,
                family_parameters[name],
                template_batch,
                state,
                discrete_batch,
            )
            record = _factor_record(factors, vegetation_pft_index)
            family_gpp[name] = float(record["gpp"])
            factor_records[name] = record
            all_factors_finite &= all(
                np.all(np.isfinite(np.asarray(value))) for value in factors.values()
            )
            all_routed_gpp_nonnegative &= bool(
                np.all(np.asarray(factors["routed_gpp"]) >= 0.0)
            )
        attribution = parameter_family_controls(
            control_gpp=float(cell_records[control_cell]["factors"]["gpp"]),
            treatment_gpp=float(cell_records[treatment_cell]["factors"]["gpp"]),
            treatment_family_only_gpp=family_gpp,
        )
        family_state_records[state_name] = {
            **attribution,
            "factorizations": factor_records,
        }
        print(f"coupled_gpp day5_cross parameter_families={state_name}", flush=True)

    paths = {
        "parameter_at_control_state": [],
        "parameter_at_treatment_state": [],
        "state_at_control_parameters": [],
        "state_at_treatment_parameters": [],
    }
    for fraction in INTERPOLATION_FRACTIONS:
        interpolated_trainable = _interpolate_tree(
            jax,
            jnp,
            control_trainable,
            treatment_trainable,
            fraction,
        )
        interpolated_parameters = replace_coupled_gpp_trainable(
            reconstruction.control_parameters,
            interpolated_trainable,
        )
        interpolated_state = (
            control_state
            if fraction == 0.0
            else treatment_state
            if fraction == 1.0
            else control_state
            + jnp.asarray(fraction, dtype=control_state.dtype)
            * (treatment_state - control_state)
        )
        for path_name, parameters, state, discrete_batch in (
            (
                "parameter_at_control_state",
                interpolated_parameters,
                control_state,
                control_discrete_batch,
            ),
            (
                "parameter_at_treatment_state",
                interpolated_parameters,
                treatment_state,
                treatment_discrete_batch,
            ),
            (
                "state_at_control_parameters",
                reconstruction.control_parameters,
                interpolated_state,
                control_discrete_batch,
            ),
            (
                "state_at_treatment_parameters",
                reconstruction.treatment_parameters,
                interpolated_state,
                treatment_discrete_batch,
            ),
        ):
            factors = _evaluate_factors(
                factor_evaluator,
                reconstruction,
                parameters,
                template_batch,
                state,
                discrete_batch,
            )
            paths[path_name].append(
                {
                    "fraction": fraction,
                    **_factor_record(factors, vegetation_pft_index),
                }
            )
            all_factors_finite &= all(
                np.all(np.isfinite(np.asarray(value))) for value in factors.values()
            )
            all_routed_gpp_nonnegative &= bool(
                np.all(np.asarray(factors["routed_gpp"]) >= 0.0)
            )
        print(f"coupled_gpp day5_cross interpolation={fraction:g}", flush=True)

    endpoint_targets = {
        "parameter_at_control_state": ("CC", "TC"),
        "parameter_at_treatment_state": ("CT", "TT"),
        "state_at_control_parameters": ("CC", "CT"),
        "state_at_treatment_parameters": ("TC", "TT"),
    }
    endpoint_records = {}
    endpoints_passed = True
    for name, (first, second) in endpoint_targets.items():
        first_error = abs(
            float(paths[name][0]["gpp"])
            - float(cell_records[first]["factors"]["gpp"])
        )
        second_error = abs(
            float(paths[name][-1]["gpp"])
            - float(cell_records[second]["factors"]["gpp"])
        )
        passed = bool(first_error <= ADDITIVITY_ATOL and second_error <= ADDITIVITY_ATOL)
        endpoints_passed &= passed
        endpoint_records[name] = {
            "first_absolute_error": first_error,
            "last_absolute_error": second_error,
            "passed": passed,
        }

    masks_passed = bool(
        tuple(source_components) == tuple(STATE_COMPONENT_COORDINATE_COUNTS)
        and all(
            int(np.sum(np.asarray(record["mask"], dtype=np.bool_)))
            == STATE_COMPONENT_COORDINATE_COUNTS[name]
            for name, record in source_components.items()
        )
    )
    routed_nonnegative = bool(all_routed_gpp_nonnegative)
    shapley_passed = bool(
        abs(float(crossed["shapley_additivity_residual"])) <= ADDITIVITY_ATOL
    )
    engineering_passed = bool(
        free_trace_reproduced
        and crossed_free_cells_reproduced
        and discrete_mismatch_count == 0
        and masks_passed
        and parameter_counts_passed
        and all_factors_finite
        and routed_nonnegative
        and cell_factorization_passed
        and crossed_day_integrity_passed
        and shapley_passed
        and endpoints_passed
    )
    if not engineering_passed:
        print(
            "coupled_gpp day5_cross engineering_gate_failed "
            f"free_trace={free_trace_reproduced} "
            f"crossed_free={crossed_free_cells_reproduced} "
            f"discrete={discrete_mismatch_count} "
            f"masks={masks_passed} parameter_counts={parameter_counts_passed} "
            f"factors={all_factors_finite} nonnegative={routed_nonnegative} "
            f"factorization={cell_factorization_passed} "
            f"daily={crossed_day_integrity_passed} shapley={shapley_passed} "
            f"endpoints={endpoints_passed}",
            flush=True,
        )

    classification = (
        str(crossed["classification"])
        if engineering_passed
        else "engineering_gate_failed"
    )
    if not engineering_passed:
        next_gate = "inspect_written_engineering_gate_report_before_rerun"
    elif classification == "state_path_not_explanatory_for_day5_jump":
        next_gate = "pause_for_user_discussion_of_direct_parameter_family_attribution"
    else:
        next_gate = "pause_for_user_discussion_of_state_owner_attribution"
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if engineering_passed else "engineering_gate_failed",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "snowheat_report_sha256": sha256_file(args.snowheat_report),
            "parent_recovery_sha256": reconstruction.recovery_digest,
        },
        "target": {
            "artifact_update": TARGET_ARTIFACT_UPDATE,
            "runtime_year": runtime_year,
            "window_index": TARGET_WINDOW_INDEX,
            "window_ordinal": TARGET_WINDOW_INDEX + 1,
            "day_index": TARGET_DAY_INDEX,
            "day_ordinal": TARGET_DAY_INDEX + 1,
        },
        "direction_reconstruction": {
            "nonzero_source_count": len(reconstruction.plan["nonzero_sources"]),
            "historical_combined_gradient_norm": reconstruction.plan[
                "historical_combined_gradient_norm"
            ],
            "reconstructed_combined_gradient_norm": reconstruction.combined_norm,
            "treatment_scale": reconstruction.treatment_scale,
            "historical_parameter_delta_l2": reconstruction.historical_delta,
            "reconstructed_parameter_delta_l2": reconstruction.realized_delta,
        },
        "free_gpp_traces": {
            "control": observed_control_trace.tolist(),
            "treatment": observed_treatment_trace.tolist(),
        },
        "parameter_state_cross": {
            "cells": cell_records,
            "attribution": crossed,
        },
        "state_owner_controls": {
            "source_routes": source_routes,
            "day5_start_delta": state_delta,
            **owner_classification,
            "components": owner_records,
        },
        "parameter_family_controls": {
            "parameter_counts": parameter_counts,
            "by_state": family_state_records,
        },
        "interpolation_paths": paths,
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "free_gpp_trace_reproduced": free_trace_reproduced,
            "crossed_free_cells_reproduced": crossed_free_cells_reproduced,
            "day5_discrete_mismatch_count": discrete_mismatch_count,
            "state_owner_masks_passed": masks_passed,
            "parameter_family_counts_passed": parameter_counts_passed,
            "all_factors_finite": all_factors_finite,
            "all_routed_gpp_nonnegative": routed_nonnegative,
            "crossed_factorization_passed": cell_factorization_passed,
            "crossed_complete_day_integrity_passed": crossed_day_integrity_passed,
            "shapley_additivity_passed": shapley_passed,
            "interpolation_endpoints_passed": endpoints_passed,
            "interpolation_endpoint_checks": endpoint_records,
            "factorization_tolerance": {
                "rtol": FACTORIZATION_RTOL,
                "atol": FACTORIZATION_ATOL,
            },
            "budget_atol": BUDGET_ATOL,
        },
        "policy": {
            "logical_updates_applied": 0,
            "historical_parameter_displacements_reconstructed_in_memory": 1,
            "optimizer_invoked": False,
            "trainable_parameters_written": False,
            "checkpoint_written": False,
            "recovery_written": False,
            "architecture_changed": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "future_teacher_state_used": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _snowheat_parser()
    parser.add_argument("--snowheat-report", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
