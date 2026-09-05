"""Attribute the accepted Day-5 GPP jump to precision and parameter subtrees."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    strong_baseline_parameter_count,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    coupled_gpp_trainable_tree,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_snowheat_causal import (
    TARGET_ARTIFACT_UPDATE,
    TARGET_DAY_INDEX,
    TARGET_WINDOW_INDEX,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_structured_precision_attribution import (
    INTERPOLATION_FRACTIONS,
    SCHEMA_VERSION,
    STRUCTURED_GPP_GROUPS,
    STRUCTURED_GPP_PARAMETER_COUNT,
    classify_precision_paths,
    classify_subtree_controls,
    float32_delta_inventory,
    float32_interpolation_membership,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution import (
    _evaluate_factors,
    _factor_record,
    _runtime_year_for_update,
    _tree_day_batch,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution import (
    _parser as _crossed_parser,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump import (
    DIRECTION_NORM_ATOL,
    DIRECTION_NORM_RTOL,
    FORWARD_REPRODUCTION_ATOL,
    FORWARD_REPRODUCTION_RTOL,
    _make_trace_evaluator,
    reconstruct_localized_jump_parameters,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal import (
    FACTORIZATION_ATOL,
    FACTORIZATION_RTOL,
    _make_factor_evaluator,
    _scenario_day_starts,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _initialization_only,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import _git_head
from scripts.hpc.run_target_oriented_full_step_preflight import _slice

CROSSED_ATTRIBUTION_SHA256 = "c014ac09558f30671bc511cfa9a63690f46f42434da428a414664581987780c5"
CROSSED_ATTRIBUTION_SCHEMA = "target_oriented_coupled_gpp_day5_crossed_attribution_result_v1"
CROSSED_ATTRIBUTION_CLASSIFICATION = "state_path_not_explanatory_for_day5_jump"
EXPECTED_TARGET = {
    "artifact_update": TARGET_ARTIFACT_UPDATE,
    "runtime_year": 1976,
    "window_index": TARGET_WINDOW_INDEX,
    "window_ordinal": TARGET_WINDOW_INDEX + 1,
    "day_index": TARGET_DAY_INDEX,
    "day_ordinal": TARGET_DAY_INDEX + 1,
}


def _validate_crossed_result(report: Mapping[str, Any], digest: str) -> None:
    attribution = report.get("parameter_state_cross", {}).get("attribution", {})
    family = report.get("parameter_family_controls", {}).get("by_state", {}).get("control_state", {})
    structured = family.get("families", {}).get("structured_gpp", {})
    feedback = family.get("families", {}).get(
        "coupled_gpp_carbon_feedback",
        {},
    )
    cells = report.get("parameter_state_cross", {}).get("cells", {})
    if (
        digest != CROSSED_ATTRIBUTION_SHA256
        or report.get("schema_version") != CROSSED_ATTRIBUTION_SCHEMA
        or report.get("status") != "passed"
        or report.get("classification") != CROSSED_ATTRIBUTION_CLASSIFICATION
        or report.get("target") != EXPECTED_TARGET
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
        or attribution.get("classification") != CROSSED_ATTRIBUTION_CLASSIFICATION
        or float(attribution.get("parameter_shapley_fraction", -1.0)) != 1.0
        or float(attribution.get("state_shapley_fraction", -1.0)) != 0.0
        or float(attribution.get("two_factor_interaction", -1.0)) != 0.0
        or family.get("eligible") is not True
        or float(structured.get("signed_sufficiency_fraction", -1.0)) != 1.0
        or float(structured.get("necessity_capture_fraction", -1.0)) != 1.0
        or float(feedback.get("signed_sufficiency_fraction", -1.0)) != 0.0
        or float(feedback.get("necessity_capture_fraction", -1.0)) != 0.0
        or frozenset(cells) != frozenset(("CC", "CT", "TC", "TT"))
    ):
        raise ValueError("coupled-GPP crossed-attribution result drift")


def _load_crossed_result(path: Path) -> tuple[Mapping[str, Any], str]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    _validate_crossed_result(report, digest)
    return report, digest


def _parent_factor_metadata(parent: Mapping[str, Any]) -> Mapping[str, Any]:
    cells = parent["parameter_state_cross"]["cells"]
    routed = np.asarray(cells["CC"]["factors"]["routed_gpp"], dtype=np.float64)
    flat = routed.reshape(-1)
    positive = np.flatnonzero(flat > 0.0)
    if positive.size != 1:
        raise ValueError("crossed-attribution vegetation PFT identity drift")
    return {
        "vegetation_pft_index": int(positive[0]),
        "control_gpp": float(cells["CC"]["factors"]["gpp"]),
        "treatment_gpp": float(cells["TC"]["factors"]["gpp"]),
        "control_gpp_trace": tuple(float(value) for value in parent["free_gpp_traces"]["control"]),
    }


def _validate_initialized_structured_tree(tree: Mapping[str, Any]) -> None:
    if frozenset(tree) != frozenset(STRUCTURED_GPP_GROUPS):
        raise ValueError("structured-GPP initial subtree registry drift")
    if strong_baseline_parameter_count(tree) != STRUCTURED_GPP_PARAMETER_COUNT:
        raise ValueError("structured-GPP initial parameter count drift")
    leaves = []
    for value in tree.values():
        leaves.extend(_array_leaves(value))
    if any(np.dtype(value.dtype) != np.dtype(np.float32) for value in leaves):
        raise ValueError("structured-GPP initial parameter dtype drift")


def _array_leaves(tree: Any) -> list[Any]:
    if isinstance(tree, Mapping):
        output = []
        for value in tree.values():
            output.extend(_array_leaves(value))
        return output
    if isinstance(tree, (tuple, list)):
        output = []
        for value in tree:
            output.extend(_array_leaves(value))
        return output
    return [tree]


def _replace_structured_gpp(parameters: Mapping[str, Any], structured: Any) -> Any:
    target = dict(parameters["target_oriented"])
    target["structured_gpp"] = structured
    return dict(parameters, target_oriented=target)


def _mixed_subtree(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    treatment_groups: Sequence[str],
) -> Mapping[str, Any]:
    expected = frozenset(STRUCTURED_GPP_GROUPS)
    if frozenset(control) != expected or frozenset(treatment) != expected:
        raise ValueError("structured-GPP subtree registry drift")
    selected = frozenset(treatment_groups)
    if not selected.issubset(STRUCTURED_GPP_GROUPS):
        raise ValueError("unknown structured-GPP subtree")
    return {name: treatment[name] if name in selected else control[name] for name in STRUCTURED_GPP_GROUPS}


def _interpolate_tree(
    jax: Any,
    jnp: Any,
    first: Any,
    second: Any,
    fraction: float,
    *,
    dtype: Any,
) -> Any:
    left = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=dtype), first)
    right = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=dtype), second)
    if fraction == 0.0:
        return left
    if fraction == 1.0:
        return right
    return jax.tree_util.tree_map(
        lambda start, stop: start + jnp.asarray(fraction, dtype=dtype) * (stop - start),
        left,
        right,
    )


def _evaluate_record(
    evaluator: Any,
    reconstruction: Any,
    parameters: Any,
    template_batch: Any,
    state: Any,
    discrete_batch: Any,
    pft_index: int,
) -> Mapping[str, Any]:
    factors = _evaluate_factors(
        evaluator,
        reconstruction,
        parameters,
        template_batch,
        state,
        discrete_batch,
    )
    return _factor_record(factors, pft_index)


def _record_is_valid(record: Mapping[str, Any]) -> bool:
    numeric = (np.asarray(value) for name, value in record.items() if name not in {"fraction", "weight_dtype"})
    values = tuple(numeric)
    return bool(all(np.all(np.isfinite(value)) for value in values) and np.all(np.asarray(record["routed_gpp"]) >= 0.0))


def _records_exact(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    names = frozenset(first) - {"fraction", "weight_dtype"}
    if names != frozenset(second) - {"fraction", "weight_dtype"}:
        return False
    return all(np.array_equal(np.asarray(first[name]), np.asarray(second[name])) for name in names)


def _inventory_conserved(inventory: Mapping[str, Any]) -> bool:
    if inventory["summary"]["parameter_count"] != STRUCTURED_GPP_PARAMETER_COUNT:
        return False
    if inventory["summary"]["ulp_count_conserved"] is not True:
        return False
    for group in inventory["groups"].values():
        if group["summary"]["ulp_count_conserved"] is not True:
            return False
        if any(record["ulp_count_conserved"] is not True for record in group["leaves"].values()):
            return False
    return True


def _membership_conserved(records: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        len(records) == len(INTERPOLATION_FRACTIONS)
        and all(
            record["summary"]["parameter_count"] == STRUCTURED_GPP_PARAMETER_COUNT
            and record["summary"]["changed_membership_conserved"] is True
            and record["summary"]["unchanged_preserved"] is True
            and record["summary"]["realized_nonfinite_count"] == 0
            for record in records
        )
    )


def _direction_integrity(
    reconstruction: Any,
    parent: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = parent["direction_reconstruction"]
    expected_delta = expected.get("reconstructed_treatment_parameter_delta_l2")
    if expected_delta is None:
        expected_delta = expected["reconstructed_parameter_delta_l2"]
    norm_passed = bool(
        reconstruction.norm_matches
        and np.isclose(
            reconstruction.combined_norm,
            float(expected["reconstructed_combined_gradient_norm"]),
            rtol=DIRECTION_NORM_RTOL,
            atol=DIRECTION_NORM_ATOL,
        )
    )
    displacement_passed = bool(
        reconstruction.nominal_delta_matches
        and np.isclose(
            reconstruction.realized_delta,
            float(expected_delta),
            rtol=DIRECTION_NORM_RTOL,
            atol=DIRECTION_NORM_ATOL,
        )
    )
    return {
        "historical_combined_gradient_norm": reconstruction.plan["historical_combined_gradient_norm"],
        "reconstructed_combined_gradient_norm": reconstruction.combined_norm,
        "combined_gradient_norm_reproduced": norm_passed,
        "treatment_scale": reconstruction.treatment_scale,
        "historical_treatment_parameter_delta_l2": reconstruction.historical_delta,
        "nominal_treatment_parameter_delta_l2": reconstruction.nominal_delta,
        "reconstructed_treatment_parameter_delta_l2": reconstruction.realized_delta,
        "treatment_parameter_displacement_reproduced": displacement_passed,
        "rtol": DIRECTION_NORM_RTOL,
        "atol": DIRECTION_NORM_ATOL,
    }


def _policy_record() -> Mapping[str, Any]:
    return {
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
    }


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    parent_path = args.crossed_attribution_report.resolve()
    parent, parent_digest = _load_crossed_result(parent_path)
    parent_metadata = _parent_factor_metadata(parent)

    bootstrap = _initialization_only(args)
    runtime = bootstrap["context"]["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    runtime_year = _runtime_year_for_update(
        bootstrap["context"]["schedule"],
        TARGET_ARTIFACT_UPDATE,
    )
    if runtime_year != EXPECTED_TARGET["runtime_year"]:
        raise ValueError("structured-precision runtime-year drift")
    initialized_structured = coupled_gpp_trainable_tree(bootstrap["initialized_parameters"])["structured_gpp"]
    _validate_initialized_structured_tree(initialized_structured)
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("structured-precision audit requires JAX float64 support")
    static_report = {
        "identity": {
            "git_head": _git_head(),
            "crossed_attribution_report_sha256": parent_digest,
        },
        "target": dict(EXPECTED_TARGET),
        "parent_cells": {
            "CC": parent_metadata["control_gpp"],
            "TC": parent_metadata["treatment_gpp"],
        },
    }

    reconstruction = reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)
    direction = _direction_integrity(reconstruction, parent)
    control_structured = coupled_gpp_trainable_tree(reconstruction.control_parameters)["structured_gpp"]
    treatment_structured = coupled_gpp_trainable_tree(reconstruction.treatment_parameters)["structured_gpp"]
    control_host = jax.device_get(control_structured)
    treatment_host = jax.device_get(treatment_structured)
    inventory = float32_delta_inventory(control_host, treatment_host)

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
    jax.block_until_ready((control_trace, control_trajectories))
    control_gpp_trace = np.asarray(
        control_trajectories.fluxes["gpp"].value,
        dtype=np.float64,
    ).reshape(-1)
    control_trace_reproduced = bool(
        np.allclose(
            control_gpp_trace,
            np.asarray(parent_metadata["control_gpp_trace"], dtype=np.float64),
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )
    templates, control_starts, control_discrete = _scenario_day_starts(
        reconstruction,
        microbatch,
        control_trace,
    )
    template_batch = _tree_day_batch(jax, templates, TARGET_DAY_INDEX)
    discrete_batch = _tree_day_batch(jax, control_discrete, TARGET_DAY_INDEX)
    control_state = jnp.asarray(control_starts[TARGET_DAY_INDEX])
    pft_index = int(parent_metadata["vegetation_pft_index"])
    factor_evaluator = _make_factor_evaluator(reconstruction)

    treatment_at_control = _replace_structured_gpp(
        reconstruction.control_parameters,
        treatment_structured,
    )
    production_direct = (
        _evaluate_record(
            factor_evaluator,
            reconstruction,
            reconstruction.control_parameters,
            template_batch,
            control_state,
            discrete_batch,
            pft_index,
        ),
        _evaluate_record(
            factor_evaluator,
            reconstruction,
            treatment_at_control,
            template_batch,
            control_state,
            discrete_batch,
            pft_index,
        ),
    )
    parent_cells_reproduced = bool(
        np.isclose(
            production_direct[0]["gpp"],
            parent_metadata["control_gpp"],
            rtol=FACTORIZATION_RTOL,
            atol=FACTORIZATION_ATOL,
        )
        and np.isclose(
            production_direct[1]["gpp"],
            parent_metadata["treatment_gpp"],
            rtol=FACTORIZATION_RTOL,
            atol=FACTORIZATION_ATOL,
        )
    )

    subtree_factorizations = {}
    treatment_group_only_gpp = {}
    treatment_without_group_gpp = {}
    evaluated_records = list(production_direct)
    for index, group in enumerate(STRUCTURED_GPP_GROUPS, start=1):
        only_tree = _mixed_subtree(control_structured, treatment_structured, (group,))
        without_tree = _mixed_subtree(
            control_structured,
            treatment_structured,
            tuple(name for name in STRUCTURED_GPP_GROUPS if name != group),
        )
        only = _evaluate_record(
            factor_evaluator,
            reconstruction,
            _replace_structured_gpp(reconstruction.control_parameters, only_tree),
            template_batch,
            control_state,
            discrete_batch,
            pft_index,
        )
        without = _evaluate_record(
            factor_evaluator,
            reconstruction,
            _replace_structured_gpp(reconstruction.control_parameters, without_tree),
            template_batch,
            control_state,
            discrete_batch,
            pft_index,
        )
        treatment_group_only_gpp[group] = float(only["gpp"])
        treatment_without_group_gpp[group] = float(without["gpp"])
        subtree_factorizations[group] = {
            "treatment_group_only": only,
            "treatment_without_group": without,
        }
        evaluated_records.extend((only, without))
        print(
            f"coupled_gpp structured_precision subtree={index}/{len(STRUCTURED_GPP_GROUPS)} name={group}",
            flush=True,
        )
    subtree_controls = classify_subtree_controls(
        control_gpp=float(production_direct[0]["gpp"]),
        treatment_gpp=float(production_direct[1]["gpp"]),
        treatment_group_only_gpp=treatment_group_only_gpp,
        treatment_without_group_gpp=treatment_without_group_gpp,
    )
    for group in STRUCTURED_GPP_GROUPS:
        subtree_controls["groups"][group]["factorizations"] = subtree_factorizations[group]

    float64_control = _interpolate_tree(
        jax,
        jnp,
        control_structured,
        treatment_structured,
        0.0,
        dtype=jnp.float64,
    )
    float64_treatment = _interpolate_tree(
        jax,
        jnp,
        control_structured,
        treatment_structured,
        1.0,
        dtype=jnp.float64,
    )
    if any(
        np.dtype(value.dtype) != np.dtype(np.float64)
        for tree in (float64_control, float64_treatment)
        for value in jax.tree_util.tree_leaves(tree)
    ):
        raise RuntimeError("structured-GPP float64 endpoint cast was not preserved")
    float64_direct = (
        _evaluate_record(
            factor_evaluator,
            reconstruction,
            _replace_structured_gpp(reconstruction.control_parameters, float64_control),
            template_batch,
            control_state,
            discrete_batch,
            pft_index,
        ),
        _evaluate_record(
            factor_evaluator,
            reconstruction,
            _replace_structured_gpp(reconstruction.control_parameters, float64_treatment),
            template_batch,
            control_state,
            discrete_batch,
            pft_index,
        ),
    )
    evaluated_records.extend(float64_direct)

    precision_paths = {
        "production_float32": [],
        "float64_weight_interpolation": [],
    }
    memberships = []
    for index, fraction in enumerate(INTERPOLATION_FRACTIONS, start=1):
        production_tree = _interpolate_tree(
            jax,
            jnp,
            control_structured,
            treatment_structured,
            fraction,
            dtype=jnp.float32,
        )
        membership = float32_interpolation_membership(
            control_host,
            treatment_host,
            jax.device_get(production_tree),
        )
        memberships.append({"fraction": fraction, **membership})
        production = {
            "fraction": fraction,
            "weight_dtype": "float32",
            **_evaluate_record(
                factor_evaluator,
                reconstruction,
                _replace_structured_gpp(
                    reconstruction.control_parameters,
                    production_tree,
                ),
                template_batch,
                control_state,
                discrete_batch,
                pft_index,
            ),
        }
        float64_tree = _interpolate_tree(
            jax,
            jnp,
            float64_control,
            float64_treatment,
            fraction,
            dtype=jnp.float64,
        )
        diagnostic = {
            "fraction": fraction,
            "weight_dtype": "float64",
            **_evaluate_record(
                factor_evaluator,
                reconstruction,
                _replace_structured_gpp(
                    reconstruction.control_parameters,
                    float64_tree,
                ),
                template_batch,
                control_state,
                discrete_batch,
                pft_index,
            ),
        }
        precision_paths["production_float32"].append(production)
        precision_paths["float64_weight_interpolation"].append(diagnostic)
        evaluated_records.extend((production, diagnostic))
        print(
            f"coupled_gpp structured_precision interpolation={index}/"
            f"{len(INTERPOLATION_FRACTIONS)} fraction={fraction:g}",
            flush=True,
        )

    path_classification = classify_precision_paths(
        fractions=INTERPOLATION_FRACTIONS,
        production_gpp=tuple(record["gpp"] for record in precision_paths["production_float32"]),
        float64_weight_gpp=tuple(record["gpp"] for record in precision_paths["float64_weight_interpolation"]),
    )
    production_endpoints_exact = bool(
        _records_exact(precision_paths["production_float32"][0], production_direct[0])
        and _records_exact(
            precision_paths["production_float32"][-1],
            production_direct[1],
        )
    )
    float64_endpoints_exact = bool(
        _records_exact(
            precision_paths["float64_weight_interpolation"][0],
            float64_direct[0],
        )
        and _records_exact(
            precision_paths["float64_weight_interpolation"][-1],
            float64_direct[1],
        )
    )
    fraction_registry_passed = bool(
        all(
            tuple(record["fraction"] for record in path) == INTERPOLATION_FRACTIONS for path in precision_paths.values()
        )
    )
    factors_passed = all(_record_is_valid(record) for record in evaluated_records)
    inventory_passed = _inventory_conserved(inventory)
    membership_passed = _membership_conserved(memberships)
    direction_passed = bool(
        direction["combined_gradient_norm_reproduced"] and direction["treatment_parameter_displacement_reproduced"]
    )
    engineering_passed = bool(
        direction_passed
        and control_trace_reproduced
        and parent_cells_reproduced
        and inventory_passed
        and membership_passed
        and factors_passed
        and fraction_registry_passed
        and production_endpoints_exact
        and float64_endpoints_exact
    )
    classification = str(path_classification["classification"]) if engineering_passed else "engineering_gate_failed"
    if not engineering_passed:
        next_gate = "inspect_written_engineering_gate_report_before_rerun"
    elif classification == "float32_weight_quantization_dominant_for_sampled_jump":
        next_gate = "pause_for_user_discussion_of_update_realization_or_master_weight_precision"
    elif classification == "structured_gpp_conditioning_persists_with_float64_weights":
        next_gate = "pause_for_user_discussion_of_conditioning_or_gpp_output_parameterization"
    else:
        next_gate = "pause_for_user_discussion_of_one_narrower_zero_training_followup"

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if engineering_passed else "engineering_gate_failed",
        "classification": classification,
        **static_report,
        "direction_reconstruction": direction,
        "control_gpp_trace": control_gpp_trace.tolist(),
        "structured_gpp_delta_inventory": inventory,
        "subtree_controls": subtree_controls,
        "precision_paths": {
            **precision_paths,
            "production_float32_membership": memberships,
            "direct_endpoints": {
                "production_float32": list(production_direct),
                "float64_weight_interpolation": list(float64_direct),
            },
            "classification": path_classification,
        },
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "direction_reproduced": direction_passed,
            "control_trace_reproduced": control_trace_reproduced,
            "parent_CC_TC_reproduced": parent_cells_reproduced,
            "structured_gpp_registry_passed": frozenset(control_structured) == frozenset(STRUCTURED_GPP_GROUPS)
            and frozenset(treatment_structured) == frozenset(STRUCTURED_GPP_GROUPS),
            "structured_gpp_inventory_conserved": inventory_passed,
            "float32_membership_conserved": membership_passed,
            "all_factors_finite_and_routed_gpp_nonnegative": factors_passed,
            "fixed_fraction_registry_passed": fraction_registry_passed,
            "production_float32_endpoints_exact": production_endpoints_exact,
            "float64_weight_endpoints_exact": float64_endpoints_exact,
            "factorization_tolerance": {
                "rtol": FACTORIZATION_RTOL,
                "atol": FACTORIZATION_ATOL,
            },
        },
        "policy": _policy_record(),
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
    }
    if not engineering_passed:
        print(
            "coupled_gpp structured_precision engineering_gate_failed "
            f"direction={direction_passed} trace={control_trace_reproduced} "
            f"parent_cells={parent_cells_reproduced} inventory={inventory_passed} "
            f"membership={membership_passed} factors={factors_passed} "
            f"fractions={fraction_registry_passed} "
            f"production_endpoints={production_endpoints_exact} "
            f"float64_endpoints={float64_endpoints_exact}",
            flush=True,
        )
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _crossed_parser()
    parser.add_argument("--crossed-attribution-report", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
