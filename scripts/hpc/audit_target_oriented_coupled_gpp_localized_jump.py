"""Trace the first rollout divergence behind the localized artifact-550 jump."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistence import (
    load_recovery,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    load_coupled_gpp_artifact_aware_persistent_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_localized_jump import (
    MATERIAL_NORMALIZED_STATE_CHANGE,
    classify_material_days,
    select_largest_positive_window,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
    TRAINABLE_PARAMETER_COUNT,
    replace_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_science_window_sufficient_statistics,
    science_window_from_sufficient_statistics,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_direction import (
    _measure_direct_artifact_gradients,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_zero_displacement import (
    _historical_records,
    _load_parent_evidence,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    _add_gradient,
    _zero_gradient,
)
from scripts.hpc.calibrate_target_oriented_coupled_gpp_objective import _state_gradient
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _horizon_gradient,
    _science_record,
)
from scripts.hpc.probe_successor_multibatch_direction import (
    _make_flat_direction,
    _make_flattened_unit_gradient,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _initialization_only,
)
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _parser as _persistent_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import (
    _git_head,
    _make_dtype_preserving_direct_virtual_step,
)
from scripts.hpc.run_target_oriented_coupled_gpp_update2_geometry import _scaled_parameters
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _hard_failure_count,
    _merge_statistics,
    _slice,
    _zero_hard,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import _hard_host

SCHEMA_VERSION = "target_oriented_coupled_gpp_localized_jump_trace_result_v1"
TARGET_ARTIFACT_UPDATE = 550
TARGET_LOGICAL_UPDATE = 3
EXPECTED_SOURCE_COUNT = 43
EXPECTED_NONZERO_SOURCE_COUNT = 26
EXPECTED_NONZERO_DIRECT_COUNT = 24
DIRECTION_NORM_RTOL = 1.0e-5
DIRECTION_NORM_ATOL = 1.0e-10
FORWARD_REPRODUCTION_RTOL = 1.0e-6
FORWARD_REPRODUCTION_ATOL = 1.0e-10


class LocalizedJumpReconstruction(NamedTuple):
    contract: Any
    recovery_digest: str
    recovered: Mapping[str, Any]
    historical_target: Mapping[str, Any]
    plan: Mapping[str, Any]
    bootstrap: Mapping[str, Any]
    context: Mapping[str, Any]
    runtime: Mapping[str, Any]
    jax: Any
    jnp: Any
    control_parameters: Any
    treatment_parameters: Any
    direct: Mapping[str, Any]
    scheduled: Mapping[str, Any]
    combined_norm: float
    norm_matches: bool
    treatment_scale: float
    historical_delta: float
    nominal_delta: float
    nominal_delta_matches: bool
    realized_delta: float


def _direction_reconstruction_plan(rejected: Mapping[str, Any]) -> Mapping[str, Any]:
    proposal = rejected.get("proposal", {})
    direction = proposal.get("direction", {})
    registry = tuple(str(value) for value in direction.get("source_registry", ()))
    selected = direction.get("selected_minimum_norm_direction", {})
    raw_weights = selected.get("normalized_simplex_weights", {})
    if not registry or set(raw_weights) != set(registry) or len(set(registry)) != len(registry):
        raise ValueError("localized-jump direction registry drift")
    weights = {name: float(raw_weights[name]) for name in registry}
    if not all(np.isfinite(value) and value >= 0.0 for value in weights.values()) or not np.isclose(
        sum(weights.values()), 1.0, rtol=1.0e-8, atol=1.0e-10
    ):
        raise ValueError("localized-jump simplex weights are invalid")

    direct_prefix = f"{DIRECT_SOURCE}.artifact_"
    nonzero = tuple(name for name in registry if weights[name] > 0.0)
    nonzero_direct = tuple(name for name in nonzero if name.startswith(direct_prefix))
    direct_updates = tuple(int(name.removeprefix(direct_prefix)) for name in nonzero_direct)
    non_gpp = tuple(name for name in nonzero if not name.startswith(direct_prefix))
    return {
        "source_registry": registry,
        "weights": weights,
        "nonzero_sources": nonzero,
        "nonzero_direct_sources": nonzero_direct,
        "nonzero_direct_updates": direct_updates,
        "nonzero_scheduled_sources": non_gpp,
        "historical_combined_gradient_norm": float(direction["combined_gradient_norm"]),
        "full_parameter_step_l2": float(direction["full_parameter_step_l2"]),
    }


def _mean_gradient(jax: Any, gradient: Any, count: int) -> Any:
    if count <= 0:
        raise ValueError("localized-jump gradient mean requires a positive count")
    return jax.tree_util.tree_map(lambda value: value / count, gradient)


def _measure_scheduled_non_gpp_gradients(
    args: Any,
    bootstrap: Mapping[str, Any],
    *,
    parameters: Any,
    trainable: Any,
    updates: Sequence[int],
    flatten_gradient: Any,
) -> Mapping[str, Any]:
    context = bootstrap["context"]
    jax = context["runtime"]["jax"]
    jnp = context["runtime"]["jnp"]
    sums = {
        STATE_SOURCE: _zero_gradient(jax, jnp, trainable),
        THIRTY_SOURCE: _zero_gradient(jax, jnp, trainable),
    }
    records = []
    started = time.perf_counter()
    for ordinal, update in enumerate(updates, start=1):
        artifact_started = time.perf_counter()
        artifact = _load_artifact(args, context, int(update))
        state, state_gradient, state_forward, state_backward = _state_gradient(
            jax,
            jnp,
            trainable,
            artifact.batch,
            bootstrap["state_steps"],
            bootstrap["layout"],
        )
        thirty, hard, thirty_gradient, thirty_timing = _horizon_gradient(
            jax,
            jnp,
            parameters,
            trainable,
            artifact.batch,
            bootstrap["steps"],
            bootstrap["gradient_steps"],
            horizon=30,
            flux_basis=bootstrap["flux_basis"],
        )
        sums[STATE_SOURCE] = _add_gradient(jax, jnp, sums[STATE_SOURCE], state_gradient)
        sums[THIRTY_SOURCE] = _add_gradient(jax, jnp, sums[THIRTY_SOURCE], thirty_gradient)
        jax.block_until_ready(tuple(sums.values()))
        hard_host = _hard_host(jax.device_get(hard))
        if _hard_failure_count(hard_host) != 0:
            raise RuntimeError(f"localized-jump scheduled artifact {update} failed the hard gate")
        records.append(
            {
                "artifact_update": int(update),
                "runtime_year": int(context["schedule"]["entries"][int(update) - 1]["runtime_year"]),
                "state_loss": float(np.asarray(state.total)),
                "thirty_day_loss": float(np.asarray(thirty.total)),
                "thirty_day_hard_counts": hard_host,
                "timing_seconds": {
                    "state_forward": state_forward,
                    "state_gradient": state_backward,
                    "thirty_day": thirty_timing,
                    "total": time.perf_counter() - artifact_started,
                },
            }
        )
        print(
            f"coupled_gpp localized_jump scheduled_gradient={ordinal}/{len(updates)} artifact={update}",
            flush=True,
        )
        del artifact, state, state_gradient, thirty, thirty_gradient, hard

    units = {}
    reports = {}
    for name in (STATE_SOURCE, THIRTY_SOURCE):
        mean = _mean_gradient(jax, sums[name], len(updates))
        unit, norm, nonfinite = flatten_gradient(mean)
        jax.block_until_ready((unit, norm, nonfinite))
        norm_host = float(np.asarray(norm))
        nonfinite_host = int(np.asarray(nonfinite))
        if (
            nonfinite_host != 0
            or not np.isfinite(norm_host)
            or norm_host <= 0.0
            or int(unit.size) != TRAINABLE_PARAMETER_COUNT
        ):
            raise RuntimeError(f"localized-jump scheduled gradient {name} is invalid")
        units[name] = unit
        reports[name] = {
            "source": name,
            "l2_norm": norm_host,
            "nonfinite_count": nonfinite_host,
            "element_count": int(unit.size),
        }
        del mean, norm, nonfinite
    return {
        "unit_gradients": units,
        "gradient_reports": reports,
        "artifact_records": records,
        "timing_seconds": {"total": time.perf_counter() - started},
    }


def _make_trace_evaluator(bootstrap: Mapping[str, Any]) -> Any:
    context = bootstrap["context"]
    runtime = context["runtime"]
    jax = runtime["jax"]

    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        rollout_daily_operator,
        trajectories_from_rollout,
    )

    dynamic = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=bootstrap["head"],
        retained_tail_template=context["template"],
    )

    @jax.jit
    def evaluate(parameters: Any, microbatch: Any) -> Any:
        templates = jax.tree_util.tree_map(lambda value: value[0], microbatch.inputs)
        day_inputs = jax.tree_util.tree_map(lambda value: value[0], microbatch.day_inputs)

        def transition(values: Any, inputs: Any) -> Any:
            return dynamic(values, inputs, day_inputs)

        trace = rollout_daily_operator(
            parameters,
            templates,
            transition=transition,
            contract=runtime["contract"],
            rematerialize=False,
        )
        return trace, trajectories_from_rollout(trace, runtime["contract"])

    return evaluate


def _state_location(contract: Any, global_index: int) -> Mapping[str, Any]:
    index = int(global_index)
    matches = tuple(
        leaf
        for leaf in contract.state_leaves
        if leaf.start is not None and leaf.stop is not None and int(leaf.start) <= index < int(leaf.stop)
    )
    if len(matches) != 1:
        raise ValueError(f"localized-jump state coordinate {index} has no unique owner")
    leaf = matches[0]
    local = tuple(int(value) for value in np.unravel_index(index - int(leaf.start), tuple(leaf.shape)))
    return {
        "global_state_index": index,
        "state_leaf_key": str(leaf.key),
        "component": str(leaf.component),
        "path": [str(value) for value in leaf.path],
        "local_coordinate": list(local),
        "axis_coordinates": [
            {"axis": str(name), "index": value} for name, value in zip(leaf.axis_names, local, strict=True)
        ],
    }


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(()).item()
    return array.tolist()


def _tree_path_string(path: Sequence[Any]) -> str:
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "name"):
            parts.append(str(entry.name))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return ".".join(parts)


def _discrete_day_comparison(jax: Any, control: Any, treatment: Any, day: int) -> Mapping[str, Any]:
    control_with_paths, control_tree = jax.tree_util.tree_flatten_with_path(control)
    treatment_with_paths, treatment_tree = jax.tree_util.tree_flatten_with_path(treatment)
    if control_tree != treatment_tree or len(control_with_paths) != len(treatment_with_paths):
        raise ValueError("localized-jump discrete-state trees differ")
    details = []
    total = 0
    for (path, first), (other_path, second) in zip(
        control_with_paths,
        treatment_with_paths,
        strict=True,
    ):
        if path != other_path:
            raise ValueError("localized-jump discrete-state paths differ")
        left = np.asarray(first)[day]
        right = np.asarray(second)[day]
        mismatch = left != right
        count = int(np.sum(mismatch))
        total += count
        if count:
            flat_index = int(np.flatnonzero(np.ravel(mismatch))[0])
            coordinate = tuple(int(value) for value in np.unravel_index(flat_index, mismatch.shape))
            details.append(
                {
                    "path": _tree_path_string(path),
                    "mismatch_count": count,
                    "first_mismatch_coordinate": list(coordinate),
                    "control_value": _json_value(left[coordinate] if coordinate else left),
                    "treatment_value": _json_value(right[coordinate] if coordinate else right),
                }
            )
    return {"mismatch_count": total, "mismatching_leaves": details}


def _masked_trajectory(value: Any, defined: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(value, dtype=np.float64), np.asarray(defined, dtype=np.bool_)


def _science_fluxes(trajectories: Any) -> Mapping[str, tuple[np.ndarray, np.ndarray]]:
    base = {name: _masked_trajectory(value.value, value.defined) for name, value in trajectories.fluxes.items()}
    required = {
        "gpp",
        "npp",
        "maintenance_respiration",
        "growth_respiration",
        "heterotrophic_respiration",
    }
    if set(base) != required:
        raise ValueError("localized-jump base science-flux registry drift")
    maintenance, maintenance_defined = base["maintenance_respiration"]
    growth, growth_defined = base["growth_respiration"]
    autotrophic = maintenance + growth
    autotrophic_defined = maintenance_defined & growth_defined
    npp, npp_defined = base["npp"]
    heterotrophic, heterotrophic_defined = base["heterotrophic_respiration"]
    nep = npp - heterotrophic
    nep_defined = npp_defined & heterotrophic_defined
    return {
        "gpp": base["gpp"],
        "maintenance_respiration": base["maintenance_respiration"],
        "growth_respiration": base["growth_respiration"],
        "autotrophic_respiration": (autotrophic, autotrophic_defined),
        "npp": base["npp"],
        "heterotrophic_respiration": base["heterotrophic_respiration"],
        "nep": (nep, nep_defined),
        "nee": (-nep, nep_defined),
    }


def _trajectory_day_comparison(
    control: Mapping[str, tuple[np.ndarray, np.ndarray]],
    treatment: Mapping[str, tuple[np.ndarray, np.ndarray]],
    day: int,
) -> Mapping[str, Any]:
    if tuple(control) != tuple(treatment):
        raise ValueError("localized-jump trajectory registries differ")
    result = {}
    for name in control:
        first, first_defined = control[name]
        second, second_defined = treatment[name]
        if first.shape != second.shape or first_defined.shape != second_defined.shape:
            raise ValueError(f"localized-jump trajectory shape drift for {name}")
        result[name] = {
            "control": _json_value(first[day]),
            "treatment": _json_value(second[day]),
            "delta": _json_value(second[day] - first[day]),
            "control_defined": _json_value(first_defined[day]),
            "treatment_defined": _json_value(second_defined[day]),
        }
    return result


def _state_day_comparison(
    contract: Any,
    control: np.ndarray,
    treatment: np.ndarray,
    control_defined: np.ndarray,
    treatment_defined: np.ndarray,
    scale: np.ndarray,
    day: int,
) -> Mapping[str, Any]:
    left = control[day]
    right = treatment[day]
    left_defined = control_defined[day]
    right_defined = treatment_defined[day]
    if (
        left.shape != right.shape
        or left_defined.shape != left.shape
        or right_defined.shape != left.shape
        or scale.shape != left.shape
    ):
        raise ValueError("localized-jump continuous-state shape drift")
    comparable = left_defined & right_defined
    if np.any(comparable & (~np.isfinite(left) | ~np.isfinite(right))):
        raise RuntimeError("localized-jump trace contains a nonfinite defined state")
    absolute = np.where(comparable, np.abs(right - left), 0.0)
    normalized = absolute / scale
    normalized_index = int(np.argmax(normalized))
    absolute_index = int(np.argmax(absolute))
    mask_mismatch_indices = np.flatnonzero(left_defined != right_defined)
    material_indices = np.flatnonzero(normalized >= MATERIAL_NORMALIZED_STATE_CHANGE)
    ordered_material = material_indices[np.argsort(normalized[material_indices])[::-1]][:20]

    def coordinate(index: int) -> Mapping[str, Any]:
        return _state_location(contract, int(index)) | {
            "control_value": float(left[index]),
            "treatment_value": float(right[index]),
            "absolute_change": float(absolute[index]),
            "training_scale": float(scale[index]),
            "normalized_change": float(normalized[index]),
        }

    return {
        "defined_mask_mismatch_count": int(mask_mismatch_indices.size),
        "defined_mask_mismatches": [
            _state_location(contract, int(index))
            | {
                "control_defined": bool(left_defined[index]),
                "treatment_defined": bool(right_defined[index]),
            }
            for index in mask_mismatch_indices[:20]
        ],
        "maximum_absolute_state_change": float(absolute[absolute_index]),
        "maximum_absolute_state_change_coordinate": coordinate(absolute_index),
        "maximum_normalized_state_change": float(normalized[normalized_index]),
        "maximum_normalized_state_change_coordinate": coordinate(normalized_index),
        "material_continuous_state_change_count": int(material_indices.size),
        "largest_material_continuous_state_changes": [coordinate(int(index)) for index in ordered_material],
    }


def _window_scores(
    jax: Any,
    jnp: Any,
    artifact: Any,
    *,
    control_parameters: Any,
    treatment_parameters: Any,
    steps: Mapping[str, Any],
    flux_basis: tuple[str, ...],
) -> Mapping[str, Any]:
    records = []
    control_statistics = []
    treatment_statistics = []
    control_hard = _zero_hard(jnp)
    treatment_hard = _zero_hard(jnp)
    for index in range(8):
        microbatch = _slice(artifact.batch.seven_day, index, index + 1)
        first_statistics, first_hard = steps["seven_forward"](control_parameters, microbatch)
        second_statistics, second_hard = steps["seven_forward"](treatment_parameters, microbatch)
        first = science_window_from_sufficient_statistics(first_statistics, flux_basis=flux_basis)
        second = science_window_from_sufficient_statistics(second_statistics, flux_basis=flux_basis)
        jax.block_until_ready((first, second, first_hard, second_hard))
        first_hard_host = _hard_host(jax.device_get(first_hard))
        second_hard_host = _hard_host(jax.device_get(second_hard))
        if _hard_failure_count(first_hard_host) != 0 or _hard_failure_count(second_hard_host) != 0:
            raise RuntimeError(f"localized-jump seven-day window {index} failed the hard gate")
        records.append(
            {
                "window_index": index,
                "window_ordinal": index + 1,
                "control_score": float(np.asarray(first.total)),
                "treatment_score": float(np.asarray(second.total)),
                "control_components": _science_record(first),
                "treatment_components": _science_record(second),
                "control_hard_counts": first_hard_host,
                "treatment_hard_counts": second_hard_host,
            }
        )
        control_statistics.append(first_statistics)
        treatment_statistics.append(second_statistics)
        control_hard = _add_hard(jax, control_hard, first_hard)
        treatment_hard = _add_hard(jax, treatment_hard, second_hard)
        print(f"coupled_gpp localized_jump window={index + 1}/8", flush=True)

    control_total = _merge_statistics(add_science_window_sufficient_statistics, control_statistics)
    treatment_total = _merge_statistics(add_science_window_sufficient_statistics, treatment_statistics)
    control_result = science_window_from_sufficient_statistics(control_total, flux_basis=flux_basis)
    treatment_result = science_window_from_sufficient_statistics(treatment_total, flux_basis=flux_basis)
    jax.block_until_ready((control_result, treatment_result, control_hard, treatment_hard))
    return {
        "selection": select_largest_positive_window(records),
        "aggregate_control": _science_record(control_result),
        "aggregate_treatment": _science_record(treatment_result),
        "aggregate_control_hard_counts": _hard_host(jax.device_get(control_hard)),
        "aggregate_treatment_hard_counts": _hard_host(jax.device_get(treatment_hard)),
    }


def reconstruct_localized_jump_parameters(
    args: Any,
    *,
    bootstrap: Mapping[str, Any] | None = None,
) -> LocalizedJumpReconstruction:
    """Rebuild the accepted control and historical rejected tiny step in memory."""

    from jax.flatten_util import ravel_pytree

    contract = load_coupled_gpp_artifact_aware_persistent_training_contract(args.artifact_aware_persistent_contract)
    parent_report, rejected = _load_parent_evidence(
        args.parent_persistent_training_report.resolve(),
        args.parent_rejected_proposal.resolve(),
        contract_sha256=contract.sha256,
    )
    recovery_digest = sha256_file(args.parent_artifact_aware_recovery)
    if recovery_digest != parent_report.get("persistence", {}).get("recovery_sha256"):
        raise ValueError("localized-jump recovery identity drift")
    recovered = load_recovery(
        args.parent_artifact_aware_recovery,
        identity=parent_report["identity"],
        parameter_count=TRAINABLE_PARAMETER_COUNT,
    )
    if recovered is None or int(recovered["completed_logical_update"]) != 2:
        raise ValueError("localized-jump recovery progress drift")

    logical_index = contract.logical_updates.index(TARGET_LOGICAL_UPDATE)
    scheduled_updates = tuple(int(value) for value in contract.train_panels[logical_index][0])
    historical, smallest = _historical_records(rejected, expected_updates=scheduled_updates)
    historical_target = next(value for value in historical if int(value["artifact_update"]) == TARGET_ARTIFACT_UPDATE)
    plan = _direction_reconstruction_plan(rejected)
    if (
        len(plan["source_registry"]) != EXPECTED_SOURCE_COUNT
        or len(plan["nonzero_sources"]) != EXPECTED_NONZERO_SOURCE_COUNT
        or len(plan["nonzero_direct_sources"]) != EXPECTED_NONZERO_DIRECT_COUNT
        or plan["nonzero_scheduled_sources"] != (STATE_SOURCE, THIRTY_SOURCE)
        or float(
            rejected["proposal"]["direction"]["selected_minimum_norm_direction"]["normalized_simplex_weights"][
                SEVEN_SOURCE
            ]
        )
        != 0.0
        or TARGET_ARTIFACT_UPDATE not in scheduled_updates
    ):
        raise ValueError("localized-jump frozen reconstruction plan drift")

    bootstrap = _initialization_only(args) if bootstrap is None else bootstrap
    context = bootstrap["context"]
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    trainable = jax.device_put(recovered["trainable"])
    control_parameters = replace_coupled_gpp_trainable(bootstrap["initialized_parameters"], trainable)
    flatten_gradient = _make_flattened_unit_gradient(jax)

    direct = _measure_direct_artifact_gradients(
        args,
        context,
        trainable=trainable,
        direct_steps=bootstrap["direct_steps"],
        updates=plan["nonzero_direct_updates"],
        flatten_gradient=flatten_gradient,
        label="localized_jump_nonzero",
    )
    direct_units = {
        report["source"]: unit
        for report, unit in zip(direct["gradient_reports"], direct["unit_gradients"], strict=True)
    }
    scheduled = _measure_scheduled_non_gpp_gradients(
        args,
        bootstrap,
        parameters=control_parameters,
        trainable=trainable,
        updates=scheduled_updates,
        flatten_gradient=flatten_gradient,
    )
    unit_by_source = direct_units | scheduled["unit_gradients"]
    if set(unit_by_source) != set(plan["nonzero_sources"]):
        raise RuntimeError("localized-jump reconstructed nonzero source registry drift")

    zero = jnp.zeros((TRAINABLE_PARAMETER_COUNT,), dtype=jnp.float32)
    unit_matrix = jnp.stack(tuple(unit_by_source.get(name, zero) for name in plan["source_registry"]))
    weights = jnp.asarray(
        [plan["weights"][name] for name in plan["source_registry"]],
        dtype=jnp.float64,
    )
    flat_direction, combined_norm, direction_nonfinite = _make_flat_direction(jax)(unit_matrix, weights)
    jax.block_until_ready((flat_direction, combined_norm, direction_nonfinite))
    combined_norm_host = float(np.asarray(combined_norm))
    norm_matches = bool(
        np.isclose(
            combined_norm_host,
            plan["historical_combined_gradient_norm"],
            rtol=DIRECTION_NORM_RTOL,
            atol=DIRECTION_NORM_ATOL,
        )
    )
    if int(np.asarray(direction_nonfinite)) != 0 or not norm_matches:
        raise RuntimeError("localized-jump combined direction does not reproduce the rejected proposal")

    flat_template, unravel = ravel_pytree(trainable)
    if int(flat_template.size) != TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("localized-jump trainable parameter width drift")
    direction_tree = unravel(flat_direction)
    proposed, proposal_nonfinite = _make_dtype_preserving_direct_virtual_step(jax)(
        trainable,
        direction_tree,
        combined_norm,
        np.asarray(plan["full_parameter_step_l2"], dtype=np.float64),
    )
    treatment_trainable = _scaled_parameters(jax, trainable, proposed, float(smallest["scale"]))
    delta_tree = jax.tree_util.tree_map(lambda first, second: second - first, trainable, treatment_trainable)
    flat_delta, _ = ravel_pytree(delta_tree)
    parameter_delta_l2 = jnp.linalg.norm(flat_delta)
    jax.block_until_ready((treatment_trainable, proposal_nonfinite, parameter_delta_l2))
    parameter_delta_l2_host = float(np.asarray(parameter_delta_l2))
    nominal_parameter_delta_l2 = float(smallest["scale"]) * plan["full_parameter_step_l2"]
    nominal_delta_matches = bool(
        np.isclose(
            nominal_parameter_delta_l2,
            float(smallest["parameter_delta_l2"]),
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    )
    if (
        int(np.asarray(proposal_nonfinite)) != 0
        or not np.isfinite(parameter_delta_l2_host)
        or parameter_delta_l2_host <= 0.0
        or not nominal_delta_matches
    ):
        raise RuntimeError("localized-jump treatment displacement is invalid")
    print(
        "coupled_gpp localized_jump direction_reproduced "
        f"combined_norm={combined_norm_host:.12g} "
        f"nominal_delta={nominal_parameter_delta_l2:.12g} "
        f"realized_delta={parameter_delta_l2_host:.12g}",
        flush=True,
    )
    treatment_parameters = replace_coupled_gpp_trainable(bootstrap["initialized_parameters"], treatment_trainable)
    del direct["unit_gradients"], scheduled["unit_gradients"]
    del unit_matrix, flat_direction, flat_template, direction_tree, proposed, delta_tree, flat_delta, zero
    del direct_units, unit_by_source
    gc.collect()

    return LocalizedJumpReconstruction(
        contract=contract,
        recovery_digest=recovery_digest,
        recovered=recovered,
        historical_target=historical_target,
        plan=plan,
        bootstrap=bootstrap,
        context=context,
        runtime=runtime,
        jax=jax,
        jnp=jnp,
        control_parameters=control_parameters,
        treatment_parameters=treatment_parameters,
        direct=direct,
        scheduled=scheduled,
        combined_norm=combined_norm_host,
        norm_matches=norm_matches,
        treatment_scale=float(smallest["scale"]),
        historical_delta=float(smallest["parameter_delta_l2"]),
        nominal_delta=nominal_parameter_delta_l2,
        nominal_delta_matches=nominal_delta_matches,
        realized_delta=parameter_delta_l2_host,
    )


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    reconstruction = reconstruct_localized_jump_parameters(args)
    contract = reconstruction.contract
    recovery_digest = reconstruction.recovery_digest
    recovered = reconstruction.recovered
    historical_target = reconstruction.historical_target
    plan = reconstruction.plan
    bootstrap = reconstruction.bootstrap
    context = reconstruction.context
    runtime = reconstruction.runtime
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    control_parameters = reconstruction.control_parameters
    treatment_parameters = reconstruction.treatment_parameters
    direct = reconstruction.direct
    scheduled = reconstruction.scheduled
    combined_norm_host = reconstruction.combined_norm
    norm_matches = reconstruction.norm_matches
    nominal_parameter_delta_l2 = reconstruction.nominal_delta
    nominal_delta_matches = reconstruction.nominal_delta_matches
    parameter_delta_l2_host = reconstruction.realized_delta

    artifact = _load_artifact(args, context, TARGET_ARTIFACT_UPDATE)
    window_scores = _window_scores(
        jax,
        jnp,
        artifact,
        control_parameters=control_parameters,
        treatment_parameters=treatment_parameters,
        steps=bootstrap["steps"],
        flux_basis=bootstrap["flux_basis"],
    )
    control_reproduced = bool(
        np.isclose(
            float(window_scores["aggregate_control"]["total"]),
            float(historical_target["gradient_baseline"]),
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )
    treatment_reproduced = bool(
        np.isclose(
            float(window_scores["aggregate_treatment"]["total"]),
            float(historical_target["smallest_nonzero_forward"]),
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )
    if not control_reproduced or not treatment_reproduced:
        raise RuntimeError("localized-jump artifact-550 aggregate does not reproduce parent evidence")

    selected_index = int(window_scores["selection"]["selected_window_index"])
    selected_batch = _slice(artifact.batch.seven_day, selected_index, selected_index + 1)
    trace_evaluator = _make_trace_evaluator(bootstrap)
    control_trace, control_trajectories = trace_evaluator(control_parameters, selected_batch)
    treatment_trace, treatment_trajectories = trace_evaluator(treatment_parameters, selected_batch)
    jax.block_until_ready((control_trace, control_trajectories, treatment_trace, treatment_trajectories))
    control_trace = jax.device_get(control_trace)
    treatment_trace = jax.device_get(treatment_trace)
    control_trajectories = jax.device_get(control_trajectories)
    treatment_trajectories = jax.device_get(treatment_trajectories)

    control_state = np.asarray(control_trace.result.next_state.other, dtype=np.float64)
    treatment_state = np.asarray(treatment_trace.result.next_state.other, dtype=np.float64)
    control_defined = np.asarray(control_trace.result.next_state_defined.other, dtype=np.bool_)
    treatment_defined = np.asarray(treatment_trace.result.next_state_defined.other, dtype=np.bool_)
    state_scale = np.asarray(context["scales"].state.other, dtype=np.float64)
    control_fluxes = _science_fluxes(control_trajectories)
    treatment_fluxes = _science_fluxes(treatment_trajectories)
    control_stocks = {
        name: _masked_trajectory(value.value, value.defined) for name, value in control_trajectories.stocks.items()
    }
    treatment_stocks = {
        name: _masked_trajectory(value.value, value.defined) for name, value in treatment_trajectories.stocks.items()
    }

    day_records = []
    for day in range(7):
        state_record = _state_day_comparison(
            runtime["contract"],
            control_state,
            treatment_state,
            control_defined,
            treatment_defined,
            state_scale,
            day,
        )
        discrete = _discrete_day_comparison(
            jax,
            control_trace.result.next_discrete_state,
            treatment_trace.result.next_discrete_state,
            day,
        )
        day_records.append(
            {
                "day_index": day,
                "day_ordinal": day + 1,
                "discrete_mismatch_count": int(discrete["mismatch_count"]),
                "discrete_mismatches": discrete["mismatching_leaves"],
                **state_record,
                "science_fluxes": _trajectory_day_comparison(
                    control_fluxes,
                    treatment_fluxes,
                    day,
                ),
                "science_stocks": _trajectory_day_comparison(
                    control_stocks,
                    treatment_stocks,
                    day,
                ),
            }
        )
    material = classify_material_days(day_records)
    first_day = material["first_material_day_index"]
    first_ordinal = None if first_day is None else int(first_day) + 1
    next_gate = {
        "discrete_state_change_first": "audit_the_source_backed_discrete_trigger_on_the_first_material_day",
        "defined_mask_change_first": "audit_the_defined_status_transition_on_the_first_material_day",
        "continuous_state_change_first": "trace_the_first_material_state_leaf_to_its_upstream_GPP_and_retained_paths",
        "no_material_state_or_branch_change": "trace_subthreshold_daily_flux_displacement_before_state_materiality",
    }[material["classification"]]

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "classification": material["classification"],
        "identity": {
            "git_head": _git_head(),
            "artifact_aware_persistent_contract_sha256": contract.sha256,
            "parent_persistent_training_report_sha256": sha256_file(args.parent_persistent_training_report),
            "parent_rejected_proposal_sha256": sha256_file(args.parent_rejected_proposal),
            "parent_recovery_sha256": recovery_digest,
        },
        "policy": {
            "logical_updates_applied": 0,
            "nonzero_direction_sources_recomputed": len(plan["nonzero_sources"]),
            "zero_weight_source_rows_recomputed": 0,
            "parameter_displacements_applied_in_memory": 1,
            "trainable_parameters_written": False,
            "checkpoint_written": False,
            "recovery_written": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "direction_reconstruction": {
            "source_count": len(plan["source_registry"]),
            "nonzero_source_count": len(plan["nonzero_sources"]),
            "nonzero_direct_source_count": len(plan["nonzero_direct_sources"]),
            "nonzero_scheduled_sources": list(plan["nonzero_scheduled_sources"]),
            "zero_weight_source_count": len(plan["source_registry"]) - len(plan["nonzero_sources"]),
            "historical_combined_gradient_norm": plan["historical_combined_gradient_norm"],
            "reconstructed_combined_gradient_norm": combined_norm_host,
            "combined_gradient_norm_reproduced": norm_matches,
            "historical_full_parameter_step_l2": plan["full_parameter_step_l2"],
            "treatment_scale": reconstruction.treatment_scale,
            "historical_treatment_parameter_delta_l2": reconstruction.historical_delta,
            "nominal_treatment_parameter_delta_l2": nominal_parameter_delta_l2,
            "nominal_treatment_parameter_delta_reproduced": nominal_delta_matches,
            "reconstructed_treatment_parameter_delta_l2": parameter_delta_l2_host,
            "direct_gradient_reports": list(direct["gradient_reports"]),
            "scheduled_gradient_reports": scheduled["gradient_reports"],
            "scheduled_artifact_records": scheduled["artifact_records"],
            "timing_seconds": {
                "direct": direct["timing_seconds"],
                "scheduled": scheduled["timing_seconds"],
            },
        },
        "window_localization": {
            "artifact_update": TARGET_ARTIFACT_UPDATE,
            "runtime_year": int(context["schedule"]["entries"][TARGET_ARTIFACT_UPDATE - 1]["runtime_year"]),
            "historical_aggregate_control": float(historical_target["gradient_baseline"]),
            "historical_aggregate_treatment": float(historical_target["smallest_nonzero_forward"]),
            "aggregate_control": window_scores["aggregate_control"],
            "aggregate_treatment": window_scores["aggregate_treatment"],
            "aggregate_control_reproduced": control_reproduced,
            "aggregate_treatment_reproduced": treatment_reproduced,
            "selection": window_scores["selection"],
        },
        "daily_trace": {
            "selected_window_index": selected_index,
            "selected_window_ordinal": selected_index + 1,
            "first_material_day_index": first_day,
            "first_material_day_ordinal": first_ordinal,
            "first_material_reasons": material["first_material_reasons"],
            "normalized_state_threshold": MATERIAL_NORMALIZED_STATE_CHANGE,
            "science_flux_registry": list(control_fluxes),
            "science_stock_registry": list(control_stocks),
            "days": material["records"],
        },
        "integrity": {
            "engineering_gate_passed": True,
            "direction_reproduced": norm_matches,
            "treatment_displacement_reproduced": True,
            "artifact_550_control_reproduced": control_reproduced,
            "artifact_550_treatment_reproduced": treatment_reproduced,
            "control_hard_failure_count": _hard_failure_count(window_scores["aggregate_control_hard_counts"]),
            "treatment_hard_failure_count": _hard_failure_count(window_scores["aggregate_treatment_hard_counts"]),
            "recovery_completed_logical_update": int(recovered["completed_logical_update"]),
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
    parser = _persistent_parser()
    parser.add_argument("--parent-persistent-training-report", type=Path, required=True)
    parser.add_argument("--parent-rejected-proposal", type=Path, required=True)
    parser.add_argument("--parent-artifact-aware-recovery", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
