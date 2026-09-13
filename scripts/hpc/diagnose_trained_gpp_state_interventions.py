"""Diagnose trained GPP recurrence with archived same-day state replacements."""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.trained_gpp_state_intervention import (
    bidirectional_gap_metrics,
    build_gpp_state_intervention_groups,
    build_intervention_arm_states,
    group_report,
    series_error_metrics,
)
from scripts.hpc import run_fresh_start_one_day as training
from scripts.hpc.trained_screen_common import WindowPool, json_safe, load_model

FLUX_NAMES = (
    "gpp",
    "npp",
    "maintenance_respiration",
    "growth_respiration",
    "heterotrophic_respiration",
)
HARD_NAMES = (
    "state_nonfinite",
    "budget_failure",
    "mask_mismatch",
    "discrete_mismatch",
    "gpp_nonfinite",
    "fraction_nonfinite",
    "learned_output_nonfinite",
    "stock_negative",
    "bare_soil_gsmean_nonzero",
)


def _archive_path(directory: Path, prior: Mapping[str, Any]) -> Path:
    name = Path(prior["raw_state_archive"]).name
    path = (directory / name).resolve()
    path.relative_to(directory.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_archived_carries(
    path: Path, *, horizon: int, state_width: int
) -> tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "free_state",
            "free_defined",
            "teacher_state",
            "teacher_defined",
            "teacher_start",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"state archive is missing arrays: {missing}")
        arrays = {name: np.asarray(payload[name]) for name in required}
    expected = (horizon, state_width)
    if any(arrays[name].shape != expected for name in required):
        shapes = {name: list(value.shape) for name, value in arrays.items()}
        raise ValueError(f"state archive shapes differ from {expected}: {shapes}")
    teacher_carry = np.asarray(arrays["teacher_start"], dtype=np.float64)
    free_carry = np.concatenate(
        (teacher_carry[:1], np.asarray(arrays["free_state"][:-1], dtype=np.float64)),
        axis=0,
    )
    return free_carry, teacher_carry, arrays


def _discrete_carry_check(jax: Any, templates: Any, expected: Any) -> int:
    mismatches = 0
    for actual, target in zip(
        jax.tree_util.tree_leaves(templates.discrete_state),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        actual_array = np.asarray(actual)
        target_array = np.asarray(target)
        mismatches += int(np.count_nonzero(actual_array[1:] != target_array[:-1]))
    return mismatches


def _series(science: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    value = science.fluxes[name]
    return np.asarray(value.value).reshape(-1), np.asarray(value.defined).reshape(-1)


def _control_comparison(actual: Any, expected: Any, replay_allowance: float = 0.0) -> Mapping[str, Any]:
    """Report graph replay differences without discarding scientific observations."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = np.abs(actual - expected)
    # Reuse the preceding attribution's replay tolerances. The free control
    # additionally inherits its recorded raw-GPP discrepancy bound.
    limit = replay_allowance + 1.0e-8 + 1.0e-6 * np.abs(expected)
    return {"valid": bool(np.all(np.isfinite(error) & (error <= limit))),
            "maximum_absolute_error": float(np.max(error)),
            "replay_allowance": float(replay_allowance), "atol": 1.0e-8, "rtol": 1.0e-6,
            "tolerance_source": "diagnose_trained_forward.py state replay allclose"}


def _summarize_aggregates(
    aggregate: Mapping[int, Mapping[str, Mapping[str, list[np.ndarray]]]],
    groups: Any,
) -> Mapping[str, Any]:
    result = {}
    for horizon, arms in aggregate.items():
        arm_metrics = {}
        for arm, fields in arms.items():
            arm_metrics[arm] = {}
            for flux in FLUX_NAMES:
                arm_metrics[arm][flux] = series_error_metrics(
                    np.concatenate(fields[f"{flux}__predicted"]),
                    np.concatenate(fields[f"{flux}__teacher"]),
                    predicted_defined=np.concatenate(fields[f"{flux}__predicted_defined"]),
                    teacher_defined=np.concatenate(fields[f"{flux}__teacher_defined"]),
                )
        causal = {}
        for group in groups:
            causal[group.name] = bidirectional_gap_metrics(
                arm_metrics["control_free"]["gpp"],
                arm_metrics["control_teacher"]["gpp"],
                arm_metrics[f"rescue__{group.name}"]["gpp"],
                arm_metrics[f"damage__{group.name}"]["gpp"],
            )
        result[str(horizon)] = {
            "day_count": int(sum(value.size for value in arms["control_free"]["gpp__teacher"])),
            "arms": arm_metrics,
            "gpp_bidirectional_attribution": causal,
        }
    return result


def _classify(
    summary: Mapping[str, Any],
    windows: list[Mapping[str, Any]],
    groups: Any,
    hard_totals: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Any]:
    consistency = {}
    for horizon in (7, 30):
        selected = [window for window in windows if int(window["selection"]["length"]) == horizon]
        consistency[str(horizon)] = {}
        for group in groups:
            rows = [window["gpp_bidirectional_attribution"][group.name] for window in selected]
            consistency[str(horizon)][group.name] = {
                "eligible_window_count": sum(bool(row["eligible"]) for row in rows),
                "positive_rescue_window_count": sum(
                    row["rescue_fraction"] is not None and row["rescue_fraction"] > 0.0 for row in rows
                ),
                "positive_damage_window_count": sum(
                    row["damage_fraction"] is not None and row["damage_fraction"] > 0.0 for row in rows
                ),
            }
    primary = []
    effect_candidates = []
    physical = {}
    controls_valid = all(window["control_reproduction"]["valid"] for window in windows)
    for group in groups:
        aggregate = summary["30"]["gpp_bidirectional_attribution"][group.name]
        counts = consistency["30"][group.name]
        relevant_arms = (
            "control_free",
            "control_teacher",
            f"rescue__{group.name}",
            f"damage__{group.name}",
        )
        physically_valid = all(
            not any(int(value) for value in hard_totals[arm].values())
            for arm in relevant_arms
        )
        physical[group.name] = physically_valid
        effect_material = (
            aggregate["bidirectionally_material"]
            and counts["positive_rescue_window_count"] >= 6
            and counts["positive_damage_window_count"] >= 6
        )
        if effect_material:
            effect_candidates.append(group.name)
        if effect_material and physically_valid and controls_valid:
            primary.append(group.name)
    return {
        "primary_t30_bidirectional_owner_candidates": primary,
        "t30_effect_candidates_before_validity_checks": effect_candidates,
        "control_reproduction_valid": controls_valid,
        "interpretation_limit": "hybrid mask/discrete changes are reported as limitations; all continuous effects remain available",
        "physically_valid_for_classification": physical,
        "window_sign_consistency": consistency,
        "classification": (
            "bidirectional_owner_candidates_identified"
            if primary
            else "no_single_group_meets_preregistered_bidirectional_rule"
        ),
    }


def run(args: Any) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
        rollout_input_from_template,
    )
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory
    from scripts.hpc.probe_public_rollout import physical_checks
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "stage": "bootstrap",
        "scope": "archived_same_day_gpp_state_intervention_not_model_acceptance",
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "optimizer_updates": 0,
        "parameter_steps": 0,
        "teacher_executions": 0,
        "windows": [],
    }

    def emit(stage: str, **values: Any) -> None:
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(args.output, json_safe(report))
        print(
            f"gpp_state_intervention stage={stage} windows={len(report['windows'])} "
            f"seconds={report['elapsed_seconds']:.1f}",
            flush=True,
        )

    try:
        emit("bootstrap")
        runtime, head, template, scales, parameters, method = load_model(args)
        jax = runtime["jax"]
        import jax.numpy as jnp

        attribution = json.loads(args.attribution_report.read_text())
        if attribution.get("status") != "completed" or len(attribution.get("windows", ())) != 16:
            raise ValueError("attribution report must contain the completed fixed 16-window panel")
        if attribution.get("checkpoint") != str(args.checkpoint):
            raise ValueError("attribution report and requested checkpoint differ")
        prior_windows = {
            (int(row["selection"]["artifact_id"]), int(row["selection"]["length"])): row
            for row in attribution["windows"]
        }
        if any(int(row["hard"]["discrete_mismatch"]) != 0 for row in attribution["windows"]):
            raise ValueError("archived free rollout did not preserve Teacher discrete state")

        pool = WindowPool(args.selection, method["source_identity"])
        groups = build_gpp_state_intervention_groups(head.contract)
        report["groups"] = group_report(groups, head.contract)
        report["state_coordinate_count"] = int(head.contract.continuous_state_width)
        report["arm_order"], _ = build_intervention_arm_states(
            np.zeros(head.contract.continuous_state_width),
            np.ones(head.contract.continuous_state_width),
            groups,
        )
        report["design"] = {
            "free_carry_source": "archived canonical free next-state from the preceding day",
            "teacher_carry_source": "prepared Teacher day-start continuous state",
            "discrete_carry": "prepared Teacher state after archived exact discrete-equality check",
            "hybrid_outputs_propagated": False,
            "same_day_forcing_and_parameters": True,
            "material_fraction": 0.10,
            "primary_window_sign_minimum_of_eight": 6,
        }

        transition = bind_daily_operator_dynamic_retained_inputs(
            input_assembler=head.input_assembler,
            process_head=head,
            retained_tail_template=template,
        )
        learned_output_names = tuple(field.name for field in head.learned_schema.fields)

        def diagnostic_batch(params: Any, states: Any, discrete: Any, day: Any) -> Any:
            def one(state: Any) -> Mapping[str, Any]:
                current = rollout_input_from_template(day[0], state, discrete, head.contract)
                result = transition(params, current, day[1])
                gpp = result.prediction.named_outputs["daily.fast_gpp"]
                return {
                    "state": result.next_state.other,
                    "defined": result.next_state_defined.other,
                    "gpp": gpp,
                    "gpp_defined": jnp.broadcast_to(
                        current.pft.active_pft_mask[None, :], gpp.shape
                    ),
                    "fraction": current.pft.pft_fraction,
                    "active": current.pft.active_pft_mask,
                    "discrete": result.next_discrete_state,
                    "budgets": jnp.stack(
                        (
                            result.diagnostics["water_budget_residual"],
                            result.diagnostics["carbon_budget_residual"],
                        )
                    ),
                    "learned_output_nonfinite": sum(
                        (
                            jnp.sum(
                                ~jnp.isfinite(result.prediction.named_outputs[name]),
                                dtype=jnp.int32,
                            )
                            for name in learned_output_names
                        ),
                        jnp.asarray(0, dtype=jnp.int32),
                    ),
                }

            return jax.lax.map(one, states)

        diagnostic = jax.jit(diagnostic_batch)

        def extract(observations: Mapping[str, Any]) -> Any:
            return extract_biomass_productivity_trajectories(
                observations["state"],
                observations["defined"],
                daily_fast_gpp=observations["gpp"],
                daily_fast_gpp_defined=observations["gpp_defined"],
                pft_fraction=observations["fraction"],
                active_pft_mask=observations["active"],
                contract=head.contract,
            )

        extract = jax.jit(extract)
        executable = None
        aggregate: Any = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        hard_totals: Any = defaultdict(lambda: {name: 0 for name in HARD_NAMES})

        for selection in pool.selection["windows"]:
            horizon = int(selection["length"])
            key = (int(selection["artifact_id"]), horizon)
            prior = prior_windows.get(key)
            if prior is None or prior["selection"] != selection:
                raise ValueError("fixed-panel attribution selection differs")
            initial, discrete, days, targets, expected_discrete = pool.case(jax, selection)
            del initial, discrete
            templates = days[0]
            discrete_carry_mismatch = _discrete_carry_check(jax, templates, expected_discrete)
            if discrete_carry_mismatch:
                raise ValueError("prepared previous expected discrete state differs from next day start")
            archive_path = _archive_path(args.archive_directory, prior)
            free_carry, teacher_carry, archive = _load_archived_carries(
                archive_path,
                horizon=horizon,
                state_width=head.contract.continuous_state_width,
            )
            if not np.array_equal(teacher_carry, np.asarray(templates.continuous_state)):
                raise ValueError("archived Teacher start differs from prepared window")
            if not np.array_equal(archive["teacher_state"], np.asarray(targets["state"])):
                raise ValueError("archived Teacher endpoint differs from prepared target")

            rows = []
            arm_names = None
            for day_index in range(horizon):
                day = jax.tree_util.tree_map(lambda value: value[day_index], days)
                names, states = build_intervention_arm_states(
                    free_carry[day_index], teacher_carry[day_index], groups
                )
                if arm_names is None:
                    arm_names = names
                elif arm_names != names:
                    raise ValueError("intervention arm order drift")
                day_discrete = jax.tree_util.tree_map(
                    lambda value: value[day_index], templates.discrete_state
                )
                if args.shape_only:
                    jax.eval_shape(diagnostic, parameters, states, day_discrete, day)
                    break
                if executable is None:
                    emit("compile_one_day_interventions")
                    executable = diagnostic.lower(
                        parameters, states, day_discrete, day
                    ).compile()
                    memory = _compiled_memory(executable)
                    report["compiled_memory"] = memory
                    if memory["estimated_live_bytes"] > 18 * 1024**3:
                        emit("stopped", status="resource_limited")
                        return report
                row = executable(parameters, states, day_discrete, day)
                jax.block_until_ready(row)
                rows.append(jax.device_get(row))
            if args.shape_only:
                report["windows"].append(
                    {
                        "selection": selection,
                        "archive": str(archive_path),
                        "discrete_carry_mismatch": discrete_carry_mismatch,
                    }
                )
                continue

            observations = jax.tree_util.tree_map(
                lambda *values: np.stack(values, axis=1), *rows
            )
            target_science = jax.device_get(extract(targets))
            teacher_fluxes = {
                flux: _series(target_science, flux)[0] for flux in FLUX_NAMES
            }
            teacher_defined = {
                flux: _series(target_science, flux)[1] for flux in FLUX_NAMES
            }
            arms = {}
            window_gpp_metrics = {}
            for arm_index, arm_name in enumerate(arm_names or ()):
                observed = jax.tree_util.tree_map(
                    lambda value: value[arm_index], observations
                )
                hard = physical_checks(
                    observed,
                    targets,
                    expected_discrete,
                    templates,
                    head.contract,
                    scales,
                )
                if set(hard) != set(HARD_NAMES):
                    raise ValueError("physical hard-count registry drift")
                for name, count in hard.items():
                    hard_totals[arm_name][name] += int(count)
                science = jax.device_get(extract(observed))
                flux_payload = {}
                for flux in FLUX_NAMES:
                    predicted, predicted_defined = _series(science, flux)
                    metrics = series_error_metrics(
                        predicted,
                        teacher_fluxes[flux],
                        predicted_defined=predicted_defined,
                        teacher_defined=teacher_defined[flux],
                    )
                    flux_payload[flux] = {
                        "predicted_daily": predicted,
                        "predicted_defined": predicted_defined,
                        "metrics": metrics,
                    }
                    bucket = aggregate[horizon][arm_name]
                    bucket[f"{flux}__predicted"].append(predicted)
                    bucket[f"{flux}__teacher"].append(teacher_fluxes[flux])
                    bucket[f"{flux}__predicted_defined"].append(predicted_defined)
                    bucket[f"{flux}__teacher_defined"].append(teacher_defined[flux])
                    if flux == "gpp":
                        window_gpp_metrics[arm_name] = metrics
                arms[arm_name] = {"hard_counts": hard, "fluxes": flux_payload}

            causal = {
                group.name: bidirectional_gap_metrics(
                    window_gpp_metrics["control_free"],
                    window_gpp_metrics["control_teacher"],
                    window_gpp_metrics[f"rescue__{group.name}"],
                    window_gpp_metrics[f"damage__{group.name}"],
                )
                for group in groups
            }
            prior_fluxes = prior["fluxes_gC_m2_day"]
            free_replay_error = float(
                np.max(
                    np.abs(
                        arms["control_free"]["fluxes"]["gpp"]["predicted_daily"]
                        - np.asarray(prior_fluxes["gpp"]["free"])
                    )
                )
            )
            teacher_reset_error = float(
                np.max(
                    np.abs(
                        arms["control_teacher"]["fluxes"]["gpp"]["predicted_daily"]
                        - np.asarray(prior_fluxes["gpp"]["reset"])
                    )
                )
            )
            target_error = float(
                np.max(
                    np.abs(
                        teacher_fluxes["gpp"]
                        - np.asarray(prior_fluxes["gpp"]["teacher"])
                    )
                )
            )
            fraction_l1 = float(np.max(np.sum(np.abs(observations["fraction"][0]), axis=-1)))
            comparisons = {
                "free": _control_comparison(
                    arms["control_free"]["fluxes"]["gpp"]["predicted_daily"],
                    prior_fluxes["gpp"]["free"],
                    float(prior["replay"]["gpp_max_abs_error"]) * fraction_l1),
                "teacher_reset": _control_comparison(
                    arms["control_teacher"]["fluxes"]["gpp"]["predicted_daily"],
                    prior_fluxes["gpp"]["reset"]),
                "teacher_target": _control_comparison(teacher_fluxes["gpp"], prior_fluxes["gpp"]["teacher"]),
            }
            report["windows"].append(
                {
                    "selection": selection,
                    "archive": str(archive_path),
                    "discrete_carry_mismatch": discrete_carry_mismatch,
                    "control_reproduction": {
                        "valid": all(row["valid"] for row in comparisons.values()),
                        "comparisons": comparisons,
                        "free_gpp_max_abs_error": free_replay_error,
                        "teacher_reset_gpp_max_abs_error": teacher_reset_error,
                        "teacher_target_gpp_max_abs_error": target_error,
                    },
                    "teacher_fluxes": teacher_fluxes,
                    "teacher_defined": teacher_defined,
                    "arms": arms,
                    "gpp_bidirectional_attribution": causal,
                }
            )
            emit("window_completed", memory=_device_memory(runtime["devices"][0]))

        if args.shape_only:
            emit(
                "completed",
                status="passed",
                classification="actual_archive_and_one_day_intervention_shapes_only",
            )
            return report
        summary = _summarize_aggregates(aggregate, groups)
        classification = _classify(summary, report["windows"], groups, hard_totals)
        emit(
            "completed",
            status="completed",
            summary=summary,
            hard_count_totals=hard_totals,
            causal_classification=classification,
            classification="same_day_state_mediation_diagnosis_not_unique_process_causality",
        )
    except Exception as error:
        emit("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return report


if __name__ == "__main__":
    parser = training.parser()
    for name in (
        "selection",
        "training-report",
        "checkpoint",
        "attribution-report",
        "archive-directory",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
