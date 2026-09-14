"""Reusable multi-day monitor for bounded mixed daily/T7/T30 learning.

This module deliberately owns no checkpoint loading, sampling, loss
cotangents, or optimizer state.  It evaluates a supplied parameter tree over
explicit prepared-window rows.  The existing development reader remains
strict: only a complete eight-T7/eight-T30 panel may be classified with the
13-key development science gates.  Smaller fixed TRAIN panels use exactly the
same rollout, physical checks, feedback diagnostics, and physical-unit
metrics, but are descriptive learning monitors rather than acceptance tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _validate_panel_rows(
    rows: Sequence[Mapping[str, Any]], *, development: bool, apply_science_gates: bool
) -> tuple[Mapping[str, Any], ...]:
    """Validate the distinct development and fixed-TRAIN monitoring roles."""

    result = tuple(rows)
    if not result:
        raise ValueError("monitor panel must contain at least one recurrent window")
    if apply_science_gates and not development:
        raise ValueError("only the strict development panel may apply science acceptance gates")
    if development:
        # Do not relax the accepted 16-window/day-disjoint selection contract
        # merely because the learning runner also needs a small TRAIN panel.
        from scripts.hpc.trained_screen_common import validate_selection

        validate_selection({"windows": list(result)})
    for row in result:
        horizon = int(row["length"])
        if horizon not in (7, 30):
            raise ValueError("mixed monitor supports only T7 or T30 windows")
        if row.get("component") != {7: "seven_day", 30: "thirty_day"}[horizon]:
            raise ValueError("monitor component/horizon mismatch")
    return result


def _rh_zero_target_caveat(physical: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make zero Teacher RH totals visible instead of treating them as evidence."""

    rh = physical["fluxes"]["heterotrophic_respiration"]
    nonzero = int(rh["teacher_nonzero_totals"])
    return {
        "teacher_nonzero_window_totals": nonzero,
        "zero_target_caveat": (
            "all defined Teacher RH totals are zero; RH errors remain reported, "
            "but this panel does not demonstrate a learned nonzero RH response"
            if nonzero == 0
            else None
        ),
    }


def _panel_classification(
    report: Mapping[str, Any], *, development: bool, apply_science_gates: bool
) -> dict[str, bool]:
    """Compute acceptance only for the declared development population."""

    horizons = report["horizons"]
    hard_pass = all(not any(window["hard"].values()) for window in report["windows"])
    population_complete = set(horizons) == {"7", "30"} and all(
        _complete_metric_population(
            value["physical"], value["science"], int(horizon), int(value["window_count"])
        )
        for horizon, value in horizons.items()
    )
    if not development:
        return {
            "physical_gate_passed": hard_pass,
            "metric_population_complete": bool(population_complete),
            "panel_science_gates_applied": False,
            "panel_science_gates_passed": False,
            "preliminary_screen_passed": False,
        }
    science_pass = bool(
        apply_science_gates
        and population_complete
        and all(value["science"]["all_key_gates_passed"] for value in horizons.values())
    )
    return {
        "physical_gate_passed": hard_pass,
        "metric_population_complete": bool(population_complete),
        "panel_science_gates_applied": bool(apply_science_gates),
        "panel_science_gates_passed": science_pass,
        "preliminary_screen_passed": bool(hard_pass and science_pass),
    }


def _complete_metric_population(
    physical: Mapping[str, Any], science: Mapping[str, Any], horizon: int, window_count: int
) -> bool:
    """Check reported metric population without applying an absolute science gate.

    ``complete_metric_gate`` is intentionally specific to the accepted
    eight-window development panel.  Fixed TRAIN diagnostics have one or a
    few windows per horizon, so they need the same missing/nonfinite coverage
    test with their declared population rather than a false 8-window failure.
    """

    if window_count < 1 or len(science.get("keys", {})) != 13:
        return False
    if len(physical.get("fluxes", {})) != 8 or len(physical.get("stocks", {})) != 5:
        return False
    if any(row.get("evaluated_values") != window_count for row in science["keys"].values()):
        return False
    for row in physical["fluxes"].values():
        for key, count in (
            ("cumulative_gC_m2", window_count),
            ("daily_gC_m2_day", window_count * horizon),
        ):
            value = row.get(key, {})
            if value.get("nonfinite_defined") or value.get("count") != count:
                return False
    for row in physical["stocks"].values():
        for value in row.values():
            if value.get("nonfinite_defined") or value.get("count") != window_count:
                return False
    return True


def build_mixed_rollout_monitor(
    runtime: Mapping[str, Any],
    head: Any,
    template: Any,
    scales: Any,
    model_protocol: Mapping[str, Any],
    *,
    compiled_memory_limit_bytes: int | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build a shared forward monitor for explicit prepared recurrent windows.

    The returned callable has signature::

        evaluate(parameters, pool, rows, *, panel_name, development,
                 apply_science_gates) -> report

    ``pool`` must expose the existing ``case(jax, row)`` boundary.  This is
    satisfied by both the strict ``WindowPool`` development reader and the
    training-specific prepared-window reader; neither branch needs to recreate
    Teacher targets, retained inputs, observations, or science extraction.
    """

    jax = runtime["jax"]
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.canonical_rollout import (
        make_canonical_step,
        make_science_probe_loss,
    )
    from research.daily_coarse_graining.daily_operator.forward_screen_metrics import (
        physical_window,
        summarize_windows,
    )
    from research.daily_coarse_graining.daily_operator.recurrent_science_loss import (
        CANOPY_KEY,
        shared_science_trajectories,
    )
    from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
    from research.daily_coarse_graining.daily_operator.successor_training import (
        extract_biomass_productivity_trajectories,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        _acceptance_trajectories,
        add_target_oriented_acceptance_statistics,
        derive_target_oriented_acceptance_scales,
        finalize_target_oriented_acceptance_statistics,
        target_oriented_acceptance_sufficient_statistics,
    )
    from scripts.hpc.probe_one_day_batch import _compiled_memory
    from scripts.hpc.probe_public_rollout import physical_checks
    from scripts.hpc.run_trained_forward_screen import feedback_diagnostics, feedback_hard_counts

    component = make_rollout_loss(
        jax, jnp, make_canonical_step(head, template), make_science_probe_loss(head.contract, scales)
    )
    forward = jax.jit(component.rollout)
    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        head.contract,
    )

    def extract(state, defined, gpp, gpp_defined, fraction, active):
        return extract_biomass_productivity_trajectories(
            state,
            defined,
            daily_fast_gpp=gpp,
            daily_fast_gpp_defined=gpp_defined,
            pft_fraction=fraction,
            active_pft_mask=active,
            contract=head.contract,
        )

    extract = jax.jit(extract)
    extract_canopy = jax.jit(
        lambda observation: shared_science_trajectories(observation, head.contract).stocks[CANOPY_KEY]
    )
    executables: dict[int, Any] = {}
    executable_metadata: dict[int, Mapping[str, Any]] = {}

    def compiled_for(arguments: tuple[Any, ...], horizon: int) -> Any:
        if horizon not in executables:
            began = time.perf_counter()
            executable = forward.lower(*arguments).compile()
            memory = _compiled_memory(executable)
            if (
                compiled_memory_limit_bytes is not None
                and memory["estimated_live_bytes"] > compiled_memory_limit_bytes
            ):
                raise MemoryError("mixed monitor forward exceeds declared compiled-memory envelope")
            executables[horizon] = executable
            executable_metadata[horizon] = {
                "compile_seconds": time.perf_counter() - began,
                "compiled_memory": memory,
            }
        return executables[horizon]

    def evaluate(
        parameters: Any,
        pool: Any,
        rows: Sequence[Mapping[str, Any]],
        *,
        panel_name: str,
        development: bool,
        apply_science_gates: bool,
    ) -> dict[str, Any]:
        selected = _validate_panel_rows(
            rows, development=development, apply_science_gates=apply_science_gates
        )
        began = time.perf_counter()
        report: dict[str, Any] = {
            "panel": str(panel_name),
            "role": "development" if development else "fixed_train_monitor",
            "windows": [],
            "horizons": {},
            "acceptance_scales": acceptance_scales._asdict(),
        }
        statistics: dict[int, Any] = {}
        physical: dict[int, list[Mapping[str, Any]]] = {}
        canopy_physical: dict[int, list[Mapping[str, Any]]] = {}

        for row_index, row in enumerate(selected, start=1):
            horizon = int(row["length"])
            print(
                f"mixed_monitor panel={panel_name} window={row_index}/{len(selected)} "
                f"horizon=T{horizon} stage=forward",
                flush=True,
            )
            initial, discrete, days, targets, expected_discrete = pool.case(jax, row)
            executable = compiled_for((parameters, initial, discrete, days), horizon)
            forward_began = time.perf_counter()
            _, observations = executable(parameters, initial, discrete, days)
            jax.block_until_ready(observations)
            elapsed = time.perf_counter() - forward_began

            hard = physical_checks(
                observations, targets, expected_discrete, days[0], head.contract, scales
            )
            predicted = extract(
                observations["state"],
                observations["defined"],
                observations["gpp"],
                observations["gpp_defined"],
                observations["fraction"],
                observations["active"],
            )
            expected = extract(
                targets["state"],
                targets["defined"],
                targets["gpp"],
                targets["gpp_defined"],
                targets["fraction"],
                targets["active"],
            )
            first = jax.tree_util.tree_map(lambda value: value[0], days[0])
            assembled = head.input_assembler.assemble(first)
            feedback = feedback_diagnostics(
                jax,
                jnp,
                runtime,
                head,
                model_protocol,
                initial,
                assembled,
                observations,
                targets,
                days[0],
            )
            hard.update(feedback_hard_counts(feedback))
            initial_science = extract(
                initial[None, :],
                assembled.state_defined.other[None, :],
                targets["gpp"][:1],
                targets["gpp_defined"][:1],
                targets["fraction"][:1],
                targets["active"][:1],
            )
            predicted_canopy = extract_canopy(observations)
            expected_canopy = extract_canopy(targets)
            initial_canopy = extract_canopy(
                {
                    "state": initial[None, :],
                    "defined": assembled.state_defined.other[None, :],
                    "gpp": targets["gpp"][:1],
                    "gpp_defined": targets["gpp_defined"][:1],
                    "fraction": targets["fraction"][:1],
                    "active": targets["active"][:1],
                }
            )
            predicted_fluxes, predicted_stocks = _acceptance_trajectories(
                predicted.fluxes, predicted.stocks
            )
            expected_fluxes, expected_stocks = _acceptance_trajectories(
                expected.fluxes, expected.stocks
            )
            _, start_stocks = _acceptance_trajectories(
                initial_science.fluxes, initial_science.stocks
            )
            metrics = physical_window(
                predicted_fluxes, expected_fluxes, predicted_stocks, expected_stocks, start_stocks
            )
            canopy_metric = physical_window(
                {},
                {},
                {CANOPY_KEY: predicted_canopy},
                {CANOPY_KEY: expected_canopy},
                {CANOPY_KEY: initial_canopy},
            )
            statistic = target_oriented_acceptance_sufficient_statistics(
                predicted.fluxes,
                expected.fluxes,
                predicted.stocks,
                expected.stocks,
                scales=acceptance_scales,
                horizon=horizon,
            )
            statistics[horizon] = (
                statistic
                if horizon not in statistics
                else add_target_oriented_acceptance_statistics(statistics[horizon], statistic)
            )
            physical.setdefault(horizon, []).append(metrics)
            canopy_physical.setdefault(horizon, []).append(canopy_metric)
            accepted_npp = predicted_fluxes["npp"]
            source_npp = predicted.fluxes["npp"]
            npp_valid = np.asarray(accepted_npp.defined) & np.asarray(source_npp.defined)
            npp_values = np.asarray(accepted_npp.value)
            source_npp_values = np.asarray(source_npp.value)
            npp_finite = np.isfinite(npp_values) & np.isfinite(source_npp_values)
            npp_error = abs(
                np.where(npp_valid & npp_finite, npp_values, 0.0)
                - np.where(npp_valid & npp_finite, source_npp_values, 0.0)
            )
            report["windows"].append(
                {
                    "selection": dict(row),
                    "prepared_day_indices": np.asarray(days[0].day_index),
                    "forward_seconds": elapsed,
                    "hard": hard,
                    "physical": metrics,
                    "canopy_physical": canopy_metric["stocks"][CANOPY_KEY],
                    "feedback_diagnostics": feedback,
                    "source_npp_identity_error": float(np.max(npp_error, initial=0.0)),
                    "source_npp_nonfinite_defined": int(np.sum(npp_valid & ~npp_finite)),
                }
            )
            print(
                f"mixed_monitor panel={panel_name} window={row_index}/{len(selected)} "
                f"horizon=T{horizon} stage=completed seconds={elapsed:.1f}",
                flush=True,
            )

        for horizon in sorted(statistics):
            summary = summarize_windows(physical[horizon])
            canopy_summary = summarize_windows(canopy_physical[horizon])["stocks"][CANOPY_KEY]
            science = finalize_target_oriented_acceptance_statistics(
                statistics[horizon], horizon=horizon, target_contract=head.target_contract
            )
            report["horizons"][str(horizon)] = {
                **executable_metadata[horizon],
                "window_count": len(physical[horizon]),
                "science": science,
                "physical": summary,
                "canopy_physical": canopy_summary,
                "rh": _rh_zero_target_caveat(summary),
            }
        report.update(
            _panel_classification(
                report, development=development, apply_science_gates=apply_science_gates
            ),
            elapsed_seconds=time.perf_counter() - began,
        )
        return report

    return evaluate
