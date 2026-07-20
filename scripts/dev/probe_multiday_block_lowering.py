from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.fast_state import (  # noqa: E402
    DriverFastStateBundle,
    fast_state_from_previous_packet,
)
from jax_orchidee.driver.orchestration import (  # noqa: E402
    _compiled_half_hour_forcing,
    _paper_1961_later_day_transition_inputs,
    _paper_daily_carbon_static_dispatch,
    paper_1961_driver_day_scaffold,
    paper_1961_driver_later_day_runtime_result,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402


configure_jax_compilation_cache(ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lower the real next-day state-dependent input boundary without executing it."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "orchidee_man_250919.yaml",
    )
    parser.add_argument(
        "--run-def",
        type=Path,
        default=ROOT / "outputs" / "reference_mode" / "used_run.def",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "performance" / "multiday_block_lowering_probe.json",
    )
    parser.add_argument(
        "--state-cache",
        type=Path,
        default=ROOT / "outputs" / "performance" / "multiday_block_probe_day1_state.pkl",
    )
    parser.add_argument("--refresh-state", action="store_true")
    parser.add_argument("--execute-two-day", action="store_true")
    args = parser.parse_args()

    context = prepare_paper_1961_driver_context(
        args.config,
        used_run_def_path=args.run_def,
    )
    if args.state_cache.exists() and not args.refresh_state:
        with args.state_cache.open("rb") as handle:
            day1_state = pickle.load(handle)
    else:
        first = paper_1961_driver_day_scaffold(
            args.config,
            year=1961,
            start_tstep=0,
            used_run_def_path=context.run_def_path,
            root=ROOT,
            prepared_context=context,
            module_jit=True,
            diffuco_local_jit=True,
            retain_stomate_step_results=False,
            use_static_jit_daily_carbon=True,
        )
        if first.first_day_end_state is None:
            raise RuntimeError("Day 1 did not produce a state for the lowering probe")
        day1_state = first.first_day_end_state
        args.state_cache.parent.mkdir(parents=True, exist_ok=True)
        with args.state_cache.open("wb") as handle:
            pickle.dump(day1_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    initial = fast_state_from_previous_packet(day1_state)
    daily_carbon_dispatch = _paper_daily_carbon_static_dispatch(context, day1_state)
    prebound = _paper_1961_later_day_transition_inputs(
        context=context,
        current_state=day1_state,
        year=1961,
        start=48,
        steps_per_stomate=48,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    ).hydrol_runtime_static_tables
    forcing_days = []
    for start_tstep in (48, 96, 144, 192):
        prepared_forcing = _paper_1961_later_day_transition_inputs(
            context=context,
            current_state=day1_state,
            year=1961,
            start=start_tstep,
            steps_per_stomate=48,
            fixed_format_trace_dir=None,
            static_trace_fields=None,
            prebuild_day_payloads=True,
            prebound_hydrol_runtime_static_tables=prebound,
        )
        forcing_days.append(
            jax.tree_util.tree_map(
                lambda *values: np.stack(tuple(np.asarray(value) for value in values)),
                *tuple(
                    _compiled_half_hour_forcing(item)
                    for item in prepared_forcing.half_hour_inputs
                ),
            )
        )
    first_forcing_block = jax.tree_util.tree_map(
        lambda *values: np.stack(values),
        *forcing_days[:2],
    )
    second_forcing_block = jax.tree_util.tree_map(
        lambda *values: np.stack(values),
        *forcing_days[2:],
    )
    first_block_day_numbers = np.asarray([2, 3], dtype=np.int32)
    second_block_day_numbers = np.asarray([4, 5], dtype=np.int32)

    def trace_boundary(values_by_component):
        state = DriverFastStateBundle(
            tstep=initial.tstep,
            values_by_component=values_by_component,
            spec=initial.spec,
        )
        prepared = _paper_1961_later_day_transition_inputs(
            context=context,
            current_state=state,
            year=1961,
            start=48,
            steps_per_stomate=48,
            fixed_format_trace_dir=None,
            static_trace_fields=None,
            prebuild_day_payloads=True,
            prebound_hydrol_runtime_static_tables=prebound,
        )
        hydrol = prepared.hydrol_static_template
        diffuco = prepared.diffuco_day_static_cache
        return (
            hydrol["veget"],
            hydrol["veget_max"],
            hydrol["soiltile"],
            hydrol["vegtot"],
            diffuco.slowproc_derivvar_payload["temp_growth"],
        )

    report: dict[str, object]
    try:
        jax.jit(trace_boundary).lower(initial.values_by_component)
    except Exception as exc:  # The first trace blocker is the probe result.
        report = {
            "status": "blocked",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    else:
        def trace_day(values_by_component):
            state = DriverFastStateBundle(
                tstep=initial.tstep,
                values_by_component=values_by_component,
                spec=initial.spec,
            )
            day = paper_1961_driver_later_day_runtime_result(
                args.config,
                previous_state=state,
                day_index=2,
                year=1961,
                start_tstep=48,
                used_run_def_path=context.run_def_path,
                prepared_context=context,
                module_jit=True,
                diffuco_local_jit=True,
                use_static_jit_daily_carbon=False,
                prebuild_day_payloads=True,
                use_compiled_sechiba_day=True,
                prebound_hydrol_runtime_static_tables=prebound,
                materialize_compiled_entries=False,
                outer_compiled_daily_carbon_dispatch=daily_carbon_dispatch,
            )
            return (
                day.day_end_state.fields_by_component[
                    "slowproc_stomate_previous_step_state"
                ]["biomass"],
                day.daily_modelout.modelout_fields["GPP"],
            )

        try:
            jax.jit(trace_day).lower(initial.values_by_component)
        except Exception as exc:
            report = {
                "status": "input_boundary_lowered_full_day_blocked",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        else:
            def trace_two_days(values_by_component, block_forcing, day_numbers):
                def body(current_values, inputs):
                    forcing, science_day_number = inputs
                    state = DriverFastStateBundle(
                        tstep=47,
                        values_by_component=current_values,
                        spec=initial.spec,
                    )
                    day = paper_1961_driver_later_day_runtime_result(
                        args.config,
                        previous_state=state,
                        day_index=2,
                        year=1961,
                        start_tstep=48,
                        used_run_def_path=context.run_def_path,
                        prepared_context=context,
                        module_jit=True,
                        diffuco_local_jit=True,
                        use_static_jit_daily_carbon=False,
                        prebuild_day_payloads=True,
                        use_compiled_sechiba_day=True,
                        prebound_hydrol_runtime_static_tables=prebound,
                        materialize_compiled_entries=False,
                        outer_compiled_daily_carbon_dispatch=daily_carbon_dispatch,
                        compiled_forcing_series=forcing,
                        model_day_number=science_day_number,
                    )
                    packet = day.day_end_state
                    next_values = tuple(
                        tuple(
                            packet.fields_by_component[component][name]
                            for name in field_names
                        )
                        for component, field_names in zip(
                            initial.spec.components,
                            initial.spec.field_names_by_component,
                            strict=True,
                        )
                    )
                    return next_values, (
                        day.daily_modelout.modelout_fields,
                        day.daily_modelout.modelout,
                    )

                return jax.lax.scan(
                    body,
                    values_by_component,
                    (block_forcing, day_numbers),
                )

            try:
                jax.jit(trace_two_days).lower(
                    initial.values_by_component,
                    first_forcing_block,
                    first_block_day_numbers,
                )
            except Exception as exc:
                report = {
                    "status": "full_day_lowered_two_day_blocked",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            else:
                if not args.execute_two_day:
                    report = {
                        "status": "lowered",
                        "boundary": "two complete later-day transitions in one lax.scan executable",
                    }
                else:
                    jitted = jax.jit(trace_two_days)
                    started = time.perf_counter()
                    executable = jitted.lower(
                        initial.values_by_component,
                        first_forcing_block,
                        first_block_day_numbers,
                    ).compile()
                    first_compiled_result = executable(
                        initial.values_by_component,
                        first_forcing_block,
                        first_block_day_numbers,
                    )
                    jax.block_until_ready(first_compiled_result)
                    compile_and_run_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    first_compiled_result = executable(
                        initial.values_by_component,
                        first_forcing_block,
                        first_block_day_numbers,
                    )
                    jax.block_until_ready(first_compiled_result)
                    first_hot_seconds = time.perf_counter() - started
                    first_values, first_outputs = first_compiled_result
                    started = time.perf_counter()
                    second_compiled_result = executable(
                        first_values,
                        second_forcing_block,
                        second_block_day_numbers,
                    )
                    jax.block_until_ready(second_compiled_result)
                    second_hot_seconds = time.perf_counter() - started

                    reference_state = day1_state
                    reference_outputs = []
                    for day_index, start_tstep in (
                        (2, 48),
                        (3, 96),
                        (4, 144),
                        (5, 192),
                    ):
                        day = paper_1961_driver_later_day_runtime_result(
                            args.config,
                            previous_state=reference_state,
                            day_index=day_index,
                            year=1961,
                            start_tstep=start_tstep,
                            used_run_def_path=context.run_def_path,
                            prepared_context=context,
                            module_jit=True,
                            diffuco_local_jit=True,
                            use_static_jit_daily_carbon=True,
                            prebuild_day_payloads=True,
                            use_compiled_sechiba_day=True,
                            prebound_hydrol_runtime_static_tables=prebound,
                        )
                        reference_state = day.day_end_state
                        reference_outputs.append(
                            (
                                day.daily_modelout.modelout_fields,
                                day.daily_modelout.modelout,
                            )
                        )
                    compiled_values, second_outputs = second_compiled_result
                    max_abs = 0.0
                    max_rel = 0.0
                    comparisons = 0
                    ok = True
                    shape_mismatches = []

                    def compare_value(actual, expected, path):
                        nonlocal comparisons, max_abs, max_rel, ok
                        actual_leaves = jax.tree_util.tree_leaves(actual)
                        expected_leaves = jax.tree_util.tree_leaves(expected)
                        if len(actual_leaves) != len(expected_leaves):
                            shape_mismatches.append(f"{path}:leaf_count")
                            ok = False
                            return
                        for actual_leaf, expected_leaf in zip(
                            actual_leaves, expected_leaves, strict=True
                        ):
                            actual_array = np.asarray(actual_leaf)
                            expected_array = np.asarray(expected_leaf)
                            comparisons += 1
                            if actual_array.shape != expected_array.shape:
                                shape_mismatches.append(
                                    f"{path}:{actual_array.shape}!={expected_array.shape}"
                                )
                                ok = False
                                continue
                            if actual_array.dtype.kind in "biu":
                                ok = ok and np.array_equal(actual_array, expected_array)
                                continue
                            delta = np.abs(actual_array - expected_array)
                            scale = np.maximum(
                                np.abs(expected_array), np.finfo(np.float64).tiny
                            )
                            max_abs = max(max_abs, float(np.max(delta, initial=0.0)))
                            with np.errstate(over="ignore", invalid="ignore"):
                                relative = delta / scale
                            finite_relative = relative[np.isfinite(relative)]
                            if finite_relative.size:
                                max_rel = max(max_rel, float(np.max(finite_relative)))
                            ok = ok and np.allclose(
                                actual_array,
                                expected_array,
                                atol=1.0e-8,
                                rtol=1.0e-10,
                                equal_nan=True,
                            )

                    for component, names, actual_fields in zip(
                        initial.spec.components,
                        initial.spec.field_names_by_component,
                        compiled_values,
                        strict=True,
                    ):
                        for name, actual in zip(names, actual_fields, strict=True):
                            compare_value(
                                actual,
                                reference_state.fields_by_component[component][name],
                                f"{component}:{name}",
                            )
                    compiled_outputs = jax.tree_util.tree_map(
                        lambda first, second: jnp.concatenate((first, second), axis=0),
                        first_outputs,
                        second_outputs,
                    )
                    for day_offset, expected in enumerate(reference_outputs):
                        actual = jax.tree_util.tree_map(
                            lambda value: value[day_offset],
                            compiled_outputs,
                        )
                        compare_value(actual, expected, f"day_{day_offset + 2}:modelout")
                    executable_reused = True
                    report = {
                        "status": "passed" if ok and executable_reused else "mismatch",
                        "boundary": "two distinct two-day blocks through one lax.scan executable",
                        "compile_and_first_run_seconds": compile_and_run_seconds,
                        "first_hot_seconds": first_hot_seconds,
                        "first_hot_seconds_per_day": first_hot_seconds / 2.0,
                        "second_distinct_block_seconds": second_hot_seconds,
                        "second_distinct_block_seconds_per_day": second_hot_seconds / 2.0,
                        "compiled_executable_identity": id(executable),
                        "executable_reused": executable_reused,
                        "compared_leaves": comparisons,
                        "max_abs": max_abs,
                        "max_rel": max_rel,
                        "shape_mismatches": shape_mismatches,
                        "promotion": (
                            "candidate_lax_scan"
                            if ok and executable_reused
                            else "rejected"
                        ),
                    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "traceback"}, indent=2))
    return 0 if report["status"] in {"lowered", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
