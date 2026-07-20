from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.fast_state import (  # noqa: E402
    DriverFastStateBundle,
    fast_state_from_previous_packet,
)
from jax_orchidee.driver.orchestration import (  # noqa: E402
    _paper_1961_later_day_half_hour_transition,
    _paper_1961_later_day_transition_inputs,
    _paper_compiled_forcing_day,
    _paper_compiled_later_day_block_executable,
    _paper_daily_carbon_static_dispatch,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402


configure_jax_compilation_cache(ROOT)


def _timed(executable, *args, repeats: int = 5) -> tuple[float, list[float]]:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = executable(*args)
        jax.block_until_ready(result)
        values.append(time.perf_counter() - started)
    return min(values), values


def main() -> int:
    config = ROOT / "configs" / "orchidee_man_250919.yaml"
    run_def = ROOT / "outputs" / "reference_mode" / "used_run.def"
    state_cache = ROOT / "outputs" / "performance" / "multiday_block_probe_day1_state.pkl"
    output = ROOT / "outputs" / "performance" / "compiled_day_stage_timing.json"
    with state_cache.open("rb") as handle:
        day1_state = pickle.load(handle)
    initial = fast_state_from_previous_packet(day1_state)
    context = prepare_paper_1961_driver_context(config, used_run_def_path=run_def)
    forcing = _paper_compiled_forcing_day(
        context,
        year=1961,
        start_tstep=48,
        steps_per_stomate=48,
    )
    prepared = _paper_1961_later_day_transition_inputs(
        context=context,
        current_state=day1_state,
        year=1961,
        start=48,
        steps_per_stomate=48,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
        compiled_forcing_series=forcing,
    )

    def sechiba_only(values_by_component, forcing_series):
        state = DriverFastStateBundle(
            tstep=47,
            values_by_component=values_by_component,
            spec=initial.spec,
        )
        day_inputs = _paper_1961_later_day_transition_inputs(
            context=context,
            current_state=state,
            year=1961,
            start=48,
            steps_per_stomate=48,
            fixed_format_trace_dir=None,
            static_trace_fields=None,
            prebuild_day_payloads=True,
            prebound_hydrol_runtime_static_tables=(
                prepared.hydrol_runtime_static_tables
            ),
            compiled_forcing_series=forcing_series,
        )
        transition = _paper_1961_later_day_half_hour_transition(
            config,
            context=context,
            initial_state=state,
            day_inputs=day_inputs,
            fixed_format_trace_dir=None,
            static_trace_fields=None,
            module_jit=True,
            diffuco_local_jit=True,
            use_compiled_sechiba_day=True,
            materialize_compiled_entries=False,
        )
        packet = transition.current_state
        return tuple(
            tuple(fields.values())
            for fields in packet.fields_by_component.values()
        )

    started = time.perf_counter()
    sechiba_executable = jax.jit(sechiba_only).lower(
        initial.values_by_component,
        forcing,
    ).compile()
    sechiba_compile_seconds = time.perf_counter() - started
    sechiba_best, sechiba_repeats = _timed(
        sechiba_executable,
        initial.values_by_component,
        forcing,
    )

    forcing_block = jax.tree_util.tree_map(lambda value: value[None, ...], forcing)
    day_numbers = np.asarray([2], dtype=np.int32)
    started = time.perf_counter()
    full_executable, _ = _paper_compiled_later_day_block_executable(
        config,
        context=context,
        initial_state=day1_state,
        year=1961,
        block_forcing=forcing_block,
        block_day_numbers=day_numbers,
        prebound_hydrol_runtime_static_tables=prepared.hydrol_runtime_static_tables,
        daily_carbon_dispatch=_paper_daily_carbon_static_dispatch(context, day1_state),
    )
    full_compile_seconds = time.perf_counter() - started
    full_best, full_repeats = _timed(
        full_executable,
        initial.values_by_component,
        forcing_block,
        day_numbers,
    )
    report = {
        "status": "passed",
        "sechiba_only": {
            "compile_seconds": sechiba_compile_seconds,
            "best_hot_seconds": sechiba_best,
            "hot_repeats_seconds": sechiba_repeats,
        },
        "complete_day": {
            "compile_seconds": full_compile_seconds,
            "best_hot_seconds": full_best,
            "hot_repeats_seconds": full_repeats,
        },
        "post_sechiba_fraction": max(full_best - sechiba_best, 0.0) / full_best,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
