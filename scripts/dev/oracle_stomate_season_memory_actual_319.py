from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.coupled import (  # noqa: E402
    paper_case_season_time_scales,
    stomate_restart_season_memory_state,
)
from jax_orchidee.driver.domain import (  # noqa: E402
    _forcing_model_step_from_cached_point,
)
from jax_orchidee.stomate.reference import (  # noqa: E402
    stomate_cold_start_season_state,
)
from jax_orchidee.stomate.season import season_memory_step  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compiler_environment,
    exact_comparison,
    float_comparison,
)
from scripts.dev.oracle_driver_forcing_actual_319 import (  # noqa: E402
    DT_FORCE,
    NB_SPREAD,
    RUN_DEF,
    SPLIT,
    _actual_inputs,
)


FAMILY = "stomate_season_memory_actual_319"
LANDPOINT_ID = "319.0-057.0"
YEAR = 1961
NDAYS = 365
STEPS_PER_DAY = 48
DT_SECHIBA = 1800.0
DT_STOMATE = 86400.0
RTOL = 1.0e-12
ATOL = 1.0e-12

CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_season.f90"
SOLAR_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/solar.f90"
BASE_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_season_memory.f90.template"
TEMPLATE = ROOT / "scripts/dev/oracle_stomate_season_memory_actual_319.f90.template"
OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY

REAL_FIELDS = (
    "t2m_daily",
    "maintenance_pre_t2m_longterm",
    "tau_longterm",
    "t2m_longterm",
    "t2m_month",
    "t2m_week",
    "t2m_longterm_unclamped",
)
INTEGER_FIELDS = ("firstcall_season", "clamp_low", "clamp_high")


def _span(path: Path, start: int, end: int) -> bytes:
    return b"".join(path.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _daily_t2m() -> tuple[float, np.ndarray, dict[str, object]]:
    cache, domain, forcing_metadata = _actual_inputs()
    initial_t2m = float(cache.Tair[0, 0, 0])
    daily = np.empty(NDAYS, dtype=np.float64)
    for day in range(NDAYS):
        accumulator = np.float64(0.0)
        for offset in range(STEPS_PER_DAY):
            model_step = day * STEPS_PER_DAY + offset
            forcing = _forcing_model_step_from_cached_point(
                cache,
                model_tstep=model_step,
                split=SPLIT,
                nb_spread=NB_SPREAD,
                lon=domain.lon,
                lat=domain.lat,
                land_indices_key=((0, 0),),
                dt_force=DT_FORCE,
            )
            increment = np.float64(forcing.Tair[0, 0]) * np.float64(DT_SECHIBA)
            accumulator = np.float64(accumulator + increment)
        daily[day] = np.float64(accumulator / np.float64(DT_STOMATE))
    metadata = {
        **forcing_metadata,
        "cold_start_t2m_source": "raw forcing record 1 used by readstart initialization",
        "cold_start_t2m": initial_t2m,
        "daily_accumulator": {
            "steps_per_day": STEPS_PER_DAY,
            "dt_sechiba_seconds": DT_SECHIBA,
            "dt_stomate_seconds": DT_STOMATE,
            "operation_order": "48 sequential field_out += t2m * dt_sechiba, then divide by dt_stomate",
            "fortran_provenance": "src_stomate/stomate.f90::stomate_accu_r1d lines 9341-9367",
        },
        "daily_t2m_sha256": hashlib.sha256(daily.tobytes()).hexdigest(),
    }
    return initial_t2m, daily, metadata


def _write_input(path: Path, initial_t2m: float, daily_t2m: np.ndarray) -> str:
    values = np.concatenate(
        (np.asarray([initial_t2m], dtype=np.float64), np.asarray(daily_t2m, dtype=np.float64))
    )
    payload = np.ascontiguousarray(values).tobytes()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _compile_unit() -> tuple[bytes, dict[str, object]]:
    base = BASE_TEMPLATE.read_bytes()
    marker = b"program main"
    if base.count(marker) != 1:
        raise ValueError("base season Oracle template program marker drift")
    stubs = base.split(marker, 1)[0]
    season_raw = SOURCE.read_bytes()
    solar_span = _span(SOLAR_SOURCE, 290, 597)
    stubs = stubs.replace(b"! <DOWNWARD_SOLAR_FLUX>", solar_span)
    stubs = stubs.replace(b"! <SEASON_MODULE>", season_raw)
    if b"! <" in stubs:
        raise ValueError("unresolved placeholder in generated Fortran stubs")
    unit = stubs + TEMPLATE.read_bytes()
    return unit, {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_procedure": "stomate_season.f90::season lines 167-1584",
        "source_sha256": hashlib.sha256(season_raw).hexdigest(),
        "solar_source": str(SOLAR_SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "solar_span": "solar.f90 lines 290-597",
        "solar_span_sha256": hashlib.sha256(solar_span).hexdigest(),
        "base_template_sha256": hashlib.sha256(base).hexdigest(),
        "harness_template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        "compile_unit_sha256": hashlib.sha256(unit).hexdigest(),
    }


def _compile_and_run(
    work: Path, input_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source = work / "oracle.f90"
    executable = work / "oracle.exe"
    real_path = work / "fortran_real.bin"
    integer_path = work / "fortran_integer.bin"
    unit, source_metadata = _compile_unit()
    source.write_bytes(unit)
    flags = (
        "-std=legacy",
        "-fdec",
        "-fdefault-real-8",
        "-ffree-line-length-none",
        "-O0",
        "-fcheck=all",
        "-ffpe-trap=invalid,zero,overflow",
    )
    command = [str(DEFAULT_COMPILER), *flags, str(source), "-o", str(executable)]
    environment = compiler_environment(DEFAULT_COMPILER)
    built = subprocess.run(
        command, cwd=work, env=environment, text=True, capture_output=True
    )
    if built.returncode:
        raise RuntimeError(f"Fortran compilation failed:\n{built.stdout}\n{built.stderr}")
    executed = subprocess.run(
        [str(executable), str(input_path), str(real_path), str(integer_path)],
        cwd=work,
        env=environment,
        text=True,
        capture_output=True,
    )
    if executed.returncode:
        raise RuntimeError(f"Fortran execution failed:\n{executed.stdout}\n{executed.stderr}")
    real = np.fromfile(real_path, dtype=np.float64)
    integer = np.fromfile(integer_path, dtype=np.int32)
    if real.size != len(REAL_FIELDS) * NDAYS:
        raise ValueError(f"unexpected Fortran real output size {real.size}")
    if integer.size != len(INTEGER_FIELDS) * NDAYS:
        raise ValueError(f"unexpected Fortran integer output size {integer.size}")
    version = subprocess.run(
        [str(DEFAULT_COMPILER), "--version"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    metadata = {
        **source_metadata,
        "compiler": str(DEFAULT_COMPILER),
        "compiler_version": version,
        "compile_flags": list(flags),
        "compile_stdout": built.stdout,
        "compile_stderr": built.stderr,
        "run_stdout": executed.stdout,
        "run_stderr": executed.stderr,
    }
    return (
        real.reshape((len(REAL_FIELDS), NDAYS), order="F"),
        integer.reshape((len(INTEGER_FIELDS), NDAYS), order="F"),
        metadata,
    )


def _jax_outputs(initial_t2m: float, daily_t2m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nvm, nslm = 14, 2
    cold = stomate_cold_start_season_state(
        t2m=np.asarray([initial_t2m]), dt_days=1.0, nvm=nvm, nslm=nslm
    )
    state = stomate_restart_season_memory_state(cold, tau_longterm=cold.tau_longterm)
    state = state._replace(
        tsurf_year=np.asarray([initial_t2m]), t2m_14=np.asarray([initial_t2m])
    )
    scales = paper_case_season_time_scales(CONFIG, used_run_def=RUN_DEF)
    real = np.empty((len(REAL_FIELDS), NDAYS), dtype=np.float64)
    integer = np.empty((len(INTEGER_FIELDS), NDAYS), dtype=np.int32)
    natural = np.zeros(nvm, dtype=bool)
    natural[13] = True
    for day, t2m in enumerate(daily_t2m):
        prior_longterm = float(np.asarray(state.t2m_longterm)[0])
        tau_next = min(float(state.tau_longterm) + 1.0, scales.tau_longterm_max)
        unclamped = (prior_longterm * (tau_next - 1.0) + float(t2m)) / tau_next
        real[0, day] = t2m
        real[1, day] = prior_longterm
        integer[:, day] = (
            int(day == 0),
            int(unclamped < scales.tlong_ref_min),
            int(unclamped > scales.tlong_ref_max),
        )
        state = season_memory_step(
            state,
            dt_days=1.0,
            end_of_year=day == NDAYS - 1,
            moiavail_daily=np.zeros((1, nvm)),
            t2m_daily=np.asarray([t2m]),
            tsurf_daily=np.asarray([t2m]),
            tsoil_daily=np.full((1, nslm), initial_t2m),
            soilhum_daily=np.zeros((1, nslm)),
            begin_leaves=np.zeros((1, nvm), dtype=bool),
            julian_diff=float(day + 1),
            when_growthinit=np.zeros((1, nvm)),
            natural=natural,
            leaf_tab=np.zeros(nvm, dtype=np.int32),
            pheno_type=np.zeros(nvm, dtype=np.int32),
            pft_to_mtc=np.zeros(nvm, dtype=np.int32),
            time_scales=scales,
            firstcall=day == 0,
            undef=1.0e20,
        ).state
        real[2:, day] = (
            float(state.tau_longterm),
            float(np.asarray(state.t2m_longterm)[0]),
            float(np.asarray(state.t2m_month)[0]),
            float(np.asarray(state.t2m_week)[0]),
            unclamped,
        )
    return real, integer


def _first_mismatch(
    fortran_real: np.ndarray,
    jax_real: np.ndarray,
    fortran_integer: np.ndarray,
    jax_integer: np.ndarray,
) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    for index, field in enumerate(REAL_FIELDS):
        passed = np.isclose(fortran_real[index], jax_real[index], rtol=RTOL, atol=ATOL)
        if not np.all(passed):
            day = int(np.flatnonzero(~passed)[0])
            candidates.append(
                {
                    "day": day + 1,
                    "field": field,
                    "fortran": float(fortran_real[index, day]),
                    "jax": float(jax_real[index, day]),
                    "abs_error": float(abs(fortran_real[index, day] - jax_real[index, day])),
                }
            )
    for index, field in enumerate(INTEGER_FIELDS):
        passed = fortran_integer[index] == jax_integer[index]
        if not np.all(passed):
            day = int(np.flatnonzero(~passed)[0])
            candidates.append(
                {
                    "day": day + 1,
                    "field": field,
                    "fortran": int(fortran_integer[index, day]),
                    "jax": int(jax_integer[index, day]),
                }
            )
    return min(candidates, key=lambda item: (int(item["day"]), str(item["field"]))) if candidates else None


def _write_evidence(
    fortran_real: np.ndarray,
    jax_real: np.ndarray,
    fortran_integer: np.ndarray,
    jax_integer: np.ndarray,
) -> None:
    with (OUTPUT_DIR / "fortran_outputs.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("day", "field", "value"))
        for day in range(NDAYS):
            for index, field in enumerate(REAL_FIELDS):
                writer.writerow((day + 1, field, f"{fortran_real[index, day]:.17e}"))
            for index, field in enumerate(INTEGER_FIELDS):
                writer.writerow((day + 1, field, int(fortran_integer[index, day])))
    with (OUTPUT_DIR / "point_comparisons.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("day", "field", "fortran", "jax", "abs_error", "passed"))
        for day in range(NDAYS):
            for index, field in enumerate(REAL_FIELDS):
                expected = fortran_real[index, day]
                actual = jax_real[index, day]
                writer.writerow(
                    (
                        day + 1,
                        field,
                        f"{expected:.17e}",
                        f"{actual:.17e}",
                        f"{abs(expected - actual):.17e}",
                        str(bool(np.isclose(expected, actual, rtol=RTOL, atol=ATOL))).lower(),
                    )
                )
            for index, field in enumerate(INTEGER_FIELDS):
                expected = int(fortran_integer[index, day])
                actual = int(jax_integer[index, day])
                writer.writerow((day + 1, field, expected, actual, abs(expected - actual), str(expected == actual).lower()))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    initial_t2m, daily_t2m, input_metadata = _daily_t2m()
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_season_actual_319_") as raw:
        work = Path(raw)
        input_path = work / "inputs.bin"
        input_metadata["oracle_input_sha256"] = _write_input(
            input_path, initial_t2m, daily_t2m
        )
        fortran_real, fortran_integer, build = _compile_and_run(work, input_path)
    jax_real, jax_integer = _jax_outputs(initial_t2m, daily_t2m)
    comparisons = [
        float_comparison(field, fortran_real[index], jax_real[index], rtol=RTOL, atol=ATOL)
        for index, field in enumerate(REAL_FIELDS)
    ]
    comparisons.extend(
        exact_comparison(field, fortran_integer[index], jax_integer[index])
        for index, field in enumerate(INTEGER_FIELDS)
    )
    first_mismatch = _first_mismatch(
        fortran_real, jax_real, fortran_integer, jax_integer
    )
    status = "passed" if all(item["passed"] for item in comparisons) else "failed"
    _write_evidence(fortran_real, jax_real, fortran_integer, jax_integer)
    (OUTPUT_DIR / "inputs.json").write_text(
        json.dumps(input_metadata, indent=2) + "\n", encoding="ascii"
    )
    result = {
        "schema_version": 2,
        "family": FAMILY,
        "status": status,
        "landpoint_id": LANDPOINT_ID,
        "year": YEAR,
        "days": NDAYS,
        "ordering_contract": "maintenance reads t2m_longterm before same-day season update",
        "comparisons": comparisons,
        "first_mismatch": first_mismatch,
        "branch_counts": {
            field: int(np.sum(fortran_integer[index]))
            for index, field in enumerate(INTEGER_FIELDS)
        },
        "build": build,
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    summary = {
        "family": FAMILY,
        "status": status,
        "days": NDAYS,
        "failed_comparisons": sum(not item["passed"] for item in comparisons),
        "first_mismatch": first_mismatch,
        "branch_counts": result["branch_counts"],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
