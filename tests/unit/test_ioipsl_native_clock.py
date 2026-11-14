"""Pure scalar contracts against the identified IOIPSL reference, not model runs."""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from jax_orchidee.driver.dim2 import Dim2RestartTime, resolve_dim2_time_control
from jax_orchidee.sechiba.ioipsl_calendar import IOIPSLCalendar, fortran_nint
from jax_orchidee.sechiba.lifecycle_completion import intersurf_time

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((Path(__file__).parent / "fixtures/ioipsl_native_clock_v1.json").read_text())
ROWS = [(case, row) for case in FIXTURE["clocks"] for row in case["observed"]]
RED_CONTROLS = []
COMPARISONS = []


def invoke(*, clock=None, **changes):
    values = dict(istp=0, date0=152385.0, dt_sechiba=1800.0,
                  calendar="gregorian", one_year=365.2425, one_day=86400.0,
                  date0_convention="ioipsl_calendar", clock=clock)
    values.update(changes)
    return intersurf_time(**values)


@pytest.mark.parametrize("case,row", ROWS, ids=[f"{c['case_id']}:{r['istp']}" for c, r in ROWS])
def test_native_clock_owner_and_primitive(case, row):
    # Native probe locks from its scalar date0, preserving that rounding path.
    clock = IOIPSLCalendar.from_scalar_startdate(case["calendar"], case["date0"])
    state = invoke(clock=clock, istp=row["istp"], date0=case["date0"],
                   dt_sechiba=case["dt"], calendar=case["calendar"],
                   one_year=case["one_year_days"], one_day=case["one_day_seconds"])
    assert state.year_length == case["year_length"]
    for key, expected in (("year_spread", case["year_spread"]),
                          ("in_julian", row["in_julian"]), ("julian0", row["jan1"]),
                          ("julian_diff", row["diff"]), ("second", row["owner_sec"])):
        assert getattr(state, key) == pytest.approx(expected, rel=0, abs=1e-10)
    assert (state.year, state.month, state.day, state.month_length) == (
        row["year"], row["month"], row["day"], row["month_length"])
    primitive = (clock.ju2ymds(state.in_julian) if case["calendar"] == "gregorian"
                 else clock.itau2ymds(row["istp"], case["dt"]))
    assert primitive[:3] == (row["year"], row["month"], row["day"])
    assert primitive[3] == pytest.approx(row["primitive_sec"], rel=0, abs=1e-10)
    COMPARISONS.append(dict(case_id=case["case_id"], istp=row["istp"], native=row,
                            python=dict(year=state.year, month=state.month, day=state.day,
                                        in_julian=state.in_julian, primitive_sec=primitive[3],
                                        owner_sec=state.second, jan1=state.julian0,
                                        diff=state.julian_diff, month_length=state.month_length,
                                        year_length=state.year_length, year_spread=state.year_spread)))


@pytest.mark.parametrize("value,expected", [(0.5, 1), (1.5, 2), (-0.5, -1), (-1.5, -2), (0.49, 0),
                                           (np.nextafter(0.5, 0.0), 0), (np.nextafter(-0.5, 0.0), 0)])
def test_fortran_nint_ties_away_from_zero(value, expected):
    assert fortran_nint(value) == expected


def test_date_aware_year_and_source_feb29_normalization():
    clock = IOIPSLCalendar("gregorian")
    assert clock.ymds2ju(2000, 1, 1, 0) == 152385.0
    assert clock.year_steps(1800, clock.ymds2ju(2000, 1, 1, 0)) == 17568
    assert clock.year_steps(1800, clock.ymds2ju(2001, 1, 1, 0)) == 17520
    assert clock.year_steps(1800, clock.ymds2ju(2000, 2, 29, 0)) == 17568
    # IOIPSL's internal ymds2ju arithmetic admits next-year Feb29 -> Mar1.
    assert clock.ymds2ju(2001, 2, 29, 0) == clock.ymds2ju(2001, 3, 1, 0)


def test_non_gregorian_split_lock_is_independent_of_owner_origin():
    clock = IOIPSLCalendar.from_ymds("noleap", 1961, 1, 1, 900.25)
    state = invoke(clock=clock, calendar="noleap", one_year=365,
                   date0=clock.ymds2ju(1962, 1, 1, 0), istp=1)
    assert (state.year, state.month, state.day, state.second) == (1961, 1, 1, 2700.25)
    assert state.julian0 == 715765.0
    assert state.julian_diff > 365
    assert clock.itau2ymds(-1, 1800) == (1960, 12, 31, 85500.25)


def test_first_call_constants_persist_across_year_in_same_run_binding():
    first = invoke()
    later = invoke(istp=17568, previous=first)
    assert later.year == 2001
    assert later.year_length == first.year_length == 17568
    assert later.year_spread == first.year_spread == 1


def test_gregorian_owner_rounding_does_not_carry_date_after_decode():
    # intersurf1919 decodes the date before its1933 NINT assignment to REALsec.
    date0 = IOIPSLCalendar("gregorian").ymds2ju(2000, 1, 1, 86399.75)
    before = invoke(date0=date0, dt_sechiba=0.25)
    after = invoke(date0=date0, dt_sechiba=0.25, istp=1)
    assert (before.year, before.month, before.day, before.second) == (2000, 1, 1, 86400)
    assert (after.year, after.month, after.day, after.second) == (2000, 1, 2, 0)


@pytest.mark.parametrize("change", [dict(date0=152386), dict(dt_sechiba=900),
                                      dict(clock=IOIPSLCalendar.from_ymds("gregorian", 2000, 1, 1, 0))])
def test_previous_binding_change_rejected(change):
    with pytest.raises(ValueError, match="binding"):
        invoke(previous=invoke(), **change)


@pytest.mark.parametrize("change", [dict(initialized=False), dict(clock_binding=None)])
def test_invalid_or_unbound_previous_rejected(change):
    with pytest.raises(ValueError, match="previous"):
        invoke(previous=replace(invoke(), **change))


@pytest.mark.parametrize("change,match", [
    (dict(date0_convention=None), "date0_convention"),
    (dict(date0_convention="astronomical_jd", date0=2451544.5), "date0_convention"),
    (dict(one_year=365 * 86400), "days"), (dict(one_day=1), "seconds"),
    (dict(non_gregorian_ymds=lambda *_: (2001, 1, 1, 0)), "legacy"),
    (dict(calendar="noleap", one_year=365), "locked"),
    (dict(clock=IOIPSLCalendar("noleap")), "calendar"),
    (dict(dt_sechiba=0), "positive"), (dict(date0=float("nan")), "finite"),
])
def test_legacy_units_unbound_origins_and_invalid_inputs_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        invoke(**change)


@pytest.mark.parametrize("calendar", ["360d", "365d", "all_leap", "unknown"])
def test_unmeasured_calendar_domains_rejected(calendar):
    with pytest.raises(ValueError, match="unsupported"):
        IOIPSLCalendar(calendar)


@pytest.mark.parametrize("case", [c for c in FIXTURE["metadata"] if "observed" in c],
                         ids=lambda c: c["case_id"])
@pytest.mark.parametrize("reset", [False, True])
def test_native_restart_metadata_to_pure_dim2_mapping(case, reset):
    row = case["observed"]
    restart = Dim2RestartTime(row["itau"], row["date0"], row["dt"], reset)
    control = resolve_dim2_time_control(dt_force=1800, dt_sechiba=900,
                                      forcing_length=100, date0=715765,
                                      restart=restart)
    assert control.itau_dep_rest == row["itau"]
    assert control.istp_old == row["itau"]
    expected_dep = 0 if reset or row["itau"] == 0 else row["itau"]
    if row["dt"] != 1800 and expected_dep > 1:
        expected_dep = fortran_nint(expected_dep * row["dt"] / 1800)
    assert control.itau_dep == expected_dep
    assert control.istp == expected_dep * 2 + 1
    assert control.date0_rest == (715765 if reset or row["itau"] == 0 else row["date0"])
    assert control.for_offset == (0 if reset or row["itau"] == 0 else
                                  fortran_nint((row["date0"] - 715765) * 48))


def test_archived_synthetic_oracle_refuses_before_compile_and_preserves_fixture():
    source = ROOT / "scripts/dev/oracle_stage4_intersurf_time_owner.py"
    tree = ast.parse(source.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("_jax_case", "run_oracle", "run_coverage"):
        node = functions[name]
        assert isinstance(node.body[0], ast.Raise)
        isolated = ast.Module(body=[node], type_ignores=[])
        env = {"Path": Path, "DEFAULT_COMPILER": Path("unused"), "DEFAULT_OUTPUT_DIR": Path("unused"),
               "GCOV": Path("unused"), "np": np}
        exec(compile(isolated, str(source), "exec"), env)
        args = (True, "gregorian", 2451544.5, 60, 366) if name == "_jax_case" else (Path("unused"),)
        with pytest.raises(RuntimeError, match="synthetic"):
            env[name](*args)
    text = source.read_text()
    assert "y=2001+n/365" in text
    assert "one_year=days*one_day" in text
    assert '("init_gregorian", True, "gregorian", 2451544.5' in text


def test_saved_baseline_clock_counterexamples_are_still_reproduced():
    source_path = os.environ.get("ORCJAX_CLOCK_BASELINE_SOURCE")
    if source_path is None:
        pytest.skip("Packaged baseline control requires the guarded TEMP source")
    path = Path(source_path)
    tree = ast.parse(path.read_text())
    names = {"IntersurfTimeState", "_julian_to_gregorian", "_gregorian_to_julian",
             "_gregorian_month_length", "intersurf_time", "_non_gregorian_month_length",
             "non_gregorian_ymds_to_year_offset"}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    env = {"dataclass": dataclass, "np": np, "NonGregorianYmds": Callable,
           "Callable": Callable}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    old = env["intersurf_time"](istp=0, date0=152385, dt_sechiba=1800,
                               calendar="gregorian", one_year=365.2425, one_day=86400)
    assert old.year_length == 0
    assert old.year != 2000
    RED_CONTROLS.extend(["baseline-day-valued-year-zero-length", "baseline-Lilian-origin-wrong-year"])
    case = next(c for c in FIXTURE["clocks"] if c["case_id"] == "nlfrac")
    row = next(r for r in case["observed"] if r["istp"] == 0)
    old_fractional = env["intersurf_time"](
        istp=0, date0=case["date0"], dt_sechiba=case["dt"], calendar="noleap",
        one_year=365, one_day=86400,
        non_gregorian_ymds=lambda *_: (row["year"], row["month"], row["day"], row["primitive_sec"]))
    assert old_fractional.second == int(row["owner_sec"])
    assert old_fractional.second != row["owner_sec"]
    RED_CONTROLS.append("baseline-native-nlfrac-REAL-second-truncated")
