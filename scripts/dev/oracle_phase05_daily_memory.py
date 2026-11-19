"""Bounded PB04 daily/season evidence; never executes a coupled model day.

The historical micro-oracle input generators and strict gates are reused
unchanged. New paper-constant component inputs are declared independently and
run through the original season subroutine once, not a 319-day trajectory.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "scripts/dev"
for path in (ROOT, DEV):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax

from jax_orchidee.stomate import carbon_kernels as ck
from jax_orchidee.stomate import daily as dm
from jax_orchidee.stomate import season as sm
from scripts.dev import oracle_lane_stomate_main_daily_prep_owner as legacy_prep
from scripts.dev import oracle_lane_stomate_season_phenology as legacy
from scripts.dev import run_stomate_daily_maintenance_oracle as legacy_daily

jax.config.update("jax_enable_x64", True)
DEFAULT_OUTPUT = (
    ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/daily-memory"
)
FIXTURE = ROOT / "tests/fixtures/phase05_daily_memory_reference.json"
RTOL, ATOL = 1e-12, 1e-14


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    if dataclasses.is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if hasattr(value, "_asdict"):
        return {k: encode(v) for k, v in value._asdict().items()}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def compare(name, expected, actual):
    a, b = np.asarray(expected), np.asarray(actual)
    valid = a.shape == b.shape
    close = np.isclose(a, b, rtol=RTOL, atol=ATOL) if valid else np.array([False])
    return {
        "name": name,
        "shape": list(a.shape),
        "passed": bool(np.all(close)),
        "failed_coordinates": int(np.sum(~close)),
        "max_abs": float(np.max(np.abs(a - b))) if valid and a.size else 0.0,
        "rtol": RTOL,
        "atol": ATOL,
    }


@contextmanager
def observe(module, names):
    """Observe arguments and real returns, always delegating to the owner."""
    calls = {name: [] for name in names}
    originals = {name: getattr(module, name) for name in names}
    for name, original in originals.items():

        def wrapper(*args, __name=name, __original=original, **kwargs):
            bound = inspect.signature(__original).bind(*args, **kwargs)
            result = __original(*args, **kwargs)
            calls[__name].append((dict(bound.arguments), result))
            return result

        setattr(module, name, wrapper)
    try:
        yield calls
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def saved_reference_checks(fixture):
    season_calls = []
    cases = []
    for cold in (False, True):
        with observe(
            sm,
            ("season_memory_step", "season_annual_step", "season_biometeorology_step", "season_update_gdd_init_date"),
        ) as calls:
            cases.append(legacy._jax_season(cold_start=cold))
        season_calls.append(calls)
    values = {k: np.concatenate([case[k] for case in cases]) for k in cases[0]}
    reference = {k: np.asarray(v) for k, v in fixture["saved_references"]["season"].items()}
    season_checks = [compare(k, v, values[k]) for k, v in reference.items()]
    with observe(ck, ("maintenance_respiration",)) as daily_calls:
        daily_values = legacy_daily._jax_values()
    daily_reference = {k: np.asarray(v) for k, v in fixture["saved_references"]["daily_maintenance"].items()}
    daily_checks = [compare(k, v, daily_values[k]) for k, v in daily_reference.items()]
    prep_values = legacy_prep._jax_values()
    prep_reference = {k: np.asarray(v) for k, v in fixture["saved_references"]["instant_daily_prep"].items()}
    prep_checks = [compare(k, v, prep_values[k]) for k, v in prep_reference.items()]
    return (
        {"season": season_checks, "daily_maintenance": daily_checks, "instant_daily_prep": prep_checks},
        season_calls,
        daily_calls,
    )


def paper_inputs(calls):
    """Explicit source-legal extension of the archived component design.

    Values are prescribed microinputs, not reconstructed observations or a
    natural trajectory. Use source undef=-9999 and source bare-soil GPP=0.
    """
    memory_args = dict(calls["season_memory_step"][0][0])
    state = memory_args.pop("state")
    state = state._replace(t2m_14=np.array([280.0, 290.0]), tsurf_year=np.array([281.0, 291.0]))
    memory_args.update(
        t2m_daily=np.array([280.0, 290.0]),
        tsurf_daily=np.array([281.0, 291.0]),
        tsoil_daily=np.array([[278.0, 278.0], [285.0, 285.0]]),
        soilhum_daily=np.array([[0.2, 0.2], [0.4, 0.4]]),
        undef=-9999.0,
    )
    annual_args = dict(calls["season_annual_step"][0][0])
    annual_state = annual_args.pop("state")
    annual_args.update(t2m_daily=memory_args["t2m_daily"])
    annual_args["gpp_daily"] = np.array(annual_args["gpp_daily"], copy=True)
    annual_args["gpp_daily"][:, 0] = 0.0
    bio_args = dict(calls["season_biometeorology_step"][0][0])
    bio_state = bio_args.pop("state")._replace(ngd_minus5=np.full((2, 14), 7.0))
    bio_args.update(
        t2m_daily=memory_args["t2m_daily"],
        undef=-9999.0,
        pheno_gdd_crit=np.full((14, 4), -9999.0),
        ncdgdd_temp=np.full(14, -9999.0),
        hum_min_time=np.full(14, -9999.0),
    )
    old_gdd = np.array([[0.0, 2000.0], [0.0, 2000.0]])
    return state, memory_args, annual_state, annual_args, bio_state, bio_args, old_gdd


def daily_payload(memory_args, annual_args):
    veget = np.zeros((2, 14))
    veget[:, 13] = 1.0
    gpp = np.array(annual_args["gpp_daily"], copy=True) / 48.0
    # All nonbare PFTs have cover in this prescribed daily-input microcomponent.
    # Season's separately declared annual-cover tuple is retained as supplied.
    veget[:, 1:13] = 1.0 / 20.0
    veget[:, 13] = 0.4
    cover = veget / veget.sum(axis=1, keepdims=True)
    gpp = gpp * cover
    return {
        "precip_rain": np.full(2, 2.0 / 48.0),
        "precip_snow": np.zeros(2),
        "veget": cover,
        "veget_max": cover,
        "totfrac_nobio": np.zeros(2),
        "gpp": gpp,
        "humrel": memory_args["moiavail_daily"],
        "litterhumdiag": np.zeros(2),
        "t2m": memory_args["t2m_daily"],
        "temp_sol": memory_args["tsurf_daily"],
        "stempdiag": memory_args["tsoil_daily"],
        "shumdiag": memory_args["soilhum_daily"],
        "t2m_min": memory_args["t2m_daily"] - 1.0,
        "t2m_max": memory_args["t2m_daily"] + 1.0,
        "wspeed": np.ones(2),
        "snow": np.zeros(2),
        "tmc_topgrass": np.zeros(2),
    }


def empty_accumulators(payload, *, nonzero=False):
    mapping = {
        "humrel_daily": "humrel",
        "litterhum_daily": "litterhumdiag",
        "t2m_daily": "t2m",
        "tsurf_daily": "temp_sol",
        "tsoil_daily": "stempdiag",
        "soilhum_daily": "shumdiag",
        "precip_daily": "precip_rain",
        "gpp_daily": "gpp",
        "wspeed_daily": "wspeed",
        "snowfall_daily": "precip_snow",
        "snowmass_daily": "snow",
        "tmc_topgrass_daily": "tmc_topgrass",
    }
    fields = {name: np.full_like(payload[source], 2.0 if nonzero else 0.0) for name, source in mapping.items()}
    fields.update(t2m_min_daily=np.full(2, 1e33), t2m_max_daily=np.full(2, -1e33))
    return SimpleNamespace(**fields)


def flatten_season(memory, annual, bio, gdd):
    values = {}
    for state in (memory, annual, bio):
        for name, value in state._asdict().items():
            if name in ("tau_longterm", "tseason"):
                continue
            arr = np.asarray(value)
            values[name] = arr.transpose(1, 0, 2, 3).ravel() if name == "turnover_longterm" else arr.ravel(order="F")
    values["gdd_init_date"] = np.asarray(gdd).ravel(order="F")
    return values


def compile_paper_reference(output_dir, *, calendar_str="gregorian", cap=False, floor=False):
    """One original season call, two points; no full-model integration."""
    template = legacy.SEASON_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "undef=1d20": "undef=-9999d0",
        "calendar_str='noleap'": f"calendar_str='{calendar_str}'",
        "t2=(/0d0,290d0/);ts=(/0d0,291d0/);tsoil=285d0;tsoil(1,:)=0d0;sh=0.4d0;sh(1,:)=0d0": "t2=(/280d0,290d0/);ts=(/281d0,291d0/);tsoil=285d0;tsoil(1,:)=278d0;sh=0.4d0;sh(1,:)=0.2d0",
        "gpp=0.2d0;gpp(1,14)=0d0": "gpp=0.2d0;gpp(1,14)=0d0;gpp(:,1)=0d0",
        "ngd=0d0;thm=0d0;hmd=0.5d0;gid=0d0;": "ngd=7d0;thm=0d0;hmd=0.5d0;gid=0d0;gid(:,2)=2000d0;",
    }
    for before, after in replacements.items():
        assert template.count(before) == 1, before
        template = template.replace(before, after)
    if cap:
        template = template.replace("t2=(/280d0,290d0/)", "t2=(/250d0,320d0/)")
        template = template.replace("tlong=(/259d0,289d0/)", "tlong=(/240d0,340d0/)")
        template = template.replace("tau=100d0", "tau=1095d0")
    if floor:
        template = template.replace("mday=0.5d0;mday(1,14)=0d0", "mday=1d-9;mday(1,14)=0d0")
        template = template.replace("mm=0.45d0;mw=0.4d0", "mm=1d-9;mw=1d-9")
        template = template.replace("nppl=0.05d0;turnl=0d0;gppw=0.1d0", "nppl=1d-9;turnl=1d-9;gppw=1d-9")
    template = (
        "\n".join(line for line in template.splitlines() if not line.strip().startswith("call season(npts,1.0,.true."))
        + "\n"
    )
    solar = b"".join(legacy.SOLAR_SOURCE.read_bytes().splitlines(keepends=True)[289:597])
    unit = (
        template.encode()
        .replace(b"! <DOWNWARD_SOLAR_FLUX>", solar)
        .replace(b"! <SEASON_MODULE>", legacy.SEASON_SOURCE.read_bytes())
    )
    (output_dir / "paper-season-single-call.f90").write_bytes(unit)
    csv_path = output_dir / "paper-fortran-outputs.csv"
    with tempfile.TemporaryDirectory(prefix="phase05_season_") as td:
        build = Path(td)
        src = build / "oracle.f90"
        exe = build / "oracle.exe"
        src.write_bytes(unit)
        metadata = legacy._compile_legacy(src, exe, legacy.DEFAULT_COMPILER)
        result = subprocess.run(
            [str(exe), str(csv_path.resolve()), "0"],
            cwd=build,
            env=legacy.compiler_environment(legacy.DEFAULT_COMPILER),
            capture_output=True,
            text=True,
        )
        (output_dir / "paper-fortran-stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output_dir / "paper-fortran-stderr.txt").write_text(result.stderr, encoding="utf-8")
        result.check_returncode()
    (output_dir / "paper-fortran-build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return legacy._read_csv(csv_path)


def compile_year_tseason_extension(output_dir):
    """Print one previously unprinted result of the unchanged old lifecycle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    template = legacy.SEASON_TEMPLATE.read_bytes()
    before = b"'tseason_length',tsyl(i);"
    after = before + b"write(u,'(A,\",\",ES26.17)')'tseason',tsy(i);"
    assert template.count(before) == 1
    template = template.replace(before, after)
    solar = b"".join(legacy.SOLAR_SOURCE.read_bytes().splitlines(keepends=True)[289:597])
    unit = template.replace(b"! <DOWNWARD_SOLAR_FLUX>", solar).replace(
        b"! <SEASON_MODULE>", legacy.SEASON_SOURCE.read_bytes()
    )
    (output_dir / "season-output-extension.f90").write_bytes(unit)
    with tempfile.TemporaryDirectory(prefix="phase05_season_year_") as td:
        build = Path(td)
        src = build / "oracle.f90"
        exe = build / "oracle.exe"
        src.write_bytes(unit)
        metadata = legacy._compile_legacy(src, exe, legacy.DEFAULT_COMPILER)
        values = []
        for mode in (0, 1):
            output = output_dir / f"outputs-mode-{mode}.csv"
            result = subprocess.run(
                [str(exe), str(output.resolve()), str(mode)],
                cwd=build,
                env=legacy.compiler_environment(legacy.DEFAULT_COMPILER),
                capture_output=True,
                text=True,
            )
            result.check_returncode()
            values.extend(legacy._read_csv(output)["tseason"].tolist())
    (output_dir / "build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return values


def paper_chain(calls, fortran_reference, *, calendar_str="gregorian", cap=False, floor=False, used_run_def=None):
    from jax_orchidee import coupled
    from jax_orchidee.driver.orchestration import (
        _paper_day_daily_reset_fields,
        _paper_day_season_fields_from_bundle_source,
        _paper_day_temp_growth_from_season_fields,
    )

    state, memory_args, annual_state, annual_args, bio_state, bio_args, old_gdd = paper_inputs(calls)
    if cap:
        state = state._replace(
            t2m_longterm=np.array([240.0, 340.0]),
            t2m_month=np.array([240.0, 340.0]),
            tau_longterm=1095.0,
            t2m_14=np.array([250.0, 320.0]),
        )
        memory_args["t2m_daily"] = np.array([250.0, 320.0])
        annual_args["t2m_daily"] = memory_args["t2m_daily"]
        bio_args["t2m_daily"] = memory_args["t2m_daily"]
    if floor:
        state = state._replace(moiavail_month=np.full((2, 14), 1e-9), moiavail_week=np.full((2, 14), 1e-9))
        moisture = np.full((2, 14), 1e-9)
        moisture[0, 13] = 0.0
        memory_args["moiavail_daily"] = moisture
        annual_args["moiavail_daily"] = moisture
        annual_state = annual_state._replace(
            npp_longterm=np.full((2, 14), 1e-9),
            turnover_longterm=np.full((2, 14, 12, 2), 1e-9),
            gpp_week=np.full((2, 14), 1e-9),
        )
    payload = daily_payload(memory_args, annual_args)
    fold = dm.stomate_daily_accumulation_fold_from_entries(
        entry_payloads=(payload,) * 48,
        accumulator_state=empty_accumulators(payload),
        do_slow_flags=(False,) * 47 + (True,),
        dt_sechiba=1800.0,
        dt_stomate=86400.0,
    )
    assert fold.ok
    checks = []
    binding = {
        "moiavail_daily": "humrel_daily",
        "t2m_daily": "t2m_daily",
        "tsurf_daily": "tsurf_daily",
        "tsoil_daily": "tsoil_daily",
        "soilhum_daily": "soilhum_daily",
    }
    for arg, field in binding.items():
        checks.append(compare("fold_to_season." + field, memory_args[arg], fold.fields[field]))
        memory_args[arg] = fold.fields[field]
    for arg, field in (("gpp_daily", "gpp_daily"), ("precip_daily", "precip_daily")):
        checks.append(compare("fold_to_annual." + field, annual_args[arg], fold.fields[field]))
        annual_args[arg] = fold.fields[field]
    memory = sm.season_memory_step(state, **memory_args)
    annual_args["tau_longterm"] = memory.state.tau_longterm
    annual = sm.season_annual_step(annual_state, **annual_args)
    gdd = sm.season_update_gdd_init_date(
        old_gdd, latitude=np.full(2, 10.0), julian_diff=100.0, calendar_str=calendar_str
    )
    bio_args.update(
        t2m_month=memory.state.t2m_month,
        t2m_week=memory.state.t2m_week,
        t2m_longterm=memory.state.t2m_longterm,
        moiavail_month=memory.state.moiavail_month,
        gdd_init_date=gdd,
    )
    bio = sm.season_biometeorology_step(bio_state, **bio_args)
    values = flatten_season(memory.state, annual.state, bio.state, gdd)
    oracle_checks = [compare("paper_reference." + k, v, values[k]) for k, v in fortran_reference.items()]
    source = SimpleNamespace(
        pre_step=SimpleNamespace(
            memory=SimpleNamespace(step=memory),
            annual=SimpleNamespace(step=annual),
            biometeorology=SimpleNamespace(step=bio, kwargs={**bio.state._asdict(), "gdd_init_date": gdd}),
        ),
        kwargs={"t2m": fold.fields["t2m_daily"]},
    )
    season_state = SimpleNamespace(
        **state._asdict(), date=99, gdd_init_date=old_gdd, begin_leaves=memory_args["begin_leaves"]
    )
    write_kwargs = dict(
        season_state=season_state,
        dt_days=1.0,
        tsurf_daily=fold.fields["tsurf_daily"],
        firstcall=True,
        latitude=np.full(2, 10.0),
        julian_diff=100.0,
    )
    if "calendar_str" in inspect.signature(_paper_day_season_fields_from_bundle_source).parameters:
        write_kwargs["calendar_str"] = calendar_str
    carry = _paper_day_season_fields_from_bundle_source(source, **write_kwargs)
    for name in ("t2m_14", "tsurf_year", "t2m_month", "moiavail_week", "tsoil_month", "soilhum_month"):
        checks.append(compare("actual_writeback." + name, getattr(memory.state, name), carry[name]))
    writeback_gdd = compare("actual_writeback.gdd_init_date", gdd, carry["gdd_init_date"])
    checks.append(compare("actual_writeback.date", 100, carry["date"]))
    temp_growth = _paper_day_temp_growth_from_season_fields(carry)
    checks.append(compare("next_actual.temp_growth", np.asarray(memory.state.t2m_month) - 273.15, temp_growth))
    # The real reset helper consumes actual component return; no downstream
    # physics is replaced. The namespace is only the documented fold interface.
    fold_iface = SimpleNamespace(ok=fold.ok, daily_fields=fold.fields)
    reset = _paper_day_daily_reset_fields(daily_fold=fold_iface, do_slow=True)
    retained = _paper_day_daily_reset_fields(daily_fold=fold_iface, do_slow=False)
    checks.extend(compare("non_slow_retains." + k, v, retained[k]) for k, v in fold.fields.items())
    checks.extend(
        compare(
            "reset." + k,
            np.full_like(np.asarray(v), 1e33 if k == "t2m_min_daily" else -1e33 if k == "t2m_max_daily" else 0.0),
            reset[k],
        )
        for k, v in fold.fields.items()
    )
    next_entry = dm.stomate_first_step_daily_accumulation_from_entry(
        entry_payload=payload,
        accumulator_state=SimpleNamespace(**reset),
        dt_sechiba=1800.0,
        dt_stomate=86400.0,
        do_slow=False,
    )
    assert next_entry.ok
    checks.append(compare("next_actual.t2m_accumulation", payload["t2m"] * 1800.0, next_entry.fields["t2m_daily"]))
    # A distinct nonzero carried-accumulator branch is exercised independently.
    restarted = dm.stomate_daily_accumulation_fold_from_entries(
        entry_payloads=(payload,),
        accumulator_state=empty_accumulators(payload, nonzero=True),
        do_slow_flags=(False,),
        dt_sechiba=1800.0,
        dt_stomate=86400.0,
    )
    checks.append(compare("nonzero_accumulator_carry", 2.0 + payload["t2m"] * 1800.0, restarted.fields["t2m_daily"]))
    # Actual production adapter: do not silently substitute the newly computed
    # GDD date. This records the present behavior against the source reference.
    adapter_season = SimpleNamespace(**bio_state._asdict(), gdd_init_date=old_gdd)
    params = SimpleNamespace(
        pheno_gdd_crit=np.full((14, 4), -9999.0), ncdgdd_temp=np.full(14, -9999.0), hum_min_time=np.full(14, -9999.0)
    )
    adapter = coupled.stomate_season_biometeorology_input_kwargs(
        season=adapter_season,
        daily=SimpleNamespace(t2m_daily=fold.fields["t2m_daily"]),
        entry_state=SimpleNamespace(when_growthinit=memory_args["when_growthinit"]),
        parameters=params,
        t2m_month=memory.state.t2m_month,
        t2m_week=memory.state.t2m_week,
        t2m_longterm=memory.state.t2m_longterm,
        moiavail_month=memory.state.moiavail_month,
        dt_days=1.0,
        julian_diff=100.0,
        firstcall=True,
    )
    adapter_check = compare("current_adapter.ngd_minus5", bio.state.ngd_minus5, adapter.step.state.ngd_minus5)
    parameters = SimpleNamespace(
        natural=memory_args["natural"],
        leaf_tab=memory_args["leaf_tab"],
        pheno_type=memory_args["pheno_type"],
        pft_to_mtc=memory_args["pft_to_mtc"],
    )
    actual_memory = coupled.stomate_season_memory_input_kwargs(
        season=season_state,
        daily=SimpleNamespace(**fold.fields),
        entry_state=SimpleNamespace(when_growthinit=memory_args["when_growthinit"]),
        parameters=parameters,
        dt_days=1.0,
        tau_longterm=state.tau_longterm,
        julian_diff=100.0,
        firstcall=True,
    )
    for name in state._fields:
        if name not in ("tsurf_year", "t2m_14"):
            checks.append(
                compare(
                    "actual_memory_adapter." + name,
                    getattr(memory.state, name),
                    getattr(actual_memory.step.state, name),
                )
            )
    # Exercise the actual common production predecessor. Optional producer
    # operands are supplied once the integration owner exposes them; before
    # that repair, the same test records the missing API and stale-date result.
    season_fields = {
        **state._asdict(),
        **annual_state._asdict(),
        **bio_state._asdict(),
        "gdd_init_date": old_gdd,
        "begin_leaves": memory_args["begin_leaves"],
    }
    entry = SimpleNamespace(
        biomass=annual_args["biomass"],
        npp_daily=annual_args["npp_daily"],
        turnover_daily=annual_args["turnover_daily"],
        when_growthinit=memory_args["when_growthinit"],
        npp_longterm=annual_state.npp_longterm,
        turnover_longterm=annual_state.turnover_longterm,
        lm_lastyearmax=annual_state.lm_lastyearmax,
    )
    common_kwargs = {
        "config_path": ROOT / "configs/orchidee_man_250919.yaml",
        "season": SimpleNamespace(**season_fields),
        "daily": SimpleNamespace(**fold.fields),
        "entry_state": entry,
        "veget": annual_args["veget"],
        "veget_max": annual_args["veget_max"],
        "dt_days": 1.0,
        "tau_longterm": state.tau_longterm,
        "julian_diff": 100.0,
        "firstcall_season": True,
    }
    if used_run_def is not None:
        common_kwargs["used_run_def"] = used_run_def
    common_signature = inspect.signature(coupled.paper_case_stomate_pre_step_input_kwargs)
    for key, value in (("latitude", np.full(2, 10.0)), ("calendar_str", calendar_str)):
        if key in common_signature.parameters:
            common_kwargs[key] = value
    actual_pre_step = coupled.paper_case_stomate_pre_step_input_kwargs(**common_kwargs)
    predecessor_check = compare(
        "actual_pre_step.pft14_ngd_minus5",
        np.asarray(bio.state.ngd_minus5)[:, 13],
        np.asarray(actual_pre_step.biometeorology.step.state.ngd_minus5)[:, 13],
    )
    return {
        "checks": checks,
        "oracle_checks": oracle_checks,
        "adapter_source_order_check": adapter_check,
        "payload": payload,
        "fold_fields": fold.fields,
        "memory_input_state": state,
        "memory_args": memory_args,
        "memory": memory,
        "annual": annual,
        "biometeorology": bio,
        "old_gdd_init_date": old_gdd,
        "updated_gdd_init_date": gdd,
        "actual_adapter_return": adapter,
        "actual_memory_adapter": actual_memory,
        "carry": carry,
        "actual_pre_step_pft14_check": predecessor_check,
        "actual_writeback_gdd_check": writeback_gdd,
        "calendar_str": calendar_str,
        "cap": cap,
        "floor": floor,
        "actual_pre_step_ngd_minus5": actual_pre_step.biometeorology.step.state.ngd_minus5,
        "actual_pre_step_producer_operands": {
            k: common_kwargs[k] for k in ("latitude", "calendar_str") if k in common_kwargs
        },
        "reset": reset,
        "next_entry": next_entry,
        "temp_growth": temp_growth,
    }


def maintenance_chain(daily_calls, calls, fixture):
    args = dict(daily_calls["maintenance_respiration"][0][0])
    one_result = daily_calls["maintenance_respiration"][0][1]
    _, memory_args, _, annual_args, _, _, _ = paper_inputs(calls)
    payload = daily_payload(memory_args, annual_args)
    payload.update(t2m=args["t2m"], stempdiag=args["stempdiag"])
    kwargs = {
        name: args[name]
        for name in (
            "biomass",
            "t2m_longterm",
            "z_soil",
            "rprof",
            "sla_calc",
            "coeff_maint_zero",
            "maint_resp_slope",
            "ext_coeff",
            "is_tree",
        )
    }
    initial = np.zeros_like(np.asarray(one_result.resp_maint_part))
    fold = dm.stomate_daily_process_fold_from_entries(
        entry_payloads=(payload,) * 48,
        accumulator_state=empty_accumulators(payload),
        resp_maint_part_current=initial,
        do_slow_flags=(False,) * 47 + (True,),
        dt_sechiba=1800.0,
        dt_stomate=86400.0,
        **kwargs,
    )
    assert fold.ok
    reference = {k: np.asarray(v) for k, v in fixture["saved_references"]["daily_maintenance"].items()}
    original = reference["maint_resp"].reshape(14, 2, 12).transpose(1, 0, 2)
    expected = np.zeros_like(original)
    for _ in range(48):
        expected = expected + original
    checks = [compare("actual48_maintenance.original_fortran_parts", expected, fold.daily_fields["resp_maint_part"])]
    # The maintained narrow public kernel is separately exercised using the
    # exact same archived per-entry tuple, not a new full-day JIT boundary.
    jit_result = jax.jit(
        lambda biomass, t2m, tlong, soil: (
            ck.maintenance_respiration(
                biomass,
                t2m,
                tlong,
                soil,
                args["z_soil"],
                args["rprof"],
                args["sla_calc"],
                args["coeff_maint_zero"],
                args["maint_resp_slope"],
                args["ext_coeff"],
                args["is_tree"],
                dt_sechiba_days=args["dt_sechiba_days"],
                min_stomate=args["min_stomate"],
                maint_resp_min_vmax=args["maint_resp_min_vmax"],
                maint_resp_coeff=args["maint_resp_coeff"],
            ).resp_maint_part
        )
    )(args["biomass"], args["t2m"], args["t2m_longterm"], args["stempdiag"])
    checks.append(compare("narrow_jit.original_fortran_parts", original, jit_result))
    return {
        "checks": checks,
        "exact_entry_inputs": args,
        "actual_return": fold.daily_fields,
        "reference_kind": "48 source-order additions of independently saved original Fortran per-entry parts; not a new whole-model oracle",
    }


def actual_consumer_join(component):
    """Observe real next consumers in a declared standalone carbon component.

    This joins independently specified component tuples. It is operand/carry
    evidence only; the growth oracle does not certify this extra tuple.
    """
    from jax_orchidee.stomate import integration
    from scripts.dev.oracle_phase05_growth_conditions import chain_inputs, run_current_chain

    def first_two(value):
        if isinstance(value, dict):
            return {k: first_two(v) for k, v in value.items()}
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == 6:
            return value[:2].copy()
        return value

    inputs = first_two(chain_inputs())
    carry = {
        **component["memory"].state._asdict(),
        **component["annual"].state._asdict(),
        **component["biometeorology"].state._asdict(),
        **component["carry"],
    }
    mappings = {
        "constraints_inputs": ("t2m_month", "tseason"),
        "alloc_inputs": ("moiavail_week", "tsoil_month", "soilhum_month"),
        "phenology_inputs": ("gdd_midwinter",),
        "post_npp_inputs": (
            "npp_longterm",
            "lm_lastyearmax",
            "tmin_spring_time",
            "maxmoiavail_lastyear",
            "minmoiavail_lastyear",
            "moiavail_week",
            "t2m_longterm",
            "t2m_month",
            "t2m_week",
            "gdd_from_growthinit",
        ),
    }
    for group, names in mappings.items():
        for name in names:
            inputs[group][name] = np.asarray(carry[name])
    # Source nelements=2 reference includes carbon and nitrogen; current growth
    # component owns carbon only. The carbon view is explicit, not invented.
    inputs["post_npp_inputs"]["turnover_longterm"] = np.asarray(carry["turnover_longterm"])[..., :1]
    inputs["post_npp_inputs"]["herbivores"] = np.asarray(component["annual"].herbivores)
    with observe(integration, ("constraints_step", "phenology_step", "allocation_step", "turnover_step")) as observed:
        result = run_current_chain(inputs)
    checks = []
    consumer_groups = {
        "constraints_step": "constraints_inputs",
        "phenology_step": "phenology_inputs",
        "allocation_step": "alloc_inputs",
        "turnover_step": "post_npp_inputs",
    }
    for consumer, group in consumer_groups.items():
        assert len(observed[consumer]) == 1
        operands = observed[consumer][0][0]
        for name in mappings[group]:
            if name in operands:
                checks.append(compare(consumer + ".operand." + name, inputs[group][name], operands[name]))
    checks.append(
        compare("phenology_none.retains_gdd_midwinter", carry["gdd_midwinter"], result.phenology.gdd_midwinter)
    )
    checks.append(
        {
            "name": "actual_consumer_returns_finite_biomass",
            "passed": bool(np.isfinite(np.asarray(result.post_npp.turnover.biomass)).all()),
        }
    )
    return {
        "checks": checks,
        "declared_inputs": inputs,
        "observed_actual_calls": observed,
        "return_summary": {
            "begin_leaves": result.phenology.begin_leaves,
            "allocation_f_alloc": result.allocation.f_alloc,
            "turnover_biomass": result.post_npp.turnover.biomass,
            "final_lai": result.post_npp.lai_after_setlai,
        },
        "level": "actual_consumer_operands_and_returns_only",
        "limit": "Additional joined tuple has no independent whole-carbon Fortran reference; none phenology retains GDD, climate memories are actual allocation/turnover operands.",
    }


def run(output_dir=DEFAULT_OUTPUT, *, compile_reference=False, external_identities=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    identity_checks = (
        [{"path": p, "passed": sha(ROOT / p) == expected} for p, expected in fixture["immutable_assets"].items()]
        if external_identities
        else []
    )
    saved, calls, daily_calls = saved_reference_checks(fixture)
    year_reference = (
        compile_year_tseason_extension(output_dir / "year-tseason-extension")
        if compile_reference
        else fixture["year_tseason_reference"]
    )
    year_actual = np.concatenate([np.asarray(case["season_memory_step"][-1][1].state.tseason) for case in calls])
    year_check = compare("original_year_tseason_additional_output", year_reference, year_actual)
    components = {}
    with tempfile.TemporaryDirectory(prefix="phase05_parameters_") as temp:
        parameter_path = Path(temp) / "materialized-input.def"
        parameter_path.write_text(
            "\n".join(f"{k} = {v}" for k, v in fixture["parameter_source"]["values"].items()) + "\n", encoding="utf-8"
        )
        for label, calendar_str, cap, floor in (
            ("paper_reference", "gregorian", False, False),
            ("paper_reference_noleap", "noleap", False, False),
            ("paper_reference_cap", "noleap", True, False),
            ("paper_reference_floor", "noleap", False, True),
        ):
            if compile_reference:
                case_dir = output_dir if label == "paper_reference" else output_dir / label
                case_dir.mkdir(parents=True, exist_ok=True)
                reference = compile_paper_reference(case_dir, calendar_str=calendar_str, cap=cap, floor=floor)
            else:
                reference = {k: np.asarray(v) for k, v in fixture[label].items()}
            components[label] = paper_chain(
                calls[0], reference, calendar_str=calendar_str, cap=cap, floor=floor, used_run_def=parameter_path
            )
    result = components["paper_reference"]
    maintenance = maintenance_chain(daily_calls, calls[0], fixture)
    consumer = actual_consumer_join(components["paper_reference_noleap"])
    checks = (
        identity_checks
        + [year_check]
        + [c for group in saved.values() for c in group]
        + [c for comp in components.values() for group in ("checks", "oracle_checks") for c in comp[group]]
        + maintenance["checks"]
    )
    integration_checks = [
        c
        for component in components.values()
        for c in (component["actual_pre_step_pft14_check"], component["actual_writeback_gdd_check"])
    ]
    passed = all(c["passed"] for c in checks + consumer["checks"] + integration_checks)
    report = {
        "schema_version": 1,
        "status": "passed_with_declared_input_phase_counterexample" if passed else "failed",
        "source_identity_mode": "external_checked" if external_identities else "external_not_requested",
        "strict_gates_unchanged": True,
        "rtol": RTOL,
        "atol": ATOL,
        "identity_checks": identity_checks,
        "saved_reference_checks": saved,
        "paper_component": result,
        "actual_maintenance_component": maintenance,
        "paper_components": components,
        "year_tseason_extension": {
            "comparison": year_check,
            "actual": year_actual,
            "reference": year_reference,
            "domain": "Unchanged old two-mode/two-call lifecycle, noleap, old sentinel1e20; only a missing output observation was added.",
        },
        "actual_consumer_join": consumer,
        "limits": [
            "Prescribed component inputs; not a demonstrated natural trajectory.",
            "No coupled step/day/year and no 319 trajectory executed.",
            "Historical sentinel=1e20 is tested only as its original microdomain.",
            "The original common-predecessor failure is preserved in execution-before-source-order-repair.json.",
            "Direct biometeorology with the old GDD date is a different-input-phase counterexample, not an integrated-chain failure.",
        ],
    }
    (output_dir / "execution.json").write_text(json.dumps(encode(report), indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compile-reference", action="store_true")
    args = parser.parse_args()
    report = run(args.output_dir, compile_reference=args.compile_reference)
    print(
        json.dumps(
            {
                "status": report["status"],
                "adapter_source_order_check": report["paper_component"]["adapter_source_order_check"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"].startswith("passed") else 1)
