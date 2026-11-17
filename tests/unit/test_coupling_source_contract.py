"""Nonzero PEQ consumer evidence; does not certify an atmospheric producer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_orchidee.sechiba.enerbil import (
    enerbil_flux_local_diagnostics,
    enerbil_surftemp_explicit_solve,
    intersurf_evaporation_rate,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "tests/fixtures/coupling_peq_nonzero_reference.json").read_text(encoding="utf-8-sig"))


def _solve(case):
    return enerbil_surftemp_explicit_solve(**{k: np.asarray(v) for k, v in case["inputs"].items()})


def _flux(case, solved):
    p = case["inputs"]
    return enerbil_flux_local_diagnostics(
        **{k: np.asarray(p[k]) for k in ["emis", "rau", "u", "v", "q_cdrag", "vbeta", "valpha", "vbeta1", "vbeta5"]},
        temp_sol=np.asarray(p["psold"]) / 1004.675,
        qair=solved.qair_new,
        epot_air=solved.epot_air_new,
        psnew=solved.psnew,
        qsol_sat_new=solved.qsol_sat_new,
        temp_sol_new=solved.temp_sol_new,
        lwdown=np.asarray([300.0, 310.0]),
        swnet=np.asarray([120.0, 130.0]),
        dt_sechiba=p["dt_sechiba"],
    )


def test_nonzero_tuple_original_source_identity():
    for source in REFERENCE["source"]:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_nonzero_tuple_surftemp_flux_and_intersurf_rate_match_original_source(case):
    solved = _solve(case)
    expected = case["outputs"]
    for name in ["psnew", "qsol_sat_new", "temp_sol_new", "qair_new", "epot_air_new"]:
        np.testing.assert_allclose(getattr(solved, name), expected[name], rtol=1e-12, atol=1e-12)
    for name in ["qsol_sat_new", "temp_sol_new"]:
        np.testing.assert_allclose(
            getattr(solved, name + "_pft")[:, 1], expected[name + "_pft2"], rtol=1e-12, atol=1e-12
        )
    flux = _flux(case, solved)
    np.testing.assert_allclose(flux.vevapp, expected["vevapp_amount"], rtol=1e-12, atol=1e-12)
    rate = intersurf_evaporation_rate(flux.vevapp, dt_sechiba=case["inputs"]["dt_sechiba"])
    np.testing.assert_allclose(rate, expected["vevapp_rate"], rtol=1e-12, atol=1e-12)
    # Distinguishable amount/rate and nonzero tuple effects prevent a zero/default pass.
    assert np.all(np.abs(np.asarray(flux.vevapp) - np.asarray(rate)) > 1e-5)


def test_nonzero_A_is_consumed_and_rate_boundary_is_jittable():
    by_name = {c["name"]: c for c in REFERENCE["cases"]}
    zero = _solve(by_name["zero-dt900"])
    positive = _solve(by_name["positive-dt900"])
    negative = _solve(by_name["negative-dt900"])
    assert np.all(np.abs(np.asarray(positive.qair_new) - np.asarray(zero.qair_new)) > 1e-7)
    assert np.all(np.abs(np.asarray(negative.qair_new) - np.asarray(zero.qair_new)) > 1e-7)
    case = by_name["positive-dt2700"]
    amount = np.asarray(case["outputs"]["vevapp_amount"])
    rate = jax.jit(lambda a, dt: intersurf_evaporation_rate(a, dt_sechiba=dt))(amount, 2700.0)
    np.testing.assert_allclose(rate, case["outputs"]["vevapp_rate"], rtol=1e-12, atol=1e-12)


def test_watchout_reader_preserves_nonzero_tuple_without_enabling_driver():
    from jax_orchidee.driver.dim2 import Dim2Switches, validate_dim2_switches
    from jax_orchidee.driver.forcing_readers import VerticalForcingOptions, forcing_just_read

    # Raw imported source tuple passes through unchanged. It carries no producer identity claim.
    def scalar(value):
        return np.asarray([[[value]]])

    values = dict(
        Tair=283.0,
        Snowf=0.0,
        Rainf=0.0002,
        Tmin=282.0,
        precip=0.0003,
        SWdown=112.0,
        LWdown=301.0,
        PSurf=98765.0,
        Qair=0.0061,
        Wind_N=2.1,
        Wind_E=3.2,
        Wind=4.3,
        Levels=4.0,
        Levels_uv=8.0,
        levels=5.0,
        SWnet=111.0,
        Eair=284123.0,
        petAcoef=1.2,
        peqAcoef=6.25,
        petBcoef=280700.0,
        peqBcoef=0.0067,
        cdrag=0.013,
        ccanopy=411.0,
    )
    fields = forcing_just_read(
        {k: scalar(v) for k, v in values.items()},
        VerticalForcingOptions(zheight=True, zlev_fixed=2.0, zlevuv_fixed=10.0),
        ttm=1,
        is_watchout=True,
        wind_n_exists=True,
    )
    for name in ["petAcoef", "peqAcoef", "petBcoef", "peqBcoef"]:
        np.testing.assert_array_equal(getattr(fields, name), [[values[name]]])
    with pytest.raises(NotImplementedError, match="WATCHOUT"):
        validate_dim2_switches(Dim2Switches(watchout=True))
    with pytest.raises(NotImplementedError, match="RELAXATION"):
        validate_dim2_switches(Dim2Switches(relaxation=True))


def test_production_precall_adapter_passes_nonzero_tuple_to_actual_energy_solver(monkeypatch):
    from types import SimpleNamespace

    import jax_orchidee.sechiba.enerbil as energy

    case = next(c for c in REFERENCE["cases"] if c["name"] == "positive-dt2700")
    p = {k: np.asarray(v) for k, v in case["inputs"].items()}
    payload = {
        **p,
        "temp_sol": p["psold"] / 1004.675,
        "dt_sechiba": 123.0,
        "temp_sol_pft": p["psold_pft"] / 1004.675,
        "pb": np.asarray([1010.0, 995.0]),
        "lwdown": np.asarray([300.0, 310.0]),
        "swnet": np.asarray([120.0, 130.0]),
        "vbeta2": np.full((2, 3), 0.03),
        "vbeta3": np.full((2, 3), 0.07),
        "vbeta3pot": np.full((2, 3), 0.09),
        "vbeta4": np.asarray([0.1, 0.2]),
        "vbeta4_pft": np.full((2, 3), 0.04),
    }
    original = energy.enerbil_surftemp_explicit_solve
    captured = []

    def observed_solver(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(energy, "enerbil_surftemp_explicit_solve", observed_solver)
    local = energy.run_enerbil_first_step_local_from_precall(
        SimpleNamespace(payload=payload, provenance=("synthetic complete component boundary",)),
        ok_laidev=p["ok_laidev"],
        dt_sechiba=2700.0,
        ok_explicitsnow=False,
    )
    assert len(captured) == 1
    assert float(captured[0]["dt_sechiba"]) == 2700.0
    assert float(local.payload["dt_sechiba"]) == 2700.0
    for name in ["petAcoef", "petBcoef", "peqAcoef", "peqBcoef"]:
        np.testing.assert_array_equal(captured[0][name], p[name])
        np.testing.assert_array_equal(local.payload[name], p[name])
    # The producer for this component is enerbil_begin, not the frozen oracle begin state.
    # This check verifies the real adapter's arguments and raw return selection only.
    np.testing.assert_array_equal(local.payload["vevapp"], local.step.flux.vevapp)
    assert np.all(np.isfinite(local.step.surftemp.qair_new))
    assert np.all(np.abs(np.asarray(local.payload["vevapp"])) > 1e-5)
