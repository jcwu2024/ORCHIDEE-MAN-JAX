"""WATCHOUT q_cdrag adapter transport through the real local drag boundary.

These bounded tests execute the assembler, kwargs adapter, wrapper, local
boundary, drag (including local aero when selected), and qsat kernels. The
downstream beta/photosynthesis chain and ENERBIL payload packaging are explicit
test delegates; no model timestep, day, or training is executed or certified.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.sechiba import diffuco

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "watchout_parameter_transport.json"


def _static_kwargs():
    # Photosynthesis is delegated below. Its unused layout is supplied so this
    # adapter test does not build or compile the full canopy calculation.
    return {
        "rstruct_const": np.full(14, 30.0),
        "trans_co2_pft_params": {"rstruct_const": 30.0, "ext_coeff": 0.5, "lai_light": None},
        "dt_sechiba": 900.0,
        "laimax": 12.0,
        "lai_level_depth": 0.15,
        "min_wind": 0.1,
        "snowcri": 1.5,
        "ok_snowfact": False,
        "rough_dyn": False,
        "ok_laidev": np.ones(14, dtype=bool),
    }


def _precall(*, q_cdrag=None, supply_q_cdrag=True, restart_q_cdrag=None, u=3.0, v=4.0):
    pft_fields = {
        "temp_sol_pft", "veget", "veget_max", "lai", "qsintveg", "rstruct", "roughheight_pft", "humrel", "height"
    }
    restart = {}
    for name in diffuco.DIFFUCO_FIRST_STEP_PRECALL_RESTART_FIELDS:
        shape = (1, 14) if name in pft_fields else (1,)
        if name in {"snowdz", "snowrho", "snowtemp"}:
            shape = (1, 3)
        elif name in {"frac_nobio", "snow_nobio", "snow_nobio_age"}:
            shape = (1, 1)
        restart[name] = np.zeros(shape, dtype=np.float64)
    restart.update(
        temp_sol=np.asarray([300.0]),
        temp_sol_pft=np.full((1, 14), 300.0),
        roughheight=np.asarray([1.0]),
        roughheight_pft=np.ones((1, 14)),
        z0h=np.asarray([0.01]),
        z0m=np.asarray([0.1]),
        qsurf=np.asarray([0.01]),
        rstruct=np.full((1, 14), 30.0),
        frac_snow_veg=np.zeros(1),
        frac_snow_nobio=np.zeros((1, 1)),
    )
    if restart_q_cdrag is not None:
        restart["q_cdrag"] = np.asarray([restart_q_cdrag])
    driver = {
        "u": np.asarray([u]),
        "v": np.asarray([v]),
        "zlev": np.asarray([30.0]),
        "temp_air": np.asarray([299.0]),
        "qair": np.asarray([0.009]),
        "pb": np.asarray([1000.0]),
        "swdown": np.asarray([400.0]),
        "ccanopy": np.asarray([400.0]),
        "precip_rain": np.zeros(1),
        "lalo": np.zeros((1, 2)),
        "neighbours": np.zeros((1, 8), dtype=np.int32),
        "resolution": np.ones((1, 2)),
    }
    if supply_q_cdrag:
        driver["q_cdrag"] = None if q_cdrag is None else np.asarray([q_cdrag])
    return diffuco.assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload=driver,
        restart_payload=restart,
        driver_albedo_payload={"albedo": np.asarray([[0.2, 0.3]])},
        slowproc_derivvar_payload={
            "qsintmax": np.ones((1, 14)),
            "assim_param": np.ones((1, 14, 1)),
            "temp_growth": np.asarray([25.0]),
        },
        diffuco_control_salinity=SimpleNamespace(payload={"control_salinity": np.ones(1)}),
        diffuco_control_inundation=SimpleNamespace(payload={"control_inudate": np.ones(1)}),
        river_routing=False,
        nbp_glo=1,
    )


@pytest.fixture
def beta_delegate(monkeypatch):
    calls = []
    sentinel = object()

    def delegated_beta_chain(**kwargs):
        calls.append(kwargs)
        return sentinel

    def delegated_payload(chain, *, pft_index, pft_output_backgrounds, passthrough):
        assert chain is sentinel
        return diffuco.DiffucoPFT14AfterMainPayload(dict(passthrough), (), tuple(passthrough), ())

    monkeypatch.setattr(diffuco, "diffuco_pft14_c3_beta_process_chain_explicit", delegated_beta_chain)
    monkeypatch.setattr(diffuco, "diffuco_pft14_enerbil_precall_payload", delegated_payload)
    return calls


def _run(precall, *, coupled):
    return diffuco.run_pft14_local_enerbil_precall_from_first_step_precall(
        precall,
        run_def_values={},
        ldq_cdrag_from_gcm=coupled,
        static_kwargs=_static_kwargs(),
        use_jit=False,
    )


@pytest.mark.parametrize("supply_q_cdrag", [False, True], ids=["absent", "none"])
def test_nonwatchout_precall_does_not_require_optional_drag(supply_q_cdrag):
    precall = _precall(supply_q_cdrag=supply_q_cdrag, restart_q_cdrag=0.9)
    assert precall.ok
    assert "q_cdrag" not in precall.payload
    assert "q_cdrag" not in precall.missing_inputs
    assert "q_cdrag" not in precall.missing_local_process_inputs


def test_watchout_precall_keeps_driver_drag_without_checkpoint_fallback():
    precall = _precall(q_cdrag=0.006, restart_q_cdrag=0.9)
    assert precall.ok
    np.testing.assert_array_equal(precall.payload["q_cdrag"], [0.006])
    assert "q_cdrag" in precall.driver_inputs
    assert "q_cdrag" not in precall.restart_inputs


@pytest.mark.parametrize("family", ["ordinary", "daily"])
def test_watchout_saved_record_drag_reaches_real_coupled_boundary(family, beta_delegate):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    seen = []
    for case in fixture[family]["cases"]:
        supplied = case["expected"]["cdrag"]
        assembly = _run(_precall(q_cdrag=supplied, restart_q_cdrag=0.9), coupled=True)
        drag = assembly.result.boundary.drag
        seen.append(float(drag.q_cdrag[0]))
        np.testing.assert_array_equal(assembly.kwargs["q_cdrag"], [supplied])
        np.testing.assert_array_equal(drag.q_cdrag, [supplied])
        # diffuco_main:633-637 broadcasts every PFT, without selecting a PFT.
        np.testing.assert_array_equal(drag.q_cdrag_pft, np.full((1, 14), supplied))
        # diffuco_main:625 and diffuco_raerod:3344-3349, independent source formula.
        np.testing.assert_allclose(drag.raero, [1.0 / (supplied * 5.0)], rtol=1e-14, atol=0.0)
        np.testing.assert_array_equal(beta_delegate[-1]["q_cdrag"], drag.q_cdrag)
        np.testing.assert_array_equal(beta_delegate[-1]["q_cdrag_pft"], drag.q_cdrag_pft)
        np.testing.assert_array_equal(assembly.result.payload.payload["raero"], drag.raero)
        assert "q_cdrag" not in assembly.kwargs["passthrough"]
    assert len(set(seen)) > 1
    assert len(beta_delegate) == len(fixture[family]["cases"])


def test_coupled_drag_keeps_source_minimum_wind(beta_delegate):
    drag = _run(_precall(q_cdrag=0.02, u=0.01, v=0.02), coupled=True).result.boundary.drag
    np.testing.assert_allclose(drag.raero, [500.0], rtol=1e-14, atol=0.0)


def test_explicit_false_computes_local_drag_even_when_watchout_drag_exists(beta_delegate):
    with_forcing = _run(_precall(q_cdrag=0.9), coupled=False)
    without_forcing = _run(_precall(supply_q_cdrag=False), coupled=False)
    assert "q_cdrag" not in with_forcing.kwargs
    np.testing.assert_array_equal(
        with_forcing.result.boundary.drag.q_cdrag, without_forcing.result.boundary.drag.q_cdrag
    )
    assert not np.array_equal(with_forcing.result.boundary.drag.q_cdrag, [0.9])
    assert len(beta_delegate) == 2


def test_coupled_flag_requires_an_explicit_driver_drag():
    precall = _precall(supply_q_cdrag=False, restart_q_cdrag=0.9)
    with pytest.raises(ValueError, match="q_cdrag"):
        diffuco.assemble_pft14_local_enerbil_precall_kwargs_from_first_step_precall(
            precall, run_def_values={}, ldq_cdrag_from_gcm=True, static_kwargs=_static_kwargs()
        )
