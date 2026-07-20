from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    read_thermosoil_initialize_restart_fields,
    read_thermosoil_initialize_static_inputs,
    thermosoil_finalize_restart_packet,
    thermosoil_initialize,
)
from jax_orchidee.driver.orchestration import (  # noqa: E402
    _thermosoil_restart_namespace_from_cold_start,
)


class _Moisture:
    def __init__(self) -> None:
        self.shumdiag_perma = np.full((1, 2), 0.3, dtype=np.float64)
        self.mc_layh = np.full((1, 2), 0.3, dtype=np.float64)
        self.mcl_layh = np.full((1, 2), 0.3, dtype=np.float64)
        self.tmc_layh = np.full((1, 2), 0.3, dtype=np.float64)
        self.mc_layh_pft = np.full((1, 2, 1), 0.3, dtype=np.float64)
        self.mcl_layh_pft = np.full((1, 2, 1), 0.3, dtype=np.float64)
        self.tmc_layh_pft = np.full((1, 2, 1), 0.3, dtype=np.float64)


def _coefficient_inputs() -> dict[str, object]:
    return {
        "moisture": _Moisture(),
        "temp_sol_new_pft": np.asarray([[280.0]], dtype=np.float64),
        "snowdz": np.zeros((1, 3), dtype=np.float64),
        "snowrho": np.full((1, 3), 50.0, dtype=np.float64),
        "snowtemp": np.full((1, 3), 273.15, dtype=np.float64),
        "njsc": np.asarray([2], dtype=np.int32),
        "pb": np.asarray([1013.0], dtype=np.float64),
        "dlt": np.asarray([0.1, 0.2, 0.4], dtype=np.float64),
        "dz1": np.asarray([10.0, 5.0], dtype=np.float64),
        "zlt": np.asarray([0.1, 0.3, 0.7], dtype=np.float64),
        "znt": np.asarray([0.05, 0.15, 0.35], dtype=np.float64),
        "dz5": np.asarray([0.5, 0.5], dtype=np.float64),
        "dt_sechiba": 1800.0,
        "frac_snow_veg": np.zeros((1,), dtype=np.float64),
        "frac_snow_nobio": np.zeros((1, 1), dtype=np.float64),
        "totfrac_nobio": np.zeros((1,), dtype=np.float64),
    }


def _initialize(**overrides):
    values = {
        "ngrnd": 3,
        "nscm": 12,
        "temp_sol_new": np.asarray([280.0], dtype=np.float64),
        "veget_max": np.asarray([[1.0]], dtype=np.float64),
        "coefficient_inputs": _coefficient_inputs(),
        "external_refsoc": np.zeros((1, 3), dtype=np.float64),
    }
    values.update(overrides)
    return thermosoil_initialize(**values)


def test_cold_start_writes_fortran_defaults_and_recomputes_coefficients():
    result = _initialize()

    assert result.ptn_source == "THERMOSOIL_TPRO"
    assert result.refsoc_source == "external_refSOC"
    assert result.calculate_coef
    np.testing.assert_array_equal(result.ptn, np.full((1, 3, 1), 280.0))
    np.testing.assert_array_equal(result.ptn_beg, result.ptn)
    np.testing.assert_array_equal(result.temp_sol_beg, [280.0])
    np.testing.assert_array_equal(result.gtemp, [0.0])
    np.testing.assert_array_equal(result.shum_ngrnd_perma, np.full((1, 3, 1), 0.3))
    np.testing.assert_array_equal(result.shum_ngrnd_permalong, np.ones((1, 3, 1)))
    np.testing.assert_array_equal(result.veget_mask_2d, np.ones((1, 1), dtype=bool))
    assert result.cgrnd.shape == (1, 2, 1)
    assert result.cgrnd_snow.shape == (1, 3)
    assert result.stempdiag.shape == (1, 2)
    assert "thermosoil_initialize lines 327-726" in result.provenance[0]


def test_cold_start_recurrence_namespace_preserves_fortran_initial_state():
    ptn = np.full((1, 3, 2), 295.0, dtype=np.float64)
    closure = _initialize(
        ngrnd=3,
        veget_max=np.asarray([[0.0, 1.0]], dtype=np.float64),
        external_ptn=ptn,
        read_reftemp=True,
        coefficient_inputs={
            **_coefficient_inputs(),
            "temp_sol_new_pft": np.asarray([[280.0, 280.0]], dtype=np.float64),
            "moisture": type(
                "Moisture",
                (),
                {
                    "shumdiag_perma": np.full((1, 2), 0.3),
                    "mc_layh": np.full((1, 2), 0.3),
                    "mcl_layh": np.full((1, 2), 0.3),
                    "tmc_layh": np.full((1, 2), 0.3),
                    "mc_layh_pft": np.full((1, 2, 2), 0.3),
                    "mcl_layh_pft": np.full((1, 2, 2), 0.3),
                    "tmc_layh_pft": np.full((1, 2, 2), 0.3),
                },
            )(),
        },
    )
    coef = type(
        "Closure",
        (),
        {
            "ok": True,
            "coef": type(
                "Coef",
                (),
                {
                    "soil": type("Soil", (), {"cgrnd": closure.cgrnd, "dgrnd": closure.dgrnd})(),
                    "cgrnd_snow": closure.cgrnd_snow,
                    "dgrnd_snow": closure.dgrnd_snow,
                    "lambda_snow": closure.lambda_snow,
                },
            )(),
            "getdiff": type("Getdiff", (), {"pcapa_en": np.ones_like(ptn)})(),
        },
    )()
    state = _thermosoil_restart_namespace_from_cold_start(
        ptn=ptn,
        coef=coef,
        refsoc=np.zeros((1, 3)),
        temp_sol_beg=np.asarray([280.0]),
        shum_ngrnd_permalong=np.ones_like(ptn),
    )

    np.testing.assert_array_equal(state.gtemp, [0.0])
    np.testing.assert_array_equal(state.temp_sol_beg, [280.0])
    np.testing.assert_array_equal(state.shum_ngrnd_permalong, np.ones_like(ptn))
    np.testing.assert_array_equal(state.veget_mask_2d, np.ones((1, 2), dtype=bool))
    assert state.e_soil_lat is None


def test_cold_restart_namespace_carries_active_ecorr_state():
    closure = _initialize(restart_fields={}, external_refsoc=np.zeros((1, 3)))
    ptn = np.full((1, 3, 2), 280.0)
    coef = type(
        "Closure",
        (),
        {
            "ok": True,
            "coef": type(
                "Coef",
                (),
                {
                    "soil": type("Soil", (), {"cgrnd": closure.cgrnd, "dgrnd": closure.dgrnd})(),
                    "cgrnd_snow": closure.cgrnd_snow,
                    "dgrnd_snow": closure.dgrnd_snow,
                    "lambda_snow": closure.lambda_snow,
                },
            )(),
            "getdiff": type("Getdiff", (), {"pcapa_en": np.ones_like(ptn)})(),
        },
    )()
    e_soil_lat = np.asarray([[0.0, 0.0]])

    state = _thermosoil_restart_namespace_from_cold_start(
        ptn=ptn,
        coef=coef,
        refsoc=np.zeros((1, 3)),
        temp_sol_beg=np.asarray([280.0]),
        shum_ngrnd_permalong=np.ones_like(ptn),
        e_soil_lat=e_soil_lat,
    )

    np.testing.assert_array_equal(state.e_soil_lat, e_soil_lat)


def test_restart_path_preserves_complete_coefficient_group_and_state():
    restart = {
        "ptn": np.full((1, 3, 1), 276.0),
        "refSOC": np.full((1, 3), 100.0),
        "shum_ngrnd_perma": np.full((1, 3, 1), 0.7),
        "shum_ngrnd_prmlng": np.full((1, 3, 1), 0.8),
        "gtemp": np.asarray([275.0]),
        "soilcap": np.asarray([11.0]),
        "soilcap_pft": np.asarray([[12.0]]),
        "soilflx": np.asarray([13.0]),
        "soilflx_pft": np.asarray([[14.0]]),
        "cgrnd": np.full((1, 2, 1), 15.0),
        "dgrnd": np.full((1, 2, 1), 16.0),
        "cgrnd_snow": np.full((1, 3), 17.0),
        "dgrnd_snow": np.full((1, 3), 18.0),
        "lambda_snow": np.asarray([19.0]),
    }
    result = _initialize(restart_fields=restart, external_refsoc=None)

    assert result.ptn_source == "restart"
    assert result.refsoc_source == "restart"
    assert not result.calculate_coef
    np.testing.assert_array_equal(result.ptn, restart["ptn"])
    np.testing.assert_array_equal(result.soilcap, restart["soilcap"])
    np.testing.assert_array_equal(result.soilflx_pft, restart["soilflx_pft"])
    np.testing.assert_array_equal(result.cgrnd, restart["cgrnd"])
    np.testing.assert_array_equal(result.lambda_snow, restart["lambda_snow"])
    np.testing.assert_array_equal(result.gtemp, restart["gtemp"])
    np.testing.assert_array_equal(result.shum_ngrnd_permalong, restart["shum_ngrnd_prmlng"])


def test_incomplete_restart_coefficient_group_is_recomputed_as_one_group():
    restart = {
        "soilcap": np.asarray([999.0]),
        "soilcap_pft": np.asarray([[999.0]]),
        "soilflx": np.asarray([999.0]),
        "soilflx_pft": np.asarray([[999.0]]),
        "cgrnd": np.full((1, 2, 1), 999.0),
        "dgrnd": np.full((1, 2, 1), 999.0),
        "cgrnd_snow": np.full((1, 3), 999.0),
        "dgrnd_snow": np.full((1, 3), 999.0),
        # lambda_snow is absent: Fortran sets calculate_coef and replaces all.
    }
    result = _initialize(restart_fields=restart)

    assert result.calculate_coef
    assert not np.array_equal(np.asarray(result.soilcap), restart["soilcap"])
    assert not np.array_equal(np.asarray(result.cgrnd), restart["cgrnd"])


def test_reference_temperature_branch_requires_and_uses_external_static_field():
    ptn = np.full((1, 3, 1), 271.25)
    result = _initialize(read_reftemp=True, external_ptn=ptn)

    assert result.ptn_source == "external_reference_temperature"
    np.testing.assert_array_equal(result.ptn, ptn)
    with pytest.raises(ValueError, match="READ_REFTEMP requires"):
        _initialize(read_reftemp=True)


def test_configuration_state_and_wetdiaglong_follow_fortran_initialization_order():
    result = _initialize(
        ok_freeze_thermix=False,
        ok_pc=False,
        ok_leak=False,
        ok_wetdiaglong=True,
        satsoil=True,
        ok_zimov=True,
        bedrock_flag=1,
    )

    assert result.ok_shum_ngrnd_permalong
    assert result.satsoil
    assert result.ok_zimov
    assert result.brk_flag == 1
    np.testing.assert_array_equal(result.shum_ngrnd_permalong, np.ones((1, 3, 1)))

    inactive = _initialize(ok_freeze_thermix=False, ok_pc=False, ok_leak=False)
    assert not inactive.ok_shum_ngrnd_permalong
    np.testing.assert_array_equal(inactive.shum_ngrnd_permalong, np.zeros((1, 3, 1)))


def test_finalize_packet_writes_exact_conditional_restart_fields():
    state = _initialize(ok_Ecorr=True, restart_fields={"e_soil_lat": np.asarray([[3.0]])})
    kwargs = dict(
        ptn=state.ptn,
        refsoc=state.refsoc,
        shum_ngrnd_perma=state.shum_ngrnd_perma,
        shum_ngrnd_permalong=state.shum_ngrnd_permalong,
        ok_shum_ngrnd_permalong=state.ok_shum_ngrnd_permalong,
        e_soil_lat=state.e_soil_lat,
        ok_Ecorr=True,
        cgrnd=state.cgrnd,
        dgrnd=state.dgrnd,
        gtemp=state.gtemp,
        soilcap=state.soilcap,
        soilcap_pft=state.soilcap_pft,
        soilflx=state.soilflx,
        soilflx_pft=state.soilflx_pft,
        cgrnd_snow=state.cgrnd_snow,
        dgrnd_snow=state.dgrnd_snow,
        lambda_snow=state.lambda_snow,
    )
    packet = thermosoil_finalize_restart_packet(**kwargs)

    assert set(packet) == {
        "ptn",
        "refSOC",
        "shum_ngrnd_prmlng",
        "shum_ngrnd_perma",
        "e_soil_lat",
        "cgrnd",
        "dgrnd",
        "gtemp",
        "soilcap",
        "soilcap_pft",
        "soilflx",
        "soilflx_pft",
        "cgrnd_snow",
        "dgrnd_snow",
        "lambda_snow",
    }
    with pytest.raises(ValueError, match="restart branch is inactive"):
        thermosoil_finalize_restart_packet(**{**kwargs, "ok_shum_ngrnd_permalong": False, "ok_Ecorr": True})


def test_static_file_boundary_reuses_driver_readers(monkeypatch):
    import jax_orchidee.driver.static as driver_static

    calls: list[str] = []

    def reftemp(*args, **kwargs):
        calls.append("reftemp")
        return np.full((1, 3, 1), 272.0), {"temp_meta": np.asarray([1])}

    def refsoc(*args, **kwargs):
        calls.append("refsoc")
        return np.full((1, 3), 2.0), {"soc_meta": np.asarray([2])}

    monkeypatch.setattr(driver_static, "read_paper_thermosoil_reftemp_static_field", reftemp)
    monkeypatch.setattr(driver_static, "read_paper_thermosoil_refsoc_static_field", refsoc)
    result = read_thermosoil_initialize_static_inputs(
        "case.yaml",
        lalo=np.asarray([[0.0, 0.0]]),
        resolution_m=np.asarray([[1.0, 1.0]]),
        ngrnd=3,
        nvm=1,
        read_reftemp=True,
        read_refsoc=True,
    )

    assert calls == ["reftemp", "refsoc"]
    np.testing.assert_array_equal(result.ptn, np.full((1, 3, 1), 272.0))
    np.testing.assert_array_equal(result.refsoc, np.full((1, 3), 2.0))
    assert set(result.metadata) == {"temp_meta", "soc_meta"}


def test_error_guards_reject_missing_assets_shapes_and_mixed_sentinel(tmp_path: Path):
    with pytest.raises(ValueError, match="use_refSOC requires"):
        _initialize(external_refsoc=None)
    with pytest.raises(ValueError, match="restart ptn must have shape"):
        _initialize(restart_fields={"ptn": np.zeros((1, 2, 1))})
    mixed = np.asarray([[[999999.0], [280.0], [280.0]]])
    with pytest.raises(ValueError, match="mixes val_exp"):
        _initialize(restart_fields={"ptn": mixed})
    with pytest.raises(ValueError, match="USDA nscm=12"):
        _initialize(nscm=3)
    with pytest.raises(ValueError, match="BEDROCK_FLAG"):
        _initialize(bedrock_flag=2)
    with pytest.raises(FileNotFoundError, match="restart file does not exist"):
        read_thermosoil_initialize_restart_fields(tmp_path / "missing.nc")
