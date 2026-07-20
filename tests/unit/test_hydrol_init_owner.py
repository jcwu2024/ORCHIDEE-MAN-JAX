from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.sechiba.hydrol import (
    HydrolInitPFT14Switches,
    USDA_MCS,
    USDA_NVAN,
    hydrol_init_pft14_owner,
)


def _inputs(*, npts: int = 2):
    nvm = 14
    nstm = 6
    nslm = 11
    znh = np.geomspace(0.001, 2.0, nslm)
    dnh = np.empty(nslm)
    dnh[0] = znh[0]
    dnh[1:] = np.diff(znh)
    dlh = np.empty(nslm)
    dlh[:-1] = (dnh[:-1] + dnh[1:]) / 2.0
    dlh[-1] = dnh[-1] / 2.0
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[:, 13] = np.asarray([0.8, 0.6][:npts])
    veget_max[:, 0] = 1.0 - veget_max[:, 13]
    soiltile = np.zeros((npts, nstm), dtype=np.float64)
    soiltile[:, 2] = 0.4
    soiltile[:, 3] = 0.6
    return {
        "veget": veget_max.copy(),
        "veget_max": veget_max,
        "soiltile": soiltile,
        "pref_soil_veg": np.asarray([1] + [3] * 12 + [4], dtype=np.int32),
        "njsc": np.asarray([6, 9][:npts], dtype=np.int32),
        "znh_m": znh,
        "dnh_m": dnh,
        "dlh_m": dlh,
        "humcste": np.linspace(0.2, 4.0, nvm),
        "altmax": np.zeros((npts, nvm), dtype=np.float64),
    }


def test_cold_start_owner_composes_source_order_without_invented_allocation_values():
    result = hydrol_init_pft14_owner(**_inputs())

    np.testing.assert_allclose(result.state["mc"], 0.3)
    np.testing.assert_allclose(result.state["mcl"], result.state["mc"])
    np.testing.assert_allclose(result.state["humrel"], 0.0)
    np.testing.assert_allclose(result.state["vegstress"], 0.0)
    np.testing.assert_allclose(result.state["free_drain_coef"][:, 3], 0.0)
    np.testing.assert_allclose(result.state["free_drain_coef"][:, 5], 0.0)
    np.testing.assert_allclose(result.state["zwt_force"], 1.0e20)
    assert result.state["zforce"] is False
    assert result.state["explicit_snow"] is not None
    np.testing.assert_allclose(result.restart_static.mcs, [USDA_MCS[5], USDA_MCS[8]])
    assert result.allocation_only["us"] == (2, 14, 6, 11)
    assert result.allocation_only["kfact_root"] == (2, 11, 6)
    assert "ru_ns" not in result.state
    assert result.restart_inputs == ()
    assert result.process_order[:4] == (
        "soil_parameter_selection",
        "allocation",
        "restart_read",
        "setvar_defaults",
    )
    assert any("hydrol_init lines 2028-2252" in item for item in result.provenance)


def test_restart_fields_follow_all_missing_fallbacks_and_are_preserved():
    inputs = _inputs(npts=1)
    mc = np.full((1, 11, 6), 0.27)
    us = np.zeros((1, 14, 6, 11))
    us[:, 13, 3, :] = 0.5
    result = hydrol_init_pft14_owner(
        **inputs,
        restart_state={
            "mc": mc,
            "us": us,
            "zwt_force": np.asarray([[1.0e20, 1.0e20, 1.0e20, 1.5, 1.0e20, 1.0e20]]),
        },
    )

    np.testing.assert_allclose(result.state["mc"], mc)
    # hydrol_init lines 2819-2821: an all-missing mcl is copied from mc.
    np.testing.assert_allclose(result.state["mcl"], mc)
    np.testing.assert_allclose(result.state["humrel"][0, 13], 5.5)
    np.testing.assert_allclose(result.state["vegstress"][0, 13], 5.5)
    assert result.state["zforce"] is True
    assert result.restart_inputs == ("mc", "us", "zwt_force")


def test_run_def_soil_parameters_drive_the_following_var_init_tables():
    nvan = np.asarray(USDA_NVAN) * 1.01
    result = hydrol_init_pft14_owner(
        **_inputs(npts=1),
        hydrol_init_config={"CWRR_N_VANGENUCHTEN": nvan},
    )

    np.testing.assert_allclose(result.soil_parameters["nvan"], nvan)
    # Texture 6 is selected by njsc=6 and must reach the lookup-table owner.
    assert result.restart_static.mineral_tables.nvan_mod[0, 0] != pytest.approx(USDA_NVAN[5])


@pytest.mark.parametrize(
    ("key", "values"),
    [
        ("CWRR_N_VANGENUCHTEN", np.zeros(12)),
        ("VWC_SAT", np.full(12, 1.1)),
        ("WETNESS_TRANSPIR_MAX", np.full(12, 0.0)),
        ("VWC_MAX_FOR_DRY_ALB", np.full(12, 0.3)),
    ],
)
def test_hydrol_init_parameter_failures_match_fortran_guards(key, values):
    with pytest.raises(ValueError):
        hydrol_init_pft14_owner(**_inputs(npts=1), hydrol_init_config={key: values})


def test_fixed_branches_are_explicit_and_explicit_snow_can_be_inactive():
    switches = HydrolInitPFT14Switches(ok_explicitsnow=False, ok_freeze_cwrr=False)
    result = hydrol_init_pft14_owner(**_inputs(npts=1), switches=switches)

    assert result.state["explicit_snow"] is None
    assert result.cold_start.profil_froz_hydro is None
    assert result.cold_start.temp_hydro is None
    with pytest.raises(ValueError, match="USDA"):
        hydrol_init_pft14_owner(
            **_inputs(npts=1),
            switches=HydrolInitPFT14Switches(soiltype_classif=3),
        )
