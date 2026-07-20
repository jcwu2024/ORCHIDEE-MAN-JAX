from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    _hydrol_main_top_diagnostics,
    build_mineral_cwrr_tables,
    hydrol_alma_step,
    hydrol_irrigation_demand_ratio,
    hydrol_main_step,
    hydrol_waterbal_step,
)
from tests.unit.test_hydrol_module_step import _module_inputs  # noqa: E402


def _bucket_main_inputs():
    module = _module_inputs()
    module["mineral_tables"] = build_mineral_cwrr_tables(
        njsc=2,
        mcs=module["mcs"],
        z_m=module["zz_mm"] / 1000.0,
    )
    module["is_crop_soil"] = np.asarray([False, False, False, True, False, False])
    npts, nvm = module["veget_max"].shape
    nslm = module["mc"].shape[1]
    nroot = np.zeros((npts, nvm, nslm))
    nroot[:, :, 1:] = 1.0 / (nslm - 1)
    diagnostic = {
        "nroot": nroot,
        "dz_mm": module["dz_mm"],
        "dh_mm": np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
        "njsc": module["njsc"],
        "mcs": module["mcs"],
        "mineral_tables": module["mineral_tables"],
        "vevapnu": module["vevapnu"],
        "evapot": module.get("evapot"),
    }
    snow = {
        "precip_rain": module["precip_rain"],
        "precip_snow": np.asarray([2.0]),
        "temp_sol_new": np.asarray([275.0]),
        "soilcap": np.asarray([2.0e6]),
        "frac_nobio": np.zeros((npts, 1)),
        "totfrac_nobio": np.zeros(npts),
        "vevapsno": np.zeros(npts),
        "snow": np.asarray([3.0]),
        "snow_age": np.zeros(npts),
        "snow_nobio": np.zeros((npts, 1)),
        "snow_nobio_age": np.zeros((npts, 1)),
    }
    irrigation = {
        "veget": module["veget"],
        "veget_max": module["veget_max"],
        "transpot": np.asarray([[0.0, 0.0, 0.0, 8.0]]),
        "evapot": np.asarray([1.0]),
        "precip_rain": module["precip_rain"],
        "vegstress_old": np.asarray([[1.0, 1.0, 1.0, 0.1]]),
        "soil_deficit": np.zeros((npts, nvm)),
        "ok_laidev": np.asarray([False, False, False, True]),
        "irrig_threshold": np.asarray([0.0, 0.0, 0.0, 0.5]),
        "irrig_fulfill": np.asarray([0.0, 0.0, 0.0, 0.8]),
        "irrig_drip": True,
        "irrig_dosmax": 10.0,
    }
    routing = {
        "vegtot": module["vegtot"],
        "returnflow": np.asarray([0.2]),
        "reinfiltration": np.asarray([0.1]),
        "irrigation": np.asarray([1.5]),
        "veget_max": module["veget_max"],
        "pref_soil_veg": module["pref_soil_veg"],
        "soiltile": module["soiltile"],
        "is_crop_soil": module["is_crop_soil"],
        "ok_laidev": irrigation["ok_laidev"],
    }
    return module, diagnostic, snow, irrigation, routing


def test_hydrol_main_bucket_branch_runs_in_source_order_and_writes_state():
    """Fortran: hydrol.f90::hydrol_main lines 1159-1650, bucket-snow arm."""

    module, diagnostic, snow, irrigation, routing = _bucket_main_inputs()
    result = hydrol_main_step(
        module_inputs=module,
        diagnostic_inputs=diagnostic,
        snow_inputs=snow,
        irrigation_inputs=irrigation,
        routing_inputs=routing,
        humcste=np.ones(module["veget_max"].shape[1]),
        dlh=np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
        tmcs=np.full(module["soiltile"].shape, 100.0),
        itopmax=2,
        zmaxh_m=2.0,
        ok_explicitsnow=False,
        drain_upd=np.asarray([0.25]),
        runoff_upd=np.asarray([0.5]),
        natural=np.asarray([True, True, True, False]),
        alma_inputs={
            "tot_watveg_beg": np.asarray([0.1]),
            "tot_watsoil_beg": np.asarray([10.0]),
            "snow_beg": np.asarray([3.0]),
            "mx_eau_var": np.asarray([100.0]),
        },
    )

    assert result.snow_step is None
    assert result.bucket_snow_step is not None
    np.testing.assert_allclose(result.module.canop.precisol.sum(axis=1), result.module.canop.precip2ground.sum(axis=1) + result.module.canop.canopy2ground.sum(axis=1))
    np.testing.assert_allclose(result.runoff, result.diagnostics.runoff + 0.5)
    np.testing.assert_allclose(result.drainage, result.diagnostics.drainage + 0.25)
    np.testing.assert_allclose(result.fwet_out, result.fwet)
    np.testing.assert_allclose(result.fsat, 0.0)
    assert result.irrigation_demand.irrig_demand_ratio[0, 3] == pytest.approx(1.0)
    assert result.routing.irrigation_soil[0, 3] > 0.0
    assert result.soil_deficit[0, 3] >= 0.0
    assert result.land_nroot.shape == (1, module["mc"].shape[1])
    assert result.land_mcs.shape == (1, module["mc"].shape[1])
    assert np.isfinite(np.asarray(result.twbr)).all()
    np.testing.assert_allclose(result.soilwet, result.diagnostics.humtot / 100.0)
    np.testing.assert_allclose(result.state_writeback["tot_watsoil_beg"], result.diagnostics.humtot)
    for name in ("mc", "mcl", "tmc", "qsintveg", "snow", "snowdepth", "soil_deficit"):
        assert name in result.state_writeback


def test_hydrol_main_explicit_snow_forcing_reaches_same_step_canopy_and_state():
    """Fortran: hydrol.f90::hydrol_main lines 1177-1190 precede lines 1232-1234."""

    module, diagnostic, _, irrigation, routing = _bucket_main_inputs()
    npts = module["veget_max"].shape[0]
    snow = {
        "precip_rain": module["precip_rain"],
        "precip_snow": np.asarray([1.0]),
        "temp_air": np.asarray([268.0]),
        "pb": np.asarray([1013.0]),
        "u": np.zeros(npts),
        "v": np.zeros(npts),
        "temp_sol_new": np.asarray([268.0]),
        "soilcap": np.asarray([2.0e6]),
        "pgflux": np.asarray([-5.0]),
        "frac_nobio": np.zeros((npts, 1)),
        "totfrac_nobio": np.zeros(npts),
        "gtemp": np.asarray([268.0]),
        "lambda_snow": np.asarray([0.5]),
        "cgrnd_snow": np.asarray([[260.0, 258.0, 0.0]]),
        "dgrnd_snow": np.asarray([[0.1, 0.2, 0.0]]),
        "vevapsno": np.zeros(npts),
        "snow_age": np.zeros(npts),
        "snow_nobio_age": np.zeros((npts, 1)),
        "snow_nobio": np.zeros((npts, 1)),
        "snowrho": np.full((npts, 3), 50.0),
        "snowgrain": np.zeros((npts, 3)),
        "snowdz": np.zeros((npts, 3)),
        "snowtemp": np.full((npts, 3), 273.15),
        "snowheat": np.zeros((npts, 3)),
        "snow": np.zeros(npts),
        "temp_sol_add": np.zeros(npts),
        "snowliq": np.zeros((npts, 3)),
        "subsnownobio": np.zeros((npts, 1)),
        "grndflux": np.zeros(npts),
        "snowmelt": np.zeros(npts),
        "soilflxresid": np.zeros(npts),
    }
    result = hydrol_main_step(
        module_inputs=module,
        diagnostic_inputs=diagnostic,
        snow_inputs=snow,
        irrigation_inputs=irrigation,
        routing_inputs=routing,
        humcste=np.ones(module["veget_max"].shape[1]),
        dlh=np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
        tmcs=np.full(module["soiltile"].shape, 100.0),
        itopmax=2,
        zmaxh_m=2.0,
        ok_explicitsnow=True,
        alma_inputs={
            "tot_watveg_beg": np.zeros(npts),
            "tot_watsoil_beg": np.zeros(npts),
            "snow_beg": np.zeros(npts),
            "mx_eau_var": np.ones(npts) * 100.0,
        },
    )

    assert result.snow_step is not None
    assert result.bucket_snow_step is None
    np.testing.assert_allclose(result.state_writeback["tot_melt"], result.snow_step.snow_state.tot_melt)
    np.testing.assert_allclose(result.state_writeback["snowliq"], result.snow_step.snowliq)
    assert "provenance" not in result.state_writeback
    assert "notes" not in result.state_writeback


def test_hydrol_main_flood_irrigation_branch_uses_soil_deficit_and_normalizes():
    """Fortran: hydrol.f90::hydrol_main lines 1262-1273."""

    result = hydrol_irrigation_demand_ratio(
        veget=np.asarray([[0.0, 0.2, 0.3]]),
        veget_max=np.asarray([[0.0, 0.4, 0.6]]),
        transpot=np.zeros((1, 3)),
        evapot=np.zeros(1),
        precip_rain=np.zeros(1),
        vegstress_old=np.asarray([[1.0, 0.1, 0.1]]),
        soil_deficit=np.asarray([[0.0, 2.0, 1.0]]),
        ok_laidev=np.asarray([False, True, True]),
        irrig_threshold=np.asarray([0.0, 0.5, 0.5]),
        irrig_fulfill=np.ones(3),
        irrig_drip=False,
        irrig_dosmax=10.0,
    )

    np.testing.assert_allclose(result.irrig_demand_ratio, [[0.0, 0.8 / 1.4, 0.6 / 1.4]])


def test_hydrol_main_top_diagnostics_preserve_landpoint_masks_and_source_order():
    """Fortran: hydrol.f90::hydrol_main lines 1330-1395."""

    mc = np.asarray([[[0.2], [0.3], [0.4]], [[0.5], [0.6], [0.7]]])
    humtop, land_nroot, land_dlh, land_mcs = _hydrol_main_top_diagnostics(
        mc=mc,
        soiltile=np.ones((2, 1)),
        vegtot=np.asarray([1.0, 0.0]),
        veget_max=np.asarray([[0.0, 1.0], [0.0, 0.0]]),
        nroot=np.asarray([[[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], [[0.0, 0.0, 0.0], [0.2, 0.3, 0.5]]]),
        dlh=np.asarray([1.0, 2.0, 3.0]),
        tmcs=np.asarray([[100.0], [200.0]]),
        dz_mm=np.asarray([0.0, 2.0, 4.0]),
        itopmax=1,
        zmaxh_m=2.0,
        min_sechiba=1.0e-8,
    )

    np.testing.assert_allclose(humtop, [0.225, 0.0])
    np.testing.assert_allclose(land_nroot[0], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(land_nroot[1], 0.0)
    np.testing.assert_allclose(land_dlh, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(land_mcs[0], [0.05, 0.05, 0.05])
    np.testing.assert_allclose(land_mcs[1], 0.0)


def test_hydrol_main_requires_raw_topmodel_process_inputs():
    """Fortran: hydrol.f90::hydrol_main lines 1159-1162."""

    with pytest.raises(ValueError, match="raw hydro_subgrid_main inputs"):
        hydrol_main_step(
            module_inputs={},
            diagnostic_inputs={},
            snow_inputs={},
            irrigation_inputs={},
            routing_inputs={},
            humcste=[],
            dlh=[],
            tmcs=[],
            itopmax=1,
            zmaxh_m=2.0,
            topm_calcul=True,
            alma_inputs={},
        )


def test_hydrol_waterbal_updates_inventory_and_only_warns_on_residual():
    """Fortran: hydrol.f90::hydrol_waterbal lines 9407-9475."""

    result = hydrol_waterbal_step(
        tot_water_beg=np.asarray([13.0]),
        vegtot=np.asarray([0.8]),
        totfrac_nobio=np.asarray([0.2]),
        qsintveg=np.asarray([[1.0, 2.0]]),
        humtot=np.asarray([10.0]),
        snow=np.asarray([1.0]),
        snow_nobio=np.asarray([[0.5]]),
        precip_rain=np.asarray([2.0]),
        precip_snow=np.asarray([0.0]),
        returnflow=np.asarray([0.0]),
        reinfiltration=np.asarray([0.0]),
        irrigation=np.asarray([0.0]),
        vevapwet=np.asarray([[0.0, 0.0]]),
        transpir=np.asarray([[0.0, 0.0]]),
        vevapnu=np.asarray([0.0]),
        vevapsno=np.asarray([0.0]),
        vevapflo=np.asarray([0.0]),
        floodout=np.asarray([0.0]),
        runoff=np.asarray([0.0]),
        drainage=np.asarray([0.0]),
    )
    np.testing.assert_allclose(result.tot_water_end, [14.5])
    np.testing.assert_allclose(result.tot_flux, [2.0])
    np.testing.assert_allclose(result.residual, [-0.5])
    assert bool(np.asarray(result.conservation_warning)[0])
    np.testing.assert_allclose(result.tot_water_beg, result.tot_water_end)


def test_hydrol_waterbal_fraction_mismatch_is_the_fatal_source_error():
    kwargs = dict(
        tot_water_beg=np.asarray([0.0]),
        vegtot=np.asarray([0.7]),
        totfrac_nobio=np.asarray([0.2]),
        qsintveg=np.zeros((1, 1)),
        humtot=np.zeros(1),
        snow=np.zeros(1),
        snow_nobio=np.zeros((1, 1)),
        precip_rain=np.zeros(1),
        precip_snow=np.zeros(1),
        returnflow=np.zeros(1),
        reinfiltration=np.zeros(1),
        irrigation=np.zeros(1),
        vevapwet=np.zeros((1, 1)),
        transpir=np.zeros((1, 1)),
        vevapnu=np.zeros(1),
        vevapsno=np.zeros(1),
        vevapflo=np.zeros(1),
        floodout=np.zeros(1),
        runoff=np.zeros(1),
        drainage=np.zeros(1),
    )
    with pytest.raises(ValueError, match="fractions fail"):
        hydrol_waterbal_step(**kwargs)


def test_hydrol_alma_init_returns_before_delta_outputs():
    """Fortran: hydrol.f90::hydrol_alma lines 9533-9545."""

    result = hydrol_alma_step(
        qsintveg=np.asarray([[1.0, 2.0], [0.5, 0.25]]),
        humtot=np.asarray([10.0, 20.0]),
        snow=np.asarray([3.0, 4.0]),
        snow_nobio=np.asarray([[1.0, 2.0], [0.0, 1.0]]),
        tot_watveg_beg=np.asarray([-1.0, -1.0]),
        tot_watsoil_beg=np.asarray([-1.0, -1.0]),
        snow_beg=np.asarray([-1.0, -1.0]),
        mx_eau_var=np.asarray([100.0, 0.0]),
        lstep_init=True,
    )
    np.testing.assert_allclose(result.tot_watveg_beg, [3.0, 0.75])
    np.testing.assert_allclose(result.tot_watsoil_beg, [10.0, 20.0])
    np.testing.assert_allclose(result.snow_beg, [6.0, 5.0])
    assert result.delintercept is None
    assert result.delsoilmoist is None
    assert result.delswe is None
    assert result.soilwet is None


def test_hydrol_alma_regular_step_updates_inventory_and_soilwet():
    """Fortran: hydrol.f90::hydrol_alma lines 9547-9573."""

    result = hydrol_alma_step(
        qsintveg=np.asarray([[1.0, 2.0], [0.5, 0.25]]),
        humtot=np.asarray([10.0, 20.0]),
        snow=np.asarray([3.0, 4.0]),
        snow_nobio=np.asarray([[1.0, 2.0], [0.0, 1.0]]),
        tot_watveg_beg=np.asarray([2.0, 1.0]),
        tot_watsoil_beg=np.asarray([8.0, 25.0]),
        snow_beg=np.asarray([5.0, 6.0]),
        mx_eau_var=np.asarray([100.0, 0.0]),
        lstep_init=False,
    )
    np.testing.assert_allclose(result.delintercept, [1.0, -0.25])
    np.testing.assert_allclose(result.delsoilmoist, [2.0, -5.0])
    np.testing.assert_allclose(result.delswe, [1.0, -1.0])
    np.testing.assert_allclose(result.soilwet, [0.1, 0.0])
