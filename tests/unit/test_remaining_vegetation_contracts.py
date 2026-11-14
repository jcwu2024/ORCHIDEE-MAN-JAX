"""Bounded field-admission/codec checks; physical initialization is replaced.

Fortran scope: slowproc_init1603-1644; hydrol_init3042-3075;
hydrol_initialize659-676; hydrol_main1241-1285; restget_p_opp_r2d344-372.
No test treats an assumed IOIPSL missing return as an observed native value.
"""
from collections import namedtuple
from types import SimpleNamespace as NS

import numpy as np
import pytest
from netCDF4 import Dataset

from jax_orchidee.sechiba import hydrol
from jax_orchidee.sechiba.restart_io import (
    read_sechiba_restart_state,
    restore_sechiba_integer_source_fields,
    sechiba_finalize_source_state_from_restart,
)
from jax_orchidee.sechiba.slowproc import SlowprocInitGapError, slowproc_init_pft14_explicit


@pytest.mark.parametrize("mode", ["present", "all_sentinel", "partial_sentinel", "omitted"])
def test_actual_small_nc_reader_preserves_fields_without_inventing_missing_defaults(tmp_path, mode):
    path = tmp_path / "declared-field-boundary.nc"
    stress = np.linspace(0.05, 0.95, 28).reshape(2, 14)
    if mode == "all_sentinel":
        stress[:] = 999999.0
    elif mode == "partial_sentinel":
        stress[0, 13] = 999999.0
    with Dataset(path, "w") as dataset:
        for name, size in (("time", 1), ("y", 1), ("x", 2), ("veget", 14)):
            dataset.createDimension(name, size)
        year = dataset.createVariable("veget_year", "f8", ("time", "y", "x"))
        year[:] = np.asarray([[[-2147483648.0, 2147483647.0]]])
        if mode != "omitted":
            variable = dataset.createVariable("vegstress", "f8", ("time", "veget", "y", "x"))
            variable[:] = stress.T[None, :, None, :]
    read = read_sechiba_restart_state(path)
    assert ("vegstress" in read.fields) == (mode != "omitted")
    if mode != "omitted":
        np.testing.assert_array_equal(read.fields["vegstress"], stress)
    restored = restore_sechiba_integer_source_fields(read.fields)
    np.testing.assert_array_equal(restored["veget_year"], [-2147483648, 2147483647])
    assert restored["veget_year"].dtype == np.int32
    # This deliberately tiny codec fixture is not a complete scientific restart.
    with pytest.raises(ValueError, match="source-state mismatch"):
        sechiba_finalize_source_state_from_restart(read)


@pytest.mark.parametrize("value", [-1.0, 0.0, 999999999.0])
def test_veget_year_integer_codec_has_no_invented_calendar_year_range(value):
    source = {"veget_year": np.asarray([value])}
    out = restore_sechiba_integer_source_fields(source)
    assert out["veget_year"].dtype == np.int32
    assert int(out["veget_year"][0]) == value
    assert source["veget_year"].dtype == np.float64


@pytest.mark.parametrize("value", [np.nan, 1.5, 2147483648.0, -2147483649.0])
def test_veget_year_codec_rejects_non_integral_or_out_of_source_storage_values(value):
    with pytest.raises(ValueError, match="finite integer|signed 32-bit"):
        restore_sechiba_integer_source_fields({"veget_year": np.asarray([value])})


def test_actual_init_rejects_map_vegetation_domain_before_physical_initialization():
    with pytest.raises(SlowprocInitGapError, match="IMPOSE_VEG=n"):
        slowproc_init_pft14_explicit(
            restart={}, veget_max_default=None, frac_nobio_default=0.0,
            height_presc=None, pref_soil_veg=None, ext_coeff_vegetfrac=None,
            nstm=6, diaglev=None, active_pft_index=13, impose_veg=False,
        )


def _init_with_replaced_physics(monkeypatch, restart):
    veget = np.zeros((2, 14))
    veget[:, 13] = [0.8, 0.6]
    tile = np.zeros((2, 6))
    tile[:, 3] = 1.0
    cold_values = dict(
        mc=np.zeros((2, 3, 6)), mcl=np.zeros((2, 3, 6)),
        us=np.zeros((2, 14, 6, 3)),
        zwt_force=np.full((2, 6), 1.0e20), free_drain_coef=np.ones((2, 6)),
        zforce=False, evap_bare_lim_ns=np.zeros((2, 6)),
        vegtot=np.asarray([0.8, 0.6]), water2infilt=np.zeros((2, 6)),
        resdist=tile.copy(), vegtot_old=np.asarray([0.8, 0.6]),
        qsintveg=np.zeros((2, 14)), explicit_snow=None,
    )
    cold = namedtuple("SyntheticColdFields", cold_values)(**cold_values)
    calls = []
    monkeypatch.setattr(hydrol, "_hydrol_init_usda_parameters", lambda values: {})
    def cold_spy(**kwargs):
        calls.append("cold")
        return cold
    def var_spy(**kwargs):
        calls.append("var_init")
        assert "vegstress" not in kwargs
        return NS(_asdict=lambda: {"adapter_marker": np.asarray([71.0, 72.0])})
    def vegetation_spy(**kwargs):
        calls.append("vegetation")
        return NS(_asdict=lambda: {})
    monkeypatch.setattr(hydrol, "hydrol_cold_start_state", cold_spy)
    monkeypatch.setattr(hydrol, "hydrol_var_init", var_spy)
    monkeypatch.setattr(hydrol, "hydrol_vegupd_static_state", vegetation_spy)
    us = np.zeros((2, 14, 6, 3))
    us[0, 13, 3] = [0.1, 0.2, 0.3]
    us[1, 13, 3] = [0.2, 0.3, 0.4]
    result = hydrol.hydrol_init_pft14_owner(
        veget=veget, veget_max=veget, soiltile=tile,
        pref_soil_veg=np.asarray([1] * 13 + [4]), njsc=np.asarray([6, 9]),
        znh_m=np.asarray([0.01, 0.1, 1.0]), dnh_m=np.ones(3), dlh_m=np.ones(3),
        humcste=np.ones(14), altmax=np.zeros((2, 14)),
        restart_state={"us": us, **restart},
    )
    assert calls == ["cold", "var_init", "vegetation"]
    np.testing.assert_array_equal(result.state["adapter_marker"], [71.0, 72.0])
    return result


@pytest.mark.parametrize("mode", ["absent", "present", "all_sentinel", "partial_sentinel"])
def test_actual_hydrol_init_adapter_distinguishes_omitted_and_explicit_values(monkeypatch, mode):
    supplied = np.full((2, 14), 0.17)
    if mode == "all_sentinel":
        supplied[:] = 999999.0
    elif mode == "partial_sentinel":
        supplied[0, 13] = 999999.0
    restart = {} if mode == "absent" else {"vegstress": supplied.copy()}
    result = _init_with_replaced_physics(monkeypatch, restart)
    if mode == "absent":
        expected = np.zeros((2, 14))
        expected[:, 13] = [0.6, 0.9]
        np.testing.assert_allclose(result.state["vegstress"], expected, atol=1e-15)
    else:
        np.testing.assert_array_equal(result.state["vegstress"], supplied)
    np.testing.assert_array_equal(result.state["vegstress"], result.span_state["vegstress"])


@pytest.mark.parametrize("drip", [True, False])
def test_first_stress_consumer_is_masked_in_selected_domain_but_sensitive_if_crop_enabled(drip):
    veget_max = np.zeros((1, 14))
    veget_max[0, 13] = 1.0
    args = dict(
        veget=veget_max * 0.75, veget_max=veget_max, transpot=np.ones((1, 14)) * 2.0,
        evapot=np.asarray([3.0]), precip_rain=np.asarray([0.0]),
        soil_deficit=np.ones((1, 14)) * 4.0, ok_laidev=np.zeros(14, dtype=bool),
        irrig_threshold=np.ones(14) * 0.5, irrig_fulfill=np.ones(14),
        irrig_drip=drip, irrig_dosmax=10.0,
    )
    for explicit_stress in (0.0, 999999.0):
        result = hydrol.hydrol_irrigation_demand_ratio(
            **args, vegstress_old=np.full((1, 14), explicit_stress))
        np.testing.assert_array_equal(result.irrig_demand_ratio, np.zeros((1, 14)))
    args["ok_laidev"][13] = True
    low = hydrol.hydrol_irrigation_demand_ratio(**args, vegstress_old=np.zeros((1, 14)))
    high = hydrol.hydrol_irrigation_demand_ratio(**args, vegstress_old=np.full((1, 14), 999999.0))
    assert float(low.irrig_demand_ratio[0, 13]) == 1.0
    np.testing.assert_array_equal(high.irrig_demand_ratio, np.zeros((1, 14)))
