from types import SimpleNamespace

import numpy as np
import pytest

import jax_orchidee.sechiba.slowproc as slowproc_module
from jax_orchidee.sechiba.slowproc import (
    SlowprocMainPft14Switches,
    slowproc_main_pft14_step,
)


def _states(nland=2):
    veget_max = np.zeros((nland, 14), dtype=np.float64)
    veget_max[:, 0] = np.asarray([0.2, 0.1])[:nland]
    veget_max[:, 13] = 1.0 - veget_max[:, 0]
    lai = np.zeros_like(veget_max)
    lai[:, 13] = np.asarray([2.0, 1.0])[:nland]
    vegetation = {
        "lai": lai,
        "frac_age": np.zeros((nland, 14, 4), dtype=np.float64),
        "height": np.zeros_like(veget_max),
        "veget": veget_max.copy(),
        "frac_nobio": np.zeros((nland, 1), dtype=np.float64),
        "veget_max": veget_max,
        "totfrac_nobio": np.zeros(nland, dtype=np.float64),
        "soiltile": np.ones((nland, 1), dtype=np.float64),
        "qsintmax": np.full_like(veget_max, 7.0),
        "temp_growth": np.asarray([6.0, 4.0], dtype=np.float64)[:nland],
    }
    soil = {
        "clayfraction": np.asarray([0.2, 0.3], dtype=np.float64)[:nland],
        "siltfraction": np.asarray([0.3, 0.1], dtype=np.float64)[:nland],
        "bulk_density": np.asarray([1400.0, -1.0], dtype=np.float64)[:nland],
        "soil_ph": np.asarray([6.5, 7.0], dtype=np.float64)[:nland],
        "poor_soils": np.asarray([0.1, 0.2], dtype=np.float64)[:nland],
    }
    slow = {
        "growth_day": np.asarray([12.0, 20.0], dtype=np.float64)[:nland],
        "peatC_ok": np.zeros(nland, dtype=np.float64),
        "GSL": np.zeros(nland, dtype=np.float64),
        "summerp_long": np.zeros(nland, dtype=np.float64),
        "summerpet_long": np.zeros(nland, dtype=np.float64),
    }
    return vegetation, soil, slow


def _stomate_result(carbon_state=None):
    return SimpleNamespace(
        lpj=None,
        carbon_state={} if carbon_state is None else carbon_state,
        output=SimpleNamespace(history={"npp": np.asarray([1.0, 2.0])}),
    )


def _call(*, nland=2, **overrides):
    vegetation, soil, slow = _states(nland)
    kwargs = {
        "month": 6,
        "day": 15,
        "sec": 1800.0,
        "dt_sechiba": 1800.0,
        "dt_stomate": 86400.0,
        "date": 100,
        "vegetation_state": vegetation,
        "soil_state": soil,
        "salinity": np.asarray([32.0, 18.0], dtype=np.float64)[:nland],
        "tide_height": np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)[:nland],
        "slow_state": slow,
        "pref_soil_veg": np.ones(14, dtype=np.int32),
        "ext_coeff_vegetfrac": np.asarray([0.0] + [0.5] * 13, dtype=np.float64),
        "nstm": 1,
        "qsintcst": 0.1,
        "stomate_result": _stomate_result(),
    }
    kwargs.update(overrides)
    return slowproc_main_pft14_step(**kwargs)


def test_calendar_branches_follow_source_order_and_fixed_dynpeat_pc_write():
    first_year = _call(month=1, day=1, sec=1800.0)
    assert first_year.calendar.first_ts_year
    assert not first_year.calendar.do_slow
    np.testing.assert_array_equal(np.asarray(first_year.slow_state["growth_day"]), [0.0, 0.0])

    jan_second = _call(month=1, day=2, sec=1800.0)
    np.testing.assert_array_equal(np.asarray(jan_second.slow_state["peatC_ok"]), [1.0, 1.0])
    assert not jan_second.calendar.update_peatfrac


def test_day_end_multiland_surface_soil_and_static_state_writeback():
    marker = np.arange(4.0).reshape(2, 2)
    result = _call(sec=0.0, stomate_result=_stomate_result({"carbon_32l": marker}))

    assert result.calendar.do_slow
    assert result.calendar.last_ts_day
    assert result.calendar.date == 101
    np.testing.assert_array_equal(np.asarray(result.calendar.growth_day), [13.0, 20.0])
    assert result.vegetation_state["veget"].shape == (2, 14)
    assert result.vegetation_state["soiltile"].shape == (2, 1)
    assert result.vegetation_state["tot_bare_soil"].shape == (2,)
    np.testing.assert_array_equal(np.asarray(result.vegetation_state["qsintmax"][:, 0]), [0.0, 0.0])
    np.testing.assert_array_equal(np.asarray(result.soil_state["carbon_32l"]), marker)
    np.testing.assert_allclose(np.asarray(result.soil_state["bulkdens"]), [1400.0, 1650.0])
    np.testing.assert_allclose(
        np.asarray(result.soil_state["textfrac"]),
        [[0.2, 0.3, 0.5], [0.3, 0.1, 0.6]],
    )
    np.testing.assert_array_equal(np.asarray(result.salinity), [32.0, 18.0])
    np.testing.assert_array_equal(np.asarray(result.tide_height), [[0.1, -0.2], [0.3, 0.4]])


def test_exact_stomate_owner_receives_source_owned_calendar_and_boundaries(monkeypatch):
    captured = {}

    def fake_stomate_main_pft14_step(**kwargs):
        captured.update(kwargs)
        return _stomate_result()

    monkeypatch.setattr(slowproc_module, "stomate_main_pft14_step", fake_stomate_main_pft14_step)
    vegetation, soil, slow = _states()
    result = _call(
        sec=0.0,
        vegetation_state=vegetation,
        soil_state=soil,
        slow_state=slow,
        stomate_result=None,
        stomate_step_kwargs={"entry_payload": {"gpp": np.zeros((2, 14))}},
    )

    assert result.stomate is not None
    assert captured["do_slow"] is True
    assert captured["entry_payload"]["date"] == 101
    np.testing.assert_array_equal(captured["entry_payload"]["totfrac_nobio_new"], [0.0, 0.0])


@pytest.mark.parametrize(
    "switches",
    [
        SlowprocMainPft14Switches(dyn_peat=True),
        SlowprocMainPft14Switches(dynpeat_pc=True),
        SlowprocMainPft14Switches(veget_update=1),
        SlowprocMainPft14Switches(rotation_update=1),
        SlowprocMainPft14Switches(ok_stomate=False),
    ],
)
def test_target_external_structural_switches_are_explicitly_rejected(switches):
    with pytest.raises(ValueError, match="unsupported slowproc_main structural switch"):
        _call(switches=switches)


def test_landpoint_shapes_are_checked_without_single_point_assumptions():
    with pytest.raises(ValueError, match="tide_height"):
        _call(tide_height=np.zeros((1, 2), dtype=np.float64))
