from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.driver.source_calendar import source_calendar_assignments
import jax_orchidee.sechiba.slowproc as slowproc_module
from tests.unit.test_slowproc_main import _call


def _apply(**overrides):
    values = {
        "growth_day": jnp.asarray([3.0, 8.0, 1.0]),
        "peatC_ok": jnp.asarray([0.0, 0.0, 1.0]),
        "temp_growth": jnp.asarray([5.1, 5.0, 4.9]),
        "peatC": jnp.asarray([50.0, 49.9, 51.0]),
        "month": 6,
        "day": 10,
        "sec": 1800,
        "dt_sechiba": 1800,
    }
    values.update(overrides)
    return source_calendar_assignments(**values)


def test_jan1_midnight_is_previous_year_end_and_next_active_step_resets():
    midnight = _apply(month=1, day=1, sec=0)
    assert bool(midnight.last_ts_day)
    assert bool(midnight.end_of_year)
    assert not bool(midnight.first_ts_year)
    np.testing.assert_array_equal(midnight.growth_day, [4.0, 8.0, 1.0])

    first_active = _apply(month=1, day=1, sec=1800)
    assert bool(first_active.first_ts_year)
    assert not bool(first_active.last_ts_day)
    np.testing.assert_array_equal(first_active.growth_day, [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("dynpeat_pc", "judge_pc", "expected"),
    [
        (False, False, [1.0, 1.0, 1.0]),
        (False, True, [1.0, 1.0, 1.0]),
        (True, False, [0.0, 0.0, 1.0]),
        (True, True, [1.0, 0.0, 1.0]),
    ],
)
def test_jan2_peat_gate_branches_and_pc_lim_equality(dynpeat_pc, judge_pc, expected):
    result = _apply(month=1, day=2, sec=1800, dynpeat_pc=dynpeat_pc, judge_pc=judge_pc)
    np.testing.assert_array_equal(result.peatC_ok, expected)


def test_midnight_growth_uses_strict_five_degree_threshold():
    result = _apply(sec=0)
    np.testing.assert_array_equal(result.growth_day, [4.0, 8.0, 1.0])


def test_inactive_initialization_leaves_memories_and_flags_unchanged():
    result = _apply(month=1, day=1, sec=1800, active=False)
    np.testing.assert_array_equal(result.growth_day, [3.0, 8.0, 1.0])
    np.testing.assert_array_equal(result.peatC_ok, [0.0, 0.0, 1.0])
    assert not bool(result.first_ts_year)
    assert not bool(result.last_ts_day)
    assert not bool(result.end_of_year)


def test_dynamic_calendar_and_configuration_values_trace_under_jit():
    compiled = jax.jit(
        lambda month, day, sec, active, dynpeat_pc, judge_pc: source_calendar_assignments(
            growth_day=jnp.asarray([2.0, 4.0]),
            peatC_ok=jnp.asarray([0.0, 0.0]),
            temp_growth=jnp.asarray([6.0, 5.0]),
            peatC=jnp.asarray([50.0, 49.0]),
            month=month,
            day=day,
            sec=sec,
            dt_sechiba=1800,
            active=active,
            dynpeat_pc=dynpeat_pc,
            judge_pc=judge_pc,
            pc_lim=50.0,
        )
    )
    jan2 = compiled(1, 2, 1800, True, True, True)
    np.testing.assert_array_equal(jan2.peatC_ok, [1.0, 0.0])
    midnight = compiled(1, 1, 0, True, False, False)
    assert bool(midnight.end_of_year)
    np.testing.assert_array_equal(midnight.growth_day, [3.0, 4.0])


@pytest.mark.parametrize("bad_shape", [(3, 1), (2,)])
def test_landpoint_vector_shape_guards(bad_shape):
    with pytest.raises(ValueError):
        _apply(peatC=np.zeros(bad_shape))


def test_slowproc_merge_keeps_pre_stomate_calendar_assignments(monkeypatch):
    calls = []

    def no_lcc_spy(*, veget_max, **kwargs):
        calls.append("no_lcc")
        nland, npft = veget_max.shape
        return SimpleNamespace(
            veget_max_new=veget_max,
            vegetnew_firstday=jnp.zeros((nland, npft)).at[:, 0].set(1.0),
            totfrac_nobio_new=jnp.zeros((nland,)),
            glccNetLCC=jnp.zeros((nland, 12)),
            glccSecondShift=jnp.zeros((nland, 12)),
            glccPrimaryShift=jnp.zeros((nland, 12)),
            harvest_matrix=jnp.zeros((nland, 12)),
            bound_spa=jnp.zeros((nland, npft)),
        )

    def nobio_spy(*, frac_nobio_lastyear, **kwargs):
        calls.append("nobio")
        nland = frac_nobio_lastyear.shape[0]
        return SimpleNamespace(
            totfrac_nobio_lastyear=jnp.zeros((nland,)),
            totfrac_nobio_new=jnp.zeros((nland,)),
        )

    def fire_spy(*, kjpindex, **kwargs):
        calls.append("fire")
        zeros = jnp.zeros((kjpindex,))
        return SimpleNamespace(
            lightn=zeros,
            observed_ba=zeros,
            cf_coarse=zeros,
            cf_fine=zeros,
            ratio=zeros,
            ratio_flag=zeros,
            popd=zeros,
            read_observed_ba=False,
            humign=zeros,
            read_cf_fine=False,
            read_cf_coarse=False,
            read_ratio_flag=False,
            read_ratio=False,
        )

    def surface_spy(**kwargs):
        calls.append("surface")
        vegetation = SimpleNamespace(
            frac_nobio=kwargs["frac_nobio"],
            veget_max=kwargs["veget_max"],
            veget=kwargs["veget"],
            totfrac_nobio=kwargs["totfrac_nobio"],
            soiltile=kwargs["soiltile"],
        )
        return SimpleNamespace(
            surface=SimpleNamespace(
                vegetation=vegetation,
                tot_bare_soil=jnp.zeros((kwargs["veget_max"].shape[0],)),
            )
        )

    monkeypatch.setattr(slowproc_module, "slowproc_no_lcc_entry_state", no_lcc_spy)
    monkeypatch.setattr(slowproc_module, "slowproc_stomate_nobio_boundary", nobio_spy)
    monkeypatch.setattr(slowproc_module, "slowproc_fire_disabled_entry_state", fire_spy)
    monkeypatch.setattr(slowproc_module, "slowproc_surface_transition_explicit", surface_spy)

    stomate_result = SimpleNamespace(
        lpj=None,
        carbon_state={},
        output=SimpleNamespace(history={}),
        daily_state={
            "growth_day": jnp.asarray([100.0, 100.0]),
            "peatC_ok": jnp.asarray([9.0, 9.0]),
        },
    )
    result = _call(sec=0.0, stomate_result=stomate_result)

    assert calls == ["no_lcc", "nobio", "fire", "surface"]
    np.testing.assert_array_equal(result.slow_state["growth_day"], [13.0, 20.0])
    np.testing.assert_array_equal(result.slow_state["peatC_ok"], [0.0, 0.0])
