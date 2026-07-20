from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver.orchestration import (
    _paper_day_season_fields_from_bundle_source,
    paper_day_active_restart_producer_fields,
)
from jax_orchidee.stomate.season import season_surface_temperature_memory_step


def _mapping_state(**values):
    return SimpleNamespace(_asdict=lambda: dict(values))


def _source(*, t2m_daily):
    empty = SimpleNamespace(step=SimpleNamespace(state=_mapping_state()))
    return SimpleNamespace(
        kwargs={"t2m": np.asarray(t2m_daily)},
        pre_step=SimpleNamespace(memory=empty, annual=empty, biometeorology=empty),
    )


def _ok_leak(*, resp_hetero_soil, deepc_peat):
    perma = (
        None
        if deepc_peat is None
        else SimpleNamespace(deepc_peat=np.asarray(deepc_peat))
    )
    soilcarbon = SimpleNamespace(
        resp_hetero_soil=np.asarray(resp_hetero_soil),
        perma_peat=perma,
    )
    return SimpleNamespace(ok_leak=SimpleNamespace(soilcarbon=soilcarbon))


def test_season_surface_temperature_memories_follow_fortran_firstcall_and_relaxation():
    result = season_surface_temperature_memory_step(
        tsurf_year=np.zeros(2),
        t2m_14=np.zeros(2),
        tsurf_daily=np.array([301.0, 299.0]),
        t2m_daily=np.array([300.0, 286.0]),
        dt_days=1.0,
        firstcall=True,
    )
    np.testing.assert_array_equal(result.tsurf_year, np.array([301.0, 299.0]))
    np.testing.assert_array_equal(result.t2m_14, np.array([300.0, 286.0]))

    next_day = season_surface_temperature_memory_step(
        tsurf_year=result.tsurf_year,
        t2m_14=result.t2m_14,
        tsurf_daily=np.array([250.0, 250.0]),
        t2m_daily=np.array([286.0, 300.0]),
        dt_days=1.0,
        firstcall=False,
    )
    np.testing.assert_array_equal(next_day.tsurf_year, result.tsurf_year)
    np.testing.assert_allclose(
        next_day.t2m_14,
        (np.array([300.0, 286.0]) * 13.0 + np.array([286.0, 300.0])) / 14.0,
        rtol=0.0,
        atol=0.0,
    )


def test_orchestration_season_packet_uses_season_owner_result():
    season = SimpleNamespace(
        t2m_14=np.array([280.0, 290.0]),
        tsurf_year=np.array([281.0, 291.0]),
        gdd_init_date=np.zeros((2, 2)),
    )
    fields = _paper_day_season_fields_from_bundle_source(
        _source(t2m_daily=np.array([294.0, 276.0])),
        season_state=season,
        dt_days=1.0,
        tsurf_daily=np.array([295.0, 277.0]),
        firstcall=False,
    )
    assert fields is not None
    np.testing.assert_array_equal(fields["tsurf_year"], season.tsurf_year)
    np.testing.assert_allclose(
        fields["t2m_14"],
        (season.t2m_14 * 13.0 + np.array([294.0, 276.0])) / 14.0,
        rtol=0.0,
        atol=0.0,
    )


def test_active_restart_fields_use_current_dt_reset_respiration_and_peat_kernel_output():
    deepc = np.arange(24.0).reshape(2, 3, 4)
    depth = np.arange(8.0).reshape(2, 4)
    fields = paper_day_active_restart_producer_fields(
        ok_leak=_ok_leak(resp_hetero_soil=np.full((2, 4), 7.5), deepc_peat=deepc),
        previous_slowproc={
            "depth_deepsoil": depth,
            "deepC_peat": np.full_like(deepc, -1.0),
        },
        dt_days=2.0,
    )
    assert fields["dt_days_read"] == 2.0
    np.testing.assert_array_equal(fields["resp_hetero"], np.zeros((2, 4)))
    np.testing.assert_array_equal(fields["deepC_peat"], deepc)
    np.testing.assert_array_equal(fields["depth_deepsoil"], depth)


def test_inactive_perma_peat_has_explicit_carry_guard_and_depth_requires_init_owner():
    prior_deepc = np.ones((2, 3, 4))
    depth = np.ones((2, 4))
    fields = paper_day_active_restart_producer_fields(
        ok_leak=_ok_leak(resp_hetero_soil=np.ones((2, 4)), deepc_peat=None),
        previous_slowproc={"depth_deepsoil": depth, "deepC_peat": prior_deepc},
        dt_days=1.0,
    )
    np.testing.assert_array_equal(fields["deepC_peat"], prior_deepc)

    with pytest.raises(ValueError, match="deepC_peat carry state"):
        paper_day_active_restart_producer_fields(
            ok_leak=_ok_leak(resp_hetero_soil=np.ones((2, 4)), deepc_peat=None),
            previous_slowproc={"depth_deepsoil": depth},
            dt_days=1.0,
        )
    with pytest.raises(ValueError, match="depth_deepsoil carry state"):
        paper_day_active_restart_producer_fields(
            ok_leak=_ok_leak(resp_hetero_soil=np.ones((2, 4)), deepc_peat=prior_deepc),
            previous_slowproc={},
            dt_days=1.0,
        )
