from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.driver.forcing_interpolation import (
    ForcingInterpolationRecord,
    ForcingInterpolationState,
    forcing_grid_metadata,
    forcing_read_interpol_transition,
)


def _records(count: int = 4) -> tuple[ForcingInterpolationRecord, ...]:
    shape = (2, 2)
    result = []
    for index in range(1, count + 1):
        def constant(value):
            return np.full(shape, value, dtype=np.float64)
        result.append(
            ForcingInterpolationRecord.from_mapping(
                {
                    "tair": constant(265.0 + index),
                    "tmax": constant(280.0 + index),
                    "pb": constant(90000.0 + 1000.0 * index),
                    "qair": constant(0.002 + 0.001 * index),
                    "u": constant(index),
                    "v": constant(-index),
                    "rainf": constant(1.0e-5 * index),
                    "snowf": constant(1.0e-6 * index),
                    "swdown": constant(100.0 * index),
                    "lwdown": constant(200.0 + 10.0 * index),
                    "zlev": constant(2.0 + index),
                    "zlevuv": constant(10.0 + index),
                    "swnet": constant(50.0 * index),
                    "eair": constant(1000.0 * index),
                }
            )
        )
    return tuple(result)


def _transition(records, state, **kwargs):
    return forcing_read_interpol_transition(
        records,
        state,
        split=kwargs.pop("split", 4),
        nb_spread=kwargs.pop("nb_spread", 2),
        dt_force=kwargs.pop("dt_force", 21600.0),
        lon=np.asarray([[-30.0, 30.0], [0.0, 60.0]]),
        lat=np.asarray([[-20.0, 30.0], [10.0, 60.0]]),
        daily_interpol=kwargs.pop("daily_interpol", False),
        is_watchout=kwargs.pop("is_watchout", False),
        **kwargs,
    )


def test_uninitialized_transition_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="initialized first"):
        _transition(_records(), ForcingInterpolationState(), itauin=1, itau_split=1)


def test_non_daily_save_state_advances_and_wraps() -> None:
    records = _records()
    state = _transition(records, ForcingInterpolationState(), itauin=0, itau_split=0).state
    first = _transition(records, state, itauin=1, itau_split=1)
    assert first.state.itau_read_nm1 == 4
    assert first.state.itau_read_n == 1
    assert np.allclose(first.values.pb, 93250.0)
    wrapped = _transition(records, first.state, itauin=5, itau_split=1)
    assert wrapped.state.itau_read_n == 1


def test_daily_save_state_advances_at_half_day_phase() -> None:
    records = _records()
    state = _transition(
        records, ForcingInterpolationState(), itauin=0, itau_split=0,
        split=24, nb_spread=6, dt_force=86400.0, daily_interpol=True,
    ).state
    for step in range(1, 13):
        result = _transition(
            records, state, itauin=1, itau_split=step,
            split=24, nb_spread=6, dt_force=86400.0, daily_interpol=True,
        )
        state = result.state
        assert state.itau_read_n == 1
    result = _transition(
        records, state, itauin=1, itau_split=13,
        split=24, nb_spread=6, dt_force=86400.0, daily_interpol=True,
    )
    assert result.state.itau_read_nm1 == 1
    assert result.state.itau_read_n == 2


def test_grid_metadata_covers_standard_default_and_watchout_mask() -> None:
    tair = np.full((3, 2), 280.0)
    standard = forcing_grid_metadata(tair, contfrac=None, watchout=False)
    assert standard.nbindex == 6
    fraction = np.asarray([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    neighbours = np.arange(3 * 2 * 8).reshape(3, 2, 8)
    resolution = np.ones((3, 2, 2))
    watchout = forcing_grid_metadata(
        tair, contfrac=fraction, watchout=True, neighbours=neighbours,
        resolution=resolution, ii_begin=2, ii_end=2,
    )
    assert np.array_equal(watchout.kindex, np.asarray([3, 4, 5], dtype=np.int32))
    assert np.array_equal(watchout.neighbours, neighbours)


def test_watchout_requires_grid_contract() -> None:
    tair = np.full((2, 2), 280.0)
    with pytest.raises(ValueError, match="requires contfrac"):
        forcing_grid_metadata(tair, contfrac=None, watchout=True)
    with pytest.raises(ValueError, match="requires neighbours"):
        forcing_grid_metadata(tair, contfrac=np.ones((2, 2)), watchout=True)
