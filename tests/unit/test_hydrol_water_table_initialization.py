"""Source tile-vector initialization, without any physical model transition."""

import jax
import numpy as np
import pytest

from jax_orchidee.parameters import soil
from jax_orchidee.sechiba.hydrol import (
    hydrol_cold_start_state,
    hydrol_force_water_table_saturation,
)


@pytest.fixture(autouse=True)
def _binary64():
    with jax.enable_x64():
        yield


def _cold(value):
    return hydrol_cold_start_state(
        veget_max=np.asarray([[0., 1.]]),
        soiltile=np.asarray([[0., 0., 0., 1., 0., 0.]]),
        pref_soil_veg=np.asarray([1, 4]), nslm=3,
        zwt_force_default=value,
    )


def test_cold_zwt_force_keeps_each_source_tile():
    # hydrol_init:2843-2850; r21setvar_p spreads the full nstm vector.
    depths = np.asarray([1e20, .25, .5, 1., 1.5, 1e20])
    state = _cold(depths)
    np.testing.assert_array_equal(state.zwt_force, depths[None, :])
    assert state.zforce is True


@pytest.mark.parametrize("depth,active", [(1e20, False), (.25, True)])
def test_cold_zwt_force_preserves_scalar_api(depth, active):
    state = _cold(depth)
    np.testing.assert_array_equal(state.zwt_force, np.full((1, 6), depth))
    assert state.zforce is active


def test_cold_zwt_force_exact_depth_gate():
    state = _cold([1e20, 1e20, 1e20, 1e20, 1e20, 2.])
    assert state.zforce is True
    above = _cold([1e20, 1e20, 1e20, 1e20, 1e20, np.nextafter(2., np.inf)])
    assert above.zforce is False


@pytest.mark.parametrize("mode", ["eager", "jit"])
def test_cold_zwt_force_reaches_real_saturation_consumer(mode):
    # The source's tile4 depth activates only nodes at/below0.5m; tile1 is
    # unforced. The old cold producer filled all tiles with tile1's sentinel.
    state = _cold([1e20, 1e20, 1e20, .5, 1e20, 1e20])
    kernel = hydrol_force_water_table_saturation
    if mode == "jit":
        kernel = jax.jit(kernel, static_argnames=("peat_hydro", "soil_tile_index"))
    common = dict(mc=np.full((1, 3), .3), dr_ns=np.zeros(1),
                  zz_mm=np.asarray([0., 500., 2000.]), dz_mm=np.asarray([0., 500., 1500.]),
                  zmaxh_m=2., mcs=np.asarray([.5]), peat_hydro=False)
    forced = kernel(**common, zwt_force=state.zwt_force[:, 3], soil_tile_index=4)
    unforced = kernel(**common, zwt_force=state.zwt_force[:, 0], soil_tile_index=1)
    np.testing.assert_array_equal(forced.mc, [[.3, .5, .5]])
    np.testing.assert_array_equal(unforced.mc, [[.3, .3, .3]])


@pytest.mark.parametrize("depths", [[.25, .5], [[.25] * 6], [np.nan] * 6])
def test_cold_zwt_force_rejects_invalid_vector(depths):
    with pytest.raises(ValueError, match="zwt_force_default"):
        _cold(depths)


def test_zwt_force_materialized_vector_resolver():
    depths = [1e20, .25, .5, 1., 1.5, 1e20]
    declarations = {f"ZWT_FORCE__{i:05d}": str(v) for i, v in enumerate(depths, 1)}
    np.testing.assert_array_equal(soil.resolve_hydrol_zwt_force(declarations, nstm=6), depths)
    np.testing.assert_array_equal(soil.resolve_hydrol_zwt_force({}, nstm=6), np.full(6, 1e20))
    np.testing.assert_array_equal(
        soil.resolve_hydrol_zwt_force({"ZWT_FORCE__00001": "1D20"}, nstm=6), np.full(6, 1e20),
    )
    np.testing.assert_array_equal(
        soil.resolve_hydrol_zwt_force({"ZWT_FORCE": "1D20 .25 .5 1 1.5 1D20"}, nstm=6), depths,
    )


@pytest.mark.parametrize("declarations", [
    {"ZWT_FORCE": ".25"},
    {"ZWT_FORCE__00001": ".25"},
    {"ZWT_FORCE__00007": ".25"},
    {"ZWT_FORCE": "0 0 0 0 0 0", "ZWT_FORCE__00001": ".25"},
])
def test_zwt_force_rejects_ambiguous_literal_contract(declarations):
    with pytest.raises(ValueError, match="ZWT_FORCE"):
        soil.resolve_hydrol_zwt_force(declarations, nstm=6)
