from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import jax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import hydrol_tmc_update  # noqa: E402


DZ = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)


def _column(mc):
    return (
        DZ[1] * (3.0 * mc[0] + mc[1]) / 8.0
        + DZ[1] * (3.0 * mc[1] + mc[0]) / 8.0
        + DZ[2] * (3.0 * mc[1] + mc[2]) / 8.0
        + DZ[2] * (3.0 * mc[2] + mc[1]) / 8.0
    )


def _base(**updates):
    mc = np.asarray([[[0.2, 0.4], [0.3, 0.5], [0.4, 0.6]]], dtype=np.float64)
    water = np.asarray([[1.0, 2.0]], dtype=np.float64)
    values = {
        "veget_max": np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64),
        "soiltile": np.asarray([[0.6, 0.4]], dtype=np.float64),
        "qsintveg": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        "mc": mc,
        "water2infilt": water,
        "tmc": np.asarray([[_column(mc[0, :, 0]) + 1.0, _column(mc[0, :, 1]) + 2.0]]),
        "resdist": np.asarray([[0.6, 0.4]], dtype=np.float64),
        "vegtot_old": np.asarray([1.0], dtype=np.float64),
        "vegtot": np.asarray([1.0], dtype=np.float64),
        "pref_soil_veg": np.asarray([1, 1, 2], dtype=np.int32),
        "dz_mm": DZ,
    }
    values.update(updates)
    return values


def test_no_fraction_change_preserves_state_and_refreshes_outputs():
    inputs = _base()
    result = hydrol_tmc_update(**inputs, check_cwrr=True, allowed_err=1.0e-12)

    np.testing.assert_array_equal(np.asarray(result.mc), inputs["mc"])
    np.testing.assert_array_equal(np.asarray(result.water2infilt), inputs["water2infilt"])
    np.testing.assert_array_equal(np.asarray(result.qsintveg), inputs["qsintveg"])
    np.testing.assert_allclose(np.asarray(result.tmc), inputs["tmc"])
    np.testing.assert_allclose(np.asarray(result.humtot), [np.sum(inputs["tmc"] * inputs["soiltile"])])
    np.testing.assert_array_equal(np.asarray(result.resdist), inputs["soiltile"])
    assert not bool(result.vegtot_updated)
    assert not bool(result.soil_updated)
    assert not np.any(np.asarray(result.water_balance_failed))


def test_disappeared_pft_canopy_water_returns_to_preferred_soil_tile():
    inputs = _base(
        veget_max=np.asarray([[0.2, 0.3, 0.0]]),
        qsintveg=np.asarray([[0.1, -0.2, 0.8]]),
    )
    result = hydrol_tmc_update(**inputs)

    expected = inputs["water2infilt"].copy()
    expected[0, 1] += 0.8 / (inputs["resdist"][0, 1] * inputs["vegtot_old"][0])
    np.testing.assert_allclose(np.asarray(result.water2infilt), expected)
    np.testing.assert_array_equal(np.asarray(result.qsintveg), [[0.1, -0.2, 0.0]])


def test_vegtot_increase_scales_each_tile_mc_and_surface_water():
    inputs = _base(vegtot_old=np.asarray([0.5]), vegtot=np.asarray([0.8]))
    result = hydrol_tmc_update(**inputs)

    scale = 0.5 / 0.8
    np.testing.assert_allclose(np.asarray(result.mc), inputs["mc"] * scale)
    np.testing.assert_allclose(np.asarray(result.water2infilt), inputs["water2infilt"] * scale)
    np.testing.assert_array_equal(np.asarray(result.drain_upd), [0.0])
    np.testing.assert_array_equal(np.asarray(result.runoff_upd), [0.0])


@pytest.mark.parametrize("new_vegtot", [0.6, 0.0])
def test_vegtot_decrease_exports_removed_soil_and_surface_water(new_vegtot):
    inputs = _base(vegtot_old=np.asarray([1.0]), vegtot=np.asarray([new_vegtot]))
    result = hydrol_tmc_update(**inputs)

    if new_vegtot > 1.0e-8:
        scale = (1.0 - new_vegtot) / new_vegtot
        expected_drain = sum(_column(inputs["mc"][0, :, jst] * scale) for jst in range(2))
        expected_runoff = np.sum(inputs["water2infilt"] * scale)
    else:
        expected_drain = sum(_column(inputs["mc"][0, :, jst]) for jst in range(2))
        expected_runoff = np.sum(inputs["water2infilt"])
    np.testing.assert_allclose(np.asarray(result.drain_upd), [expected_drain])
    np.testing.assert_allclose(np.asarray(result.runoff_upd), [expected_runoff])
    np.testing.assert_array_equal(np.asarray(result.mc), inputs["mc"])


def test_soiltile_loss_is_mixed_into_growing_tile_and_deleted_tile_is_zeroed():
    inputs = _base(soiltile=np.asarray([[1.0, 0.0]], dtype=np.float64))
    result = hydrol_tmc_update(**inputs)

    expected_mc = np.zeros_like(inputs["mc"])
    expected_mc[0, :, 0] = inputs["mc"][0, :, 0] * 0.6 + inputs["mc"][0, :, 1] * 0.4
    expected_water = np.asarray([[1.0 * 0.6 + 2.0 * 0.4, 0.0]])
    np.testing.assert_allclose(np.asarray(result.mc), expected_mc)
    np.testing.assert_allclose(np.asarray(result.water2infilt), expected_water)
    assert bool(result.soil_updated)


def test_multiple_shrinking_tiles_are_area_weighted_before_growing_tile_update():
    mc = np.asarray([[[0.1, 0.4, 0.9], [0.2, 0.5, 1.0], [0.3, 0.6, 1.1]]])
    inputs = _base(
        veget_max=np.asarray([[0.2, 0.3, 0.5]]),
        qsintveg=np.asarray([[0.0, 0.0, 0.0]]),
        mc=mc,
        water2infilt=np.asarray([[1.0, 4.0, 9.0]]),
        tmc=np.asarray([[_column(mc[0, :, j]) + [1.0, 4.0, 9.0][j] for j in range(3)]]),
        resdist=np.asarray([[0.5, 0.3, 0.2]]),
        soiltile=np.asarray([[0.2, 0.2, 0.6]]),
        pref_soil_veg=np.asarray([1, 2, 3]),
    )
    result = hydrol_tmc_update(**inputs)

    mixed_loss_mc = mc[0, :, 0] * 0.75 + mc[0, :, 1] * 0.25
    expected_growing = (mc[0, :, 2] * 0.2 + mixed_loss_mc * 0.4) / 0.6
    expected_infil = (9.0 * 0.2 + (1.0 * 0.75 + 4.0 * 0.25) * 0.4) / 0.6
    np.testing.assert_allclose(np.asarray(result.mc)[0, :, 2], expected_growing)
    np.testing.assert_allclose(np.asarray(result.water2infilt)[0, 2], expected_infil)


def test_domain_wide_update_flags_preserve_unchanged_landpoint():
    one = _base()
    values = {}
    for name, value in one.items():
        values[name] = value if name in {"pref_soil_veg", "dz_mm"} else np.concatenate((value, value), axis=0)
    values["vegtot"][0] = 0.8
    values["soiltile"][0] = [0.7, 0.3]
    result = hydrol_tmc_update(**values)

    np.testing.assert_array_equal(np.asarray(result.mc)[1], values["mc"][1])
    np.testing.assert_array_equal(np.asarray(result.water2infilt)[1], values["water2infilt"][1])
    assert bool(result.vegtot_updated)
    assert bool(result.soil_updated)


def test_jitted_transition_accepts_dynamic_water_state():
    inputs = _base(vegtot_old=np.asarray([0.5]), vegtot=np.asarray([0.8]))
    static = {name: value for name, value in inputs.items() if name not in {"mc", "water2infilt"}}

    transition = jax.jit(
        lambda mc, water: hydrol_tmc_update(
            **static,
            mc=mc,
            water2infilt=water,
        )
    )
    result = transition(inputs["mc"], inputs["water2infilt"])

    np.testing.assert_allclose(np.asarray(result.mc), inputs["mc"] * (0.5 / 0.8))
    np.testing.assert_allclose(np.asarray(result.water2infilt), inputs["water2infilt"] * (0.5 / 0.8))


def test_water_balance_fatal_branch_and_shape_validation_are_explicit():
    inputs = _base()
    with pytest.raises(RuntimeError, match="water-balance"):
        hydrol_tmc_update(**inputs, check_cwrr=True, allowed_err=-1.0)
    diagnostic = hydrol_tmc_update(
        **inputs,
        check_cwrr=True,
        allowed_err=-1.0,
        raise_on_water_balance_error=False,
    )
    assert np.all(np.asarray(diagnostic.water_balance_failed))

    with pytest.raises(ValueError, match="pref_soil_veg"):
        hydrol_tmc_update(**{**inputs, "pref_soil_veg": np.asarray([1, 3, 2])})
