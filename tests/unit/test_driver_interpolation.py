from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.driver.interpolation import (
    interpweight_masking_input1d,
    interpweight_masking_input2d,
    interpweight_masking_input3d,
    interpweight_masking_input4d,
    interpweight_modifying_input1d,
    interpweight_modifying_input2d,
    interpweight_modifying_input3d,
    interpweight_modifying_input4d,
    interpweight_provide_fractions1d,
    interpweight_provide_fractions2d,
    interpweight_provide_fractions4d,
    interpweight_calc_resolution_in,
    interpweight_valvecr,
    interpweight_provide_interpolation2d,
)


@pytest.mark.parametrize(
    ("owner", "shape"),
    [
        (interpweight_modifying_input1d, (4,)),
        (interpweight_modifying_input2d, (2, 2)),
        (interpweight_modifying_input3d, (1, 2, 2)),
        (interpweight_modifying_input4d, (1, 1, 2, 2)),
    ],
)
def test_modifying_input_clamps_only_values_strictly_below_zero_value(owner, shape):
    source = np.array([-2.0, -1.0, 0.0, 3.0]).reshape(shape)
    result = owner(source, -1.0)
    np.testing.assert_array_equal(result, np.array([-1.0, -1.0, 0.0, 3.0]).reshape(shape))
    np.testing.assert_array_equal(source, np.array([-2.0, -1.0, 0.0, 3.0]).reshape(shape))


def test_masking_input1d_preserves_inout_mask_and_rejects_source_fatal_branch():
    values = np.array([-1.0, 0.0, 2.0])
    initial = np.array([7, 8, 9])
    _, below = interpweight_masking_input1d(values, initial, "mbelow", [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(below, [1, 8, 9])
    _, above = interpweight_masking_input1d(values, initial, "mabove", [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(above, [7, 8, 1])
    with pytest.raises(RuntimeError, match="no sens"):
        interpweight_masking_input1d(values, initial, "msumrange", [1.0, 0.0, 2.0])


def test_masking_input2d_preserves_source_ordered_repeated_row_normalization():
    values = np.array([[2.0, 2.0]])
    normalized, mask = interpweight_masking_input2d(
        values,
        np.zeros((1, 2), dtype=np.int32),
        "msumrange",
        [3.0, 1.0, 5.0],
    )
    # j=1 divides 2 by 4; j=2 then sees a new row sum of 2.5 and is not divided.
    np.testing.assert_allclose(normalized, [[0.5, 2.0]])
    np.testing.assert_array_equal(mask, [[1, 1]])


def test_masking_input3d_uses_any_for_threshold_masks_and_sum_for_normalization():
    values = np.array([[[0.0, 3.0], [1.0, 1.0]]])
    initial = np.zeros((1, 2), dtype=np.int32)
    _, above = interpweight_masking_input3d(values, initial, "mabove", [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(above, [[1, 0]])
    normalized, mask = interpweight_masking_input3d(
        values,
        initial,
        "msumrange",
        [2.5, 1.0, 4.0],
    )
    np.testing.assert_allclose(normalized[0, 0], [0.0, 1.0])
    np.testing.assert_allclose(normalized[0, 1], [1.0, 1.0])
    np.testing.assert_array_equal(mask, [[1, 1]])


def test_masking_input4d_preserves_shared_mask_gate_and_per_layer_normalization():
    values = np.array([[[[1.0, 2.0], [1.0, 2.0]]]])
    normalized, mask = interpweight_masking_input4d(
        values,
        np.zeros((1, 1), dtype=np.int32),
        "msumrange",
        [3.0, 1.0, 5.0],
    )
    np.testing.assert_allclose(normalized[0, 0, :, 0], [1.0, 1.0])
    np.testing.assert_allclose(normalized[0, 0, :, 1], [0.5, 0.5])
    np.testing.assert_array_equal(mask, [[1]])


def test_masking_input_rejects_unknown_mode_and_wrong_shapes():
    with pytest.raises(ValueError, match="not ready"):
        interpweight_masking_input2d(np.ones((2, 2)), np.zeros((2, 2)), "var", [0, 0, 0])
    with pytest.raises(ValueError, match="msk must have shape"):
        interpweight_masking_input3d(np.ones((2, 2, 2)), np.zeros((2, 1)), "nomask", [0, 0, 0])


def test_provide_fractions1d_uses_first_counted_overlap_slots_and_normalizes_area():
    result = interpweight_provide_fractions1d(
        np.array([1.0, 2.0, 1.0]),
        np.array([[0.25, 0.75, 0.0]]),
        np.array([[1, 2, 3]]),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(result.fractions, [[0.25, 0.75]])
    np.testing.assert_allclose(result.availability, [1.0])


def test_provide_fractions2d_uses_fortran_indices_and_no_overlap_fallback():
    result = interpweight_provide_fractions2d(
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[0.4, 0.6], [0.0, 0.0]]),
        np.array([[[1, 1], [2, 1]], [[1, 1], [1, 1]]]),
        np.array([1.0, 2.0]),
        vmin=1.0,
        vmax=3.0,
    )
    np.testing.assert_allclose(result.fractions[0], [0.4, 0.6])
    np.testing.assert_allclose(result.fractions[1], [0.0, 1.0])
    np.testing.assert_allclose(result.availability, [1.0, -1.0])


def test_provide_fractions_rejects_all_zero_source_and_unknown_interpolation():
    with pytest.raises(ValueError, match="wrong 1D"):
        interpweight_provide_fractions1d([0.0], [[1.0]], [[1]], [1.0])
    with pytest.raises(ValueError, match="not ready"):
        interpweight_provide_fractions2d([[1.0]], [[1.0]], [[[1, 1]]], [1.0], tint="linear")


def test_provide_fractions4d_filters_invalid_values_and_preserves_source_fallback_loop():
    values = np.array([[[[2.0, 25.0], [4.0, -1.0]]]])
    weighted = interpweight_provide_fractions4d(
        values,
        np.array([[0.5]]),
        np.array([[[1, 1]]]),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(weighted.fractions[0], [[2.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(weighted.availability, [0.5])

    fallback = interpweight_provide_fractions4d(
        values,
        np.array([[0.0]]),
        np.array([[[1, 1]]]),
        np.array([1.0, 2.0]),
        vmin=np.array([1.0, 3.0]),
        vmax=np.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(fallback.fractions[0], [[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(fallback.availability, [-1.0])


def test_calc_resolution_in_preserves_edge_and_center_difference_formulas():
    lon_axis = np.array([0.0, 2.0, 6.0])
    lat_axis = np.array([10.0, 13.0, 19.0])
    lon = np.repeat(lon_axis[:, None], 3, axis=1)
    lat = np.repeat(lat_axis[None, :], 3, axis=0)
    result = interpweight_calc_resolution_in(lon, lat, r_earth=1.0)
    np.testing.assert_allclose(result[0, 1, 0], 2.0 * np.pi / 180.0 * np.cos(np.deg2rad(13.0)))
    np.testing.assert_allclose(result[1, 1, 0], 3.0 * np.pi / 180.0 * np.cos(np.deg2rad(13.0)))
    np.testing.assert_allclose(result[1, 0, 1], 3.0 * np.pi / 180.0)
    np.testing.assert_allclose(result[1, 1, 1], 4.5 * np.pi / 180.0)


@pytest.mark.parametrize(
    ("oper", "count", "positions"),
    [("eq", 2, [2, 3]), ("ge", 3, [2, 3, 4]), ("le", 3, [1, 2, 3]), ("neq", 2, [1, 4])],
)
def test_valvecr_returns_fortran_count_and_one_based_stored_positions(oper, count, positions):
    result = interpweight_valvecr(np.array([0.0, 1.0, 1.0, 2.0]), 1.0, oper)
    assert result.count == count
    np.testing.assert_array_equal(result.stored_positions, positions)


def test_valvecr_preserves_full_match_last_position_omission():
    result = interpweight_valvecr(np.ones(3), 1.0, "eq")
    assert result.count == 3
    np.testing.assert_array_equal(result.stored_positions, [1, 2])
    with pytest.raises(ValueError, match="supports only"):
        interpweight_valvecr(np.ones(3), 1.0, "gt")


def test_provide_interpolation2d_default_stops_at_first_nonpositive_area():
    result = interpweight_provide_interpolation2d(
        np.array([[2.0, 8.0], [4.0, 16.0]]),
        np.array([[0.25, 0.75, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[[1, 1], [2, 1], [2, 2]], [[1, 1], [2, 2], [1, 1]]]),
        defaultval=9.0,
    )
    np.testing.assert_allclose(result.fractions, [3.5, 9.0])
    np.testing.assert_allclose(result.availability, [1.0, -1.0])


def test_provide_interpolation2d_slopecalc_caps_each_source_before_weighting():
    result = interpweight_provide_interpolation2d(
        np.array([[1.0], [4.0]]),
        np.array([[0.25, 0.75]]),
        np.array([[[1, 1], [2, 1]]]),
        tint="slopecalc",
        default_no_value=2.0,
    )
    np.testing.assert_allclose(result.fractions, [0.125])
    np.testing.assert_allclose(result.availability, [1.0])
