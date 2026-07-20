from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.static import (
    aggregate_2d_overlap,
    bbox_center_mean,
    hydrol_refsoc_1d_from_explicit_overlap,
    model_bbox_from_lalo_resolution,
    njsc_from_soilclass,
    read_paper_condveg_background_soilalbedo,
    read_paper_hydrol_refsoc_1d_static_field,
    read_paper_thermosoil_reftemp_static_field,
    read_paper_thermosoil_refsoc_static_field,
    regular_lonlat_grid_geometry,
    slowproc_nearest_indices,
    thermosoil_refsoc_from_explicit_overlap,
    usda_soil_fields_from_explicit_overlap,
    weighted_mean_from_explicit_overlap,
    zobler_soilclass_from_explicit_overlap,
)


def test_model_bbox_from_explicit_lalo_resolution_uses_orchidee_units():
    lalo = np.asarray([[0.0, 10.0]], dtype=np.float64)
    resolution = np.asarray([[111_317.09969219834, 111_317.09969219834]], dtype=np.float64)

    bbox = model_bbox_from_lalo_resolution(lalo, resolution)

    assert bbox.shape == (1, 4)
    assert np.allclose(bbox[0], [9.5, 10.5, -0.5, 0.5])


def test_bbox_center_mean_requires_explicit_geometry_and_uses_fortran_bounds():
    lalo = np.asarray([[0.0, 10.0]], dtype=np.float64)
    resolution = np.asarray([[111_317.09969219834, 111_317.09969219834]], dtype=np.float64)
    bbox = model_bbox_from_lalo_resolution(lalo, resolution)[0]
    source_lon = np.asarray([bbox[0], 9.75, 10.0, 10.25, bbox[1]], dtype=np.float64)
    source_lat = np.asarray([bbox[2], -0.25, 0.0, 0.25, bbox[3]], dtype=np.float64)
    values = np.arange(25, dtype=np.float64).reshape(5, 5)

    mean, counts = bbox_center_mean(lalo, resolution, source_lon, source_lat, values)

    included = values[:4, :4]
    assert np.array_equal(counts, [16])
    assert np.allclose(mean, [float(np.mean(included))])


def test_bbox_center_mean_preserves_trailing_time_axis():
    lalo = np.asarray([[0.0, 0.0]], dtype=np.float64)
    resolution = np.asarray([[111_317.09969219834, 111_317.09969219834]], dtype=np.float64)
    source_lon = np.asarray([-0.25, 0.25], dtype=np.float64)
    source_lat = np.asarray([-0.25, 0.25], dtype=np.float64)
    values = np.asarray(
        [
            [[1.0, 10.0], [3.0, 30.0]],
            [[5.0, 50.0], [7.0, 70.0]],
        ],
        dtype=np.float64,
    )

    mean, counts = bbox_center_mean(lalo, resolution, source_lon, source_lat, values)

    assert np.array_equal(counts, [4])
    assert mean.shape == (1, 2)
    assert np.allclose(mean, [[4.0, 40.0]])


def test_bbox_center_mean_raises_when_no_source_center_is_inside_bbox():
    lalo = np.asarray([[0.0, 0.0]], dtype=np.float64)
    resolution = np.asarray([[10_000.0, 10_000.0]], dtype=np.float64)
    source_lon = np.asarray([5.0], dtype=np.float64)
    source_lat = np.asarray([5.0], dtype=np.float64)
    values = np.asarray([[1.0]], dtype=np.float64)

    try:
        bbox_center_mean(lalo, resolution, source_lon, source_lat, values)
    except ValueError as exc:
        assert "no valid source centers" in str(exc)
    else:
        raise AssertionError("empty bbox did not raise")


def test_slowproc_nearest_indices_matches_fortran_cosang_maxloc():
    source_lon = np.asarray([0.0, 5.0, 20.0], dtype=np.float64)
    source_lat = np.asarray([0.0, 1.0, 20.0], dtype=np.float64)
    lalo = np.asarray([[1.1, 5.2], [19.0, 20.0]], dtype=np.float64)

    nearest = slowproc_nearest_indices(source_lon, source_lat, lalo)

    assert np.array_equal(nearest, np.asarray([1, 2], dtype=np.int32))


def test_bbox_center_mean_can_use_fortran_nearest_fallback_for_empty_targets():
    lalo = np.asarray([[0.0, 0.0]], dtype=np.float64)
    resolution = np.asarray([[10_000.0, 10_000.0]], dtype=np.float64)
    source_lon = np.asarray([5.0, 1.0], dtype=np.float64)
    source_lat = np.asarray([5.0, 0.1], dtype=np.float64)
    values = np.asarray([[10.0, 11.0], [12.0, 20.0]], dtype=np.float64)

    mean, counts = bbox_center_mean(
        lalo,
        resolution,
        source_lon,
        source_lat,
        values,
        allow_nearest_fallback=True,
    )

    assert np.array_equal(counts, [0])
    assert np.allclose(mean, [20.0])


def test_bbox_center_mean_nearest_fallback_preserves_trailing_time_axis():
    lalo = np.asarray([[0.0, 0.0]], dtype=np.float64)
    resolution = np.asarray([[10_000.0, 10_000.0]], dtype=np.float64)
    source_lon = np.asarray([5.0, 1.0], dtype=np.float64)
    source_lat = np.asarray([5.0, 0.1], dtype=np.float64)
    values = np.asarray(
        [
            [[10.0, 11.0], [12.0, 13.0]],
            [[14.0, 15.0], [20.0, 21.0]],
        ],
        dtype=np.float64,
    )

    mean, counts = bbox_center_mean(
        lalo,
        resolution,
        source_lon,
        source_lat,
        values,
        allow_nearest_fallback=True,
    )

    assert np.array_equal(counts, [0])
    assert mean.shape == (1, 2)
    assert np.allclose(mean, [[20.0, 21.0]])


def test_weighted_mean_from_explicit_overlap_uses_sub_area_and_one_based_indices():
    source_values = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    sub_index = np.asarray([[[1, 1], [2, 2], [0, 0]]], dtype=np.int32)
    sub_area = np.asarray([[1.0, 3.0, 0.0]], dtype=np.float64)

    out = weighted_mean_from_explicit_overlap(source_values, sub_index, sub_area)

    assert out.shape == (1,)
    assert np.allclose(out, [(10.0 * 1.0 + 40.0 * 3.0) / 4.0])


def test_thermosoil_refsoc_overlap_uses_fortran_zero_fallback_for_empty_targets():
    source_values = np.asarray(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[50.0, 60.0], [70.0, 80.0]],
        ],
        dtype=np.float64,
    )
    sub_index = np.asarray(
        [
            [[1, 1], [2, 2]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    sub_area = np.asarray(
        [
            [1.0, 3.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    refsoc = thermosoil_refsoc_from_explicit_overlap(source_values, sub_index, sub_area)

    assert refsoc.shape == (2, 2)
    assert np.allclose(refsoc[0], [(10.0 + 70.0 * 3.0) / 4.0, (20.0 + 80.0 * 3.0) / 4.0])
    assert np.allclose(refsoc[1], [0.0, 0.0])


def test_hydrol_refsoc_1d_overlap_uses_fortran_zero_fallback_for_empty_targets():
    source_values = np.asarray([[10.0, 30.0], [50.0, 70.0]], dtype=np.float64)
    sub_index = np.asarray(
        [
            [[1, 1], [2, 2]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.int32,
    )
    sub_area = np.asarray(
        [
            [1.0, 3.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    refsoc = hydrol_refsoc_1d_from_explicit_overlap(source_values, sub_index, sub_area)

    assert refsoc.shape == (2,)
    assert np.allclose(refsoc[0], (10.0 + 70.0 * 3.0) / 4.0)
    assert np.allclose(refsoc[1], 0.0)


def test_zobler_soilclass_from_explicit_overlap_and_njsc_follow_fortran_maxloc():
    soiltext = np.asarray([[1, 2], [5, 6]], dtype=np.float64)
    sub_index = np.asarray(
        [
            [[1, 1], [1, 2], [2, 1]],
            [[1, 1], [2, 2], [0, 0]],
        ],
        dtype=np.int32,
    )
    sub_area = np.asarray(
        [
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    soilclass = zobler_soilclass_from_explicit_overlap(soiltext, sub_index, sub_area)
    njsc = njsc_from_soilclass(soilclass)

    assert np.allclose(soilclass[0], [0.25, 0.5, 0.25])
    assert np.allclose(soilclass[1], [0.5, 0.5, 0.0])
    assert np.array_equal(njsc, np.asarray([2, 1], dtype=np.int32))


def test_aggregate_2d_overlap_uses_mask_and_fortran_area_formula():
    lalo = np.asarray([[21.0, 109.0]], dtype=np.float64)
    resolution = np.asarray([[111106.370838091, 111111.0]], dtype=np.float64)
    lon = np.asarray([[108.5, 108.5], [109.5, 109.5]], dtype=np.float64)
    lat = np.asarray([[21.5, 20.5], [21.5, 20.5]], dtype=np.float64)
    mask = np.asarray([[False, False], [True, False]])

    sub_index, sub_area = aggregate_2d_overlap(lalo, resolution, lon, lat, mask)

    assert np.array_equal(sub_index, np.asarray([[[2, 1]]], dtype=np.int32))
    assert sub_area.shape == (1, 1)
    assert sub_area[0, 0] > 0.0


def test_regular_lonlat_grid_geometry_matches_haversine_neighbour_order():
    lon_axis = np.asarray([107.0, 109.0, 111.0], dtype=np.float64)
    lat_axis = np.asarray([23.0, 21.0, 19.0], dtype=np.float64)
    lon = np.repeat(lon_axis[:, None], 3, axis=1)
    lat = np.repeat(lat_axis[None, :], 3, axis=0)
    kindex = np.arange(1, 10, dtype=np.int32)

    resolution, neighbours, area, seglength = regular_lonlat_grid_geometry(lon, lat, kindex)

    np.testing.assert_allclose(
        resolution,
        np.asarray(
            [
                [204904.64835356, 222634.1993844],
                [204904.64835356, 222634.1993844],
                [204904.64835356, 222634.1993844],
                [207815.27471986, 222634.1993844],
                [207815.27471986, 222634.1993844],
                [207815.27471986, 222634.1993844],
                [210472.71018539, 222634.1993844],
                [210472.71018539, 222634.1993844],
                [210472.71018539, 222634.1993844],
            ]
        ),
    )
    np.testing.assert_array_equal(
        neighbours,
        np.asarray(
            [
                [-1, -1, 2, 5, 4, -1, -1, -1],
                [-1, -1, 3, 6, 5, 4, 1, -1],
                [-1, -1, -1, -1, 6, 5, 2, -1],
                [1, 2, 5, 8, 7, -1, -1, -1],
                [2, 3, 6, 9, 8, 7, 4, 1],
                [3, -1, -1, -1, 9, 8, 5, 2],
                [4, 5, 8, -1, -1, -1, -1, -1],
                [5, 6, 9, -1, -1, -1, 7, 4],
                [6, -1, -1, -1, -1, -1, 8, 5],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_allclose(area, resolution[:, 0] * resolution[:, 1])
    np.testing.assert_allclose(resolution[:, 0], (seglength[:, 0] + seglength[:, 2]) / 2.0)
    np.testing.assert_allclose(resolution[:, 1], (seglength[:, 1] + seglength[:, 3]) / 2.0)


def test_usda_soil_fields_from_explicit_overlap_follows_usda_table_and_maxloc():
    soiltext = np.asarray([[5.0]], dtype=np.float64)
    soilbd = np.asarray([[1.38594591617584]], dtype=np.float64)
    soilph = np.asarray([[5.66892290115356]], dtype=np.float64)
    poorsol = np.asarray([[8.333333535119891e-4]], dtype=np.float64)
    sub_index = np.asarray([[[1, 1]]], dtype=np.int32)
    sub_area = np.asarray([[3075829029.14936]], dtype=np.float64)

    fields = usda_soil_fields_from_explicit_overlap(soiltext, soilbd, soilph, poorsol, sub_index, sub_area)

    expected_soilclass = np.zeros((1, 12), dtype=np.float64)
    expected_soilclass[0, 4] = 1.0
    assert np.allclose(fields["soilclass"], expected_soilclass)
    assert np.array_equal(fields["njsc"], np.asarray([5], dtype=np.int32))
    assert np.allclose(fields["clay_frac"], [0.10])
    assert np.allclose(fields["sand_frac"], [0.06])
    assert np.allclose(fields["silt_frac"], [0.84])
    assert np.allclose(fields["bulk_dens"], [1.38594591617584])
    assert np.allclose(fields["soil_ph"], [5.66892290115356])
    assert np.allclose(fields["poor_soils"], [8.333333535119891e-4])


def test_read_paper_condveg_background_soilalbedo_returns_two_band_local_field():
    refsoc_config = ROOT / "configs" / "orchidee_man_250919.yaml"
    lalo = np.asarray([[21.0, 109.0]], dtype=np.float64)
    resolution = np.asarray([[111106.370838091, 111111.0]], dtype=np.float64)

    soilalb, meta = read_paper_condveg_background_soilalbedo(
        refsoc_config,
        lalo=lalo,
        resolution_m=resolution,
    )

    assert soilalb.shape == (1, 2)
    assert np.all(np.isfinite(soilalb))
    assert np.all((soilalb >= 0.0) & (soilalb <= 1.0))
    assert meta["soilalbedo_bg_source_count_vis"].shape == (1,)


def test_read_paper_thermosoil_reftemp_static_field_returns_repeated_profile():
    refsoc_config = ROOT / "configs" / "orchidee_man_250919.yaml"
    lalo = np.asarray([[21.0, 109.0]], dtype=np.float64)
    resolution = np.asarray([[111106.370838091, 111111.0]], dtype=np.float64)

    ptn, meta = read_paper_thermosoil_reftemp_static_field(
        refsoc_config,
        lalo=lalo,
        resolution_m=resolution,
        ngrnd=4,
        nvm=3,
    )

    assert ptn.shape == (1, 4, 3)
    assert np.all(np.isfinite(ptn))
    np.testing.assert_allclose(ptn[:, :, 0], ptn[:, :, 1])
    np.testing.assert_allclose(ptn[:, 0, :], ptn[:, -1, :])
    assert meta["reftemp_sub_index"].shape[0] == 1


def test_read_paper_thermosoil_refsoc_static_field_returns_vertical_profile():
    refsoc_config = ROOT / "configs" / "orchidee_man_250919.yaml"
    lalo = np.asarray([[21.0, 109.0]], dtype=np.float64)
    resolution = np.asarray([[111106.370838091, 111111.0]], dtype=np.float64)

    refsoc, meta = read_paper_thermosoil_refsoc_static_field(
        refsoc_config,
        lalo=lalo,
        resolution_m=resolution,
    )

    assert refsoc.ndim == 2
    assert refsoc.shape[0] == 1
    assert refsoc.shape[1] > 1
    assert np.all(np.isfinite(refsoc))
    assert np.all(refsoc >= 0.0)
    assert meta["refsoc_sub_index"].shape[0] == 1


def test_read_paper_hydrol_refsoc_1d_static_field_returns_local_field():
    refsoc_config = ROOT / "configs" / "orchidee_man_250919.yaml"
    lalo = np.asarray([[21.0, 109.0]], dtype=np.float64)
    resolution = np.asarray([[111106.370838091, 111111.0]], dtype=np.float64)

    refsoc, meta = read_paper_hydrol_refsoc_1d_static_field(
        refsoc_config,
        lalo=lalo,
        resolution_m=resolution,
    )

    assert refsoc.shape == (1,)
    assert np.all(np.isfinite(refsoc))
    assert refsoc[0] >= 0.0
    assert meta["refsoc_1d_sub_index"].shape[0] == 1
    assert meta["refsoc_1d_sub_area"].shape[0] == 1
