from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (
    SelectedMineralHydrolCoefficients,
    bottom_drainage,
    build_mineral_cwrr_tables,
    drainage_correction,
    hydrol_soil_coef_mineral_from_selected_coefficients,
    hydrol_soil_coef_mineral_profile_from_tables,
    hydrol_soil_coef_mineral_from_tables,
    hydrol_soil_setup_coefficients,
    select_mineral_cwrr_bin,
)


TRACE_DIR = ROOT / "traces" / "hydrol_1961_v9"
COEF_CSV = TRACE_DIR / "jcwu_hydrol_soil_coef_tile4_bottom_v8.csv"
DRAINAGE_CSV = TRACE_DIR / "jcwu_hydrol_tile4_trace.csv"
REFERENCE_Z_M = np.asarray(
    [
        0.000977517107,
        0.003910068428,
        0.00977517107,
        0.021505376354,
        0.044965786922,
        0.091886608058,
        0.185728250332,
        0.37341153488,
        0.748778104,
        1.49951124224,
        2.0,
    ],
    dtype=np.float64,
)
REFERENCE_DZ_MM = np.asarray(
    [
        0.0,
        2.932551321,
        5.865102642,
        11.730205284,
        23.460410568,
        46.920821136,
        93.841642274,
        187.683284548,
        375.36656912,
        750.73313824,
        750.73313824,
    ],
    dtype=np.float64,
)


def _read_float_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    values = {column: [] for column in columns}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                values[column].append(float(row[column]))
    return {column: np.asarray(items, dtype=np.float64) for column, items in values.items()}


def _read_string_columns(path: Path, columns: list[str]) -> dict[str, list[str]]:
    values = {column: [] for column in columns}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                values[column].append(row[column].strip())
    return values


def test_hydrol_soil_coef_mineral_trace_rows_match_fortran_trace():
    flags = _read_string_columns(COEF_CSV, ["branch_peat", "ok_freeze_cwrr", "peat_hydro"])
    assert set(flags["branch_peat"]) == {"F"}
    assert set(flags["ok_freeze_cwrr"]) == {"T"}
    assert set(flags["peat_hydro"]) == {"F"}

    cols = _read_float_columns(
        COEF_CSV,
        [
            "mc",
            "profil_froz",
            "mc_used",
            "kfact_root",
            "a_raw",
            "b_raw",
            "d_raw",
            "k_floor_raw",
            "k_eval_raw",
            "k_result",
            "a_scaled",
            "b_scaled",
            "d_scaled",
        ],
    )

    assert np.isclose(cols["k_result"][0], 6.0648789143219517, rtol=0.0, atol=1e-14)

    coefficients = SelectedMineralHydrolCoefficients(
        a_raw=cols["a_raw"],
        b_raw=cols["b_raw"],
        d_raw=cols["d_raw"],
        k_floor_raw=cols["k_floor_raw"],
    )
    result = hydrol_soil_coef_mineral_from_selected_coefficients(
        mc=cols["mc"],
        profil_froz=cols["profil_froz"],
        kfact_root=cols["kfact_root"],
        coefficients=coefficients,
    )

    mc_used = np.asarray(result.mc_used)
    k_eval = np.asarray(result.k_eval)
    k_result = np.asarray(result.k)

    assert np.allclose(mc_used, cols["mc_used"], rtol=0.0, atol=1e-15)
    assert np.allclose(k_eval, cols["k_eval_raw"], rtol=0.0, atol=1e-13)
    assert np.allclose(k_result, cols["k_result"], rtol=0.0, atol=1e-13)
    assert np.allclose(k_result, np.maximum(cols["k_floor_raw"], cols["a_raw"] * mc_used + cols["b_raw"]))
    assert np.allclose(np.asarray(result.a), cols["a_scaled"], rtol=0.0, atol=1e-13)
    assert np.allclose(np.asarray(result.b), cols["b_scaled"], rtol=0.0, atol=1e-13)
    assert np.allclose(np.asarray(result.d), cols["d_scaled"], rtol=0.0, atol=1e-9)


def test_source_driven_mineral_cwrr_tables_generate_trace_coefficients():
    cols = _read_float_columns(
        COEF_CSV,
        [
            "bin_i",
            "mc",
            "profil_froz",
            "kfact_root",
            "a_raw",
            "b_raw",
            "d_raw",
            "k_floor_raw",
            "k_result",
        ],
    )

    tables = build_mineral_cwrr_tables(
        njsc=2,
        mcs=0.41,
        z_m=REFERENCE_Z_M,
    )

    bottom_layer = 10
    first_bin = int(cols["bin_i"][0])
    first_offset = first_bin - tables.imin
    assert first_bin == 35

    assert np.isclose(np.asarray(tables.a_lin)[first_offset, bottom_layer, 0], 175.91824059050936)
    assert np.isclose(np.asarray(tables.b_lin)[first_offset, bottom_layer, 0], -46.710593262830855)
    assert np.isclose(np.asarray(tables.d_lin)[first_offset, bottom_layer, 0], 27504.887686410824)
    assert np.isclose(np.asarray(tables.k_lin)[1, bottom_layer, 0], 2.0635528515138059e-09)

    coefficients = SelectedMineralHydrolCoefficients(
        a_raw=np.asarray(tables.a_lin)[first_offset, bottom_layer, 0],
        b_raw=np.asarray(tables.b_lin)[first_offset, bottom_layer, 0],
        d_raw=np.asarray(tables.d_lin)[first_offset, bottom_layer, 0],
        k_floor_raw=np.asarray(tables.k_lin)[1, bottom_layer, 0],
    )
    result = hydrol_soil_coef_mineral_from_selected_coefficients(
        mc=0.3,
        profil_froz=0.0,
        kfact_root=1.0,
        coefficients=coefficients,
    )
    assert np.isclose(np.asarray(result.k), 6.0648789143219517, rtol=0.0, atol=1e-14)

    offsets = cols["bin_i"].astype(np.int64) - tables.imin
    a_selected = np.asarray(tables.a_lin)[offsets, bottom_layer, 0]
    b_selected = np.asarray(tables.b_lin)[offsets, bottom_layer, 0]
    d_selected = np.asarray(tables.d_lin)[offsets, bottom_layer, 0]
    k_floor = np.asarray(tables.k_lin)[1, bottom_layer, 0]

    assert np.allclose(a_selected, cols["a_raw"], rtol=0.0, atol=1e-11)
    assert np.allclose(b_selected, cols["b_raw"], rtol=0.0, atol=1e-11)
    assert np.allclose(d_selected, cols["d_raw"], rtol=0.0, atol=1e-7)
    assert np.allclose(np.full_like(cols["k_floor_raw"], k_floor), cols["k_floor_raw"], rtol=0.0, atol=1e-20)

    coefficients = SelectedMineralHydrolCoefficients(
        a_raw=a_selected,
        b_raw=b_selected,
        d_raw=d_selected,
        k_floor_raw=np.full_like(cols["k_floor_raw"], k_floor),
    )
    result = hydrol_soil_coef_mineral_from_selected_coefficients(
        mc=cols["mc"],
        profil_froz=cols["profil_froz"],
        kfact_root=cols["kfact_root"],
        coefficients=coefficients,
    )
    assert np.allclose(np.asarray(result.k), cols["k_result"], rtol=0.0, atol=1e-13)


def test_table_driven_mineral_hydrol_soil_coef_rebuilds_trace_rows():
    flags = _read_string_columns(COEF_CSV, ["branch_peat", "ok_freeze_cwrr", "peat_hydro"])
    assert set(flags["branch_peat"]) == {"F"}
    assert set(flags["ok_freeze_cwrr"]) == {"T"}
    assert set(flags["peat_hydro"]) == {"F"}

    cols = _read_float_columns(
        COEF_CSV,
        [
            "mc",
            "profil_froz",
            "mc_used",
            "bin_i",
            "kfact_root",
            "a_raw",
            "b_raw",
            "d_raw",
            "k_floor_raw",
            "k_eval_raw",
            "k_result",
            "a_scaled",
            "b_scaled",
            "d_scaled",
        ],
    )

    tables = build_mineral_cwrr_tables(
        njsc=2,
        mcs=0.41,
        z_m=REFERENCE_Z_M,
    )

    bottom_layer = 10
    mcr = 0.057
    mcs = 0.41

    selection = select_mineral_cwrr_bin(
        mc=cols["mc"],
        profil_froz=cols["profil_froz"],
        mcr=mcr,
        mcs=mcs,
        imin=tables.imin,
        imax=tables.imax,
    )
    assert np.allclose(np.asarray(selection.mc_used), cols["mc_used"], rtol=0.0, atol=1e-15)
    assert np.array_equal(np.asarray(selection.bin_i), cols["bin_i"].astype(np.int32))
    assert np.array_equal(np.asarray(selection.bin_offset), cols["bin_i"].astype(np.int32) - tables.imin)

    result = hydrol_soil_coef_mineral_from_tables(
        mc=cols["mc"],
        profil_froz=cols["profil_froz"],
        kfact_root=cols["kfact_root"],
        tables=tables,
        mcr=mcr,
        mcs=mcs,
        layer_index=bottom_layer,
        point_index=0,
    )

    assert np.allclose(np.asarray(result.mc_used), cols["mc_used"], rtol=0.0, atol=1e-15)
    assert np.array_equal(np.asarray(result.bin_i), cols["bin_i"].astype(np.int32))
    assert np.allclose(np.asarray(result.a_raw), cols["a_raw"], rtol=0.0, atol=1e-11)
    assert np.allclose(np.asarray(result.b_raw), cols["b_raw"], rtol=0.0, atol=1e-11)
    assert np.allclose(np.asarray(result.d_raw), cols["d_raw"], rtol=0.0, atol=1e-7)
    assert np.allclose(np.asarray(result.k_floor_raw), cols["k_floor_raw"], rtol=0.0, atol=1e-20)
    assert np.allclose(np.asarray(result.k_eval), cols["k_eval_raw"], rtol=0.0, atol=1e-13)
    assert np.allclose(np.asarray(result.k), cols["k_result"], rtol=0.0, atol=1e-13)
    assert np.allclose(np.asarray(result.a), cols["a_scaled"], rtol=0.0, atol=1e-11)
    assert np.allclose(np.asarray(result.b), cols["b_scaled"], rtol=0.0, atol=1e-11)
    assert np.allclose(np.asarray(result.d), cols["d_scaled"], rtol=0.0, atol=1e-7)


def test_profile_mineral_hydrol_soil_coef_matches_layerwise_freezing_helper():
    tables = build_mineral_cwrr_tables(
        njsc=np.asarray([2, 3]),
        mcs=np.asarray([0.41, 0.45]),
        z_m=REFERENCE_Z_M,
    )
    nslm = REFERENCE_Z_M.size
    mc = np.vstack(
        [
            np.linspace(0.22, 0.36, nslm),
            np.linspace(0.24, 0.40, nslm),
        ]
    )
    profil_froz = np.vstack(
        [
            np.linspace(0.0, 0.20, nslm),
            np.linspace(0.10, 0.00, nslm),
        ]
    )
    kfact_root = np.vstack(
        [
            np.linspace(1.0, 1.3, nslm),
            np.linspace(0.9, 1.1, nslm),
        ]
    )
    mcr = np.asarray([0.057, 0.065])
    mcs = np.asarray([0.41, 0.45])

    profile = hydrol_soil_coef_mineral_profile_from_tables(
        mc=mc,
        profil_froz=profil_froz,
        kfact_root=kfact_root,
        tables=tables,
        mcr=mcr,
        mcs=mcs,
        ok_freeze_cwrr=True,
    )

    for point in range(mc.shape[0]):
        layerwise = hydrol_soil_coef_mineral_from_tables(
            mc=mc[point],
            profil_froz=profil_froz[point],
            kfact_root=kfact_root[point],
            tables=tables,
            mcr=mcr[point],
            mcs=mcs[point],
            layer_index=np.arange(nslm),
            point_index=point,
        )
        np.testing.assert_allclose(np.asarray(profile.a[point]), np.asarray(layerwise.a))
        np.testing.assert_allclose(np.asarray(profile.b[point]), np.asarray(layerwise.b))
        np.testing.assert_allclose(np.asarray(profile.d[point]), np.asarray(layerwise.d))
        np.testing.assert_allclose(np.asarray(profile.k[point]), np.asarray(layerwise.k))
        np.testing.assert_allclose(np.asarray(profile.mc_used[point]), np.asarray(layerwise.mc_used))
        np.testing.assert_array_equal(np.asarray(profile.bin_i[point]), np.asarray(layerwise.bin_i))


def test_profile_mineral_hydrol_soil_coef_no_freeze_uses_source_mc_ratio_and_total_mc_for_k():
    tables = build_mineral_cwrr_tables(
        njsc=2,
        mcs=0.41,
        z_m=REFERENCE_Z_M,
    )
    mc = np.linspace(0.04, 0.46, REFERENCE_Z_M.size)[None, :]
    profil_froz = np.full_like(mc, 0.75)
    kfact_root = np.linspace(0.8, 1.2, REFERENCE_Z_M.size)[None, :]
    mcr = np.asarray([0.057])
    mcs = np.asarray([0.41])

    result = hydrol_soil_coef_mineral_profile_from_tables(
        mc=mc,
        profil_froz=profil_froz,
        kfact_root=kfact_root,
        tables=tables,
        mcr=mcr,
        mcs=mcs,
        ok_freeze_cwrr=False,
    )

    mc_ratio = np.maximum(mc[0] - mcr[0], 0.0) / (mcs[0] - mcr[0])
    expected_bin_i = np.maximum(
        np.minimum(((tables.imax - tables.imin) * mc_ratio).astype(np.int32) + tables.imin, tables.imax - 1),
        tables.imin,
    )
    expected_bin_offset = expected_bin_i - tables.imin
    layer_index = np.arange(REFERENCE_Z_M.size)
    a_raw = np.asarray(tables.a_lin)[expected_bin_offset, layer_index, 0]
    b_raw = np.asarray(tables.b_lin)[expected_bin_offset, layer_index, 0]
    d_raw = np.asarray(tables.d_lin)[expected_bin_offset, layer_index, 0]
    k_floor = np.asarray(tables.k_lin)[tables.imin + 1 - tables.imin, layer_index, 0]

    np.testing.assert_array_equal(np.asarray(result.bin_i[0]), expected_bin_i)
    np.testing.assert_allclose(np.asarray(result.mc_used[0]), mc[0])
    np.testing.assert_allclose(np.asarray(result.a[0]), a_raw * kfact_root[0])
    np.testing.assert_allclose(np.asarray(result.b[0]), b_raw * kfact_root[0])
    np.testing.assert_allclose(np.asarray(result.d[0]), d_raw * kfact_root[0])
    np.testing.assert_allclose(np.asarray(result.k_eval[0]), a_raw * mc[0] + b_raw)
    np.testing.assert_allclose(np.asarray(result.k[0]), np.maximum(k_floor, a_raw * mc[0] + b_raw))


def test_hydrol_soil_setup_coefficients_match_source_formula_invariants():
    tables = build_mineral_cwrr_tables(
        njsc=2,
        mcs=0.41,
        z_m=REFERENCE_Z_M,
    )
    mc_profile = np.linspace(0.30, 0.34, REFERENCE_Z_M.size)
    coef = hydrol_soil_coef_mineral_from_tables(
        mc=mc_profile,
        profil_froz=np.zeros_like(mc_profile),
        kfact_root=np.ones_like(mc_profile),
        tables=tables,
        mcr=0.057,
        mcs=0.41,
        layer_index=np.arange(REFERENCE_Z_M.size),
        point_index=0,
    )

    a = np.asarray(coef.a)[None, :]
    d = np.asarray(coef.d)[None, :]
    dt_days = 1.0 / 48.0
    free_drain_coef = np.asarray([1.0])
    setup = hydrol_soil_setup_coefficients(
        a=a,
        d=d,
        dz_mm=REFERENCE_DZ_MM,
        dt_days=dt_days,
        free_drain_coef=free_drain_coef,
    )

    e = np.asarray(setup.e)
    f = np.asarray(setup.f)
    g1 = np.asarray(setup.g1)
    ep = np.asarray(setup.ep)
    fp = np.asarray(setup.fp)
    gp = np.asarray(setup.gp)

    assert np.allclose(e[:, 0], 0.0)
    assert np.allclose(g1[:, -1], 0.0)
    assert np.allclose(ep[:, 0], 0.0)
    assert np.allclose(gp[:, -1], 0.0)

    temp3 = dt_days / 2.0
    assert np.allclose(fp[0, 0], 3.0 * REFERENCE_DZ_MM[1] / 8.0)
    assert np.allclose(gp[0, 0], REFERENCE_DZ_MM[1] / 8.0)
    assert np.allclose(ep[0, 1:-1], REFERENCE_DZ_MM[1:-1] / 8.0)
    assert np.allclose(fp[0, 1:-1], 3.0 * (REFERENCE_DZ_MM[1:-1] + REFERENCE_DZ_MM[2:]) / 8.0)
    assert np.allclose(gp[0, 1:-1], REFERENCE_DZ_MM[2:] / 8.0)
    assert np.allclose(ep[0, -1], REFERENCE_DZ_MM[-1] / 8.0)
    assert np.allclose(fp[0, -1], 3.0 * REFERENCE_DZ_MM[-1] / 8.0)

    expected_f_top = 3.0 * REFERENCE_DZ_MM[1] / 8.0 + temp3 * (
        (d[0, 0] + d[0, 1]) / REFERENCE_DZ_MM[1] + a[0, 0]
    )
    expected_g_top = REFERENCE_DZ_MM[1] / 8.0 - temp3 * (
        (d[0, 0] + d[0, 1]) / REFERENCE_DZ_MM[1] - a[0, 1]
    )
    assert np.allclose(f[0, 0], expected_f_top)
    assert np.allclose(g1[0, 0], expected_g_top)

    jsl = 5
    expected_e_mid = REFERENCE_DZ_MM[jsl] / 8.0 - temp3 * (
        (d[0, jsl] + d[0, jsl - 1]) / REFERENCE_DZ_MM[jsl] + a[0, jsl - 1]
    )
    expected_f_mid = 3.0 * (REFERENCE_DZ_MM[jsl] + REFERENCE_DZ_MM[jsl + 1]) / 8.0 + temp3 * (
        (d[0, jsl] + d[0, jsl - 1]) / REFERENCE_DZ_MM[jsl]
        + (d[0, jsl] + d[0, jsl + 1]) / REFERENCE_DZ_MM[jsl + 1]
    )
    expected_g_mid = REFERENCE_DZ_MM[jsl + 1] / 8.0 - temp3 * (
        (d[0, jsl] + d[0, jsl + 1]) / REFERENCE_DZ_MM[jsl + 1] - a[0, jsl + 1]
    )
    assert np.allclose(e[0, jsl], expected_e_mid)
    assert np.allclose(f[0, jsl], expected_f_mid)
    assert np.allclose(g1[0, jsl], expected_g_mid)

    last = REFERENCE_Z_M.size - 1
    expected_e_bottom = REFERENCE_DZ_MM[last] / 8.0 - temp3 * (
        (d[0, last] + d[0, last - 1]) / REFERENCE_DZ_MM[last] + a[0, last - 1]
    )
    expected_f_bottom = 3.0 * REFERENCE_DZ_MM[last] / 8.0 + temp3 * (
        (d[0, last] + d[0, last - 1]) / REFERENCE_DZ_MM[last]
        - a[0, last] * (1.0 - 2.0 * free_drain_coef[0])
    )
    assert np.allclose(e[0, last], expected_e_bottom)
    assert np.allclose(f[0, last], expected_f_bottom)


def test_bottom_drainage_and_correction_trace_rows_match_fortran_trace():
    cols = _read_float_columns(
        DRAINAGE_CSV,
        [
            "dr_ns_before_corr_at_6012",
            "k_6012_bottom",
            "free_drain_coef_6012",
            "dt_days",
            "check_tr_ns_6047",
            "dr_corrnum_ns_6047",
            "dr_after_corr_6047",
        ],
    )

    assert np.isclose(
        cols["dr_ns_before_corr_at_6012"][0],
        0.12635164404837398,
        rtol=0.0,
        atol=1e-15,
    )

    dr_before = np.asarray(
        bottom_drainage(
            k_bottom=cols["k_6012_bottom"],
            free_drain_coef=cols["free_drain_coef_6012"],
            dt_days=cols["dt_days"],
        )
    )

    expected_before = cols["k_6012_bottom"] * cols["free_drain_coef_6012"] * cols["dt_days"]
    assert np.allclose(dr_before, expected_before, rtol=0.0, atol=1e-15)
    assert np.allclose(dr_before, cols["dr_ns_before_corr_at_6012"], rtol=0.0, atol=1e-15)

    correction = drainage_correction(dr_before, cols["check_tr_ns_6047"])
    dr_corrnum = np.asarray(correction.dr_corrnum)
    dr_after = np.asarray(correction.dr_after)

    expected_corrnum = np.where(
        cols["check_tr_ns_6047"] < 0.0,
        -cols["check_tr_ns_6047"],
        -np.minimum(dr_before, cols["check_tr_ns_6047"]),
    )
    assert np.allclose(dr_corrnum, expected_corrnum, rtol=0.0, atol=1e-15)
    assert np.allclose(dr_corrnum, cols["dr_corrnum_ns_6047"], rtol=0.0, atol=1e-15)
    assert np.allclose(dr_after, dr_before + dr_corrnum, rtol=0.0, atol=1e-15)
    assert np.allclose(dr_after, cols["dr_after_corr_6047"], rtol=0.0, atol=1e-15)
    assert np.all(dr_after >= -1e-15)
