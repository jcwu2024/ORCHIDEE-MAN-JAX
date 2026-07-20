from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    PEAT_KS,
    PEAT_MCR,
    PEAT_MCS,
    build_peat_cwrr_tables,
    hydrol_soil_coef_peat_profile_from_tables,
    hydrol_soil_tile_explicit_step,
)


def test_build_peat_cwrr_tables_matches_source_bounds_and_depth_factors():
    z_m = np.asarray([0.00098, 0.00391, 0.00978, 0.02151], dtype=np.float64)
    tables = build_peat_cwrr_tables(z_m)

    np.testing.assert_allclose(np.asarray(tables.mc_lin[0]), PEAT_MCR)
    np.testing.assert_allclose(np.asarray(tables.mc_lin[-1]), PEAT_MCS)
    np.testing.assert_allclose(np.asarray(tables.k_lin[-1]), PEAT_KS * np.asarray(tables.kfact))
    assert np.all(np.asarray(tables.kfact) <= 1.0)
    assert np.all(np.asarray(tables.kfact) >= 0.1)
    assert np.all(np.asarray(tables.d_lin[0]) > 0.0)


def test_hydrol_soil_coef_peat_profile_freeze_and_no_freeze_follow_source_formulas():
    z_m = np.asarray([0.00098, 0.00391, 0.00978, 0.02151], dtype=np.float64)
    tables = build_peat_cwrr_tables(z_m)
    mc = np.asarray([[0.10, 0.30, 0.61, 0.95]], dtype=np.float64)
    profil_froz = np.asarray([[0.0, 0.25, 0.50, 0.75]], dtype=np.float64)
    kfact_root = np.asarray([[1.0, 1.1, 1.2, 1.3]], dtype=np.float64)

    frozen = hydrol_soil_coef_peat_profile_from_tables(
        mc=mc,
        profil_froz=profil_froz,
        kfact_root=kfact_root,
        tables=tables,
        ok_freeze_cwrr=True,
    )
    no_freeze = hydrol_soil_coef_peat_profile_from_tables(
        mc=mc,
        profil_froz=profil_froz,
        kfact_root=kfact_root,
        tables=tables,
        ok_freeze_cwrr=False,
    )

    expected_mc_used = PEAT_MCR + (1.0 - profil_froz[0]) * np.maximum(mc[0] - PEAT_MCR, 0.0)
    np.testing.assert_allclose(np.asarray(frozen.mc_used[0]), expected_mc_used)
    np.testing.assert_allclose(np.asarray(no_freeze.mc_used[0]), mc[0])

    ratio = np.maximum(mc[0] - PEAT_MCR, 0.0) / (PEAT_MCS - PEAT_MCR)
    expected_bin_i = np.maximum(
        np.minimum(((tables.imax - tables.imin) * ratio).astype(np.int32) + tables.imin, tables.imax - 1),
        tables.imin,
    )
    np.testing.assert_array_equal(np.asarray(no_freeze.bin_i[0]), expected_bin_i)

    layer = np.arange(z_m.size)
    a_raw = np.asarray(tables.a_lin)[expected_bin_i - tables.imin, layer]
    b_raw = np.asarray(tables.b_lin)[expected_bin_i - tables.imin, layer]
    k_floor = np.asarray(tables.k_lin)[tables.imin + 1 - tables.imin, layer]
    np.testing.assert_allclose(np.asarray(no_freeze.k_eval[0]), a_raw * mc[0] + b_raw)
    np.testing.assert_allclose(np.asarray(no_freeze.k[0]), np.maximum(k_floor, a_raw * mc[0] + b_raw))
    np.testing.assert_allclose(np.asarray(no_freeze.a[0]), a_raw * kfact_root[0])


def test_tile_adapter_uses_peat_tables_for_active_pft14_soil_tile():
    z_m = np.asarray([0.00098, 0.00391, 0.00978, 0.02151], dtype=np.float64)
    tables = build_peat_cwrr_tables(z_m)
    inputs = {
        "water2infilt": np.asarray([0.10], dtype=np.float64),
        "ae_ns_tile": np.asarray([0.02], dtype=np.float64),
        "subsinksoil": np.asarray([0.00], dtype=np.float64),
        "precisol_ns_tile": np.asarray([0.40], dtype=np.float64),
        "reinfiltration_soil": np.asarray([0.00], dtype=np.float64),
        "is_crop_soil": False,
        "mc": np.asarray([[0.30, 0.34, 0.38, 0.42]], dtype=np.float64),
        "mcl": np.asarray([[0.30, 0.34, 0.38, 0.42]], dtype=np.float64),
        "profil_froz": np.asarray([[0.0, 0.05, 0.10, 0.15]], dtype=np.float64),
        "mcr": np.asarray([PEAT_MCR], dtype=np.float64),
        "mcs": np.asarray([PEAT_MCS], dtype=np.float64),
        "dz_mm": np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64),
        "dt_days": 1.0 / 48.0,
        "free_drain_coef": np.asarray([1.0], dtype=np.float64),
        "rootsink": np.asarray([[0.0, 0.01, 0.01, 0.0]], dtype=np.float64),
        "resolv": np.asarray([True]),
        "mask_soiltile": np.asarray([1.0], dtype=np.float64),
        "kfact_root": np.asarray([[1.0, 1.1, 1.2, 1.3]], dtype=np.float64),
        "ks": np.asarray([PEAT_KS], dtype=np.float64),
        "kfact": np.asarray(tables.kfact),
        "reinf_slope": np.asarray([0.25], dtype=np.float64),
        "peat_hydro": True,
        "soil_tile_index": 4,
        "peat_tables": tables,
        "ok_freeze_cwrr": True,
        "temp_hydro": np.asarray([[274.0, 274.0, 274.0, 274.0]], dtype=np.float64),
    }

    result = hydrol_soil_tile_explicit_step(**inputs)

    assert result.coef_before_infilt is not None
    assert result.coef_after_infilt is not None
    assert np.asarray(result.infilt.mc)[0, 0] > inputs["mc"][0, 0]
    np.testing.assert_allclose(
        np.asarray(result.coef_before_infilt.mc_used[0]),
        PEAT_MCR + (1.0 - inputs["profil_froz"][0]) * np.maximum(inputs["mc"][0] - PEAT_MCR, 0.0),
    )
