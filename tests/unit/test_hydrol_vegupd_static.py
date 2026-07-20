from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    PEAT_KS,
    USDA_KS,
    hydrol_kfact_root_from_vegupd,
    hydrol_split_soil_fluxes,
    hydrol_vegupd_static_state,
)


def test_hydrol_vegupd_static_state_matches_fortran_masks_and_bare_fractions():
    veget_max = np.asarray(
        [
            [0.10, 0.30, 0.20, 0.00],
            [0.00, 0.25, 0.00, 0.15],
        ],
        dtype=np.float64,
    )
    veget = np.asarray(
        [
            [0.10, 0.24, 0.05, 0.00],
            [0.00, 0.10, 0.00, 0.15],
        ],
        dtype=np.float64,
    )
    soiltile = np.asarray([[0.60, 0.40], [0.50, 0.00]], dtype=np.float64)
    vegtot = np.sum(veget_max, axis=1)
    pref_soil_veg = np.asarray([1, 1, 2, 2], dtype=np.int32)

    result = hydrol_vegupd_static_state(
        veget=veget,
        veget_max=veget_max,
        soiltile=soiltile,
        vegtot=vegtot,
        pref_soil_veg=pref_soil_veg,
    )

    expected_mask_soiltile = (soiltile > 1.0e-8).astype(np.float64)
    expected_mask_veget = (veget_max > 1.0e-8).astype(np.float64)
    expected_vegetmax_soil = np.zeros((2, 4, 2), dtype=np.float64)
    for ji in range(2):
        for jv, jst_fortran in enumerate(pref_soil_veg):
            jst = jst_fortran - 1
            if expected_mask_soiltile[ji, jst] > 0 and vegtot[ji] > 1.0e-8:
                expected_vegetmax_soil[ji, jv, jst] = veget_max[ji, jv] / soiltile[ji, jst]

    expected_frac_bare = np.zeros_like(veget)
    expected_frac_bare[:, 0] = np.where(veget_max[:, 0] > 1.0e-8, 1.0, 0.0)
    for jv in range(1, veget.shape[1]):
        safe_veget_max = np.where(veget_max[:, jv] > 1.0e-8, veget_max[:, jv], 1.0)
        expected_frac_bare[:, jv] = np.where(
            veget_max[:, jv] > 1.0e-8,
            1.0 - veget[:, jv] / safe_veget_max,
            0.0,
        )
    expected_frac_bare_ns = np.zeros_like(soiltile)
    for jst in range(soiltile.shape[1]):
        expected_frac_bare_ns[:, jst] = np.where(
            vegtot > 1.0e-8,
            np.sum(expected_vegetmax_soil[:, :, jst] * expected_frac_bare, axis=1) / vegtot,
            0.0,
        )

    np.testing.assert_allclose(np.asarray(result.mask_soiltile), expected_mask_soiltile)
    np.testing.assert_allclose(np.asarray(result.mask_veget), expected_mask_veget)
    np.testing.assert_allclose(np.asarray(result.vegetmax_soil), expected_vegetmax_soil)
    np.testing.assert_allclose(np.asarray(result.frac_bare), expected_frac_bare)
    np.testing.assert_allclose(np.asarray(result.frac_bare_ns), expected_frac_bare_ns)


def test_hydrol_vegupd_static_state_feeds_split_soil_without_trace_arrays():
    veget_max = np.asarray([[0.10, 0.30, 0.20, 0.00]], dtype=np.float64)
    veget = np.asarray([[0.10, 0.18, 0.05, 0.00]], dtype=np.float64)
    soiltile = np.asarray([[0.60, 0.40]], dtype=np.float64)
    vegtot = np.sum(veget_max, axis=1)
    pref_soil_veg = np.asarray([1, 1, 2, 2], dtype=np.int32)
    static = hydrol_vegupd_static_state(
        veget=veget,
        veget_max=veget_max,
        soiltile=soiltile,
        vegtot=vegtot,
        pref_soil_veg=pref_soil_veg,
    )

    result = hydrol_split_soil_fluxes(
        veget_max=veget_max,
        soiltile=soiltile,
        precisol=np.asarray([[0.2, 0.6, 0.4, 0.0]], dtype=np.float64),
        vevapnu=np.asarray([0.3], dtype=np.float64),
        vevapnu_pft=np.asarray([[0.01, 0.02, 0.03, 0.0]], dtype=np.float64),
        transpir=np.asarray([[0.0, 0.09, 0.06, 0.0]], dtype=np.float64),
        humrel=np.asarray([[0.0, 0.45, 0.30, 0.0]], dtype=np.float64),
        humrelv=np.asarray([[[0.0, 0.0], [0.45, 0.0], [0.0, 0.30], [0.0, 0.0]]], dtype=np.float64),
        us=np.asarray([[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.45], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.30]], [[0.0, 0.0], [0.0, 0.0]]]], dtype=np.float64),
        ae_ns=np.asarray([[0.0, 0.0]], dtype=np.float64),
        evap_bare_lim=np.asarray([0.0], dtype=np.float64),
        evap_bare_lim_ns=np.asarray([[0.0, 0.0]], dtype=np.float64),
        frac_bare_ns=static.frac_bare_ns,
        tot_bare_soil=np.sum(np.asarray(static.frac_bare) * veget_max, axis=1),
        vegetmax_soil=static.vegetmax_soil,
        vegtot=vegtot,
        pref_soil_veg=pref_soil_veg,
    )

    assert np.asarray(result.precisol_ns).shape == (1, 2)
    assert np.asarray(result.ae_ns).shape == (1, 2)
    assert np.asarray(result.rootsink).shape == (1, 2, 2)
    np.testing.assert_allclose(
        np.sum(np.asarray(result.precisol_ns) * soiltile * vegtot[:, None], axis=1),
        [1.2],
    )


def test_hydrol_kfact_root_from_vegupd_matches_source_loop_and_peat_tile_branch():
    veget_max = np.asarray([[0.0, 0.2, 0.3, 0.5]], dtype=np.float64)
    veget = veget_max.copy()
    soiltile = np.asarray([[0.2, 0.3, 0.0, 0.5]], dtype=np.float64)
    pref_soil_veg = np.asarray([1, 1, 2, 4], dtype=np.int32)
    vegupd = hydrol_vegupd_static_state(
        veget=veget,
        veget_max=veget_max,
        soiltile=soiltile,
        vegtot=np.sum(veget_max, axis=1),
        pref_soil_veg=pref_soil_veg,
    )
    humcste = np.asarray([0.0, 0.7, 0.8, 0.9], dtype=np.float64)
    zz_mm = np.asarray([100.0, 500.0, 1000.0], dtype=np.float64)

    result = np.asarray(
        hydrol_kfact_root_from_vegupd(
            vegetmax_soil=vegupd.vegetmax_soil,
            soiltile=soiltile,
            pref_soil_veg=pref_soil_veg,
            humcste=humcste,
            zz_mm=zz_mm,
            njsc=np.asarray([2], dtype=np.int32),
            peat_hydro=True,
        )
    )

    expected = np.ones((1, 3, 4), dtype=np.float64)
    max_ks = np.max(np.asarray(USDA_KS, dtype=np.float64))
    for jv in range(1, veget_max.shape[1]):
        jst_fortran = int(pref_soil_veg[jv])
        jst = jst_fortran - 1
        denom = PEAT_KS if jst_fortran in {4, 5} else USDA_KS[1]
        exponent = -np.asarray(vegupd.vegetmax_soil)[0, jv, jst] / 2.0 * (humcste[jv] * zz_mm / 1000.0 - 1.0) / 2.0
        expected[0, :, jst] *= np.maximum((max_ks / denom) ** exponent, 1.0)

    np.testing.assert_allclose(result, expected)
    assert np.any(result[0, :, 3] > 1.0)
