from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import hydrol_split_soil_fluxes  # noqa: E402


def test_hydrol_split_soil_fluxes_match_fortran_formula_blocks():
    veget_max = np.asarray(
        [
            [0.10, 0.30, 0.20, 0.00],
            [0.05, 0.00, 0.30, 0.15],
        ],
        dtype=np.float64,
    )
    soiltile = np.asarray([[0.60, 0.40], [0.50, 0.50]], dtype=np.float64)
    vegtot = np.sum(veget_max, axis=1)
    pref_soil_veg = np.asarray([1, 1, 2, 2], dtype=np.int32)
    precisol = np.asarray([[0.2, 0.6, 0.4, 0.0], [0.1, 0.0, 0.5, 0.2]], dtype=np.float64)
    vevapnu = np.asarray([0.30, 0.24], dtype=np.float64)
    vevapnu_pft = np.asarray([[0.01, 0.02, 0.03, 0.00], [0.02, 0.00, 0.04, 0.01]], dtype=np.float64)
    transpir = np.asarray([[0.0, 0.09, 0.06, 0.0], [0.0, 0.0, 0.10, 0.05]], dtype=np.float64)
    humrel = np.asarray([[0.0, 0.45, 0.30, 0.0], [0.0, 0.0, 0.50, 0.25]], dtype=np.float64)
    humrelv = np.zeros((2, 4, 2), dtype=np.float64)
    humrelv[:, :, 0] = np.asarray([[0.0, 0.45, 0.00, 0.0], [0.0, 0.0, 0.00, 0.0]])
    humrelv[:, :, 1] = np.asarray([[0.0, 0.00, 0.30, 0.0], [0.0, 0.0, 0.50, 0.25]])
    us = np.zeros((2, 4, 2, 3), dtype=np.float64)
    us[0, 1, 0, :] = [0.0, 0.18, 0.27]
    us[0, 2, 1, :] = [0.0, 0.12, 0.18]
    us[1, 2, 1, :] = [0.0, 0.20, 0.30]
    us[1, 3, 1, :] = [0.0, 0.10, 0.15]
    ae_ns = np.asarray([[0.05, 0.10], [0.00, 0.00]], dtype=np.float64)
    evap_bare_lim = np.asarray([0.20, 0.0], dtype=np.float64)
    evap_bare_lim_ns = np.asarray([[0.12, 0.08], [0.00, 0.00]], dtype=np.float64)
    frac_bare_ns = np.asarray([[0.3, 0.2], [0.2, 0.4]], dtype=np.float64)
    tot_bare_soil = np.asarray([0.50, 0.60], dtype=np.float64)
    vegetmax_soil = np.zeros((2, 4, 2), dtype=np.float64)
    for ji in range(2):
        for jv, jst_fortran in enumerate(pref_soil_veg):
            jst = jst_fortran - 1
            if soiltile[ji, jst] > 0:
                vegetmax_soil[ji, jv, jst] = veget_max[ji, jv] / soiltile[ji, jst]

    result = hydrol_split_soil_fluxes(
        veget_max=veget_max,
        soiltile=soiltile,
        precisol=precisol,
        vevapnu=vevapnu,
        vevapnu_pft=vevapnu_pft,
        transpir=transpir,
        humrel=humrel,
        humrelv=humrelv,
        us=us,
        ae_ns=ae_ns,
        evap_bare_lim=evap_bare_lim,
        evap_bare_lim_ns=evap_bare_lim_ns,
        frac_bare_ns=frac_bare_ns,
        tot_bare_soil=tot_bare_soil,
        vegetmax_soil=vegetmax_soil,
        vegtot=vegtot,
        pref_soil_veg=pref_soil_veg,
    )

    expected_precisol_ns = np.zeros_like(soiltile)
    expected_vevapnu_ns = np.zeros_like(soiltile)
    for jv, jst_fortran in enumerate(pref_soil_veg):
        jst = jst_fortran - 1
        denom = soiltile[:, jst] * vegtot
        active = (veget_max[:, jv] > 1.0e-8) & (denom > 1.0e-8)
        expected_precisol_ns[:, jst] += np.where(active, precisol[:, jv] / denom, 0.0)
        expected_vevapnu_ns += np.where(
            veget_max[:, jv, None] > 1.0e-8,
            vevapnu_pft[:, jv, None] * vegetmax_soil[:, jv, :] / vegtot[:, None] / np.where(
                veget_max[:, jv, None] > 1.0e-8,
                veget_max[:, jv, None],
                1.0,
            ),
            0.0,
        )

    expected_vevapnu_old = np.sum(ae_ns * soiltile * vegtot[:, None], axis=1)
    expected_ae = ae_ns.copy()
    for ji in range(2):
        for jst in range(2):
            if expected_vevapnu_old[ji] > 1.0e-8:
                if evap_bare_lim[ji] > 1.0e-8:
                    expected_ae[ji, jst] = vevapnu[ji] * evap_bare_lim_ns[ji, jst] / evap_bare_lim[ji]
                elif expected_ae[ji, jst] > 1000.0 * expected_vevapnu_old[ji]:
                    expected_ae[ji, jst] = 0.0
                else:
                    expected_ae[ji, jst] = expected_ae[ji, jst] * vevapnu[ji] / expected_vevapnu_old[ji]
            elif frac_bare_ns[ji, jst] > 1.0e-8:
                if evap_bare_lim[ji] > 1.0e-8:
                    expected_ae[ji, jst] = vevapnu[ji] * evap_bare_lim_ns[ji, jst] / evap_bare_lim[ji]
                elif tot_bare_soil[ji] > 1.0e-8:
                    expected_ae[ji, jst] = vevapnu[ji] * frac_bare_ns[ji, jst] / tot_bare_soil[ji]
                else:
                    expected_ae[ji, jst] = 0.0

    expected_tr = np.zeros_like(soiltile)
    expected_rootsink = np.zeros((2, 3, 2), dtype=np.float64)
    for jv, jst_fortran in enumerate(pref_soil_veg):
        jst = jst_fortran - 1
        denom = soiltile[:, jst] * vegtot
        active = (humrel[:, jv] > 1.0e-8) & (denom > 1.0e-8)
        expected_tr[:, jst] += np.where(
            active,
            transpir[:, jv] * (humrelv[:, jv, jst] / np.where(humrel[:, jv] > 1.0e-8, humrel[:, jv], 1.0)) / denom,
            0.0,
        )
        expected_rootsink[:, :, jst] += np.where(
            active[:, None],
            transpir[:, jv, None] * (us[:, jv, jst, :] / np.where(humrel[:, jv, None] > 1.0e-8, humrel[:, jv, None], 1.0))
            / denom[:, None],
            0.0,
        )

    np.testing.assert_allclose(np.asarray(result.precisol_ns), expected_precisol_ns)
    np.testing.assert_allclose(np.asarray(result.vevapnu_ns), expected_vevapnu_ns)
    np.testing.assert_allclose(np.asarray(result.vevapnu_old), expected_vevapnu_old)
    np.testing.assert_allclose(np.asarray(result.ae_ns), expected_ae)
    np.testing.assert_allclose(np.asarray(result.tr_ns), expected_tr)
    np.testing.assert_allclose(np.asarray(result.rootsink), expected_rootsink)
    np.testing.assert_allclose(
        np.sum(np.asarray(result.precisol_ns) * soiltile * vegtot[:, None], axis=1),
        np.sum(precisol, axis=1),
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(result.tr_ns) * soiltile * vegtot[:, None], axis=1),
        np.sum(transpir, axis=1),
    )
    np.testing.assert_allclose(np.sum(np.asarray(result.rootsink), axis=1), np.asarray(result.tr_ns))
