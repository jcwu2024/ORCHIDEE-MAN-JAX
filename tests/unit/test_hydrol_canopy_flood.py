from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import hydrol_canop_interception, hydrol_flood_reservoir  # noqa: E402


def test_hydrol_canop_interception_matches_fortran_update_order():
    precip_rain = np.asarray([10.0, 4.0], dtype=np.float64)
    vevapwet = np.asarray([[0.0, 0.2, 0.1], [0.0, 0.05, 0.02]], dtype=np.float64)
    veget_max = np.asarray([[0.2, 0.5, 0.3], [0.1, 0.2, 0.0]], dtype=np.float64)
    veget = np.asarray([[0.2, 0.4, 0.1], [0.1, 0.15, 0.0]], dtype=np.float64)
    qsintmax = np.asarray([[0.0, 1.0, 0.5], [0.0, 0.2, 0.1]], dtype=np.float64)
    qsintveg = np.asarray([[0.0, 0.8, 0.3], [0.0, 0.1, 0.0]], dtype=np.float64)
    tot_melt = np.asarray([0.6, 1.2], dtype=np.float64)
    vegtot = np.sum(veget_max, axis=1)
    throughfall_by_pft = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)

    result = hydrol_canop_interception(
        precip_rain=precip_rain,
        vevapwet=vevapwet,
        veget_max=veget_max,
        veget=veget,
        qsintmax=qsintmax,
        qsintveg=qsintveg,
        tot_melt=tot_melt,
        vegtot=vegtot,
        throughfall_by_pft=throughfall_by_pft,
    )

    qs_work = qsintveg.copy()
    qs_work[:, 1:] -= vevapwet[:, 1:]
    precip2canopy = np.zeros_like(qsintveg)
    qs_work[:, 0] = 0.0
    for jv in range(1, qsintveg.shape[1]):
        addition = veget[:, jv] * (1.0 - throughfall_by_pft[jv]) * precip_rain
        qs_work[:, jv] += addition
        precip2canopy[:, jv] = addition

    expected_precisol = np.zeros_like(qsintveg)
    expected_precip2ground = np.zeros_like(qsintveg)
    expected_canopy2ground = np.zeros_like(qsintveg)
    expected_precisol[:, 0] = veget_max[:, 0] * precip_rain
    expected_precip2ground[:, 0] = expected_precisol[:, 0]
    zqsintvegnew = np.minimum(qs_work, qsintmax)
    for jv in range(1, qsintveg.shape[1]):
        expected_precisol[:, jv] = (
            veget[:, jv] * throughfall_by_pft[jv] * precip_rain
            + qs_work[:, jv]
            - zqsintvegnew[:, jv]
            + (veget_max[:, jv] - veget[:, jv]) * precip_rain
        )
        expected_precip2ground[:, jv] = (
            veget[:, jv] * throughfall_by_pft[jv] * precip_rain
            + (veget_max[:, jv] - veget[:, jv]) * precip_rain
        )
        expected_canopy2ground[:, jv] = np.maximum(
            expected_precisol[:, jv] - expected_precip2ground[:, jv],
            0.0,
        )
    melt_share = tot_melt[:, None] * veget_max / vegtot[:, None]
    expected_precisol += melt_share
    expected_precip2ground += melt_share
    qs_expected = qs_work.copy()
    qs_expected[:, 1:] = zqsintvegnew[:, 1:]

    np.testing.assert_allclose(np.asarray(result.qsintveg), qs_expected)
    np.testing.assert_allclose(np.asarray(result.precisol), expected_precisol)
    np.testing.assert_allclose(np.asarray(result.precip2canopy), precip2canopy)
    np.testing.assert_allclose(np.asarray(result.precip2ground), expected_precip2ground)
    np.testing.assert_allclose(np.asarray(result.canopy2ground), expected_canopy2ground)


def test_hydrol_flood_reservoir_matches_fortran_module_state_updates():
    vevapflo = np.asarray([3.0, 1.0], dtype=np.float64)
    flood_frac = np.asarray([0.25, 0.0], dtype=np.float64)
    flood_res = np.asarray([2.0, 5.0], dtype=np.float64)
    precisol = np.asarray([[1.0, 2.0, 3.0], [0.5, 0.25, 0.25]], dtype=np.float64)
    subsinksoil = np.asarray([0.1, 0.2], dtype=np.float64)

    result = hydrol_flood_reservoir(
        vevapflo=vevapflo,
        flood_frac=flood_frac,
        flood_res=flood_res,
        precisol=precisol,
        subsinksoil=subsinksoil,
    )

    temp = np.minimum(flood_res, vevapflo)
    expected_flood_res = flood_res - temp
    expected_subsinksoil = subsinksoil + vevapflo - temp
    expected_vevapflo = temp
    expected_floodout = expected_vevapflo - flood_frac * np.sum(precisol, axis=1)
    expected_precisol = precisol * (1.0 - flood_frac[:, None])

    np.testing.assert_allclose(np.asarray(result.vevapflo), expected_vevapflo)
    np.testing.assert_allclose(np.asarray(result.flood_res), expected_flood_res)
    np.testing.assert_allclose(np.asarray(result.subsinksoil), expected_subsinksoil)
    np.testing.assert_allclose(np.asarray(result.floodout), expected_floodout)
    np.testing.assert_allclose(np.asarray(result.precisol), expected_precisol)
