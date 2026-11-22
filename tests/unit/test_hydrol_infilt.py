from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import hydrol_soil_infilt_explicit, hydrol_total_moisture_content  # noqa: E402


def test_hydrol_soil_infilt_explicit_matches_mineral_fortran_loop():
    mc = np.asarray([[0.20, 0.25, 0.30, 0.35]], dtype=np.float64)
    flux_infilt = np.asarray([0.80], dtype=np.float64)
    k = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=np.float64)
    dz = np.asarray([0.0, 2.0, 4.0, 8.0], dtype=np.float64)
    dt_days = 1.0 / 48.0
    mcs = np.asarray([0.50], dtype=np.float64)
    ks = np.asarray([10.0], dtype=np.float64)
    kfact = np.asarray([1.0, 0.8, 0.6, 0.4], dtype=np.float64)
    kfact_root = np.asarray([[1.0, 1.1, 1.2, 1.3]], dtype=np.float64)

    result = hydrol_soil_infilt_explicit(
        mc=mc,
        flux_infilt=flux_infilt,
        k=k,
        dz_mm=dz,
        dt_days=dt_days,
        mcs=mcs,
        ks=ks,
        kfact=kfact,
        kfact_root=kfact_root,
        check_cwrr2=True,
    )

    expected_mc = mc.copy()
    wat_inf_pot = np.maximum((mcs - expected_mc[:, 0]) * dz[1] / 2.0, 0.0)
    wat_inf_first = np.minimum(wat_inf_pot, flux_infilt)
    expected_mc[:, 0] += wat_inf_first * 2.0 / dz[1]
    dt_tmp = np.full((1,), dt_days, dtype=np.float64)
    infilt_tot = wat_inf_first.copy()
    flux_tmp = (flux_infilt - wat_inf_first) / dt_tmp
    for jsl in range(1, expected_mc.shape[1] - 1):
        k_m = (k[:, jsl] + ks * kfact[jsl - 1] * kfact_root[:, jsl]) / 2.0
        infilt_tmp = k_m * (1.0 - np.exp(-flux_tmp / k_m))
        wat_inf_pot = np.maximum((mcs - expected_mc[:, jsl]) * (dz[jsl] + dz[jsl + 1]) / 2.0, 0.0)
        dt_inf = np.where(infilt_tmp > 1.0e-8, np.minimum(wat_inf_pot / infilt_tmp, dt_tmp), dt_tmp)
        dt_inf = np.where(
            (infilt_tmp > 1.0e-8) & (dt_inf * infilt_tmp > flux_infilt - infilt_tot),
            np.maximum(flux_infilt - infilt_tot, 0.0) / infilt_tmp,
            dt_inf,
        )
        wat_inf = dt_inf * infilt_tmp
        expected_mc[:, jsl] += wat_inf * 2.0 / (dz[jsl] + dz[jsl + 1])
        dt_tmp -= dt_inf
        infilt_tot += infilt_tmp * dt_inf

    np.testing.assert_allclose(np.asarray(result.mc), expected_mc)
    np.testing.assert_allclose(np.asarray(result.wat_inf_first), wat_inf_first)
    np.testing.assert_allclose(np.asarray(result.infilt_tot), infilt_tot)
    np.testing.assert_allclose(np.asarray(result.qinfilt), infilt_tot)
    np.testing.assert_allclose(np.asarray(result.ru_infilt), flux_infilt - infilt_tot)
    expected_check = hydrol_total_moisture_content(expected_mc, dz) - (
        hydrol_total_moisture_content(mc, dz) + infilt_tot
    )
    np.testing.assert_allclose(np.asarray(result.check), np.asarray(expected_check))


@pytest.mark.parametrize(
    ("layout", "selected"),
    [("point", [.1, .4, .7]), ("texture", [.3, .1, .2]), (None, [.3, .1, .2])],
    ids=["point", "texture", "legacy-texture"],
)
def test_hydrol_soil_infilt_square_kfact_uses_the_declared_layout(layout, selected):
    profile = np.asarray([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
    inputs = dict(
        mc=np.full((3, 3), .2), flux_infilt=np.full(3, 20.), k=np.full((3, 3), .01),
        dz_mm=np.full(3, 100.), dt_days=.02, mcs=np.full(3, .5), ks=1.,
        kfact=profile, kfact_root=np.ones((3, 3)), njsc=np.array([3, 1, 2]),
    )
    options = {"kfact_layout": layout} if layout is not None else {}

    # The first layer takes 15 mm. Fortran line 7520 sets the rate on the only interior layer.
    k_m = (.01 + np.asarray(selected)) / 2
    interior_mm = .02 * k_m * (1 - np.exp(-250. / k_m))
    expected_mc = inputs["mc"].copy()
    expected_mc[:, 0] = .5
    expected_mc[:, 1] += interior_mm / 100.
    compiled = jax.jit(hydrol_soil_infilt_explicit, static_argnames=("kfact_layout",))
    for kernel in (hydrol_soil_infilt_explicit, compiled):
        result = kernel(**inputs, **options)
        np.testing.assert_allclose(result.qinfilt, 15. + interior_mm, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.ru_infilt, 5. - interior_mm, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(result.mc, expected_mc, rtol=1e-12, atol=1e-14)


def test_hydrol_soil_infilt_explicit_uses_freeze_and_peat_branches():
    mc = np.asarray([[0.20, 0.25, 0.30]], dtype=np.float64)
    flux_infilt = np.asarray([1.00], dtype=np.float64)
    k = np.asarray([[4.0, 0.5, 2.0]], dtype=np.float64)
    dz = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)
    kfact_root = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64)

    peat = hydrol_soil_infilt_explicit(
        mc=mc,
        flux_infilt=flux_infilt,
        k=k,
        dz_mm=dz,
        dt_days=1.0 / 48.0,
        mcs=np.asarray([0.50], dtype=np.float64),
        ks=np.asarray([10.0], dtype=np.float64),
        kfact=np.ones(3, dtype=np.float64),
        kfact_root=kfact_root,
        soil_tile_index=4,
        peat_hydro=True,
        mcs_peat=0.80,
        ks_peat=20.0,
        kfact_peat=np.asarray([1.0, 0.5, 0.25], dtype=np.float64),
        ok_freeze_cwrr=True,
        temp_hydro=np.asarray([[274.0, 270.0, 274.0]], dtype=np.float64),
    )
    no_freeze = hydrol_soil_infilt_explicit(
        mc=mc,
        flux_infilt=flux_infilt,
        k=k,
        dz_mm=dz,
        dt_days=1.0 / 48.0,
        mcs=np.asarray([0.50], dtype=np.float64),
        ks=np.asarray([10.0], dtype=np.float64),
        kfact=np.ones(3, dtype=np.float64),
        kfact_root=kfact_root,
        soil_tile_index=4,
        peat_hydro=True,
        mcs_peat=0.80,
        ks_peat=20.0,
        kfact_peat=np.asarray([1.0, 0.5, 0.25], dtype=np.float64),
        ok_freeze_cwrr=False,
    )

    assert np.asarray(peat.mc)[0, 0] > mc[0, 0]
    assert np.asarray(peat.mc)[0, 1] < np.asarray(no_freeze.mc)[0, 1]
