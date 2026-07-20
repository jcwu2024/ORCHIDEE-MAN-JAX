from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    PEAT_MCR,
    PEAT_MCS,
    build_mineral_cwrr_tables,
    build_peat_cwrr_tables,
    hydrol_soil_all_tiles_explicit_step,
)


def _base_inputs(nstm=6):
    npts = 1
    nslm = 4
    dz_mm = np.asarray([0.0, 3.0, 6.0, 6.0], dtype=np.float64)
    zz_mm = np.asarray([1.0, 4.0, 10.0, 20.0], dtype=np.float64)
    mc = np.zeros((npts, nslm, nstm), dtype=np.float64)
    for jst in range(nstm):
        mc[:, :, jst] = np.asarray([[0.30, 0.32, 0.34, 0.36]], dtype=np.float64) + 0.01 * jst
    return {
        "water2infilt": np.full((npts, nstm), 0.05, dtype=np.float64),
        "ae_ns": np.full((npts, nstm), 0.01, dtype=np.float64),
        "subsinksoil": np.asarray([0.0], dtype=np.float64),
        "precisol_ns": np.full((npts, nstm), 0.20, dtype=np.float64),
        "reinfiltration_soil": np.zeros((npts, nstm), dtype=np.float64),
        "is_crop_soil": False,
        "mc": mc,
        "mcl": mc.copy(),
        "profil_froz": np.zeros_like(mc),
        "mcr": np.asarray([0.057], dtype=np.float64),
        "mcs": np.asarray([0.50], dtype=np.float64),
        "dz_mm": dz_mm,
        "dt_days": 1.0 / 48.0,
        "free_drain_coef": np.ones((npts, nstm), dtype=np.float64),
        "rootsink": np.zeros((npts, nslm, nstm), dtype=np.float64),
        "resolv": np.ones((npts, nstm), dtype=bool),
        "mask_soiltile": np.ones((npts, nstm), dtype=np.float64),
        "kfact_root": np.ones((npts, nslm, nstm), dtype=np.float64),
        "ks": np.asarray([10.0], dtype=np.float64),
        "kfact": np.ones(nslm, dtype=np.float64),
        "soiltile": np.asarray([[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]], dtype=np.float64),
        "tmc": np.zeros((npts, nstm), dtype=np.float64),
        "njsc": np.asarray([2], dtype=np.int32),
        "reinf_slope": np.asarray([0.25], dtype=np.float64),
        "zz_mm": zz_mm,
        "zmaxh_m": 2.0,
        "zwt_force": np.full((npts, nstm), 1.0e20, dtype=np.float64),
    }


def test_hydrol_soil_all_tiles_explicit_step_runs_full_tile_loop_with_mineral_tables():
    inputs = _base_inputs()
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)

    result = hydrol_soil_all_tiles_explicit_step(
        **inputs,
        mineral_tables=tables,
        peat_hydro=False,
        tides=False,
        ok_ru2peat=False,
        ok_wt_ab=False,
    )

    assert len(result.tile_results) == 6
    assert result.mc.shape == inputs["mc"].shape
    assert result.qflux.shape == inputs["mc"].shape
    assert result.ru_ns.shape == (1, 6)
    assert result.dr_ns.shape == (1, 6)
    assert result.water2infilt.shape == (1, 6)
    assert np.all(np.asarray(result.tmc) >= np.asarray(result.tmc_soil))
    assert np.all(np.asarray(result.wtd_ns) == 1.0e20)
    assert np.all(np.asarray(result.runoff2peat) == 0.0)

    for jst, tile in enumerate(result.tile_results):
        np.testing.assert_allclose(np.asarray(result.mc[:, :, jst]), np.asarray(tile.solve.mc_final))
        np.testing.assert_allclose(np.asarray(result.dr_ns[:, jst]), np.asarray(tile.solve.dr_ns_final))
        np.testing.assert_allclose(np.asarray(result.qflux[:, -1, jst]), np.asarray(tile.solve.dr_ns_final))


def test_hydrol_soil_all_tiles_explicit_step_runs_pft14_peat_tide_routing():
    inputs = _base_inputs()
    z_m = inputs["zz_mm"] / 1000.0
    mineral_tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=z_m)
    peat_tables = build_peat_cwrr_tables(z_m)
    inputs["mc"][:, :, 3] = np.asarray([[0.30, 0.35, 0.40, 0.45]], dtype=np.float64)
    inputs["mcl"] = inputs["mc"].copy()
    inputs["mcr"] = np.asarray([PEAT_MCR], dtype=np.float64)
    inputs["mcs"] = np.asarray([PEAT_MCS], dtype=np.float64)
    inputs["soiltile"] = np.asarray([[0.2, 0.3, 0.1, 0.25, 0.05, 0.10]], dtype=np.float64)
    inputs["precisol_ns"][:, :3] = np.asarray([[0.5, 0.4, 0.3]], dtype=np.float64)

    result = hydrol_soil_all_tiles_explicit_step(
        **inputs,
        mineral_tables=mineral_tables,
        peat_tables=peat_tables,
        peat_hydro=True,
        tides=True,
        ok_ru2peat=True,
        ok_wt_ab=True,
        wtp_tide=0.7,
        dwtp_tide=0.2,
    )

    assert result.tile_results[3].coef_before_infilt is not None
    assert np.asarray(result.run2man)[0] >= 0.0
    assert np.asarray(result.run2peat)[0] >= 0.0
    assert np.asarray(result.wt_ab_tide)[0] >= 0.0
    assert np.any(np.asarray(result.runoff2peat[:, :3]) >= 0.0)
    assert np.asarray(result.water2infilt[0, 3]) >= 0.0
    assert np.asarray(result.water2infilt[0, 5]) >= 0.0
