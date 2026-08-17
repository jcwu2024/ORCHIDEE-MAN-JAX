from __future__ import annotations

import numpy as np
import pytest

from research.daily_coarse_graining.thermal_coefficient_exact_reconstruction_panel import (
    THERMAL_COEFFICIENT_FIELDS,
    _compact_static_pft,
    _paper_vertical_grid,
    parse_selection,
    reconstruct_thermal_coefficient_fields,
)


def test_parse_selection_and_compact_static_pft():
    assert parse_selection("001.0-071.0:1961") == ("001.0-071.0", 1961)
    np.testing.assert_array_equal(_compact_static_pft(np.arange(14), (0, 13)), [0, 13])
    with pytest.raises(ValueError, match="LANDPOINT:YEAR"):
        parse_selection("001.0-071.0")


def test_thermal_reconstruction_returns_finite_source_fields():
    batch = 2
    nslm = 11
    ngrnd = 32
    nvm = 2
    nstm = 6
    ptn = np.broadcast_to(np.linspace(268.0, 280.0, ngrnd)[None, :, None], (batch, ngrnd, nvm)).copy()
    mc = np.full((batch, nslm, nstm), 0.25, dtype=np.float64)
    mcl = np.full_like(mc, 0.2)
    veget_max = np.asarray([[0.7, 0.3], [0.6, 0.4]], dtype=np.float64)
    frac_nobio = np.zeros((batch, 1), dtype=np.float64)
    grid = _paper_vertical_grid()
    fields, metadata = reconstruct_thermal_coefficient_fields(
        ptn=ptn,
        shum_ngrnd_permalong=np.full((batch, ngrnd, nvm), 0.5, dtype=np.float64),
        ref_soc=np.full((batch, ngrnd), 10.0, dtype=np.float64),
        mc=mc,
        mcl=mcl,
        veget_max=veget_max,
        frac_nobio=frac_nobio,
        lai=np.asarray([[0.0, 3.0], [0.0, 2.0]], dtype=np.float64),
        temp_sol=np.asarray([274.0, 275.0], dtype=np.float64),
        temp_sol_pft=np.asarray([[274.0, 274.5], [275.0, 275.5]], dtype=np.float64),
        snow=np.asarray([10.0, 5.0], dtype=np.float64),
        snow_nobio=np.zeros((batch, 1), dtype=np.float64),
        snowrho=np.full((batch, 3), 250.0, dtype=np.float64),
        snowtemp=np.full((batch, 3), 270.0, dtype=np.float64),
        snowdz=np.full((batch, 3), 0.02, dtype=np.float64),
        pressure_hpa=np.asarray([1000.0, 990.0]),
        njsc=np.asarray([6, 6], dtype=np.int32),
        hydrol_zz_mm=np.asarray(grid.znh) * 1000.0,
        hydrol_dz_mm=np.asarray(grid.dnh) * 1000.0,
        pref_soil_veg=np.asarray([1, 6], dtype=np.int32),
        ext_coeff_vegetfrac=np.asarray([0.0, 0.5], dtype=np.float64),
        model_interval_seconds=1800.0,
    )
    assert tuple(fields) == THERMAL_COEFFICIENT_FIELDS
    assert all(np.all(np.isfinite(np.asarray(value))) for value in fields.values())
    assert metadata["forcing_records_materialized"] == 0
    assert metadata["subdaily_state_scan_steps"] == 0
