from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.domain import read_domain_grid
from jax_orchidee.driver.init import (
    initialize_imposed_veget_max,
    initialize_imposed_vegetation_state,
    parse_run_def,
    read_run_scalars,
    reference_run_def_path,
    soil_tiles_from_veget_max,
)


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def test_reference_run_def_simple_key_value_parser_reads_template_file():
    run_def_path = reference_run_def_path(CONFIG)
    values = parse_run_def(run_def_path)

    assert values["READ_SALINITY"] == "TRUE"
    assert values["READ_TIDE"] == "TRUE"
    assert values["IMPOSE_VEG"] == "n"
    assert values["ATM_CO2"] == "285"


def test_run_scalars_use_audited_paper_case_overrides_for_pft14():
    scalars = read_run_scalars(CONFIG)

    assert scalars.impose_veg is True
    assert scalars.nvm == 14
    assert scalars.nstm == 6
    assert scalars.raw_run_def["IMPOSE_VEG"] == "n"

    expected_vegmax = np.zeros(14, dtype=np.float64)
    expected_vegmax[13] = 1.0
    assert np.allclose(scalars.sechiba_vegmax, expected_vegmax)

    assert scalars.pft_to_mtc.shape == (14,)
    assert scalars.pft_to_mtc[13] == 2
    assert scalars.pref_soil_veg.shape == (14,)
    assert scalars.pref_soil_veg[13] == 4
    assert scalars.natural.shape == (14,)
    assert bool(scalars.natural[13]) is True
    assert bool(scalars.is_peat[13]) is True
    assert bool(scalars.is_c4[13]) is False
    assert np.allclose(scalars.ext_coeff_vegetfrac, np.ones(14))
    assert scalars.slowproc_height.shape == (14,)
    assert scalars.slowproc_height[13] == 30.0
    expected_z0_over_height = np.full(14, 0.0625)
    expected_z0_over_height[0] = 0.0
    assert np.allclose(scalars.z0_over_height, expected_z0_over_height)
    assert scalars.z0_over_height[0] == 0.0
    assert np.allclose(scalars.ratio_z0m_z0h, np.ones(14))
    assert scalars.nleafages == 4
    assert scalars.read_lai is False
    assert scalars.ok_stomate is True


def test_imposed_pft14_veget_max_keeps_npts_nvm_shape():
    grid = read_domain_grid(CONFIG, year=1961)
    scalars = read_run_scalars(CONFIG)

    veget_max = initialize_imposed_veget_max(scalars, npts=grid.nbindex)

    assert veget_max.shape == (1, 14)
    assert np.allclose(veget_max[:, :13], 0.0)
    assert np.allclose(veget_max[:, 13], 1.0)


def test_soil_tiles_from_pft14_cover_use_pref_soil_veg_tile4():
    grid = read_domain_grid(CONFIG, year=1961)
    scalars = read_run_scalars(CONFIG)
    veget_max = initialize_imposed_veget_max(scalars, npts=grid.nbindex)

    soiltile, normalized_veget_max, totfrac_nobio = soil_tiles_from_veget_max(
        veget_max,
        scalars.pref_soil_veg,
        nstm=scalars.nstm,
    )

    expected_soiltile = np.zeros((1, 6), dtype=np.float64)
    expected_soiltile[0, 3] = 1.0
    assert np.allclose(normalized_veget_max, veget_max)
    assert np.allclose(totfrac_nobio, [0.0])
    assert np.allclose(soiltile, expected_soiltile)


def test_imposed_vegetation_state_derives_cold_start_veget_from_val_exp_lai():
    grid = read_domain_grid(CONFIG, year=1961)
    scalars = read_run_scalars(CONFIG)

    state = initialize_imposed_vegetation_state(scalars, npts=grid.nbindex)

    assert state.veget_max.shape == (1, 14)
    assert np.allclose(state.veget_max[0, 13], 1.0)
    assert np.allclose(state.frac_nobio, [[0.0]])
    assert np.allclose(state.totfrac_nobio, [0.0])
    assert np.allclose(state.soiltile, [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    assert state.veget.shape == (1, 14)
    assert np.allclose(state.veget[0, :13], 0.0)
    assert np.allclose(state.veget[0, 13], 1.0)
