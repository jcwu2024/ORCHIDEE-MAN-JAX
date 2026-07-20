from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.carbon_kernels import ICARBON, ILEAF, NPARTS
from jax_orchidee.stomate.modelout import compute_modelout_from_fields
from jax_orchidee.stomate.reference import (
    HISTORY_VALIDATION_FIELDS,
    RESTART_STATE_FIELDS,
    find_stomate_reference_files,
    inventory_netcdf,
    pack_history_modelout_fields,
    read_history_pools_as_biomass,
    read_restart_biomass_carbon,
    read_restart_maint_resp_part,
    read_restart_pft_field,
    variable_presence,
)


def test_reference_run_contains_expected_stomate_restart_and_history_files():
    files = find_stomate_reference_files(ROOT)

    assert files.start.name == "stomate_start.nc"
    assert files.restart.name == "stomate_restart.nc"
    assert len(files.histories) == 50
    assert files.histories[0].name == "stomate_history_1961.nc"
    assert files.histories[-1].name == "stomate_history_2010.nc"


def test_restart_inventory_marks_state_truth_and_missing_process_outputs():
    files = find_stomate_reference_files(ROOT)
    inventory = inventory_netcdf(files.restart)
    presence = variable_presence(files.restart, RESTART_STATE_FIELDS + ("f_alloc", "bm_alloc", "lai"))

    assert inventory["biomass"].dimensions == ("time", "m_a", "l_a", "z_a", "y", "x")
    assert inventory["biomass"].shape == (1, 1, NPARTS, 14, 1, 1)
    assert inventory["maint_resp"].dimensions == ("time", "l_a", "z_a", "y", "x")
    assert inventory["maint_resp"].shape == (1, NPARTS, 14, 1, 1)

    for name in RESTART_STATE_FIELDS:
        assert presence[name], name
    assert not presence["f_alloc"]
    assert not presence["bm_alloc"]
    assert not presence["lai"]


def test_history_inventory_marks_modelout_validation_truth_not_full_restart_state():
    files = find_stomate_reference_files(ROOT)
    history = files.histories[0]
    inventory = inventory_netcdf(history)
    presence = variable_presence(history, HISTORY_VALIDATION_FIELDS + ("biomass", "f_alloc", "bm_alloc", "PFTpresent"))

    assert inventory["LEAF_M"].dimensions == ("time_counter", "veget", "lat", "lon")
    assert inventory["LEAF_M"].shape == (1, 14, 1, 1)
    assert inventory["GPP"].shape == (1, 14, 1, 1)
    assert inventory["BM_ALLOC_LEAF"].shape == (1, 14, 1, 1)

    for name in HISTORY_VALIDATION_FIELDS:
        assert presence[name], name
    assert not presence["biomass"]
    assert not presence["f_alloc"]
    assert not presence["bm_alloc"]
    assert not presence["PFTpresent"]


def test_pack_history_modelout_fields_matches_formula_on_real_reference_history():
    files = find_stomate_reference_files(ROOT)
    fields = pack_history_modelout_fields(files.histories[0])

    result = compute_modelout_from_fields(fields)

    expected_agb = 0.02 * (
        fields["LEAF_M"]
        + fields["SAP_M_AB"]
        + fields["HEART_M_AB"]
        + fields["AGR_SAP_ST_M"]
        + fields["AGR_HRT_ST_M"]
        + fields["AGR_SAP_PN_M"]
        + fields["AGR_HRT_PN_M"]
    )
    expected_bgb = 0.02 * (fields["SAP_M_BE"] + fields["HEART_M_BE"] + fields["ROOT_M"])
    assert np.allclose(np.asarray(result.AGB_model), expected_agb)
    assert np.allclose(np.asarray(result.BGB_model), expected_bgb)
    assert np.allclose(np.asarray(result.GPP_model), fields["GPP"])
    assert np.allclose(np.asarray(result.NPP_model), fields["NPP"])


def test_restart_axis_adapters_preserve_pft_pool_axes_for_kernel_readiness():
    files = find_stomate_reference_files(ROOT)

    biomass = read_restart_biomass_carbon(files.restart)
    maint_resp = read_restart_maint_resp_part(files.restart)
    pft_present = read_restart_pft_field(files.restart, "PFTpresent")
    sla_calc = read_restart_pft_field(files.restart, "sla_calc")

    assert biomass.shape == (1, 14, NPARTS, 1)
    assert maint_resp.shape == (1, 14, NPARTS)
    assert pft_present.shape == (1, 14)
    assert sla_calc.shape == (1, 14)
    history_pools = read_history_pools_as_biomass(files.histories[-1])
    assert history_pools.shape == (1, 14, NPARTS, 1, 1)
    assert biomass[0, :, ILEAF, ICARBON].shape == history_pools[0, :, ILEAF, 0, 0].shape
    assert np.isfinite(biomass[0, :, ILEAF, ICARBON]).all()
    assert np.asarray(pft_present).dtype != object
