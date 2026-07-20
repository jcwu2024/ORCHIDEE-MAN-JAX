from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.carbon_kernels import NPARTS
from jax_orchidee.stomate.modelout import compute_modelout_from_fields
from jax_orchidee.stomate.reference import (
    find_stomate_reference_files,
    pack_history_modelout_fields,
    read_restart_carbon_readiness,
)


def test_real_history_modelout_formula_closes_for_first_and_last_years():
    files = find_stomate_reference_files(ROOT)

    for history in (files.histories[0], files.histories[-1]):
        fields = pack_history_modelout_fields(history)
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


def test_restart_carbon_readiness_shapes_for_explicit_adapter_boundary():
    files = find_stomate_reference_files(ROOT)
    readiness = read_restart_carbon_readiness(files.restart)

    assert readiness.biomass.shape == (1, 14, NPARTS, 1)
    assert readiness.maint_resp_part.shape == (1, 14, NPARTS)
    assert readiness.gpp_daily.shape == (1, 14)
    assert readiness.npp_daily.shape == (1, 14)
    assert readiness.resp_maint.shape == (1, 14)
    assert readiness.resp_growth.shape == (1, 14)
    assert readiness.pft_present.shape == (1, 14)
    assert readiness.pft_present.dtype == np.bool_
    assert np.isfinite(readiness.biomass).all()

