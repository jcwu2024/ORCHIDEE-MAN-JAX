from __future__ import annotations

import numpy as np

from research.daily_coarse_graining.hydrology_canopy_boundary_audit import (
    CanopyDebtArrays,
    ReconstructionComparison,
    reconstruct_hydrology_boundary_fields,
    summarize_canopy_debt,
    summarize_comparisons,
)


def test_reconstruction_comparison_panel_uses_field_tolerance() -> None:
    first = ReconstructionComparison(
        reconstructed=np.asarray([1.0, 1.0e6 + 5.0e-7]),
        target=np.asarray([1.0 + 5.0e-11, 1.0e6]),
        defined=np.asarray([True, True]),
    )
    second = ReconstructionComparison(
        reconstructed=np.asarray([2.0]),
        target=np.asarray([2.0]),
        defined=np.asarray([True]),
    )

    summary = summarize_comparisons((first, second))

    assert summary["point_year_count"] == 2
    assert summary["value_count"] == 3
    assert summary["passed"]


def test_canopy_debt_uses_daily_wet_evaporation_for_new_debt() -> None:
    record = CanopyDebtArrays(
        start_debt=np.asarray([0.0, 0.4, 0.2, 0.0]),
        end_storage=np.asarray([0.3, 0.0, 0.0, 0.0]),
        end_debt=np.asarray([0.0, 0.6, 0.1, 0.5]),
        capacity=np.asarray([0.5, 0.5, 0.5, 0.5]),
        wet_canopy_evaporation=np.asarray([0.1, 0.3, 0.0, 0.4]),
        defined=np.ones(4, dtype=np.bool_),
    )

    summary = summarize_canopy_debt((record,))

    assert summary["end_debt_positive_count"] == 3
    assert summary["end_storage_positive_count"] == 1
    assert summary["end_zero_count"] == 0
    assert summary["debt_incoming_exceeds_daily_wet_evaporation_count"] == 1
    assert (
        summary[
            "end_debt_exceeds_start_debt_plus_daily_wet_evaporation_count"
        ]
        == 1
    )
    assert not summary["daily_wet_evaporation_is_valid_debt_incoming_scale"]
    assert not summary[
        "start_debt_plus_daily_wet_evaporation_is_valid_end_debt_scale"
    ]


def test_synthetic_hydrology_boundary_shapes_and_identities() -> None:
    batch = 2
    nslm = 3
    nvm = 2
    nstm = 2
    mc = np.full((batch, nslm, nstm), 0.3, dtype=np.float64)
    mcl = np.full_like(mc, 0.2)
    nroot = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.6, 0.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
        ],
        dtype=np.float64,
    )
    veget_max = np.asarray([[0.2, 0.8], [0.3, 0.7]], dtype=np.float64)
    soiltile = np.asarray([[0.2, 0.8], [0.3, 0.7]], dtype=np.float64)
    evap_bare_lim_ns = np.asarray([[0.5, 0.25], [0.1, 0.4]], dtype=np.float64)

    fields, metadata = reconstruct_hydrology_boundary_fields(
        mc=mc,
        mcl=mcl,
        start_nroot=nroot,
        end_nroot=nroot,
        veget=veget_max,
        veget_max=veget_max,
        soiltile=soiltile,
        pref_soil_veg=np.asarray([1, 2], dtype=np.int32),
        dz_mm=np.asarray([10.0, 30.0, 60.0]),
        zz_mm=np.asarray([10.0, 40.0, 100.0]),
        diaglev_m=np.asarray([0.01, 0.04, 0.1]),
        njsc=np.asarray([6, 6], dtype=np.int32),
        is_tree=np.asarray([False, True]),
        end_stempdiag=np.asarray([[280.0, 279.0, 278.0], [275.0, 274.0, 273.0]]),
        end_snow=np.zeros(batch),
        end_snowdz=np.zeros((batch, 3)),
        evap_bare_lim_ns=evap_bare_lim_ns,
    )

    assert fields["from_true_end_nroot.us"].shape == (batch, nvm, nstm, nslm)
    assert fields["from_true_end_nroot.humrelv"].shape == (batch, nvm, nstm)
    assert fields["from_end_stempdiag.temp_hydro"].shape == (batch, nslm)
    np.testing.assert_allclose(
        fields["from_true_end_nroot.humrel"],
        np.sum(fields["from_true_end_nroot.humrelv"], axis=2),
    )
    np.testing.assert_allclose(
        fields["from_tile_beta.evap_bare_lim"],
        np.sum(evap_bare_lim_ns * soiltile, axis=1),
    )
    assert metadata["forcing_records_materialized"] == 0
    assert metadata["subdaily_state_scan_steps"] == 0
