from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.stomate.initialize import (
    STOMATE_INITIALIZE_PROVENANCE,
    StomateInitializationBoundaryError,
    StomateInitializationOwners,
    StomateInitializeDimensions,
    StomateInitializePFT14Switches,
    stomate_initialize_pft14_explicit,
    stomate_initialize_forcing_boundaries,
    stomate_var_init_carbon_owner,
)


def _state(npts=3, nvm=16, ndeep=2):
    veget_max = np.zeros((npts, nvm))
    veget_max[:, 0] = 0.2
    veget_max[:, 13] = np.linspace(0.2, 0.6, npts)
    return {
        "veget": veget_max * 0.5,
        "veget_max": veget_max,
        "totfrac_nobio": np.asarray([0.0, 0.25, 1.0])[:npts],
        "deepC_a": np.ones((npts, ndeep, nvm)),
        "deepC_s": np.ones((npts, ndeep, nvm)) * 2.0,
        "deepC_p": np.ones((npts, ndeep, nvm)) * 3.0,
        "dt_days_read": 2.0,
        "t2m_month": np.linspace(273.15, 275.15, npts),
        "assim_param": np.zeros((npts, nvm, 4)),
        "deadleaf_cover": np.linspace(0.0, 0.2, npts),
    }


def _owners(calls, initial):
    def owner(name, result):
        def run(**kwargs):
            calls.append(name)
            assert kwargs["position"] == len(calls)
            return result

        return run

    return StomateInitializationOwners(
        stomate_init=owner("stomate_init", {"allocation_marker": 1}),
        data=owner("data", {"continuous_pft_parameter": np.arange(initial["veget"].shape[1]) + 0.25}),
        readstart=owner("readstart", {"entry_state": initial, "report": "not state"}),
        season=owner("season", {"season_marker": 1}),
        carbon=owner(
            "carbon",
            {"assim_param": initial["assim_param"], "deadleaf_cover": initial["deadleaf_cover"]},
        ),
    )


def _run(*, npts=3, nvm=16, switches=StomateInitializePFT14Switches()):
    calls = []
    initial = _state(npts=npts, nvm=nvm)
    owner_inputs = {
        name: {"position": position}
        for position, name in enumerate(("stomate_init", "data", "readstart", "season", "carbon"), 1)
    }
    result = stomate_initialize_pft14_explicit(
        dimensions=StomateInitializeDimensions(npts=npts, nvm=nvm),
        owner_inputs=owner_inputs,
        owners=_owners(calls, initial),
        dt_days=1.0,
        dt_sechiba_seconds=1800.0,
        max_dt_days=10.0,
        switches=switches,
    )
    return result, calls, initial


def test_composes_existing_owners_in_source_order_and_preserves_continuous_parameters():
    result, calls, initial = _run()

    assert calls == ["stomate_init", "data", "readstart", "season", "carbon"]
    assert result.process_order.index("stomate_init") < result.process_order.index("data")
    assert result.process_order.index("data") < result.process_order.index("flux_zero")
    assert result.process_order.index("readstart") < result.process_order.index("season")
    assert result.process_order.index("vegetation_cover") < result.process_order.index("carbon")
    np.testing.assert_allclose(result.state["continuous_pft_parameter"], np.arange(16) + 0.25)
    np.testing.assert_allclose(result.state["soilc_total"], 6.0)
    np.testing.assert_allclose(result.state["heat_Zimov"], 0.0)
    np.testing.assert_allclose(result.state["co2_flux"], 0.0)
    np.testing.assert_allclose(result.state["fco2_lu"], 0.0)
    assert result.state["N_limfert"] is None
    assert result.source_undefined_fields == ("N_limfert",)
    assert result.state["dt_days_changed"]
    np.testing.assert_allclose(result.state["temp_growth"], initial["t2m_month"] - 273.15)
    assert result.state["frozen_respiration_func"] == 1


def test_dynamic_landpoints_extensible_nvm_and_zero_pft14_cover():
    result, _, initial = _run(npts=2, nvm=19)
    assert np.asarray(result.state["co2_flux"]).shape == (2, 19)
    assert np.asarray(result.state["heat_Zimov"]).shape == (2, 2, 19)
    expected = initial["veget_max"] / (1.0 - initial["totfrac_nobio"][:, None])
    np.testing.assert_allclose(result.state["veget_cov_max"], expected)

    initial["veget_max"][:, 13] = 0.0
    calls = []
    owners = _owners(calls, initial)
    zero_cover = stomate_initialize_pft14_explicit(
        dimensions=StomateInitializeDimensions(npts=2, nvm=19),
        owner_inputs={name: {"position": i} for i, name in enumerate(
            ("stomate_init", "data", "readstart", "season", "carbon"), 1
        )},
        owners=owners,
        dt_days=1.0,
        dt_sechiba_seconds=1800.0,
        max_dt_days=10.0,
    )
    np.testing.assert_allclose(np.asarray(zero_cover.state["veget_cov_max"])[:, 13], 0.0)


@pytest.mark.parametrize(
    "switches,match",
    [
        (StomateInitializePFT14Switches(spinup_analytic=True), "spinup_analytic"),
        (StomateInitializePFT14Switches(ok_laidev=(False,) * 13 + (True,) + (False,) * 2), "ok_LAIdev"),
        (StomateInitializePFT14Switches(stomate_cforcing_name="soil.nc"), "stomate_cforcing_name"),
        (StomateInitializePFT14Switches(ch4_calcul=True), "ch4_calcul"),
        (StomateInitializePFT14Switches(peat_occur=True), "peat_occur"),
    ],
)
def test_fixed_structural_switches_reject_external_branches(switches, match):
    with pytest.raises(NotImplementedError, match=match):
        _run(switches=switches)


@pytest.mark.parametrize(
    "dt_days,dt_sechiba,max_dt,match",
    [
        (1.25, 1800.0, 10.0, "multiple of a full day"),
        (11.0, 1800.0, 10.0, "exceeds max_dt_days"),
        (1.0, 90000.0, 10.0, "smaller than the forcing"),
    ],
)
def test_exact_fortran_timestep_failures(dt_days, dt_sechiba, max_dt, match):
    calls = []
    initial = _state()
    with pytest.raises(ValueError, match=match):
        stomate_initialize_pft14_explicit(
            dimensions=StomateInitializeDimensions(npts=3, nvm=16),
            owner_inputs={name: {"position": i} for i, name in enumerate(
                ("stomate_init", "data", "readstart", "season", "carbon"), 1
            )},
            owners=_owners(calls, initial),
            dt_days=dt_days,
            dt_sechiba_seconds=dt_sechiba,
            max_dt_days=max_dt,
        )


def test_missing_owner_is_an_explicit_boundary_error():
    with pytest.raises(StomateInitializationBoundaryError, match="stomate_init"):
        stomate_initialize_pft14_explicit(
            dimensions=StomateInitializeDimensions(npts=1, nvm=14),
            owner_inputs={},
            owners=StomateInitializationOwners(),
            dt_days=1.0,
            dt_sechiba_seconds=1800.0,
            max_dt_days=10.0,
        )


def test_carbon_var_init_preserves_restart_assim_and_computes_deadleaf_cover():
    npts, nvm = 2, 16
    assim = np.arange(npts * nvm * 3, dtype=float).reshape(npts, nvm, 3)
    dead_leaves = np.zeros((npts, nvm, 2))
    dead_leaves[:, 13, 0] = [0.5, 1.0]
    veget_cov_max = np.zeros((npts, nvm))
    veget_cov_max[:, 13] = [0.4, 0.8]
    sla_calc = np.ones((npts, nvm)) * 2.0

    result = stomate_var_init_carbon_owner(
        assim_param=assim,
        leaf_age=np.zeros((npts, nvm, 4)),
        leaf_frac=np.zeros((npts, nvm, 4)),
        dead_leaves=dead_leaves,
        veget_cov_max=veget_cov_max,
        sla_calc=sla_calc,
        val_exp=999999.0,
    )

    np.testing.assert_array_equal(result.assim_param, assim)
    expected = 1.0 - np.exp(-0.5 * np.asarray([0.5, 1.0]) * 2.0 * np.asarray([0.4, 0.8]))
    np.testing.assert_allclose(result.deadleaf_cover, expected)
    assert "stomate_var_init lines 9249-9321" in result.provenance[0]


def test_carbon_var_init_requires_explicit_n_limfert_when_restart_value_is_missing():
    with pytest.raises(StomateInitializationBoundaryError, match="n_limfert"):
        stomate_var_init_carbon_owner(
            assim_param=np.full((1, 14, 3), 999999.0),
            leaf_age=np.zeros((1, 14, 4)),
            leaf_frac=np.zeros((1, 14, 4)),
            dead_leaves=np.zeros((1, 14, 2)),
            veget_cov_max=np.zeros((1, 14)),
            sla_calc=np.ones((1, 14)),
            val_exp=999999.0,
        )


def test_carbon_var_init_cold_start_computes_vcmax_and_zeroes_bare_soil():
    npts, nvm, nleafages = 1, 14, 4
    leaf_frac = np.zeros((npts, nvm, nleafages))
    leaf_frac[:, :, 0] = 1.0
    result = stomate_var_init_carbon_owner(
        assim_param=np.full((npts, nvm, 3), 999999.0),
        leaf_age=np.zeros((npts, nvm, nleafages)),
        leaf_frac=leaf_frac,
        dead_leaves=np.zeros((npts, nvm, 2)),
        veget_cov_max=np.zeros((npts, nvm)),
        sla_calc=np.ones((npts, nvm)),
        val_exp=999999.0,
        vmax_inputs={
            "vcmax25": np.arange(nvm, dtype=float) + 10.0,
            "n_limfert": np.ones((npts, nvm)),
            "leaf_timecst": np.full(nvm, 25.0),
            "leafagecrit": np.full(nvm, 100.0),
            "pheno_type": np.ones(nvm, dtype=int),
            "leaf_tab": np.ones(nvm, dtype=int),
            "ok_laidev": np.zeros(nvm, dtype=bool),
        },
    )
    assert result.assim_param[0, 0, 0] == 0.0
    np.testing.assert_allclose(
        result.assim_param[0, 1:, 0],
        0.3 * (np.arange(1, nvm, dtype=float) + 10.0),
    )


def test_provenance_names_exact_file_subroutine_and_owned_line_span():
    assert STOMATE_INITIALIZE_PROVENANCE[0] == (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_initialize lines 1465-2350"
    )
    assert any("stomate_var_init" in item and "2249-2254" in item for item in STOMATE_INITIALIZE_PROVENANCE)


def test_forcing_boundaries_classify_nested_io_without_scientific_values():
    records = stomate_initialize_forcing_boundaries(StomateInitializePFT14Switches())
    assert [record.line for record in records] == [1866, 2059, 2115, 2150, 2180, 2199, 2205, 2220]
    assert not any(record.taken for record in records)
    assert records[0].classification == "io_gate"
    assert records[2].classification == "nested_allocation_gate"
    assert "disabled carbon forcing writer" in records[2].reason

    result, _, _ = _run()
    assert result.state["forcing_boundary_dispositions"] == records
