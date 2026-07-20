from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.stomate.finalize import (
    STOMATE_FINALIZE_PROVENANCE,
    StomateFinalizeDimensions,
    StomateFinalizePFT14Switches,
    stomate_finalize_pft14_explicit,
)
from jax_orchidee.stomate.reference import (
    stomate_cold_start_daily_accumulator_state,
    stomate_cold_start_entry_state,
    stomate_cold_start_season_state,
)


def _states(*, npts: int = 2, nvm: int = 17):
    t2m = np.linspace(273.15, 275.15, npts)
    entry = stomate_cold_start_entry_state(t2m=t2m, nvm=nvm, nslm=2, ndeep=3)
    pft_present = np.zeros((npts, nvm), dtype=bool)
    pft_present[:, 13] = True
    entry = entry._replace(pft_present=pft_present)
    season = stomate_cold_start_season_state(t2m=t2m, dt_days=1.0, nvm=nvm, nslm=2)
    daily = stomate_cold_start_daily_accumulator_state(t2m=t2m, nvm=nvm, nslm=2)
    return entry, season, daily


def _run(*, npts: int = 2, nvm: int = 17, switches=StomateFinalizePFT14Switches()):
    entry, season, daily = _states(npts=npts, nvm=nvm)
    calls = []

    def writer(request):
        calls.append(request)
        return {"write_id": 7}

    result = stomate_finalize_pft14_explicit(
        dimensions=StomateFinalizeDimensions(npts=npts, nvm=nvm),
        entry_state=entry,
        season_state=season,
        daily_state=daily,
        restart_writer=writer,
        switches=switches,
    )
    return result, calls, (entry, season, daily)


def test_restart_contract_is_handed_to_callback_once_before_inactive_io_branches():
    result, calls, states = _run()

    assert calls == [result.restart_request]
    assert result.restart_request.entry_state is states[0]
    assert result.restart_request.season_state is states[1]
    assert result.restart_request.daily_state is states[2]
    assert result.restart_request.as_writer_kwargs() == {
        "entry_state": states[0],
        "season_state": states[1],
        "daily_state": states[2],
    }
    assert result.restart_write_result == {"write_id": 7}
    assert result.process_order == (
        "writerestart",
        "sticslai_io_writestart",
        "forcing_write",
        "carbon_forcing_write",
        "permafrost_carbon_forcing_write",
    )
    assert [boundary.active for boundary in result.io_boundaries] == [True, False, False, False, False]


def test_dynamic_landpoints_and_nvm_preserve_active_fortran_pft14():
    result, _, _ = _run(npts=4, nvm=20)
    assert result.restart_request.entry_state.pft_present.shape == (4, 20)


def test_finalize_does_not_invent_a_pft_present_constraint():
    """stomate_finalize lines 5440-5496 write PFTpresent but never gate on it."""

    entry, season, daily = _states(npts=3, nvm=18)
    entry = entry._replace(pft_present=entry.pft_present.at[:, 13].set(False)) if hasattr(
        entry.pft_present, "at"
    ) else entry._replace(pft_present=np.zeros((3, 18), dtype=bool))
    result = stomate_finalize_pft14_explicit(
        dimensions=StomateFinalizeDimensions(npts=3, nvm=18),
        entry_state=entry,
        season_state=season,
        daily_state=daily,
        restart_writer=lambda request: request,
    )
    np.testing.assert_array_equal(result.restart_request.entry_state.pft_present, False)


@pytest.mark.parametrize(
    "switches,match",
    [
        (StomateFinalizePFT14Switches(ok_laidev=True), "ok_LAIdev"),
        (StomateFinalizePFT14Switches(stomate_forcing_name="forcing.nc"), "stomate_forcing_name"),
        (StomateFinalizePFT14Switches(stomate_cforcing_name="carbon.nc"), "stomate_cforcing_name"),
        (
            StomateFinalizePFT14Switches(cforcing_permafrost_name="permafrost.nc"),
            "cforcing_permafrost_name",
        ),
        (StomateFinalizePFT14Switches(ok_leak=False), "ok_leak"),
        (StomateFinalizePFT14Switches(ok_peat=True), "ok_peat"),
    ],
)
def test_fixed_paper_switches_reject_unimplemented_source_branches(switches, match):
    with pytest.raises(NotImplementedError, match=match):
        _run(switches=switches)


def test_shape_or_contract_failure_occurs_before_writer_side_effect():
    entry, season, daily = _states()
    calls = []
    with pytest.raises(ValueError, match="moiavail_month must have shape"):
        stomate_finalize_pft14_explicit(
            dimensions=StomateFinalizeDimensions(npts=2, nvm=17),
            entry_state=entry,
            season_state=season._replace(moiavail_month=np.zeros((1, 17))),
            daily_state=daily,
            restart_writer=calls.append,
        )
    assert calls == []


def test_shared_gpp_daily_mismatch_fails_before_writer_side_effect():
    entry, season, daily = _states()
    calls = []
    daily = daily._replace(gpp_daily=np.ones_like(daily.gpp_daily))
    with pytest.raises(ValueError, match="shared restart field gpp_daily differs"):
        stomate_finalize_pft14_explicit(
            dimensions=StomateFinalizeDimensions(npts=2, nvm=17),
            entry_state=entry,
            season_state=season,
            daily_state=daily,
            restart_writer=calls.append,
        )
    assert calls == []


def test_provenance_covers_owned_span_and_existing_writerestart_contract():
    joined = "\n".join(STOMATE_FINALIZE_PROVENANCE)
    assert "stomate_finalize lines 5499-5523" in joined
    assert "stomate_finalize lines 5535-6132" in joined
    assert "stomate_finalize lines 6133-6205" in joined
    assert "stomate_io.f90::writerestart lines 1751-2944" in joined


def test_each_finalize_decision_has_exact_source_provenance_and_disposition():
    result, _, _ = _run()

    assert [boundary.line_span for boundary in result.io_boundaries] == [
        (5440, 5496),
        (5499, 5523),
        (5525, 5533),
        (5535, 6132),
        (6133, 6205),
    ]
    for boundary in result.io_boundaries:
        assert boundary.source_file == "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
        assert boundary.source_subroutine == "stomate_finalize"
        assert boundary.provenance.endswith(
            f"lines {boundary.line_span[0]}-{boundary.line_span[1]}"
        )
        assert boundary.gate
        assert boundary.disposition
    assert result.io_boundaries[0].classification == "restart_serialization"
    assert all(boundary.classification.endswith("io") for boundary in result.io_boundaries[1:])
