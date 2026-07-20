from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.daily_inputs import (
    daily_input_availability_from_entry,
    daily_input_availability_with_supplied_totfrac_nobio,
    entry_normalization_availability_from_record,
    entry_normalization_availability_with_supplied_totfrac_nobio,
    entry_normalization_availability_with_supplied_totfrac_nobio_new,
    stomate_entry_local_prep_explicit,
    stomate_gpp_daily_increment,
    stomate_gpp_daily_increment_from_veget_cov_max,
    stomate_normalize_by_bio_fraction,
    stomate_precip_increment,
    stomate_veget_cover_fractions,
    stomate_veget_cov_max,
    stomate_veget_cov_max_new,
    stomate_vegetnew_firstday,
)
from jax_orchidee.stomate.entry import read_first_pft14_entry
from jax_orchidee.driver.init import initialize_imposed_vegetation_state, read_run_scalars


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def test_first_step_entry_can_build_precip_increment_from_fortran_formula():
    payload = read_first_pft14_entry()
    availability = daily_input_availability_from_entry(payload.record)

    precip = stomate_precip_increment(
        payload.record["precip_rain"],
        payload.record["precip_snow"],
        dt_sechiba=payload.record["dt_sechiba"],
    )

    assert availability.precip_status == "covered"
    assert availability.precip_fields == ("precip_rain", "precip_snow", "dt_sechiba")
    assert np.asarray(precip).shape == ()
    assert np.asarray(precip) == pytest.approx(0.0, abs=0.0)


def test_first_step_entry_gpp_increment_contract_keeps_totfrac_nobio_gap_visible():
    payload = read_first_pft14_entry()
    availability = daily_input_availability_from_entry(payload.record)

    assert availability.gpp_status == "partial"
    assert availability.gpp_fields == ("gpp", "veget_max", "dt_sechiba", "totfrac_nobio")
    assert availability.missing_for_gpp == ("totfrac_nobio",)
    with pytest.raises(KeyError, match="totfrac_nobio"):
        payload.require("totfrac_nobio")


def test_tot_bare_soil_does_not_satisfy_totfrac_nobio_contract():
    payload = read_first_pft14_entry()
    record = dict(payload.record)
    record["tot_bare_soil"] = 0.0

    availability = daily_input_availability_from_entry(record)

    assert availability.gpp_status == "partial"
    assert availability.missing_for_gpp == ("totfrac_nobio",)
    assert any("not the same contract" in item for item in availability.totfrac_nobio_provenance)


def test_entry_normalization_contract_keeps_current_and_new_totfrac_separate():
    payload = read_first_pft14_entry()
    record = dict(payload.record)
    record["tot_bare_soil"] = 0.0

    availability = entry_normalization_availability_from_record(record)

    assert availability.current_status == "partial"
    assert availability.new_firstday_status == "partial"
    assert availability.new_lcchange_status == "partial"
    assert availability.missing_for_current == ("totfrac_nobio",)
    assert availability.missing_for_new_firstday == ("vegetnew_firstday", "totfrac_nobio_new", "date")
    assert availability.missing_for_new_lcchange == ("totfrac_nobio_new", "do_now_stomate_lcchange")


def test_supplied_current_totfrac_nobio_only_closes_current_normalization():
    payload = read_first_pft14_entry()
    scalars = read_run_scalars(CONFIG)
    state = initialize_imposed_vegetation_state(scalars, npts=1)

    availability = entry_normalization_availability_with_supplied_totfrac_nobio(
        payload.record,
        totfrac_nobio=state.totfrac_nobio,
    )
    veget_cov, veget_cov_max = stomate_veget_cover_fractions(
        np.asarray([[payload.record["veget"]]], dtype=np.float64),
        np.asarray([[payload.record["veget_max"]]], dtype=np.float64),
        state.totfrac_nobio,
    )

    assert availability.current_status == "covered"
    assert availability.missing_for_current == ()
    assert availability.missing_for_new_lcchange == ("totfrac_nobio_new", "do_now_stomate_lcchange")
    assert np.asarray(veget_cov)[0, 0] == pytest.approx(1.0)
    assert np.asarray(veget_cov_max)[0, 0] == pytest.approx(1.0)


def test_supplied_totfrac_nobio_new_only_closes_new_denominator_not_branch_inputs():
    payload = read_first_pft14_entry()

    availability = entry_normalization_availability_with_supplied_totfrac_nobio_new(
        payload.record,
        totfrac_nobio_new=np.asarray([0.25], dtype=np.float64),
    )

    assert availability.current_status == "partial"
    assert availability.missing_for_current == ("totfrac_nobio",)
    assert availability.missing_for_new_firstday == ("vegetnew_firstday", "date")
    assert availability.missing_for_new_lcchange == ("do_now_stomate_lcchange",)


def test_normalization_helpers_match_source_formula_and_zero_non_bio_pixels():
    field = np.asarray([[0.5, 0.25], [9.0, 4.0]], dtype=np.float64)
    totfrac = np.asarray([0.5, 1.0 - 1.0e-10], dtype=np.float64)

    normalized = stomate_normalize_by_bio_fraction(field, totfrac)
    vegetnew = stomate_vegetnew_firstday(field, totfrac)
    veget_cov_max_new = stomate_veget_cov_max_new(field, totfrac)

    expected = np.asarray([[1.0, 0.5], [0.0, 0.0]], dtype=np.float64)
    assert np.allclose(np.asarray(normalized), expected)
    assert np.allclose(np.asarray(vegetnew), expected)
    assert np.allclose(np.asarray(veget_cov_max_new), expected)


def test_driver_static_totfrac_nobio_closes_gpp_contract_without_trace_aliasing():
    payload = read_first_pft14_entry()
    scalars = read_run_scalars(CONFIG)
    state = initialize_imposed_vegetation_state(scalars, npts=1)

    availability = daily_input_availability_with_supplied_totfrac_nobio(
        payload.record,
        totfrac_nobio=state.totfrac_nobio,
    )
    gpp = np.zeros((1, 14), dtype=np.float64)
    gpp[0, 13] = 2.0
    gpp_d = stomate_gpp_daily_increment(
        gpp,
        state.veget_max,
        state.totfrac_nobio,
        dt_sechiba=payload.record["dt_sechiba"],
    )

    assert np.allclose(state.totfrac_nobio, [0.0])
    assert availability.gpp_status == "covered"
    assert availability.missing_for_gpp == ()
    assert np.asarray(gpp_d).shape == (1, 14)
    assert np.asarray(gpp_d)[0, 0] == pytest.approx(0.0, abs=0.0)
    assert np.asarray(gpp_d)[0, 13] == pytest.approx(96.0)


def test_gpp_increment_matches_source_formula_when_totfrac_nobio_is_supplied():
    gpp = np.asarray([[0.0, 2.0, 3.0]], dtype=np.float64)
    veget_max = np.asarray([[1.0, 0.5, 0.0]], dtype=np.float64)
    totfrac_nobio = np.asarray([0.25], dtype=np.float64)

    veget_cov_max = stomate_veget_cov_max(veget_max, totfrac_nobio)
    gpp_d = stomate_gpp_daily_increment(gpp, veget_max, totfrac_nobio, dt_sechiba=43200.0)

    assert np.allclose(np.asarray(veget_cov_max), [[4.0 / 3.0, 2.0 / 3.0, 0.0]])
    assert np.allclose(np.asarray(gpp_d), [[0.0, 6.0, 0.0]])


def test_gpp_increment_from_veget_cov_max_matches_post_normalization_source_formula():
    gpp = np.asarray([[0.0, 2.0, 3.0]], dtype=np.float64)
    veget_cov_max = np.asarray([[1.0, 2.0 / 3.0, 0.0]], dtype=np.float64)

    gpp_d = stomate_gpp_daily_increment_from_veget_cov_max(
        gpp,
        veget_cov_max,
        dt_sechiba=43200.0,
    )

    assert np.allclose(np.asarray(gpp_d), [[0.0, 6.0, 0.0]])


def test_entry_local_prep_groups_stomate_main_section_3_without_extra_inference():
    veget = np.asarray([[0.25, 0.5, 0.0], [8.0, 4.0, 2.0]], dtype=np.float64)
    veget_max = np.asarray([[0.5, 0.25, 0.0], [9.0, 8.0, 7.0]], dtype=np.float64)
    totfrac_nobio = np.asarray([0.5, 1.0 - 1.0e-10], dtype=np.float64)
    gpp = np.asarray([[0.0, 3.0, 7.0], [0.0, 99.0, 88.0]], dtype=np.float64)
    glcc = np.asarray([[0.12, 0.24], [9.0, 8.0]], dtype=np.float64)

    result = stomate_entry_local_prep_explicit(
        precip_rain=np.asarray([1.0, 2.0], dtype=np.float64),
        precip_snow=np.asarray([3.0, 4.0], dtype=np.float64),
        dt_sechiba=43200.0,
        veget=veget,
        veget_max=veget_max,
        totfrac_nobio=totfrac_nobio,
        gpp=gpp,
        glccNetLCC=glcc,
        glccSecondShift=glcc + 1.0,
        glccPrimaryShift=glcc + 2.0,
        harvest_matrix=glcc + 3.0,
    )

    assert np.allclose(np.asarray(result.precip), [8.0, 12.0])
    assert np.allclose(np.asarray(result.veget_cov), [[0.5, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(np.asarray(result.veget_cov_max), [[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(np.asarray(result.gpp_d), [[0.0, 12.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(np.asarray(result.glccNetLCC), [[0.24, 0.48], [0.0, 0.0]])
    assert np.allclose(np.asarray(result.glccSecondShift), [[2.24, 2.48], [0.0, 0.0]])
    assert np.allclose(np.asarray(result.glccPrimaryShift), [[4.24, 4.48], [0.0, 0.0]])
    assert np.allclose(np.asarray(result.harvest_matrix), [[6.24, 6.48], [0.0, 0.0]])
    assert result.vegetnew_firstday is None
    assert result.veget_cov_max_new is None


def test_entry_local_prep_normalizes_firstday_and_lcchange_branches_only_when_active():
    veget = np.asarray([[1.0, 0.0]], dtype=np.float64)
    veget_max = np.asarray([[1.0, 0.0]], dtype=np.float64)
    gpp = np.zeros((1, 2), dtype=np.float64)
    totfrac_nobio = np.asarray([0.0], dtype=np.float64)
    totfrac_nobio_new = np.asarray([0.25], dtype=np.float64)
    vegetnew_firstday = np.asarray([[0.75, 0.375]], dtype=np.float64)
    veget_max_new = np.asarray([[0.375, 0.75]], dtype=np.float64)

    result = stomate_entry_local_prep_explicit(
        precip_rain=np.asarray([0.0], dtype=np.float64),
        precip_snow=np.asarray([0.0], dtype=np.float64),
        dt_sechiba=1800.0,
        veget=veget,
        veget_max=veget_max,
        totfrac_nobio=totfrac_nobio,
        gpp=gpp,
        date=1,
        vegetnew_firstday=vegetnew_firstday,
        totfrac_nobio_new=totfrac_nobio_new,
        veget_max_new=veget_max_new,
        do_now_stomate_lcchange=True,
    )

    assert np.allclose(np.asarray(result.vegetnew_firstday), [[1.0, 0.5]])
    assert np.allclose(np.asarray(result.veget_cov_max_new), [[0.5, 1.0]])


def test_entry_local_prep_preserves_firstday_state_when_date_is_not_one():
    veget = np.asarray([[1.0, 0.0]], dtype=np.float64)
    veget_max = np.asarray([[1.0, 0.0]], dtype=np.float64)
    gpp = np.zeros((1, 2), dtype=np.float64)
    vegetnew_firstday = np.asarray([[0.75, 0.375]], dtype=np.float64)

    result = stomate_entry_local_prep_explicit(
        precip_rain=np.asarray([0.0], dtype=np.float64),
        precip_snow=np.asarray([0.0], dtype=np.float64),
        dt_sechiba=1800.0,
        veget=veget,
        veget_max=veget_max,
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        gpp=gpp,
        date=2,
        vegetnew_firstday=vegetnew_firstday,
        totfrac_nobio_new=np.asarray([0.25], dtype=np.float64),
    )

    assert np.allclose(np.asarray(result.vegetnew_firstday), vegetnew_firstday)
    assert result.veget_cov_max_new is None


def test_entry_local_prep_requires_explicit_new_fraction_for_active_branches():
    veget = np.asarray([[1.0, 0.0]], dtype=np.float64)
    veget_max = np.asarray([[1.0, 0.0]], dtype=np.float64)
    gpp = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="totfrac_nobio_new"):
        stomate_entry_local_prep_explicit(
            precip_rain=np.asarray([0.0], dtype=np.float64),
            precip_snow=np.asarray([0.0], dtype=np.float64),
            dt_sechiba=1800.0,
            veget=veget,
            veget_max=veget_max,
            totfrac_nobio=np.asarray([0.0], dtype=np.float64),
            gpp=gpp,
            date=1,
            vegetnew_firstday=np.asarray([[1.0, 0.0]], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="veget_max_new"):
        stomate_entry_local_prep_explicit(
            precip_rain=np.asarray([0.0], dtype=np.float64),
            precip_snow=np.asarray([0.0], dtype=np.float64),
            dt_sechiba=1800.0,
            veget=veget,
            veget_max=veget_max,
            totfrac_nobio=np.asarray([0.0], dtype=np.float64),
            gpp=gpp,
            totfrac_nobio_new=np.asarray([0.0], dtype=np.float64),
            dyn_peat=True,
            update_peatfrac=True,
        )
