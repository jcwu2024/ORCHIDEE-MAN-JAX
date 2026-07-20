from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.daily import (
    accumulate_resp_maint_part,
    prepare_daily_carbon_inputs,
    require_explicit_f_alloc,
    reset_daily_on_slow,
    stomate_accumulate_daily,
    stomate_accumulate_named_daily,
    stomate_cforcing_accumulate_slot,
    stomate_cforcing_finalize_slots,
    stomate_cforcing_slot_index,
    stomate_crop_growth_degree_hour,
    stomate_crop_photoperiod_unit,
    stomate_daily_accumulation_fold_from_entries,
    stomate_daily_process_fold_from_entries,
    stomate_end_of_month,
    stomate_first_step_daily_accumulation_from_entry,
    stomate_instant_daily_prep,
    stomate_maintenance_fold_from_entries,
    stomate_update_temperature_extrema_daily,
    sum_resp_maint_radia,
)
from jax_orchidee.stomate.carbon_kernels import ICARBON, ILEAF, IROOT, NPARTS, maintenance_respiration


def test_stomate_accumulate_daily_matches_source_formula_before_slow_step():
    current = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    increment = np.asarray([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)

    result = stomate_accumulate_daily(
        current,
        increment,
        False,
        dt_sechiba=1800.0,
        dt_stomate=86400.0,
    )

    assert np.allclose(np.asarray(result), current + increment * 1800.0)
    assert np.asarray(result).shape == current.shape


def test_stomate_accumulate_daily_divides_by_dt_stomate_on_slow_step():
    current = np.asarray([10.0, 20.0], dtype=np.float64)
    increment = np.asarray([1.0, 2.0], dtype=np.float64)

    result = stomate_accumulate_daily(
        current,
        increment,
        True,
        dt_sechiba=6.0,
        dt_stomate=24.0,
    )

    assert np.allclose(np.asarray(result), (current + increment * 6.0) / 24.0)


def test_stomate_accumulate_daily_preserves_3d_generic_semantics():
    current = np.ones((2, 3, 4), dtype=np.float64)
    increment = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    result = stomate_accumulate_daily(
        current,
        increment,
        True,
        dt_sechiba=1800.0,
        dt_stomate=86400.0,
    )

    assert np.allclose(np.asarray(result), (current + increment * 1800.0) / 86400.0)
    assert np.asarray(result).shape == current.shape


def test_stomate_end_of_month_matches_first_step_of_month_source_condition():
    assert stomate_end_of_month(1, 0.0, dt_sechiba=1800.0) is True
    assert stomate_end_of_month(1, 1799.0, dt_sechiba=1800.0) is True
    assert stomate_end_of_month(1, 1800.0, dt_sechiba=1800.0) is False
    assert stomate_end_of_month(2, 0.0, dt_sechiba=1800.0) is False


def test_cforcing_slot_index_matches_fortran_iatt_formula_zero_based():
    assert stomate_cforcing_slot_index(1, one_year=10.0, nbyear=1, dt_forcesoil=2.0, nslots=5) == 0
    assert stomate_cforcing_slot_index(2, one_year=10.0, nbyear=1, dt_forcesoil=2.0, nslots=5) == 0
    assert stomate_cforcing_slot_index(3, one_year=10.0, nbyear=1, dt_forcesoil=2.0, nslots=5) == 1
    assert stomate_cforcing_slot_index(11, one_year=10.0, nbyear=1, dt_forcesoil=2.0, nslots=5) == 0


def test_cforcing_accumulate_slot_resets_on_window_wrap_when_not_cumulative():
    fields = {
        "soilcarbon_input": np.ones((1, 2, 3, 5), dtype=np.float64) * 99.0,
        "tprof_2pfcforcing": np.ones((1, 2, 3, 5), dtype=np.float64) * 77.0,
    }
    increments = {
        "soilcarbon_input": np.ones((1, 2, 3), dtype=np.float64) * 2.0,
        "tprof_2pfcforcing": np.ones((1, 2, 3), dtype=np.float64) * 4.0,
    }

    result = stomate_cforcing_accumulate_slot(
        fields,
        increments,
        np.ones(5, dtype=np.float64) * 3.0,
        date=11,
        iatt_old=4,
        cumul_cforcing=False,
        one_year=10.0,
        nbyear=1,
        dt_forcesoil=2.0,
    )

    assert result.iatt == 0
    assert result.reset_window is True
    np.testing.assert_allclose(np.asarray(result.nforce), [1.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(result.fields["soilcarbon_input"])[..., 0], 2.0)
    np.testing.assert_allclose(np.asarray(result.fields["soilcarbon_input"])[..., 1:], 0.0)
    np.testing.assert_allclose(np.asarray(result.fields["tprof_2pfcforcing"])[..., 0], 4.0)


def test_cforcing_finalize_slots_averages_and_applies_permafrost_reciprocal_fields():
    fields = {
        "soilcarbon_input": np.asarray([[[[4.0, 0.0, 12.0]]]], dtype=np.float64),
        "fbact_2pfcforcing": np.asarray([[[[8.0, 0.0, 6.0]]]], dtype=np.float64),
    }
    nforce = np.asarray([2.0, 0.0, 3.0], dtype=np.float64)

    result = stomate_cforcing_finalize_slots(
        fields,
        nforce,
        reciprocal_fields=("fbact_2pfcforcing",),
    )

    np.testing.assert_allclose(np.asarray(result["soilcarbon_input"]), [[[[2.0, 0.0, 4.0]]]])
    np.testing.assert_allclose(np.asarray(result["fbact_2pfcforcing"]), [[[[0.25, 0.0, 0.5]]]])


def test_crop_growth_degree_hour_clips_by_pft_thresholds_and_keeps_bare_soil_zero():
    t2m = np.asarray([273.15 + 5.0, 273.15 + 20.0], dtype=np.float64)
    ok_laidev = np.asarray([True, True, False, True])
    tdmin = np.asarray([-99.0, 3.0, 4.0, 10.0], dtype=np.float64)
    tdmax = np.asarray([99.0, 12.0, 14.0, 15.0], dtype=np.float64)

    ugdh = stomate_crop_growth_degree_hour(t2m, ok_laidev, tdmin, tdmax)

    assert np.allclose(
        np.asarray(ugdh),
        [
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 5.0],
        ],
    )


def test_crop_photoperiod_unit_uses_positive_swdown_only():
    swdown = np.asarray([-1.0, 0.0, 0.1, 50.0], dtype=np.float64)

    uphoi = stomate_crop_photoperiod_unit(swdown)

    assert np.allclose(np.asarray(uphoi), [0.0, 0.0, 1.0, 1.0])


def test_instant_daily_prep_skips_crop_fields_when_no_ok_laidev():
    prep = stomate_instant_daily_prep(
        day=1,
        sec=0.0,
        dt_sechiba=1800.0,
        t2m=np.asarray([300.0], dtype=np.float64),
        ok_LAIdev=np.asarray([False, False, False]),
    )

    assert prep.end_of_month is True
    assert np.allclose(np.asarray(prep.st2m), [26.85])
    assert prep.ugdh is None
    assert prep.uphoi is None


def test_instant_daily_prep_requires_crop_thresholds_when_any_ok_laidev():
    with pytest.raises(ValueError, match="SP_tdmin"):
        stomate_instant_daily_prep(
            day=2,
            sec=0.0,
            dt_sechiba=1800.0,
            t2m=np.asarray([300.0], dtype=np.float64),
            ok_LAIdev=np.asarray([False, True]),
            SP_tdmax=np.asarray([0.0, 10.0], dtype=np.float64),
            swdown=np.asarray([1.0], dtype=np.float64),
        )


def test_update_temperature_extrema_daily_matches_min_max_assignments():
    t2m_min_daily, t2m_max_daily = stomate_update_temperature_extrema_daily(
        t2m_min=np.asarray([280.0, 270.0], dtype=np.float64),
        t2m_max=np.asarray([300.0, 295.0], dtype=np.float64),
        t2m_min_daily=np.asarray([275.0, 275.0], dtype=np.float64),
        t2m_max_daily=np.asarray([299.0, 301.0], dtype=np.float64),
    )

    assert np.allclose(np.asarray(t2m_min_daily), [275.0, 270.0])
    assert np.allclose(np.asarray(t2m_max_daily), [300.0, 301.0])


def test_accumulate_named_daily_updates_only_explicit_accumulators():
    current = {
        "humrel_daily": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "snowfall_daily": np.asarray([0.5], dtype=np.float64),
        "pb_pa_daily": np.asarray([100.0], dtype=np.float64),
    }
    increments = {
        "humrel_daily": np.asarray([[0.1, 0.2]], dtype=np.float64),
        "snowfall_daily": np.asarray([0.3], dtype=np.float64),
        "pb_pa_daily": np.asarray([101300.0], dtype=np.float64),
    }

    result = stomate_accumulate_named_daily(
        current,
        increments,
        False,
        dt_sechiba=2.0,
        dt_stomate=10.0,
    )

    assert np.allclose(np.asarray(result["humrel_daily"]), [[1.2, 2.4]])
    assert np.allclose(np.asarray(result["snowfall_daily"]), [1.1])
    assert np.allclose(np.asarray(result["pb_pa_daily"]), [202700.0])
    with pytest.raises(KeyError, match="missing_daily"):
        stomate_accumulate_named_daily(
            current,
            {"missing_daily": np.asarray([1.0], dtype=np.float64)},
            False,
            dt_sechiba=2.0,
            dt_stomate=10.0,
        )


def test_maintenance_respiration_scheduling_helpers_preserve_axes_and_bare_soil_zero():
    current = np.ones((2, 14, 12), dtype=np.float64)
    radia = np.arange(2 * 14 * 12, dtype=np.float64).reshape(2, 14, 12)
    flood_frac = np.asarray([0.25, 0.5], dtype=np.float64)

    accumulated = accumulate_resp_maint_part(current, radia)
    total, flood_root = sum_resp_maint_radia(radia, flood_frac=flood_frac, root_index=5)

    assert np.allclose(np.asarray(accumulated), current + radia)
    assert np.asarray(total).shape == (2, 14)
    assert np.asarray(flood_root).shape == (2, 14)
    assert np.allclose(np.asarray(total)[:, 0], 0.0)
    assert np.allclose(np.asarray(flood_root)[:, 0], 0.0)
    assert np.allclose(np.asarray(total)[:, 13], radia[:, 13, :].sum(axis=1))
    assert np.allclose(np.asarray(flood_root)[:, 13], flood_frac * radia[:, 13, 5])


def test_reset_daily_on_slow_resets_only_explicit_fields():
    fields = {
        "gpp_daily": np.ones((1, 14), dtype=np.float64),
        "resp_maint_part": np.ones((1, 14, 12), dtype=np.float64) * 2.0,
        "biomass": np.ones((1, 14, 12, 1), dtype=np.float64) * 3.0,
    }

    unchanged = reset_daily_on_slow(fields, False, ("gpp_daily", "resp_maint_part"))
    reset = reset_daily_on_slow(fields, True, ("gpp_daily", "resp_maint_part"))

    assert np.allclose(np.asarray(unchanged["gpp_daily"]), fields["gpp_daily"])
    assert np.allclose(np.asarray(reset["gpp_daily"]), 0.0)
    assert np.allclose(np.asarray(reset["resp_maint_part"]), 0.0)
    assert np.allclose(np.asarray(reset["biomass"]), fields["biomass"])


def test_prepare_daily_carbon_inputs_reports_missing_alloc_trace_without_fabricating():
    gpp_daily = np.ones((1, 14), dtype=np.float64)
    biomass = np.ones((1, 14, 12, 1), dtype=np.float64)
    resp_maint_part = np.ones((1, 14, 12), dtype=np.float64)

    boundary = prepare_daily_carbon_inputs(
        gpp_daily=gpp_daily,
        biomass=biomass,
        resp_maint_part=resp_maint_part,
        pft_present=np.ones((1, 14), dtype=bool),
    )

    assert boundary.gpp_daily.shape == (1, 14)
    assert boundary.biomass.shape == (1, 14, 12, 1)
    assert boundary.f_alloc is None
    assert "f_alloc" in boundary.requires_trace
    assert "bm_alloc" in boundary.requires_trace
    assert "resp_maint_part" not in boundary.requires_trace
    with pytest.raises(ValueError, match="f_alloc is required"):
        require_explicit_f_alloc(boundary)


def test_prepare_daily_carbon_inputs_accepts_explicit_f_alloc_but_keeps_other_trace_gaps():
    f_alloc = np.zeros((2, 14, 12), dtype=np.float64)

    boundary = prepare_daily_carbon_inputs(
        gpp_daily=np.ones((2, 14), dtype=np.float64),
        biomass=np.ones((2, 14, 12, 1), dtype=np.float64),
        resp_maint_part=np.ones((2, 14, 12), dtype=np.float64),
        f_alloc=f_alloc,
    )

    assert np.allclose(np.asarray(require_explicit_f_alloc(boundary)), f_alloc)
    assert "f_alloc" not in boundary.requires_trace
    assert "bm_alloc" in boundary.requires_trace


class _DailyAccumulatorState:
    def __init__(self):
        self.humrel_daily = np.asarray([[0.0, 0.0]], dtype=np.float64)
        self.litterhum_daily = np.asarray([0.0], dtype=np.float64)
        self.t2m_daily = np.asarray([0.0], dtype=np.float64)
        self.t2m_min_daily = np.asarray([999.0], dtype=np.float64)
        self.t2m_max_daily = np.asarray([-999.0], dtype=np.float64)
        self.wspeed_daily = np.asarray([0.0], dtype=np.float64)
        self.tsurf_daily = np.asarray([0.0], dtype=np.float64)
        self.tsoil_daily = np.asarray([[0.0, 0.0]], dtype=np.float64)
        self.soilhum_daily = np.asarray([[0.0, 0.0]], dtype=np.float64)
        self.precip_daily = np.asarray([0.0], dtype=np.float64)
        self.gpp_daily = np.asarray([[0.0, 0.0]], dtype=np.float64)
        self.snowfall_daily = np.asarray([0.0], dtype=np.float64)
        self.snowmass_daily = np.asarray([0.0], dtype=np.float64)
        self.tmc_topgrass_daily = np.asarray([0.0], dtype=np.float64)
        self.fwet_daily = np.asarray([0.0], dtype=np.float64)
        self.liqwt_daily = np.asarray([0.0], dtype=np.float64)


def _entry_payload(*, scale: float, t2m_min: float, t2m_max: float) -> dict[str, np.ndarray]:
    return {
        "precip_rain": np.asarray([scale], dtype=np.float64),
        "precip_snow": np.asarray([0.5 * scale], dtype=np.float64),
        "veget": np.asarray([[0.0, 0.8]], dtype=np.float64),
        "veget_max": np.asarray([[0.0, 0.8]], dtype=np.float64),
        "totfrac_nobio": np.asarray([0.2], dtype=np.float64),
        "gpp": np.asarray([[0.0, 2.0 * scale]], dtype=np.float64),
        "humrel": np.asarray([[scale, 2.0 * scale]], dtype=np.float64),
        "litterhumdiag": np.asarray([3.0 * scale], dtype=np.float64),
        "t2m": np.asarray([280.0 + scale], dtype=np.float64),
        "temp_sol": np.asarray([281.0 + scale], dtype=np.float64),
        "stempdiag": np.asarray([[282.0 + scale, 283.0 + scale]], dtype=np.float64),
        "shumdiag": np.asarray([[0.1 * scale, 0.2 * scale]], dtype=np.float64),
        "t2m_min": np.asarray([t2m_min], dtype=np.float64),
        "t2m_max": np.asarray([t2m_max], dtype=np.float64),
        "wspeed": np.asarray([4.0 * scale], dtype=np.float64),
        "snow": np.asarray([5.0 * scale], dtype=np.float64),
        "tmc_topgrass": np.asarray([0.3 * scale], dtype=np.float64),
    }


def test_daily_accumulation_fold_advances_explicit_entries_until_slow_boundary():
    state = _DailyAccumulatorState()
    payloads = (
        _entry_payload(scale=1.0, t2m_min=279.0, t2m_max=290.0),
        _entry_payload(scale=2.0, t2m_min=277.0, t2m_max=292.0),
        _entry_payload(scale=3.0, t2m_min=278.0, t2m_max=291.0),
    )

    folded = stomate_daily_accumulation_fold_from_entries(
        entry_payloads=payloads,
        accumulator_state=state,
        do_slow_flags=(False, False, True),
        dt_sechiba=2.0,
        dt_stomate=6.0,
    )

    assert folded.ok
    assert folded.failed_step is None
    assert len(folded.step_results) == 3
    np.testing.assert_allclose(np.asarray(folded.fields["humrel_daily"]), [[2.0, 4.0]])
    np.testing.assert_allclose(np.asarray(folded.fields["litterhum_daily"]), [6.0])
    np.testing.assert_allclose(np.asarray(folded.fields["wspeed_daily"]), [8.0])
    np.testing.assert_allclose(np.asarray(folded.fields["precip_daily"]), [129600.0])
    np.testing.assert_allclose(np.asarray(folded.fields["gpp_daily"]), [[0.0, 172800.0]])
    np.testing.assert_allclose(np.asarray(folded.fields["snowfall_daily"]), [1.0])
    np.testing.assert_allclose(np.asarray(folded.fields["snowmass_daily"]), [10.0])
    np.testing.assert_allclose(np.asarray(folded.fields["tmc_topgrass_daily"]), [0.6])
    np.testing.assert_allclose(np.asarray(folded.fields["t2m_min_daily"]), [277.0])
    np.testing.assert_allclose(np.asarray(folded.fields["t2m_max_daily"]), [292.0])


def test_first_step_daily_accumulation_passes_lcc_new_cover_switches_to_local_prep():
    state = _DailyAccumulatorState()
    payload = _entry_payload(scale=1.0, t2m_min=279.0, t2m_max=290.0)
    payload.update(
        {
            "do_now_stomate_lcchange": True,
            "veget_max_new": np.asarray([[0.0, 0.4]], dtype=np.float64),
            "totfrac_nobio_new": np.asarray([0.2], dtype=np.float64),
        }
    )

    result = stomate_first_step_daily_accumulation_from_entry(
        entry_payload=payload,
        accumulator_state=state,
        dt_sechiba=2.0,
        dt_stomate=4.0,
        do_slow=False,
    )

    assert result.local_prep is not None
    np.testing.assert_allclose(np.asarray(result.local_prep.veget_cov_max_new), [[0.0, 0.5]])


def test_first_step_daily_accumulation_rejects_active_lcc_without_new_cover_inputs():
    state = _DailyAccumulatorState()
    payload = _entry_payload(scale=1.0, t2m_min=279.0, t2m_max=290.0)
    payload["do_now_stomate_lcchange"] = True

    with pytest.raises(ValueError, match="veget_cov_max_new normalization"):
        stomate_first_step_daily_accumulation_from_entry(
            entry_payload=payload,
            accumulator_state=state,
            dt_sechiba=2.0,
            dt_stomate=4.0,
            do_slow=False,
        )


def test_daily_accumulation_fold_reports_first_missing_entry_without_fabricating_state():
    state = _DailyAccumulatorState()
    first = _entry_payload(scale=1.0, t2m_min=279.0, t2m_max=290.0)
    missing = _entry_payload(scale=2.0, t2m_min=277.0, t2m_max=292.0)
    del missing["gpp"]

    folded = stomate_daily_accumulation_fold_from_entries(
        entry_payloads=(first, missing),
        accumulator_state=state,
        do_slow_flags=(False, True),
        dt_sechiba=2.0,
        dt_stomate=4.0,
    )

    assert not folded.ok
    assert folded.failed_step == 1
    assert folded.missing_inputs == ("gpp",)
    np.testing.assert_allclose(np.asarray(folded.fields["humrel_daily"]), [[2.0, 4.0]])
    assert "gpp_daily" in folded.fields


def test_daily_accumulation_runtime_fold_matches_debug_path_bit_exact():
    state = _DailyAccumulatorState()
    payloads = (
        _entry_payload(scale=1.0, t2m_min=279.0, t2m_max=290.0),
        _entry_payload(scale=2.0, t2m_min=277.0, t2m_max=292.0),
        _entry_payload(scale=3.0, t2m_min=278.0, t2m_max=291.0),
    )
    common = {
        "entry_payloads": payloads,
        "accumulator_state": state,
        "do_slow_flags": (False, False, True),
        "dt_sechiba": 2.0,
        "dt_stomate": 6.0,
    }

    debug = stomate_daily_accumulation_fold_from_entries(**common)
    runtime = stomate_daily_accumulation_fold_from_entries(**common, retain_step_results=False)

    assert runtime.ok
    assert runtime.step_results == ()
    assert set(runtime.fields) == set(debug.fields)
    for name in runtime.fields:
        np.testing.assert_allclose(np.asarray(runtime.fields[name]), np.asarray(debug.fields[name]), rtol=0, atol=0)


def test_daily_accumulation_fold_rejects_internal_slow_boundary():
    state = _DailyAccumulatorState()
    payload = _entry_payload(scale=1.0, t2m_min=279.0, t2m_max=290.0)

    with pytest.raises(ValueError, match="final entry"):
        stomate_daily_accumulation_fold_from_entries(
            entry_payloads=(payload, payload),
            accumulator_state=state,
            do_slow_flags=(True, False),
            dt_sechiba=2.0,
            dt_stomate=4.0,
        )


def test_maintenance_fold_accumulates_each_entry_respiration_in_fortran_order():
    npts, nvm, pft = 1, 14, 13
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 100.0
    biomass[0, pft, IROOT, ICARBON] = 50.0
    z_soil = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    rprof = np.ones((npts, nvm), dtype=np.float64) * 0.8
    sla_calc = np.ones((npts, nvm), dtype=np.float64) * 0.02
    coeff_maint_zero = np.zeros((nvm, NPARTS), dtype=np.float64)
    coeff_maint_zero[pft, :] = 0.01
    maint_resp_slope = np.zeros((nvm, 3), dtype=np.float64)
    ext_coeff = np.ones(nvm, dtype=np.float64) * 0.5
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[pft] = True
    common = {
        "biomass": biomass,
        "t2m_longterm": np.asarray([293.15], dtype=np.float64),
        "z_soil": z_soil,
        "rprof": rprof,
        "sla_calc": sla_calc,
        "coeff_maint_zero": coeff_maint_zero,
        "maint_resp_slope": maint_resp_slope,
        "ext_coeff": ext_coeff,
        "is_tree": is_tree,
        "dt_sechiba": 43200.0,
        "flood_frac": np.asarray([0.25], dtype=np.float64),
    }
    entries = (
        {
            "t2m": np.asarray([293.15], dtype=np.float64),
            "stempdiag": np.asarray([[283.15, 284.15]], dtype=np.float64),
        },
        {
            "t2m": np.asarray([294.15], dtype=np.float64),
            "stempdiag": np.asarray([[285.15, 286.15]], dtype=np.float64),
        },
    )

    first = maintenance_respiration(**entries[0], **{k: v for k, v in common.items() if k not in {"dt_sechiba", "flood_frac"}}, dt_sechiba_days=0.5)
    second = maintenance_respiration(**entries[1], **{k: v for k, v in common.items() if k not in {"dt_sechiba", "flood_frac"}}, dt_sechiba_days=0.5)
    folded = stomate_maintenance_fold_from_entries(
        entry_payloads=entries,
        resp_maint_part_current=np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        **common,
    )

    assert folded.ok
    assert len(folded.step_results) == 2
    np.testing.assert_allclose(
        np.asarray(folded.resp_maint_part),
        np.asarray(first.resp_maint_part + second.resp_maint_part),
    )
    expected_radia, expected_flood = sum_resp_maint_radia(second.resp_maint_part, flood_frac=common["flood_frac"])
    np.testing.assert_allclose(np.asarray(folded.resp_maint_radia), np.asarray(expected_radia))
    np.testing.assert_allclose(np.asarray(folded.flood_root_radia), np.asarray(expected_flood))


def test_maintenance_default_uses_fortran_min_stomate_threshold():
    npts, nvm, pft = 1, 14, 13
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 0.5e-8
    common = {
        "biomass": biomass,
        "t2m": np.asarray([293.15], dtype=np.float64),
        "t2m_longterm": np.asarray([293.15], dtype=np.float64),
        "stempdiag": np.asarray([[283.15, 284.15]], dtype=np.float64),
        "z_soil": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        "rprof": np.ones((npts, nvm), dtype=np.float64),
        "sla_calc": np.ones((npts, nvm), dtype=np.float64),
        "coeff_maint_zero": np.ones((nvm, NPARTS), dtype=np.float64) * 0.01,
        "maint_resp_slope": np.zeros((nvm, 3), dtype=np.float64),
        "ext_coeff": np.ones(nvm, dtype=np.float64) * 0.5,
        "is_tree": np.zeros(nvm, dtype=bool),
    }

    below = maintenance_respiration(**common)
    assert np.asarray(below.resp_maint_part)[0, pft, ILEAF] == 0.0

    biomass[0, pft, ILEAF, ICARBON] = 2.0e-8
    above = maintenance_respiration(**{**common, "biomass": biomass})
    assert np.asarray(above.resp_maint_part)[0, pft, ILEAF] > 0.0


def test_maintenance_fold_jit_matches_explicit_path():
    npts, nvm, pft = 1, 14, 13
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 100.0
    biomass[0, pft, IROOT, ICARBON] = 50.0
    common = {
        "resp_maint_part_current": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "biomass": biomass,
        "t2m_longterm": np.asarray([293.15], dtype=np.float64),
        "z_soil": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        "rprof": np.ones((npts, nvm), dtype=np.float64) * 0.8,
        "sla_calc": np.ones((npts, nvm), dtype=np.float64) * 0.02,
        "coeff_maint_zero": np.ones((nvm, NPARTS), dtype=np.float64) * 0.01,
        "maint_resp_slope": np.zeros((nvm, 3), dtype=np.float64),
        "ext_coeff": np.ones(nvm, dtype=np.float64) * 0.5,
        "is_tree": np.zeros(nvm, dtype=bool),
        "dt_sechiba": 43200.0,
        "flood_frac": np.asarray([0.25], dtype=np.float64),
    }
    common["is_tree"][pft] = True
    entries = (
        {
            "t2m": np.asarray([293.15], dtype=np.float64),
            "stempdiag": np.asarray([[283.15, 284.15]], dtype=np.float64),
        },
        {
            "t2m": np.asarray([294.15], dtype=np.float64),
            "stempdiag": np.asarray([[285.15, 286.15]], dtype=np.float64),
        },
    )

    explicit = stomate_maintenance_fold_from_entries(entry_payloads=entries, **common)
    compiled = stomate_maintenance_fold_from_entries(entry_payloads=entries, use_jit=True, **common)

    assert compiled.ok
    np.testing.assert_allclose(np.asarray(compiled.resp_maint_part), np.asarray(explicit.resp_maint_part), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(compiled.resp_maint_radia), np.asarray(explicit.resp_maint_radia), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(compiled.flood_root_radia), np.asarray(explicit.flood_root_radia), rtol=0, atol=0)


def test_maintenance_runtime_fold_matches_debug_path_bit_exact():
    npts, nvm, pft = 1, 14, 13
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 100.0
    biomass[0, pft, IROOT, ICARBON] = 50.0
    common = {
        "entry_payloads": (
            {
                "t2m": np.asarray([293.15], dtype=np.float64),
                "stempdiag": np.asarray([[283.15, 284.15]], dtype=np.float64),
            },
            {
                "t2m": np.asarray([294.15], dtype=np.float64),
                "stempdiag": np.asarray([[285.15, 286.15]], dtype=np.float64),
            },
            {
                "t2m": np.asarray([295.15], dtype=np.float64),
                "stempdiag": np.asarray([[287.15, 288.15]], dtype=np.float64),
            },
        ),
        "resp_maint_part_current": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "biomass": biomass,
        "t2m_longterm": np.asarray([293.15], dtype=np.float64),
        "z_soil": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        "rprof": np.ones((npts, nvm), dtype=np.float64) * 0.8,
        "sla_calc": np.ones((npts, nvm), dtype=np.float64) * 0.02,
        "coeff_maint_zero": np.ones((nvm, NPARTS), dtype=np.float64) * 0.01,
        "maint_resp_slope": np.zeros((nvm, 3), dtype=np.float64),
        "ext_coeff": np.ones(nvm, dtype=np.float64) * 0.5,
        "is_tree": np.zeros(nvm, dtype=bool),
        "dt_sechiba": 43200.0,
        "flood_frac": np.asarray([0.25], dtype=np.float64),
    }
    common["is_tree"][pft] = True

    debug = stomate_maintenance_fold_from_entries(**common)
    runtime = stomate_maintenance_fold_from_entries(**common, retain_step_results=False)

    assert runtime.ok
    assert runtime.step_results == ()
    np.testing.assert_allclose(np.asarray(runtime.resp_maint_part), np.asarray(debug.resp_maint_part), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(runtime.resp_maint_radia), np.asarray(debug.resp_maint_radia), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(runtime.flood_root_radia), np.asarray(debug.flood_root_radia), rtol=0, atol=0)


def test_maintenance_fold_reports_missing_entry_fields_without_fabricating():
    npts, nvm = 1, 14
    current = np.zeros((npts, nvm, NPARTS), dtype=np.float64)
    payload = {"t2m": np.asarray([293.15], dtype=np.float64)}

    folded = stomate_maintenance_fold_from_entries(
        entry_payloads=(payload,),
        resp_maint_part_current=current,
        biomass=np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        t2m_longterm=np.asarray([293.15], dtype=np.float64),
        z_soil=np.asarray([0.0, 1.0], dtype=np.float64),
        rprof=np.ones((npts, nvm), dtype=np.float64),
        sla_calc=np.ones((npts, nvm), dtype=np.float64),
        coeff_maint_zero=np.zeros((nvm, NPARTS), dtype=np.float64),
        maint_resp_slope=np.zeros((nvm, 3), dtype=np.float64),
        ext_coeff=np.ones(nvm, dtype=np.float64),
        is_tree=np.zeros(nvm, dtype=bool),
        dt_sechiba=1800.0,
    )

    assert not folded.ok
    assert folded.failed_step == 0
    assert folded.missing_inputs == ("stempdiag",)
    np.testing.assert_allclose(np.asarray(folded.resp_maint_part), current)


def test_daily_process_fold_combines_accumulators_and_maintenance_boundary_fields():
    npts, nvm, pft = 1, 14, 13
    accumulator = _DailyAccumulatorState()
    entries = (
        _entry_payload(scale=1.0, t2m_min=279.0, t2m_max=290.0),
        _entry_payload(scale=2.0, t2m_min=277.0, t2m_max=292.0),
    )
    biomass = np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64)
    biomass[0, pft, ILEAF, ICARBON] = 100.0
    biomass[0, pft, IROOT, ICARBON] = 50.0
    coeff_maint_zero = np.zeros((nvm, NPARTS), dtype=np.float64)
    coeff_maint_zero[pft, :] = 0.01
    is_tree = np.zeros(nvm, dtype=bool)
    is_tree[pft] = True

    folded = stomate_daily_process_fold_from_entries(
        entry_payloads=entries,
        accumulator_state=accumulator,
        resp_maint_part_current=np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        do_slow_flags=(False, True),
        dt_sechiba=2.0,
        dt_stomate=4.0,
        biomass=biomass,
        t2m_longterm=np.asarray([293.15], dtype=np.float64),
        z_soil=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        rprof=np.ones((npts, nvm), dtype=np.float64) * 0.8,
        sla_calc=np.ones((npts, nvm), dtype=np.float64) * 0.02,
        coeff_maint_zero=coeff_maint_zero,
        maint_resp_slope=np.zeros((nvm, 3), dtype=np.float64),
        ext_coeff=np.ones(nvm, dtype=np.float64) * 0.5,
        is_tree=is_tree,
        flood_frac=np.asarray([0.25], dtype=np.float64),
    )

    assert folded.ok
    assert "gpp_daily" in folded.daily_fields
    assert "resp_maint_part" in folded.daily_fields
    assert "resp_maint_radia" in folded.daily_fields
    assert "flood_root_radia" in folded.daily_fields
    np.testing.assert_allclose(
        np.asarray(folded.daily_fields["gpp_daily"]),
        np.asarray(folded.accumulator.fields["gpp_daily"]),
    )
    np.testing.assert_allclose(
        np.asarray(folded.daily_fields["resp_maint_part"]),
        np.asarray(folded.maintenance.resp_maint_part),
    )


def test_daily_process_fold_single_pass_matches_default_path_bit_exact():
    npts, nvm, pft = 1, 14, 13
    accumulator = _DailyAccumulatorState()
    entries = (
        _entry_payload(scale=1.0, t2m_min=279.0, t2m_max=290.0),
        _entry_payload(scale=2.0, t2m_min=277.0, t2m_max=292.0),
        _entry_payload(scale=3.0, t2m_min=278.0, t2m_max=291.0),
    )
    common = {
        "entry_payloads": entries,
        "accumulator_state": accumulator,
        "resp_maint_part_current": np.zeros((npts, nvm, NPARTS), dtype=np.float64),
        "do_slow_flags": (False, False, True),
        "dt_sechiba": 2.0,
        "dt_stomate": 6.0,
        "biomass": np.zeros((npts, nvm, NPARTS, 1), dtype=np.float64),
        "t2m_longterm": np.asarray([293.15], dtype=np.float64),
        "z_soil": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        "rprof": np.ones((npts, nvm), dtype=np.float64) * 0.8,
        "sla_calc": np.ones((npts, nvm), dtype=np.float64) * 0.02,
        "coeff_maint_zero": np.zeros((nvm, NPARTS), dtype=np.float64),
        "maint_resp_slope": np.zeros((nvm, 3), dtype=np.float64),
        "ext_coeff": np.ones(nvm, dtype=np.float64) * 0.5,
        "is_tree": np.zeros(nvm, dtype=bool),
        "flood_frac": np.asarray([0.25], dtype=np.float64),
    }
    common["biomass"][0, pft, ILEAF, ICARBON] = 100.0
    common["biomass"][0, pft, IROOT, ICARBON] = 50.0
    common["coeff_maint_zero"][pft, :] = 0.01
    common["is_tree"][pft] = True

    default = stomate_daily_process_fold_from_entries(**common)
    single = stomate_daily_process_fold_from_entries(**common, single_pass=True)

    assert single.ok
    assert single.missing_inputs == default.missing_inputs
    assert single.accumulator.failed_step == default.accumulator.failed_step
    assert single.maintenance.failed_step == default.maintenance.failed_step
    assert set(single.daily_fields) == set(default.daily_fields)
    for name in single.daily_fields:
        np.testing.assert_allclose(np.asarray(single.daily_fields[name]), np.asarray(default.daily_fields[name]), rtol=0, atol=0)
