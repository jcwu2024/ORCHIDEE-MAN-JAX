from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax_orchidee.stomate.main as main  # noqa: E402


class _State(NamedTuple):
    marker: object


class _Pools(NamedTuple):
    pool: object


class _Prep(NamedTuple):
    turnover_littercalc: object
    bm_to_littercalc: object
    soil_mc_32l: object


def _install_owner_fakes(monkeypatch, calls, *, nland, slow):
    gpp_daily = np.arange(nland * 14, dtype=np.float64).reshape(nland, 14)
    resp_daily = np.ones((nland, 14, 12), dtype=np.float64) * 7.0
    daily_fields = {
        "gpp_daily": gpp_daily,
        "resp_maint_part": resp_daily,
        "t2m_min_daily": np.ones(nland) * 270.0,
        "t2m_max_daily": np.ones(nland) * 305.0,
        "precip_daily": np.ones(nland) * 4.0,
    }

    def daily_owner(**kwargs):
        calls.append(("daily", kwargs))
        return SimpleNamespace(
            ok=True,
            missing_inputs=(),
            daily_fields=daily_fields,
            maintenance=SimpleNamespace(
                step_results=(SimpleNamespace(resp_maint_part=np.ones((nland, 14, 12)) * 3.0),),
                flood_root_radia=np.ones((nland, 14)) * 2.0,
            ),
            accumulator=SimpleNamespace(
                step_results=(SimpleNamespace(local_prep="entry-prep"),),
            ),
        )

    def prep_owner(soil_mc, turnover, bm_to_litter, **kwargs):
        calls.append(("prep", turnover.copy(), bm_to_litter.copy(), kwargs))
        return _Prep(
            turnover_littercalc=turnover * 0.5,
            bm_to_littercalc=bm_to_litter * 0.5,
            soil_mc_32l=np.ones((nland, 32, 6)),
        )

    def leak_owner(**kwargs):
        calls.append(("leak", kwargs))
        assert kwargs["perma_peat"] is True
        assert kwargs["ok_cryoturb"] is False
        agg = np.arange(nland * 3 * 4, dtype=np.float64).reshape(nland, 3, 4)
        return SimpleNamespace(
            littercalc=_Pools(np.ones(nland)),
            soilcarbon=_Pools(np.ones(nland) * 2.0),
            interception_storage=np.ones(nland) * 9.0,
            doc_export=SimpleNamespace(doc_exp_agg=agg),
        )

    monkeypatch.setattr(main, "stomate_daily_process_fold_from_entries", daily_owner)
    monkeypatch.setattr(main, "stomate_littercalc_entry_prep", prep_owner)
    monkeypatch.setattr(main, "stomate_ok_leak_explicit", leak_owner)

    if slow:
        def memory_owner(state, **kwargs):
            calls.append(("season_memory", state, kwargs))
            return SimpleNamespace(state=_State("memory-owner"))

        def bio_owner(state, **kwargs):
            calls.append(("season_bio", state, kwargs))
            assert kwargs["marker"] == "memory-owner"
            return SimpleNamespace(state=_State("bio-owner"))

        def annual_owner(state, **kwargs):
            calls.append(("season_annual", state, kwargs))
            assert kwargs["marker"] == "bio-owner"
            return SimpleNamespace(state=_State("annual-owner"), herbivores=np.ones((nland, 14)))

        def lpj_owner(**kwargs):
            calls.append(("lpj", kwargs))
            assert kwargs["alloc_inputs"]["marker"] == "annual-owner"
            assert kwargs["prescribe_inputs"]["ok_dgvm"] is False
            assert kwargs["post_npp_inputs"]["wire_cover"] is True
            assert kwargs["post_npp_inputs"]["wire_harvest_agri"] is True
            assert kwargs["post_npp_inputs"]["wire_vmax"] is True
            assert kwargs["post_npp_inputs"]["wire_modelout"] is True
            np.testing.assert_array_equal(kwargs["post_npp_inputs"]["gpp_daily"], gpp_daily)
            np.testing.assert_array_equal(kwargs["post_npp_inputs"]["resp_maint_part"], resp_daily)
            return SimpleNamespace(
                post_npp=SimpleNamespace(
                    modelout_fields={"NPP": np.ones((nland, 14))},
                    daily_carbon=SimpleNamespace(
                        npp_update=SimpleNamespace(
                            resp_maint=np.ones((nland, 14)) * 2.0,
                            resp_growth=np.ones((nland, 14)) * 3.0,
                        )
                    ),
                    harvest_agri=SimpleNamespace(harvest_above=np.ones(nland) * 0.25),
                )
            )

        monkeypatch.setattr(main, "season_memory_step", memory_owner)
        monkeypatch.setattr(main, "season_biometeorology_step", bio_owner)
        monkeypatch.setattr(main, "season_annual_step", annual_owner)
        monkeypatch.setattr(main, "stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit", lpj_owner)


def _base_kwargs(nland):
    return {
        "entry_payload": {"gpp": np.ones((nland, 14)), "date": 17},
        "accumulator_state": object(),
        "maintenance_inputs": {"resp_maint_part_current": np.zeros((nland, 14, 12)), "token": 1},
        "turnover_daily": np.ones((nland, 14, 12, 1)) * 11.0,
        "bm_to_litter": np.ones((nland, 14, 12, 1)) * 13.0,
        "ok_leak_inputs": {
            "soil_mc": np.ones((nland, 11, 6)),
            "carbon_32l": np.ones((nland, 3, 14, 32, 1)),
        },
        "activity_control_inputs": {
            "tdeep": np.ones((nland, 32, 14)) * 278.15,
            "hsdeep": np.ones((nland, 32, 14)) * 0.6,
            "shumdiag_peat": np.ones((nland, 11)) * 0.7,
            "shumdiag_man": np.ones((nland, 11)) * 0.8,
            "zz_deep": np.linspace(0.01, 3.2, 32),
            "poor_soils": np.linspace(0.0, 0.2, nland),
            "frozen_respiration_func": 0,
            "is_peat": np.array([False] * 13 + [True]),
        },
        "dt_sechiba": 1800.0,
        "dt_stomate": 86400.0,
    }


def test_stomate_output_fluxes_matches_fortran_5033_5049():
    nland, nvm = 2, 14
    shape = (nland, nvm)
    cover = np.linspace(0.0, 1.0, nland * nvm).reshape(shape)
    resp_maint_radia = np.arange(nland * nvm, dtype=np.float64).reshape(shape) + 1.0
    resp_growth_d = np.full(shape, 8.0)
    resp_hetero_radia = np.full(shape, 3.0)
    co2_fire = np.full(shape, 6.0)
    co2_to_bm_dgvm = np.full(shape, 2.0)
    gpp = np.full(shape, 0.25)

    result = main.stomate_output_fluxes(
        resp_maint_radia=resp_maint_radia,
        resp_growth_d=resp_growth_d,
        resp_hetero_radia=resp_hetero_radia,
        co2_fire=co2_fire,
        co2_to_bm_dgvm=co2_to_bm_dgvm,
        veget_cov_max=cover,
        gpp=gpp,
        t2m_month=np.array([273.15, 283.15]),
        dt_sechiba=1800.0,
        one_day=86400.0,
    )

    expected_maint = resp_maint_radia * cover
    expected_maint[:, 0] = 0.0
    expected_growth = resp_growth_d * cover * 1800.0 / 86400.0
    expected_hetero = resp_hetero_radia * cover
    expected_co2 = (
        expected_hetero
        + expected_maint
        + expected_growth
        + (co2_fire - co2_to_bm_dgvm) * cover / 86400.0
        - gpp
    )
    np.testing.assert_allclose(result.resp_maint, expected_maint)
    np.testing.assert_allclose(result.resp_growth, expected_growth)
    np.testing.assert_allclose(result.resp_hetero, expected_hetero)
    np.testing.assert_allclose(result.co2_flux, expected_co2)
    np.testing.assert_allclose(result.temp_growth, np.array([0.0, 10.0]))


def test_stomate_output_fluxes_rejects_inconsistent_shapes():
    shape = (2, 14)
    kwargs = {
        "resp_maint_radia": np.ones(shape),
        "resp_growth_d": np.ones(shape),
        "resp_hetero_radia": np.ones(shape),
        "co2_fire": np.ones(shape),
        "co2_to_bm_dgvm": np.ones(shape),
        "veget_cov_max": np.ones(shape),
        "gpp": np.ones(shape),
        "t2m_month": np.ones(2),
        "dt_sechiba": 1800.0,
    }
    with pytest.raises(ValueError, match="share shape"):
        main.stomate_output_fluxes(**{**kwargs, "resp_growth_d": np.ones((1, 14))})
    with pytest.raises(ValueError, match="t2m_month"):
        main.stomate_output_fluxes(**{**kwargs, "t2m_month": np.ones(3)})


def test_fixed_structural_switches_reject_unreachable_branches():
    main.validate_stomate_main_pft14_switches(main.StomateMainPFT14Switches())
    with pytest.raises(ValueError, match="ok_pc=True"):
        main.validate_stomate_main_pft14_switches(replace(main.StomateMainPFT14Switches(), ok_pc=True))
    with pytest.raises(ValueError, match="ok_laidev"):
        main.validate_stomate_main_pft14_switches(
            replace(main.StomateMainPFT14Switches(), ok_laidev=(False,) * 13 + (True,))
        )
    with pytest.raises(ValueError, match="stomate_forcing_name"):
        main.validate_stomate_main_pft14_switches(
            replace(main.StomateMainPFT14Switches(), stomate_forcing_name="forcing.nc")
        )
    for name, value in (("ok_pheno", False), ("ok_rotate", True), ("nitrogen_use", True)):
        with pytest.raises(ValueError, match=name):
            main.validate_stomate_main_pft14_switches(
                replace(main.StomateMainPFT14Switches(), **{name: value})
            )


def test_fixed_path_control_matches_daily_and_year_end_source_order():
    switches = main.StomateMainPFT14Switches()
    fast = main.stomate_fixed_path_control(
        initialized=True,
        do_slow=False,
        end_of_year=False,
        dt_stomate=86400.0,
        one_day=86400.0,
        tday_counter=np.array([17.0, 29.0]),
        vday_counter=np.array([7.0, 11.0]),
        switches=switches,
    )
    np.testing.assert_array_equal(fast.tday_counter, np.array([17.0, 29.0]))
    np.testing.assert_array_equal(fast.vday_counter, np.array([7.0, 11.0]))
    assert fast.phenology_enabled is True

    daily = main.stomate_fixed_path_control(
        initialized=True,
        do_slow=True,
        end_of_year=False,
        dt_stomate=86400.0,
        one_day=86400.0,
        tday_counter=np.array([17.0, 29.0]),
        vday_counter=np.array([7.0, 11.0]),
        switches=switches,
    )
    np.testing.assert_array_equal(daily.tday_counter, np.array([18.0, 30.0]))
    np.testing.assert_array_equal(daily.vday_counter, np.array([8.0, 12.0]))

    year_end = main.stomate_fixed_path_control(
        initialized=True,
        do_slow=True,
        end_of_year=True,
        dt_stomate=86400.0,
        one_day=86400.0,
        tday_counter=np.array([364.0, 364.0]),
        vday_counter=np.array([364.0, 364.0]),
        switches=switches,
    )
    np.testing.assert_array_equal(year_end.tday_counter, np.zeros(2))
    np.testing.assert_array_equal(year_end.vday_counter, np.full(2, 365.0))


def test_fixed_path_control_enforces_initialization_guard_before_state_change():
    with pytest.raises(RuntimeError, match="Initialization not yet done"):
        main.stomate_fixed_path_control(
            initialized=False,
            do_slow=True,
            end_of_year=True,
            dt_stomate=86400.0,
            one_day=86400.0,
            tday_counter=364.0,
            vday_counter=364.0,
            switches=main.StomateMainPFT14Switches(),
        )


def test_non_slow_multiland_step_preserves_old_turnover_order_and_output_boundary(monkeypatch):
    calls = []
    _install_owner_fakes(monkeypatch, calls, nland=3, slow=False)
    result = main.stomate_main_pft14_step(**_base_kwargs(3), do_slow=False)

    assert [call[0] for call in calls] == ["daily", "prep", "leak"]
    np.testing.assert_array_equal(calls[1][1], np.ones((3, 14, 12, 1)) * 11.0)
    np.testing.assert_array_equal(calls[2][1]["turnover"], np.ones((3, 14, 12, 1)) * 5.5)
    np.testing.assert_array_equal(result.daily_state["precip_daily"], np.ones(3) * 4.0)
    assert result.lpj is None
    assert len(result.output.xios) == 9
    assert result.output.history == {}
    assert result.output.xios["zz_EXP_DIC_RUNOFF"].shape == (3,)


def test_slow_step_runs_season_then_lpj_and_resets_after_output(monkeypatch):
    calls = []
    _install_owner_fakes(monkeypatch, calls, nland=2, slow=True)
    kwargs = _base_kwargs(2)
    kwargs.update(
        season_memory_state="memory-state",
        season_memory_inputs={},
        season_biometeorology_state="bio-state",
        season_biometeorology_inputs={"marker": "stale"},
        season_annual_state="annual-state",
        season_annual_inputs={"marker": "stale"},
        lpj_inputs={
            "prescribe_inputs": {},
            "constraints_inputs": {},
            "phenology_inputs": {},
            "alloc_inputs": {"marker": "stale"},
            "post_npp_inputs": {},
        },
        monthly_carbon_state=main.StomateMonthlyCarbonState(
            np.zeros((2, 14)),
            np.zeros(2),
            np.zeros((2, 2)),
        ),
        monthly_carbon_inputs={
            "resp_hetero_d": np.ones((2, 14)) * 4.0,
            "veget_cov_max": np.ones((2, 14)) * 0.5,
            "contfrac": np.array([0.8, 0.9]),
            "convflux": np.ones((2, 2)) * 0.1,
            "cflux_prod10": np.ones((2, 2)) * 0.2,
            "cflux_prod100": np.ones((2, 2)) * 0.3,
            "resolution": np.ones((2, 2)) * 1000.0,
            "totfrac_nobio": np.array([0.1, 0.2]),
            "end_of_month": True,
        },
    )
    result = main.stomate_main_pft14_step(**kwargs, do_slow=True)

    assert [call[0] for call in calls] == [
        "daily",
        "prep",
        "leak",
        "season_memory",
        "season_bio",
        "season_annual",
        "lpj",
    ]
    np.testing.assert_array_equal(result.output.history["NPP"], np.ones((2, 14)))
    np.testing.assert_array_equal(result.daily_state["precip_daily"], np.zeros(2))
    np.testing.assert_array_equal(result.daily_state["t2m_min_daily"], np.ones(2) * 1.0e33)
    np.testing.assert_array_equal(result.daily_state["t2m_max_daily"], np.ones(2) * -1.0e33)
    assert result.monthly_carbon is not None
    np.testing.assert_array_equal(result.carbon_state["co2_flux_monthly"], np.zeros((2, 14)))
    np.testing.assert_array_equal(result.output.history["NONBIOFRAC"], np.array([0.1, 0.2]))
    np.testing.assert_array_equal(result.output.xios["nep"], result.monthly_carbon.nep)
    np.testing.assert_array_equal(result.output.history["nep"], result.monthly_carbon.nep)


def test_runtime_owned_inputs_cannot_be_overridden(monkeypatch):
    calls = []
    _install_owner_fakes(monkeypatch, calls, nland=1, slow=False)
    kwargs = _base_kwargs(1)
    kwargs["ok_leak_inputs"]["turnover"] = np.zeros((1, 14, 12, 1))
    with pytest.raises(ValueError, match="orchestration-owned fields: turnover"):
        main.stomate_main_pft14_step(**kwargs, do_slow=False)


def test_activity_control_precomputation_extends_layers_and_selects_peat_pfts():
    nland, nslm, ndeep = 2, 3, 12
    tdeep = np.ones((nland, ndeep, 14), dtype=np.float64) * 278.15
    hsdeep = np.linspace(0.2, 0.8, nland * ndeep * 14).reshape(nland, ndeep, 14)
    peat = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    man = peat + 0.1
    is_peat = np.array([False, True] + [False] * 12)

    result = main.stomate_activity_control_precomputation(
        tdeep=tdeep,
        hsdeep=hsdeep,
        shumdiag_peat=peat,
        shumdiag_man=man,
        zz_deep=np.linspace(0.01, 1.2, ndeep),
        poor_soils=np.array([0.0, 0.5]),
        frozen_respiration_func=0,
        is_peat=is_peat,
    )

    expected_peat = np.concatenate((peat, np.repeat(peat[:, -1:], ndeep - nslm, axis=1)), axis=1)
    expected_man = np.concatenate((man, np.repeat(man[:, -1:], ndeep - nslm, axis=1)), axis=1)
    np.testing.assert_allclose(result.mc_peat, expected_peat)
    np.testing.assert_allclose(result.mc_man, expected_man)
    np.testing.assert_allclose(result.hsdeep_new[:, :, 1], expected_peat)
    np.testing.assert_allclose(result.hsdeep_new[:, :, 0], hsdeep[:, :, 0])
    factor_ratio = np.asarray(result.controls.prmfrst_soilc_tempctrl[1, :, 0]) / np.asarray(
        result.controls.prmfrst_soilc_tempctrl[0, :, 0]
    )
    np.testing.assert_allclose(factor_ratio, 0.75)
    assert "lines 3035-3124" in result.provenance[0]


def test_nep_monthly_non_end_of_month_accumulates_without_reset():
    state = main.StomateMonthlyCarbonState(
        np.ones((1, 3)) * 10.0,
        np.array([2.0]),
        np.ones((1, 2)) * 3.0,
    )
    result = main.stomate_nep_monthly_step(
        state,
        resp_maint_d=np.ones((1, 3)) * 2.0,
        resp_growth_d=np.ones((1, 3)) * 3.0,
        resp_hetero_d=np.ones((1, 3)) * 4.0,
        co2_fire=np.ones((1, 3)) * 5.0,
        co2_to_bm_dgvm=np.ones((1, 3)),
        gpp_daily=np.ones((1, 3)) * 6.0,
        veget_cov_max=np.array([[0.2, 0.3, 0.5]]),
        contfrac=np.array([0.8]),
        harvest_above=np.array([0.25]),
        convflux=np.array([[0.1, 0.2]]),
        cflux_prod10=np.array([[0.3, 0.4]]),
        cflux_prod100=np.array([[0.5, 0.6]]),
        resolution=np.array([[10.0, 20.0]]),
        totfrac_nobio=np.array([0.1]),
        end_of_month=False,
    )

    np.testing.assert_allclose(result.co2_flux_daily, 7.0)
    np.testing.assert_allclose(result.nep, 7.0 / 1.0e3 / 86400.0 * 0.8)
    np.testing.assert_allclose(result.state.co2_flux_monthly, 17.0)
    np.testing.assert_allclose(result.state.harvest_above_monthly, 2.25)
    np.testing.assert_allclose(result.state.cflux_prod_monthly, [[3.9, 4.2]])
    assert result.co2_flux_monthly_history is None


def test_nep_monthly_end_of_month_captures_history_scales_totals_then_resets():
    state = main.StomateMonthlyCarbonState(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    zeros = np.zeros((2, 3))
    result = main.stomate_nep_monthly_step(
        state,
        resp_maint_d=np.ones((2, 3)),
        resp_growth_d=zeros,
        resp_hetero_d=zeros,
        co2_fire=zeros,
        co2_to_bm_dgvm=zeros,
        gpp_daily=zeros,
        veget_cov_max=np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]),
        contfrac=np.array([0.5, 0.25]),
        harvest_above=np.array([0.5, 0.25]),
        convflux=np.ones((2, 2)),
        cflux_prod10=np.ones((2, 2)) * 2.0,
        cflux_prod100=np.ones((2, 2)) * 3.0,
        resolution=np.array([[10.0, 20.0], [30.0, 40.0]]),
        totfrac_nobio=np.array([0.1, 0.2]),
        end_of_month=True,
    )

    accumulated_co2 = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    area = np.array([100.0, 300.0])
    expected_net_co2 = (
        3.0 * area[0] * 0.3 + 4.0 * area[0] * 0.5
        + 6.0 * area[1] * 0.4 + 7.0 * area[1] * 0.5
    ) * 1.0e-15
    accumulated_products = np.array([[7.0, 8.0], [9.0, 10.0]])
    expected_products = np.sum(np.sum(accumulated_products, axis=1) * area) * 1.0e-15
    expected_harvest = np.sum(np.array([1.5, 2.25]) * area) * 1.0e-15
    np.testing.assert_allclose(result.co2_flux_monthly_history, accumulated_co2)
    np.testing.assert_allclose(result.net_co2_flux_monthly_sum, expected_net_co2)
    np.testing.assert_allclose(result.net_cflux_prod_monthly_tot, expected_products)
    np.testing.assert_allclose(result.net_harvest_above_monthly_tot, expected_harvest)
    np.testing.assert_allclose(
        result.net_biosp_prod_monthly_tot,
        expected_net_co2 + expected_products + expected_harvest,
    )
    np.testing.assert_array_equal(result.state.co2_flux_monthly, np.zeros((2, 3)))
    np.testing.assert_array_equal(result.state.harvest_above_monthly, np.zeros(2))
    np.testing.assert_array_equal(result.state.cflux_prod_monthly, np.zeros((2, 2)))
    assert "lines 4851-4937" in result.provenance[0]
