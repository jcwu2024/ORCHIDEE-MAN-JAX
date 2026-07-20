from types import SimpleNamespace

import numpy as np
import pytest

import jax_orchidee.stomate.integration as integration
from jax_orchidee.stomate.carbon_kernels import IABOVE, ICARBON, NLITT


def _state(nland=2):
    nvm = 14
    nparts = 4
    nelements = 1
    shape = (nland, nvm)
    biomass = np.ones(shape + (nparts, nelements))
    litter = np.zeros((nland, NLITT, nvm, 2, nelements))
    litter[:, :, :, IABOVE, ICARBON] = np.arange(1, NLITT + 1)[None, :, None]
    return {
        "biomass": biomass,
        "veget_max": np.full(shape, 1.0 / nvm),
        "gpp_daily": np.full(shape, 3.0),
        "resp_maint_part": np.ones(shape + (nparts,)),
        "pft_present": np.ones(shape, dtype=bool),
        "everywhere": np.zeros(shape),
        "when_growthinit": np.ones(shape),
        "leaf_frac": np.ones(shape + (4,)) / 4.0,
        "leaf_age": np.ones(shape + (4,)),
        "age": np.ones(shape),
        "ind": np.ones(shape),
        "cn_ind": np.ones(shape),
        "adapted": np.ones(shape),
        "regenerate": np.ones(shape),
        "npp_longterm": np.ones(shape),
        "turnover_longterm": np.ones(shape),
        "lm_lastyearmax": np.ones(shape),
        "rip_time": np.ones(shape),
        "height": np.ones(shape),
        "tmin_spring_time": np.ones(shape),
        "turnover_time": np.ones(shape),
        "litter": litter,
        "carbon": np.ones((nland, 3, nvm)),
        "fuel_1hr": np.ones(shape + (2, nelements)),
        "fuel_10hr": np.ones(shape + (2, nelements)),
        "fuel_100hr": np.ones(shape + (2, nelements)),
        "fuel_1000hr": np.ones(shape + (2, nelements)),
        "resp_hetero": np.ones(shape),
        "deepC_a": np.ones((nland, 3, nvm)),
        "deepC_s": np.ones((nland, 3, nvm)),
        "deepC_p": np.ones((nland, 3, nvm)),
    }


def _fake_chain(state, captured):
    shape = state["gpp_daily"].shape
    biomass_shape = state["biomass"].shape
    leaf_shape = state["leaf_age"].shape
    final_biomass = np.full(biomass_shape, 21.0)
    final_leaf_age = np.full(leaf_shape, 22.0)
    final_leaf_frac = np.full(leaf_shape, 0.25)
    cover = SimpleNamespace(
        biomass=final_biomass,
        veget_max=np.full(shape, 23.0),
        litter=np.full_like(state["litter"], 24.0),
        litter_avail=np.full((shape[0], NLITT, shape[1]), 25.0),
        litter_not_avail=np.full((shape[0], NLITT, shape[1]), 26.0),
        carbon=np.full_like(state["carbon"], 27.0),
        fuel_1hr=np.full_like(state["fuel_1hr"], 28.0),
        fuel_10hr=np.full_like(state["fuel_10hr"], 29.0),
        fuel_100hr=np.full_like(state["fuel_100hr"], 30.0),
        fuel_1000hr=np.full_like(state["fuel_1000hr"], 31.0),
        deepC_a=np.full_like(state["deepC_a"], 32.0),
        deepC_s=np.full_like(state["deepC_s"], 33.0),
        deepC_p=np.full_like(state["deepC_p"], 34.0),
        co2_to_bm=np.full(shape, 35.0),
        co2_fire=np.zeros(shape),
        resp_hetero=np.full(shape, 36.0),
        gpp_daily=np.full(shape, 37.0),
    )
    harvest = SimpleNamespace(
        turnover_daily=np.full(biomass_shape, 38.0),
        bm_to_litter=np.full(biomass_shape, 39.0),
        harvest_above=np.full(shape[:1], 40.0),
    )
    vmax = SimpleNamespace(
        leaf_age=final_leaf_age,
        leaf_frac=final_leaf_frac,
        vcmax=np.full(shape, 41.0),
    )
    kill = SimpleNamespace(
        pft_present=np.ones(shape, dtype=bool),
        everywhere=np.full(shape, 42.0),
        when_growthinit=np.full(shape, 43.0),
        ind=np.full(shape, 44.0),
        cn_ind=np.full(shape, 45.0),
        npp_longterm=np.full(shape, 46.0),
        rip_time=np.full(shape, 47.0),
    )
    turnover = SimpleNamespace(
        senescence=np.zeros(shape, dtype=bool),
        age=np.full(shape, 48.0),
        turnover_time=np.full(shape, 49.0),
        c_export=np.full(shape, 50.0),
    )
    npp = SimpleNamespace(
        resp_maint=np.full(shape, 51.0),
        resp_growth=np.full(shape, 52.0),
        npp=np.full(shape, 53.0),
    )
    post = SimpleNamespace(
        cover=cover,
        harvest_agri=harvest,
        vmax=vmax,
        kill_after_gap=kill,
        turnover=turnover,
        daily_carbon=SimpleNamespace(
            npp_update=npp,
            age_sla=SimpleNamespace(sla_calc=np.full(shape, 54.0)),
        ),
        lai_after_setlai=np.full(shape, 55.0),
        output_diagnostics=SimpleNamespace(carb_mass_total=np.full(shape[:1], 58.0)),
    )
    return SimpleNamespace(
        constraints=SimpleNamespace(adapted=np.full(shape, 56.0), regenerate=np.full(shape, 57.0)),
        post_npp=post,
    )


def _inputs(state):
    return {
        "state": state,
        "prescribe_inputs": {"natural": (False,) * 13 + (True,)},
        "constraints_inputs": {},
        "phenology_inputs": {},
        "alloc_inputs": {},
        "post_npp_inputs": {"litter_avail_frac": np.full((len(state["biomass"]), NLITT, 14), 0.25)},
        "firstcall": True,
    }


def test_fixed_owner_rejects_nonpaper_structural_branch():
    """Fortran: StomateLpj lines 1193-1215; paper path has fire disabled."""

    with pytest.raises(NotImplementedError, match="disable_fire=False"):
        integration.stomate_lpj_main_pft14_fixed(
            **_inputs(_state()),
            switches=integration.StomateLpjPFT14FixedSwitches(disable_fire=False),
        )


def test_fixed_owner_rejects_caller_override_of_process_state():
    """Fortran: StomateLpj lines 873-901 initialize and own daily state."""

    inputs = _inputs(_state())
    inputs["post_npp_inputs"]["bm_to_litter"] = np.ones_like(inputs["state"]["biomass"])
    with pytest.raises(ValueError, match="bm_to_litter"):
        integration.stomate_lpj_main_pft14_fixed(**inputs)


def test_fixed_owner_threads_source_order_and_writes_final_multiland_state(monkeypatch):
    """Fortran: StomateLpj lines 944-1559 and state/output lines 1578-2437."""

    state = _state(nland=2)
    captured = {}

    def fake_owner(**kwargs):
        captured.update(kwargs)
        return _fake_chain(state, captured)

    monkeypatch.setattr(
        integration,
        "stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit",
        fake_owner,
    )
    result = integration.stomate_lpj_main_pft14_fixed(**_inputs(state))

    post = captured["post_npp_inputs"]
    np.testing.assert_array_equal(post["bm_to_litter"], 0.0)
    np.testing.assert_array_equal(post["co2_to_bm"], 0.0)
    np.testing.assert_allclose(post["cover_litter_avail"], state["litter"][:, :, :, IABOVE, ICARBON] * 0.25)
    assert post["ok_dgvm"] is False
    assert post["lpj_gap_const_mort"] is True
    assert post["wire_cover"] is True
    assert post["update_peatfrac"] is False
    assert post["wire_harvest_agri"] is True
    assert captured["prescribe_inputs"]["firstcall"] is True
    assert captured["phenology_inputs"]["pheno_model"] == ("none",) * 14

    after = result.state_after
    assert after["biomass"].shape[:2] == (2, 14)
    np.testing.assert_array_equal(after["biomass"], 21.0)
    np.testing.assert_array_equal(after["turnover_daily"], 38.0)
    np.testing.assert_array_equal(after["bm_to_litter"], 39.0)
    np.testing.assert_array_equal(after["vcmax"], 41.0)
    np.testing.assert_array_equal(after["resp_maint"], 51.0)
    np.testing.assert_array_equal(after["npp_daily"], 53.0)
    np.testing.assert_array_equal(after["lai"], 55.0)
    np.testing.assert_array_equal(after["adapted"], 56.0)
    np.testing.assert_array_equal(after["soilc_total"], 99.0)
    np.testing.assert_array_equal(after["carb_mass_total"], 58.0)
    assert result.process_order.index("cover") < result.process_order.index("harvest")
    assert result.process_order.index("harvest") < result.process_order.index("setlai")
    assert result.provenance == integration.STOMATE_LPJ_PFT14_PROVENANCE
