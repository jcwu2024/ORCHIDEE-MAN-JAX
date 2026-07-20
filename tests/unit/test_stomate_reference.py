from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.carbon_kernels import ICARBON, NPARTS
from jax_orchidee.sechiba.slowproc import (
    read_slowproc_restart_entry_state,
    slowproc_dyn_peat_disabled_entry_state,
    slowproc_derivvar_explicit,
    slowproc_erosion_daily_zero_entry_state,
    slowproc_fire_disabled_entry_state,
    slowproc_no_lcc_entry_state,
    slowproc_static_entry_state,
    slowproc_thermosoil_entry_init_state,
)
from jax_orchidee.stomate.entry import (
    assemble_stomate_main_payload,
    stomate_erosion_disabled_erodepth_entry_source,
    stomate_restart_entry_source,
    stomate_slowproc_dyn_peat_disabled_entry_source,
    stomate_slowproc_derivvar_entry_source,
    stomate_slowproc_erosion_daily_zero_entry_source,
    stomate_slowproc_fire_disabled_entry_source,
    stomate_slowproc_no_lcc_entry_source,
    stomate_slowproc_restart_entry_source,
    stomate_slowproc_static_entry_source,
    stomate_slowproc_thermosoil_entry_source,
)
from jax_orchidee.stomate.reference import (
    OK_PC_RESTART_GAS_FIELDS,
    RESTART_ENTRY_STATE_FIELDS,
    encode_restart_pft_bool_field,
    find_stomate_reference_files,
    inventory_netcdf,
    read_restart_carbon_32l,
    read_restart_deep_carbon_pool,
    read_restart_deep_carbon_total,
    read_restart_doc,
    read_restart_interception_storage,
    read_restart_leaf_age_field,
    read_restart_legacy_carbon,
    read_restart_lignin_struc_below,
    read_restart_lignin_struc_legacy,
    read_restart_litter_above,
    read_restart_litter_below,
    read_restart_litter_legacy,
    read_restart_product_pool,
    read_restart_product_total,
    read_restart_pft_vertical_field,
    read_variables,
    read_stomate_ok_pc_restart_gas_state,
    read_stomate_restart_entry_state,
    read_stomate_restart_season_state,
    read_stomate_daily_accumulator_state,
    variable_presence,
)


def test_restart_pft_bool_encoding_matches_fortran_need_adjacent_roundtrip():
    flags = np.asarray([[True, False, True], [False, False, True]], dtype=bool)

    encoded = encode_restart_pft_bool_field(flags)

    np.testing.assert_allclose(encoded, [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert encoded.dtype == np.float64
    decoded = encoded >= 0.5
    np.testing.assert_array_equal(decoded, flags)
    threshold_values = np.asarray([[0.49, 0.5, 1.0, 0.0]], dtype=np.float64)
    np.testing.assert_array_equal(threshold_values >= 0.5, [[False, True, True, False]])


def test_restart_entry_inventory_contains_source_backed_state_variables():
    files = find_stomate_reference_files(ROOT)
    inventory = inventory_netcdf(files.restart)
    presence = variable_presence(
        files.restart,
        RESTART_ENTRY_STATE_FIELDS
        + (
            "lai",
            "height",
            "deadleaf_cover",
            "veget_max_new",
            "vegetnew_firstday",
            "sat_duration",
            "thawed_humidity",
            "depth_organic_soil",
            "deepC_a",
            "deepC_s",
            "deepC_p",
        ),
    )

    for name in RESTART_ENTRY_STATE_FIELDS:
        assert presence[name], name
    for name in ("lai", "height", "deadleaf_cover", "veget_max_new", "vegetnew_firstday", "sat_duration"):
        assert not presence[name], name
    for name in ("thawed_humidity", "depth_organic_soil", "deepC_a", "deepC_s", "deepC_p"):
        assert presence[name], name

    assert inventory["litter_above"].dimensions == ("time", "m_a", "l_d", "z_b", "y", "x")
    assert inventory["litter_below"].dimensions == ("time", "n_a", "m_c", "l_d", "z_b", "y", "x")
    assert inventory["freedoc"].dimensions == ("time", "n_a", "m_d", "l_e", "z_a", "y", "x")
    assert inventory["carbon_32l_a"].shape == (1, 32, 14, 1, 1)
    assert inventory["deepC_a"].shape == (1, 14, 32, 1, 1)
    assert inventory["fixed_cryoturb_depth"].dimensions == ("time", "z_a", "y", "x")


def test_restart_state_axis_adapters_match_stomate_main_fortran_shapes():
    files = find_stomate_reference_files(ROOT)

    leaf_age = read_restart_leaf_age_field(files.restart, "leaf_age")
    litter_above = read_restart_litter_above(files.restart)
    litter_below = read_restart_litter_below(files.restart)
    litter = read_restart_litter_legacy(files.restart)
    lignin_struc = read_restart_lignin_struc_legacy(files.restart)
    carbon_32l = read_restart_carbon_32l(files.restart)
    carbon = read_restart_legacy_carbon(files.restart)
    deepC_a = read_restart_deep_carbon_pool(files.restart, "deepC_a")
    deepC_s = read_restart_deep_carbon_pool(files.restart, "deepC_s")
    deepC_p = read_restart_deep_carbon_pool(files.restart, "deepC_p")
    soilc_total = read_restart_deep_carbon_total(files.restart)
    doc = read_restart_doc(files.restart)
    interception_storage = read_restart_interception_storage(files.restart)
    lignin_below = read_restart_lignin_struc_below(files.restart)
    prod10_total = read_restart_product_total(files.restart, "prod10")
    prod100_total = read_restart_product_total(files.restart, "prod100")
    prod10 = read_restart_product_pool(files.restart, "prod10", 11)
    prod100 = read_restart_product_pool(files.restart, "prod100", 101)
    flux10 = read_restart_product_pool(files.restart, "flux10", 10)
    flux100 = read_restart_product_pool(files.restart, "flux100", 100)

    assert leaf_age.shape == (1, 14, 4)
    assert litter_above.shape == (1, 2, 14, 1)
    assert litter_below.shape == (1, 2, 14, 32, 1)
    assert litter.shape == (1, 2, 14, 2, 1)
    assert lignin_struc.shape == (1, 14, 2)
    assert carbon.shape == (1, 3, 14)
    assert carbon_32l.shape == (1, 3, 14, 32)
    assert deepC_a.shape == (1, 32, 14)
    assert deepC_s.shape == (1, 32, 14)
    assert deepC_p.shape == (1, 32, 14)
    assert soilc_total.shape == (1, 32, 14)
    assert doc.shape == (1, 14, 32, 2, 7, 1)
    assert interception_storage.shape == (1, 14, 1)
    assert lignin_below.shape == (1, 14, 32)
    assert prod10_total.shape == (1,)
    assert prod100_total.shape == (1,)
    assert prod10.shape == (1, 11, 2)
    assert prod100.shape == (1, 101, 2)
    assert flux10.shape == (1, 10, 2)
    assert flux100.shape == (1, 100, 2)

    assert np.isfinite(litter_above).all()
    assert np.isfinite(litter_below).all()
    assert np.isfinite(carbon_32l).all()
    assert np.isfinite(carbon).all()
    assert np.isfinite(doc).all()
    assert np.isfinite(interception_storage).all()
    np.testing.assert_allclose(prod10_total, np.sum(read_variables(files.restart, ("prod10",))["prod10"][0]))
    np.testing.assert_allclose(prod100_total, np.sum(read_variables(files.restart, ("prod100",))["prod100"][0]))
    np.testing.assert_allclose(prod10_total, np.sum(prod10, axis=(1, 2)))
    np.testing.assert_allclose(prod100_total, np.sum(prod100, axis=(1, 2)))
    np.testing.assert_allclose(soilc_total, deepC_a + deepC_s + deepC_p)


def test_read_stomate_ok_pc_restart_gas_state_preserves_restart_axes():
    files = find_stomate_reference_files(ROOT)
    inventory = inventory_netcdf(files.restart)
    presence = variable_presence(files.restart, OK_PC_RESTART_GAS_FIELDS)

    for name in OK_PC_RESTART_GAS_FIELDS:
        assert presence[name], name
        assert inventory[name].dimensions[:2] == ("time", "l_d")

    gas = read_stomate_ok_pc_restart_gas_state(files.restart)
    assert gas.O2_soil.shape == (1, 32, 14)
    assert gas.CH4_soil.shape == (1, 32, 14)
    assert gas.O2_snow.shape == (1, 3, 14)
    assert gas.CH4_snow.shape == (1, 3, 14)
    np.testing.assert_allclose(gas.O2_soil, read_restart_pft_vertical_field(files.restart, "O2_soil"))
    np.testing.assert_allclose(gas.CH4_soil, read_restart_pft_vertical_field(files.restart, "CH4_soil"))
    np.testing.assert_allclose(gas.O2_snow, read_restart_pft_vertical_field(files.restart, "O2_snow"))
    np.testing.assert_allclose(gas.CH4_snow, read_restart_pft_vertical_field(files.restart, "CH4_snow"))


def test_read_stomate_restart_entry_state_preserves_restart_values_without_fills():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)
    inventory = inventory_netcdf(files.restart)

    assert state.biomass.shape == (1, 14, NPARTS, 1)
    assert state.resp_maint_part.shape == (1, 14, NPARTS)
    assert state.leaf_frac.shape == (1, 14, 4)
    assert state.age.shape == (1, 14)
    assert state.sla_calc.shape == (1, 14)
    assert state.pft_present.shape == (1, 14)
    assert state.gpp_daily.shape == (1, 14)
    assert state.npp_daily.shape == (1, 14)
    assert state.turnover_daily.shape == (1, 14, NPARTS, 1)
    assert state.resp_maint.shape == (1, 14)
    assert state.resp_growth.shape == (1, 14)
    assert state.ind.shape == (1, 14)
    assert state.adapted.shape == (1, 14)
    assert state.regenerate.shape == (1, 14)
    assert state.turnover_longterm.shape == (1, 14, NPARTS, 1)
    assert state.bm_to_litter.shape == (1, 14, NPARTS, 1)
    assert state.senescence.shape == (1, 14)
    assert state.veget_lastlight.shape == (1, 14)
    assert state.need_adjacent.shape == (1, 14)
    assert state.litterpart.shape == (1, 14, 2)
    assert state.dead_leaves.shape == (1, 14, 2)
    assert state.litter.shape == (1, 2, 14, 2, 1)
    assert state.lignin_struc.shape == (1, 14, 2)
    assert state.carbon.shape == (1, 3, 14)
    assert state.fuel_1hr.shape == (1, 14, 2, 1)
    assert state.fuel_10hr.shape == (1, 14, 2, 1)
    assert state.fuel_100hr.shape == (1, 14, 2, 1)
    assert state.fuel_1000hr.shape == (1, 14, 2, 1)
    assert state.prod10.shape == (1, 11, 2)
    assert state.prod100.shape == (1, 101, 2)
    assert state.flux10.shape == (1, 10, 2)
    assert state.flux100.shape == (1, 100, 2)
    assert state.prod10_total.shape == (1,)
    assert state.prod100_total.shape == (1,)
    assert state.assim_param.shape == (1, 14, 1)
    assert state.altmax.shape == (1, 14)
    assert state.fixed_cryoturbation_depth.shape == (1, 14)
    assert state.fpeat.shape == (1,)
    assert state.deepC_a.shape == (1, 32, 14)
    assert state.deepC_s.shape == (1, 32, 14)
    assert state.deepC_p.shape == (1, 32, 14)
    assert state.soilc_total.shape == (1, 32, 14)
    assert state.interception_storage.shape == (1, 14, 1)
    assert state.thawed_humidity.shape == (1,)
    assert state.depth_organic_soil.shape == (1,)
    assert state.carb_mass_total.shape == (1,)

    raw_biomass = inventory["biomass"].shape
    raw_carbon = inventory["carbon"].shape
    raw_doc = inventory["freedoc"].shape
    assert raw_biomass == (1, 1, NPARTS, 14, 1, 1)
    assert raw_carbon == (1, 14, 3, 1, 1)
    assert raw_doc == (1, 1, 7, 32, 14, 1, 1)
    np.testing.assert_allclose(state.carbon, read_restart_legacy_carbon(files.restart))
    np.testing.assert_allclose(state.soilc_total, read_restart_deep_carbon_total(files.restart))
    np.testing.assert_allclose(state.soilc_total, state.deepC_a + state.deepC_s + state.deepC_p)
    np.testing.assert_allclose(state.interception_storage, read_restart_interception_storage(files.restart))
    np.testing.assert_allclose(
        state.fixed_cryoturbation_depth,
        read_variables(files.restart, ("fixed_cryoturb_depth",))["fixed_cryoturb_depth"][0].transpose(1, 2, 0).reshape(1, 14),
    )
    assert np.asarray(state.pft_present).dtype == np.bool_
    assert np.asarray(state.senescence).dtype == np.bool_
    assert np.asarray(state.need_adjacent).dtype == np.bool_
    assert np.isfinite(state.biomass[:, :, :, ICARBON]).all()
    np.testing.assert_allclose(state.prod10_total, np.sum(state.prod10, axis=(1, 2)))
    np.testing.assert_allclose(state.prod100_total, np.sum(state.prod100, axis=(1, 2)))


def test_read_stomate_daily_accumulator_state_preserves_restart_shapes_without_fills():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_daily_accumulator_state(files.start)

    assert state.humrel_daily.shape == (1, 14)
    assert state.litterhum_daily.shape == (1,)
    assert state.t2m_daily.shape == (1,)
    assert state.t2m_min_daily.shape == (1,)
    assert state.t2m_max_daily.shape == (1,)
    assert state.wspeed_daily.shape == (1,)
    assert state.tsurf_daily.shape == (1,)
    assert state.tsoil_daily.shape == (1, 11)
    assert state.soilhum_daily.shape == (1, 11)
    assert state.precip_daily.shape == (1,)
    assert state.gpp_daily.shape == (1, 14)
    assert state.snowfall_daily.shape == (1,)
    assert state.snowmass_daily.shape == (1,)
    assert state.tmc_topgrass_daily.shape == (1,)
    assert state.fwet_daily.shape == (1,)
    assert state.liqwt_daily.shape == (1,)
    np.testing.assert_allclose(state.gpp_daily[:, :13], 0.0)
    np.testing.assert_allclose(state.snowfall_daily, 0.0)
    np.testing.assert_allclose(state.snowmass_daily, 0.0)
    np.testing.assert_allclose(state.tmc_topgrass_daily, 0.0)
    assert any("stomate_main lines 3187-3240" in item for item in state.provenance)
    assert any("snowfall_daily" in item for item in state.provenance)


def test_stomate_cold_start_readstart_fallbacks_match_fortran_defaults():
    from jax_orchidee.stomate.reference import (
        stomate_cold_start_daily_accumulator_state,
        stomate_cold_start_entry_state,
        stomate_cold_start_season_state,
    )

    t2m = np.asarray([289.0], dtype=np.float64)
    daily = stomate_cold_start_daily_accumulator_state(t2m=t2m, nvm=14, nslm=11)
    season = stomate_cold_start_season_state(t2m=t2m, dt_days=1.0, nvm=14, nslm=11)
    entry = stomate_cold_start_entry_state(t2m=t2m, nvm=14, nslm=11, ndeep=32)

    np.testing.assert_allclose(daily.t2m_daily, 0.0)
    np.testing.assert_allclose(daily.tsurf_daily, t2m)
    np.testing.assert_allclose(daily.t2m_min_daily, 1.0e33)
    np.testing.assert_allclose(daily.t2m_max_daily, -1.0e33)
    np.testing.assert_allclose(daily.wspeed_daily, 1.0e33)
    assert season.tau_longterm == 2.0
    np.testing.assert_allclose(season.t2m_longterm, t2m)
    np.testing.assert_allclose(season.t2m_month, t2m)
    np.testing.assert_allclose(season.tsoil_month, np.repeat(t2m[:, None], 11, axis=1))
    np.testing.assert_allclose(season.gdd_init_date[:, 0], 365.0)
    np.testing.assert_allclose(season.minmoiavail_lastyear, 1.0)
    np.testing.assert_allclose(season.gdd0_lastyear, 150.0)
    np.testing.assert_allclose(season.precip_lastyear, 100.0)
    assert entry.biomass.shape == (1, 14, NPARTS, 1)
    assert entry.litter_above.shape == (1, 2, 14, 1)
    assert entry.litter_below.shape == (1, 2, 14, 32, 1)
    assert entry.litter.shape == (1, 2, 14, 2, 1)
    assert entry.lignin_struc.shape == (1, 14, 2)
    assert entry.carbon.shape == (1, 3, 14)
    assert entry.carbon_32l.shape == (1, 3, 14, 32)
    assert entry.deepC_a.shape == (1, 32, 14)
    assert entry.deepC_s.shape == (1, 32, 14)
    assert entry.deepC_p.shape == (1, 32, 14)
    assert entry.prod10.shape == (1, 11, 2)
    assert entry.prod100.shape == (1, 101, 2)
    assert entry.flux10.shape == (1, 10, 2)
    assert entry.flux100.shape == (1, 100, 2)
    assert entry.DOC.shape == (1, 14, 32, 2, 7, 1)
    assert entry.veget_lastlight.shape == (1, 14)
    assert entry.need_adjacent.shape == (1, 14)
    np.testing.assert_allclose(entry.biomass, 0.0)
    np.testing.assert_allclose(entry.turnover_time, 100.0)
    np.testing.assert_allclose(entry.rip_time, 1.0e33)
    assert np.asarray(entry.need_adjacent).dtype == np.bool_
    np.testing.assert_allclose(entry.thawed_humidity, 1.0e20)
    assert any("stomate_io.f90::readstart lines 501-637" in item for item in daily.provenance)


def test_read_stomate_restart_season_state_preserves_yearly_statistics_without_fills():
    files = find_stomate_reference_files(ROOT)
    season = read_stomate_restart_season_state(files.restart)

    assert season.dt_days_read == 1.0
    assert season.date == 18250
    assert season.tau_longterm == 1095.0
    assert season.moiavail_month.shape == (1, 14)
    assert season.gpp_week.shape == (1, 14)
    assert season.maxmoiavail_lastyear.shape == (1, 14)
    assert season.maxmoiavail_thisyear.shape == (1, 14)
    assert season.minmoiavail_lastyear.shape == (1, 14)
    assert season.minmoiavail_thisyear.shape == (1, 14)
    assert season.maxgppweek_lastyear.shape == (1, 14)
    assert season.maxgppweek_thisyear.shape == (1, 14)
    assert season.gdd0_lastyear.shape == (1,)
    assert season.gdd0_thisyear.shape == (1,)
    assert season.precip_lastyear.shape == (1,)
    assert season.precip_thisyear.shape == (1,)
    assert season.gdd_init_date.shape == (1, 2)
    assert season.time_hum_min.shape == (1, 14)
    assert season.hum_min_dormance.shape == (1, 14)
    assert season.maxfpc_lastyear.shape == (1, 14)
    assert season.maxfpc_thisyear.shape == (1, 14)
    assert season.lm_thisyearmax.shape == (1, 14)
    assert any("stomate_io.f90::readstart lines 719-792" in item for item in season.provenance)
    assert any("stomate_io.f90::readstart lines 516-520" in item for item in season.provenance)


def test_restart_entry_source_closes_only_exact_stomate_main_arguments():
    files = find_stomate_reference_files(ROOT)
    state = read_stomate_restart_entry_state(files.restart)

    source = stomate_restart_entry_source(state)
    assembly = assemble_stomate_main_payload(source)

    assert set(source) == {
        "assim_param",
        "altmax",
        "fpeat",
        "biomass",
        "litter_above",
        "litter_below",
        "carbon_32l",
        "DOC",
        "lignin_struc_above",
        "lignin_struc_below",
        "soilc_total",
        "thawed_humidity",
        "depth_organic_soil",
    }
    for name in source:
        assert name in assembly.covered_inputs

    assert "biomass" not in assembly.missing_by_source.get("stomate_state", ())
    assert "litter_above" not in assembly.missing_by_source.get("stomate_state", ())
    assert "carbon_32l" not in assembly.missing_by_source.get("stomate_state", ())
    assert "DOC" not in assembly.missing_by_source.get("stomate_state", ())
    assert "soilc_total" not in assembly.missing_by_source.get("stomate_state", ())
    assert "thawed_humidity" not in assembly.missing_by_source.get("thermosoil", ())
    assert "depth_organic_soil" not in assembly.missing_by_source.get("driver_static", ())
    assert "fpeat" not in assembly.missing_by_source.get("slowproc_peat_state", ())

    assert "lai" in assembly.missing_by_source["slowproc_stomate_state"]
    assert "height" in assembly.missing_by_source["slowproc_stomate_state"]
    assert "deadleaf_cover" in assembly.missing_by_source["slowproc_stomate_state"]
    assert "sat_duration" in assembly.missing_by_source["slowproc_peat_state"]


def test_slowproc_restart_and_no_lcc_sources_reduce_stomate_entry_gaps():
    files = find_stomate_reference_files(ROOT)
    sechiba = read_slowproc_restart_entry_state(files.run_dir / "sechiba_start.nc")
    derivvar = slowproc_derivvar_explicit(
        veget=sechiba.veget,
        lai=sechiba.lai,
        vcmax_fix=np.zeros((sechiba.lai.shape[1],), dtype=np.float64),
        height_presc=sechiba.height[0],
        qsintcst=0.1,
    )
    no_lcc = slowproc_no_lcc_entry_state(
        veget_max=sechiba.veget_max,
        use_age_class=False,
        veget_update="0Y",
    )

    assembly = assemble_stomate_main_payload(
        stomate_slowproc_restart_entry_source(sechiba),
        stomate_slowproc_derivvar_entry_source(derivvar),
        stomate_slowproc_no_lcc_entry_source(no_lcc),
    )

    for name in (
        "lai",
        "frac_age",
        "height",
        "veget",
        "veget_max",
        "totfrac_nobio",
        "deadleaf_cover",
        "veget_max_new",
        "vegetnew_firstday",
        "totfrac_nobio_new",
        "glccNetLCC",
        "glccSecondShift",
        "glccPrimaryShift",
        "harvest_matrix",
        "bound_spa",
    ):
        assert name in assembly.covered_inputs

    assert "lai" not in assembly.missing_by_source.get("slowproc_stomate_state", ())
    assert "height" not in assembly.missing_by_source.get("slowproc_stomate_state", ())
    assert "veget_max_new" not in assembly.missing_by_source.get("slowproc_land_cover", ())
    assert "sat_duration" in assembly.missing_by_source["slowproc_peat_state"]


def test_fire_and_dyn_peat_disabled_sources_reduce_paper_case_entry_gaps():
    fire = slowproc_fire_disabled_entry_state(kjpindex=1, fire_disable=True)
    peat = slowproc_dyn_peat_disabled_entry_state(kjpindex=1, dyn_peat=False)

    assembly = assemble_stomate_main_payload(
        stomate_slowproc_fire_disabled_entry_source(fire),
        stomate_slowproc_dyn_peat_disabled_entry_source(peat),
    )

    for name in (
        "lightn",
        "popd",
        "read_observed_ba",
        "observed_ba",
        "humign",
        "read_cf_fine",
        "cf_fine",
        "read_cf_coarse",
        "cf_coarse",
        "read_ratio_flag",
        "ratio_flag",
        "read_ratio",
        "ratio",
        "sat_duration",
    ):
        assert name in assembly.covered_inputs

    assert "lightn" not in assembly.missing_by_source.get("slowproc_read_data", ())
    assert "popd" not in assembly.missing_by_source.get("slowproc_read_annual", ())
    assert "sat_duration" not in assembly.missing_by_source.get("slowproc_peat_state", ())
    np.testing.assert_allclose(assembly.payload["lightn"], [0.0])
    assert assembly.payload["read_observed_ba"] is False
    assert "branch-inactive dummy" in peat.notes[0]


def test_thermosoil_static_and_erosion_entry_sources_reduce_only_source_backed_gaps():
    thermosoil = slowproc_thermosoil_entry_init_state(
        kjpindex=1,
        nvm=14,
        znt=np.linspace(0.001, 38.0, 32, dtype=np.float64),
        zlt=np.linspace(0.002, 38.0, 32, dtype=np.float64),
    )
    static = slowproc_static_entry_state(
        njsc=np.asarray([6], dtype=np.int32),
        soil_classif="usda",
        pft_to_mtc=np.arange(1, 15, dtype=np.int32),
        zmaxh=2.0,
    )
    erosion = slowproc_erosion_daily_zero_entry_state(kjpindex=1, ncarb=3)

    assembly = assemble_stomate_main_payload(
        stomate_slowproc_thermosoil_entry_source(thermosoil),
        stomate_slowproc_static_entry_source(static),
        stomate_slowproc_erosion_daily_zero_entry_source(erosion),
    )

    for name in (
        "tdeep",
        "hsdeep",
        "zz_deep",
        "zz_coef_deep",
        "fc_grazing",
        "humcste_use",
        "sed_deposition_d",
        "poc_deposition_d",
    ):
        assert name in assembly.covered_inputs

    for name in ("thawed_humidity",):
        assert name in assembly.missing_by_source["thermosoil"]
    for name in ("heat_Zimov", "sfluxCH4_deep", "sfluxCO2_deep"):
        assert name in assembly.output_arguments
    assert "depth_organic_soil" in assembly.missing_by_source["driver_static"]
    assert "soilc_total" in assembly.missing_by_source["stomate_state"]
    assert "erodepth" in assembly.missing_by_source["hydrol"]

    assert "soilc_total" not in stomate_slowproc_thermosoil_entry_source(thermosoil)
    assert "erodepth" not in stomate_slowproc_erosion_daily_zero_entry_source(erosion)

    closed = assemble_stomate_main_payload(
        stomate_slowproc_erosion_daily_zero_entry_source(erosion),
        stomate_erosion_disabled_erodepth_entry_source(
            erosion_module=False,
            kjpindex=1,
            nvm=14,
        ),
    )
    assert "erodepth" in closed.covered_inputs
    assert "erodepth" not in closed.missing_by_source.get("hydrol", ())
