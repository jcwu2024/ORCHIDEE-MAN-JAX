from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.carbon_kernels import ICARBON, ILEAF
from jax_orchidee.stomate.reference import (
    find_stomate_reference_files,
    read_restart_biomass_carbon,
    read_restart_pft_field,
)
from jax_orchidee.trace.fixed_format import read_records
from jax_orchidee.trace.server_1961 import (
    BRIDGE_TRACE_ROOT,
    DIFFUCO_TRACE_ROOT,
    TRACE_SCHEMAS,
    find_server_record,
    missing_traces,
    read_server_records,
    trace_available,
    trace_path,
    trace_schema,
)


def test_fixed_format_reader_streams_tagged_multiline_records_without_csv_header():
    path = trace_path("hydrol_main")

    records = read_records(path, tags="pre", limit=2)

    assert len(records) == 2
    assert records[0].tag == "pre"
    assert records[0].start_line == 1
    assert records[0].line_count == 10
    assert len(records[0].values) == 32
    assert records[0].values[0:7] == (1, 1, 1, 1, 5, False, 0)


def test_server_trace_registry_records_package_and_fortran_provenance():
    schema = trace_schema("stomate_daily")

    assert schema.trace_file == "orchjax_stomate_daily_trace.txt"
    assert any("server_1961_trace_full_20260623/MANIFEST.txt" in item for item in schema.provenance)
    assert any("stomate.f90" in item for item in schema.tags["gpp_after_accu"].provenance)


@pytest.mark.parametrize(
    ("name", "tag", "expected"),
    (
        (
            "driver_forcing",
            "first_forcing",
            {
                "tstep": 1,
                "lon": pytest.approx(109.0),
                "lat": pytest.approx(21.0),
                "tair": pytest.approx(291.323883056641),
                "psurf": pytest.approx(98422.8671875),
                "height_levuv": pytest.approx(10.0),
            },
        ),
        (
            "intersurf_main",
            "main",
            {
                "tstep": 1,
                "temp_air": pytest.approx(289.76685333252),
                "pb": pytest.approx(987.963802083333),
                "co2_ppm": pytest.approx(317.27),
            },
        ),
        (
            "slowproc_read_data",
            "tide",
            {
                "ik": 1,
                "time_index": 1,
                "source_count": 4,
                "fallback_nearest": False,
                "final_value": pytest.approx(-1.39575001597404),
            },
        ),
        (
            "hydrol_main",
            "pre",
                {
                    "kjit": 1,
                    "jst": 1,
                    "jsl": 1,
                    "resolv": False,
                    "mask_soiltile": pytest.approx(0.0),
                    "mc": pytest.approx(0.3),
                    "mcr": pytest.approx(0.034),
                    "rootsink": pytest.approx(0.0),
                    "flux_top": pytest.approx(0.0),
                    "dt_days": pytest.approx(2.083333333333333e-2),
                },
            ),
        (
            "hydrol_post",
            "post_tile",
                {
                    "kjit": 1,
                    "jst": 1,
                    "tmci": pytest.approx(600.0000002766),
                    "rootsink_sum": pytest.approx(0.0),
                    "k_bottom": pytest.approx(3.931147101839984e-05),
                    "free_drain_coef": pytest.approx(1.0),
                    "dt_days": pytest.approx(2.083333333333333e-2),
                },
            ),
        (
            "stomate_daily",
            "gpp_before_accu",
            {
                "itime": 1,
                "pft": 14,
                "do_slow": False,
                "dt_sechiba": pytest.approx(1800.0),
                "dt_stomate": pytest.approx(86400.0),
                "gpp_daily": pytest.approx(0.0),
            },
        ),
    ),
)
def test_registered_server_trace_schemas_parse_first_record_key_values(name, tag, expected):
    rows = read_server_records(name, tags=tag, limit=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["tag"] == tag
    for key, value in expected.items():
        assert row[key] == value


def test_reader_filters_tags_before_limit_for_mixed_hydrol_post_trace():
    rows = read_server_records("hydrol_post", tags="post_layer", limit=2)

    assert [row["tag"] for row in rows] == ["post_layer", "post_layer"]
    assert [row["jsl"] for row in rows] == [1, 2]
    assert rows[0]["mc"] == pytest.approx(0.3)
    assert rows[0]["mcs"] == pytest.approx(37.1059226989746)


@pytest.mark.parametrize(
    ("name", "tag", "expected"),
    (
        (
            "grid",
            "grid_scatter",
            {
                "ik": 1,
                "area": pytest.approx(12345139970.1911),
                "resolution_x": pytest.approx(111106.370838091),
                "neighbour_1": -1,
                "corner_1_lon": pytest.approx(108.465421088962),
            },
        ),
        (
            "intersurf_boundary",
            "initialize",
            {
                "tstep": 0,
                "ik": 1,
                "temp_air": pytest.approx(289.76685333252),
                "co2_ppm": pytest.approx(317.27),
            },
        ),
        (
            "slowproc_soilt",
            "soil_overlap",
            {
                "ik": 1,
                "soil_texture_source": "usda",
                "source_i": 290,
                "source_j": 69,
                "soilclass_1": pytest.approx(0.1),
                "soilclass_3": pytest.approx(0.84),
            },
        ),
        (
            "slowproc_read_annual",
            "salinity",
            {
                "ik": 1,
                "source_count": 4,
                "fallback_nearest": False,
                "final_value": pytest.approx(33.227593421936),
            },
        ),
        (
            "hydrol_update",
            "mc_after_update",
            {
                "kjit": 1,
                "ji": 1,
                "jst": 1,
                "jsl": 1,
                "mc": pytest.approx(0.3),
                "mcs": pytest.approx(37.1059226989746),
            },
        ),
        (
            "hydrol_alt",
            "alt_first_solve",
            {
                "kjit": 1,
                "ji": 1,
                "jst": 1,
                "flux_top": pytest.approx(0.0),
                "mcr": pytest.approx(0.034),
                "min_sechiba": pytest.approx(1.0e-8),
            },
        ),
        (
            "hydrol_alt_residual",
            "alt_residual",
            {
                "kjit": 1,
                "ji": 1,
                "jst": 1,
                "jsl": 1,
                "resolv": False,
                "rhs": pytest.approx(0.034),
            },
        ),
        (
            "stomate_maint",
            "maint_after",
            {
                "itime": 1,
                "ji": 1,
                "jv": 14,
                "part": 1,
                "do_slow": False,
                "t2m": pytest.approx(289.76685333252),
            },
        ),
        (
            "stomate_npp",
            "after_npp",
            {
                "ip": 1,
                "jv": 14,
                "part": 1,
                "dt_days": pytest.approx(1.0),
                "gpp_daily": pytest.approx(0.0),
                "f_alloc": pytest.approx(0.269777947230288),
                "pft_present": True,
                "age": pytest.approx(2.739726027397260e-3),
            },
        ),
        (
            "stomate_lpj",
            "after_alloc",
            {
                "ip": 1,
                "jv": 14,
                "part": 1,
                "dt_days": pytest.approx(1.0),
                "f_alloc": pytest.approx(0.269777947230288),
                "biomass_after_alloc": pytest.approx(50.9244200281159),
                "senescence": False,
                "rprof": pytest.approx(1.25),
            },
        ),
    ),
)
def test_extended_server_trace_schemas_parse_first_record_key_values(name, tag, expected):
    rows = read_server_records(name, tags=tag, limit=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["tag"] == tag
    for key, value in expected.items():
        assert row[key] == value


def test_find_server_record_matches_tag_and_key_indices_without_full_scan():
    row = find_server_record(
        "hydrol_update",
        tag="mc_after_update",
        criteria={"itime": 1, "ji": 1, "jst": 1, "isl": 2},
        scan_limit=3,
    )

    assert row is not None
    assert row["jsl"] == 2
    assert row["mc"] == pytest.approx(0.3)
    assert row["mcs"] == pytest.approx(37.1059226989746)


def test_find_server_record_uses_pft_alias_for_stomate_traces():
    row = find_server_record("stomate_maint", tag="maint_after", criteria={"itime": 1, "pft": 14}, scan_limit=1)

    assert row is not None
    assert row["jv"] == 14
    assert row["part"] == 1


def test_stomate_maint_nonzero_record_preserves_biomass_and_resp_columns():
    row = find_server_record(
        "stomate_maint",
        tag="maint_after",
        criteria={"itime": 49, "pft": 14, "part": 1},
        scan_limit=49 * 14 * 12,
    )

    assert row is not None
    assert row["biomass"] == pytest.approx(50.9226760411286)
    assert row["resp_maint_part_radia"] == pytest.approx(4.040910557551783e-3)
    assert row["resp_maint_part_after_accum"] == pytest.approx(4.040910557551783e-3)
    assert row["resp_maint_radia_after_sum"] == pytest.approx(4.972527902884661e-3)


def test_target_trace_contracts_have_minimal_field_schemas():
    expected = {
        "driver_forcing",
        "grid",
        "intersurf_boundary",
        "intersurf_main",
        "slowproc_read_data",
        "slowproc_read_annual",
        "slowproc_soilt",
        "hydrol_alt",
        "hydrol_alt_residual",
        "hydrol_update",
        "hydrol_main",
        "hydrol_post",
        "stomate_daily",
        "stomate_lpj",
        "stomate_maint",
        "stomate_npp",
        "diffuco_trans_co2",
        "sechiba_bridge_diffuco",
        "sechiba_bridge_enerbil",
        "sechiba_bridge_hydrol",
        "sechiba_bridge_thermosoil",
        "sechiba_bridge_slowproc",
    }

    assert expected.issubset(TRACE_SCHEMAS)
    for name in expected:
        schema = TRACE_SCHEMAS[name]
        assert schema.tags
        for tag_schema in schema.tags.values():
            assert tag_schema.fields
            assert tag_schema.provenance


def test_sechiba_bridge_trace_schemas_are_registered_from_patch_plan_without_reading_missing_files():
    expected = {
        "sechiba_bridge_diffuco": (
            "orchjax_sechiba_bridge_diffuco_trace.txt",
            {
                "after_diffuco_main": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "gpp",
                    "gsmean",
                    "rveget",
                    "rstruct",
                    "cimean",
                    "vbeta",
                    "vbeta_pft",
                    "vbeta1",
                    "vbeta2",
                    "vbeta3",
                    "vbeta3pot",
                    "vbeta4",
                    "vbeta4_pft",
                    "vbeta5",
                    "q_cdrag",
                    "q_cdrag_pft",
                    "humrel",
                    "qsintveg",
                    "qsintmax",
                    "salinity",
                    "tide_height_1",
                    "veget",
                    "veget_max",
                    "lai",
                    "temp_sol",
                    "temp_sol_pft",
                    "qsurf",
                    "evapot",
                    "evapot_corr",
                )
            },
        ),
        "diffuco_trans_co2": (
            "orchjax_diffuco_trans_co2_trace.txt",
            {
                "after_diffuco_trans_co2_pft14": (
                    "kjit",
                    "ji",
                    "jv",
                    "swdown",
                    "pb",
                    "qsurf",
                    "qsatt",
                    "t2m",
                    "temp_growth",
                    "ca",
                    "vcmax",
                    "humrel",
                    "veget",
                    "veget_max",
                    "lai",
                    "qsintveg",
                    "qsintmax",
                    "vbeta23",
                    "q_cdrag",
                    "q_cdrag_pft",
                    "wind",
                    "control_salinity",
                    "control_inudate",
                    "gpp",
                    "gsmean",
                    "rveget",
                    "rstruct",
                    "cimean",
                    "vbeta3",
                    "vbeta3pot",
                    "assimtot",
                    "rdtot",
                    "gstot",
                    "leaf_gs_top",
                    "laisum",
                    "cim",
                    "ilai",
                    "gamma_star",
                    "fvpd",
                    "g0var",
                )
            },
        ),
        "sechiba_bridge_enerbil": (
            "orchjax_sechiba_bridge_enerbil_trace.txt",
            {
                "after_enerbil_main": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "transpir",
                    "transpot",
                    "vevapwet",
                    "vevapnu",
                    "vevapnu_pft",
                    "vevapsno",
                    "vevapflo",
                    "vevapp",
                    "evapot",
                    "evapot_corr",
                    "temp_sol",
                    "temp_sol_pft",
                    "temp_sol_new",
                    "temp_sol_new_pft",
                    "qsurf",
                    "t2mdiag",
                    "fluxsens",
                    "fluxlat",
                    "soilcap",
                    "soilcap_pft",
                    "snowdz_1",
                    "precip_rain",
                )
            },
        ),
        "sechiba_bridge_hydrol": (
            "orchjax_sechiba_bridge_hydrol_trace.txt",
            {
                "after_hydrol_main_pft": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "humrel",
                    "vegstress",
                    "qsintveg",
                    "precip2canopy",
                    "precip2ground",
                    "canopy2ground",
                    "runoff",
                    "drainage",
                    "drysoil_frac",
                    "evap_bare_lim",
                    "k_litt",
                    "litterhumdiag",
                    "snow",
                    "snow_age",
                    "snow_nobio_1",
                    "snow_nobio_age_1",
                    "tot_melt",
                    "floodout",
                    "fwet_out",
                    "wtp",
                    "fwet_new",
                    "mc_peat_above",
                    "liqwt_ratio",
                    "mc_man_above",
                ),
                "after_hydrol_main_layer": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "jsl",
                    "shumdiag",
                    "shumdiag_perma",
                    "shumdiag_peat",
                    "shumdiag_croppeat",
                    "shumdiag_man",
                    "mc_layh",
                    "mcl_layh",
                    "soilmoist",
                    "mc_layh_pft",
                    "mcl_layh_pft",
                    "soilmoist_pft",
                ),
                "after_hydrol_main_tile_layer": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst",
                    "jsl",
                    "soil_mc",
                    "wat_flux",
                    "mc_layh_s",
                    "mcl_layh_s",
                ),
                "after_hydrol_main_tile": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst",
                    "runoff_per_soil",
                    "runoff2peat",
                    "drainage_per_soil",
                    "soiltile",
                    "reinf_slope",
                    "drunoff_tot",
                ),
            },
        ),
        "sechiba_bridge_thermosoil": (
            "orchjax_sechiba_bridge_thermosoil_trace.txt",
            {
                "after_thermosoil_before_slowproc_pft": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "temp_sol",
                    "temp_sol_new",
                    "temp_sol_pft",
                    "temp_sol_new_pft",
                    "t2mdiag",
                    "qsurf",
                    "soilflx",
                    "soilflx_pft",
                    "soilcap",
                    "soilcap_pft",
                    "grndflux",
                    "gtemp",
                ),
                "after_thermosoil_before_slowproc_layer": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "jsl",
                    "stempdiag",
                    "shumdiag",
                    "shumdiag_perma",
                    "mc_layh",
                    "mcl_layh",
                    "soilmoist",
                ),
            },
        ),
        "sechiba_bridge_slowproc": (
            "orchjax_sechiba_bridge_slowproc_trace.txt",
            {
                "before_slowproc_main": (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "t2mdiag",
                    "temp_sol",
                    "gpp",
                    "humrel",
                    "vegstress",
                    "litterhumdiag",
                    "precip_rain",
                    "precip_snow",
                    "swdown",
                    "evapot_corr",
                    "snow",
                    "snowdz_1",
                    "snowrho_1",
                    "tot_bare_soil",
                    "veget",
                    "veget_max",
                    "lai",
                    "frac_age_1",
                    "height",
                    "qsintmax",
                    "wspeed",
                    "wtp",
                    "fwet_new",
                    "fpeat",
                    "mc_peat_above",
                    "liqwt_ratio",
                    "mc_man_above",
                    "flood_frac_stream",
                ),
                "before_stomate_main": (
                    "kjit",
                    "ji",
                    "jv",
                    "do_slow",
                    "end_of_year",
                    "dt_sechiba",
                    "dt_stomate",
                    "dt_days",
                    "t2m",
                    "t2m_min",
                    "temp_sol",
                    "humrel",
                    "litterhumdiag",
                    "precip_rain",
                    "precip_snow",
                    "wspeed",
                    "lightn",
                    "popd",
                    "gpp",
                    "lai",
                    "veget",
                    "veget_max",
                    "veget_max_new",
                    "t2mdiag",
                    "evapot_corr",
                    "tdeep",
                    "hsdeep_long",
                    "snow",
                    "snowdz_1",
                    "snowrho_1",
                    "wtp",
                    "fwet_new",
                    "fpeat",
                    "shumdiag_peat_1",
                    "mc_peat_above",
                    "liqwt_ratio",
                    "shumdiag_croppeat_1",
                    "mc_croppeat_above",
                    "shumdiag_man_1",
                    "mc_man_above",
                    "soil_mc_top_tile",
                    "wat_flux_top_tile",
                    "drainage_per_soil_tile",
                    "runoff_per_soil_tile",
                    "runoff2peat_tile",
                    "flood_frac",
                    "precip2canopy",
                    "precip2ground",
                    "canopy2ground",
                ),
            },
        ),
    }

    for name, (trace_file, tag_fields) in expected.items():
        schema = trace_schema(name)

        assert schema.trace_file == trace_file
        assert set(schema.tags) == set(tag_fields)
        assert any("server_1961_bridge_trace_patch_plan.md" in item for item in schema.provenance)
        assert any("bridge_trace_patch_plan.py" in item for item in schema.provenance)
        for tag, fields in tag_fields.items():
            tag_schema = schema.tags[tag]
            assert tag_schema.fields == fields
            assert len(tag_schema.fields) == len(set(tag_schema.fields))
            assert any(trace_file in item for item in tag_schema.provenance)
            assert tag_schema.notes


def test_sechiba_bridge_trace_availability_reports_copied_bridge_package_files():
    names = (
        "sechiba_bridge_diffuco",
        "sechiba_bridge_enerbil",
        "sechiba_bridge_hydrol",
        "sechiba_bridge_thermosoil",
        "sechiba_bridge_slowproc",
    )

    assert missing_traces(names) == ()
    assert all(trace_available(name) for name in names)


def test_diffuco_trans_co2_trace_defaults_to_enhanced_diffuco_package():
    path = trace_path("diffuco_trans_co2")

    assert path == DIFFUCO_TRACE_ROOT / "orchjax_diffuco_trans_co2_trace.txt"
    assert trace_available("diffuco_trans_co2")


def test_active_enerbil_trace_schema_defaults_to_enhanced_enerbil_package():
    schema = trace_schema("sechiba_bridge_enerbil_active")
    path = trace_path("sechiba_bridge_enerbil_active")

    from jax_orchidee.trace.server_1961 import ENERBIL_TRACE_ROOT

    assert path == ENERBIL_TRACE_ROOT / "orchjax_sechiba_bridge_enerbil_active_trace.txt"
    assert trace_available("sechiba_bridge_enerbil_active")
    assert set(schema.tags) == {
        "before_enerbil_main_active_pft14",
        "after_enerbil_main_active_pft14",
    }
    assert "swnet" in schema.tags["before_enerbil_main_active_pft14"].fields
    assert "soilflx_pft" in schema.tags["before_enerbil_main_active_pft14"].fields
    assert "temp_sol_add" in schema.tags["after_enerbil_main_active_pft14"].fields


def test_active_enerbil_pottemp_trace_schema_defaults_to_enhanced_enerbil_package():
    schema = trace_schema("enerbil_pottemp_active")
    path = trace_path("enerbil_pottemp_active")

    from jax_orchidee.trace.server_1961 import ENERBIL_TRACE_ROOT

    assert path == ENERBIL_TRACE_ROOT / "orchjax_enerbil_pottemp_active_trace.txt"
    assert trace_available("enerbil_pottemp_active")
    assert set(schema.tags) == {
        "before_enerbil_pottemp_active",
        "after_enerbil_pottemp_active",
    }
    assert schema.tags["before_enerbil_pottemp_active"].fields == (
        "kjit",
        "ji",
        "q_sol_pot",
        "temp_sol_pot",
        "qsurf",
        "temp_sol",
    )
    assert schema.tags["after_enerbil_pottemp_active"].fields == (
        "kjit",
        "ji",
        "q_sol_pot",
        "temp_sol_pot",
    )
    rows = read_server_records("enerbil_pottemp_active", tags="after_enerbil_pottemp_active", limit=1)
    assert rows[0]["q_sol_pot"] == pytest.approx(9.809188079088926e-3)
    assert rows[0]["temp_sol_pot"] == pytest.approx(280.0)


def test_server_stomate_trace_is_not_a_whole_driver_sentinel_for_local_reference_restart():
    reference_files = find_stomate_reference_files(ROOT)
    local_biomass = read_restart_biomass_carbon(reference_files.start)
    local_sla = read_restart_pft_field(reference_files.start, "sla_calc")
    server_lpj = read_server_records("stomate_lpj", tags="after_alloc", limit=1)[0]

    local_leaf_biomass = float(local_biomass[0, 13, ILEAF, ICARBON])
    local_lai = float(local_leaf_biomass * local_sla[0, 13])
    server_leaf_after_alloc = float(server_lpj["biomass_after_alloc"])
    server_lai_after_alloc = float(server_leaf_after_alloc * server_lpj["sla_calc"])

    assert reference_files.start.name == "stomate_start.nc"
    assert server_lpj["ip"] == 1
    assert server_lpj["jv"] == 14
    assert server_lpj["part"] == ILEAF + 1
    assert local_leaf_biomass == pytest.approx(236.51477764695636)
    assert server_leaf_after_alloc == pytest.approx(50.9244200281159)
    assert not np.isclose(local_leaf_biomass, server_leaf_after_alloc, rtol=1e-6, atol=1e-12)
    assert not np.isclose(local_lai, server_lai_after_alloc, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "tag", "expected"),
    (
        (
            "sechiba_bridge_diffuco",
            "after_diffuco_main",
            {
                "kjit": 1,
                "ji": 1,
                "jv": 14,
                "jst_pref": 4,
                "gpp": pytest.approx(0.0),
                "salinity": pytest.approx(33.227593421936),
                "tide_height_1": pytest.approx(-1.39575001597404),
                "veget_max": pytest.approx(1.0),
                "evapot": pytest.approx(0.0),
            },
        ),
        (
            "sechiba_bridge_enerbil",
            "after_enerbil_main",
            {
                "kjit": 1,
                "ji": 1,
                "jv": 14,
                "jst_pref": 4,
                "evapot": pytest.approx(5.719239060617407e-2),
                "evapot_corr": pytest.approx(2.580207366855209e-2),
                "temp_sol_new": pytest.approx(295.744633030835),
                "t2mdiag": pytest.approx(289.76685333252),
                "precip_rain": pytest.approx(0.0),
            },
        ),
        (
            "sechiba_bridge_hydrol",
            "after_hydrol_main_pft",
            {
                "kjit": 1,
                "ji": 1,
                "jv": 14,
                "jst_pref": 4,
                "humrel": pytest.approx(1.0),
                "vegstress": pytest.approx(1.0),
                "runoff": pytest.approx(0.0),
                "drainage": pytest.approx(8.189820446528955e-7),
                "k_litt": pytest.approx(2.61836446146161),
                "mc_man_above": pytest.approx(0.0),
            },
        ),
        (
            "sechiba_bridge_thermosoil",
            "after_thermosoil_before_slowproc_pft",
            {
                "kjit": 1,
                "ji": 1,
                "jv": 14,
                "jst_pref": 4,
                "temp_sol": pytest.approx(280.0),
                "temp_sol_new": pytest.approx(295.744633030835),
                "t2mdiag": pytest.approx(289.76685333252),
                "gtemp": pytest.approx(295.905865017108),
            },
        ),
        (
            "sechiba_bridge_slowproc",
            "before_slowproc_main",
            {
                "kjit": 1,
                "ji": 1,
                "jv": 14,
                "jst_pref": 4,
                "t2mdiag": pytest.approx(289.76685333252),
                "humrel": pytest.approx(1.0),
                "veget_max": pytest.approx(1.0),
                "evapot_corr": pytest.approx(2.580207366855209e-2),
                "mc_man_above": pytest.approx(0.0),
            },
        ),
    ),
)
def test_copied_sechiba_bridge_traces_parse_first_record_key_values(name, tag, expected):
    rows = read_server_records(name, tags=tag, limit=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["tag"] == tag
    for key, value in expected.items():
        assert row[key] == value
