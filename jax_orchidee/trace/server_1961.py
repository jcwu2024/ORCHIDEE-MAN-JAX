"""Schema registry for the 1961 server fixed-format trace package."""

from __future__ import annotations

from pathlib import Path

from jax_orchidee.trace.fixed_format import (
    FixedTraceRecord,
    FixedTraceSchema,
    FixedTraceTagSchema,
    first_matching_record,
    parse_record,
    read_records,
    stream_records,
)


PACKAGE_ROOT = Path("outputs/server_1961_trace_full_20260623")
TRACE_ROOT = PACKAGE_ROOT / "traces"
BRIDGE_PACKAGE_ROOT = Path("outputs/server_1961_bridge_trace_20260624")
BRIDGE_TRACE_ROOT = BRIDGE_PACKAGE_ROOT / "traces"
DIFFUCO_PACKAGE_ROOT = Path("outputs/server_1961_diffuco_active_after_main_trace_20260624_2244")
DIFFUCO_TRACE_ROOT = DIFFUCO_PACKAGE_ROOT / "traces"
ENERBIL_PACKAGE_ROOT = Path("outputs/server_1961_enerbil_active_trace_20260625_0054")
ENERBIL_TRACE_ROOT = ENERBIL_PACKAGE_ROOT / "traces"

PACKAGE_PROVENANCE = (
    "outputs/server_1961_trace_full_20260623/MANIFEST.txt",
    "source_root=/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_trace_clean_20260619_012142",
    "run_script=/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_trace_clean_20260619_012142/scripts/run_trace_full_1961.ksh",
    "run_output=/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_trace_clean_20260619_012142/outputs/run_1961_trace_full",
)

BRIDGE_TRACE_PROVENANCE = (
    "Schema provenance: docs/source_audits/server_1961_bridge_trace_patch_plan.md",
    "Machine-readable draft: outputs/server_trace_patch/bridge_trace_patch_plan.py",
    "Trace package: outputs/server_1961_bridge_trace_20260624 copied from the private jcwu server run.",
)

DIFFUCO_TRACE_PROVENANCE = (
    "Trace package: outputs/server_1961_diffuco_active_after_main_trace_20260624_2244 copied from private jcwu server run.",
    "source_root=/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_diffuco_trace_20260624_181143",
    "run_output=/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_diffuco_trace_20260624_181143/outputs/run_1961_diffuco_active_after_main_trace",
)

ENERBIL_TRACE_PROVENANCE = (
    "Trace package: outputs/server_1961_enerbil_active_trace_20260625_0054 copied from private jcwu server run.",
    "source_root=/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_enerbil_trace_20260624_2355",
    "run_output=/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_enerbil_trace_20260624_2355/outputs/run_1961_enerbil_active_trace",
)


def _tag(tag: str, fields: tuple[str, ...], provenance: tuple[str, ...], notes: str = "") -> FixedTraceTagSchema:
    return FixedTraceTagSchema(tag=tag, fields=fields, provenance=provenance, notes=notes)


TRACE_SCHEMAS: dict[str, FixedTraceSchema] = {
    "driver_forcing": FixedTraceSchema(
        name="driver_forcing",
        trace_file="orchjax_driver_forcing_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_driver_forcing_trace.txt, tag first_forcing.",
            "Fortran source truth: dim2_driver.f90 and readdim2.f90 driver forcing path.",
        ),
        tags={
            "first_forcing": _tag(
                "first_forcing",
                (
                    "tstep",
                    "ik",
                    "i_fortran",
                    "j_fortran",
                    "lon",
                    "lat",
                    "contfrac",
                    "resolution_x",
                    "resolution_y",
                    "tair",
                    "psurf",
                    "qair",
                    "wind_e",
                    "wind_n",
                    "rainf",
                    "snowf",
                    "swdown",
                    "lwdown",
                    "height_lev1",
                    "height_levuv",
                ),
                (
                    "orchjax_driver_forcing_trace.txt:first_forcing",
                    "dim2_driver.f90/readdim2.f90 forcing selection; exact patch lines not included in package.",
                ),
            )
        },
    ),
    "grid": FixedTraceSchema(
        name="grid",
        trace_file="orchjax_grid_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_grid_trace.txt, tag grid_scatter.",
            "Fortran source truth: grid.f90 grid_scatter/grid geometry path.",
        ),
        tags={
            "grid_scatter": _tag(
                "grid_scatter",
                (
                    "ik",
                    "area",
                    "unmapped_grid_01",
                    "resolution_x",
                    "resolution_y",
                    "neighbour_1",
                    "neighbour_2",
                    "neighbour_3",
                    "neighbour_4",
                    "neighbour_5",
                    "neighbour_6",
                    "neighbour_7",
                    "neighbour_8",
                    "seglength_1",
                    "seglength_2",
                    "seglength_3",
                    "seglength_4",
                    "corner_1_lon",
                    "corner_1_lat",
                    "corner_2_lon",
                    "corner_2_lat",
                    "corner_3_lon",
                    "corner_3_lat",
                    "corner_4_lon",
                    "corner_4_lat",
                ),
                (
                    "orchjax_grid_trace.txt:grid_scatter",
                    "grid.f90/haversine.f90 geometry spans audited in docs/source_audits/driver_static_trace_instrumentation_plan.md.",
                ),
                notes="unmapped_grid_01 is the 1.0e20 sentinel-like field emitted by the trace patch.",
            )
        },
    ),
    "intersurf_boundary": FixedTraceSchema(
        name="intersurf_boundary",
        trace_file="orchjax_intersurf_boundary_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_intersurf_boundary_trace.txt, tag initialize.",
            "Fortran source truth: intersurf.f90 initialize boundary path.",
        ),
        tags={
            "initialize": _tag(
                "initialize",
                (
                    "tstep",
                    "ik",
                    "kindex",
                    "lon",
                    "lat",
                    "contfrac",
                    "temp_air",
                    "pb",
                    "qair",
                    "u",
                    "v",
                    "precip_rain",
                    "precip_snow",
                    "swdown",
                    "lwdown",
                    "co2_ppm",
                ),
                (
                    "orchjax_intersurf_boundary_trace.txt:initialize",
                    "intersurf.f90 boundary initialization; exact patch lines not included in package.",
                ),
            )
        },
    ),
    "slowproc_soilt": FixedTraceSchema(
        name="slowproc_soilt",
        trace_file="orchjax_slowproc_soilt_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_slowproc_soilt_trace.txt, tag soil_overlap.",
            "Fortran source truth: slowproc.f90 slowproc_soilt and interpol_help.f90 overlap path.",
        ),
        tags={
            "soil_overlap": _tag(
                "soil_overlap",
                (
                    "ik",
                    "overlap_index",
                    "soil_texture_source",
                    "source_i",
                    "source_j",
                    "sub_area",
                    "soiltext",
                    "bulk_dens_source",
                    "soil_ph_source",
                    "unmapped_soilt_01",
                    "unmapped_soilt_02",
                    "unmapped_soilt_03",
                    "unmapped_soilt_04",
                    "soilclass_1",
                    "soilclass_2",
                    "soilclass_3",
                    "bulk_dens",
                    "soil_ph",
                    "poor_soils",
                ),
                (
                    "orchjax_slowproc_soilt_trace.txt:soil_overlap",
                    "slowproc.f90 slowproc_soilt and interpol_help.f90 overlap aggregation; exact patch lines not included in package.",
                ),
                notes="The source/final soil scalar names need patch audit; soilclass_1..3 are the normalized final triplet.",
            )
        },
    ),
    "slowproc_read_annual": FixedTraceSchema(
        name="slowproc_read_annual",
        trace_file="orchjax_slowproc_read_annual_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_slowproc_read_annual_trace.txt, tag salinity.",
            "Fortran source truth: slowproc.f90 slowproc_read_annual salinity path.",
        ),
        tags={
            "salinity": _tag(
                "salinity",
                (
                    "ik",
                    "lon_low",
                    "lon_up",
                    "lat_low",
                    "lat_up",
                    "source_count",
                    "fallback_nearest",
                    "nearest_source_index",
                    "final_value",
                ),
                (
                    "orchjax_slowproc_read_annual_trace.txt:salinity",
                    "slowproc.f90 slowproc_read_annual; exact patch lines not included in package.",
                ),
            )
        },
    ),
    "intersurf_main": FixedTraceSchema(
        name="intersurf_main",
        trace_file="orchjax_intersurf_main_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_intersurf_main_trace.txt, tag main.",
            "Fortran source truth: intersurf.f90 main boundary call path.",
        ),
        tags={
            "main": _tag(
                "main",
                (
                    "tstep",
                    "ik",
                    "kindex",
                    "lon",
                    "lat",
                    "contfrac",
                    "temp_air",
                    "pb",
                    "qair",
                    "u",
                    "v",
                    "precip_rain",
                    "precip_snow",
                    "swdown",
                    "lwdown",
                    "co2_ppm",
                ),
                (
                    "orchjax_intersurf_main_trace.txt:main",
                    "intersurf.f90 main call boundary; exact patch lines not included in package.",
                ),
            )
        },
    ),
    "slowproc_read_data": FixedTraceSchema(
        name="slowproc_read_data",
        trace_file="orchjax_slowproc_read_data_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_slowproc_read_data_trace.txt, tag tide.",
            "Fortran source truth: slowproc.f90 slowproc_read_data tide read path.",
        ),
        tags={
            "tide": _tag(
                "tide",
                (
                    "ik",
                    "time_index",
                    "lon_low",
                    "lon_up",
                    "lat_low",
                    "lat_up",
                    "source_count",
                    "fallback_nearest",
                    "nearest_source_index",
                    "final_value",
                ),
                (
                    "orchjax_slowproc_read_data_trace.txt:tide",
                    "slowproc.f90 slowproc_read_data; exact patch lines not included in package.",
                ),
            )
        },
    ),
    "hydrol_main": FixedTraceSchema(
        name="hydrol_main",
        trace_file="orchjax_hydrol_main_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_hydrol_main_trace.txt, tag pre.",
            "Fortran source truth: hydrol.f90 hydrol_soil pre-solve/RHS path.",
        ),
        tags={
            "pre": _tag(
                "pre",
                (
                    "kjit",
                    "ji",
                    "jst",
                    "jsl",
                    "njsc",
                    "resolv",
                    "mask_soiltile",
                    "mc",
                    "mcl",
                    "mclint",
                    "profil_froz_hydro_ns",
                    "mcr",
                    "mcs",
                    "a",
                    "b",
                    "d",
                    "k",
                    "e",
                    "f",
                    "g1",
                    "ep",
                    "fp",
                    "gp",
                    "rhs",
                    "tmat_e",
                    "tmat_f",
                    "tmat_g1",
                    "rootsink",
                    "tmci",
                    "flux_top",
                    "free_drain_coef",
                    "dt_days",
                ),
                (
                    "orchjax_hydrol_main_trace.txt:pre",
                    "hydrol.f90 hydrol_soil pre-solve spans audited in docs/source_audits/hydrol_full_solve_trace_contract.md.",
                ),
            )
        },
    ),
    "hydrol_post": FixedTraceSchema(
        name="hydrol_post",
        trace_file="orchjax_hydrol_post_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_hydrol_post_trace.txt, tags post_tile and post_layer.",
            "Fortran source truth: hydrol.f90 hydrol_soil post-solve path.",
        ),
        tags={
            "post_tile": _tag(
                "post_tile",
                (
                    "kjit",
                    "ji",
                    "jst",
                    "branch_peat",
                    "tmci",
                    "tmcf",
                    "flux_top",
                    "rootsink_sum",
                    "dr_ns",
                    "dr_corrnum_ns",
                    "check_tr_ns",
                    "k_bottom",
                    "free_drain_coef",
                    "dt_days",
                ),
                (
                    "orchjax_hydrol_post_trace.txt:post_tile",
                    "hydrol.f90 hydrol_soil post-solve/drainage spans audited in docs/source_audits/hydrol_full_solve_trace_contract.md.",
                ),
            ),
            "post_layer": _tag(
                "post_layer",
                (
                    "kjit",
                    "ji",
                    "jst",
                    "jsl",
                    "mcl",
                    "mc",
                    "profil_froz_hydro_ns",
                    "mcr",
                    "mcs",
                ),
                (
                    "orchjax_hydrol_post_trace.txt:post_layer",
                    "hydrol.f90 hydrol_soil post-solve layer update spans audited in docs/source_audits/hydrol_full_solve_trace_contract.md.",
                ),
            ),
        },
    ),
    "hydrol_update": FixedTraceSchema(
        name="hydrol_update",
        trace_file="orchjax_hydrol_update_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_hydrol_update_trace.txt, tag mc_after_update.",
            "Fortran source truth: hydrol.f90 hydrol_soil layer update path.",
        ),
        tags={
            "mc_after_update": _tag(
                "mc_after_update",
                (
                    "kjit",
                    "ji",
                    "jst",
                    "jsl",
                    "mc",
                    "mcl",
                    "profil_froz_hydro_ns",
                    "mcr",
                    "mcs",
                ),
                (
                    "orchjax_hydrol_update_trace.txt:mc_after_update",
                    "hydrol.f90 hydrol_soil post-solve layer update spans audited in docs/source_audits/hydrol_full_solve_trace_contract.md.",
                ),
            )
        },
    ),
    "hydrol_alt": FixedTraceSchema(
        name="hydrol_alt",
        trace_file="orchjax_hydrol_alt_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_hydrol_alt_trace.txt, tag alt_first_solve.",
            "Fortran source truth: hydrol.f90 hydrol_soil alternate first solve path.",
        ),
        tags={
            "alt_first_solve": _tag(
                "alt_first_solve",
                (
                    "kjit",
                    "ji",
                    "jst",
                    "mcl_top_after_first_solve",
                    "flux_top",
                    "mcr",
                    "min_sechiba",
                ),
                (
                    "orchjax_hydrol_alt_trace.txt:alt_first_solve",
                    "hydrol.f90 alternate bare-soil solve spans audited in docs/source_audits/hydrol_full_solve_trace_contract.md.",
                ),
            )
        },
    ),
    "hydrol_alt_residual": FixedTraceSchema(
        name="hydrol_alt_residual",
        trace_file="orchjax_hydrol_alt_residual_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_hydrol_alt_residual_trace.txt, tag alt_residual.",
            "Fortran source truth: hydrol.f90 hydrol_soil alternate residual-boundary path.",
        ),
        tags={
            "alt_residual": _tag(
                "alt_residual",
                (
                    "kjit",
                    "ji",
                    "jst",
                    "jsl",
                    "resolv",
                    "rhs",
                    "tmat_e",
                    "tmat_f",
                    "tmat_g1",
                    "mcl_after_residual_solve",
                ),
                (
                    "orchjax_hydrol_alt_residual_trace.txt:alt_residual",
                    "hydrol.f90 residual-boundary spans audited in docs/source_audits/hydrol_full_solve_trace_contract.md.",
                ),
            )
        },
    ),
    "stomate_daily": FixedTraceSchema(
        name="stomate_daily",
        trace_file="orchjax_stomate_daily_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_stomate_daily_trace.txt, tags gpp_before_accu and gpp_after_accu.",
            "Fortran source truth: stomate.f90 stomate_main daily GPP accumulation path.",
        ),
        tags={
            "gpp_before_accu": _tag(
                "gpp_before_accu",
                (
                    "itime",
                    "ik",
                    "pft",
                    "do_slow",
                    "dt_sechiba",
                    "dt_stomate",
                    "dt_days",
                    "gpp_d",
                    "gpp_daily",
                ),
                (
                    "orchjax_stomate_daily_trace.txt:gpp_before_accu",
                    "stomate.f90 stomate_main lines 3198-3208; stomate_accu_r2d lines 9365-9387.",
                ),
            ),
            "gpp_after_accu": _tag(
                "gpp_after_accu",
                (
                    "itime",
                    "ik",
                    "pft",
                    "do_slow",
                    "dt_sechiba",
                    "dt_stomate",
                    "dt_days",
                    "gpp_d",
                    "gpp_daily",
                ),
                (
                    "orchjax_stomate_daily_trace.txt:gpp_after_accu",
                    "stomate.f90 stomate_main lines 3198-3208; stomate_accu_r2d lines 9365-9387.",
                ),
            ),
        },
    ),
    "stomate_maint": FixedTraceSchema(
        name="stomate_maint",
        trace_file="orchjax_stomate_maint_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_stomate_maint_trace.txt, tag maint_after.",
            "Fortran source truth: stomate.f90 maint_respiration call and stomate_resp.f90 maintenance path.",
        ),
        tags={
            "maint_after": _tag(
                "maint_after",
                (
                    "itime",
                    "ji",
                    "jv",
                    "part",
                    "do_slow",
                    "dt_sechiba",
                    "dt_stomate",
                    "dt_days",
                    "lai",
                    "t2m",
                    "t2m_longterm",
                    "rprof",
                    "sla_calc",
                    "biomass",
                    "resp_maint_part_radia",
                    "resp_maint_part_after_accum",
                    "resp_maint_radia_after_sum",
                ),
                (
                    "orchjax_stomate_maint_trace.txt:maint_after",
                    "stomate.f90 stomate_main lines 3244-3267; stomate_resp.f90 maint_respiration audited in docs/source_audits/stomate_daily_trace_contract.md.",
                ),
            )
        },
    ),
    "stomate_npp": FixedTraceSchema(
        name="stomate_npp",
        trace_file="orchjax_stomate_npp_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_stomate_npp_trace.txt, tag after_npp.",
            "Fortran source truth: stomate_lpj.f90 npp_calc call and stomate_npp.f90.",
        ),
        tags={
            "after_npp": _tag(
                "after_npp",
                (
                    "ip",
                    "jv",
                    "part",
                    "dt_days",
                    "gpp_daily",
                    "f_alloc",
                    "bm_alloc",
                    "biomass_after_npp",
                    "resp_maint_part",
                    "resp_maint",
                    "resp_growth",
                    "npp_daily",
                    "pft_present",
                    "age",
                ),
                (
                    "orchjax_stomate_npp_trace.txt:after_npp",
                    "stomate_lpj.f90 StomateLpj lines 1115-1131; stomate_npp.f90 npp_calc audited in docs/source_audits/stomate_daily_trace_contract.md.",
                ),
                notes="Trace patch emits ip, PFT14, part, dt_days, then after-NPP diagnostics for the first three grid points.",
            )
        },
    ),
    "stomate_lpj": FixedTraceSchema(
        name="stomate_lpj",
        trace_file="orchjax_stomate_lpj_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + (
            "Trace provenance: orchjax_stomate_lpj_trace.txt, tag after_alloc.",
            "Fortran source truth: stomate_lpj.f90 allocation call path.",
        ),
        tags={
            "after_alloc": _tag(
                "after_alloc",
                (
                    "ip",
                    "jv",
                    "part",
                    "dt_days",
                    "f_alloc",
                    "biomass_after_alloc",
                    "lai",
                    "slai",
                    "senescence",
                    "rprof",
                    "sla_calc",
                ),
                (
                    "orchjax_stomate_lpj_trace.txt:after_alloc",
                    "stomate_lpj.f90 StomateLpj lines 1093-1102; stomate_alloc.f90 alloc audited in docs/source_audits/stomate_daily_trace_contract.md.",
                ),
                notes="Trace patch emits ip, PFT14, part, dt_days, f_alloc, biomass after allocation, LAI diagnostics, senescence, rprof, and sla_calc.",
            )
        },
    ),
    "sechiba_bridge_diffuco": FixedTraceSchema(
        name="sechiba_bridge_diffuco",
        trace_file="orchjax_sechiba_bridge_diffuco_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + BRIDGE_TRACE_PROVENANCE
        + (
            "Trace schema provenance: orchjax_sechiba_bridge_diffuco_trace.txt, tag after_diffuco_main.",
            "Fortran source truth: sechiba.f90 sechiba_main lines 997-1005 calls diffuco_main; diffuco.f90 diffuco_main process boundary.",
        ),
        tags={
            "after_diffuco_main": _tag(
                "after_diffuco_main",
                (
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
                ),
                (
                    "orchjax_sechiba_bridge_diffuco_trace.txt:after_diffuco_main",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_diffuco_main",
                    "sechiba.f90:997-1005; diffuco.f90::diffuco_main process boundary.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            )
        },
    ),
    "sechiba_bridge_diffuco_active": FixedTraceSchema(
        name="sechiba_bridge_diffuco_active",
        trace_file="orchjax_sechiba_bridge_diffuco_active_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + BRIDGE_TRACE_PROVENANCE
        + DIFFUCO_TRACE_PROVENANCE
        + (
            "Trace schema provenance: outputs/server_trace_patch/bridge_trace_patch_plan.py:after_diffuco_main_active_pft14.",
            "Fortran source truth: sechiba.f90 sechiba_main lines 997-1005 calls diffuco_main before enerbil_main.",
            "Fortran source truth: diffuco.f90 diffuco_main lines 665-710 runs trans_co2, bare, and comb before returning.",
        ),
        tags={
            "after_diffuco_main_active_pft14": _tag(
                "after_diffuco_main_active_pft14",
                (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "qair",
                    "temp_air",
                    "pb",
                    "qsurf",
                    "evap_bare_lim",
                    "tot_bare_soil",
                    "gpp",
                    "gsmean",
                    "rveget",
                    "rstruct",
                    "cimean",
                    "valpha",
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
                    "evapot",
                    "evapot_corr",
                ),
                (
                    "orchjax_sechiba_bridge_diffuco_active_trace.txt:after_diffuco_main_active_pft14",
                    "outputs/server_trace_patch/bridge_trace_patch_plan.py:after_diffuco_main_active_pft14",
                    "sechiba.f90:997-1005; diffuco.f90::diffuco_main lines 665-710.",
                ),
                notes="Planned active DIFFUCO module-closure boundary; not present in older copied bridge packages.",
            )
        },
    ),
    "diffuco_trans_co2": FixedTraceSchema(
        name="diffuco_trans_co2",
        trace_file="orchjax_diffuco_trans_co2_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + BRIDGE_TRACE_PROVENANCE
        + DIFFUCO_TRACE_PROVENANCE
        + (
            "Trace schema: orchjax_diffuco_trans_co2_trace.txt, tag after_diffuco_trans_co2_pft14.",
            "Fortran source truth: diffuco.f90 diffuco_trans_co2 C3/output path lines 2338-2969.",
        ),
        tags={
            "after_diffuco_trans_co2_pft14": _tag(
                "after_diffuco_trans_co2_pft14",
                (
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
                ),
                (
                    "orchjax_diffuco_trans_co2_trace.txt:after_diffuco_trans_co2_pft14",
                    "outputs/server_trace_patch/apply_bridge_trace_patch.py planned PFT14 diffuco_trans_co2 trace.",
                    "diffuco.f90::diffuco_trans_co2 lines 2338-2969.",
                ),
                notes="Planned enhanced DIFFUCO parity trace; older copied bridge packages may not include this file.",
            )
        },
    ),
    "sechiba_bridge_enerbil": FixedTraceSchema(
        name="sechiba_bridge_enerbil",
        trace_file="orchjax_sechiba_bridge_enerbil_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + BRIDGE_TRACE_PROVENANCE
        + (
            "Trace schema provenance: orchjax_sechiba_bridge_enerbil_trace.txt, tag after_enerbil_main.",
            "Fortran source truth: sechiba.f90 sechiba_main lines 1013-1019 calls enerbil_main; enerbil.f90 enerbil_main process boundary.",
        ),
        tags={
            "after_enerbil_main": _tag(
                "after_enerbil_main",
                (
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
                ),
                (
                    "orchjax_sechiba_bridge_enerbil_trace.txt:after_enerbil_main",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_enerbil_main",
                    "sechiba.f90:1013-1019; enerbil.f90::enerbil_main process boundary.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            )
        },
    ),
    "sechiba_bridge_enerbil_active": FixedTraceSchema(
        name="sechiba_bridge_enerbil_active",
        trace_file="orchjax_sechiba_bridge_enerbil_active_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + BRIDGE_TRACE_PROVENANCE
        + DIFFUCO_TRACE_PROVENANCE
        + ENERBIL_TRACE_PROVENANCE
        + (
            "Trace schema provenance: outputs/server_trace_patch/apply_enerbil_active_trace_patch.py.",
            "Fortran source truth: sechiba.f90 sechiba_main lines 997-1019 calls diffuco_main before enerbil_main.",
            "Fortran source truth: enerbil.f90 enerbil_main lines 390-598 runs begin, surftemp, flux, t2mdiag, and evapveg.",
        ),
        tags={
            "before_enerbil_main_active_pft14": _tag(
                "before_enerbil_main_active_pft14",
                (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "lai",
                    "gpp",
                    "veget_max",
                    "zlev",
                    "lwdown",
                    "swdown",
                    "swnet",
                    "epot_air",
                    "temp_air",
                    "u",
                    "v",
                    "petAcoef",
                    "petBcoef",
                    "qair",
                    "peqAcoef",
                    "peqBcoef",
                    "pb",
                    "rau",
                    "valpha",
                    "vbeta",
                    "vbeta_pft",
                    "vbeta1",
                    "vbeta2",
                    "vbeta3",
                    "vbeta3pot",
                    "vbeta4",
                    "vbeta4_pft",
                    "vbeta5",
                    "emis",
                    "soilflx",
                    "soilflx_pft",
                    "soilcap",
                    "soilcap_pft",
                    "q_cdrag",
                    "q_cdrag_pft",
                    "humrel",
                    "precip_rain",
                    "snowdz_1",
                    "pgflux",
                    "temp_sol",
                    "temp_sol_pft",
                    "qsurf",
                    "evapot",
                    "evapot_corr",
                ),
                (
                    "orchjax_sechiba_bridge_enerbil_active_trace.txt:before_enerbil_main_active_pft14",
                    "outputs/server_trace_patch/apply_enerbil_active_trace_patch.py",
                    "sechiba.f90:1013-1019; enerbil.f90::enerbil_main input boundary lines 410-472.",
                ),
                notes="Planned active ENERBIL pre-call trace for source-equivalent PFT14 module closure.",
            ),
            "after_enerbil_main_active_pft14": _tag(
                "after_enerbil_main_active_pft14",
                (
                    "kjit",
                    "ji",
                    "jv",
                    "jst_pref",
                    "lai",
                    "gpp",
                    "veget_max",
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
                    "pgflux",
                    "temp_sol_add",
                    "soilcap",
                    "soilcap_pft",
                    "soilflx",
                    "soilflx_pft",
                    "snowdz_1",
                    "precip_rain",
                ),
                (
                    "orchjax_sechiba_bridge_enerbil_active_trace.txt:after_enerbil_main_active_pft14",
                    "outputs/server_trace_patch/apply_enerbil_active_trace_patch.py",
                    "sechiba.f90:1013-1019; enerbil.f90::enerbil_main output boundary lines 449-472.",
                ),
                notes="Planned active ENERBIL after-call trace including HYDROL explicit-snow boundary fields.",
            ),
        },
    ),
    "enerbil_pottemp_active": FixedTraceSchema(
        name="enerbil_pottemp_active",
        trace_file="orchjax_enerbil_pottemp_active_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + ENERBIL_TRACE_PROVENANCE
        + (
            "Trace schema provenance: outputs/server_trace_patch/apply_enerbil_pottemp_active_trace_patch.py.",
            "Fortran source truth: enerbil.f90 enerbil_main lines 523-526 calls enerbil_pottemp.",
            "Fortran source truth: enerbil.f90 enerbil_pottemp lines 1263-1317 updates q_sol_pot and temp_sol_pot.",
        ),
        tags={
            "before_enerbil_pottemp_active": _tag(
                "before_enerbil_pottemp_active",
                (
                    "kjit",
                    "ji",
                    "q_sol_pot",
                    "temp_sol_pot",
                    "qsurf",
                    "temp_sol",
                ),
                (
                    "orchjax_enerbil_pottemp_active_trace.txt:before_enerbil_pottemp_active",
                    "outputs/server_trace_patch/apply_enerbil_pottemp_active_trace_patch.py",
                    "enerbil.f90:523-526; pre-call q_sol_pot/temp_sol_pot module state.",
                ),
                notes="Planned active ENERBIL potential-surface trace before enerbil_pottemp.",
            ),
            "after_enerbil_pottemp_active": _tag(
                "after_enerbil_pottemp_active",
                (
                    "kjit",
                    "ji",
                    "q_sol_pot",
                    "temp_sol_pot",
                ),
                (
                    "orchjax_enerbil_pottemp_active_trace.txt:after_enerbil_pottemp_active",
                    "outputs/server_trace_patch/apply_enerbil_pottemp_active_trace_patch.py",
                    "enerbil.f90:523-526,1263-1317; post-call q_sol_pot/temp_sol_pot module state.",
                ),
                notes="Planned active ENERBIL potential-surface trace after enerbil_pottemp.",
            ),
        },
    ),
    "sechiba_bridge_hydrol": FixedTraceSchema(
        name="sechiba_bridge_hydrol",
        trace_file="orchjax_sechiba_bridge_hydrol_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + BRIDGE_TRACE_PROVENANCE
        + (
            "Trace schema provenance: orchjax_sechiba_bridge_hydrol_trace.txt, tags after_hydrol_main_*.",
            "Fortran source truth: sechiba.f90 sechiba_main lines 1049-1072 calls hydrol_main and lines 1093-1102 map PFT layer moisture.",
        ),
        tags={
            "after_hydrol_main_pft": _tag(
                "after_hydrol_main_pft",
                (
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
                (
                    "orchjax_sechiba_bridge_hydrol_trace.txt:after_hydrol_main_pft",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_hydrol_main_pft",
                    "sechiba.f90:1049-1072; hydrol.f90::hydrol_main exported boundary.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            ),
            "after_hydrol_main_layer": _tag(
                "after_hydrol_main_layer",
                (
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
                (
                    "orchjax_sechiba_bridge_hydrol_trace.txt:after_hydrol_main_layer",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_hydrol_main_layer",
                    "sechiba.f90:1093-1102 maps hydrol tile moisture to PFT arrays.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            ),
            "after_hydrol_main_tile_layer": _tag(
                "after_hydrol_main_tile_layer",
                (
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
                (
                    "orchjax_sechiba_bridge_hydrol_trace.txt:after_hydrol_main_tile_layer",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_hydrol_main_tile_layer",
                    "sechiba.f90:1049-1072; hydrol.f90::hydrol_main tile-layer exports.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            ),
            "after_hydrol_main_tile": _tag(
                "after_hydrol_main_tile",
                (
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
                (
                    "orchjax_sechiba_bridge_hydrol_trace.txt:after_hydrol_main_tile",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_hydrol_main_tile",
                    "sechiba.f90:1049-1072; hydrol.f90::hydrol_main tile runoff/drainage exports.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            ),
        },
    ),
    "sechiba_bridge_thermosoil": FixedTraceSchema(
        name="sechiba_bridge_thermosoil",
        trace_file="orchjax_sechiba_bridge_thermosoil_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + BRIDGE_TRACE_PROVENANCE
        + (
            "Trace schema provenance: orchjax_sechiba_bridge_thermosoil_trace.txt, tags after_thermosoil_before_slowproc_*.",
            "Fortran source truth: sechiba.f90 sechiba_main lines 1109-1118 calls thermosoil_main before slowproc.",
        ),
        tags={
            "after_thermosoil_before_slowproc_pft": _tag(
                "after_thermosoil_before_slowproc_pft",
                (
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
                (
                    "orchjax_sechiba_bridge_thermosoil_trace.txt:after_thermosoil_before_slowproc_pft",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_thermosoil_before_slowproc_pft",
                    "sechiba.f90:1109-1118; thermosoil.f90::thermosoil_main process boundary.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            ),
            "after_thermosoil_before_slowproc_layer": _tag(
                "after_thermosoil_before_slowproc_layer",
                (
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
                (
                    "orchjax_sechiba_bridge_thermosoil_trace.txt:after_thermosoil_before_slowproc_layer",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#after_thermosoil_before_slowproc_layer",
                    "sechiba.f90:1109-1118; thermosoil.f90::thermosoil_main stempdiag boundary.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            ),
        },
    ),
    "sechiba_bridge_slowproc": FixedTraceSchema(
        name="sechiba_bridge_slowproc",
        trace_file="orchjax_sechiba_bridge_slowproc_trace.txt",
        provenance=PACKAGE_PROVENANCE
        + BRIDGE_TRACE_PROVENANCE
        + (
            "Trace schema provenance: orchjax_sechiba_bridge_slowproc_trace.txt, tags before_slowproc_main and before_stomate_main.",
            "Fortran source truth: sechiba.f90 sechiba_main lines 1184-1216 calls slowproc_main; slowproc.f90 slowproc_main lines 973-985 begins stomate_main call.",
        ),
        tags={
            "before_slowproc_main": _tag(
                "before_slowproc_main",
                (
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
                (
                    "orchjax_sechiba_bridge_slowproc_trace.txt:before_slowproc_main",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#before_slowproc_main",
                    "sechiba.f90:1184-1216 passes final SECHIBA boundary into slowproc_main.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            ),
            "before_stomate_main": _tag(
                "before_stomate_main",
                (
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
                (
                    "orchjax_sechiba_bridge_slowproc_trace.txt:before_stomate_main",
                    "docs/source_audits/server_1961_bridge_trace_patch_plan.md#before_stomate_main",
                    "slowproc.f90:973-985 passes final slowproc boundary into stomate_main.",
                ),
                notes="Trace-backed schema from the copied server bridge package.",
            ),
        },
    ),
}


def _uses_bridge_trace_root(name: str) -> bool:
    """Return whether a schema belongs to the supplemental bridge package."""

    return any(item in TRACE_SCHEMAS[name].provenance for item in BRIDGE_TRACE_PROVENANCE)


def _uses_diffuco_trace_root(name: str) -> bool:
    """Return whether a schema belongs to the enhanced DIFFUCO trace package."""

    return any(item in TRACE_SCHEMAS[name].provenance for item in DIFFUCO_TRACE_PROVENANCE)


def _uses_enerbil_trace_root(name: str) -> bool:
    """Return whether a schema belongs to the enhanced ENERBIL trace package."""

    return any(item in TRACE_SCHEMAS[name].provenance for item in ENERBIL_TRACE_PROVENANCE)


def trace_path(name: str, *, root: str | Path = TRACE_ROOT) -> Path:
    """Return the local path for a registered server trace."""

    trace_root = Path(root)
    if trace_root == TRACE_ROOT and _uses_enerbil_trace_root(name):
        return ENERBIL_TRACE_ROOT / TRACE_SCHEMAS[name].trace_file
    if trace_root == TRACE_ROOT and _uses_diffuco_trace_root(name):
        return DIFFUCO_TRACE_ROOT / TRACE_SCHEMAS[name].trace_file
    if trace_root == TRACE_ROOT and _uses_bridge_trace_root(name):
        return BRIDGE_TRACE_ROOT / TRACE_SCHEMAS[name].trace_file
    return trace_root / TRACE_SCHEMAS[name].trace_file


def trace_available(name: str, *, root: str | Path = TRACE_ROOT) -> bool:
    """Return whether a registered trace file is present in the local package."""

    return trace_path(name, root=root).is_file()


def missing_traces(names: tuple[str, ...] | None = None, *, root: str | Path = TRACE_ROOT) -> tuple[str, ...]:
    """Return registered trace names whose files are not present locally.

    This is intentionally a filesystem availability check, not a schema
    validator. It lets bridge trace workers distinguish "schema prepared" from
    "supplemental server trace copied back".
    """

    selected = names if names is not None else tuple(TRACE_SCHEMAS)
    return tuple(name for name in selected if not trace_available(name, root=root))


def trace_schema(name: str) -> FixedTraceSchema:
    """Return one server trace schema by name."""

    return TRACE_SCHEMAS[name]


def parse_server_record(name: str, record: FixedTraceRecord) -> dict[str, object]:
    """Parse one server trace record using its registered tag schema."""

    schema = trace_schema(name)
    try:
        tag_schema = schema.tags[record.tag]
    except KeyError as exc:
        raise KeyError(f"trace {name!r} has no schema for tag {record.tag!r}") from exc
    parsed = parse_record(record, tag_schema)
    parsed["tag"] = record.tag
    parsed["start_line"] = record.start_line
    return parsed


def read_server_records(
    name: str,
    *,
    tags: str | tuple[str, ...] | None = None,
    limit: int | None = None,
    root: str | Path = TRACE_ROOT,
) -> tuple[dict[str, object], ...]:
    """Read a bounded set of parsed records from a registered trace."""

    records = read_records(trace_path(name, root=root), tags=tags, limit=limit)
    return tuple(parse_server_record(name, record) for record in records)


def _criteria_matches(row: dict[str, object], criteria: dict[str, object]) -> bool:
    aliases = {
        "jv": ("jv", "pft"),
        "pft": ("pft", "jv"),
        "isl": ("isl", "jsl"),
        "jsl": ("jsl", "isl"),
        "itime": ("itime", "tstep", "kjit"),
        "tstep": ("tstep", "itime", "kjit"),
    }
    for key, expected in criteria.items():
        keys = aliases.get(key, (key,))
        if not any(row.get(candidate) == expected for candidate in keys):
            return False
    return True


def find_server_record(
    name: str,
    *,
    tag: str | tuple[str, ...] | None = None,
    criteria: dict[str, object] | None = None,
    scan_limit: int | None = None,
    root: str | Path = TRACE_ROOT,
) -> dict[str, object] | None:
    """Find the first parsed record matching tag and key-index criteria.

    Common criteria include `itime`, `ji`, `jv`, and `isl`/`jsl`. Matching is
    streaming and bounded by `scan_limit` after tag filtering, so HYDROL and
    STOMATE workers can probe large trace files without loading them.
    """

    if not criteria:
        return first_matching_record(
            trace_path(name, root=root),
            tags=tag,
            criteria={},
            scan_limit=scan_limit,
            parser=lambda record: parse_server_record(name, record),
        )
    return _find_server_record_with_aliases(name, tag=tag, criteria=criteria, scan_limit=scan_limit, root=root)


def _find_server_record_with_aliases(
    name: str,
    *,
    tag: str | tuple[str, ...] | None,
    criteria: dict[str, object],
    scan_limit: int | None,
    root: str | Path,
) -> dict[str, object] | None:
    for row in stream_server_records(name, tags=tag, limit=scan_limit, root=root):
        if _criteria_matches(row, criteria):
            return row
    return None


def stream_server_records(
    name: str,
    *,
    tags: str | tuple[str, ...] | None = None,
    limit: int | None = None,
    root: str | Path = TRACE_ROOT,
):
    """Stream parsed records from a registered trace."""

    for record in stream_records(trace_path(name, root=root), tags=tags, limit=limit):
        yield parse_server_record(name, record)
