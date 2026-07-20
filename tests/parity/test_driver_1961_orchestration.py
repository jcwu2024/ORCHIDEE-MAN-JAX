from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    DriverFirstDayStomateStatePacket,
    DriverPreviousStepStatePacket,
    DriverRuntimeHalfHourInput,
    LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES,
    STOMATE_DAY_SEASON_ANNUAL_FIELDS,
    STOMATE_DAY_SEASON_BIOMETEOROLOGY_FIELDS,
    STOMATE_DAY_SEASON_MEMORY_FIELDS,
    STOMATE_DAY_SEASON_PERSISTED_FIELDS,
    _paper_end_of_year,
    _compiled_half_hour_forcing,
    _paper_1961_step_bundle_from_context,
    _paper_compiled_forcing_day,
    _paper_day_end_state_packet,
    _paper_day_final_lai_from_daily_carbon,
    _paper_day_post_npp_owned_daily_fields,
    _paper_day_stomate_daily_carbon_from_bundles,
    _paper_first_step_do_slow,
    _paper_first_step_stomate_restart_input_bundles,
    _paper_later_day_stomate_entry_state_from_previous,
    driver_previous_step_state_from_next_step_hydrol_scaffold,
    driver_previous_step_state_from_timestep_scaffold,
    driver_year_handoff_state_gaps,
    paper_1961_next_step_diffuco_precall_scaffold,
    paper_1961_next_step_enerbil_precall_scaffold,
    paper_1961_next_step_hydrol_precall_scaffold,
    paper_1961_driver_cold_start_first_step_coverage,
    paper_1961_driver_cold_start_day_scaffold,
    paper_1961_driver_day_scaffold,
    paper_1961_driver_later_day_scaffold,
    paper_1961_driver_multiday_modelout_lite_run,
    paper_1961_driver_multiday_modelout_run,
    paper_1961_driver_multiday_scaffold,
    paper_1961_driver_restart_year_multiday_modelout_lite_run,
    paper_1961_driver_restart_year_start_day_result,
    paper_1961_driver_year_scaffold,
    paper_1961_driver_timestep_scaffold,
    prepare_paper_1961_driver_context,
    read_driver_runtime_scalars,
    rebase_driver_state_for_year_start,
)
from jax_orchidee.driver.bundle import load_paper_1961_step_bundle  # noqa: E402
from jax_orchidee.driver.init import parse_run_def_float, parse_run_def_indexed_vector  # noqa: E402
from jax_orchidee.driver.fast_state import (  # noqa: E402
    fast_state_from_previous_packet,
    previous_packet_from_fast_state,
)
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state  # noqa: E402
from jax_orchidee.driver.restart_bundle import paper_restart_bundle_state_from_day_end_packet  # noqa: E402
from jax_orchidee.driver.stomate_boundary import paper_1961_first_step_stomate_boundary  # noqa: E402
from jax_orchidee.driver.sechiba_boundary import build_intersurf_first_step_payload  # noqa: E402
from jax_orchidee.driver.trace import read_static_fields_from_fixed_format_trace_dir  # noqa: E402
from jax_orchidee.sechiba.slowproc import slowproc_surface_update_explicit  # noqa: E402
from jax_orchidee.stomate.season import season_update_gdd_init_date  # noqa: E402
from jax_orchidee.stomate.reference import (  # noqa: E402
    StomateRestartEntryState,
    StomateRestartSeasonState,
)
from jax_orchidee.stomate.restart_io import STOMATE_FIXED_PATH_CARRY_FIELDS  # noqa: E402
from jax_orchidee.stomate.integration import (  # noqa: E402
    stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_static_jit,
)
from jax_orchidee.trace.server_1961 import find_server_record  # noqa: E402


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
TRACE_DIR = ROOT / "outputs" / "server_1961_trace_full_20260623" / "traces"
THERMOSOIL_TRACE_DIR = ROOT / "outputs" / "server_1961_diffuco_trace_20260624_181143" / "traces"


def _assert_values_exact(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_values_exact(actual[key], expected[key])
        return
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_values_exact(left, right)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_values_exact(left, right)
        return
    try:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=0)
    except (TypeError, ValueError):
        assert actual == expected


def _assert_previous_step_packets_exact(actual, expected):
    assert actual.tstep == expected.tstep
    assert tuple(actual.fields_by_component.keys()) == tuple(expected.fields_by_component.keys())
    assert actual.provenance_by_component == expected.provenance_by_component
    for component, expected_fields in expected.fields_by_component.items():
        actual_fields = actual.fields_by_component[component]
        assert tuple(actual_fields.keys()) == tuple(expected_fields.keys())
        for name, expected_value in expected_fields.items():
            _assert_values_exact(actual_fields[name], expected_value)


def _assert_values_close(actual, expected, *, rtol=1.0e-10, atol=1.0e-8):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_values_close(actual[key], expected[key], rtol=rtol, atol=atol)
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_values_close(left, right, rtol=rtol, atol=atol)
        return
    try:
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    except (TypeError, ValueError):
        assert actual == expected


def test_compiled_forcing_day_matches_audited_payload_assembly_exactly():
    context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=USED_RUN_DEF)
    actual = _paper_compiled_forcing_day(
        context,
        year=1961,
        start_tstep=48,
        steps_per_stomate=48,
    )
    expected_steps = []
    for tstep in range(48, 96):
        step = _paper_1961_step_bundle_from_context(
            context,
            year=1961,
            tstep=tstep,
        )
        expected_steps.append(
            _compiled_half_hour_forcing(
                DriverRuntimeHalfHourInput(
                    step=step,
                    base_payload=build_intersurf_first_step_payload(
                        step,
                        driver_z0_for_wind=0.1,
                        dt_sechiba=context.dt_sechiba,
                    ),
                )
            )
        )
    expected = jax.tree_util.tree_map(
        lambda *values: np.stack(values),
        *expected_steps,
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_driver_runtime_scalars_read_actual_used_run_def_values():
    runtime = read_driver_runtime_scalars(USED_RUN_DEF)

    assert runtime.dt_sechiba == 1800.0
    assert runtime.dt_stomate == 86400.0
    assert runtime.dt_days == 1.0
    assert runtime.stomate_ok_stomate is True
    assert runtime.stomate_ok_dgvm is False
    assert any("used_run.def lines 57-59" in item for item in runtime.provenance)


def test_paper_end_of_year_follows_slowproc_last_timestep_of_noleap_year():
    runtime = read_driver_runtime_scalars(USED_RUN_DEF)
    steps_per_day = int(round(runtime.dt_stomate / runtime.dt_sechiba))
    last_tstep = 365 * steps_per_day - 1

    assert _paper_first_step_do_slow(last_tstep, runtime) is True
    assert _paper_end_of_year(last_tstep - 1, runtime) is False
    assert _paper_end_of_year(last_tstep, runtime) is True
    assert _paper_end_of_year(last_tstep + steps_per_day, runtime) is False


def test_year_start_rebase_drops_hydrol_nroot_without_mutating_previous_state():
    nroot = np.ones((1, 14, 11), dtype=np.float64)
    state = DriverPreviousStepStatePacket(
        tstep=17519,
        fields_by_component={
            "hydrol_previous_step_state": {
                "mc": np.ones((1, 11, 3), dtype=np.float64),
                "nroot": nroot,
            },
            "diffuco_previous_step_state": {"temp_sol": np.asarray([300.0], dtype=np.float64)},
        },
        provenance_by_component={
            "hydrol_previous_step_state": ("hydrol_end restart writes us but not nroot",),
            "diffuco_previous_step_state": ("diffuco restart fields",),
        },
    )

    rebased = rebase_driver_state_for_year_start(state)

    assert rebased.tstep == -1
    assert rebased.fields_by_component is not state.fields_by_component
    assert rebased.fields_by_component["hydrol_previous_step_state"] is not state.fields_by_component[
        "hydrol_previous_step_state"
    ]
    assert "nroot" in state.fields_by_component["hydrol_previous_step_state"]
    assert "nroot" not in rebased.fields_by_component["hydrol_previous_step_state"]
    np.testing.assert_allclose(
        rebased.fields_by_component["hydrol_previous_step_state"]["mc"],
        state.fields_by_component["hydrol_previous_step_state"]["mc"],
    )
    assert any("HYDROL dynamic nroot is not a Fortran restart field" in item for item in rebased.provenance_by_component["hydrol_previous_step_state"])


def test_later_day_stomate_entry_overlay_carries_fixed_cryoturbation_depth_from_previous_day():
    restart = reference_case_first_step_restart_state(CONFIG, root=ROOT)
    template = restart.stomate
    slowproc = {name: getattr(template, name) for name in LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES}
    previous_fixed = np.full_like(template.fixed_cryoturbation_depth, 0.37)
    slowproc["fixed_cryoturbation_depth"] = previous_fixed
    state = DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component={"slowproc_stomate_previous_step_state": slowproc},
        provenance_by_component={
            "slowproc_stomate_previous_step_state": (
                "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90 lines 1235-1240 and 2825-2828",
            )
        },
    )

    entry, gaps = _paper_later_day_stomate_entry_state_from_previous(
        template=template,
        previous_state=state,
    )

    assert gaps == ()
    assert entry is not None
    np.testing.assert_allclose(entry.fixed_cryoturbation_depth, previous_fixed)


def test_later_day_stomate_entry_overlay_carries_persisted_peat_and_deep_carbon_state():
    restart = reference_case_first_step_restart_state(CONFIG, root=ROOT)
    template = restart.stomate
    slowproc = {name: getattr(template, name) for name in LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES}
    overrides = {
        "fpeat": np.asarray([0.23], dtype=np.float64),
        "deepC_a": np.full_like(template.deepC_a, 7.0),
        "deepC_s": np.full_like(template.deepC_s, 8.0),
        "deepC_p": np.full_like(template.deepC_p, 9.0),
        "soilc_total": np.full_like(template.soilc_total, 9.0),
        "thawed_humidity": np.asarray([0.44], dtype=np.float64),
        "depth_organic_soil": np.asarray([1.7], dtype=np.float64),
    }
    slowproc.update(overrides)
    state = DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component={"slowproc_stomate_previous_step_state": slowproc},
        provenance_by_component={
            "slowproc_stomate_previous_step_state": (
                "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90 lines 1201-1225 and 1466-1471",
            )
        },
    )

    entry, gaps = _paper_later_day_stomate_entry_state_from_previous(
        template=template,
        previous_state=state,
    )

    assert gaps == ()
    assert entry is not None
    for name, expected in overrides.items():
        np.testing.assert_allclose(getattr(entry, name), expected)


def test_later_day_stomate_overlay_contract_has_only_documented_nondynamic_exclusions():
    entry_fields = set(StomateRestartEntryState._fields)
    season_fields = set(StomateRestartSeasonState._fields)

    assert entry_fields - set(LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES) == {
        "prod10_total",
        "prod100_total",
    }
    assert set(LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES) - entry_fields == set()
    assert season_fields - set(
        (
            *STOMATE_DAY_SEASON_MEMORY_FIELDS,
            *STOMATE_DAY_SEASON_ANNUAL_FIELDS,
            *STOMATE_DAY_SEASON_BIOMETEOROLOGY_FIELDS,
            *STOMATE_DAY_SEASON_PERSISTED_FIELDS,
        )
    ) == {
        "date",
        "dt_days_read",
        "provenance",
    }


def test_paper_case_restart_branch_switches_match_ledger_classification():
    reference_run_def = ROOT / "outputs" / "reference_mode" / "used_run.def"
    values = {}
    for line in reference_run_def.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()

    assert values["FIRE_DISABLE"] == "y"
    assert values["LAND_COVER_CHANGE"] == "n"
    assert values["ENABLE_GRAZING"] == "n"
    assert values["GRM_ENABLE_GRAZING"] == "FALSE"
    assert values["CH4_CALCUL"] == "n"
    assert values["OK_PC"] == "n"
    assert values["DYN_PEAT"] == "n"
    assert values["PERMA_PEAT"] == "y"
    assert values["TF_DOC"] == "y"


def test_day_end_packet_carries_active_ok_leak_restart_state_fields():
    active_ok_leak_fields = {
        "litter_above",
        "litter_below",
        "litterpart",
        "dead_leaves",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "lignin_struc_above",
        "lignin_struc_below",
        "carbon_32l",
        "DOC",
        "interception_storage",
        "altmax",
        "fixed_cryoturbation_depth",
    }
    previous = DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component={"slowproc_stomate_previous_step_state": {}},
        provenance_by_component={"slowproc_stomate_previous_step_state": ("previous state",)},
    )
    stomate_state = DriverFirstDayStomateStatePacket(
        day_index=1,
        fields={name: np.asarray([index], dtype=np.float64) for index, name in enumerate(sorted(active_ok_leak_fields))},
        provenance_by_field={
            name: ("fortran_source/ORCHIDEE/src_stomate/stomate.f90 lines 3288-3489",)
            for name in active_ok_leak_fields
        },
    )

    day_end = _paper_day_end_state_packet(previous_state=previous, stomate_state=stomate_state)

    assert day_end is not None
    slowproc = day_end.fields_by_component["slowproc_stomate_previous_step_state"]
    assert active_ok_leak_fields <= set(slowproc)


def test_day_end_packet_carries_persisted_legacy_lcc_restart_state_fields():
    persisted_fields = {
        "carbon",
        "litter",
        "lignin_struc",
        "prod10",
        "prod100",
        "flux10",
        "flux100",
        "deepC_a",
        "deepC_s",
        "deepC_p",
    }
    previous = DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component={"slowproc_stomate_previous_step_state": {}},
        provenance_by_component={"slowproc_stomate_previous_step_state": ("previous state",)},
    )
    stomate_state = DriverFirstDayStomateStatePacket(
        day_index=1,
        fields={name: np.asarray([index], dtype=np.float64) for index, name in enumerate(sorted(persisted_fields))},
        provenance_by_field={
            name: (
                "fortran_source/ORCHIDEE/src_stomate/stomate_io.f90 reads legacy/LCC restart state",
                "fortran_source/ORCHIDEE/src_stomate/stomate.f90 keeps STOMATE inout state live across calls",
            )
            for name in persisted_fields
        },
    )

    day_end = _paper_day_end_state_packet(previous_state=previous, stomate_state=stomate_state)

    assert day_end is not None
    slowproc = day_end.fields_by_component["slowproc_stomate_previous_step_state"]
    assert persisted_fields <= set(slowproc)


def test_driver_step_bundle_uses_requested_year_for_co2_ccanopy():
    step1961 = load_paper_1961_step_bundle(CONFIG, year=1961, tstep=0)
    step1962 = load_paper_1961_step_bundle(CONFIG, year=1962, tstep=0)

    assert step1961.year == 1961
    assert step1962.year == 1962
    assert step1961.co2_ppm != step1962.co2_ppm
    np.testing.assert_allclose(step1961.ccanopy, np.full((1,), step1961.co2_ppm))
    np.testing.assert_allclose(step1962.ccanopy, np.full((1,), step1962.co2_ppm))


def test_first_timestep_scaffold_builds_driver_payload_and_stomate_restart_boundary():
    scaffold = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert scaffold.step.tstep == 0
    assert scaffold.initial_state_mode == "restart_backed_reference_case"
    assert scaffold.payload.kjpindex == 1
    assert scaffold.stomate_entry_assembly.payload["kjit"] == 1
    assert "precip_rain" in scaffold.stomate_entry_assembly.covered_inputs
    assert "DOC_to_topsoil" in scaffold.stomate_entry_assembly.covered_inputs
    assert "t2m" not in scaffold.stomate_entry_assembly.missing_by_source.get("forcing", ())
    assert "temp_sol" not in scaffold.stomate_entry_assembly.missing_by_source.get("enerbil", ())
    assert "shumdiag" not in scaffold.stomate_entry_assembly.missing_by_source.get("hydrol", ())
    assert "stempdiag" not in scaffold.stomate_entry_assembly.missing_by_source.get("thermosoil", ())
    assert "biomass" in scaffold.stomate_entry_assembly.covered_inputs
    assert "litter_above" in scaffold.stomate_entry_assembly.covered_inputs
    assert "carbon_32l" in scaffold.stomate_entry_assembly.covered_inputs
    assert "lai" in scaffold.stomate_entry_assembly.covered_inputs
    assert "height" in scaffold.stomate_entry_assembly.covered_inputs
    assert "veget_max_new" in scaffold.stomate_entry_assembly.covered_inputs
    assert "sat_duration" in scaffold.stomate_entry_assembly.covered_inputs
    assert scaffold.stomate_pre_step is not None
    assert scaffold.first_step_restart_state is not None
    assert scaffold.first_step_restart_state.stomate_input.name == "stomate_start.nc"
    assert scaffold.diffuco_control_coverage is not None
    assert scaffold.diffuco_control_coverage.complete
    assert scaffold.diffuco_control_coverage.z_soil.status == "constructible"
    assert scaffold.diffuco_control_coverage.rprof.status == "constructible"
    assert scaffold.diffuco_control_coverage.biomass.status == "covered"
    assert scaffold.diffuco_control_coverage.tide_height.status == "covered"
    assert scaffold.diffuco_control_coverage.missing_inputs == ()
    assert scaffold.diffuco_control_salinity is not None
    assert scaffold.diffuco_control_salinity.ok
    assert scaffold.diffuco_control_salinity.missing_inputs == ()
    assert "control_salinity" in scaffold.diffuco_control_salinity.payload
    assert scaffold.diffuco_control_assembly is not None
    assert scaffold.diffuco_control_assembly.ok
    assert scaffold.diffuco_control_assembly.missing_inputs == ()
    assert "z_soil" in scaffold.diffuco_control_assembly.payload
    assert "rprof" in scaffold.diffuco_control_assembly.payload
    assert "biomass" in scaffold.diffuco_control_assembly.payload
    assert "control_inudate" in scaffold.diffuco_control_assembly.payload
    assert scaffold.enerbil_first_step_coverage is not None
    assert "swnet" in scaffold.enerbil_first_step_coverage.covered_by_source_kernel
    assert "soilcap" in scaffold.enerbil_first_step_coverage.covered_by_source_kernel
    assert "soilflx_pft" in scaffold.enerbil_first_step_coverage.covered_by_source_kernel
    assert "emis" in scaffold.enerbil_first_step_coverage.covered_by_source_kernel
    assert "rau" in scaffold.enerbil_first_step_coverage.covered_by_source_kernel
    assert scaffold.enerbil_first_step_coverage.ok
    assert "vbeta" not in scaffold.enerbil_first_step_coverage.missing
    assert "q_cdrag" not in scaffold.enerbil_first_step_coverage.missing
    assert "temp_sol" not in scaffold.enerbil_first_step_coverage.missing
    assert "qsurf" not in scaffold.enerbil_first_step_coverage.missing
    assert scaffold.diffuco_first_step_precall is not None
    assert scaffold.diffuco_first_step_precall.ok
    assert scaffold.diffuco_first_step_local_enerbil_precall is not None
    assert "temp_sol" in scaffold.diffuco_first_step_precall.restart_inputs
    assert "roughheight_pft" in scaffold.diffuco_first_step_precall.restart_inputs
    assert "qsintmax" in scaffold.diffuco_first_step_precall.source_kernel_inputs
    assert "tot_bare_soil" in scaffold.diffuco_first_step_precall.source_kernel_inputs
    assert "frac_snow_veg" in scaffold.diffuco_first_step_precall.source_kernel_inputs
    assert "frac_snow_nobio" in scaffold.diffuco_first_step_precall.source_kernel_inputs
    assert "flood_frac" in scaffold.diffuco_first_step_precall.source_kernel_inputs
    assert "flood_res" in scaffold.diffuco_first_step_precall.source_kernel_inputs
    assert "control_salinity" not in scaffold.diffuco_first_step_precall.missing_local_process_inputs
    assert "frac_snow_veg" not in scaffold.diffuco_first_step_precall.missing_local_process_inputs
    assert "flood_frac" not in scaffold.diffuco_first_step_precall.missing_local_process_inputs
    assert "flood_res" not in scaffold.diffuco_first_step_precall.missing_local_process_inputs
    assert scaffold.enerbil_first_step_precall is not None
    assert scaffold.enerbil_first_step_local is not None
    assert scaffold.enerbil_first_step_precall.ok
    assert "swnet" in scaffold.enerbil_first_step_precall.payload
    expected_swnet = (
        1.0
        - (
            scaffold.first_step_restart_state.driver_albedo.albedo[:, 0]
            + scaffold.first_step_restart_state.driver_albedo.albedo[:, 1]
        )
        / 2.0
    ) * scaffold.payload.swdown
    np.testing.assert_allclose(
        scaffold.enerbil_first_step_precall.payload["swnet"],
        expected_swnet,
    )
    assert "emis" in scaffold.enerbil_first_step_precall.source_kernel_inputs
    assert "rau" in scaffold.enerbil_first_step_precall.source_kernel_inputs
    assert "soilcap" in scaffold.enerbil_first_step_precall.restart_inputs
    assert "soilflx_pft" in scaffold.enerbil_first_step_precall.restart_inputs
    assert "temp_sol" not in scaffold.enerbil_first_step_precall.missing_inputs
    assert "qsurf" not in scaffold.enerbil_first_step_precall.missing_inputs
    assert "evapot_corr" not in scaffold.enerbil_first_step_precall.missing_inputs
    assert "q_cdrag" not in scaffold.enerbil_first_step_precall.missing_inputs
    assert "vbeta" not in scaffold.enerbil_first_step_precall.missing_inputs
    assert "swnet" not in scaffold.enerbil_first_step_precall.missing_inputs
    assert "driver_forcing_step" in scaffold.covered_components
    assert "reference_case_first_step_restart_state" in scaffold.covered_components
    assert scaffold.sechiba_first_step_coverage is not None
    assert not scaffold.sechiba_first_step_coverage.ready_for_sechiba_explicit_step
    assert "hydrol_restart_anchors" in scaffold.sechiba_first_step_coverage.closed_module_boundaries
    assert "thermosoil_restart_recurrence_state" in scaffold.sechiba_first_step_coverage.closed_module_boundaries
    assert "slowproc_restart_state" in scaffold.sechiba_first_step_coverage.closed_module_boundaries
    assert "DIFFUCO:tide_height" not in scaffold.sechiba_first_step_coverage.missing_components
    assert "DIFFUCO:frac_snow_veg" not in scaffold.sechiba_first_step_coverage.missing_components
    assert "DIFFUCO:flood_frac" not in scaffold.sechiba_first_step_coverage.missing_components
    assert "DIFFUCO:pft14_trans_co2_gpp" not in scaffold.sechiba_first_step_coverage.missing_components
    assert "ENERBIL:vbeta" not in scaffold.sechiba_first_step_coverage.missing_components
    assert "HYDROL:hydrol_to_stomate_diagnostics" not in scaffold.sechiba_first_step_coverage.missing_components
    assert "THERMOSOIL:thermosoil_to_stomate_stempdiag" not in scaffold.sechiba_first_step_coverage.missing_components
    assert "SLOWPROC:post_stomate_surface_update" in scaffold.sechiba_first_step_coverage.missing_components
    assert "SLOWPROC:slowproc_to_stomate_dynamic_surface_state" in scaffold.sechiba_first_step_coverage.missing_components
    assert "slowproc_restart_stomate_entry_state" in scaffold.covered_components
    assert "stomate_restart_entry_state" in scaffold.covered_components
    assert "diffuco_control_inundation_input_coverage" in scaffold.covered_components
    assert "diffuco_first_step_precall_partial_payload" in scaffold.covered_components
    assert "diffuco_first_step_local_enerbil_precall" in scaffold.covered_components
    assert "enerbil_first_step_input_coverage" in scaffold.covered_components
    assert "enerbil_first_step_local_step" in scaffold.covered_components
    assert "hydrol_first_step_module_closure" in scaffold.covered_components
    assert "condveg_first_step_module_closure" in scaffold.covered_components
    assert "thermosoil_first_step_module_closure" in scaffold.covered_components
    assert "stomate_restart_ok_leak_pre_step_boundary" in scaffold.covered_components
    assert "sechiba_prognostic_state_before_step" not in scaffold.missing_components
    assert "hydrol_soil_water_state_before_step" not in scaffold.missing_components
    assert "thermosoil_restart_recurrence_state" in scaffold.covered_components
    assert "diffuco_same_step_precall_state_and_gpp" in scaffold.missing_components
    assert "hydrol_same_step_diagnostics_to_stomate" not in scaffold.missing_components
    assert "thermosoil_same_step_stempdiag_to_stomate" not in scaffold.missing_components
    assert "stomate_accumulator_state_before_step" not in scaffold.missing_components
    assert scaffold.stomate_first_step_daily_accumulation is not None
    assert scaffold.stomate_first_step_daily_accumulation.ok
    assert scaffold.stomate_bundle_source is None
    assert scaffold.stomate_restart_input_bundles is None
    assert "stomate_main_chain_restart_input_bundles_waiting_for_do_slow" in scaffold.covered_components
    assert "stomate_daily_process_waiting_for_do_slow" in scaffold.missing_components
    assert not scaffold.ready_to_run_processes

    boundary = scaffold.stomate_pre_step.pre_step.boundary_inputs
    np.testing.assert_allclose(boundary.output_inputs["contfrac"], [0.5625])
    assert boundary.ok_leak_inputs["nslm"] == 11
    assert boundary.ok_leak_inputs["ndeep"] == 32


def test_driver_scaffolds_mark_reference_restart_mode_without_full_day_run():
    first = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    multi0 = paper_1961_driver_multiday_scaffold(
        CONFIG,
        year=1961,
        ndays=0,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert first.initial_state_mode == "restart_backed_reference_case"
    assert multi0.initial_state_mode == "restart_backed_reference_case"
    assert "cold_start" not in first.initial_state_mode


def test_cold_start_first_step_coverage_exposes_initialization_gaps():
    coverage = paper_1961_driver_cold_start_first_step_coverage(
        CONFIG,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
    )

    assert coverage.mode == "cold_start_no_restart"
    assert coverage.ready_for_cold_start_first_step
    assert "cold_start_biomass_zero_fallback" in coverage.covered_by_component["diffuco"]
    assert "temp_sol_constant_ENERBIL_TSURF" in coverage.covered_by_component["enerbil"]
    assert "qsurf_initialized_from_qair" in coverage.covered_by_component["enerbil"]
    assert "static_fc_grazing_humcste_use" in coverage.covered_by_component["slowproc"]
    slowproc_init = coverage.initialized_payloads["slowproc_init_pft14"]
    assert slowproc_init.found_restart is False
    assert slowproc_init.veget.shape == (1, 14)
    first_step_payload = coverage.initialized_payloads["first_step_payload"]
    np.testing.assert_allclose(np.asarray(slowproc_init.salinity), first_step_payload.salinity)
    np.testing.assert_allclose(np.asarray(slowproc_init.tide_height), first_step_payload.tide_height)
    assert "mc_mcl_constant_HYDROL_MOISTURE_CONTENT" in coverage.covered_by_component["hydrol"]
    assert "explicit_snow_initialize_defaults" in coverage.covered_by_component["hydrol"]
    assert "ptn_read_reftempfile_static_input" in coverage.covered_by_component["thermosoil"]
    enerbil_surface = coverage.initialized_payloads["enerbil_surface_state"]
    np.testing.assert_allclose(np.asarray(enerbil_surface.temp_sol), 280.0)
    assert np.asarray(enerbil_surface.qsurf).shape == (1,)
    hydrol_cold = coverage.initialized_payloads["hydrol_cold_start_state"]
    np.testing.assert_allclose(np.asarray(hydrol_cold.mc), 0.3)
    np.testing.assert_allclose(np.asarray(hydrol_cold.water2infilt), 0.0)
    np.testing.assert_allclose(np.asarray(hydrol_cold.explicit_snow.snowrho), 50.0)
    assert "hydrol_cold_start_thermosoil_moisture" in coverage.initialized_payloads
    refsoc = coverage.initialized_payloads["thermosoil_refSOC"]
    assert refsoc.shape[0] == 1
    assert refsoc.shape[1] > 1
    assert np.all(np.isfinite(refsoc))
    thermosoil_cold_coef = coverage.initialized_payloads["thermosoil_cold_start_coef"]
    assert thermosoil_cold_coef.ok
    assert "refSOC_non_restart_initialization" not in thermosoil_cold_coef.missing_inputs
    assert "refSOC_read_refSOCfile_static_input" in coverage.covered_by_component["thermosoil"]
    assert "soilcap_soilflx_non_restart_initialization" in coverage.covered_by_component["thermosoil"]
    np.testing.assert_allclose(
        np.asarray(coverage.initialized_payloads["thermosoil_ptn_constant"])[0, :, 13],
        296.454946279493,
    )
    assert coverage.missing_by_component["diffuco"] == ()
    assert "hydrol:soil_water_non_restart_initialization" not in coverage.missing_components
    assert coverage.missing_by_component["hydrol"] == ()
    assert "hydrol_first_step_module_closed" in coverage.covered_by_component["hydrol"]
    assert coverage.initialized_payloads["diffuco_first_step_precall"].ok
    assert coverage.initialized_payloads["diffuco_first_step_local_enerbil_precall"] is not None
    assert coverage.initialized_payloads["enerbil_first_step_precall"].ok
    assert coverage.initialized_payloads["enerbil_first_step_local"] is not None
    assert coverage.initialized_payloads["hydrol_first_step_precall"].ok
    assert coverage.initialized_payloads["hydrol_first_step_module"].ok
    hydrol_module = coverage.initialized_payloads["hydrol_first_step_module"]
    assert hydrol_module.module_inputs["peat_hydro"] is False
    np.testing.assert_allclose(np.asarray(hydrol_module.module_inputs["free_drain_coef"])[0, 3], 1.0)
    tile4 = hydrol_module.module.soil.tile_results[3]
    np.testing.assert_allclose(np.asarray(tile4.coef_after_infilt.a)[0, 0], 13.6733100285188)
    np.testing.assert_allclose(np.asarray(tile4.solve.dr_ns_final)[0], 8.189820446528955e-07)
    np.testing.assert_allclose(np.asarray(tile4.solve.mcl_after_tridiag)[0, 0], 0.299441884927089)
    np.testing.assert_allclose(np.asarray(hydrol_module.thermosoil_moisture.shumdiag_perma)[0, 0], 0.299459936689220)
    np.testing.assert_allclose(np.asarray(hydrol_module.thermosoil_moisture.tmc_layh)[0, 0], 0.292727210974850)
    assert coverage.initialized_payloads["condveg_first_step_module"].ok
    assert coverage.initialized_payloads["thermosoil_first_step_module"].ok
    thermosoil_module = coverage.initialized_payloads["thermosoil_first_step_module"]
    thermosoil_trace_pft = find_server_record(
        "sechiba_bridge_thermosoil",
        tag="after_thermosoil_before_slowproc_pft",
        criteria={"kjit": 1, "ji": 1, "jv": 14},
        root=THERMOSOIL_TRACE_DIR,
    )
    thermosoil_trace_layer = find_server_record(
        "sechiba_bridge_thermosoil",
        tag="after_thermosoil_before_slowproc_layer",
        criteria={"kjit": 1, "ji": 1, "jv": 14, "jsl": 1},
        root=THERMOSOIL_TRACE_DIR,
    )
    assert thermosoil_trace_pft is not None
    assert thermosoil_trace_layer is not None
    np.testing.assert_allclose(
        np.asarray(thermosoil_module.downstream_payload["soilcap"])[0],
        thermosoil_trace_pft["soilcap"],
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(thermosoil_module.downstream_payload["soilflx"])[0],
        thermosoil_trace_pft["soilflx"],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(thermosoil_module.downstream_payload["gtemp"])[0],
        thermosoil_trace_pft["gtemp"],
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(thermosoil_module.downstream_payload["stempdiag"])[0, 0],
        thermosoil_trace_layer["stempdiag"],
        rtol=0,
        atol=1e-8,
    )
    previous = coverage.initialized_payloads["cold_start_first_step_previous_state"]
    assert previous.tstep == 0
    cold_hydrol_state = previous.fields_by_component["hydrol_previous_step_state"]
    assert "njsc" in cold_hydrol_state
    assert "profil_froz_hydro_ns" in cold_hydrol_state
    assert "temp_hydro" in cold_hydrol_state
    assert "z0m" in previous.fields_by_component["diffuco_previous_step_state"]
    assert "soilcap" in previous.fields_by_component["enerbil_previous_step_state"]
    assert "ptn" in previous.fields_by_component["thermosoil_previous_step_state"]
    assert "stempdiag" in previous.fields_by_component["thermosoil_previous_step_state"]
    first_entry = coverage.initialized_payloads["cold_start_first_step_entry_payload"]
    assert first_entry["kjit"] == 1
    assert "gpp" in first_entry
    assert "stempdiag" in first_entry
    assert "fwet_new" in first_entry
    assert "liqwt_ratio" in first_entry
    assert "hydrol:vegstress_grid_pft_restart_fallback_target_initialization" not in coverage.missing_components
    assert coverage.missing_by_component["stomate"] == ()
    stomate_daily = coverage.initialized_payloads["stomate_cold_start_daily_accumulators"]
    np.testing.assert_allclose(np.asarray(stomate_daily.t2m_daily), 0.0)
    np.testing.assert_allclose(np.asarray(stomate_daily.tsurf_daily), np.asarray(coverage.initialized_payloads["forcing_t2m_for_stomate"]))
    np.testing.assert_allclose(np.asarray(stomate_daily.t2m_min_daily), 1.0e33)
    np.testing.assert_allclose(np.asarray(stomate_daily.t2m_max_daily), -1.0e33)
    stomate_season = coverage.initialized_payloads["stomate_cold_start_season_state"]
    assert stomate_season.tau_longterm == 2.0
    np.testing.assert_allclose(np.asarray(stomate_season.gdd_init_date[:, 0]), 365.0)
    np.testing.assert_allclose(np.asarray(stomate_season.t2m_longterm), np.asarray(coverage.initialized_payloads["forcing_t2m_for_stomate"]))
    stomate_entry = coverage.initialized_payloads["stomate_cold_start_entry_state"]
    np.testing.assert_allclose(np.asarray(stomate_entry.biomass), 0.0)
    np.testing.assert_allclose(np.asarray(stomate_entry.sla_calc)[0, 13], 0.0153)
    np.testing.assert_allclose(np.asarray(stomate_entry.litter_above), 0.0)
    np.testing.assert_allclose(np.asarray(stomate_entry.carbon_32l), 0.0)
    np.testing.assert_allclose(np.asarray(stomate_entry.DOC), 0.0)
    np.testing.assert_allclose(
        np.asarray(previous.fields_by_component["slowproc_stomate_previous_step_state"]["sla_calc"]),
        np.asarray(stomate_entry.sla_calc),
    )
    assert "soilcap_from_thermosoil_coef" in coverage.covered_by_component["enerbil"]
    assert "enerbil:soilcap_non_restart_thermosoil_coef" not in coverage.missing_components
    assert "thermosoil:thermosoil_coef_non_restart_soilcap_soilflx_waiting_for_refSOC" not in coverage.missing_components


def test_cold_start_previous_state_drives_later_sechiba_steps_without_restart_aliasing():
    coverage = paper_1961_driver_cold_start_first_step_coverage(
        CONFIG,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
    )
    previous = coverage.initialized_payloads["cold_start_first_step_previous_state"]

    for tstep in (1, 2):
        step = paper_1961_next_step_hydrol_precall_scaffold(
            CONFIG,
            year=1961,
            tstep=tstep,
            used_run_def_path=USED_RUN_DEF,
            previous_state=previous,
        )
        assert step.missing_previous_state_fields == ()
        assert step.hydrol_precall is not None
        assert step.hydrol_precall.ok
        assert "profil_froz" not in step.hydrol_precall.missing_inputs
        assert "kfact_root" not in step.hydrol_precall.missing_inputs
        assert step.ready_for_next_state
        previous = driver_previous_step_state_from_next_step_hydrol_scaffold(step)

    assert previous.tstep == 2
    assert "njsc" in previous.fields_by_component["hydrol_previous_step_state"]
    assert "profil_froz_hydro_ns" in previous.fields_by_component["hydrol_previous_step_state"]
    assert "soilcap" in previous.fields_by_component["enerbil_previous_step_state"]


def test_cold_start_day_scaffold_reaches_limited_stomate_entry_chain():
    day = paper_1961_driver_cold_start_day_scaffold(
        CONFIG,
        year=1961,
        max_steps=3,
        used_run_def_path=USED_RUN_DEF,
    )

    assert day.initial_state_mode == "cold_start_no_restart"
    assert day.completed_steps == 3
    assert len(day.completed_entry_payloads) == 3
    assert [payload["kjit"] for payload in day.completed_entry_payloads] == [1, 2, 3]
    assert day.stopped_at_tstep is None
    assert day.state_gaps == ()
    assert not day.ready_for_day_boundary
    assert day.daily_process_fold is None
    assert day.stomate_restart_input_bundles is None
    assert day.stomate_daily_carbon is None
    assert day.first_day_end_state is None
    assert "cold_start_stomate_daily_carbon_and_day_end_state" not in day.missing_components
    assert day.previous_step_state is not None
    assert day.previous_step_state.tstep == 2
    assert "gpp" in day.completed_entry_payloads[0]
    assert "stempdiag" in day.completed_entry_payloads[0]
    assert "fwet_new" in day.completed_entry_payloads[0]
    assert "liqwt_ratio" in day.completed_entry_payloads[0]
    assert "soilcap" in day.previous_step_state.fields_by_component["enerbil_previous_step_state"]


def test_cold_start_day_scaffold_reaches_daily_fold_from_no_restart_sla_state():
    day = paper_1961_driver_cold_start_day_scaffold(
        CONFIG,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
    )

    assert day.initial_state_mode == "cold_start_no_restart"
    assert day.ready_for_day_boundary
    assert day.daily_process_fold is not None
    assert day.daily_process_fold.ok
    assert day.stomate_restart_input_bundles.prescribe_inputs["stomate_restart_none"] is True
    slowproc = day.previous_step_state.fields_by_component["slowproc_stomate_previous_step_state"]
    assert np.asarray(slowproc["sla_calc"]).shape == (1, 14)
    assert len(day.daily_process_fold.accumulator.step_results) == day.steps_per_stomate
    assert day.daily_process_fold.maintenance.failed_step is None
    assert "cold_start_stomate_daily_process_fold" not in day.missing_components


def test_cold_start_day_end_feeds_second_day_dynamic_vcmax_and_temp_growth():
    day1 = paper_1961_driver_cold_start_day_scaffold(
        CONFIG,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
    )
    end_slowproc = day1.first_day_end_state.fields_by_component["slowproc_stomate_previous_step_state"]

    np.testing.assert_allclose(np.asarray(end_slowproc["temp_growth"]), 16.626392745971714)
    np.testing.assert_allclose(np.asarray(end_slowproc["assim_param"])[0, 13, 0], 16.598173515981735)

    day2 = paper_1961_driver_later_day_scaffold(
        CONFIG,
        year=1961,
        start_tstep=48,
        used_run_def_path=USED_RUN_DEF,
        previous_state=day1.first_day_end_state,
    )

    assert day2.ready_for_day_boundary
    assert day2.completed_entry_payloads[0]["kjit"] == 49
    np.testing.assert_allclose(np.asarray(day2.completed_entry_payloads[0]["humrel"])[0, 13], 1.0)
    np.testing.assert_allclose(
        np.asarray(day2.completed_entry_payloads[0]["gpp"])[0, 13],
        2.434493100106178e-02,
        rtol=1.0e-3,
    )


def test_second_day_second_half_hour_keeps_temp_growth_and_diffuco_trace_parity():
    context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=USED_RUN_DEF)
    day1 = paper_1961_driver_cold_start_day_scaffold(
        CONFIG,
        year=1961,
        fixed_format_trace_dir=TRACE_DIR,
        used_run_def_path=USED_RUN_DEF,
        prepared_context=context,
    )
    step49 = paper_1961_next_step_hydrol_precall_scaffold(
        CONFIG,
        year=1961,
        tstep=48,
        fixed_format_trace_dir=TRACE_DIR,
        used_run_def_path=USED_RUN_DEF,
        previous_state=day1.first_day_end_state,
        prepared_context=context,
    )
    state49 = driver_previous_step_state_from_next_step_hydrol_scaffold(step49)
    step50 = paper_1961_next_step_hydrol_precall_scaffold(
        CONFIG,
        year=1961,
        tstep=49,
        fixed_format_trace_dir=TRACE_DIR,
        used_run_def_path=USED_RUN_DEF,
        previous_state=state49,
        prepared_context=context,
    )
    trans = step50.enerbil.diffuco_local_enerbil_precall.result.boundary.process_chain.trans_co2
    trace = find_server_record(
        "diffuco_trans_co2",
        tag="after_diffuco_trans_co2_pft14",
        criteria={"kjit": 50, "ji": 1, "jv": 14},
        scan_limit=5000,
    )

    assert step50.ready_for_next_state
    np.testing.assert_allclose(
        np.asarray(step50.enerbil.diffuco_precall.payload["temp_growth"])[0],
        trace["temp_growth"],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.asarray(trans.output.gpp)[0, 0], trace["gpp"], rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(trans.controlled_assimtot)[0], trace["assimtot"], rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(trans.vpd_boundary.fvpd)[0, 0], trace["fvpd"], rtol=1.0e-12)


def test_prepared_driver_context_matches_runtime_scalars_and_reuses_static_bundle():
    runtime = read_driver_runtime_scalars(USED_RUN_DEF)
    context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=USED_RUN_DEF)

    assert context.runtime == runtime
    assert context.nflow == 6
    assert context.nvm == 14
    assert len(context.ok_laidev) == 14
    assert context.first_step_bundle is not None
    assert context.first_step_bundle.domain.nbindex == 1
    np.testing.assert_allclose(context.cwrr_grid.znt, context.znt)


def test_later_timestep_scaffold_keeps_previous_step_state_gap_visible():
    scaffold = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=1,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert scaffold.step.tstep == 1
    assert scaffold.initial_state_mode == "previous_step_state_required"
    assert np.allclose(scaffold.payload.temp_air, [289.9084014892578])
    assert scaffold.stomate_entry_assembly.payload["kjit"] == 2
    assert "DOC_to_subsoil" in scaffold.stomate_entry_assembly.covered_inputs
    assert "biomass" in scaffold.stomate_entry_assembly.missing_by_source["stomate_state"]
    assert scaffold.stomate_pre_step is None
    assert scaffold.first_step_restart_state is None
    assert scaffold.diffuco_control_coverage is None
    assert scaffold.diffuco_control_salinity is None
    assert scaffold.diffuco_control_assembly is None
    assert scaffold.diffuco_first_step_precall is None
    assert scaffold.enerbil_first_step_coverage is None
    assert scaffold.enerbil_first_step_precall is None
    assert scaffold.hydrol_first_step_precall is None
    assert scaffold.sechiba_first_step_coverage is None
    assert "stomate_restart_state_after_previous_step" in scaffold.missing_components
    assert any("previous-step model state is required" in note for note in scaffold.notes)
    assert not scaffold.ready_to_run_processes


def test_first_timestep_scaffold_diffuco_coverage_closes_when_full_tide_trace_is_supplied():
    static_truth = read_static_fields_from_fixed_format_trace_dir(TRACE_DIR)
    scaffold = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    assert scaffold.diffuco_control_coverage is not None
    assert scaffold.diffuco_control_coverage.complete

    traced_step = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    traced_coverage = traced_step.diffuco_control_coverage
    assert traced_coverage is not None
    assert static_truth.tide_height is not None
    assert traced_coverage.complete
    assert traced_coverage.tide_height.status == "covered"
    assert traced_coverage.missing_inputs == ()
    assert traced_step.diffuco_control_assembly is not None
    assert traced_step.diffuco_control_salinity is not None
    assert traced_step.diffuco_control_salinity.ok
    assert traced_step.diffuco_control_salinity.missing_inputs == ()
    assert "control_salinity" in traced_step.diffuco_control_salinity.payload
    assert traced_step.diffuco_control_assembly.ok
    assert traced_step.diffuco_control_assembly.missing_inputs == ()
    assert "control_inudate" in traced_step.diffuco_control_assembly.payload
    assert np.asarray(traced_step.diffuco_control_assembly.payload["control_inudate"]).shape == (1,)
    assert traced_step.diffuco_first_step_precall is not None
    assert traced_step.diffuco_first_step_precall.ok
    assert traced_step.diffuco_first_step_precall.missing_local_process_inputs == ()
    assert "control_salinity" in traced_step.diffuco_first_step_precall.control_inputs
    assert "control_inudate" in traced_step.diffuco_first_step_precall.control_inputs
    assert traced_step.diffuco_first_step_local_enerbil_precall is not None
    local_payload = traced_step.diffuco_first_step_local_enerbil_precall.result.payload.payload
    assert "q_cdrag" in local_payload
    assert "vbeta" in local_payload
    assert traced_step.enerbil_first_step_precall is not None
    assert "q_cdrag" not in traced_step.enerbil_first_step_precall.missing_inputs
    assert "q_cdrag_pft" not in traced_step.enerbil_first_step_precall.missing_inputs
    assert "vbeta" not in traced_step.enerbil_first_step_precall.missing_inputs
    assert "vbeta_pft" not in traced_step.enerbil_first_step_precall.missing_inputs
    assert "vbeta3" not in traced_step.enerbil_first_step_precall.missing_inputs
    assert "q_cdrag" in traced_step.enerbil_first_step_precall.diffuco_inputs
    assert "vbeta" in traced_step.enerbil_first_step_precall.diffuco_inputs
    assert traced_step.enerbil_first_step_local is not None
    assert np.asarray(traced_step.enerbil_first_step_local.payload["temp_sol_new"]).shape == (1,)
    assert np.asarray(traced_step.enerbil_first_step_local.payload["qsurf"]).shape == (1,)
    assert traced_step.hydrol_first_step_precall is not None
    assert traced_step.hydrol_first_step_precall.enerbil_boundary.ok
    assert "humrelv" not in traced_step.hydrol_first_step_precall.missing_inputs
    assert "us" not in traced_step.hydrol_first_step_precall.missing_inputs
    assert "evap_bare_lim_ns" not in traced_step.hydrol_first_step_precall.missing_inputs
    assert "profil_froz" not in traced_step.hydrol_first_step_precall.missing_inputs
    assert "kfact_root" not in traced_step.hydrol_first_step_precall.missing_inputs
    assert traced_step.hydrol_first_step_precall.ok
    assert np.asarray(traced_step.hydrol_first_step_precall.payload["kfact_root"]).shape == (1, 11, 6)
    assert np.asarray(traced_step.hydrol_first_step_precall.payload["profil_froz"]).shape == (1, 11, 6)
    assert np.asarray(traced_step.hydrol_first_step_precall.payload["stempdiag"]).shape == (1, 11)
    assert np.asarray(traced_step.hydrol_first_step_precall.payload["temp_hydro"]).shape == (1, 11)
    assert "same_step_enerbil_evaporation_inputs" not in traced_step.hydrol_first_step_precall.missing_inputs
    np.testing.assert_allclose(np.asarray(traced_step.hydrol_first_step_precall.payload["soiltile"]), [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    assert traced_step.hydrol_first_step_module is not None
    assert traced_step.hydrol_first_step_module.ok
    assert traced_step.hydrol_first_step_module.missing_inputs == ()
    assert np.asarray(traced_step.hydrol_first_step_module.outputs.wat_flux).shape == (1, 11, 6)
    assert np.asarray(traced_step.hydrol_first_step_module.diagnostics.humrel).shape == (1, 14)
    assert np.asarray(traced_step.hydrol_first_step_module.thermosoil_moisture.mc_layh_pft).shape == (1, 11, 14)
    for name in (
        "humrel",
        "shumdiag",
        "soil_mc",
        "wat_flux",
        "stempdiag",
        "tdeep",
        "hsdeep",
        "zz_deep",
        "zz_coef_deep",
        "snow",
        "snowdz",
        "snowrho",
        "gpp",
        "temp_sol",
        "fwet_new",
    ):
        assert name in traced_step.stomate_entry_assembly.covered_inputs
    for source, name in (
        ("hydrol", "humrel"),
        ("thermosoil", "stempdiag"),
        ("thermosoil_static", "zz_deep"),
        ("hydrol_condveg", "snow"),
        ("condveg", "snowrho"),
        ("diffuco", "gpp"),
        ("enerbil", "temp_sol"),
        ("hydrol", "fwet_new"),
        ("hydrol", "erodepth"),
    ):
        assert name not in traced_step.stomate_entry_assembly.missing_by_source.get(source, ())
    assert traced_step.stomate_first_step_daily_accumulation is not None
    assert traced_step.stomate_first_step_daily_accumulation.ok
    assert "gpp_daily" in traced_step.stomate_first_step_daily_accumulation.fields
    assert "precip_daily" in traced_step.stomate_first_step_daily_accumulation.fields
    assert "snowfall_daily" in traced_step.stomate_first_step_daily_accumulation.fields
    assert "snowmass_daily" in traced_step.stomate_first_step_daily_accumulation.fields
    assert "tmc_topgrass_daily" in traced_step.stomate_first_step_daily_accumulation.fields
    assert "stomate_accumulator_state_before_step" not in traced_step.missing_components
    np.testing.assert_allclose(
        np.asarray(traced_step.stomate_first_step_daily_accumulation.local_prep.gpp_d)[0, 13],
        6.339203652842375,
    )
    assert traced_step.sechiba_first_step_coverage is not None
    assert "diffuco_control_inundation_inputs" in traced_step.sechiba_first_step_coverage.closed_module_boundaries
    assert "diffuco_first_step_local_enerbil_precall" in traced_step.sechiba_first_step_coverage.closed_module_boundaries
    assert "DIFFUCO:tide_height" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "DIFFUCO:salinity_control" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "DIFFUCO:inundation_control" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "DIFFUCO:pft14_trans_co2_gpp" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "DIFFUCO:diffuco_comb_final_beta_bundle" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "DIFFUCO:after_diffuco_precall_payload" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "ENERBIL:q_cdrag" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "ENERBIL:vbeta" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "ENERBIL:psold" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "ENERBIL:qsol_sat_new" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "ENERBIL:temp_sol_new" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "ENERBIL:qair_new" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:same_step_enerbil_evaporation_inputs" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:humrelv" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:us" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:evap_bare_lim_ns" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:profil_froz" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:kfact_root" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:hydrol_main_outputs" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:hydrol_to_stomate_diagnostics" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "HYDROL:hydrol_to_thermosoil_moisture" not in traced_step.sechiba_first_step_coverage.missing_components
    assert "enerbil_to_hydrol_first_step_boundary" in traced_step.sechiba_first_step_coverage.closed_module_boundaries
    assert "hydrol_first_step_module_outputs" in traced_step.sechiba_first_step_coverage.closed_module_boundaries
    assert "hydrol_to_stomate_diagnostics" in traced_step.sechiba_first_step_coverage.closed_module_boundaries
    assert "hydrol_to_thermosoil_moisture" in traced_step.sechiba_first_step_coverage.closed_module_boundaries
    assert "diffuco_control_salinity" in traced_step.sechiba_first_step_coverage.closed_module_boundaries
    assert "diffuco_control_inundation" in traced_step.sechiba_first_step_coverage.closed_module_boundaries
    assert not traced_step.sechiba_first_step_coverage.ready_for_sechiba_explicit_step


def test_restart_backed_stomate_chain_bundles_are_constructible_only_at_do_slow_boundary():
    runtime = read_driver_runtime_scalars(USED_RUN_DEF)
    restart = reference_case_first_step_restart_state(CONFIG, root=ROOT)

    with np.testing.assert_raises_regex(ValueError, "do_slow boundary"):
        _paper_first_step_stomate_restart_input_bundles(
            config_path=CONFIG,
            run_def_path=USED_RUN_DEF,
            restart_state=restart,
            runtime=runtime,
            tstep=0,
        )

    source, bundles = _paper_first_step_stomate_restart_input_bundles(
        config_path=CONFIG,
        run_def_path=USED_RUN_DEF,
        restart_state=restart,
        runtime=runtime,
        tstep=47,
    )

    np.testing.assert_allclose(bundles.prescribe_inputs["cn_ind"], 0.0)
    assert bundles.prescribe_inputs.get("stomate_restart_none", False) is False
    assert bundles.maintenance_inputs["do_slow"] is True
    assert bundles.maintenance_inputs["dt_sechiba"] == 1800.0
    assert bundles.maintenance_inputs["dt_stomate"] == 86400.0
    np.testing.assert_allclose(bundles.maintenance_inputs["rprof"], source.boundary.kwargs["rprof"])
    np.testing.assert_allclose(bundles.prescribe_inputs["bm_sapl"], source.data.kwargs["bm_sapl"])
    np.testing.assert_array_equal(bundles.post_npp_inputs["nrec"], source.inactive_crop.kwargs["nrec"])
    np.testing.assert_allclose(bundles.post_npp_inputs["herbivores"], source.pre_step.annual.step.herbivores)

    daily_fields = {
        "gpp_daily": np.full_like(bundles.daily_process_inputs["gpp_daily"], 2.0),
        "t2m_daily": np.full_like(bundles.daily_process_inputs["t2m_daily"], 288.0),
        "t2m_min_daily": np.full_like(bundles.daily_process_inputs["t2m_min_daily"], 277.0),
        "snowfall_daily": np.full_like(bundles.daily_process_inputs["t2m_daily"], 0.1),
        "snowmass_daily": np.full_like(bundles.daily_process_inputs["t2m_daily"], 0.2),
        "tmc_topgrass_daily": np.full_like(bundles.daily_process_inputs["t2m_daily"], 0.3),
    }
    _, folded_bundles = _paper_first_step_stomate_restart_input_bundles(
        config_path=CONFIG,
        run_def_path=USED_RUN_DEF,
        restart_state=restart,
        runtime=runtime,
        tstep=47,
        daily_fields=daily_fields,
    )
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["gpp_daily"], daily_fields["gpp_daily"])
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["t2m_daily"], daily_fields["t2m_daily"])
    np.testing.assert_allclose(folded_bundles.post_npp_inputs["t2m_min_daily"], daily_fields["t2m_min_daily"])
    np.testing.assert_allclose(folded_bundles.daily_process_inputs["snowmass_daily"], daily_fields["snowmass_daily"])


def test_static_jit_daily_carbon_matches_explicit_paper_case_bundles():
    context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=USED_RUN_DEF)
    day = paper_1961_driver_day_scaffold(
        CONFIG,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        prepared_context=context,
        retain_stomate_step_results=False,
        single_pass_daily_fold=False,
    )
    bundles = day.stomate_restart_input_bundles
    assert bundles is not None
    nvm = int(np.asarray(bundles.prescribe_inputs["veget_max"]).shape[1])
    allocation_kwargs = {
        "f_fruit": parse_run_def_float(context.run_def_values, "F_FRUIT"),
        "ecureuil": parse_run_def_indexed_vector(context.run_def_values, "ECUREUIL", nvm),
        "alloc_sap_above_grass": parse_run_def_float(context.run_def_values, "ALLOC_SAP_ABOVE_GRASS"),
        "min_l_to_lsr": parse_run_def_float(context.run_def_values, "MIN_LTOLSR"),
        "max_l_to_lsr": parse_run_def_float(context.run_def_values, "MAX_LTOLSR"),
        "z_nitrogen": parse_run_def_float(context.run_def_values, "Z_NITROGEN"),
    }
    explicit = _paper_day_stomate_daily_carbon_from_bundles(
        bundles,
        run_def_values=context.run_def_values,
    )
    compiled = stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_static_jit(
        prescribe_inputs=bundles.prescribe_inputs,
        constraints_inputs=bundles.constraints_inputs,
        phenology_inputs=bundles.phenology_inputs,
        alloc_inputs=bundles.alloc_inputs,
        post_npp_inputs={
            **bundles.post_npp_inputs,
            **_paper_day_post_npp_owned_daily_fields(bundles),
        },
        allocation_kwargs=allocation_kwargs,
    )

    np.testing.assert_allclose(compiled.prescribe.biomass, explicit.prescribe.biomass, rtol=0, atol=0)
    np.testing.assert_allclose(compiled.constraints.regenerate, explicit.constraints.regenerate, rtol=0, atol=0)
    np.testing.assert_allclose(compiled.allocation.biomass, explicit.allocation.biomass, rtol=0, atol=0)
    np.testing.assert_allclose(
        compiled.post_npp.daily_carbon.npp_update.npp,
        explicit.post_npp.daily_carbon.npp_update.npp,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(compiled.post_npp.lai_after_setlai, explicit.post_npp.lai_after_setlai, rtol=0, atol=0)
    assert compiled.post_npp.daily_carbon.boundary.requires_trace == explicit.post_npp.daily_carbon.boundary.requires_trace


def test_first_step_stomate_boundary_refuses_later_step_restart_aliasing():
    scaffold = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=1,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    with np.testing.assert_raises_regex(ValueError, "only valid for tstep=0"):
        paper_1961_first_step_stomate_boundary(
            CONFIG,
            bundle=scaffold.step,
            payload=scaffold.payload,
            used_run_def_path=USED_RUN_DEF,
            root=ROOT,
        )


def test_year_scaffold_previews_ordered_steps_without_running_process_state():
    year = paper_1961_driver_year_scaffold(
        CONFIG,
        year=1961,
        preview_nsteps=2,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert year.year == 1961
    assert year.total_forcing_steps == 1460
    assert [step.step.tstep for step in year.preview_steps] == [0, 1]
    assert not year.ready_to_run_processes
    assert "diffuco_same_step_precall_state_and_gpp" in year.missing_components
    assert "stomate_restart_state_after_previous_step" in year.missing_components


def test_trace_free_first_day_scaffold_reaches_day_end_state_without_static_trace():
    day = paper_1961_driver_day_scaffold(
        CONFIG,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert day.year == 1961
    assert day.start_tstep == 0
    assert day.steps_per_stomate == 48
    assert len(day.do_slow_flags) == 48
    assert day.do_slow_flags[:-1] == (False,) * 47
    assert day.do_slow_flags[-1] is True
    assert [step.step.tstep for step in day.preview_steps] == [0]
    assert len(day.completed_entry_payloads) == day.steps_per_stomate
    assert day.completed_entry_payloads[0]["kjit"] == 1
    assert day.completed_entry_payloads[-1]["kjit"] == 48
    assert day.stopped_at_tstep is None
    assert day.state_gaps == ()
    assert day.daily_process_ready
    assert day.ready_for_day_boundary
    assert day.ready_for_first_daily_carbon
    assert day.ready_for_ok_leak
    assert day.ready_for_outputs
    assert day.ready_for_first_day_end_state
    assert day.daily_process_fold is not None
    assert day.daily_process_fold.ok
    assert day.daily_carbon_boundary is not None
    assert day.stomate_daily_carbon is not None
    assert day.ok_leak_boundary_inputs is not None
    assert day.ok_leak_boundary_gaps == ()
    assert day.stomate_ok_leak is not None
    assert day.stomate_outputs is not None
    assert day.stomate_outputs.requires_trace == ()
    assert day.daily_reset_fields is not None
    assert day.first_day_stomate_state is not None
    assert day.first_day_end_state is not None
    assert day.stomate_restart_input_bundles is not None
    assert day.stomate_daily_carbon.requires_trace == ()
    assert day.stomate_ok_leak.requires_trace == ()
    assert day.ok_leak_boundary_inputs.ok_leak_inputs["carbon_32l"].shape[-1] == 32
    assert day.stomate_outputs.modelout_fields["GPP"].shape == (1, 14)
    end = day.first_day_end_state
    assert len(end.fields_by_component["driver_previous_step_state"]) >= 13
    assert len(end.fields_by_component["sechiba_finalize_state"]) == 93
    restart = day.preview_steps[0].first_step_restart_state
    bundle = paper_restart_bundle_state_from_day_end_packet(
        end, base_stomate=restart.stomate_readstart, kjit=48
    )
    assert len(bundle.sechiba.fields) == 93


def test_first_day_scaffold_stops_at_previous_step_state_after_traced_first_entry():
    day = paper_1961_driver_day_scaffold(
        CONFIG,
        year=1961,
        fixed_format_trace_dir=TRACE_DIR,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert [step.step.tstep for step in day.preview_steps] == [0]
    assert len(day.completed_entry_payloads) == day.steps_per_stomate
    assert day.completed_entry_payloads[0]["kjit"] == 1
    assert day.completed_entry_payloads[-1]["kjit"] == 48
    assert "gpp" in day.completed_entry_payloads[0]
    assert day.previous_step_state is not None
    covered = day.previous_step_state.covered_fields
    assert "temp_sol" in covered["diffuco_previous_step_state"]
    assert "qsurf" in covered["diffuco_previous_step_state"]
    assert "temp_sol" in covered["enerbil_previous_step_state"]
    assert "soilcap" in covered["enerbil_previous_step_state"]
    assert "mc" in covered["hydrol_previous_step_state"]
    assert "mcl" in covered["hydrol_previous_step_state"]
    assert "ptn" in covered["thermosoil_previous_step_state"]
    assert "cgrnd" in covered["thermosoil_previous_step_state"]
    assert "daily_accumulators" in covered["slowproc_stomate_previous_step_state"]
    assert day.stopped_at_tstep is None
    assert day.daily_process_ready
    assert day.ready_for_day_boundary
    assert day.ready_for_first_daily_carbon
    assert day.ready_for_ok_leak
    assert day.ready_for_outputs
    assert day.ready_for_first_day_end_state
    assert day.daily_process_fold is not None
    assert day.daily_process_fold.ok
    assert day.daily_carbon_boundary is not None
    assert day.stomate_daily_carbon is not None
    assert day.ok_leak_boundary_inputs is not None
    assert day.ok_leak_boundary_gaps == ()
    assert day.stomate_ok_leak is not None
    assert day.stomate_outputs is not None
    assert day.stomate_outputs.requires_trace == ()
    assert day.daily_reset_fields is not None
    assert day.first_day_stomate_state is not None
    assert day.first_day_end_state is not None
    assert day.stomate_restart_input_bundles is not None
    assert day.state_gaps == ()
    np.testing.assert_allclose(
        np.asarray(day.daily_carbon_boundary.gpp_daily),
        np.asarray(day.stomate_restart_input_bundles.daily_process_inputs["gpp_daily"]),
    )
    np.testing.assert_allclose(
        np.asarray(day.daily_carbon_boundary.resp_maint_part),
        np.asarray(day.stomate_restart_input_bundles.daily_process_inputs["resp_maint_part"]),
    )
    np.testing.assert_allclose(
        np.asarray(day.daily_process_fold.daily_fields["snowfall_daily"]),
        np.asarray(day.stomate_restart_input_bundles.daily_process_inputs["snowfall_daily"]),
    )
    assert day.daily_carbon_boundary.f_alloc is None
    assert "f_alloc" in day.daily_carbon_boundary.requires_trace
    assert "resp_maint_part" not in day.daily_carbon_boundary.requires_trace
    np.testing.assert_allclose(
        np.asarray(day.stomate_daily_carbon.allocation.f_alloc.sum(axis=2)),
        1.0,
    )
    assert bool(np.asarray(day.stomate_daily_carbon.prescribe.pft_present)[0, 13])
    np.testing.assert_allclose(
        np.asarray(day.stomate_daily_carbon.post_npp.daily_carbon.boundary.gpp_daily),
        np.asarray(day.daily_carbon_boundary.gpp_daily),
    )
    np.testing.assert_allclose(
        np.asarray(day.stomate_daily_carbon.post_npp.daily_carbon.boundary.resp_maint_part),
        np.asarray(day.daily_carbon_boundary.resp_maint_part),
    )
    assert day.stomate_daily_carbon.requires_trace == ()
    assert day.stomate_ok_leak.requires_trace == ()
    assert day.ok_leak_boundary_inputs.ok_leak_inputs["carbon_32l"].shape[-1] == 32
    assert day.ok_leak_boundary_inputs.ok_leak_inputs["soil_mc_32l"].shape[1] == 32
    assert day.ok_leak_boundary_inputs.ok_leak_inputs["cmax_peat"].shape == (32,)
    assert day.stomate_ok_leak.ok_leak.soilcarbon.doc.shape[2] == 32
    assert day.stomate_outputs.output_diagnostics.tot_litter_soil_carb.shape == (1, 14, 32)
    assert day.stomate_outputs.output_diagnostics.carbon_32l_conct.shape == (1, 3, 32)
    assert day.stomate_outputs.modelout_fields["LEAF_M"].shape == (1, 14)
    assert day.stomate_outputs.modelout.AGB_model.shape == (1, 14)
    np.testing.assert_allclose(
        np.asarray(day.stomate_outputs.modelout_fields["GPP"]),
        np.asarray(day.daily_process_fold.daily_fields["gpp_daily"]),
    )
    for name in ("gpp_daily", "resp_maint_part", "precip_daily", "snowmass_daily", "fwet_daily", "liqwt_daily"):
        np.testing.assert_allclose(np.asarray(day.daily_reset_fields[name]), 0.0)
    np.testing.assert_allclose(np.asarray(day.daily_reset_fields["t2m_min_daily"]), 1.0e33)
    np.testing.assert_allclose(np.asarray(day.daily_reset_fields["t2m_max_daily"]), -1.0e33)
    state = day.first_day_stomate_state
    assert state.day_index == 1
    for name in (
        "biomass",
        "litter_above",
        "litterpart",
        "dead_leaves",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "carbon_32l",
        "DOC",
        "carb_mass_total",
        "daily_accumulators",
        "lai",
        "veget",
        "veget_max",
        "height",
        "frac_age",
        "soiltile",
        "tot_bare_soil",
    ):
        assert name in state.covered_fields
    final_lai = _paper_day_final_lai_from_daily_carbon(day.stomate_daily_carbon)
    expected_surface = slowproc_surface_update_explicit(
        lai=final_lai,
        frac_nobio=day.preview_steps[0].first_step_restart_state.slowproc.frac_nobio,
        veget_max=day.preview_steps[0].payload.veget_max,
        pref_soil_veg=day.preview_steps[0].step.run_scalars.pref_soil_veg,
        ext_coeff_vegetfrac=prepare_paper_1961_driver_context(
            CONFIG,
            used_run_def_path=USED_RUN_DEF,
        ).ext_coeff_vegetfrac,
        nstm=day.preview_steps[0].step.run_scalars.nstm,
        ok_dgvm=False,
    )
    np.testing.assert_allclose(
        np.asarray(state.fields["biomass"]),
        np.asarray(day.stomate_daily_carbon.post_npp.turnover.biomass),
    )
    np.testing.assert_allclose(
        np.asarray(state.fields["carbon_32l"]),
        np.asarray(day.stomate_ok_leak.ok_leak.soilcarbon.carbon_32l),
    )
    np.testing.assert_allclose(
        np.asarray(state.fields["lignin_struc_above"]),
        np.asarray(day.stomate_ok_leak.ok_leak.littercalc.lignin_struc_above),
    )
    np.testing.assert_allclose(
        np.asarray(state.fields["lignin_struc_below"]),
        np.asarray(day.stomate_ok_leak.ok_leak.littercalc.lignin_struc_below),
    )
    np.testing.assert_allclose(
        np.asarray(state.fields["altmax"]),
        np.asarray(day.ok_leak_boundary_inputs.output_inputs["altmax"]),
    )
    np.testing.assert_allclose(
        np.asarray(state.fields["carb_mass_total"]),
        np.asarray(day.stomate_outputs.output_diagnostics.carb_mass_total),
    )
    np.testing.assert_allclose(
        np.asarray(state.fields["daily_accumulators"]["gpp_daily"]),
        0.0,
    )
    np.testing.assert_allclose(np.asarray(state.fields["lai"]), np.asarray(final_lai))
    np.testing.assert_allclose(np.asarray(state.fields["veget"]), np.asarray(expected_surface.vegetation.veget))
    np.testing.assert_allclose(np.asarray(state.fields["veget_max"]), np.asarray(expected_surface.vegetation.veget_max))
    final_crown = (
        day.stomate_daily_carbon.post_npp.crown_after_establish
        if day.stomate_daily_carbon.post_npp.crown_after_establish is not None
        else day.stomate_daily_carbon.post_npp.crown_after_npp
    )
    np.testing.assert_array_equal(
        np.asarray(state.fields["height"]), np.asarray(final_crown.height)
    )
    final_leaf_frac = (
        day.stomate_daily_carbon.post_npp.vmax.leaf_frac
        if day.stomate_daily_carbon.post_npp.vmax is not None
        else day.stomate_daily_carbon.post_npp.turnover.leaf_frac
    )
    np.testing.assert_array_equal(
        np.asarray(state.fields["frac_age"]), np.asarray(final_leaf_frac)
    )
    np.testing.assert_allclose(np.asarray(state.fields["soiltile"]), np.asarray(expected_surface.vegetation.soiltile))
    np.testing.assert_allclose(np.asarray(state.fields["tot_bare_soil"]), np.asarray(expected_surface.tot_bare_soil))
    assert "stomate_lpj.f90::StomateLpj" in state.provenance_by_field["biomass"][0]
    assert "stomate_soilcarbon.f90::soilcarbon_leak" in state.provenance_by_field["carbon_32l"][2]
    assert "slowproc.f90::slowproc_main" in state.provenance_by_field["tot_bare_soil"][1]
    end_state = day.first_day_end_state
    end_slowproc = end_state.fields_by_component["slowproc_stomate_previous_step_state"]
    end_diffuco = end_state.fields_by_component["diffuco_previous_step_state"]
    np.testing.assert_allclose(np.asarray(end_slowproc["biomass"]), np.asarray(state.fields["biomass"]))
    np.testing.assert_allclose(np.asarray(end_slowproc["carbon_32l"]), np.asarray(state.fields["carbon_32l"]))
    np.testing.assert_allclose(np.asarray(end_slowproc["DOC"]), np.asarray(state.fields["DOC"]))
    np.testing.assert_allclose(np.asarray(end_slowproc["lignin_struc_above"]), np.asarray(state.fields["lignin_struc_above"]))
    np.testing.assert_allclose(np.asarray(end_slowproc["lignin_struc_below"]), np.asarray(state.fields["lignin_struc_below"]))
    np.testing.assert_allclose(np.asarray(end_slowproc["altmax"]), np.asarray(state.fields["altmax"]))
    np.testing.assert_array_equal(
        np.asarray(end_slowproc["height"]), np.asarray(state.fields["height"])
    )
    np.testing.assert_array_equal(
        np.asarray(end_slowproc["frac_age"]), np.asarray(state.fields["frac_age"])
    )
    assert "assim_param" in end_slowproc
    assert day.stomate_daily_carbon.post_npp.vmax is not None
    np.testing.assert_allclose(
        np.asarray(end_slowproc["assim_param"]),
        np.asarray(day.stomate_daily_carbon.post_npp.vmax.vcmax)[:, :, None],
    )
    assert "temp_growth" in end_slowproc
    np.testing.assert_allclose(
        np.asarray(end_slowproc["temp_growth"]),
        np.asarray(state.fields["t2m_month"]) - 273.15,
    )
    for name in ("fpeat", "soilc_total", "thawed_humidity", "depth_organic_soil"):
        assert name in end_slowproc
        np.testing.assert_allclose(
            np.asarray(end_slowproc[name]),
            np.asarray(day.previous_step_state.fields_by_component["slowproc_stomate_previous_step_state"][name]),
        )
    np.testing.assert_allclose(np.asarray(end_slowproc["daily_accumulators"]["gpp_daily"]), 0.0)
    np.testing.assert_allclose(np.asarray(end_slowproc["lai"]), np.asarray(state.fields["lai"]))
    np.testing.assert_allclose(np.asarray(end_slowproc["veget"]), np.asarray(state.fields["veget"]))
    np.testing.assert_allclose(np.asarray(end_slowproc["tot_bare_soil"]), np.asarray(state.fields["tot_bare_soil"]))
    for name in ("t2m_longterm", "t2m_month", "moiavail_week", "tsoil_month", "soilhum_month", "gdd_from_growthinit"):
        assert name in end_slowproc
        np.testing.assert_allclose(np.asarray(end_slowproc[name]), np.asarray(state.fields[name]))
    for name in (*STOMATE_DAY_SEASON_ANNUAL_FIELDS, *STOMATE_DAY_SEASON_BIOMETEOROLOGY_FIELDS):
        assert name in end_slowproc
        np.testing.assert_allclose(np.asarray(end_slowproc[name]), np.asarray(state.fields[name]))
    for name in STOMATE_DAY_SEASON_PERSISTED_FIELDS:
        assert name in end_slowproc
        np.testing.assert_allclose(np.asarray(end_slowproc[name]), np.asarray(state.fields[name]))
    np.testing.assert_allclose(np.asarray(end_diffuco["veget"]), np.asarray(state.fields["veget"]))
    np.testing.assert_allclose(np.asarray(end_diffuco["tot_bare_soil"]), np.asarray(state.fields["tot_bare_soil"]))
    assert "temp_sol" in end_diffuco
    assert "qsurf" in end_diffuco
    assert any(
        "OK_LEAK state update" in item
        for item in end_state.provenance_by_component["slowproc_stomate_previous_step_state"]
    )
    assert "diffuco_previous_step_state" not in day.missing_components
    assert "hydrol_previous_step_state" not in day.missing_components
    assert "slowproc_stomate_previous_step_state" not in day.missing_components


def test_previous_step_state_packet_extracts_only_source_backed_timestep_outputs():
    static_truth = read_static_fields_from_fixed_format_trace_dir(TRACE_DIR)
    scaffold = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    state = driver_previous_step_state_from_timestep_scaffold(scaffold)
    slowproc = state.fields_by_component["slowproc_stomate_previous_step_state"]
    assert set(STOMATE_FIXED_PATH_CARRY_FIELDS) <= set(slowproc)
    np.testing.assert_array_equal(
        slowproc["MatrixV"],
        scaffold.first_step_restart_state.stomate_readstart.remainder_state.MatrixV,
    )

    assert state.tstep == 0
    np.testing.assert_allclose(
        state.fields_by_component["diffuco_previous_step_state"]["temp_sol"],
        scaffold.enerbil_first_step_local.payload["temp_sol_new"],
    )
    np.testing.assert_allclose(
        state.fields_by_component["enerbil_previous_step_state"]["soilcap"],
        scaffold.thermosoil_first_step_module.downstream_payload["soilcap"],
    )
    np.testing.assert_allclose(
        state.fields_by_component["hydrol_previous_step_state"]["mc"],
        scaffold.hydrol_first_step_module.module.soil.mc,
    )
    np.testing.assert_allclose(
        state.fields_by_component["hydrol_previous_step_state"]["nroot"],
        scaffold.hydrol_first_step_module.diagnostics.nroot,
    )
    np.testing.assert_allclose(
        state.fields_by_component["thermosoil_previous_step_state"]["ptn"],
        scaffold.thermosoil_first_step_module.module.profile.ptn,
    )
    np.testing.assert_allclose(
        state.fields_by_component["thermosoil_previous_step_state"]["pcapa_en"],
        scaffold.thermosoil_first_step_module.module.getdiff.pcapa_en,
    )
    np.testing.assert_allclose(
        state.fields_by_component["thermosoil_previous_step_state"]["temp_sol_beg"],
        scaffold.thermosoil_first_step_module.module.energy.temp_sol_beg,
    )
    np.testing.assert_allclose(
        state.fields_by_component["slowproc_stomate_previous_step_state"]["altmax"],
        scaffold.first_step_restart_state.stomate.altmax,
    )
    assert "carbon" in state.fields_by_component["slowproc_stomate_previous_step_state"]
    np.testing.assert_allclose(
        state.fields_by_component["slowproc_stomate_previous_step_state"]["carbon"],
        scaffold.first_step_restart_state.stomate.carbon,
    )
    assert any("sechiba_end lines 3114-3140" in item for item in state.provenance_by_component["diffuco_previous_step_state"])


def test_previous_step_state_feeds_diagnosed_hydrol_evap_bare_lim_to_next_diffuco():
    npts, nstm, nslm, nvm = 1, 2, 3, 1
    hydrol_module = SimpleNamespace(
        ok=True,
        module=SimpleNamespace(
            soil=SimpleNamespace(
                mc=np.full((npts, nslm, nstm), 0.31, dtype=np.float64),
                mcl=np.full((npts, nslm, nstm), 0.30, dtype=np.float64),
                water2infilt=np.zeros((npts, nstm), dtype=np.float64),
                run2peat=np.asarray([0.0], dtype=np.float64),
                run2man=np.asarray([0.0], dtype=np.float64),
                wt_ab=np.asarray([0.0], dtype=np.float64),
                wt_ab_tide=np.asarray([0.0], dtype=np.float64),
            ),
            split=SimpleNamespace(ae_ns=np.zeros((npts, nstm), dtype=np.float64)),
            canop=SimpleNamespace(qsintveg=np.zeros((npts, nvm), dtype=np.float64)),
            flood=SimpleNamespace(flood_res=np.asarray([0.0], dtype=np.float64)),
        ),
        diagnostics=SimpleNamespace(
            humrel=np.asarray([[0.8]], dtype=np.float64),
            humrelv=np.full((npts, nvm, nstm), 0.8, dtype=np.float64),
            us=np.full((npts, nvm, nstm, nslm), 0.2, dtype=np.float64),
            evap_bare_lim=np.asarray([0.73], dtype=np.float64),
            evap_bare_lim_ns=np.asarray([[0.50, 0.23]], dtype=np.float64),
            fwet_new=np.asarray([0.0], dtype=np.float64),
            nroot=np.full((npts, nvm, nslm), 1.0 / nslm, dtype=np.float64),
            mc_layh_s=np.full((npts, nslm, nstm), 0.31, dtype=np.float64),
            drysoil_frac=np.asarray([0.2], dtype=np.float64),
            shumdiag_peat=None,
        ),
        outputs=SimpleNamespace(
            drainage_per_soil=np.zeros((npts, nstm), dtype=np.float64),
            runoff_per_soil=np.zeros((npts, nstm), dtype=np.float64),
            wat_flux=np.zeros((npts, nslm, nstm), dtype=np.float64),
            runoff2peat=np.zeros((npts, nstm), dtype=np.float64),
            canopy2ground=np.asarray([0.0], dtype=np.float64),
            precip2ground=np.asarray([0.0], dtype=np.float64),
            precip2canopy=np.asarray([0.0], dtype=np.float64),
        ),
        snow_state=None,
        module_inputs={},
    )
    scaffold = SimpleNamespace(
        step=SimpleNamespace(tstep=0),
        enerbil_first_step_local=None,
        diffuco_first_step_local_enerbil_precall=None,
        diffuco_control_salinity=None,
        diffuco_control_assembly=None,
        diffuco_first_step_precall=None,
        thermosoil_first_step_module=None,
        condveg_first_step_module=None,
        hydrol_first_step_module=hydrol_module,
        first_step_restart_state=None,
    )

    state = driver_previous_step_state_from_timestep_scaffold(scaffold)
    hydrol_state = state.fields_by_component["hydrol_previous_step_state"]
    diffuco_state = state.fields_by_component["diffuco_previous_step_state"]

    np.testing.assert_allclose(diffuco_state["evap_bare_lim"], [0.73])
    assert "evap_bare_lim_ns" not in diffuco_state
    np.testing.assert_allclose(hydrol_state["evap_bare_lim"], [0.73])
    np.testing.assert_allclose(hydrol_state["evap_bare_lim_ns"], [[0.50, 0.23]])


def test_next_step_diffuco_precall_uses_previous_state_without_restart_aliasing():
    static_truth = read_static_fields_from_fixed_format_trace_dir(TRACE_DIR)
    first = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    previous = driver_previous_step_state_from_timestep_scaffold(first)

    next_diffuco = paper_1961_next_step_diffuco_precall_scaffold(
        CONFIG,
        year=1961,
        tstep=1,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        previous_state=previous,
    )

    assert next_diffuco.tstep == 1
    assert next_diffuco.previous_state.tstep == 0
    assert next_diffuco.payload.temp_air.shape == (1,)
    assert "temp_sol" in next_diffuco.used_previous_state_fields
    assert "qsurf" in next_diffuco.used_previous_state_fields
    assert "humrel" in next_diffuco.used_previous_state_fields
    assert "qsintveg" in next_diffuco.used_previous_state_fields
    assert "z0m" in next_diffuco.used_previous_state_fields
    assert "evapot" in next_diffuco.used_previous_state_fields
    assert "evapot_corr" in next_diffuco.used_previous_state_fields
    assert "frac_nobio" in next_diffuco.used_previous_state_fields
    assert "snow" not in next_diffuco.missing_previous_state_fields
    assert "snowdz" not in next_diffuco.missing_previous_state_fields
    assert "snowrho" not in next_diffuco.missing_previous_state_fields
    assert "snowtemp" not in next_diffuco.missing_previous_state_fields
    assert "snow_age" not in next_diffuco.missing_previous_state_fields
    assert "snow_nobio" not in next_diffuco.missing_previous_state_fields
    assert "snow_nobio_age" not in next_diffuco.missing_previous_state_fields
    assert "rstruct" not in next_diffuco.missing_previous_state_fields
    assert "temp_sol" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "qsurf" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "humrel" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "z0m" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "rstruct" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "qsintmax" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "assim_param" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "temp_growth" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "frac_nobio" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "totfrac_nobio" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "evapot" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "evapot_corr" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "flood_frac" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "flood_res" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "control_salinity" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "control_inudate" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "control_salinity" in next_diffuco.static_passthrough_fields
    assert "control_inudate" in next_diffuco.static_passthrough_fields
    np.testing.assert_array_equal(
        next_diffuco.diffuco_precall.payload["control_salinity"],
        previous.fields_by_component["diffuco_previous_step_state"]["control_salinity"],
    )
    np.testing.assert_array_equal(
        next_diffuco.diffuco_precall.payload["control_inudate"],
        previous.fields_by_component["diffuco_previous_step_state"]["control_inudate"],
    )
    assert "snow" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "frac_snow_veg" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert "frac_snow_nobio" not in next_diffuco.diffuco_precall.missing_local_process_inputs
    assert next_diffuco.ready_for_local_diffuco


def test_next_step_enerbil_precall_consumes_previous_surface_and_thermal_state():
    static_truth = read_static_fields_from_fixed_format_trace_dir(TRACE_DIR)
    first = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    previous = driver_previous_step_state_from_timestep_scaffold(first)

    next_enerbil = paper_1961_next_step_enerbil_precall_scaffold(
        CONFIG,
        year=1961,
        tstep=1,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        previous_state=previous,
    )

    assert next_enerbil.tstep == 1
    assert "temp_sol" in next_enerbil.used_previous_state_fields
    assert "soilcap" in next_enerbil.used_previous_state_fields
    assert "soilflx_pft" in next_enerbil.used_previous_state_fields
    assert "temp_sol" not in next_enerbil.enerbil_precall.missing_inputs
    assert "temp_sol_pft" not in next_enerbil.enerbil_precall.missing_inputs
    assert "soilcap" not in next_enerbil.enerbil_precall.missing_inputs
    assert "soilflx" not in next_enerbil.enerbil_precall.missing_inputs
    assert "qsurf" not in next_enerbil.enerbil_precall.missing_inputs
    assert next_enerbil.diffuco_local_enerbil_precall is not None
    assert next_enerbil.diffuco_local_enerbil_precall.result is not None
    assert "q_cdrag" not in next_enerbil.enerbil_precall.missing_inputs
    assert "vbeta" not in next_enerbil.enerbil_precall.missing_inputs
    assert "snowdz" not in next_enerbil.enerbil_precall.missing_inputs
    assert next_enerbil.ready_for_local_enerbil
    assert next_enerbil.enerbil_local is not None
    assert np.asarray(next_enerbil.enerbil_local.payload["temp_sol_new"]).shape == (1,)
    assert np.asarray(next_enerbil.enerbil_local.payload["qsurf"]).shape == (1,)


def test_next_step_hydrol_precall_and_module_consume_previous_state_without_restart_aliasing():
    static_truth = read_static_fields_from_fixed_format_trace_dir(TRACE_DIR)
    first = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    previous = driver_previous_step_state_from_timestep_scaffold(first)

    next_hydrol = paper_1961_next_step_hydrol_precall_scaffold(
        CONFIG,
        year=1961,
        tstep=1,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        previous_state=previous,
    )

    assert next_hydrol.tstep == 1
    assert next_hydrol.missing_previous_state_fields == ()
    assert next_hydrol.hydrol_precall is not None
    assert next_hydrol.hydrol_precall.ok
    assert "profil_froz" not in next_hydrol.hydrol_precall.missing_inputs
    assert np.asarray(next_hydrol.hydrol_precall.payload["profil_froz"]).shape == (1, 11, 6)
    assert next_hydrol.ready_for_local_hydrol
    assert next_hydrol.hydrol_module is not None
    assert next_hydrol.hydrol_module.ok
    assert next_hydrol.hydrol_module.missing_inputs == ()
    assert next_hydrol.ready_for_next_state
    assert np.asarray(next_hydrol.hydrol_module.outputs.wat_flux).shape == (1, 11, 6)
    assert np.asarray(next_hydrol.hydrol_module.diagnostics.humrel).shape == (1, 14)
    np.testing.assert_allclose(np.asarray(next_hydrol.hydrol_module.snow_state.snow), [0.0])
    assert next_hydrol.condveg_module is not None
    assert next_hydrol.condveg_module.ok
    assert "roughheight" in next_hydrol.condveg_module.covered_fields
    assert next_hydrol.thermosoil_module is not None
    assert next_hydrol.thermosoil_module.ok
    assert next_hydrol.thermosoil_module.missing_inputs == ()
    assert "soilcap" in next_hydrol.thermosoil_module.downstream_payload
    assert "cgrnd_snow" in next_hydrol.thermosoil_module.downstream_payload

    second_previous = driver_previous_step_state_from_next_step_hydrol_scaffold(next_hydrol)
    assert second_previous.tstep == 1
    assert "mc" in second_previous.fields_by_component["hydrol_previous_step_state"]
    assert "ptn" in second_previous.fields_by_component["thermosoil_previous_step_state"]
    assert "soilcap" in second_previous.fields_by_component["enerbil_previous_step_state"]
    assert "z0m" in second_previous.fields_by_component["diffuco_previous_step_state"]
    assert "altmax" in second_previous.fields_by_component["slowproc_stomate_previous_step_state"]

    tstep2 = paper_1961_next_step_hydrol_precall_scaffold(
        CONFIG,
        year=1961,
        tstep=2,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        previous_state=second_previous,
    )
    assert tstep2.missing_previous_state_fields == ()
    assert tstep2.ready_for_next_state
    assert tstep2.hydrol_module is not None
    assert tstep2.hydrol_module.ok
    assert tstep2.thermosoil_module is not None
    assert tstep2.thermosoil_module.ok


def test_first_day_end_state_drives_second_day_first_sechiba_step_trace_free():
    day = paper_1961_driver_day_scaffold(
        CONFIG,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    previous = day.first_day_end_state

    second_day_first = paper_1961_next_step_hydrol_precall_scaffold(
        CONFIG,
        year=1961,
        tstep=48,
        used_run_def_path=USED_RUN_DEF,
        previous_state=previous,
    )

    slowproc_state = previous.fields_by_component["slowproc_stomate_previous_step_state"]
    diffuco_state = previous.fields_by_component["diffuco_previous_step_state"]
    assert second_day_first.missing_previous_state_fields == ()
    assert second_day_first.enerbil.diffuco_precall.ok
    assert second_day_first.enerbil.enerbil_precall.ok
    assert second_day_first.ready_for_next_state
    np.testing.assert_allclose(
        np.asarray(second_day_first.enerbil.diffuco_precall.payload["veget"]),
        np.asarray(slowproc_state["veget"]),
    )
    np.testing.assert_allclose(
        np.asarray(second_day_first.enerbil.diffuco_precall.payload["tot_bare_soil"]),
        np.asarray(slowproc_state["tot_bare_soil"]),
    )
    np.testing.assert_allclose(np.asarray(diffuco_state["veget"]), np.asarray(slowproc_state["veget"]))
    np.testing.assert_allclose(np.asarray(diffuco_state["tot_bare_soil"]), np.asarray(slowproc_state["tot_bare_soil"]))
    assert second_day_first.hydrol_module is not None
    assert second_day_first.condveg_module is not None
    assert second_day_first.thermosoil_module is not None


def test_first_day_end_state_drives_second_day_daily_sechiba_fold_trace_free():
    day1 = paper_1961_driver_day_scaffold(
        CONFIG,
        year=1961,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    day2 = paper_1961_driver_later_day_scaffold(
        CONFIG,
        year=1961,
        start_tstep=48,
        used_run_def_path=USED_RUN_DEF,
        previous_state=day1.first_day_end_state,
    )

    assert day2.ready_for_day_boundary
    assert len(day2.completed_entry_payloads) == day2.steps_per_stomate
    assert day2.completed_entry_payloads[0]["kjit"] == 49
    assert day2.completed_entry_payloads[-1]["kjit"] == 96
    assert day2.stopped_at_tstep is None
    assert day2.missing_components == ()
    assert day2.daily_process_fold is not None
    assert day2.daily_process_fold.ok
    assert day2.stomate_restart_input_bundles is not None
    assert day2.stomate_bundle_gaps == ()
    assert day2.stomate_daily_carbon is not None
    assert day2.stomate_daily_carbon.requires_trace == ()
    assert day2.ok_leak_boundary_inputs is not None
    assert day2.ok_leak_boundary_gaps == ()
    assert day2.stomate_ok_leak is not None
    assert day2.stomate_ok_leak.requires_trace == ()
    assert day2.stomate_outputs is not None
    assert day2.stomate_outputs.requires_trace == ()
    assert day2.daily_reset_fields is not None
    assert day2.day_stomate_state is not None
    assert day2.day_end_state is not None
    assert day2.previous_step_state.tstep == 95
    slowproc_state = day2.previous_step_state.fields_by_component["slowproc_stomate_previous_step_state"]
    assert "t2m_longterm" in slowproc_state
    for name in (
        "pft_present",
        "leaf_frac",
        "leaf_age",
        "turnover_daily",
        "bm_to_litter",
        "senescence",
        "tot_bare_soil",
        "daily_accumulators",
    ):
        assert name in slowproc_state
    assert np.asarray(day2.daily_process_fold.daily_fields["gpp_daily"]).shape == (1, 14)
    assert np.asarray(day2.daily_process_fold.daily_fields["resp_maint_part"]).shape[0:2] == (1, 14)
    assert np.isfinite(np.asarray(day2.daily_process_fold.daily_fields["t2m_min_daily"])).all()
    np.testing.assert_allclose(
        np.asarray(day2.stomate_daily_carbon.post_npp.daily_carbon.boundary.gpp_daily),
        np.asarray(day2.daily_process_fold.daily_fields["gpp_daily"]),
    )
    for name in (
        *STOMATE_DAY_SEASON_MEMORY_FIELDS,
        *STOMATE_DAY_SEASON_ANNUAL_FIELDS,
        *STOMATE_DAY_SEASON_BIOMETEOROLOGY_FIELDS,
        *STOMATE_DAY_SEASON_PERSISTED_FIELDS,
    ):
        np.testing.assert_allclose(
            np.asarray(day2.day_stomate_state.fields[name]),
            np.asarray(day2.day_end_state.fields_by_component["slowproc_stomate_previous_step_state"][name]),
        )
    np.testing.assert_allclose(
        np.asarray(day2.stomate_restart_input_bundles.alloc_inputs["moiavail_week"]),
        np.asarray(day2.day_stomate_state.fields["moiavail_week"]),
    )
    np.testing.assert_allclose(
        np.asarray(day2.stomate_restart_input_bundles.post_npp_inputs["gdd_from_growthinit"]),
        np.asarray(day2.day_stomate_state.fields["gdd_from_growthinit"]),
    )
    assert np.asarray(day2.stomate_restart_input_bundles.post_npp_inputs["gdd_from_growthinit"]).shape == np.asarray(
        day1.first_day_end_state.fields_by_component["slowproc_stomate_previous_step_state"]["gdd_from_growthinit"]
    ).shape
    np.testing.assert_allclose(
        np.asarray(day2.day_stomate_state.fields["gdd_init_date"]),
        np.asarray(
            season_update_gdd_init_date(
                day1.first_day_end_state.fields_by_component["slowproc_stomate_previous_step_state"]["gdd_init_date"],
                latitude=day2.completed_entry_payloads[0]["lalo"][:, 0],
                julian_diff=2.0,
            )
        ),
    )
    assert day2.ok_leak_boundary_inputs.ok_leak_inputs["carbon_32l"].shape[-1] == 32
    assert day2.stomate_ok_leak.ok_leak.soilcarbon.doc.shape[2] == 32
    assert day2.stomate_outputs.modelout_fields["GPP"].shape == (1, 14)
    np.testing.assert_allclose(
        np.asarray(day2.stomate_outputs.modelout_fields["GPP"]),
        np.asarray(day2.daily_process_fold.daily_fields["gpp_daily"]),
    )
    for name in ("gpp_daily", "resp_maint_part", "precip_daily", "snowmass_daily", "fwet_daily", "liqwt_daily"):
        np.testing.assert_allclose(np.asarray(day2.daily_reset_fields[name]), 0.0)
    np.testing.assert_allclose(np.asarray(day2.daily_reset_fields["t2m_min_daily"]), 1.0e33)
    np.testing.assert_allclose(np.asarray(day2.daily_reset_fields["t2m_max_daily"]), -1.0e33)
    for name in (
        "biomass",
        "litter_above",
        "carbon_32l",
        "DOC",
        "carb_mass_total",
        "daily_accumulators",
        "lai",
        "veget",
        "veget_max",
        "height",
        "frac_age",
        "soiltile",
        "tot_bare_soil",
    ):
        assert name in day2.day_stomate_state.covered_fields
    end_slowproc = day2.day_end_state.fields_by_component["slowproc_stomate_previous_step_state"]
    final_day2_crown = (
        day2.stomate_daily_carbon.post_npp.crown_after_establish
        if day2.stomate_daily_carbon.post_npp.crown_after_establish is not None
        else day2.stomate_daily_carbon.post_npp.crown_after_npp
    )
    np.testing.assert_array_equal(
        np.asarray(end_slowproc["height"]), np.asarray(final_day2_crown.height)
    )
    final_day2_leaf_frac = (
        day2.stomate_daily_carbon.post_npp.vmax.leaf_frac
        if day2.stomate_daily_carbon.post_npp.vmax is not None
        else day2.stomate_daily_carbon.post_npp.turnover.leaf_frac
    )
    np.testing.assert_array_equal(
        np.asarray(end_slowproc["frac_age"]), np.asarray(final_day2_leaf_frac)
    )
    np.testing.assert_allclose(
        np.asarray(end_slowproc["carbon_32l"]),
        np.asarray(day2.stomate_ok_leak.ok_leak.soilcarbon.carbon_32l),
    )
    np.testing.assert_allclose(
        np.asarray(end_slowproc["DOC"]),
        np.asarray(day2.stomate_ok_leak.ok_leak.soilcarbon.doc),
    )
    np.testing.assert_allclose(
        np.asarray(end_slowproc["litterpart"]),
        np.asarray(day2.stomate_ok_leak.ok_leak.littercalc.litterpart),
    )
    np.testing.assert_allclose(
        np.asarray(end_slowproc["dead_leaves"]),
        np.asarray(day2.stomate_ok_leak.ok_leak.littercalc.dead_leaves),
    )
    for name in ("fuel_1hr", "fuel_10hr", "fuel_100hr", "fuel_1000hr"):
        np.testing.assert_allclose(
            np.asarray(end_slowproc[name]),
            np.asarray(getattr(day2.stomate_ok_leak.ok_leak.littercalc.fuel, name)),
        )
    np.testing.assert_allclose(
        np.asarray(end_slowproc["lai"]),
        np.asarray(day2.day_stomate_state.fields["lai"]),
    )
    np.testing.assert_allclose(
        np.asarray(end_slowproc["veget"]),
        np.asarray(day2.day_stomate_state.fields["veget"]),
    )


def test_multiday_scaffold_reuses_model_state_through_second_day():
    multi = paper_1961_driver_multiday_scaffold(
        CONFIG,
        year=1961,
        ndays=2,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert multi.ready_for_requested_days
    assert multi.initial_state_mode == "restart_backed_reference_case"
    assert multi.stopped_day_index is None
    assert multi.missing_components == ()
    assert len(multi.days) == 2
    assert multi.steps_per_stomate == 48
    day1, day2 = multi.days
    assert day1.ready_for_first_day_end_state
    assert day2.ready_for_day_end_state
    assert day2.start_tstep == 48
    assert day2.completed_entry_payloads[0]["kjit"] == 49
    assert day2.completed_entry_payloads[-1]["kjit"] == 96
    assert multi.last_day_end_state is day2.day_end_state
    assert day2.input_previous_state is day1.first_day_end_state
    assert day2.day_end_state.tstep == 95
    assert day2.stomate_ok_leak.requires_trace == ()
    assert day2.stomate_outputs.requires_trace == ()
    end_slowproc = day2.day_end_state.fields_by_component["slowproc_stomate_previous_step_state"]
    np.testing.assert_allclose(
        np.asarray(end_slowproc["carbon_32l"]),
        np.asarray(day2.stomate_ok_leak.ok_leak.soilcarbon.carbon_32l),
    )
    np.testing.assert_allclose(
        np.asarray(end_slowproc["dead_leaves"]),
        np.asarray(day2.stomate_ok_leak.ok_leak.littercalc.dead_leaves),
    )
    np.testing.assert_allclose(np.asarray(end_slowproc["daily_accumulators"]["gpp_daily"]), 0.0)


def test_multiday_scaffold_reuses_model_state_through_third_day():
    multi = paper_1961_driver_multiday_scaffold(
        CONFIG,
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert multi.ready_for_requested_days
    assert multi.stopped_day_index is None
    assert multi.missing_components == ()
    assert len(multi.days) == 3
    day3 = multi.days[-1]
    assert day3.ready_for_day_end_state
    assert day3.start_tstep == 96
    assert day3.completed_entry_payloads[0]["kjit"] == 97
    assert day3.completed_entry_payloads[-1]["kjit"] == 144
    assert multi.last_day_end_state is day3.day_end_state
    assert day3.input_previous_state is multi.days[1].day_end_state
    assert day3.day_end_state.tstep == 143
    assert day3.stomate_ok_leak.requires_trace == ()
    assert day3.stomate_outputs.requires_trace == ()


def test_multiday_modelout_run_collects_closed_daily_outputs_through_third_day():
    run = paper_1961_driver_multiday_modelout_run(
        CONFIG,
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert run.ready_for_requested_days
    assert run.missing_components == ()
    assert len(run.daily_modelout) == 3
    assert run.last_day_end_state is run.scaffold.days[-1].day_end_state
    for index, daily in enumerate(run.daily_modelout, start=1):
        assert daily.day_index == index
        assert daily.start_tstep == (index - 1) * run.scaffold.steps_per_stomate
        assert daily.end_tstep == index * run.scaffold.steps_per_stomate - 1
        assert daily.modelout_fields["LEAF_M"].shape == (1, 14)
        assert daily.modelout.AGB_model.shape == (1, 14)
        assert daily.modelout.NPP_model.shape == (1, 14)


def test_multiday_modelout_lite_run_matches_scaffold_outputs_through_third_day():
    scaffold_run = paper_1961_driver_multiday_modelout_run(
        CONFIG,
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    lite_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    compact_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
    )
    static_jit_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
    )

    for run in (lite_run, compact_run, static_jit_run):
        assert run.ready_for_requested_days
        assert run.missing_components == scaffold_run.missing_components
        assert run.last_day_end_state.tstep == scaffold_run.last_day_end_state.tstep
        assert len(run.daily_modelout) == len(scaffold_run.daily_modelout)
        for lite, scaffold in zip(run.daily_modelout, scaffold_run.daily_modelout, strict=True):
            assert lite.day_index == scaffold.day_index
            assert lite.start_tstep == scaffold.start_tstep
            assert lite.end_tstep == scaffold.end_tstep
            np.testing.assert_allclose(lite.modelout_fields["GPP"], scaffold.modelout_fields["GPP"], rtol=0, atol=0)
            np.testing.assert_allclose(lite.modelout_fields["LEAF_M"], scaffold.modelout_fields["LEAF_M"], rtol=0, atol=0)
            np.testing.assert_allclose(lite.modelout.NPP_model, scaffold.modelout.NPP_model, rtol=0, atol=0)
            np.testing.assert_allclose(lite.modelout.AGB_model, scaffold.modelout.AGB_model, rtol=0, atol=0)


def test_multiday_lite_fast_state_loop_matches_compact_outputs_through_second_day():
    compact_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=2,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
    )
    fast_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=2,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
        use_fast_state_loop=True,
    )

    assert fast_run.ready_for_requested_days
    assert fast_run.missing_components == compact_run.missing_components
    assert len(fast_run.daily_modelout) == len(compact_run.daily_modelout)
    for fast_day, compact_day in zip(fast_run.daily_modelout, compact_run.daily_modelout, strict=True):
        assert fast_day.day_index == compact_day.day_index
        assert fast_day.modelout_fields.keys() == compact_day.modelout_fields.keys()
        for name, compact_value in compact_day.modelout_fields.items():
            np.testing.assert_allclose(fast_day.modelout_fields[name], compact_value, rtol=0, atol=0)
        np.testing.assert_allclose(fast_day.modelout.GPP_model, compact_day.modelout.GPP_model, rtol=0, atol=0)
        np.testing.assert_allclose(fast_day.modelout.NPP_model, compact_day.modelout.NPP_model, rtol=0, atol=0)
        np.testing.assert_allclose(fast_day.modelout.AGB_model, compact_day.modelout.AGB_model, rtol=0, atol=0)
    _assert_previous_step_packets_exact(fast_run.last_day_end_state, compact_run.last_day_end_state)


def test_multiday_lite_prebuilt_day_payloads_match_step_local_payloads_through_second_day():
    local_payload_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=2,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
        prebuild_day_payloads=False,
    )
    prebuilt_payload_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=2,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
        prebuild_day_payloads=True,
    )

    assert prebuilt_payload_run.ready_for_requested_days
    assert prebuilt_payload_run.missing_components == local_payload_run.missing_components
    assert len(prebuilt_payload_run.daily_modelout) == len(local_payload_run.daily_modelout)
    for prebuilt_day, local_day in zip(prebuilt_payload_run.daily_modelout, local_payload_run.daily_modelout, strict=True):
        assert prebuilt_day.day_index == local_day.day_index
        assert prebuilt_day.modelout_fields.keys() == local_day.modelout_fields.keys()
        for name, local_value in local_day.modelout_fields.items():
            np.testing.assert_allclose(prebuilt_day.modelout_fields[name], local_value, rtol=0, atol=0)
        np.testing.assert_allclose(prebuilt_day.modelout.GPP_model, local_day.modelout.GPP_model, rtol=0, atol=0)
        np.testing.assert_allclose(prebuilt_day.modelout.NPP_model, local_day.modelout.NPP_model, rtol=0, atol=0)
        np.testing.assert_allclose(prebuilt_day.modelout.AGB_model, local_day.modelout.AGB_model, rtol=0, atol=0)
    _assert_previous_step_packets_exact(prebuilt_payload_run.last_day_end_state, local_payload_run.last_day_end_state)


def test_multiday_lite_compiled_sechiba_day_matches_strict_outputs_and_restart_state():
    common = dict(
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
    )
    strict_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        **common,
        use_compiled_sechiba_day=False,
    )
    compiled_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        **common,
        use_compiled_sechiba_day=True,
    )

    assert strict_run.ready_for_requested_days
    assert compiled_run.ready_for_requested_days
    for compiled_day, strict_day in zip(compiled_run.daily_modelout, strict_run.daily_modelout, strict=True):
        assert compiled_day.modelout_fields.keys() == strict_day.modelout_fields.keys()
        for name, expected in strict_day.modelout_fields.items():
            _assert_values_close(compiled_day.modelout_fields[name], expected)

    actual_state = compiled_run.last_day_end_state
    expected_state = strict_run.last_day_end_state
    assert actual_state.tstep == expected_state.tstep
    assert actual_state.fields_by_component.keys() == expected_state.fields_by_component.keys()
    for component, expected_fields in expected_state.fields_by_component.items():
        actual_fields = actual_state.fields_by_component[component]
        assert actual_fields.keys() == expected_fields.keys()
        for name, expected in expected_fields.items():
            _assert_values_close(actual_fields[name], expected)
    assert driver_year_handoff_state_gaps(actual_state) == ()


def test_multiday_lite_compiled_day_blocks_match_daily_compiled_path():
    common = dict(
        year=1961,
        ndays=5,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    daily_run = paper_1961_driver_multiday_modelout_lite_run(CONFIG, **common)
    block_run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        **common,
        compiled_day_block_size=2,
    )

    assert daily_run.ready_for_requested_days
    assert block_run.ready_for_requested_days
    for block_day, daily_day in zip(
        block_run.daily_modelout,
        daily_run.daily_modelout,
        strict=True,
    ):
        assert block_day.day_index == daily_day.day_index
        assert block_day.start_tstep == daily_day.start_tstep
        assert block_day.end_tstep == daily_day.end_tstep
        _assert_values_close(block_day.modelout_fields, daily_day.modelout_fields)
        _assert_values_close(block_day.modelout, daily_day.modelout)

    actual_state = block_run.last_day_end_state
    expected_state = daily_run.last_day_end_state
    assert actual_state.tstep == expected_state.tstep
    for component, expected_fields in expected_state.fields_by_component.items():
        actual_fields = actual_state.fields_by_component[component]
        for name, expected in expected_fields.items():
            _assert_values_close(actual_fields[name], expected)
    assert driver_year_handoff_state_gaps(actual_state) == ()


def test_multiday_lite_end_state_satisfies_year_handoff_contract_and_rebases_counter():
    run = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
        use_fast_state_loop=True,
    )

    assert run.ready_for_requested_days
    fast_state = fast_state_from_previous_packet(run.last_day_end_state)
    restored_state = previous_packet_from_fast_state(fast_state)
    assert fast_state.components == tuple(run.last_day_end_state.fields_by_component.keys())
    _assert_previous_step_packets_exact(restored_state, run.last_day_end_state)

    gaps = driver_year_handoff_state_gaps(run.last_day_end_state)
    assert gaps == ()

    rebased = rebase_driver_state_for_year_start(run.last_day_end_state)
    assert rebased.tstep == -1
    assert rebased.fields_by_component is not run.last_day_end_state.fields_by_component
    assert "nroot" not in rebased.fields_by_component["hydrol_previous_step_state"]
    assert "nroot" in run.last_day_end_state.fields_by_component["hydrol_previous_step_state"]
    assert set(rebased.provenance_by_component) == set(run.last_day_end_state.provenance_by_component)


def test_restart_year_start_day_result_reports_missing_handoff_state_without_fabricating():
    day = paper_1961_driver_restart_year_start_day_result(
        CONFIG,
        previous_year_end_state=None,
        year=1962,
        used_run_def_path=USED_RUN_DEF,
        use_static_jit_daily_carbon=True,
    )

    assert not day.ready_for_day_end_state
    assert day.year == 1962
    assert day.day_index == 1
    assert day.start_tstep == 0
    assert "year_handoff_state" in day.missing_components


def test_restart_year_multiday_lite_reports_missing_handoff_state_without_fabricating():
    run = paper_1961_driver_restart_year_multiday_modelout_lite_run(
        CONFIG,
        previous_year_end_state=None,
        year=1962,
        ndays=2,
        used_run_def_path=USED_RUN_DEF,
        use_static_jit_daily_carbon=True,
    )

    assert not run.ready_for_requested_days
    assert run.year == 1962
    assert run.stopped_day_index == 1
    assert "day_1:year_handoff_state" in run.missing_components


def test_restart_year_compiled_day_blocks_match_daily_compiled_path():
    source = paper_1961_driver_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=3,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        use_compiled_sechiba_day=True,
    )
    common = dict(
        previous_year_end_state=source.last_day_end_state,
        year=1962,
        ndays=5,
        used_run_def_path=USED_RUN_DEF,
        use_static_jit_daily_carbon=True,
        use_compiled_sechiba_day=True,
    )
    daily_run = paper_1961_driver_restart_year_multiday_modelout_lite_run(
        CONFIG,
        **common,
    )
    block_run = paper_1961_driver_restart_year_multiday_modelout_lite_run(
        CONFIG,
        **common,
        compiled_day_block_size=2,
    )

    assert daily_run.ready_for_requested_days
    assert block_run.ready_for_requested_days
    for block_day, daily_day in zip(
        block_run.daily_modelout,
        daily_run.daily_modelout,
        strict=True,
    ):
        _assert_values_close(block_day.modelout_fields, daily_day.modelout_fields)
        _assert_values_close(block_day.modelout, daily_day.modelout)
    for component, expected_fields in daily_run.last_day_end_state.fields_by_component.items():
        actual_fields = block_run.last_day_end_state.fields_by_component[component]
        for name, expected in expected_fields.items():
            _assert_values_close(actual_fields[name], expected)
