from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / "source_audits" / name).read_text(encoding="utf-8")


def test_pft14_execution_plan_keeps_long_runs_after_branch_and_performance_work():
    plan = _read_doc("pft14_completion_execution_plan.md")

    single_case_index = plan.index("Close one active paper-case PFT14 single-landpoint path")
    mosaic_index = plan.index("Generalize that closed single-landpoint path into the paper mosaic runner")
    subset_index = plan.index("Run a small representative landpoint subset")
    branch_index = plan.index("Complete source-backed SECHIBA/STOMATE structural branches")
    performance_index = plan.index("Optimize performance before scaling")
    long_run_index = plan.index("Use long annual or multiyear runs, including the 2010 paper CSV target")

    assert single_case_index < mosaic_index < subset_index < branch_index < performance_index < long_run_index
    assert "mosaic of independent single-landpoint\n  cases, not one `nbp_glo>1`" in plan
    assert "669 independent single-landpoint case orchestration" in plan
    assert "emits deterministic case-year tasks for validation runners" in plan
    assert "threads the selected case `run.def` through\n   both domain zoom metadata and run-scalar" in plan
    assert "Archived case `run.def` files are not assumed to contain materialized\n   Fortran defaults" in plan
    assert "materialize_paper_mosaic_run_defs.py" in plan
    assert "prepared driver context accept an explicit\n   case run directory" in plan
    assert "Passing all 669 paper mosaic cases is still only the paper-workflow target" in plan
    assert "changing forcing data, landpoint state,\n  hydrological/salinity/tide conditions" in plan
    assert "one-week execution focus is PFT14 semantic closure" in plan
    assert "they do not\n  multiply the amount of model process logic to port" in plan
    assert "Near-term work must return to source-driven PFT14 branch closure" in plan
    assert "continue the PFT14-relevant\n    branch ledger" in plan
    assert "do not rely\n   on long single-point runs to discover them" in plan
    assert "sechiba_pft14_branch_microcase_matrix.md" in plan
    assert "Full daily `OK_PC`/`deep_carbcycle`\n   coupling remains explicitly unsupported" not in plan


def test_sechiba_pft14_branch_matrix_classifies_reachable_and_deferred_branches():
    matrix = _read_doc("sechiba_pft14_branch_microcase_matrix.md")

    required_rows = (
        "Driver land-vector and static aggregation (`PFT14 active`)",
        "Driver static thermal/albedo/SOC readers (`PFT14 conditional`)",
        "SLOWPROC external LAI map (`PFT14 conditional`)",
        "DIFFUCO mangrove salinity/tide controls (`PFT14 active`)",
        "ENERBIL active energy/evaporation boundary (`PFT14 active`)",
        "HYDROL runoff-to-peat/tide routing (`PFT14 active/conditional for mangrove`)",
        "HYDROL 1D refSOC hydraulic thresholds (`PFT14 conditional`)",
        "HYDROL alternate residual bare-soil evaporation diagnostic (`PFT14 conditional`)",
        "CONDVEG static roughness (`PFT14 conditional`)",
        "THERMOSOIL CWRR thermal profile and coefficients (`PFT14 active`)",
        "SECHIBA routing river network (`PFT14 conditional by domain size`)",
        "Long-distance DOC routing boundary (`PFT14 conditional`)",
        "DIFFUCO BVOC chemistry (`Non-PFT14 current target`)",
    )
    for row in required_rows:
        assert row in matrix

    assert "Long paper-case runs are validation gates only" in matrix
    assert "Do not spend near-term effort on Choisnel, BVOC, crop/grassland" in matrix
    assert "monthly `mean`/`inter` LAI selection feeds the cold-start `slowproc_veget` surface state" in matrix
    assert "`slowproc_interlai` file reading/regridding remains an explicit source boundary" in matrix
    assert "`READ_REFTEMP` repeats the Celsius-plus-Kelvin field across thermal layers and PFTs" in matrix
    assert "Driver cold-start and later-day HYDROL static templates now carry `refSOC_1d`" in matrix


def test_routing_flow_sequence_ledger_keeps_wrapper_branch_composition_open():
    ledger = _read_doc("routing_flow_sequence_ledger.md")

    required_helpers = (
        "routing_reservoir_outflow_step",
        "routing_stream_erosion_step",
        "routing_flood_pond_input_step",
        "routing_floodplain_flux_step",
        "routing_pond_flux_step",
        "routing_transport_between_basins_step",
        "routing_swamp_flood_step",
        "routing_reservoir_update_step",
        "routing_area_fractions_step",
        "routing_co2_chemistry_step",
        "routing_return_reinfiltration_step",
        "routing_irrigation_step",
        "routing_flow_diagnostics_step",
        "routing_lake_overflow_step",
    )
    for helper in required_helpers:
        assert helper in ledger

    assert "`routing_flow_step` now passes explicit state" in ledger
    assert "`routing_daily_boundary_step` now covers" in ledger
    assert "Do not claim full multi-landpoint/global routing-flow parity yet" in ledger
    assert "wrapper-level branch composition" in ledger
    assert "production asset wiring" in ledger


def test_pft14_closure_sweep_separates_stale_trace_gaps_from_real_work():
    sweep = _read_doc("pft14_closure_sweep_20260709.md")
    plan = _read_doc("pft14_completion_execution_plan.md")
    surface = _read_doc("enerbil_surface_state_input_chain.md")
    soil = _read_doc("enerbil_soil_thermal_state_input_chain.md")
    diffuco = _read_doc("diffuco_pft14_minimal_closure_audit.md")
    hydrol = _read_doc("hydrol_full_solve_trace_contract.md")
    routing = _read_doc("routing_pft14_branch_ledger.md")

    assert "stale" in plan and "bridge-trace wording separately" in plan
    assert "No new model semantic blocker was found" in sweep
    assert "Stale bridge-trace gap, not a production process gap" in sweep
    assert "Source-covered only with same-step two-band albedo" in sweep
    assert "Source-covered by coherent restart fields or `thermosoil_coef` path" in sweep
    assert "Diagnostic trace limitation, not current production HYDROL export gap" in sweep
    assert "PFT14 active DIFFUCO beta/resistance/GPP boundary is closed" in sweep
    assert "Asset/production-wiring gate for `nbp_glo>1`, not 669 independent one-cell paper blocker" in sweep
    assert "Non-PFT14 natural triggers or out-of-target management branches" in sweep

    assert "`after_enerbil_main` is never used as a pre-call source" in surface
    assert "Non-restart initialization when the four fields are absent is now covered" in soil
    assert "Numeric ENERBIL evaporation/transpiration and HYDROL exported-boundary\n  closure are now covered" in diffuco
    assert "not a claim that the current\nproduction driver has a HYDROL-to-STOMATE process gap" in hydrol
    assert "not a blocker for the paper workflow\nthat runs the 669 target cells as independent `nbp_glo=1` cases" in routing
