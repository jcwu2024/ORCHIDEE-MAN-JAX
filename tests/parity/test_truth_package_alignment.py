from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.init import parse_run_def  # noqa: E402
from jax_orchidee.driver.orchestration import paper_1961_driver_cold_start_multiday_modelout_lite_run  # noqa: E402
from jax_orchidee.driver.restart import reference_restart_path  # noqa: E402
from jax_orchidee.stomate.carbon_kernels import ICARBON, ILEAF  # noqa: E402
from jax_orchidee.stomate.reference import (  # noqa: E402
    compare_modelout_fields_to_history_point,
    find_stomate_reference_files,
    pack_history_modelout_fields,
    read_restart_biomass_carbon,
)


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
SERVER_PACKAGE = ROOT / "outputs" / "server_1961_trace_full_20260623"
SERVER_USED_RUN_DEF = SERVER_PACKAGE / "run" / "used_run.def"
SERVER_TRACE_DIR = SERVER_PACKAGE / "traces"


def _time_counter_len(path: Path) -> int:
    with Dataset(path) as dataset:
        return int(np.asarray(dataset.variables["time_counter"][:]).reshape(-1).size)


def test_reference_history_1961_is_sparse_not_daily_truth():
    files = find_stomate_reference_files(ROOT)

    assert _time_counter_len(files.run_dir / "stomate_history_1961.nc") == 1


def test_server_trace_package_history_1961_is_sparse_not_daily_truth():
    history = SERVER_PACKAGE / "netcdf" / "stomate_history_1961.nc"

    assert history.exists()
    assert _time_counter_len(history) == 1


def test_current_runner_uses_reference_start_files_not_server_trace_restarts():
    files = find_stomate_reference_files(ROOT)
    reference_run = parse_run_def(files.run_dir / "run.def")
    server_run = parse_run_def(SERVER_PACKAGE / "run" / "run.def")

    assert reference_restart_path(CONFIG, "sechiba_start.nc").name == "sechiba_start.nc"
    assert files.start.name == "stomate_start.nc"
    assert reference_run["SECHIBA_restart_in"] == "sechiba_start.nc"
    assert reference_run["STOMATE_RESTART_FILEIN"] == "stomate_start.nc"
    assert server_run["SECHIBA_restart_in"] == "NONE"
    assert server_run["STOMATE_RESTART_FILEIN"] == "NONE"


def test_server_stomate_restart_is_not_same_initial_state_as_reference_start():
    files = find_stomate_reference_files(ROOT)
    reference_start = read_restart_biomass_carbon(files.start)
    server_restart = read_restart_biomass_carbon(SERVER_PACKAGE / "netcdf" / "stomate_restart.nc")

    reference_leaf = float(reference_start[0, 13, ILEAF, ICARBON])
    server_leaf = float(server_restart[0, 13, ILEAF, ICARBON])

    assert reference_leaf > 200.0
    assert server_leaf < 200.0
    assert abs(reference_leaf - server_leaf) > 50.0


def test_history_modelout_comparison_closes_when_inputs_are_same_fortran_history():
    history = SERVER_PACKAGE / "netcdf" / "stomate_history_1961.nc"
    fields = pack_history_modelout_fields(history)

    comparison = compare_modelout_fields_to_history_point(fields, history)

    assert comparison.ok
    assert comparison.max_abs_difference == 0.0
    assert comparison.jax_values["GPP"] == comparison.history_values["GPP"]
    assert comparison.jax_values["LEAF_M"] == comparison.history_values["LEAF_M"]


def test_short_cold_start_run_is_not_a_1961_year_end_history_closure():
    run = paper_1961_driver_cold_start_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=3,
        fixed_format_trace_dir=SERVER_TRACE_DIR,
        used_run_def_path=SERVER_USED_RUN_DEF,
        single_pass_daily_fold=True,
    )
    history = SERVER_PACKAGE / "netcdf" / "stomate_history_1961.nc"

    comparison = compare_modelout_fields_to_history_point(run.daily_modelout[-1].modelout_fields, history)

    assert run.ready_for_requested_days
    assert comparison.max_abs_difference > 1.0
    assert comparison.abs_differences["LEAF_M"] > 50.0
    assert comparison.abs_differences["GPP"] > 1.0


@pytest.mark.skipif(
    not bool(int(os.environ.get("ORCHJAX_RUN_SLOW_YEAR", "0"))),
    reason="Set ORCHJAX_RUN_SLOW_YEAR=1 to run the cold-start 1961 annual history parity check.",
)
def test_slow_cold_start_1961_year_end_modelout_matches_server_history():
    run = paper_1961_driver_cold_start_multiday_modelout_lite_run(
        CONFIG,
        year=1961,
        ndays=365,
        fixed_format_trace_dir=SERVER_TRACE_DIR,
        used_run_def_path=SERVER_USED_RUN_DEF,
        single_pass_daily_fold=True,
    )
    history = SERVER_PACKAGE / "netcdf" / "stomate_history_1961.nc"

    comparison = compare_modelout_fields_to_history_point(run.daily_modelout[-1].modelout_fields, history)

    assert run.ready_for_requested_days
    np.testing.assert_allclose(
        tuple(comparison.jax_values[name] for name in comparison.fields),
        tuple(comparison.history_values[name] for name in comparison.fields),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
