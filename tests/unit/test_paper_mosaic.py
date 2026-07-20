from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from jax_orchidee.driver.paper_mosaic import (
    build_paper_mosaic_manifest,
    build_paper_mosaic_runtime_tasks,
    build_paper_mosaic_year_tasks,
    discover_paper_mosaic_cases,
    history_is_single_landpoint,
    history_lat_lon_shape,
    materialize_paper_mosaic_case_run_defs,
    materialized_case_run_def_path,
    paper_mosaic_case_manifest,
    summarize_paper_mosaic_manifest,
    summarize_paper_mosaic_runtime_tasks,
    summarize_paper_mosaic_year_tasks,
)
from jax_orchidee.driver.init import parse_run_def
from jax_orchidee.driver.domain import load_case_config


ROOT = Path(__file__).resolve().parents[2]


def _write_history(path: Path, *, lat: int = 1, lon: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={"GPP": (("time_counter", "lat", "lon", "veget"), np.zeros((1, lat, lon, 14)))},
        coords={
            "time_counter": [0.0],
            "lat": np.arange(lat, dtype=np.float64),
            "lon": np.arange(lon, dtype=np.float64),
            "veget": np.arange(14, dtype=np.int32),
        },
    )
    ds.to_netcdf(path)


def test_discover_paper_mosaic_cases_finds_independent_single_landpoint_run_defs(tmp_path):
    root = tmp_path / "orc_calibrate_250919_sen"
    first = root / "arg2_1.0" / "001.0-071.0" / "I10" / "S2_63.206_0.0876_0.2019_50.658"
    second = root / "arg2_0.9" / "001.0-073.0" / "I11" / "S3_1.0_2.0_3.0_4.0"
    for case_dir in (first, second):
        case_dir.mkdir(parents=True)
        (case_dir / "run.def").write_text("RIVER_ROUTING=y\nLIMIT_WEST=108.0\n", encoding="utf-8")
    (first / "z1").mkdir()
    (first / "z1" / "run.def.2010").write_text("RIVER_ROUTING=y\n", encoding="utf-8")

    cases = discover_paper_mosaic_cases(root)

    assert tuple(case.key for case in cases) == (
        "arg2_0.9/001.0-073.0/I11/S3_1.0_2.0_3.0_4.0",
        "arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658",
    )
    assert cases[1].case_tokens == (1.0, 71.0)
    assert cases[1].independent_single_landpoint is True
    assert "nbp_glo>1 routing remains a separate workflow" in cases[1].routing_scope_note
    assert cases[1].run_def_values()["RIVER_ROUTING"] == "y"


def test_paper_mosaic_case_reports_missing_history_years(tmp_path):
    root = tmp_path / "orc_calibrate_250919_sen"
    case_dir = root / "arg2_1.0" / "001.0-071.0" / "I10" / "S2"
    case_dir.mkdir(parents=True)
    (case_dir / "run.def").write_text("RIVER_ROUTING=y\n", encoding="utf-8")
    _write_history(case_dir / "stomate_history_1961.nc")

    case = discover_paper_mosaic_cases(root)[0]

    assert case.history_path(1961).name == "stomate_history_1961.nc"
    assert case.missing_history_years((1961, 1962, 2010)) == (1962, 2010)
    assert history_lat_lon_shape(case.history_path(1961)) == (1, 1)
    assert history_is_single_landpoint(case.history_path(1961))


def test_paper_mosaic_manifest_records_run_protocol_without_claiming_branch_coverage(tmp_path):
    root = tmp_path / "orc_calibrate_250919_sen"
    case_dir = root / "arg2_1.0" / "001.0-071.0" / "I10" / "S2"
    case_dir.mkdir(parents=True)
    (case_dir / "run.def").write_text(
        "\n".join(
            [
                "LIMIT_WEST=108.0",
                "LIMIT_EAST=110.0",
                "LIMIT_NORTH=22.0",
                "LIMIT_SOUTH=20.0",
                "FORCING_FILE=/forcing/cruncep_twodeg_1961.nc",
                "ATM_CO2=317.27",
            ]
        ),
        encoding="utf-8",
    )
    _write_history(case_dir / "stomate_history_1961.nc")

    case = discover_paper_mosaic_cases(root)[0]
    row = paper_mosaic_case_manifest(case, years=(1961, 1962), check_history_shape=True)

    assert row.domain_limits == {"west": 108.0, "east": 110.0, "south": 20.0, "north": 22.0}
    assert row.archived_run_def_forcing_file == "/forcing/cruncep_twodeg_1961.nc"
    assert row.archived_run_def_atm_co2 == 317.27
    assert row.missing_history_years == (1962,)
    assert row.first_history_lat_lon_shape == (1, 1)
    assert row.has_single_landpoint_history is True
    assert row.to_json_row()["case_tokens"] == [1.0, 71.0]


def test_paper_mosaic_manifest_summary_keeps_changed_forcing_branch_audit_scope(tmp_path):
    root = tmp_path / "orc_calibrate_250919_sen"
    for label in ("001.0-071.0", "001.0-073.0"):
        case_dir = root / "arg2_1.0" / label / "I10" / "S2"
        case_dir.mkdir(parents=True)
        (case_dir / "run.def").write_text("LIMIT_WEST=108.0\n", encoding="utf-8")
    _write_history(root / "arg2_1.0" / "001.0-071.0" / "I10" / "S2" / "stomate_history_1961.nc")

    rows = build_paper_mosaic_manifest(root, years=(1961,), check_history_shape=True)
    summary = summarize_paper_mosaic_manifest(rows, discovered_cases=2, expected_landpoint_cases=669)

    assert summary["expected_landpoint_cases"] == 669
    assert summary["discovered_cases"] == 2
    assert summary["manifest_rows"] == 2
    assert summary["missing_history_case_count_in_rows"] == 1
    assert summary["paper_target_scope"] == "669 independent single-landpoint cases"
    assert "changed forcing, landpoint state, or continuous parameters" in summary["branch_coverage_scope"]


def test_paper_mosaic_year_tasks_schedule_each_case_year_and_report_missing_reference(tmp_path):
    root = tmp_path / "orc_calibrate_250919_sen"
    first = root / "arg2_1.0" / "001.0-071.0" / "I10" / "S2"
    second = root / "arg2_1.0" / "001.0-073.0" / "I10" / "S2"
    for case_dir in (first, second):
        case_dir.mkdir(parents=True)
        (case_dir / "run.def").write_text("LIMIT_WEST=108.0\n", encoding="utf-8")
    _write_history(first / "stomate_history_1961.nc")
    _write_history(first / "stomate_history_1962.nc")
    _write_history(second / "stomate_history_1961.nc")

    tasks = build_paper_mosaic_year_tasks(root, years=(1961, 1962))
    summary = summarize_paper_mosaic_year_tasks(tasks)

    assert tuple(task.task_key for task in tasks) == (
        "arg2_1.0/001.0-071.0/I10/S2/1961",
        "arg2_1.0/001.0-071.0/I10/S2/1962",
        "arg2_1.0/001.0-073.0/I10/S2/1961",
        "arg2_1.0/001.0-073.0/I10/S2/1962",
    )
    assert tasks[-1].reference_history_exists is False
    assert tasks[-1].to_json_row()["independent_single_landpoint"] is True
    assert summary["task_count"] == 4
    assert summary["case_count"] == 2
    assert summary["years"] == [1961, 1962]
    assert summary["missing_reference_task_count"] == 1
    assert summary["missing_reference_tasks"] == ["arg2_1.0/001.0-073.0/I10/S2/1962"]


def test_materialize_paper_mosaic_case_run_defs_writes_complete_case_used_run_def(tmp_path):
    root = tmp_path / "orc_calibrate_250919_sen"
    case_dir = root / "arg2_1.0" / "001.0-071.0" / "I10" / "S2"
    case_dir.mkdir(parents=True)
    (case_dir / "run.def").write_text(
        "\n".join(
            [
                "LIMIT_WEST=-180.0",
                "LIMIT_EAST=-178.0",
                "LIMIT_NORTH=-18.0",
                "LIMIT_SOUTH=-20.0",
                "ATM_CO2=387.99",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    base = tmp_path / "base_used.def"
    base.write_text("DT_SECHIBA=1800\nATM_CO2=317.27\nNVM=14\n", encoding="utf-8")
    output_root = tmp_path / "materialized"
    case = discover_paper_mosaic_cases(root)[0]

    rows = materialize_paper_mosaic_case_run_defs(root, base_used_run_def=base, output_root=output_root)
    output = materialized_case_run_def_path(output_root, case)
    values = parse_run_def(output)

    assert rows[0]["key"] == "arg2_1.0/001.0-071.0/I10/S2"
    assert rows[0]["written"] is True
    assert output.exists()
    assert values["DT_SECHIBA"] == "1800"
    assert values["LIMIT_WEST"] == "-180.0"
    assert values["LIMIT_EAST"] == "-178.0"
    assert values["ATM_CO2"] == "317.27"
    assert values["LAND_COVER_CHANGE"] == "n"


def test_paper_mosaic_runtime_tasks_report_materialized_run_def_readiness(tmp_path):
    root = tmp_path / "orc_calibrate_250919_sen"
    ready_dir = root / "arg2_1.0" / "001.0-071.0" / "I10" / "S2"
    missing_dir = root / "arg2_1.0" / "001.0-073.0" / "I10" / "S2"
    incomplete_dir = root / "arg2_1.0" / "001.0-075.0" / "I10" / "S2"
    for case_dir in (ready_dir, missing_dir, incomplete_dir):
        case_dir.mkdir(parents=True)
        (case_dir / "run.def").write_text("LIMIT_WEST=-180.0\n", encoding="utf-8")
        _write_history(case_dir / "stomate_history_1961.nc")
        for filename in ("driver_start.nc", "sechiba_start.nc", "stomate_start.nc"):
            (case_dir / filename).write_bytes(b"placeholder")

    materialized_root = tmp_path / "materialized"
    ready_case, missing_case, incomplete_case = discover_paper_mosaic_cases(root)
    materialized_case_run_def_path(materialized_root, ready_case).parent.mkdir(parents=True)
    materialized_case_run_def_path(materialized_root, ready_case).write_text(
        "\n".join(
            [
                "DT_SECHIBA=1800",
                "DT_STOMATE=86400",
                "STOMATE_OK_STOMATE=y",
                "STOMATE_OK_DGVM=n",
                "NVM=14",
                "NSTM=6",
                "LIMIT_WEST=-180",
                "LIMIT_EAST=-178",
                "LIMIT_SOUTH=-20",
                "LIMIT_NORTH=-18",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    materialized_case_run_def_path(materialized_root, incomplete_case).parent.mkdir(parents=True)
    materialized_case_run_def_path(materialized_root, incomplete_case).write_text(
        "DT_SECHIBA=1800\nLIMIT_WEST=-180\n",
        encoding="utf-8",
    )

    tasks = build_paper_mosaic_runtime_tasks(
        root,
        years=(1961,),
        materialized_run_def_root=materialized_root,
    )
    summary = summarize_paper_mosaic_runtime_tasks(tasks)

    assert tasks[0].ready_for_runtime is True
    assert tasks[1].readiness_gaps() == ("missing_materialized_run_def", "missing_materialized_run_def_keys:DT_SECHIBA,DT_STOMATE,STOMATE_OK_STOMATE,STOMATE_OK_DGVM,NVM,NSTM,LIMIT_WEST,LIMIT_EAST,LIMIT_SOUTH,LIMIT_NORTH")
    assert "DT_STOMATE" in tasks[2].missing_run_def_keys()
    assert summary["task_count"] == 3
    assert summary["ready_task_count"] == 1
    assert summary["missing_start_file_task_count"] == 0
    assert summary["missing_materialized_run_def_task_count"] == 1
    assert summary["missing_required_run_def_key_task_count"] == 1


def test_history_single_landpoint_check_distinguishes_true_multicell_output(tmp_path):
    one = tmp_path / "one.nc"
    multi = tmp_path / "multi.nc"
    _write_history(one, lat=1, lon=1)
    _write_history(multi, lat=2, lon=3)

    assert history_is_single_landpoint(one)
    assert not history_is_single_landpoint(multi)
    assert history_lat_lon_shape(multi) == (2, 3)


def test_config_records_paper_mosaic_as_independent_single_landpoint_cases():
    config = load_case_config(ROOT / "configs" / "orchidee_man_250919.yaml")
    mosaic = config["paper_mosaic"]

    assert mosaic["execution_unit"] == "independent_single_landpoint"
    assert mosaic["expected_landpoint_cases"] == 669
    assert "nbp_glo>1 routing run" in " ".join(mosaic["notes"])
