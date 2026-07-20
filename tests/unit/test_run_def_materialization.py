from __future__ import annotations

from pathlib import Path

from jax_orchidee.driver.run_def_materialization import (
    materialize_case_run_def_values,
    materialize_run_def_values,
    static_run_def_values,
    write_materialized_run_def,
)


def _write_run_def(path: Path, values: dict[str, str]) -> None:
    path.write_text("\n".join(f"{key} = {value}" for key, value in values.items()) + "\n", encoding="utf-8")


def test_static_run_def_values_excludes_year_dynamic_archived_keys():
    values = {
        "LIMIT_WEST": "108.0",
        "ATM_CO2": "387.99",
        "FORCING_FILE": "cruncep_twodeg_2010.nc",
        "RESTART_FILEIN": "driver_start.nc",
    }

    static = static_run_def_values(values)

    assert static == {"LIMIT_WEST": "108.0"}


def test_materialize_run_def_values_keeps_defaults_and_applies_case_overrides():
    merged = materialize_run_def_values(
        {"DT_SECHIBA": "1800", "LIMIT_WEST": "0", "LAND_COVER_CHANGE": "y"},
        {"LIMIT_WEST": "108.0", "NVM": "14"},
    )

    assert merged["DT_SECHIBA"] == "1800"
    assert merged["LIMIT_WEST"] == "108.0"
    assert merged["NVM"] == "14"
    assert merged["LAND_COVER_CHANGE"] == "n"
    assert merged["SECHIBA_VEGMAX__00014"] == "1.0"


def test_materialize_case_run_def_values_merges_incomplete_case_run_def_with_defaults(tmp_path):
    base = tmp_path / "base_used.def"
    case = tmp_path / "case.def"
    _write_run_def(base, {"DT_SECHIBA": "1800", "NVM": "14", "LIMIT_WEST": "0"})
    _write_run_def(case, {"LIMIT_WEST": "-180.0", "ATM_CO2": "387.99"})

    values = materialize_case_run_def_values(base_used_run_def=base, case_run_def=case)

    assert values["DT_SECHIBA"] == "1800"
    assert values["LIMIT_WEST"] == "-180.0"
    assert values["NVM"] == "14"
    assert "ATM_CO2" not in values


def test_materialized_fortran_used_truth_is_not_reoverridden_by_submission_case(tmp_path):
    used = tmp_path / "used.def"
    case = tmp_path / "case.def"
    _write_run_def(used, {"TIDES": "FALSE", "VCMAX25__00014": "69.0"})
    _write_run_def(case, {"TIDES": "y", "VCMAX25__00014": "70.0"})

    values = materialize_case_run_def_values(
        base_used_run_def=used,
        case_run_def=case,
        base_is_fortran_used_truth=True,
    )

    assert values["TIDES"] == "FALSE"
    assert values["VCMAX25__00014"] == "69.0"


def test_case_override_keys_preserve_fortran_getin_case_sensitivity():
    """pft_parameters.f90 reads uppercase TIDES; lowercase tides is distinct."""

    merged = materialize_run_def_values(
        {"TIDES": "FALSE", "READ_TIDE": "TRUE"},
        {"tides": "y"},
        job_protocol_overrides={},
    )

    assert merged["TIDES"] == "FALSE"
    assert merged["tides"] == "y"
    assert merged["READ_TIDE"] == "TRUE"


def test_write_materialized_run_def_writes_header_and_sorted_values(tmp_path):
    output = tmp_path / "used.def"

    write_materialized_run_def({"B": "2", "A": "1"}, output, header_lines=("# generated",))

    assert output.read_text(encoding="utf-8").splitlines() == ["# generated", "", "A = 1", "B = 2"]
