"""Actual numerical -> NetCDF -> readstart -> maintained mapper -> physics."""

from __future__ import annotations

import json

import numpy as np
import pytest
from netCDF4 import Dataset

from scripts.dev.oracle_phase06_scientific_restart import (
    ENTRY_PRODUCERS,
    OUT,
    REFERENCE,
    ROOT,
    run,
    sha,
)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    folder = tmp_path_factory.mktemp("phase06_scientific_restart")
    return run(folder), folder


def checks_with(result, prefix):
    return [check for check in result["checks"] if check["name"].startswith(prefix)]


def test_independent_original_scientific_producer_and_both_next_returns(evidence):
    result, _folder = evidence
    for prefix, count in (
        ("original_producer.", 49),
        ("original_next.eager.", 49),
        ("original_next.public_compiled.", 49),
    ):
        checks = checks_with(result, prefix)
        assert len(checks) == count
        assert all(check["passed"] for check in checks), [c for c in checks if not c["passed"]]
    assert result["status"] == "passed"
    assert len(result["checks"]) == 1402


@pytest.mark.parametrize("cell", range(6))
def test_six_cell_reference_survives_actual_singlepoint_writer_and_readstart(evidence, cell):
    result, folder = evidence
    record = result["cell_records"][cell]
    checks = checks_with(result, f"cell{cell}.actual_io.")
    assert len(checks) > 150
    assert all(check["passed"] for check in checks), [c for c in checks if not c["passed"]]
    assert not record["mapper_gaps"] and not record["season_mapper_gaps"]
    assert len(record["write_report"].written_fields) == 161
    source_locals = ("read_input_depth_organic_soil", "read_input_thawed_humidity")
    assert record["actual_readstart_report"].source_undefined_fields == source_locals
    assert record["actual_source_local_flags"] == dict.fromkeys(source_locals)
    assert "depth_deepsoil" in record["declared_seed_report"].defaulted_fields
    assert record["declared_seed_report"].source_undefined_fields == ()
    with Dataset(folder / record["output"]) as dataset:
        assert dataset.variables["biomass"].shape == (1, 1, 12, 14, 1, 1)
        assert dataset.variables["leaf_age"].shape == (1, 4, 14, 1, 1)
        assert dataset.variables["gdd_midwinter"].shape == (1, 14, 1, 1)
        assert dataset.variables["depth_deepsoil"].shape == (1, 14, 1, 1)
        np.testing.assert_array_equal(dataset.variables["depth_deepsoil"][:], 0)
    # Each file contains one slice of the six-cell reference, not a new
    # independently compiled singleton oracle.
    assert np.asarray(result["actual_producer_fields"]["turnover.biomass"]).shape[0] == 6


def test_maintained_mapper_and_real_next_components_receive_reloaded_values(evidence):
    result, _folder = evidence
    for cell in range(6):
        checks = checks_with(result, f"cell{cell}.mapper.")
        assert len(checks) == len(ENTRY_PRODUCERS) + 1
        assert all(check["passed"] for check in checks)
    for prefix in ("restored_next_operand.", "actual_next_consumer.", "actual_setlai_from_restart"):
        checks = checks_with(result, prefix)
        assert checks and all(check["passed"] for check in checks), [c for c in checks if not c["passed"]]
    assert result["complete_restored_next_input"]["prescribe_inputs"]["firstcall"] is False
    assert result["complete_restored_next_input"]["prescribe_inputs"]["stomate_restart_none"] is False


def test_reference_fixture_and_source_local_limits_are_explicit(evidence):
    result, _folder = evidence
    assert result["independent_reference"]["sha256"] == sha(REFERENCE)
    assert result["independent_reference"]["cases"] == ["environment_partition", "environment_partition_reentry"]
    assert set(result["source_local_operands"]) == {"cn_ind", "sla_age1", "lai"}
    assert "Not a natural restart" in result["limit"]


@pytest.mark.external_data
def test_external_original_source_identities_do_not_gate_portable_numerics():
    fixture = json.loads(REFERENCE.read_text())
    identities = [
        value
        for value in fixture["source_identities"].values()
        if isinstance(value, dict) and "file" in value and "sha256" in value
    ]
    missing = [item["file"] for item in identities if not (ROOT / item["file"]).is_file()]
    if missing:
        pytest.skip("External original source assets unavailable: " + ", ".join(missing))
    assert identities
    assert all(sha(ROOT / item["file"]) == item["sha256"] for item in identities)
    assert OUT.is_absolute()
