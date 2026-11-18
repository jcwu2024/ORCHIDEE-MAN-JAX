"""Original-Fortran PB-02 references through eager and production narrow JIT.

The frozen tuple is local component input coverage, not a natural trajectory
or independently verified atmospheric/PBL or DIFFUCO producer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.dev.oracle_physics_branch_energy import certify

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/physics_branch_energy_reference.json"
CASE_NAMES = (
    "dt900-explicit1-laidev0",
    "dt2700-explicit1-laidev0",
    "dt900-explicit0-laidev0",
    "dt900-explicit1-laidev1",
    "dt900-explicit0-laidev1",
    "qsat",
)


@pytest.fixture(scope="module")
def reference():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checks(reference):
    return certify(reference)


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("mode", ("eager", "jit"))
def test_original_fortran_energy_reference(checks, case, mode):
    selected = [row for row in checks if row["case"] == case and row["mode"] == mode]
    assert selected
    assert all(row["passed"] for row in selected), [row for row in selected if not row["passed"]]


def test_declared_energy_inputs_match_versioned_reference(reference):
    """Input identity uses the versioned recipe and fixture, without original source."""
    from scripts.dev.oracle_physics_branch_energy import NAMES, json_inputs, make_inputs

    assert reference["point_names"] == NAMES
    assert [case["name"] for case in reference["cases"]] == list(CASE_NAMES[:-1])
    declarations = ((900.0, True, False), (2700.0, True, False), (900.0, False, False),
                    (900.0, True, True), (900.0, False, True))
    for case, (dt, explicit, laidev) in zip(reference["cases"], declarations, strict=True):
        assert json_inputs(make_inputs(dt, explicit, laidev)) == case["inputs"]


@pytest.mark.external_data
def test_reference_still_bound_to_original_fortran(reference):
    """Optional original-source gate; absence is reported separately from regression."""
    missing = sorted({source["path"] for source in reference["source"] if not (ROOT / source["path"]).is_file()})
    if missing:
        pytest.skip("Original-Fortran source/span gate NOT EXECUTED; missing source: " + ", ".join(missing))
    for source in reference["source"]:
        raw = (ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["source_sha256"]
        span = b"".join(raw.splitlines(keepends=True)[source["start"] - 1 : source["end"]])
        assert hashlib.sha256(span).hexdigest() == source["span_sha256"]
