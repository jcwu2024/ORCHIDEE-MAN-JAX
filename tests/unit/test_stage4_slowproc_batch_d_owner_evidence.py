from __future__ import annotations

import json
from pathlib import Path

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/slowproc_batch_d"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
OWNER_IDS = {
    "pft14-owner-contract-1efd6322fc73": 84,
    "pft14-owner-contract-463396c23861": 51,
}


def _load(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="ascii"))


def test_batch_d_closes_exactly_the_two_slowproc_owners() -> None:
    inputs = _load("inputs.json")
    coverage = _load("branch_coverage.json")
    evidence = _load("owner_region_evidence.json")
    aggregate = _load("aggregate_audit.json")

    assert inputs["owner_ids"] == list(OWNER_IDS)
    assert coverage["required_by_owner"] == OWNER_IDS
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 135
    assert coverage["missing_arm_ids"] == []
    assert coverage["branch_complete"] is True
    assert {item["support"] for item in coverage["arms"]} == {
        "real_gcov",
        "exact_fatal_witness",
        "pinned_source_proof",
    }
    assert len(coverage["fatal_witness_arm_ids"]) == 2
    assert evidence["complete"] is True
    assert {record["owner_region_id"] for record in evidence["records"]} == set(OWNER_IDS)
    for record in evidence["records"]:
        assert len(record["required_arm_ids"]) == OWNER_IDS[record["owner_region_id"]]
        assert record["required_arm_ids"] == record["covered_arm_ids"]
        assert record["missing_arm_ids"] == []
        assert record["direct_state_writeback_comparison"]["status"] == "passed"
        assert record["direct_state_writeback_comparison"]["tolerance_policy"]["changed"] is False

    accepted = {
        record["owner_region_id"]: record
        for record in aggregate["records"]
        if record["owner_region_id"] in OWNER_IDS
    }
    assert set(accepted) == set(OWNER_IDS)
    assert all(record["passed"] for record in accepted.values())
    assert not any(aggregate["errors"].values())


def test_batch_d_assets_pin_original_bytes_real_format_and_fatals() -> None:
    inputs = _load("inputs.json")
    init = _load("slowproc_init_comparison.json")
    soilt = _load("slowproc_soilt_comparison.json")
    witnesses = _load("execution_witnesses.json")

    for procedure, metadata in inputs["owner_procedure_spans"].items():
        span = extract_procedure_bytes(SOURCE, procedure)
        assert metadata["sha256"] == span.span_sha256
        assert (OUTPUT / metadata["asset"]).read_bytes() == span.span_bytes

    assert init["status"] == soilt["status"] == "passed"
    assert len(init["comparisons"]) == 92
    assert len(soilt["comparisons"]) == 17
    assert init["tolerance_policy"] == soilt["tolerance_policy"] == {
        "rtol": 1.0e-12,
        "atol": 1.0e-14,
        "changed": False,
    }
    assert Path(soilt["real_format_inputs"]["netcdf_path"]).suffix == ".nc"
    assert all(item["matched"] for item in witnesses["fatal_witnesses"])
    assert {item["case"] for item in witnesses["fatal_witnesses"]} == {
        "retry",
        "fatal",
    }
