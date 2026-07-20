from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "outputs/reference_mode/micro_oracles/interpweight_rank2_owner_closure"
)
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpweight.f90"
AGGREGATE_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpol_help.f90"
OWNER_IDS = {
    "pft14-owner-contract-252914bbabc7",
    "pft14-owner-contract-372b01d64eab",
}


def _json(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _fortran_fields() -> dict[tuple[str, str], list[float]]:
    fields: dict[tuple[str, str], list[float]] = defaultdict(list)
    with (OUTPUT / "fortran_outputs.csv").open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields[(row["case"], row["field"])].append(float(row["value"]))
    return fields


def test_original_owner_and_scientific_helper_bytes_are_materialized_exactly() -> None:
    comparison = _json("comparison.json")
    composed = (OUTPUT / "composed_oracle.f90").read_bytes()
    for procedure, metadata in comparison["procedure_spans"].items():
        source = AGGREGATE_SOURCE if procedure == "aggregate_2d" else SOURCE
        span = extract_procedure_bytes(source, procedure)
        extracted = ROOT / metadata["extracted_file"]
        assert extracted.read_bytes() == span.span_bytes
        assert metadata["span_sha256"] == span.span_sha256
        assert metadata["start_line"] == span.start_line
        assert metadata["end_line"] == span.end_line
        assert composed.count(span.span_bytes) == 1


def test_rank2_numerics_masks_dateline_boundaries_and_retries_pass() -> None:
    comparison = _json("comparison.json")
    assert comparison["status"] == "passed"
    assert comparison["tolerance"] == {
        "rtol": 1e-12,
        "atol": 1e-14,
        "discrete": "exact",
    }
    assert comparison["comparisons"]
    assert all(record["passed"] for record in comparison["comparisons"])
    names = {record["name"] for record in comparison["comparisons"]}
    for case in (
        "frac_east_nomask",
        "frac_west_var",
        "frac_boundary_mbelow",
        "frac_mabove",
        "frac_msumrange",
        "cont_east_default",
        "cont_west_slope_var",
        "cont_boundary_mbelow",
        "cont_mabove",
        "cont_msumrange",
    ):
        assert f"{case}.output" in names
        assert f"{case}.mask" in names
        assert f"{case}.overlap_index" in names

    fields = _fortran_fields()
    assert fields[("frac_dense_retry", "attempt_width")] == [200.0, 400.0]
    assert fields[("cont_dense_retry", "attempt_width")] == [
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        256.0,
    ]
    assert fields[("frac_boundary_mbelow", "availability")] == [-1.0, -1.0]
    assert fields[("cont_boundary_mbelow", "output")] == [7.5, 7.5]


def test_rank3_rank4_and_default_source_contracts_are_explicit_process_failures() -> None:
    comparison = _json("comparison.json")
    contracts = comparison["fatal_and_undefined_contracts"]
    assert len(contracts) == 12
    assert all(record["passed"] for record in contracts)
    assert {record["owner_region_id"] for record in contracts} == OWNER_IDS
    defaults = [record for record in contracts if "default" in record["mode"]]
    undefined = [record for record in contracts if "default" not in record["mode"]]
    assert len(defaults) == 2
    assert len(undefined) == 10
    assert all(record["source_contract"] == "explicit_ipslerr_p_fatal" for record in defaults)
    assert all(record["returncode"] == 73 for record in defaults)
    assert all("IPSLERR" in record["stdout"] for record in defaults)
    assert all(
        record["source_contract"] == "unallocated_invar2D_undefined"
        for record in undefined
    )
    assert all("is not allocated" in record["stderr"] for record in undefined)


def test_exactly_two_canonical_owner_arm_sets_are_accepted_by_formal_audit() -> None:
    canonical = json.loads(
        (ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        record["owner_region_id"]: set(record["arm_ids"])
        for record in canonical["owner_regions"]
        if record["owner_region_id"] in OWNER_IDS
    }
    source_proofs = json.loads(
        (ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    passed_source_proof_ids = {
        record["arm_id"]
        for record in source_proofs["records"]
        if record.get("passed") is True
    }
    evidence = _json("owner_region_evidence.json")
    assert evidence["complete"] is True
    assert len(evidence["records"]) == 2
    assert {record["owner_region_id"] for record in evidence["records"]} == OWNER_IDS
    expected_route_counts = {
        "pft14-owner-contract-252914bbabc7": (71, 48, 5, 18),
        "pft14-owner-contract-372b01d64eab": (66, 44, 7, 15),
    }
    for record in evidence["records"]:
        owner_id = record["owner_region_id"]
        required = set(record["required_arm_ids"])
        gcov = set(record["gcov_arm_ids"])
        fatal = set(record["fatal_boundary_arm_ids"])
        source_proof = set(record["source_proof_arm_ids"])
        assert set(record["required_arm_ids"]) == expected[owner_id]
        assert set(record["covered_arm_ids"]) == expected[owner_id]
        assert gcov | fatal | source_proof == required
        assert gcov.isdisjoint(fatal)
        assert gcov.isdisjoint(source_proof)
        assert fatal.isdisjoint(source_proof)
        assert source_proof <= passed_source_proof_ids
        assert (
            len(required),
            len(gcov),
            len(fatal),
            len(source_proof),
        ) == expected_route_counts[owner_id]
        assert record["missing_arm_ids"] == []
        assert record["passed"] is True

    coverage = _json("branch_coverage.json")
    assert coverage["branch_complete"] is True
    assert coverage["required_arm_count"] == 104
    assert coverage["covered_arm_count"] == 104
    assert coverage["missing_arm_ids"] == []
    assert len(coverage["arms"]) == 104
    for arm in coverage["arms"]:
        routes = (
            arm["gcov_passed"],
            arm["fatal_boundary_passed"],
            arm["source_proof_passed"],
        )
        assert sum(routes) == 1
        assert arm["passed"] is True
        if arm["gcov_passed"]:
            assert (
                arm["gcov_taken"] > 0
                or arm["execution_witness_count"] > 0
            )
        if arm["fatal_boundary_passed"]:
            assert arm["fatal_boundary_witness_count"] > 0
        assert arm["source_proof_passed"] is False

    audit = _json("formal_audit_report.json")
    assert audit["complete"] is True
    assert audit["counts"]["required_owner_regions"] == 2
    assert audit["counts"]["passed_owner_regions"] == 2
    assert audit["counts"]["missing_owner_regions"] == 0
    assert all(not values for values in audit["errors"].values())
    assert audit["missing_owner_region_ids"] == []
