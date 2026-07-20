from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes
from scripts.dev.oracle_lane_constantes_owners import (
    CASES,
    OWNER_IDS,
    PROCEDURES,
    SOURCE,
    VAR_SOURCE,
    compose,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/constantes_owners"


def test_constantes_owner_evidence_closes_all_ledger_arms() -> None:
    coverage = json.loads((OUTPUT / "branch_coverage.json").read_text(encoding="ascii"))
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text(encoding="ascii"))
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 17
    assert not coverage["missing_arm_ids"]
    assert owners["complete"]
    assert {record["owner_region_id"] for record in owners["records"]} == OWNER_IDS
    assert all(record["passed"] for record in owners["records"])


def test_constantes_oracle_compares_every_fortran_writeback_strictly() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="ascii"))
    assert comparison["status"] == "passed"
    assert comparison["tolerance"]["float_rtol"] <= 1e-12
    assert comparison["tolerance"]["discrete"] == "exact"
    assert all(record["passed"] for record in comparison["comparisons"])

    expected_writebacks = 0
    for procedure, modes in CASES.items():
        for mode in modes:
            with (OUTPUT / f"{procedure}_{mode}.csv").open(newline="", encoding="ascii") as handle:
                rows = list(csv.DictReader(handle))
            expected_writebacks += sum(
                not row["key"].startswith("__") and row["before"] != "derived"
                for row in rows
            )
    writeback_records = [
        record for record in comparison["comparisons"]
        if record["name"].split(".")[-1] not in {"read_order", "warnings", "SNEIGE"}
    ]
    assert len(writeback_records) == expected_writebacks


def test_constantes_lane_embeds_real_module_and_original_procedure_bytes(tmp_path: Path) -> None:
    generated_path = tmp_path / "oracle.f90"
    hashes = compose(generated_path)
    generated = generated_path.read_bytes()
    assert generated.count(VAR_SOURCE.read_bytes()) == 1
    assert b"interface getin_p" in generated
    assert b"call emit(trim(key),'real_array'" in generated
    for procedure in PROCEDURES:
        span = extract_procedure_bytes(SOURCE, procedure)
        assert hashes[procedure]["span_sha256"] == span.span_sha256
        assert generated.count(span.span_bytes) == 1
