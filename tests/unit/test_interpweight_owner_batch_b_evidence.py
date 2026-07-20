from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/interpweight_owner_batch_b"


def test_batch_b_preserves_original_owner_bytes_and_declares_harness_blocker() -> None:
    comparison = json.loads((OUTPUT / "comparison.json").read_text(encoding="ascii"))
    owners = json.loads((OUTPUT / "owner_region_evidence.json").read_text(encoding="ascii"))
    assert comparison["status"] == "partial"
    assert set(comparison["procedure_span_sha256"]) == {
        "pft14-owner-contract-252914bbabc7",
        "pft14-owner-contract-372b01d64eab",
    }
    assert len(owners["records"]) == 2
    assert all((OUTPUT / item["extracted_file"]).is_file() for item in owners["records"])
    assert owners["source_level_impossibility"]["interpweight_2D"]["unconditional_unallocated_bcast"] == 663
    assert owners["source_level_impossibility"]["interpweight_2Dcont"]["unconditional_unallocated_writeback"] == 2189
    assert owners["pft14_forcing_route"]["active_owner"] == "interpweight_3D"
    assert all(item["passed"] for item in comparison["boundary_comparisons"])
