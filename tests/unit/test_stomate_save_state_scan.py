from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev.scan_stomate_save_state import DEFAULT_SOURCE_DIR, scan_fortran_save_state  # noqa: E402


def test_stomate_save_state_scan_finds_known_annual_memory_fields():
    scan = scan_fortran_save_state(DEFAULT_SOURCE_DIR)
    by_name = {record["name"]: record for record in scan["records"]}
    records = scan["records"]

    assert scan["summary"]["records"] > 100
    assert scan["summary"]["unique_names"] > 100
    assert scan["summary"]["files"] > 10

    assert by_name["carbon"]["file"].endswith("src_stomate\\stomate.f90")
    assert by_name["carbon"]["threadprivate"] is True
    assert by_name["lm_lastyearmax"]["file"].endswith("src_stomate\\stomate.f90")
    assert any(
        record["name"] == "firstcall" and record["file"].endswith("src_stomate\\stomate_lpj.f90")
        for record in records
    )
    assert any(
        record["name"] == "altmax_lastyear" and record["file"].endswith("src_stomate\\stomate_soilcarbon.f90")
        for record in records
    )
    assert any(
        record["name"] == "use_fixed_cryoturbation_depth"
        and record["file"].endswith("src_stomate\\stomate_soilcarbon.f90")
        for record in records
    )
