from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/run_qsat_owner_region_coverage.py"
SPEC = importlib.util.spec_from_file_location("run_qsat_owner_region_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_gcov_preserves_branch_counts(tmp_path: Path) -> None:
    path = tmp_path / "sample.gcov"
    path.write_text(
        "        3:   10: IF (flag) THEN\n"
        "branch  0 taken 2 (fallthrough)\n"
        "branch  1 taken 1\n"
        "    #####:   11: value = 1\n",
        encoding="ascii",
    )
    assert MODULE.parse_gcov(path) == {
        10: [
            {"branch_index": 0, "taken": 2},
            {"branch_index": 1, "taken": 1},
        ]
    }


def test_gcov_arm_mapping_distinguishes_if_and_where_edges() -> None:
    branches = [
        {"branch_index": 0, "taken": 30},
        {"branch_index": 1, "taken": 2},
        {"branch_index": 23, "taken": 2},
        {"branch_index": 24, "taken": 28},
    ]
    assert MODULE.select_arm_branch("source.f90:10:if:true", branches) == branches[-2]
    assert MODULE.select_arm_branch("source.f90:10:if:false", branches) == branches[-1]
    assert MODULE.select_arm_branch("source.f90:20:where:true", branches) == branches[-2]
    assert MODULE.select_arm_branch("source.f90:20:where:false", branches) == branches[-1]


def test_qsat_owner_evidence_assets_are_complete() -> None:
    import json

    output = ROOT / "outputs/reference_mode/micro_oracles/qsat_moisture"
    coverage = json.loads((output / "branch_coverage.json").read_text())
    owners = json.loads((output / "owner_region_evidence.json").read_text())
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"]
    assert owners["complete"]
    assert all(record["passed"] for record in owners["records"])
