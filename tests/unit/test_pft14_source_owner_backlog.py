from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "build_pft14_source_owner_backlog.py"
SPEC = importlib.util.spec_from_file_location("build_pft14_source_owner_backlog", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_backlog_groups_only_scientific_arms_without_owners():
    disposition = {
        "records": [
            {
                "arm_id": "f:2:if:true",
                "branch_id": "f:2:if",
                "file": "f.f90",
                "procedure": "step",
                "line": 2,
                "source": "if (soil_wet)",
                "disposition": "pft14_conditional",
                "scientific_source_owner_present": False,
            },
            {
                "arm_id": "f:2:if:false",
                "branch_id": "f:2:if",
                "file": "f.f90",
                "procedure": "step",
                "line": 2,
                "source": "if (soil_wet)",
                "disposition": "pft14_conditional",
                "scientific_source_owner_present": True,
            },
            {
                "arm_id": "f:3:if:true",
                "branch_id": "f:3:if",
                "file": "f.f90",
                "procedure": "step",
                "line": 3,
                "source": "if (bad_input)",
                "disposition": "outside_declared_workflow",
                "scientific_source_owner_present": False,
            },
        ]
    }

    result = BUILD.build_backlog(disposition)

    assert result["scientific_reachable_arms"] == 2
    assert result["scientific_arms_with_source_owner"] == 1
    assert result["scientific_arms_missing_source_owner"] == 1
    assert result["procedures_missing_source_owner"] == 1
    assert result["rows"][0]["branch_sites"] == 1
    assert not result["closed"]
