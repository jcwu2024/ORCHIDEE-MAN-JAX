from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "run_teacher_branch_parity.py"
SPEC = importlib.util.spec_from_file_location("run_teacher_branch_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_snapshot_roundtrip_and_exact_discrete_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    metadata = {"input_files": [{"path": "forcing.nc", "sha256": "abc", "size": 1}]}
    payload = {
        "days": {
            "day_0001": {
                "state": {"carbon": np.array([1.0, 2.0]), "mask": np.array([True, False])}
            }
        }
    }
    MODULE._write_snapshot(baseline, payload, metadata)
    MODULE._write_snapshot(candidate, payload, metadata)
    summary, rows = MODULE._compare_case(
        case={"id": "case", "landpoint_id": "point"},
        baseline_dir=baseline,
        candidate_dir=candidate,
        atol=0.0,
        rtol=0.0,
    )
    assert summary["passed"]
    assert len(rows) == 2

    changed = {
        "days": {
            "day_0001": {
                "state": {"carbon": np.array([1.0, 2.0]), "mask": np.array([False, False])}
            }
        }
    }
    MODULE._write_snapshot(candidate, changed, metadata)
    summary, rows = MODULE._compare_case(
        case={"id": "case", "landpoint_id": "point"},
        baseline_dir=baseline,
        candidate_dir=candidate,
        atol=0.0,
        rtol=0.0,
    )
    assert not summary["passed"]
    assert summary["first_differing_day"] == 1
    assert any(row["status"] == "discrete_mismatch" for row in rows)


def test_input_identity_ignores_worktree_path(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    payload = {"days": {"day_0001": {"state": np.array([1.0])}}}
    MODULE._write_snapshot(
        baseline,
        payload,
        {"input_files": [{"path": "main/config.yaml", "sha256": "abc", "size": 3}]},
    )
    MODULE._write_snapshot(
        candidate,
        payload,
        {"input_files": [{"path": "research/config.yaml", "sha256": "abc", "size": 3}]},
    )
    summary, _ = MODULE._compare_case(
        case={"id": "case", "landpoint_id": "point"},
        baseline_dir=baseline,
        candidate_dir=candidate,
        atol=0.0,
        rtol=0.0,
    )
    assert summary["passed"]


def test_disposition_is_fail_closed_and_oracle_bound(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle.json"
    oracle.write_text('{"status":"passed"}\n', encoding="utf-8")
    disposition = {
        "id": "source_correction",
        "case_id": "case",
        "first_differing_day": 4,
        "first_day_max_abs_error": 1.0e-8,
        "first_day_fields": ["root/days/day_0004/state/carbon"],
        "oracle": {
            "path": "oracle.json",
            "sha256": MODULE._sha256(oracle),
            "required_status": "passed",
        },
    }
    summary = {
        "case_id": "case",
        "passed": False,
        "first_differing_day": 4,
    }
    rows = [
        {
            "day": 4,
            "path": "root/days/day_0004/state/carbon",
            "status": "float_mismatch",
            "max_abs_error": 1.0e-9,
        }
    ]
    accepted = MODULE._apply_disposition(
        summary=summary,
        rows=rows,
        dispositions=[disposition],
        asset_root=tmp_path,
    )
    assert accepted["accepted"]

    rows[0]["status"] = "discrete_mismatch"
    rejected = MODULE._apply_disposition(
        summary=summary,
        rows=rows,
        dispositions=[disposition],
        asset_root=tmp_path,
    )
    assert not rejected["accepted"]


def test_manifest_declares_required_lifecycle_matrix() -> None:
    manifest = json.loads((ROOT / "configs" / "teacher_branch_parity.json").read_text(encoding="utf-8"))
    assert manifest["baseline"] == {"id": "main", "revision": "main"}
    assert manifest["candidate"] == {"id": "research", "revision": "HEAD"}
    assert manifest["dispositions"] == []
    cases = manifest["cases"]
    assert {case["scenario"] for case in cases} == {"cold", "reference_start", "restart"}
    assert any(case["days"] == 365 for case in cases)
    assert len({case["landpoint_id"] for case in cases}) >= 3
    assert any(
        case.get("source_scenario") == "cold" and case.get("source_days") == 365
        for case in cases
    )
