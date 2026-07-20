from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/audit_pft14_production_coverage.py"


def _module():
    spec = importlib.util.spec_from_file_location("production_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _span(path: Path, root: Path = ROOT) -> dict:
    data = path.read_bytes()
    return {
        "file": path.relative_to(root).as_posix(),
        "symbol": "fixture",
        "start_line": 1,
        "end_line": len(data.splitlines()),
        "file_sha256": sha256(data).hexdigest(),
        "span_sha256": sha256(data).hexdigest(),
    }


def test_production_gate_requires_executed_comparison_and_matching_oracle(tmp_path):
    module = _module()
    module.ROOT = tmp_path
    source = tmp_path / "owner.py"
    source.write_text("def fixture():\n    return 1\n", encoding="ascii")
    asset = tmp_path / "comparison.json"
    asset.write_text(
        json.dumps(
            {
                "status": "passed",
                "production_execution": True,
                "comparisons": [{"name": "state", "passed": True}],
            }
        ),
        encoding="utf-8",
    )
    record = {
        "ledger_entry": "example.active.process",
        "oracle_family": "example_oracle",
        "status": "verified",
        "production_owner": _span(source, tmp_path),
        "callsites": [_span(source, tmp_path)],
        "contracts": {"inputs": ["forcing"], "outputs": ["flux"], "state_writeback": ["store"]},
        "oracle_case_ids": ["warm"],
        "comparison_asset": asset.relative_to(tmp_path).as_posix(),
    }
    ledger = {"entries": [{"id": "example.active.process", "reachability": "active"}]}
    oracle = {
        "records": [
            {"ledger_entry": "example.active.process", "family": "example_oracle", "passed": True}
        ]
    }
    report = module.build_report([record], ledger, oracle)
    assert report["complete"]
    assert report["passed_entries"] == 1

    record["oracle_family"] = "wrong"
    rejected = module.build_report([record], ledger, oracle)
    assert not rejected["complete"]
    assert "oracle_family does not match" in " ".join(rejected["records"][0]["errors"])


def test_static_or_missing_execution_asset_cannot_pass(tmp_path):
    module = _module()
    asset = tmp_path / "comparison.json"
    asset.write_text(
        json.dumps({"status": "passed", "comparisons": [{"passed": True}]}),
        encoding="utf-8",
    )
    assert not module._comparison_passed(asset)
