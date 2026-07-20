from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/sync_pft14_production_evidence_scope.py"


def _module():
    scripts = str(ROOT / "scripts/dev")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("production_scope_sync", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sync_rejects_partial_coverage(tmp_path):
    module = _module()
    module.COVERAGE = tmp_path / "coverage.json"
    module.SCOPE = tmp_path / "scope.yaml"
    module.COVERAGE.write_text(
        json.dumps({"complete": False, "reachable_entries": 1, "records": []}),
        encoding="utf-8",
    )
    module.SCOPE.write_text("entries: {}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="not complete"):
        module.main()


def test_sync_promotes_only_after_complete_gate(tmp_path):
    module = _module()
    module.COVERAGE = tmp_path / "coverage.json"
    module.SCOPE = tmp_path / "scope.yaml"
    module.COVERAGE.write_text(
        json.dumps(
            {
                "complete": True,
                "reachable_entries": 1,
                "records": [
                    {
                        "ledger_entry": "example.active.step",
                        "comparison_asset": "outputs/example.json",
                        "passed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    module.SCOPE.write_text(
        yaml.safe_dump(
            {
                "entries": {
                    "example.active.step": {
                        "kind": "fortran_micro_oracle",
                        "scope": "oracle",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert module.main() == 0
    scope = yaml.safe_load(module.SCOPE.read_text(encoding="utf-8"))
    assert scope["entries"]["example.active.step"]["kind"] == "production_verified"
    assert scope["entries"]["example.active.step"]["evidence"] == "outputs/example.json"
