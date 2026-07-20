import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/run_stomate_prescribe_owner_coverage.py"


def _module():
    spec = importlib.util.spec_from_file_location("prescribe_owner_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prescribe_owner_contract_covers_all_canonical_arms(tmp_path):
    result = _module().run_coverage(tmp_path)
    coverage = result["branch_coverage"]
    evidence = result["owner_evidence"]
    assert coverage["required_arm_count"] == 13
    assert coverage["covered_arm_count"] == 13
    assert coverage["missing_arm_ids"] == []
    assert evidence["complete"] is True
    assert evidence["records"][0]["owner_region_id"] == "pft14-owner-contract-261c1baede4c"
    assert (tmp_path / "central_audit_report.json").is_file()
