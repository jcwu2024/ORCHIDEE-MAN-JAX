from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_PATH = ROOT / "scripts" / "dev" / "extract_fortran_micro_oracle.py"


def _extractor():
    spec = importlib.util.spec_from_file_location(
        "framework_oracle_extractor", EXTRACTOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_family_manifest_source_hashes_are_current():
    module = _extractor()
    records = module.validate_and_extract(module.load_manifest(), root=ROOT)
    family_ids = {entry.get("family_id") for entry, _ in records}
    assert {"diffuco_surface_exchange", "stomate_daily_maintenance"} <= family_ids


def test_pilot_family_cached_comparisons_are_passing():
    for family in ("diffuco_surface_exchange", "stomate_daily_maintenance"):
        path = (
            ROOT
            / "outputs"
            / "reference_mode"
            / "micro_oracles"
            / family
            / "comparison.json"
        )
        result = json.loads(path.read_text(encoding="ascii"))
        assert result["family"] == family
        assert result["status"] == "passed"
        assert result["comparisons"]
        assert all(item["passed"] for item in result["comparisons"])
        assert all("max_abs_error" in item for item in result["comparisons"])
        assert all("max_ulp_error" in item for item in result["comparisons"])
