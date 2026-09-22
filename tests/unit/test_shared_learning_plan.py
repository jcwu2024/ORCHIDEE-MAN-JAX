import json
from pathlib import Path

import pytest

from scripts.hpc.prepare_shared_learning_data import day_keys
from scripts.hpc.shared_learning_plan import build_learning_plan

ROOT = Path(__file__).resolve().parents[2]


def test_plan_is_repeatable_without_replacement_and_separates_fit_role():
    config = json.loads((ROOT / "manifests/coarse_graining/shared_learning_v1.json").read_text())
    manifest = {"identity": {"dataset": "train"}, "entries": [
        {"update": i, "runtime_year": 1961 + i % 44} for i in range(1000)]}
    plan = build_learning_plan(manifest, config)
    reordered = {**manifest, "entries": list(reversed(manifest["entries"]))}
    assert plan == build_learning_plan(reordered, config)
    assert len(set(plan["training_artifact_ids"])) == 100
    assert plan["fit_artifact_ids"] == plan["training_artifact_ids"][:8]
    assert plan["summary"]["daily_draws"] == 3200
    assert plan["summary"]["unique_daily_rows"] is None
    assert not plan["summary"]["is_epoch"]


@pytest.mark.parametrize("identifiers", [range(99), [0] * 101])
def test_plan_rejects_insufficient_or_ambiguous_artifacts(identifiers):
    with pytest.raises(ValueError, match="distinct available"):
        build_learning_plan({"entries": [{"update": i} for i in identifiers]},
                            {"max_updates": 100})


def test_window_coverage_counts_overlapping_days_once():
    rows = [{"landpoint_id": "A", "year": 1961, "start_rows": [1, 2]},
            {"landpoint_id": "B", "year": 1961, "start_rows": [1]}]
    keys = day_keys(rows, horizon=7)
    assert len(keys) == 15
    assert ("A", 1961, 8) in keys
    assert ("A", 1961, 9) not in keys
