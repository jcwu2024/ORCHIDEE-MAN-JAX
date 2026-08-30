import json
from pathlib import Path

import pytest

from scripts.hpc.run_target_oriented_structured_gpp_short_horizon_screen import (
    _contract_sha256,
    _load_screen_contract,
)
from scripts.hpc.run_target_oriented_training import _validation_records

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "manifests/coarse_graining/target_oriented_structured_gpp_short_horizon_screen_v1.json"
)


def _panel_and_prepared():
    panel = {"slices": {}}
    prepared = {"entries": []}
    for slice_id in ("temporal", "spatial", "joint"):
        panel["slices"][slice_id] = {"horizons": {}}
        for horizon in (7, 30):
            windows = []
            for ordinal in range(16):
                entry = {"window_id": f"{slice_id}-{horizon}-{ordinal}"}
                windows.append(entry)
                prepared["entries"].append({"artifact": entry["window_id"]})
            panel["slices"][slice_id]["horizons"][str(horizon)] = windows
    return panel, prepared


def test_structured_gpp_short_horizon_contract_freezes_read_only_96_window_screen():
    contract = _load_screen_contract(CONTRACT)
    assert contract["contract_sha256"] == contract["sha256"]
    assert contract["evaluation"]["formal_window_count"] == 96
    assert contract["evaluation"]["admission_window_count"] == 6
    assert contract["execution"]["optimizer_updates"] == 0
    assert contract["execution"]["checkpoint_writes"] == 0
    assert contract["acceptance"]["relative_comparison_or_control_required"] is False


def test_structured_gpp_short_horizon_contract_rejects_policy_drift(tmp_path):
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["execution"]["optimizer_updates"] = 1
    payload["contract_sha256"] = _contract_sha256(payload)
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract drift"):
        _load_screen_contract(path)


def test_structured_gpp_short_horizon_modes_select_six_or_ninety_six_windows():
    panel, prepared = _panel_and_prepared()
    assert len(_validation_records(panel, prepared, mode="admission")) == 6
    assert len(_validation_records(panel, prepared, mode="formal")) == 96


def test_structured_gpp_short_horizon_launcher_is_free_node_read_only():
    launcher = (
        ROOT / "scripts/hpc/run_target_oriented_structured_gpp_short_horizon_screen.sh"
    ).read_text(encoding="utf-8")
    assert "sbatch" not in launcher
    assert "--mode \"$MODE\"" in launcher
    assert "stage-update-0498.pkl" in launcher
    assert "--selected-checkpoint" in launcher
    assert "--screen-contract" in launcher
