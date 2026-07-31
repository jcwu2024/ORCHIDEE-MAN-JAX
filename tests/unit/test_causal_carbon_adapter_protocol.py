from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.daily_coarse_graining.causal_carbon_adapter_protocol import (
    load_causal_carbon_adapter_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "manifests" / "coarse_graining" / "canonical_669_causal_carbon_adapter_experiment.json"


def test_causal_carbon_adapter_protocol_is_frozen_and_loadable():
    protocol = load_causal_carbon_adapter_protocol(PROTOCOL)

    assert protocol.raw["parent"]["parameters_frozen"] is True
    assert protocol.raw["data_policy"]["sealed_test_used"] is False
    assert protocol.raw["train_only_feasibility_gate"]["updates_per_arm"] == 8
    assert len(protocol.sha256) == 64


def test_protocol_rejects_validation_leak_and_candidate_checkpoint_reuse(
    tmp_path,
):
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    raw["data_policy"]["optimization_and_coefficient_calibration"]["spatial_split"] = "validation"
    leaked = tmp_path / "leaked.json"
    leaked.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="train/train"):
        load_causal_carbon_adapter_protocol(leaked)

    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    raw["rejected_predecessor"]["reuse_candidate_checkpoint"] = True
    reused = tmp_path / "reused.json"
    reused.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be reused"):
        load_causal_carbon_adapter_protocol(reused)
