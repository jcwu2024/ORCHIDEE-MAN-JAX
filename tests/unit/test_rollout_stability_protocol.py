from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.daily_coarse_graining import rollout_stability_protocol as protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "manifests"
    / "coarse_graining"
    / "canonical_669_rollout_stability_protocol.json"
)


def _write_mutation(tmp_path: Path, mutate) -> Path:
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "rollout_stability_protocol.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_frozen_rollout_stability_protocol_loads_and_binds_parent_assets():
    loaded = protocol.load_rollout_stability_protocol(PROTOCOL)

    assert loaded.sha256 == (
        "a0ecfd4c89ff3c5691f153168f3ba80939582774689b92ff04b0cf77e699cb40"
    )
    assert loaded.raw["protocol_id"] == "canonical-669-rollout-stability-v2"
    assert loaded.raw["amendment"]["predecessor_canonical_sha256"] == (
        "370011f6d8edc447bd0ebf037249df1a18f3bfb30071204001366a2aecde310b"
    )
    assert loaded.raw["amendment"]["sealed_test_used"] is False
    assert loaded.raw["data_policy"]["sealed_test_used"] is False
    assert loaded.raw["mixed_horizon_sampling"]["horizons"][-1]["days"] == 30
    assert loaded.raw["post_screening_promotion"][-1]["id"] == (
        "sealed_1961_2010_test"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["mixed_horizon_sampling"]["horizons"].reverse(),
        lambda raw: raw["mixed_horizon_sampling"]["horizons"][0].update(
            probability=0.5
        ),
    ],
)
def test_rollout_stability_protocol_rejects_horizon_or_probability_drift(
    tmp_path,
    mutate,
):
    path = _write_mutation(tmp_path, mutate)

    with pytest.raises(ValueError, match="mixed-horizon inventory"):
        protocol.load_rollout_stability_protocol(path)


def test_rollout_stability_protocol_rejects_unsealed_test(tmp_path):
    path = _write_mutation(
        tmp_path,
        lambda raw: raw["data_policy"].update(sealed_test_used=True),
    )

    with pytest.raises(ValueError, match="test split must remain sealed"):
        protocol.load_rollout_stability_protocol(path)


def test_rollout_stability_protocol_rejects_amendment_drift(tmp_path):
    path = _write_mutation(
        tmp_path,
        lambda raw: raw["amendment"].update(
            change="silently accept all defined-status mismatches"
        ),
    )

    with pytest.raises(ValueError, match="amendment evidence drift"):
        protocol.load_rollout_stability_protocol(path)


def test_rollout_stability_protocol_rejects_parent_decision_drift(tmp_path):
    def mutate(raw):
        raw["parent_architecture_screen"]["required_decision"] = (
            "advance_flat_to_rollout_stability_experiment"
        )

    path = _write_mutation(tmp_path, mutate)

    with pytest.raises(ValueError, match="parent architecture decision drift"):
        protocol.load_rollout_stability_protocol(path)


def test_rollout_stability_protocol_rejects_open_ended_weight_search(tmp_path):
    path = _write_mutation(
        tmp_path,
        lambda raw: raw["stop_rules"].update(open_ended_weight_search=True),
    )

    with pytest.raises(ValueError, match="open-ended coefficient search"):
        protocol.load_rollout_stability_protocol(path)


def test_rollout_stability_protocol_rejects_unknown_schema_fields(tmp_path):
    def mutate(raw):
        raw["unfrozen_override"] = copy.deepcopy(raw["optimization"])

    path = _write_mutation(tmp_path, mutate)

    with pytest.raises(ValueError, match="key inventory drift"):
        protocol.load_rollout_stability_protocol(path)
