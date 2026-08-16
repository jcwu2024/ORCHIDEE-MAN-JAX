from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.attention_mixed_continuation import (
    classify_attention_continuation,
    continuation_learning_rate,
    load_attention_continuation_state,
    load_attention_mixed_continuation_contract,
    mixed_update_groups,
    write_attention_continuation_state,
)

ROOT = Path(__file__).resolve().parents[2]


def _validation_report(value: float, *, maintenance: float = 1.0):
    keys = (
        "cumulative.npp",
        "terminal.aboveground_biomass_carbon",
        "terminal.belowground_biomass_carbon",
        "terminal.total_live_biomass_carbon",
    )
    return {
        "slices": {
            slice_id: {
                "hard_counts": {"nonfinite": 0, "budget": 0},
                "horizons": {
                    str(horizon): {
                        "keys": {
                            **{
                                key: {"normalized_rmse": float(value)}
                                for key in keys
                            },
                            "cumulative.maintenance_respiration": {
                                "normalized_rmse": float(maintenance)
                            },
                        }
                    }
                    for horizon in (7, 30)
                },
            }
            for slice_id in ("temporal", "spatial", "joint")
        }
    }


def test_attention_mixed_continuation_contract_and_cosine_schedule():
    contract = load_attention_mixed_continuation_contract()
    initial = 2.981e-4
    rates = [
        continuation_learning_rate(contract, update, initial_learning_rate=initial)
        for update in range(1, 251)
    ]
    assert rates[0] == pytest.approx(initial)
    assert rates[-1] == pytest.approx(3.0e-5)
    assert all(left > right for left, right in zip(rates, rates[1:]))
    assert (rates[124] + rates[125]) / 2.0 == pytest.approx(
        (initial + 3.0e-5) / 2.0
    )
    with pytest.raises(ValueError, match="outside"):
        continuation_learning_rate(contract, 251, initial_learning_rate=initial)


def test_attention_mixed_groups_cover_unique_point_years_once():
    entries = [
        {"landpoint_id": f"point-{index % 4}", "year": 1900 + index}
        for index in range(12)
    ]
    groups = mixed_update_groups(
        entries,
        point_years_per_update=4,
        expected_updates=3,
    )
    assert len(groups) == 3
    assert all(len(group) == 4 for group in groups)
    repeated = list(entries)
    repeated[-1] = dict(repeated[0])
    with pytest.raises(ValueError, match="repeated"):
        mixed_update_groups(
            repeated,
            point_years_per_update=4,
            expected_updates=3,
        )


def test_attention_mixed_continuation_classification_uses_scientific_outputs():
    contract = load_attention_mixed_continuation_contract()
    baseline = _validation_report(1.0, maintenance=1.0)
    candidate = _validation_report(0.98, maintenance=1.1)
    result = classify_attention_continuation(
        baseline,
        candidate,
        contract=contract,
    )
    assert result["passed"] is True
    assert result["research_primary"]["relative_improvement"] == pytest.approx(0.02)
    assert result["research_primary"]["improved_cell_count"] == 6
    assert result["maintenance"]["relative_change"] == pytest.approx(0.1)

    failed = _validation_report(1.01, maintenance=0.9)
    failed_result = classify_attention_continuation(
        baseline,
        failed,
        contract=contract,
    )
    assert failed_result["passed"] is False
    assert failed_result["checks"]["research_primary_composite_improved"] is False


def test_attention_mixed_continuation_state_round_trip(tmp_path):
    path = tmp_path / "state.pkl"
    identity = {"contract": "accepted"}
    parameters = {"kernel": np.asarray([1.0, 2.0], dtype=np.float64)}
    optimizer = {
        "step": np.asarray(3, dtype=np.int32),
        "first": np.asarray([0.1, 0.2], dtype=np.float64),
    }
    record = write_attention_continuation_state(
        path,
        identity=identity,
        parameters=parameters,
        optimizer=optimizer,
        completed_updates=5,
        parameter_count=2,
    )
    assert record["promotion_eligible"] is False
    loaded = load_attention_continuation_state(
        path,
        identity=identity,
        parameter_count=2,
        maximum_updates=250,
    )
    assert loaded is not None
    assert loaded["completed_updates"] == 5
    np.testing.assert_array_equal(loaded["parameters"]["kernel"], [1.0, 2.0])


def test_attention_mixed_continuation_assets_are_candidate_only_and_free():
    contract = json.loads(
        (
            ROOT
            / "manifests"
            / "coarse_graining"
            / "successor_attention_mixed_annealed_continuation_v1.json"
        ).read_text(encoding="utf-8")
    )
    launcher = (
        ROOT
        / "scripts"
        / "hpc"
        / "run_successor_attention_mixed_continuation.sh"
    ).read_text(encoding="utf-8")
    runner = (
        ROOT
        / "scripts"
        / "hpc"
        / "run_successor_attention_mixed_continuation.py"
    ).read_text(encoding="utf-8")
    assert contract["training"]["continuation_updates"] == 250
    assert contract["model"]["graph_control_continued"] is False
    assert contract["training"]["free_gln01_only"] is True
    assert "#SBATCH" not in launcher
    assert "source_routed_graph_mlp_control_v2" not in runner
    assert "write_attention_continuation_state(" in runner
