from __future__ import annotations

from dataclasses import dataclass

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    build_target_oriented_training_schedule,
    load_target_oriented_training_contract,
)


@dataclass(frozen=True)
class Parent:
    landpoint_id: str
    year: int
    sha256: str


@dataclass(frozen=True)
class Reference:
    parent: Parent
    sha256: str
    day_count: int = 364


def _references(count: int = 32):
    return tuple(
        Reference(
            parent=Parent(
                landpoint_id=f"point-{index:03d}",
                year=1961 + index % 10,
                sha256=f"parent-{index:03d}",
            ),
            sha256=f"typed-{index:03d}",
        )
        for index in range(count)
    )


def test_target_oriented_training_contract_freezes_diverse_effective_batch():
    contract = load_target_oriented_training_contract()
    assert contract.data["teacher_days_per_update"] == 208
    assert contract.data["distinct_point_years_per_update"] == 28
    assert contract.data["distinct_landpoints_per_update"] == 28
    assert contract.payload["microbatch"] == {
        "one_day_size": 4,
        "seven_day_window_size": 1,
        "thirty_day_window_size": 1,
        "gradient_accumulation_dtype": "float32",
        "loss_reduction_dtype": "float64",
        "optimizer_updates_per_effective_batch": 1,
        "global_bias_reduction": "exact_sufficient_statistics_across_the_full_effective_batch",
        "rollout_rematerialization": True,
        "multi_gpu_required": False,
    }
    assert contract.optimizer["stage_milestones"] == [250, 1000, 3000]
    assert contract.payload["validation"]["promotion"] == ("absolute_target_oriented_7_30_day_gates_only")


def test_target_oriented_schedule_is_deterministic_diverse_and_complete():
    contract = load_target_oriented_training_contract()
    references = _references()
    first = build_target_oriented_training_schedule(references, contract, updates=2)
    second = build_target_oriented_training_schedule(references, contract, updates=2)
    assert first == second
    assert first["schedule_sha256"] == second["schedule_sha256"]

    for entry in first["entries"]:
        components = entry["components"]
        assert {name: len(values) for name, values in components.items()} == {
            "one_day": 16,
            "seven_day": 8,
            "thirty_day": 4,
        }
        all_records = sum(components.values(), [])
        assert len({(item["landpoint_id"], item["year"]) for item in all_records}) == 28
        assert len({item["landpoint_id"] for item in all_records}) == 28
        assert sum(len(item["start_rows"]) for item in components["one_day"]) == 32
        assert sum(len(item["start_rows"]) for item in components["seven_day"]) == 8
        assert sum(len(item["start_rows"]) for item in components["thirty_day"]) == 4
        assert all(min(item["start_rows"]) >= 1 for item in all_records)
        assert all(max(item["start_rows"]) <= 334 for item in components["thirty_day"])


def test_target_oriented_schedule_fails_when_landpoint_diversity_is_impossible():
    contract = load_target_oriented_training_contract()
    repeated = tuple(
        Reference(Parent("same-point", 1961 + index, f"parent-{index}"), f"typed-{index}") for index in range(28)
    )
    with pytest.raises(ValueError, match="28 distinct landpoints"):
        build_target_oriented_training_schedule(repeated, contract, updates=1)
