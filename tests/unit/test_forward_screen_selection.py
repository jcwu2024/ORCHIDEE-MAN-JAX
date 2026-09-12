from __future__ import annotations

import pytest

from research.daily_coarse_graining.daily_operator.forward_screen_selection import (
    select_windows,
    training_day_keys,
)


def _record(landpoint_id: str, year: int, starts: list[int]):
    return {"landpoint_id": landpoint_id, "year": year, "start_rows": starts}


def _entry(
    artifact_id: int,
    *,
    seven_day=(),
    thirty_day=(),
):
    return {
        "update": artifact_id,
        "runtime_year": 2000,
        "components": {
            "seven_day": list(seven_day),
            "thirty_day": list(thirty_day),
        },
    }


def _covered_days(window):
    return set(map(tuple, window["teacher_day_keys"]))


def test_training_day_keys_includes_every_catalogue_row_and_monitor_row():
    catalogue = {
        "one_day_entries": [
            {
                "update": 1,
                "components": {
                    "one_day": [
                        _record("site-a", 2000, [0, 4]),
                        _record("site-b", 2001, [7]),
                    ]
                },
            },
            {
                "update": 2,
                "components": {"one_day": [_record("site-a", 2000, [4, 9])]},
            },
        ],
        "monitor_rows": [
            {
                "artifact_id": 3,
                "batch_row": 0,
                "landpoint_id": "monitor-only",
                "year": 2002,
                "start_row": 11,
            }
        ],
    }

    assert training_day_keys(catalogue) == {
        ("site-a", 2000, 0),
        ("site-a", 2000, 4),
        ("site-a", 2000, 9),
        ("site-b", 2001, 7),
        ("monitor-only", 2002, 11),
    }


def test_interior_overlap_is_rejected_without_renumbering_actual_window_index():
    schedule = [
        _entry(
            10,
            seven_day=[
                _record("site", 2000, [4, 40]),
                _record("site", 2000, [80]),
            ],
            thirty_day=[_record("other", 2000, [200])],
        )
    ]
    excluded = {
        ("site", 2000, 6),  # Interior of the row-4 window.
        ("site", 2000, 40),
    }

    selected = select_windows(schedule, excluded, seed=17, per_horizon=1)

    assert [window["component"] for window in selected] == ["seven_day", "thirty_day"]
    seven_day = selected[0]
    assert seven_day["start_row"] == 80
    assert seven_day["window_index"] == 2
    assert not _covered_days(seven_day) & excluded


def test_selection_is_deterministic_for_a_fixed_pcg64_seed():
    schedule = [
        _entry(
            artifact_id,
            seven_day=[_record(f"seven-{artifact_id}", 2000, [artifact_id, artifact_id + 100])],
            thirty_day=[_record(f"thirty-{artifact_id}", 2000, [artifact_id + 300, artifact_id + 400])],
        )
        for artifact_id in range(1, 13)
    ]

    first = select_windows(schedule, set(), seed=20260913)
    second = select_windows(schedule, set(), seed=20260913)

    assert first == second
    assert [window["component"] for window in first] == ["seven_day"] * 8 + ["thirty_day"] * 8
    assert len({(window["landpoint_id"], window["year"]) for window in first}) == 16
    for component in ("seven_day", "thirty_day"):
        artifact_ids = [window["artifact_id"] for window in first if window["component"] == component]
        assert len(artifact_ids) == len(set(artifact_ids)) == 8


def test_seven_and_thirty_day_selections_are_mutually_day_disjoint():
    schedule = [
        _entry(
            1,
            seven_day=[_record("shared", 2000, [10])],
            thirty_day=[_record("shared", 2000, [0])],
        ),
        _entry(2, seven_day=[_record("shared", 2000, [50])]),
    ]

    selected = select_windows(schedule, set(), seed=3, per_horizon=1)
    seven_day, thirty_day = selected

    assert seven_day["start_row"] == 50
    assert thirty_day["start_row"] == 0
    assert _covered_days(seven_day).isdisjoint(_covered_days(thirty_day))


def test_cold_start_row_is_preserved_as_the_actual_storage_row():
    schedule = [
        _entry(
            1,
            seven_day=[_record("cold-start", 1961, [4])],
            thirty_day=[_record("independent", 1961, [100])],
        )
    ]

    selected = select_windows(schedule, {("cold-start", 1961, 3)}, seed=9, per_horizon=1)
    seven_day = selected[0]

    assert seven_day["start_row"] == 4
    assert seven_day["teacher_day_keys"] == tuple(
        ("cold-start", 1961, row) for row in range(4, 11)
    )


def test_selection_reports_a_direct_shortage_without_widening_metadata_scope():
    schedule = [
        _entry(
            1,
            seven_day=[_record("site", 2000, [4])],
            thirty_day=[_record("other", 2000, [100])],
        )
    ]

    with pytest.raises(ValueError, match=r"window shortage.*seven_day required=2 selected=1"):
        select_windows(schedule, set(), seed=1, per_horizon=2)
