from __future__ import annotations

import pytest

from scripts.hpc.prepare_mixed_learning_schedule import select_mixed_learning_windows


def _record(landpoint_id: str, year: int, starts: list[int]):
    return {"landpoint_id": landpoint_id, "year": year, "start_rows": starts}


def _entry(artifact_id: int, *, seven_day=(), thirty_day=()):
    return {
        "update": artifact_id,
        "components": {"seven_day": list(seven_day), "thirty_day": list(thirty_day)},
    }


def _schedule(count: int):
    return [
        _entry(
            index,
            seven_day=[_record(f"site-{index}", 2000, [10 * index])],
            thirty_day=[_record(f"site-{index}", 2001, [100 + 40 * index])],
        )
        for index in range(count)
    ]


def _days(row):
    return set(map(tuple, row["teacher_day_keys"]))


def _semantic(row):
    return row["landpoint_id"], row["year"], row["start_row"], row["length"]


def test_schedule_is_deterministic_in_update_order_and_reports_cross_update_coverage():
    first = select_mixed_learning_windows(_schedule(5), set(), updates=4)
    second = select_mixed_learning_windows(_schedule(5), set(), updates=4)

    assert first == second
    assert [row["update"] for row in first["update_rows"]] == [1, 2, 3, 4]
    assert [row["horizon"] for row in first["window_table"]] == ["T7", "T30"] * 4
    assert first["coverage"]["selected_window_count"] == 8
    assert first["coverage"]["selected_unique_window_identity_count"] == 8
    assert first["candidate_counts"]["seven_day"]["selected_window_count"] == 4
    assert first["candidate_counts"]["thirty_day"]["selected_window_count"] == 4


def test_selection_excludes_actual_semantic_day_keys_and_keeps_real_window_coordinates():
    schedule = [
        _entry(
            1,
            seven_day=[_record("blocked", 2000, [10]), _record("usable", 2000, [40])],
            thirty_day=[_record("other", 2001, [100])],
        )
    ]
    result = select_mixed_learning_windows(schedule, {("blocked", 2000, 13)}, updates=1)

    chosen_t7 = result["update_rows"][0]["T7"]
    assert chosen_t7["landpoint_id"] == "usable"
    assert chosen_t7["start_row"] == 40
    assert chosen_t7["teacher_day_keys"] == tuple(("usable", 2000, day) for day in range(40, 47))
    assert not _days(chosen_t7) & {("blocked", 2000, 13)}


def test_same_update_overlap_is_avoided_without_repeating_windows():
    schedule = [
        _entry(
            1,
            seven_day=[_record("shared", 2000, [10])],
            thirty_day=[_record("shared", 2000, [0]), _record("separate", 2000, [100])],
        )
    ]
    result = select_mixed_learning_windows(schedule, set(), updates=1)
    row = result["update_rows"][0]

    assert _days(row["T7"]).isdisjoint(_days(row["T30"]))
    assert result["coverage"]["same_update_day_overlap_count"] == 0
    assert result["coverage"]["selected_unique_window_identity_count"] == 2


def test_same_semantic_window_in_multiple_artifacts_is_selected_once_and_reported():
    schedule = [
        _entry(
            1,
            seven_day=[_record("shared", 2000, [10])],
            thirty_day=[_record("t30-a", 2001, [100])],
        ),
        _entry(
            2,
            seven_day=[_record("shared", 2000, [10])],
            thirty_day=[_record("t30-b", 2002, [140])],
        ),
        _entry(
            3,
            seven_day=[_record("unique", 2003, [50])],
        ),
    ]
    result = select_mixed_learning_windows(schedule, set(), updates=2)

    selected_t7 = [row["T7"] for row in result["update_rows"]]
    assert len({_semantic(row) for row in selected_t7}) == 2
    assert result["candidate_counts"]["seven_day"]["eligible_alias_count_removed"] == 1
    assert result["coverage"]["selected_unique_semantic_window_count"] == 4
    assert result["coverage"]["selected_unique_site_count"] == 4
    assert result["coverage"]["selected_unique_year_count"] == 4


def test_duplicate_window_identity_is_rejected():
    duplicate = _entry(
        1,
        seven_day=[_record("site", 2000, [0])],
        thirty_day=[_record("site", 2001, [100])],
    )

    with pytest.raises(ValueError, match="duplicate seven_day window identity"):
        select_mixed_learning_windows([duplicate, duplicate], set(), updates=1)


def test_insufficient_distinct_pairs_fail_closed():
    with pytest.raises(ValueError, match="mixed-learning window shortage.*required_updates=2"):
        select_mixed_learning_windows(_schedule(1), set(), updates=2)
