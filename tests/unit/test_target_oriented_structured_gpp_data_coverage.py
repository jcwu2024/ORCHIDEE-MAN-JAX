from dataclasses import dataclass

from scripts.hpc.audit_target_oriented_structured_gpp_data_coverage import (
    _direct_day_keys,
    _select_monitor_entries,
)


def _record(landpoint, year, parent, typed, rows):
    return {
        "landpoint_id": landpoint,
        "year": year,
        "parent_shard_sha256": parent,
        "typed_shard_sha256": typed,
        "start_rows": rows,
    }


def test_structured_gpp_coverage_expands_direct_days_by_component_horizon():
    entry = {
        "components": {
            "one_day": [_record("a", 2000, "p1", "t1", [2, 8])],
            "seven_day": [_record("b", 2000, "p2", "t2", [3])],
            "thirty_day": [_record("c", 2000, "p3", "t3", [4])],
        }
    }
    keys = _direct_day_keys(entry)
    assert {value[-1] for value in keys["one_day"]} == {2, 8}
    assert {value[-1] for value in keys["seven_day"]} == set(range(3, 10))
    assert {value[-1] for value in keys["thirty_day"]} == set(range(4, 34))


@dataclass(frozen=True)
class _Parent:
    landpoint_id: str
    year: int
    sha256: str


@dataclass(frozen=True)
class _Reference:
    parent: _Parent
    sha256: str
    day_count: int


def test_structured_gpp_monitor_is_point_year_disjoint_and_landpoint_distinct():
    references = tuple(
        _Reference(_Parent(f"p{index}", 2000 + index % 2, f"parent-{index}"), f"typed-{index}", 40)
        for index in range(6)
    )
    exposed = {("p0", 2000, "parent-0", "typed-0")}
    selected = _select_monitor_entries(
        references,
        exposed_point_years=exposed,
        point_year_count=4,
        days_per_point_year=3,
        seed=7,
    )
    identities = {
        (
            value["landpoint_id"],
            value["year"],
            value["parent_shard_sha256"],
            value["typed_shard_sha256"],
        )
        for value in selected
    }
    assert len(selected) == len({value["landpoint_id"] for value in selected}) == 4
    assert identities.isdisjoint(exposed)
    assert all(len(value["monitor_rows"]) == 3 for value in selected)
    assert all(min(value["monitor_rows"]) >= 1 for value in selected)
