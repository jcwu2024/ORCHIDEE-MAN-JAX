from __future__ import annotations

import json
from dataclasses import dataclass

from research.daily_coarse_graining.daily_operator.conditioned_independent_panel import (
    consumed_point_years,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_screen import (
    canonical_sha256,
)
from scripts.hpc.prepare_fresh_method_validation import (
    CONSUMED_PANEL_NAMES,
    HORIZONS,
    SLICE_SPLITS,
    build_fresh_method_panel,
    load_consumed_panels,
)


@dataclass(frozen=True)
class Parent:
    landpoint_id: str
    year: int
    sha256: str
    spatial_split: str
    temporal_split: str


@dataclass(frozen=True)
class Reference:
    parent: Parent
    sha256: str
    day_count: int = 365


class Index:
    def __init__(self, references):
        self.references = tuple(references)

    def select(self, *, spatial_split: str, temporal_split: str):
        return tuple(
            reference
            for reference in self.references
            if reference.parent.spatial_split == spatial_split
            and reference.parent.temporal_split == temporal_split
        )


def _fixtures():
    references = []
    for slice_id, (spatial, temporal) in SLICE_SPLITS.items():
        references.extend(
            Reference(
                Parent(
                    landpoint_id=f"{slice_id}-{ordinal:03d}",
                    year=1961 + ordinal,
                    sha256=f"parent-{slice_id}-{ordinal}",
                    spatial_split=spatial,
                    temporal_split=temporal,
                ),
                sha256=f"typed-{slice_id}-{ordinal}",
            )
            for ordinal in range(80)
        )
    return Index(references), references


def _consumed_panel(references, offset: int):
    slices = {}
    for slice_id in SLICE_SPLITS:
        available = [
            value
            for value in references
            if value.parent.landpoint_id.startswith(f"{slice_id}-")
        ]
        slices[slice_id] = {
            "horizons": {
                str(horizon): [
                    {
                        "landpoint_id": value.parent.landpoint_id,
                        "year": value.parent.year,
                        "parent_shard_sha256": value.parent.sha256,
                        "typed_shard_sha256": value.sha256,
                        "day_count": value.day_count,
                        "horizon_days": horizon,
                        "start_rows": [1],
                    }
                ]
                for horizon, value in zip(
                    HORIZONS,
                    available[offset : offset + 2],
                    strict=True,
                )
            }
        }
    panel = {"sealed_test_used": False, "slices": slices}
    panel["panel_sha256"] = canonical_sha256(panel, hash_key="panel_sha256")
    return panel


def test_fresh_method_panel_is_reproducible_and_has_eight_windows_per_cell():
    index, references = _fixtures()
    consumed = tuple(_consumed_panel(references, offset) for offset in (0, 2, 4, 6))
    first = build_fresh_method_panel(index, type("Target", (), {"sha256": "target"})(), consumed)
    second = build_fresh_method_panel(index, type("Target", (), {"sha256": "target"})(), consumed)

    assert first == second
    assert first["schema_version"] == "fresh_method_validation_panel_v1"
    assert first["method"]["method_id"] == "mixed_learning_fresh_litter_v1"
    assert first["selection_seed"] == 20260921
    assert first["target_values_read_during_selection"] is False
    assert first["sealed_test_used"] is False
    assert first["consumed_panel_count"] == 4
    assert first["panel_sha256"]

    rows = [
        entry
        for slice_id in SLICE_SPLITS
        for horizon in HORIZONS
        for entry in first["slices"][slice_id]["horizons"][str(horizon)]
    ]
    assert len(rows) == 48
    assert all(
        len(first["slices"][slice_id]["horizons"][str(horizon)]) == 8
        for slice_id in SLICE_SPLITS
        for horizon in HORIZONS
    )


def test_fresh_method_panel_excludes_union_and_preserves_validation_splits():
    index, references = _fixtures()
    consumed = tuple(_consumed_panel(references, offset) for offset in (0, 2, 4, 6))
    panel = build_fresh_method_panel(index, type("Target", (), {"sha256": "target"})(), consumed)
    excluded = set().union(*(consumed_point_years(value) for value in consumed))

    selected = set()
    for slice_id, (spatial, temporal) in SLICE_SPLITS.items():
        cell = panel["slices"][slice_id]
        assert (cell["spatial_split"], cell["temporal_split"]) == (spatial, temporal)
        for horizon in HORIZONS:
            for entry in cell["horizons"][str(horizon)]:
                point_year = (entry["landpoint_id"], entry["year"])
                assert point_year not in excluded
                assert point_year not in selected
                selected.add(point_year)

    assert [item["directory"] for item in panel["consumed_panels"]] == list(
        CONSUMED_PANEL_NAMES
    )


def test_consumed_loader_accepts_legacy_staged_entries_schema(tmp_path):
    staged = {
        "schema_version": "successor_training_validation_panel_v1",
        "sealed_test_used": False,
        "slices": {
            name: {
                "spatial_split": spatial,
                "temporal_split": temporal,
                "entries": [
                    {"landpoint_id": f"{name}-{i}", "year": 2000 + i}
                    for i in range(8)
                ],
            }
            for name, (spatial, temporal) in SLICE_SPLITS.items()
        },
    }
    staged["panel_sha256"] = canonical_sha256(staged, hash_key="panel_sha256")
    for name in CONSUMED_PANEL_NAMES:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "validation_panel.json").write_text(
            json.dumps(staged), encoding="utf-8"
        )

    loaded = load_consumed_panels(tmp_path)
    assert len(loaded) == 4
    assert [item["window_count"] for item in [
        {"window_count": sum(len(value["entries"]) for value in panel["slices"].values())}
        for panel in loaded
    ]] == [24] * 4
