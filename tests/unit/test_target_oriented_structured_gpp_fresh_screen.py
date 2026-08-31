from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
    load_target_oriented_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_screen import (
    StructuredGPPFreshScreenContract,
    build_fresh_validation_panel,
    canonical_sha256,
    load_structured_gpp_fresh_screen_contract,
    verify_fresh_validation_panel,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    ordered_target_oriented_validation_entries,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_fresh_screen_v1.json"
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


def _consumed_panel(references):
    slices = {}
    for slice_id in ("temporal", "spatial", "joint"):
        available = [
            value
            for value in references
            if value.parent.landpoint_id.startswith(f"{slice_id}-")
        ]
        horizons = {}
        for horizon, reference in zip((7, 30), available[:2], strict=True):
            horizons[str(horizon)] = [
                {
                    "landpoint_id": reference.parent.landpoint_id,
                    "year": reference.parent.year,
                    "parent_shard_sha256": reference.parent.sha256,
                    "typed_shard_sha256": reference.sha256,
                    "day_count": reference.day_count,
                    "horizon_days": horizon,
                    "start_rows": [1],
                }
            ]
        slices[slice_id] = {"horizons": horizons}
    panel = {"sealed_test_used": False, "slices": slices}
    panel["panel_sha256"] = canonical_sha256(panel, hash_key="panel_sha256")
    return panel


def test_fresh_screen_contract_is_frozen_before_target_read():
    contract = load_structured_gpp_fresh_screen_contract(CONTRACT)
    assert contract.sha256 == contract.payload["contract_sha256"]
    assert contract.selection["selection_uses_metadata_only"] is True
    assert contract.payload["evaluation"]["formal_window_count"] == 96
    assert contract.payload["evaluation"]["direct_teacher_state_gpp"]["report_only"] is True
    assert contract.payload["acceptance"]["relative_comparison_or_control_required"] is False


def test_fresh_panel_excludes_consumed_point_years_and_is_deterministic():
    references = []
    splits = {
        "temporal": ("train", "validation"),
        "spatial": ("validation", "train"),
        "joint": ("validation", "validation"),
    }
    for slice_id, (spatial, temporal) in splits.items():
        references.extend(
            Reference(
                Parent(
                    landpoint_id=f"{slice_id}-{ordinal:02d}",
                    year=2000 + ordinal,
                    sha256=f"parent-{slice_id}-{ordinal}",
                    spatial_split=spatial,
                    temporal_split=temporal,
                ),
                sha256=f"typed-{slice_id}-{ordinal}",
            )
            for ordinal in range(24)
        )
    index = Index(references)
    consumed = _consumed_panel(references)
    frozen = load_structured_gpp_fresh_screen_contract(CONTRACT)
    payload = dict(frozen.payload)
    payload["identity"] = dict(
        payload["identity"],
        consumed_validation_panel_sha256=consumed["panel_sha256"],
    )
    contract = StructuredGPPFreshScreenContract(payload, frozen.source_path, frozen.sha256)
    target_contract = load_target_oriented_contract()

    first = build_fresh_validation_panel(index, contract, target_contract, consumed)
    second = build_fresh_validation_panel(index, contract, target_contract, consumed)
    assert first == second
    assert first["target_values_read_during_selection"] is False
    assert len(ordered_target_oriented_validation_entries(first)) == 96
    for slice_id in splits:
        consumed_point_years = {
            (entry["landpoint_id"], entry["year"])
            for horizon in (7, 30)
            for entry in consumed["slices"][slice_id]["horizons"][str(horizon)]
        }
        for horizon in (7, 30):
            entries = first["slices"][slice_id]["horizons"][str(horizon)]
            assert len(entries) == 16
            assert len({entry["landpoint_id"] for entry in entries}) == 16
            assert not {
                (entry["landpoint_id"], entry["year"]) for entry in entries
            } & consumed_point_years
    verify_fresh_validation_panel(
        first,
        index,
        contract,
        target_contract,
        consumed,
    )


def test_fresh_screen_launchers_use_only_free_nodes_and_selected_broad_checkpoint():
    prepare = (
        ROOT
        / "scripts"
        / "hpc"
        / "prepare_target_oriented_structured_gpp_fresh_validation.sh"
    ).read_text(encoding="utf-8")
    evaluate = (
        ROOT
        / "scripts"
        / "hpc"
        / "run_target_oriented_structured_gpp_fresh_screen.sh"
    ).read_text(encoding="utf-8")
    assert "test01|test02|test03|test04" in prepare
    assert "sbatch" not in prepare
    assert "sbatch" not in evaluate
    assert "target-oriented-structured-gpp-broad-14883081" in evaluate
    assert "stage-update-1000.pkl" in evaluate
    assert "--fresh-validation-prepared-manifest" in evaluate
