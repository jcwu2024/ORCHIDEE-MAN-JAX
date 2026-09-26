from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
    load_target_oriented_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    TargetOrientedAcceptanceScales,
    build_target_oriented_validation_panel,
    finalize_target_oriented_acceptance_statistics,
    load_target_oriented_training_contract,
    ordered_target_oriented_validation_entries,
    target_oriented_acceptance_sufficient_statistics,
    verify_target_oriented_validation_panel,
)
from scripts.hpc.target_oriented_training_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedTargetOrientedUpdateArtifact,
    atomic_json,
    load_prepared_target_manifest,
    load_prepared_target_manifest_range,
    write_prepared_target_artifact,
)


def _trajectory(value: float, days: int = 30) -> MaskedTrajectory:
    return MaskedTrajectory(
        value=jnp.full((days, 1), value, dtype=jnp.float64),
        defined=jnp.ones((days, 1), dtype=jnp.bool_),
    )


def test_target_oriented_acceptance_reports_all_derived_science_keys():
    fluxes = {
        "gpp": _trajectory(5.0),
        "maintenance_respiration": _trajectory(1.0),
        "growth_respiration": _trajectory(0.5),
        "heterotrophic_respiration": _trajectory(0.75),
        "npp": _trajectory(999.0),
    }
    stocks = {
        "aboveground_biomass_carbon": _trajectory(10.0),
        "belowground_biomass_carbon": _trajectory(4.0),
        "total_live_biomass_carbon": _trajectory(14.0),
        "ecosystem_carbon": _trajectory(20.0),
    }
    scales = TargetOrientedAcceptanceScales(
        fluxes={
            "gpp": 1.0,
            "maintenance_respiration": 1.0,
            "growth_respiration": 1.0,
            "autotrophic_respiration": 1.0,
            "npp": 1.0,
            "heterotrophic_respiration": 1.0,
            "nep": 1.0,
            "nee": 1.0,
        },
        stocks={
            "aboveground_biomass_carbon": 1.0,
            "belowground_biomass_carbon": 1.0,
            "total_live_biomass_carbon": 1.0,
            "nonliving_ecosystem_carbon": 1.0,
            "ecosystem_carbon": 1.0,
        },
    )
    statistics = target_oriented_acceptance_sufficient_statistics(
        fluxes,
        fluxes,
        stocks,
        stocks,
        scales=scales,
        horizon=7,
    )
    statistics = statistics._replace(
        fluxes=dict(sorted(statistics.fluxes.items())),
        stocks=dict(sorted(statistics.stocks.items())),
    )
    report = finalize_target_oriented_acceptance_statistics(
        statistics,
        horizon=7,
        target_contract=load_target_oriented_contract(),
    )

    assert tuple(report["keys"]) == (
        "gpp",
        "maintenance_respiration",
        "growth_respiration",
        "autotrophic_respiration",
        "npp",
        "heterotrophic_respiration",
        "nep",
        "nee",
        "aboveground_biomass_carbon",
        "belowground_biomass_carbon",
        "total_live_biomass_carbon",
        "nonliving_ecosystem_carbon",
        "ecosystem_carbon",
    )
    assert report["all_key_gates_passed"] is True
    assert report["science_macro_normalized_rmse"] == 0.0


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
    day_count: int = 364


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


def test_target_oriented_validation_panel_is_fixed_diverse_and_non_test():
    references = []
    for slice_id, (spatial, temporal) in {
        "temporal": ("train", "validation"),
        "spatial": ("validation", "train"),
        "joint": ("validation", "validation"),
    }.items():
        references.extend(
            Reference(
                parent=Parent(
                    landpoint_id=f"{slice_id}-point-{index:02d}",
                    year=2001 + index % 3,
                    sha256=f"{slice_id}-parent-{index:02d}",
                    spatial_split=spatial,
                    temporal_split=temporal,
                ),
                sha256=f"{slice_id}-typed-{index:02d}",
            )
            for index in range(20)
        )
    index = Index(references)
    training_contract = load_target_oriented_training_contract()
    target_contract = load_target_oriented_contract()

    first = build_target_oriented_validation_panel(index, training_contract, target_contract)
    second = build_target_oriented_validation_panel(index, training_contract, target_contract)

    assert first == second
    assert first["sealed_test_used"] is False
    assert len(ordered_target_oriented_validation_entries(first)) == 96
    for slice_id in ("temporal", "spatial", "joint"):
        for horizon in (7, 30):
            entries = first["slices"][slice_id]["horizons"][str(horizon)]
            assert len(entries) == 16
            assert len({entry["landpoint_id"] for entry in entries}) == 16
            assert all(entry["start_rows"][0] >= 1 for entry in entries)
    verify_target_oriented_validation_panel(first, index, training_contract, target_contract)
