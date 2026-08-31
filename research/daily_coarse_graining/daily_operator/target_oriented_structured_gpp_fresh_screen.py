"""Fresh independent 7/30-day screen selection for the broad GPP stage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_training import _rank, _start_rows

SCHEMA_VERSION = "target_oriented_structured_gpp_fresh_screen_v1"
FROZEN_CONTRACT_SHA256 = "894a0b2dc329179566e624e7c04b3a63522588aed34feb8dc00e3187eb73e6a7"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_fresh_screen_v1.json"
)
SLICE_SPLITS = {
    "temporal": ("train", "validation"),
    "spatial": ("validation", "train"),
    "joint": ("validation", "validation"),
}
HORIZONS = (7, 30)


def canonical_sha256(payload: Mapping[str, Any], *, hash_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(hash_key, None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StructuredGPPFreshScreenContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def selection(self) -> Mapping[str, Any]:
        return self.payload["selection"]


def load_structured_gpp_fresh_screen_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPFreshScreenContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = canonical_sha256(payload, hash_key="contract_sha256")
    identity = payload.get("identity", {})
    selection = payload.get("selection", {})
    evaluation = payload.get("evaluation", {})
    execution = payload.get("execution", {})
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_fresh_panel_targets_are_read"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("consumed_validation_panel_sha256")
        != "f4952f5fa106aae107fcee6595a6b69897c67b0a54c3a17d3323abc7ed933e79"
        or identity.get("selected_broad_checkpoint_sha256")
        != "3b173b8980c37b49db3d1547dddad2ab98ec00c733e2f564469490ee382d4ec0"
        or int(identity.get("selected_broad_checkpoint_update", -1)) != 1000
        or tuple(selection.get("slices", ())) != tuple(SLICE_SPLITS)
        or tuple(int(value) for value in selection.get("horizons_days", ()))
        != HORIZONS
        or int(selection.get("windows_per_slice_per_horizon", -1)) != 16
        or selection.get("exclude_consumed_point_years_within_slice") is not True
        or selection.get("distinct_landpoints_within_cell") is not True
        or int(evaluation.get("formal_window_count", -1)) != 96
        or int(evaluation.get("admission_window_count", -1)) != 6
        or execution.get("teacher_data_regeneration_allowed") is not False
        or execution.get("historical_v5_allowed") is not False
        or execution.get("sealed_test_allowed") is not False
        or int(execution.get("optimizer_updates", -1)) != 0
        or int(execution.get("checkpoint_writes", -1)) != 0
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
    ):
        raise ValueError("structured GPP fresh-screen contract drift")
    return StructuredGPPFreshScreenContract(payload, source, observed)


def _reference_identity(reference: Any) -> tuple[str, int, str, str]:
    return (
        str(reference.parent.landpoint_id),
        int(reference.parent.year),
        str(reference.parent.sha256),
        str(reference.sha256),
    )


def _consumed_point_years(
    panel: Mapping[str, Any],
    *,
    slice_id: str,
) -> set[tuple[str, int]]:
    return {
        (str(entry["landpoint_id"]), int(entry["year"]))
        for horizon in HORIZONS
        for entry in panel["slices"][slice_id]["horizons"][str(horizon)]
    }


def build_fresh_validation_panel(
    index: Any,
    contract: StructuredGPPFreshScreenContract,
    target_contract: Any,
    consumed_panel: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select metadata-only windows without consulting any target values."""

    identity = contract.payload["identity"]
    selection = contract.selection
    if (
        consumed_panel.get("panel_sha256")
        != identity["consumed_validation_panel_sha256"]
        or canonical_sha256(consumed_panel, hash_key="panel_sha256")
        != consumed_panel.get("panel_sha256")
        or consumed_panel.get("sealed_test_used") is not False
    ):
        raise ValueError("consumed validation panel identity drift")

    seed = int(selection["selection_seed"])
    per_cell = int(selection["windows_per_slice_per_horizon"])
    panel_slices = {}
    for slice_id, (spatial, temporal) in SLICE_SPLITS.items():
        consumed = _consumed_point_years(consumed_panel, slice_id=slice_id)
        references = tuple(
            reference
            for reference in index.select(
                spatial_split=spatial,
                temporal_split=temporal,
            )
            if (str(reference.parent.landpoint_id), int(reference.parent.year))
            not in consumed
        )
        if not references or any(
            "test" in (reference.parent.spatial_split, reference.parent.temporal_split)
            for reference in references
        ):
            raise ValueError(f"fresh validation slice drift: {slice_id}")

        horizon_entries = {}
        for horizon in HORIZONS:
            ranked = sorted(
                (
                    reference
                    for reference in references
                    if int(reference.day_count) >= horizon + 1
                ),
                key=lambda reference: (
                    _rank(
                        seed,
                        horizon,
                        ("fresh_validation", slice_id, *_reference_identity(reference)),
                    ),
                    _reference_identity(reference),
                ),
            )
            selected = []
            seen_landpoints = set()
            for reference in ranked:
                landpoint = str(reference.parent.landpoint_id)
                if landpoint in seen_landpoints:
                    continue
                seen_landpoints.add(landpoint)
                selected.append(reference)
                if len(selected) == per_cell:
                    break
            if len(selected) != per_cell:
                raise ValueError(
                    f"fresh validation slice has too few landpoints: {slice_id}/{horizon}"
                )
            entries = []
            for reference in selected:
                landpoint, year, parent_sha256, typed_sha256 = _reference_identity(
                    reference
                )
                entries.append(
                    {
                        "landpoint_id": landpoint,
                        "year": year,
                        "parent_shard_sha256": parent_sha256,
                        "typed_shard_sha256": typed_sha256,
                        "day_count": int(reference.day_count),
                        "horizon_days": horizon,
                        "start_rows": list(
                            _start_rows(
                                reference,
                                seed=seed,
                                update=horizon,
                                component=f"fresh_validation/{slice_id}",
                                count=1,
                                horizon=horizon,
                                exclude_first=True,
                            )
                        ),
                    }
                )
            horizon_entries[str(horizon)] = entries
        panel_slices[slice_id] = {
            "spatial_split": spatial,
            "temporal_split": temporal,
            "excluded_consumed_point_year_count": len(consumed),
            "horizons": horizon_entries,
        }

    panel = {
        "schema_version": "target_oriented_fresh_validation_panel_v1",
        "screen_contract_sha256": contract.sha256,
        "target_contract_sha256": target_contract.sha256,
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "consumed_validation_panel_sha256": identity[
            "consumed_validation_panel_sha256"
        ],
        "selection_seed": seed,
        "windows_per_slice_per_horizon": per_cell,
        "target_values_read_during_selection": False,
        "sealed_test_used": False,
        "slices": panel_slices,
    }
    panel["panel_sha256"] = canonical_sha256(panel, hash_key="panel_sha256")
    return panel


def verify_fresh_validation_panel(
    panel: Mapping[str, Any],
    index: Any,
    contract: StructuredGPPFreshScreenContract,
    target_contract: Any,
    consumed_panel: Mapping[str, Any],
) -> None:
    if panel.get("panel_sha256") != canonical_sha256(
        panel,
        hash_key="panel_sha256",
    ):
        raise ValueError("fresh validation panel hash drift")
    expected = build_fresh_validation_panel(
        index,
        contract,
        target_contract,
        consumed_panel,
    )
    if dict(panel) != dict(expected):
        raise ValueError("fresh validation panel selection drift")


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "HORIZONS",
    "SCHEMA_VERSION",
    "SLICE_SPLITS",
    "StructuredGPPFreshScreenContract",
    "build_fresh_validation_panel",
    "canonical_sha256",
    "load_structured_gpp_fresh_screen_contract",
    "verify_fresh_validation_panel",
]
