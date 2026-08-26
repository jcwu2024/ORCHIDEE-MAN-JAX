"""Frozen component-resolved flux-basis amendment for target-oriented training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "target_oriented_flux_basis_amendment_v1"
FROZEN_CONTRACT_SHA256 = "6c81e8f56c2c81fc2ac024a71da98f3e875c4ebc429f9e3d1ff4e8e3ef6bfb45"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_flux_basis_amendment_v1.json"
)

LEGACY_AGGREGATE_FLUX_BASIS = (
    "gpp",
    "autotrophic_respiration",
    "heterotrophic_respiration",
)
COMPONENT_RESOLVED_FLUX_BASIS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
    "heterotrophic_respiration",
)
DERIVED_FLUX_IDENTITIES = (
    "autotrophic_respiration",
    "npp",
    "nep",
    "nee",
)
INDEPENDENT_STOCK_BASIS = (
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "nonliving_ecosystem_carbon",
)


def contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TargetOrientedFluxBasisAmendment:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def flux_basis(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.payload["flux_basis"]["component_resolved_basis"])

    @property
    def stock_basis(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.payload["unchanged_boundary"]["stock_basis"])


def _validate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("target-oriented flux-basis amendment schema drift")
    if payload.get("status") != "frozen_before_zero_update_geometry_audit":
        raise ValueError("target-oriented flux-basis amendment is not frozen")
    identity = payload.get("identity", {})
    if (
        identity.get("parent_target_contract_sha256")
        != "7c9857c0ca57ee65cfcf7514a216a217c3b1661f6cc869c6dfb85d3786bfce5c"
        or identity.get("parent_training_contract_sha256")
        != "91faf01123008a60c59038c074779a2a2b4553b82c3739475dd4829fe91ea25b"
        or identity.get("update_1000_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
        or identity.get("diagnostic_report_sha256")
        != "1de8962a64aad6a7b76cf311f1dd14932780c67691ec472fbc342c5c8eb42fbe"
        or identity.get("candidate") != "target_oriented_hierarchical_coordinate_attention_v1"
        or int(identity.get("parameter_count", 0)) != 83_755_173
    ):
        raise ValueError("target-oriented flux-basis amendment identity drift")
    data = payload.get("data", {})
    if data.get("release_id") != "pft14-daily-teacher-v6" or data.get("split") != "train/train":
        raise ValueError("target-oriented flux-basis amendment data identity drift")
    if any(
        bool(data.get(name))
        for name in (
            "teacher_regeneration_allowed",
            "historical_v5_allowed",
            "sealed_test_allowed",
        )
    ):
        raise ValueError("target-oriented flux-basis amendment opens a forbidden data path")
    basis = payload.get("flux_basis", {})
    if (
        tuple(basis.get("superseded_aggregate_basis", ())) != LEGACY_AGGREGATE_FLUX_BASIS
        or tuple(basis.get("component_resolved_basis", ())) != COMPONENT_RESOLVED_FLUX_BASIS
        or tuple(basis.get("derived_report_only", ())) != DERIVED_FLUX_IDENTITIES
        or basis.get("component_weighting") != "equal_active_independent_fields"
        or basis.get("flux_vs_stock_weighting") != "unchanged_equal_macro_groups"
    ):
        raise ValueError("target-oriented component-resolved flux basis drift")
    unchanged = payload.get("unchanged_boundary", {})
    if (
        tuple(unchanged.get("stock_basis", ())) != INDEPENDENT_STOCK_BASIS
        or any(
            bool(unchanged.get(name))
            for name in (
                "architecture_changed",
                "operator_boundary_changed",
                "output_schema_changed",
            )
        )
        or (
            float(unchanged.get("one_day_state_anchor_coefficient", 0.0)),
            float(unchanged.get("seven_day_science_coefficient", 0.0)),
            float(unchanged.get("thirty_day_science_coefficient", 0.0)),
            float(unchanged.get("necessary_process_supervision_coefficient", 0.0)),
        )
        != (1.0, 1.0, 1.0, 0.1)
    ):
        raise ValueError("target-oriented flux-basis unchanged boundary drift")
    gate = payload.get("zero_update_gate", {})
    if (
        tuple(gate.get("audit_updates", ())) != (251, 358, 465, 572, 679, 786, 893, 1000)
        or int(gate.get("required_parameter_coverage", 0)) != 83_755_173
        or int(gate.get("required_finite_gradient_sources", 0)) != 5
        or gate.get("required_zero_physical_hard_failures") is not True
        or gate.get("optimizer_update_retained") is not False
        or gate.get("checkpoint_written") is not False
    ):
        raise ValueError("target-oriented flux-basis zero-update gate drift")


def load_target_oriented_flux_basis_amendment(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedFluxBasisAmendment:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("target-oriented flux-basis amendment SHA256 drift")
    if observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("target-oriented flux-basis amendment is not frozen v1")
    return TargetOrientedFluxBasisAmendment(payload, source, observed)


__all__ = [
    "COMPONENT_RESOLVED_FLUX_BASIS",
    "DEFAULT_CONTRACT_PATH",
    "DERIVED_FLUX_IDENTITIES",
    "FROZEN_CONTRACT_SHA256",
    "INDEPENDENT_STOCK_BASIS",
    "LEGACY_AGGREGATE_FLUX_BASIS",
    "SCHEMA_VERSION",
    "TargetOrientedFluxBasisAmendment",
    "contract_sha256",
    "load_target_oriented_flux_basis_amendment",
]
