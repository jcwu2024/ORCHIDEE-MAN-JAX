"""Machine audit for true-daily water, carbon, and energy labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

AVAILABILITY_CLASSES = frozenset(
    {"present", "exactly_derivable", "missing_non_identifiable"}
)
DOMAINS = frozenset({"water", "carbon", "energy"})
REQUIRED_DOMAIN_ROLES = frozenset(
    {"inventory_endpoint", "external_input", "external_output", "internal_transfer", "budget_residual"}
)
CONTRACT_COLLECTIONS = frozenset(
    {"state_leaves", "fast_day_target_leaves", "diagnostic_leaves", "native_forcing"}
)
DEFAULT_INVENTORY_PATH = (
    Path(__file__).resolve().parents[2]
    / "manifests"
    / "coarse_graining"
    / "daily_flux_label_inventory_v1.json"
)


@dataclass(frozen=True)
class ContractReference:
    collection: str
    key: str


@dataclass(frozen=True)
class DailyFluxLabel:
    label_id: str
    domain: str
    role: str
    units: str
    availability: str
    required_for: tuple[str, ...]
    source_owner: str
    contract_refs: tuple[ContractReference, ...]
    dependencies: tuple[str, ...]
    derivation: str | None
    capture: Mapping[str, Any] | None
    reason: str | None


@dataclass(frozen=True)
class DailyFluxLabelInventory:
    schema_version: str
    status: str
    teacher_contract_schema: str
    teacher_contract_sha256: str
    capture_family_ids: tuple[str, ...]
    labels: tuple[DailyFluxLabel, ...]
    source_path: Path

    @property
    def labels_by_id(self) -> dict[str, DailyFluxLabel]:
        return {label.label_id: label for label in self.labels}


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _text(mapping: Mapping[str, Any], key: str, *, label: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} requires nonempty {key}")
    return value


def load_daily_flux_label_inventory(
    path: str | Path = DEFAULT_INVENTORY_PATH,
) -> DailyFluxLabelInventory:
    """Load and fail-closed validate the Gate C label inventory."""

    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "daily_flux_label_inventory_v1":
        raise ValueError("unsupported daily flux label inventory schema")
    teacher = _mapping(payload.get("teacher_contract"), label="teacher_contract")
    availability = tuple(str(item) for item in payload.get("availability_classes", ()))
    if set(availability) != AVAILABILITY_CLASSES:
        raise ValueError("availability_classes must match the frozen three-way classification")

    capture_families = tuple(
        _text(_mapping(raw, label="capture family"), "id", label="capture family")
        for raw in payload.get("capture_families", ())
    )
    if not capture_families or len(set(capture_families)) != len(capture_families):
        raise ValueError("capture family IDs must be nonempty and unique")

    labels: list[DailyFluxLabel] = []
    for raw_value in payload.get("labels", ()):
        raw = _mapping(raw_value, label="daily flux label")
        availability_class = _text(raw, "availability", label="daily flux label")
        if availability_class not in AVAILABILITY_CLASSES:
            raise ValueError(f"unknown availability class {availability_class!r}")
        domain = _text(raw, "domain", label="daily flux label")
        if domain not in DOMAINS:
            raise ValueError(f"unknown daily flux domain {domain!r}")
        refs = tuple(
            ContractReference(
                collection=_text(ref, "collection", label="contract reference"),
                key=_text(ref, "key", label="contract reference"),
            )
            for ref_value in raw.get("contract_refs", ())
            for ref in (_mapping(ref_value, label="contract reference"),)
        )
        if any(ref.collection not in CONTRACT_COLLECTIONS for ref in refs):
            raise ValueError("contract reference uses an unknown collection")
        derivation = str(raw.get("derivation", "")).strip() or None
        capture_raw = raw.get("capture")
        capture = None if capture_raw is None else dict(_mapping(capture_raw, label="capture"))
        reason = str(raw.get("reason", "")).strip() or None
        if availability_class == "present" and not refs:
            raise ValueError("present labels require at least one contract reference")
        if availability_class == "exactly_derivable" and not derivation:
            raise ValueError("exactly_derivable labels require an explicit derivation")
        if availability_class == "missing_non_identifiable":
            if capture is None or not reason:
                raise ValueError("missing labels require capture metadata and a non-identifiability reason")
            family = str(capture.get("family", ""))
            if family not in capture_families:
                raise ValueError(f"missing label references unknown capture family {family!r}")
            if not capture.get("source_terms") or not str(capture.get("aggregation", "")).strip():
                raise ValueError("missing label capture requires source terms and daily aggregation")
        labels.append(
            DailyFluxLabel(
                label_id=_text(raw, "id", label="daily flux label"),
                domain=domain,
                role=_text(raw, "role", label="daily flux label"),
                units=_text(raw, "units", label="daily flux label"),
                availability=availability_class,
                required_for=tuple(str(item) for item in raw.get("required_for", ())),
                source_owner=_text(raw, "source_owner", label="daily flux label"),
                contract_refs=refs,
                dependencies=tuple(str(item) for item in raw.get("dependencies", ())),
                derivation=derivation,
                capture=capture,
                reason=reason,
            )
        )
    ids = tuple(label.label_id for label in labels)
    if not labels or len(set(ids)) != len(ids):
        raise ValueError("daily flux label IDs must be nonempty and unique")
    id_set = set(ids)
    for label in labels:
        unknown = sorted(set(label.dependencies) - id_set)
        if unknown:
            raise ValueError(f"{label.label_id} has unknown dependencies {unknown}")
        if label.label_id in label.dependencies:
            raise ValueError(f"{label.label_id} cannot depend on itself")
    for domain in DOMAINS:
        roles = {label.role for label in labels if label.domain == domain}
        required = REQUIRED_DOMAIN_ROLES - ({"inventory_endpoint"} if domain == "energy" else set())
        missing_roles = sorted(required - roles)
        if missing_roles:
            raise ValueError(f"{domain} inventory is missing roles {missing_roles}")
    _topological_label_ids(labels)
    return DailyFluxLabelInventory(
        schema_version="daily_flux_label_inventory_v1",
        status=_text(payload, "status", label="inventory"),
        teacher_contract_schema=_text(teacher, "schema_version", label="teacher_contract"),
        teacher_contract_sha256=_text(teacher, "sha256", label="teacher_contract"),
        capture_family_ids=capture_families,
        labels=tuple(labels),
        source_path=source_path,
    )


def _topological_label_ids(labels: list[DailyFluxLabel] | tuple[DailyFluxLabel, ...]) -> tuple[str, ...]:
    by_id = {label.label_id: label for label in labels}
    visiting: set[str] = set()
    visited: set[str] = set()
    result: list[str] = []

    def visit(label_id: str) -> None:
        if label_id in visited:
            return
        if label_id in visiting:
            raise ValueError(f"daily flux derivation cycle includes {label_id}")
        visiting.add(label_id)
        for dependency in by_id[label_id].dependencies:
            visit(dependency)
        visiting.remove(label_id)
        visited.add(label_id)
        result.append(label_id)

    for label_id in by_id:
        visit(label_id)
    return tuple(result)


def _contract_key_sets(contract_metadata: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for collection in ("state_leaves", "fast_day_target_leaves"):
        result[collection] = {str(item["key"]) for item in contract_metadata.get(collection, ())}
    result["diagnostic_leaves"] = {
        str(item["name"]) for item in contract_metadata.get("diagnostic_leaves", ())
    }
    forcing = _mapping(contract_metadata.get("native_forcing"), label="native_forcing")
    result["native_forcing"] = {str(item) for item in forcing.get("fields", ())}
    return result


def audit_daily_flux_label_inventory(
    inventory: DailyFluxLabelInventory,
    contract_metadata: Mapping[str, Any],
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    """Audit inventory references and readiness against one v5 contract."""

    if contract_metadata.get("schema_version") != inventory.teacher_contract_schema:
        raise ValueError("Teacher contract schema does not match the label inventory")
    if contract_sha256 != inventory.teacher_contract_sha256:
        raise ValueError("Teacher contract hash does not match the frozen label inventory")
    keys = _contract_key_sets(contract_metadata)
    missing_refs = []
    for label in inventory.labels:
        for ref in label.contract_refs:
            if ref.key not in keys[ref.collection]:
                missing_refs.append(
                    {"label_id": label.label_id, "collection": ref.collection, "key": ref.key}
                )
    if missing_refs:
        raise ValueError(f"daily flux inventory references absent Teacher fields: {missing_refs}")

    by_id = inventory.labels_by_id
    ready_now: dict[str, bool] = {}
    ready_after_capture: dict[str, bool] = {}
    for label_id in _topological_label_ids(inventory.labels):
        label = by_id[label_id]
        if label.availability == "present":
            ready_now[label_id] = True
            ready_after_capture[label_id] = True
        elif label.availability == "missing_non_identifiable":
            ready_now[label_id] = False
            ready_after_capture[label_id] = True
        else:
            ready_now[label_id] = all(ready_now[item] for item in label.dependencies)
            ready_after_capture[label_id] = all(
                ready_after_capture[item] for item in label.dependencies
            )

    counts = {
        domain: {
            availability: sum(
                label.domain == domain and label.availability == availability
                for label in inventory.labels
            )
            for availability in sorted(AVAILABILITY_CLASSES)
        }
        for domain in sorted(DOMAINS)
    }
    missing = [
        label for label in inventory.labels if label.availability == "missing_non_identifiable"
    ]
    capture = {
        family: [label.label_id for label in missing if label.capture["family"] == family]
        for family in inventory.capture_family_ids
    }
    state_update_ids = [
        label.label_id for label in inventory.labels if "state_update" in label.required_for
    ]
    budget_ids = [
        label.label_id for label in inventory.labels if "budget" in label.required_for
    ]
    return {
        "schema_version": "daily_flux_label_inventory_audit_v1",
        "inventory_schema_version": inventory.schema_version,
        "inventory_status": inventory.status,
        "teacher_contract_schema": inventory.teacher_contract_schema,
        "teacher_contract_sha256": contract_sha256,
        "label_count": len(inventory.labels),
        "counts_by_domain_and_availability": counts,
        "missing_label_count": len(missing),
        "minimal_supplemental_capture": capture,
        "existing_shards_state_update_ready": all(ready_now[item] for item in state_update_ids),
        "existing_shards_budget_ready": all(ready_now[item] for item in budget_ids),
        "ready_after_declared_capture": all(ready_after_capture.values()),
        "decision": "capture_three_daily_reduced_families_then_run_non_neural_replay",
        "forbidden": [
            "derive_process_fluxes_from_endpoint_differences",
            "regenerate_669_shards_before_minimal_capture_gate",
            "store_48_step_trajectory_as_final_surrogate_input",
            "start_neural_training_before_true_label_replay",
        ],
    }
