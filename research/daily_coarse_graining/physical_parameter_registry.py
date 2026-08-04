"""Fail-closed loader for the source-driven physical-parameter registry."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "physical_parameter_candidate_registry_v1"
CLASSIFICATIONS = frozenset(
    {"inversion_candidate", "sensitivity_only", "discrete_control", "not_identifiable_now"}
)
MATHEMATICAL_DISCRETE_PREFIXES = ("logical_", "integer_")
ARCHITECTURE_OWNERSHIP = frozenset(
    {
        "retained_exact",
        "explicit_parameterized_learned_process",
        "learned_conditional",
        "configuration_only",
    }
)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = (
    ROOT / "manifests" / "coarse_graining" / "physical_parameter_candidate_registry_v1.json"
)
_SOURCE_SPAN_RE = re.compile(r"lines?\s+(\d+)(?:-(\d+))?", re.IGNORECASE)


@dataclass(frozen=True)
class PhysicalParameterCandidate:
    id: str
    run_def_keys: tuple[str, ...]
    component_count: int
    classification: str
    priority: int
    axis_owner: str
    mathematical_class: str
    architecture_ownership: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PhysicalParameterRegistry:
    schema_version: str
    status: str
    candidates: tuple[PhysicalParameterCandidate, ...]
    first_wave: tuple[str, ...]
    second_wave: tuple[str, ...]
    source_path: Path

    @property
    def by_id(self) -> dict[str, PhysicalParameterCandidate]:
        return {candidate.id: candidate for candidate in self.candidates}

    @property
    def classification_counts(self) -> Counter[str]:
        return Counter(candidate.classification for candidate in self.candidates)

    @property
    def scalar_component_count(self) -> int:
        return sum(candidate.component_count for candidate in self.candidates)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _text(mapping: Mapping[str, Any], key: str, *, label: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} requires nonempty {key}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sources(payload: Mapping[str, Any], *, root: Path) -> str:
    combined_source = []
    seen_paths: set[str] = set()
    for raw_value in payload.get("source_files", ()):
        raw = _mapping(raw_value, label="source file")
        relative = _text(raw, "path", label="source file")
        expected = _text(raw, "sha256", label="source file").lower()
        if relative in seen_paths:
            raise ValueError(f"duplicate source file {relative!r}")
        seen_paths.add(relative)
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing registry source file {relative!r}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"source hash drift for {relative!r}: {actual} != {expected}")
        combined_source.append(path.read_text(encoding="utf-8", errors="replace").upper())
    if not combined_source:
        raise ValueError("registry requires source files")
    return "\n".join(combined_source)


def _validate_valid_domain(candidate: Mapping[str, Any], *, label: str) -> None:
    domain = _mapping(candidate.get("valid_domain"), label=f"{label} valid_domain")
    _text(domain, "constraint", label=f"{label} valid_domain")
    _text(domain, "prior_status", label=f"{label} valid_domain")
    if "range" not in domain:
        return
    bounds = domain["range"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{label} range must contain two bounds")
    lower, upper = (float(value) for value in bounds)
    if not lower < upper:
        raise ValueError(f"{label} range must be strictly increasing")


def _validate_candidate(
    raw: Mapping[str, Any], *, source_text: str
) -> PhysicalParameterCandidate:
    candidate_id = _text(raw, "id", label="parameter")
    label = f"parameter {candidate_id!r}"
    keys = tuple(str(value).strip().upper() for value in raw.get("run_def_keys", ()))
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError(f"{label} requires unique run_def_keys")
    component_count = int(raw.get("component_count", 0))
    if component_count < len(keys):
        raise ValueError(f"{label} component_count cannot be smaller than run_def_keys")
    missing_source_keys = tuple(key for key in keys if key not in source_text)
    if missing_source_keys:
        raise ValueError(f"{label} source files do not contain {missing_source_keys}")

    classification = _text(raw, "classification", label=label)
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"{label} has unknown classification {classification!r}")
    ownership = _text(raw, "architecture_ownership", label=label)
    if ownership not in ARCHITECTURE_OWNERSHIP:
        raise ValueError(f"{label} has unknown architecture ownership {ownership!r}")
    mathematical_class = _text(raw, "mathematical_class", label=label)
    if classification == "discrete_control":
        if ownership != "configuration_only" or not mathematical_class.startswith(
            MATHEMATICAL_DISCRETE_PREFIXES
        ):
            raise ValueError(f"{label} discrete controls must be configuration-only and discrete")
    elif ownership == "configuration_only":
        raise ValueError(f"{label} continuous candidates cannot be configuration-only")

    priority = int(raw.get("priority", 0))
    if not 1 <= priority <= 5:
        raise ValueError(f"{label} priority must be in 1..5")
    for field in (
        "axis_owner",
        "applicability",
        "units",
        "fortran_parameter_owner",
        "fortran_process_consumer",
        "jax_owner",
        "lifecycle",
        "gradient_evidence",
    ):
        _text(raw, field, label=label)
    for field in ("expected_observables", "confounding_risks"):
        values = tuple(str(value).strip() for value in raw.get(field, ()) if str(value).strip())
        if not values:
            raise ValueError(f"{label} requires {field}")
    _validate_valid_domain(raw, label=label)
    owner = _text(raw, "fortran_parameter_owner", label=label)
    span = _SOURCE_SPAN_RE.search(owner)
    if "lines" in owner.lower() and span is None:
        raise ValueError(f"{label} has an invalid Fortran line span")

    return PhysicalParameterCandidate(
        id=candidate_id,
        run_def_keys=keys,
        component_count=component_count,
        classification=classification,
        priority=priority,
        axis_owner=_text(raw, "axis_owner", label=label),
        mathematical_class=mathematical_class,
        architecture_ownership=ownership,
        metadata=dict(raw),
    )


def load_physical_parameter_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH, *, root: str | Path = ROOT
) -> PhysicalParameterRegistry:
    """Load the registry and reject source drift or internally inconsistent claims."""

    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    source_text = _validate_sources(payload, root=Path(root).resolve())
    candidates = tuple(
        _validate_candidate(_mapping(value, label="parameter"), source_text=source_text)
        for value in payload.get("parameters", ())
    )
    ids = tuple(candidate.id for candidate in candidates)
    if not candidates or len(ids) != len(set(ids)):
        raise ValueError("parameter IDs must be nonempty and unique")
    all_keys = tuple(key for candidate in candidates for key in candidate.run_def_keys)
    if len(all_keys) != len(set(all_keys)):
        duplicates = tuple(key for key, count in Counter(all_keys).items() if count > 1)
        raise ValueError(f"run_def_keys must have one registry owner; duplicates={duplicates}")

    expected = _mapping(payload.get("expected_counts"), label="expected_counts")
    counts = Counter(candidate.classification for candidate in candidates)
    declared_counts = {
        str(key): int(value)
        for key, value in _mapping(
            expected.get("classification_counts"), label="classification_counts"
        ).items()
    }
    actual_counts = {name: counts[name] for name in CLASSIFICATIONS}
    if int(expected.get("parameter_families", -1)) != len(candidates):
        raise ValueError("parameter family count drift")
    if int(expected.get("scalar_components", -1)) != sum(
        candidate.component_count for candidate in candidates
    ):
        raise ValueError("scalar component count drift")
    if declared_counts != actual_counts:
        raise ValueError(f"classification count drift: {actual_counts} != {declared_counts}")
    if int(expected.get("priority_1_families", -1)) != sum(
        candidate.priority == 1 for candidate in candidates
    ):
        raise ValueError("priority-1 count drift")
    if int(expected.get("priority_2_families", -1)) != sum(
        candidate.priority == 2 for candidate in candidates
    ):
        raise ValueError("priority-2 count drift")

    shortlist = _mapping(payload.get("shortlist"), label="shortlist")
    first_wave = tuple(str(value) for value in shortlist.get("first_wave", ()))
    second_wave = tuple(
        str(value) for value in shortlist.get("second_wave_after_priors_and_controlled_perturbations", ())
    )
    by_id = {candidate.id: candidate for candidate in candidates}
    if set(first_wave + second_wave) - set(by_id):
        raise ValueError("shortlist contains an unknown parameter family")
    if set(first_wave) != {candidate.id for candidate in candidates if candidate.priority == 1}:
        raise ValueError("first-wave shortlist must exactly match priority-1 families")
    if any(by_id[name].classification != "inversion_candidate" for name in first_wave + second_wave):
        raise ValueError("shortlist may contain only inversion candidates")
    if set(first_wave) & set(second_wave):
        raise ValueError("shortlist waves must not overlap")

    return PhysicalParameterRegistry(
        schema_version=SCHEMA_VERSION,
        status=_text(payload, "status", label="registry"),
        candidates=candidates,
        first_wave=first_wave,
        second_wave=second_wave,
        source_path=source_path,
    )


def registry_summary(registry: PhysicalParameterRegistry) -> dict[str, object]:
    """Return a stable compact summary for CI and human-facing verification."""

    return {
        "schema_version": registry.schema_version,
        "status": registry.status,
        "parameter_families": len(registry.candidates),
        "scalar_components": registry.scalar_component_count,
        "classification_counts": dict(sorted(registry.classification_counts.items())),
        "first_wave": list(registry.first_wave),
        "second_wave": list(registry.second_wave),
    }
