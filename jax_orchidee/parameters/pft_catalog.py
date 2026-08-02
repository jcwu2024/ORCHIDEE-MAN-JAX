"""Stable PFT identities and source-backed execution layouts.

The catalog is an orchestration boundary. Scientific formulas remain in the
SECHIBA and STOMATE owners; catalog entries select source identities,
parameter rows, and explicit process capabilities without copying formulas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SUPPORTED_CAPABILITY_STATUS = "supported"
SUPPORTED_PFT_STATUS = frozenset({"paper_validated", "source_validated"})


@dataclass(frozen=True)
class PFTCapability:
    """One source process family available to catalog entries."""

    capability_id: str
    status: str
    jax_owner: str
    fortran_provenance: tuple[str, ...]


@dataclass(frozen=True)
class PFTCatalogEntry:
    """Stable PFT identity plus source parameter and process ownership."""

    pft_id: str
    fortran_pft_id: int
    mtc_id: int
    role: str
    support_status: str
    traits: Mapping[str, bool | int | str]
    parameter_source: str
    capabilities: tuple[str, ...]
    fortran_provenance: tuple[str, ...]


@dataclass(frozen=True)
class PFTCatalog:
    """Versioned source-backed PFT catalog and named execution layouts."""

    schema_version: int
    catalog_id: str
    entries: tuple[PFTCatalogEntry, ...]
    capabilities: Mapping[str, PFTCapability]
    layouts: Mapping[str, tuple[str, ...]]
    source_path: Path

    @property
    def entries_by_id(self) -> dict[str, PFTCatalogEntry]:
        return {entry.pft_id: entry for entry in self.entries}


@dataclass(frozen=True)
class PFTRunLayout:
    """Dense JAX PFT-axis ordering with stable identities retained."""

    catalog_id: str
    layout_id: str
    entries: tuple[PFTCatalogEntry, ...]
    fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.entries) != len(self.fractions):
            raise ValueError("PFT entries and fractions must have the same length")
        if not self.entries:
            raise ValueError("a PFT run layout cannot be empty")
        if len(set(self.pft_ids)) != len(self.entries):
            raise ValueError("a PFT run layout cannot repeat a stable pft_id")
        if any(not np.isfinite(value) or value < 0.0 for value in self.fractions):
            raise ValueError("PFT fractions must be finite and nonnegative")
        bare = [index for index, entry in enumerate(self.entries) if entry.role == "bare_soil"]
        if bare != [0]:
            raise ValueError("the bare-soil entry must be the first and only bare-soil slot")

    @property
    def n_pft(self) -> int:
        return len(self.entries)

    @property
    def pft_ids(self) -> tuple[str, ...]:
        return tuple(entry.pft_id for entry in self.entries)

    @property
    def fortran_pft_ids(self) -> tuple[int, ...]:
        return tuple(entry.fortran_pft_id for entry in self.entries)

    @property
    def mtc_ids(self) -> tuple[int, ...]:
        return tuple(entry.mtc_id for entry in self.entries)

    @property
    def active_mask(self) -> tuple[bool, ...]:
        return tuple(
            entry.role == "bare_soil" or fraction > 0.0
            for entry, fraction in zip(self.entries, self.fractions, strict=True)
        )

    def index_for_id(self, pft_id: str) -> int:
        try:
            return self.pft_ids.index(pft_id)
        except ValueError as exc:
            raise KeyError(f"PFT {pft_id!r} is not present in layout {self.layout_id!r}") from exc

    def metadata(self) -> dict[str, object]:
        return {
            "catalog_id": self.catalog_id,
            "layout_id": self.layout_id,
            "pft_ids": list(self.pft_ids),
            "fortran_pft_ids": list(self.fortran_pft_ids),
            "mtc_ids": list(self.mtc_ids),
            "fractions": list(self.fractions),
            "active_mask": list(self.active_mask),
        }


def _required_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_pft_catalog(path: str | Path) -> PFTCatalog:
    """Load and fail-closed validate a version-1 PFT catalog.

    Source ownership: ``src_parameters/pft_parameters.f90``
    ``pft_parameters_main`` lines 97-169 owns PFT-to-MTC selection;
    ``config_pft_parameters`` lines 247-293 and 2962-3051 owns MTC-derived
    traits. Distinct process capabilities cite their own Fortran owners in the
    catalog asset.
    """

    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported PFT catalog schema version")
    catalog_id = str(payload.get("catalog_id", "")).strip()
    if not catalog_id:
        raise ValueError("PFT catalog_id is required")

    capability_payload = _required_mapping(payload.get("capabilities"), name="capabilities")
    capabilities: dict[str, PFTCapability] = {}
    for capability_id, raw in capability_payload.items():
        declaration = _required_mapping(raw, name=f"capability {capability_id}")
        provenance = tuple(str(item) for item in declaration.get("fortran_provenance", ()))
        if not provenance:
            raise ValueError(f"capability {capability_id!r} has no Fortran provenance")
        capabilities[str(capability_id)] = PFTCapability(
            capability_id=str(capability_id),
            status=str(declaration.get("status", "")),
            jax_owner=str(declaration.get("jax_owner", "")),
            fortran_provenance=provenance,
        )

    entries: list[PFTCatalogEntry] = []
    for raw in payload.get("entries", ()):
        declaration = _required_mapping(raw, name="catalog entry")
        entry = PFTCatalogEntry(
            pft_id=str(declaration.get("pft_id", "")).strip(),
            fortran_pft_id=int(declaration["fortran_pft_id"]),
            mtc_id=int(declaration["mtc_id"]),
            role=str(declaration.get("role", "vegetated")),
            support_status=str(declaration.get("support_status", "unvalidated")),
            traits=dict(_required_mapping(declaration.get("traits"), name="entry traits")),
            parameter_source=str(declaration.get("parameter_source", "")),
            capabilities=tuple(str(item) for item in declaration.get("capabilities", ())),
            fortran_provenance=tuple(str(item) for item in declaration.get("fortran_provenance", ())),
        )
        if not entry.pft_id or not entry.parameter_source or not entry.fortran_provenance:
            raise ValueError("every PFT entry requires identity, parameter source, and provenance")
        unknown = sorted(set(entry.capabilities) - set(capabilities))
        if unknown:
            raise ValueError(f"PFT {entry.pft_id!r} requires unknown capabilities {unknown}")
        entries.append(entry)
    if not entries:
        raise ValueError("PFT catalog has no entries")
    if len({entry.pft_id for entry in entries}) != len(entries):
        raise ValueError("PFT catalog contains duplicate stable pft_id values")
    if len({entry.fortran_pft_id for entry in entries}) != len(entries):
        raise ValueError("PFT catalog contains duplicate Fortran PFT identities")

    layouts_payload = _required_mapping(payload.get("layouts"), name="layouts")
    entry_ids = {entry.pft_id for entry in entries}
    layouts: dict[str, tuple[str, ...]] = {}
    for layout_id, raw_ids in layouts_payload.items():
        ids = tuple(str(item) for item in raw_ids)
        unknown = sorted(set(ids) - entry_ids)
        if unknown:
            raise ValueError(f"layout {layout_id!r} selects unknown PFT IDs {unknown}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"layout {layout_id!r} repeats a PFT ID")
        layouts[str(layout_id)] = ids

    return PFTCatalog(
        schema_version=1,
        catalog_id=catalog_id,
        entries=tuple(entries),
        capabilities=capabilities,
        layouts=layouts,
        source_path=source_path,
    )


def build_pft_run_layout(
    catalog: PFTCatalog,
    *,
    layout_id: str,
    fractions: Sequence[float],
) -> PFTRunLayout:
    """Materialize one named catalog layout in dense execution order."""

    if layout_id not in catalog.layouts:
        raise KeyError(f"unknown PFT layout {layout_id!r}")
    return build_selected_pft_run_layout(
        catalog,
        layout_id=layout_id,
        pft_ids=catalog.layouts[layout_id],
        fractions=fractions,
    )


def build_selected_pft_run_layout(
    catalog: PFTCatalog,
    *,
    layout_id: str,
    pft_ids: Sequence[str],
    fractions: Sequence[float],
) -> PFTRunLayout:
    """Build a declared ad hoc selection while retaining catalog identities."""

    by_id = catalog.entries_by_id
    unknown = sorted(set(pft_ids) - set(by_id))
    if unknown:
        raise ValueError(f"selected PFT IDs are absent from the catalog: {unknown}")
    entries = tuple(by_id[pft_id] for pft_id in pft_ids)
    return PFTRunLayout(
        catalog_id=catalog.catalog_id,
        layout_id=layout_id,
        entries=entries,
        fractions=tuple(float(value) for value in fractions),
    )


def validate_active_capabilities(
    catalog: PFTCatalog,
    layout: PFTRunLayout,
) -> None:
    """Reject an active PFT whose declared source process is unsupported."""

    failures: list[str] = []
    for entry, active in zip(layout.entries, layout.active_mask, strict=True):
        if not active:
            continue
        if entry.support_status not in SUPPORTED_PFT_STATUS:
            failures.append(
                f"{entry.pft_id}:scientific_support={entry.support_status}"
            )
        for capability_id in entry.capabilities:
            status = catalog.capabilities[capability_id].status
            if status != SUPPORTED_CAPABILITY_STATUS:
                failures.append(f"{entry.pft_id}:{capability_id}={status}")
    if failures:
        raise ValueError("active PFT capabilities are not supported: " + ", ".join(failures))


def pft_layout_netcdf_attributes(layout: PFTRunLayout) -> dict[str, str]:
    """Serialize stable PFT-axis metadata as NetCDF-safe global strings."""

    return {
        "orchidee_jax_pft_catalog_id": layout.catalog_id,
        "orchidee_jax_pft_layout_id": layout.layout_id,
        "orchidee_jax_pft_ids_json": json.dumps(layout.pft_ids, separators=(",", ":")),
        "orchidee_jax_fortran_pft_ids_json": json.dumps(
            layout.fortran_pft_ids, separators=(",", ":")
        ),
        "orchidee_jax_mtc_ids_json": json.dumps(layout.mtc_ids, separators=(",", ":")),
    }


def remap_pft_axis(
    values: object,
    *,
    source: PFTRunLayout,
    target: PFTRunLayout,
    axis: int,
    fill_value: float | int | bool = 0,
) -> np.ndarray:
    """Remap an array by stable PFT identity, never by slot coincidence."""

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("a scalar has no PFT axis to remap")
    normalized_axis = int(axis) % array.ndim
    if array.shape[normalized_axis] != source.n_pft:
        raise ValueError("source PFT axis length does not match its declared layout")
    output_shape = list(array.shape)
    output_shape[normalized_axis] = target.n_pft
    output = np.full(output_shape, fill_value, dtype=array.dtype)
    source_indices = {pft_id: index for index, pft_id in enumerate(source.pft_ids)}
    for target_index, pft_id in enumerate(target.pft_ids):
        if pft_id not in source_indices:
            continue
        source_slice = [slice(None)] * array.ndim
        target_slice = [slice(None)] * array.ndim
        source_slice[normalized_axis] = source_indices[pft_id]
        target_slice[normalized_axis] = target_index
        output[tuple(target_slice)] = array[tuple(source_slice)]
    return output
