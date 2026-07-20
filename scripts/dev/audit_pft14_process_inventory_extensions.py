from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "docs" / "source_audits" / "pft14_process_inventory_extensions.yaml"
DEFAULT_RESOLUTION = ROOT / "outputs" / "reference_mode" / "pft14_control_flow_resolution.json"
ALLOWED_CATEGORIES = {
    "scientific_process",
    "state_io",
    "driver_regridding",
    "valid_input_infrastructure",
    "paper_static_inactive",
}
REQUIRED_PROCEDURES = {
    "pft_parameters_alloc",
    "readstart",
    "ioipslctrl_history",
    "ioipslctrl_histstom",
    "interpweight_1d",
    "interpweight_2d",
    "interpweight_2dcont",
    "interpweight_3d",
    "interpweight_4d",
    "sechiba_init",
    "hydrol_init",
    "stomate_init",
}
SCIENTIFIC_INIT_PROCEDURES = {"sechiba_init", "hydrol_init", "stomate_init"}
REQUIRED_FIELDS = {
    "id",
    "category",
    "fortran_file",
    "fortran_subroutine",
    "fortran_lines",
    "inventory_resolution_sites",
    "paper_path",
    "jax_owner",
    "gap",
    "reason",
}
LINE_SPAN = re.compile(r"^(\d+)-(\d+)$")
SUBROUTINE_START = re.compile(r"^\s*SUBROUTINE\s+([A-Za-z][A-Za-z0-9_]*)\b", re.IGNORECASE)
SUBROUTINE_END = re.compile(r"^\s*END\s+SUBROUTINE\s+([A-Za-z][A-Za-z0-9_]*)\b", re.IGNORECASE)


def load_inventory(path: Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("inventory root must be a mapping")
    return document


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _python_symbol_exists(reference: str, *, root: Path) -> bool:
    parts = reference.split(".")
    if len(parts) < 3 or parts[0] != "jax_orchidee":
        return False
    path = root.joinpath(*parts[:-1]).with_suffix(".py")
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbol = parts[-1]
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol
        for node in tree.body
    )


def _declared_span(source_path: Path, procedure: str) -> tuple[int, int] | None:
    target = procedure.lower()
    start: int | None = None
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if start is None:
            match = SUBROUTINE_START.match(line)
            if match and match.group(1).lower() == target:
                start = line_number
        else:
            match = SUBROUTINE_END.match(line)
            if match and match.group(1).lower() == target:
                return start, line_number
    return None


def _inventory_resolution_counts(resolution: dict[str, Any]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in resolution.get("rows", []):
        if not str(row.get("resolution", "")).startswith("process_inventory_") or not isinstance(
            row.get("procedure"), str
        ):
            continue
        key = (str(row.get("file", "")).replace("\\", "/"), row["procedure"].lower())
        counts[key] = counts.get(key, 0) + 1
    return counts


def validate_inventory(
    document: dict[str, Any],
    *,
    root: Path = ROOT,
    resolution: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if set(document.get("allowed_categories", [])) != ALLOWED_CATEGORIES:
        errors.append("allowed_categories must exactly match the audit categories")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        return errors + ["entries must be a non-empty list"]

    resolution_counts = _inventory_resolution_counts(resolution) if resolution is not None else None
    seen_ids: set[str] = set()
    seen_procedures: set[str] = set()
    represented_categories: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be a mapping")
            continue
        if _nonempty_string(entry.get("id")):
            label = entry["id"]
            if label in seen_ids:
                errors.append(f"{label}: duplicate id")
            seen_ids.add(label)
        missing = sorted(field for field in REQUIRED_FIELDS if field not in entry)
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(missing)}")

        category = entry.get("category")
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"{label}: invalid category {category!r}")
        elif isinstance(category, str):
            represented_categories.add(category)
        procedure_value = entry.get("fortran_subroutine")
        procedure = procedure_value.lower() if _nonempty_string(procedure_value) else ""
        if procedure:
            if procedure in seen_procedures:
                errors.append(f"{label}: duplicate procedure {procedure_value}")
            seen_procedures.add(procedure)
        if procedure in SCIENTIFIC_INIT_PROCEDURES and category != "scientific_process":
            errors.append(f"{label}: {procedure_value} must remain scientific_process")

        paper_path = entry.get("paper_path")
        if paper_path not in {"active", "inactive"}:
            errors.append(f"{label}: paper_path must be active or inactive")
        if category == "paper_static_inactive" and paper_path != "inactive":
            errors.append(f"{label}: paper_static_inactive requires paper_path inactive")
        if category != "paper_static_inactive" and paper_path == "inactive":
            errors.append(f"{label}: inactive paper paths must use paper_static_inactive")

        owner = entry.get("jax_owner")
        gap = entry.get("gap")
        if owner is None and not _nonempty_string(gap):
            errors.append(f"{label}: requires a JAX owner or an explicit gap")
        if owner is not None:
            if not _nonempty_string(owner) or not _python_symbol_exists(owner, root=root):
                errors.append(f"{label}: JAX owner does not exist: {owner!r}")
        if gap is not None and not _nonempty_string(gap):
            errors.append(f"{label}: gap must be null or a non-empty explanation")
        if not _nonempty_string(entry.get("reason")):
            errors.append(f"{label}: reason must be non-empty")

        fortran_file = entry.get("fortran_file")
        source_path = root / fortran_file if _nonempty_string(fortran_file) else None
        if source_path is None or not source_path.is_file():
            errors.append(f"{label}: Fortran file does not exist: {fortran_file!r}")
            continue
        declared_span = _declared_span(source_path, procedure)
        if declared_span is None:
            errors.append(f"{label}: Fortran subroutine does not exist: {procedure_value!r}")
            continue
        span_match = LINE_SPAN.fullmatch(str(entry.get("fortran_lines", "")))
        if span_match is None:
            errors.append(f"{label}: fortran_lines must be one contiguous start-end span")
        else:
            recorded_span = (int(span_match.group(1)), int(span_match.group(2)))
            if recorded_span != declared_span:
                errors.append(f"{label}: recorded span {recorded_span} != declared span {declared_span}")

        inventory_sites = entry.get("inventory_resolution_sites")
        minimum_sites = 0
        if (
            not isinstance(inventory_sites, int)
            or isinstance(inventory_sites, bool)
            or inventory_sites < minimum_sites
        ):
            errors.append(f"{label}: inventory_resolution_sites must be a non-negative integer")
        if resolution_counts is not None:
            key = (str(fortran_file).replace("\\", "/"), procedure)
            current = resolution_counts.get(key, 0)
            if inventory_sites != current:
                errors.append(
                    f"{label}: inventory_resolution_sites {inventory_sites!r} "
                    f"!= current resolution count {current}"
                )

    missing_procedures = sorted(REQUIRED_PROCEDURES - seen_procedures)
    if missing_procedures:
        errors.append(f"required procedures missing: {', '.join(missing_procedures)}")
    if represented_categories != ALLOWED_CATEGORIES:
        missing_categories = sorted(ALLOWED_CATEGORIES - represented_categories)
        errors.append(f"categories without entries: {', '.join(missing_categories)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit PFT14 process inventory extensions.")
    parser.add_argument("inventory", nargs="?", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--resolution", type=Path, default=DEFAULT_RESOLUTION)
    args = parser.parse_args(argv)
    resolution = json.loads(args.resolution.read_text(encoding="utf-8"))
    errors = validate_inventory(load_inventory(args.inventory), root=ROOT, resolution=resolution)
    if errors:
        print(f"FAIL {args.inventory.resolve()}")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"OK {args.inventory.resolve()} entries={len(load_inventory(args.inventory)['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
