from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = "pft14_source_owners_*.yaml"
DEFAULT_INVENTORY = ROOT / "outputs" / "reference_mode" / "fortran_control_flow_inventory.json"
ALLOWED_STATUS = {"implemented", "partial", "missing"}


def _line_numbers(value: str | int) -> set[int]:
    result: set[int] = set()
    for item in str(value).split(","):
        bounds = [int(part) for part in item.strip().split("-")]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _python_symbol_exists(reference: str) -> bool:
    parts = reference.split(".")
    if len(parts) < 3 or parts[0] != "jax_orchidee":
        return False
    path = ROOT.joinpath(*parts[:-1]).with_suffix(".py")
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbol = parts[-1]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            return True
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == symbol for target in node.targets
        ):
            return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
            return True
    return False


def load_entries(paths: list[Path] | None = None) -> list[dict[str, Any]]:
    paths = paths or sorted((ROOT / "docs" / "source_audits").glob(DEFAULT_GLOB))
    entries: list[dict[str, Any]] = []
    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ValueError(f"{path}: schema_version must be 1")
        if not isinstance(document.get("entries"), list):
            raise ValueError(f"{path}: entries must be a list")
        entries.extend(document["entries"])
    return entries


def validate_entries(entries: list[dict[str, Any]], inventory: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    procedures = {
        (procedure["file"], procedure["name"].lower()): procedure
        for procedure in inventory.get("procedures", [])
    }
    seen_ids: set[str] = set()
    implemented_lines: dict[tuple[str, str], set[int]] = {}
    for index, entry in enumerate(entries):
        label = str(entry.get("id", f"entries[{index}]"))
        if label in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(label)
        required = {
            "id",
            "fortran_file",
            "fortran_subroutine",
            "fortran_lines",
            "status",
            "jax_owner",
            "gap",
            "state_contract",
            "validation",
        }
        missing = sorted(required - set(entry))
        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue
        key = (str(entry["fortran_file"]).replace("\\", "/"), str(entry["fortran_subroutine"]).lower())
        procedure = procedures.get(key)
        if procedure is None:
            errors.append(f"{label}: unknown Fortran procedure {key}")
            continue
        try:
            lines = _line_numbers(entry["fortran_lines"])
        except (TypeError, ValueError):
            errors.append(f"{label}: invalid fortran_lines {entry['fortran_lines']!r}")
            continue
        procedure_lines = set(range(int(procedure["start_line"]), int(procedure["end_line"]) + 1))
        if not lines or not lines.issubset(procedure_lines):
            errors.append(f"{label}: source lines leave procedure span")

        status = entry["status"]
        if status not in ALLOWED_STATUS:
            errors.append(f"{label}: invalid status {status!r}")
        owner = entry["jax_owner"]
        gap = entry["gap"]
        if status == "implemented":
            if not isinstance(owner, str) or not _python_symbol_exists(owner):
                errors.append(f"{label}: implemented owner does not exist: {owner!r}")
            if gap is not None:
                errors.append(f"{label}: implemented entry cannot retain a gap")
            prior = implemented_lines.setdefault(key, set())
            overlap = prior.intersection(lines)
            if overlap:
                errors.append(f"{label}: implemented source lines overlap another owner")
            prior.update(lines)
        elif status == "partial":
            if not isinstance(owner, str) or not _python_symbol_exists(owner):
                errors.append(f"{label}: partial owner does not exist: {owner!r}")
            if not isinstance(gap, str) or not gap.strip():
                errors.append(f"{label}: partial entry requires a gap")
        elif status == "missing":
            if owner is not None:
                errors.append(f"{label}: missing entry cannot declare a JAX owner")
            if not isinstance(gap, str) or not gap.strip():
                errors.append(f"{label}: missing entry requires a gap")
        if status in {"implemented", "partial"}:
            if not isinstance(entry["state_contract"], list) or not entry["state_contract"]:
                errors.append(f"{label}: owner requires state_contract evidence")
            if not isinstance(entry["validation"], list) or not entry["validation"]:
                errors.append(f"{label}: owner requires validation evidence")
            else:
                for reference in entry["validation"]:
                    path = ROOT / str(reference).split("::", 1)[0]
                    if not path.exists():
                        errors.append(f"{label}: validation path does not exist: {reference}")
            for contract in entry["state_contract"] if isinstance(entry["state_contract"], list) else []:
                reference = str(contract)
                if reference.startswith("jax_orchidee.") and not _python_symbol_exists(reference):
                    errors.append(f"{label}: state-contract symbol does not exist: {reference}")
                elif "/" in reference and not (ROOT / reference.split("::", 1)[0]).exists():
                    errors.append(f"{label}: state-contract path does not exist: {reference}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit exact PFT14 Fortran-to-JAX source owners.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args(argv)
    entries = load_entries()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    errors = validate_entries(entries, inventory)
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"OK entries={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
