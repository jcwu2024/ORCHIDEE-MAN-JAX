from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "docs" / "source_audits" / "pft14_reachable_ledger.yaml"
REQUIRED_MODULES = {
    "DRIVER",
    "SECHIBA",
    "HYDROL",
    "DIFFUCO",
    "ENERBIL",
    "THERMOSOIL",
    "CONDVEG",
    "SLOWPROC",
    "ROUTING",
    "STOMATE",
    "RESTART",
    "MODELOUT",
}
REQUIRED_BRANCH_IDS = {
    "driver.active.forcing_landpoint_interpolation",
    "driver.active.salinity_bbox",
    "driver.active.tide_bbox",
    "slowproc.active.daily_surface_writeback",
    "routing.active.paper_single_landpoint_zero_branch",
    "restart.active.day_and_year_handoff",
    "modelout.active.paper_history_formula",
    "hydrol.active.dynamic_root_water_stress",
    "hydrol.conditional.canopy_interception",
    "hydrol.active.freeze_hydraulics",
    "hydrol.conditional.infiltration_and_excess_routing",
    "hydrol.conditional.explicit_snow_nonzero",
    "hydrol.conditional.water_table_forcing",
    "hydrol.conditional.peat_hydraulic_coefficients",
    "condveg.active.dynamic_roughness",
    "condveg.active.emissivity",
    "condveg.active.snow_fraction",
    "condveg.active.background_albedo",
    "diffuco.active.aerodynamic_boundary",
    "diffuco.conditional.snow_sublimation",
    "diffuco.conditional.canopy_interception",
    "diffuco.active.cwrr_bare_soil",
    "diffuco.conditional.dew_freezing_combination",
    "stomate.active.prescribe_restart_gate",
    "stomate.active.season_memory",
    "stomate.active.phenology",
    "stomate.active.constraints",
    "stomate.active.allocation",
    "stomate.active.npp_growth",
    "stomate.active.gap_mortality",
    "stomate.active.turnover",
    "stomate.active.final_lai_vmax",
    "stomate.active.ok_leak_litter",
    "stomate.active.ok_leak_soilcarbon",
    "stomate.active.tf_doc",
    "stomate.active.ok_leak_firstcall_active_layer",
    "thermosoil.active.refsoc_freeze_properties",
    "thermosoil.conditional.explicit_snow_thermal_profile",
}
REQUIRED_ENTRY_FIELDS = {
    "id",
    "module",
    "fortran_file",
    "fortran_subroutine",
    "fortran_lines",
    "condition",
    "reachability",
    "kernel",
    "production_caller",
    "state_inputs",
    "same_step_outputs",
    "next_step_restart_behavior",
    "microcase",
    "validation_evidence",
    "status",
}
REACHABILITY = {"active", "conditional", "inactive"}
STATUS = {"verified", "open", "unsupported"}
LINE_SPAN = re.compile(r"^\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*$")
FORTRAN_SUBROUTINE = re.compile(r"^\s*SUBROUTINE\s+([A-Za-z][A-Za-z0-9_]*)", re.IGNORECASE | re.MULTILINE)


def _python_symbol_exists(reference: str, *, root: Path) -> bool:
    parts = reference.split(".")
    if len(parts) < 3 or parts[0] != "jax_orchidee":
        return False
    path = root.joinpath(*parts[:-1]).with_suffix(".py")
    if not path.is_file():
        return False
    symbol = parts[-1]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol
        for node in tree.body
    )


def _evidence_exists(reference: str, *, root: Path) -> bool:
    path_text, separator, node_id = reference.partition("::")
    path = root / path_text
    if not path.is_file():
        return False
    if not separator:
        return True
    if not node_id or path.suffix != ".py":
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function_name = node_id.split("[", 1)[0]
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name for node in tree.body)


def load_ledger(path: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("ledger root must be a mapping")
    return document


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def validate_ledger(document: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        return ["entries must be a non-empty list"]

    seen_ids: set[str] = set()
    represented_modules: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"entries[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be a mapping")
            continue
        entry_id = entry.get("id")
        if _nonempty(entry_id):
            label = str(entry_id)
            if label in seen_ids:
                errors.append(f"{label}: duplicate id")
            seen_ids.add(label)

        missing = sorted(field for field in REQUIRED_ENTRY_FIELDS if not _nonempty(entry.get(field)))
        if missing:
            errors.append(f"{label}: missing or empty fields: {', '.join(missing)}")

        module = entry.get("module")
        if isinstance(module, str):
            represented_modules.add(module)
        reachability = entry.get("reachability")
        status = entry.get("status")
        if reachability not in REACHABILITY:
            errors.append(f"{label}: invalid reachability {reachability!r}")
        if status not in STATUS:
            errors.append(f"{label}: invalid status {status!r}")
        if reachability in {"active", "conditional"} and status == "unsupported":
            errors.append(f"{label}: {reachability} entries cannot have status unsupported")

        lines = entry.get("fortran_lines")
        if isinstance(lines, int):
            lines = str(lines)
        if _nonempty(lines) and (not isinstance(lines, str) or LINE_SPAN.fullmatch(lines) is None):
            errors.append(f"{label}: invalid fortran_lines {lines!r}")

        fortran_file = entry.get("fortran_file")
        if isinstance(fortran_file, str) and fortran_file.strip():
            source_path = root / fortran_file
            if not source_path.is_file():
                errors.append(f"{label}: Fortran file does not exist: {fortran_file}")
            else:
                source_text = source_path.read_text(encoding="utf-8", errors="replace")
                subroutine = entry.get("fortran_subroutine")
                declared = {match.group(1).lower() for match in FORTRAN_SUBROUTINE.finditer(source_text)}
                if isinstance(subroutine, str) and subroutine.lower() not in declared:
                    errors.append(f"{label}: Fortran subroutine does not exist in {fortran_file}: {subroutine}")
            if source_path.is_file() and isinstance(lines, str) and LINE_SPAN.fullmatch(lines):
                source_line_count = sum(1 for _ in source_path.open("r", encoding="utf-8", errors="replace"))
                endpoints = [int(value) for span in lines.split(",") for value in span.strip().split("-")]
                if min(endpoints) < 1 or max(endpoints) > source_line_count:
                    errors.append(f"{label}: Fortran line span exceeds source file ({source_line_count} lines)")

        kernel = entry.get("kernel")
        if reachability in {"active", "conditional"} and isinstance(kernel, str) and not _python_symbol_exists(kernel, root=root):
            errors.append(f"{label}: kernel symbol does not exist: {kernel}")
        production_caller = entry.get("production_caller")
        if reachability in {"active", "conditional"} and isinstance(production_caller, str) and not _python_symbol_exists(production_caller, root=root):
            errors.append(f"{label}: production caller symbol does not exist: {production_caller}")

        for list_field in ("state_inputs", "same_step_outputs", "microcase", "validation_evidence"):
            value = entry.get(list_field)
            if _nonempty(value) and not isinstance(value, list):
                errors.append(f"{label}: {list_field} must be a non-empty list")
        if reachability in {"active", "conditional"}:
            for evidence_field in ("microcase", "validation_evidence"):
                for reference in entry.get(evidence_field, []):
                    if not isinstance(reference, str) or not _evidence_exists(reference, root=root):
                        errors.append(f"{label}: {evidence_field} target does not exist: {reference}")

    missing_modules = sorted(REQUIRED_MODULES - represented_modules)
    if missing_modules:
        errors.append(f"priority modules without entries: {', '.join(missing_modules)}")
    missing_branch_ids = sorted(REQUIRED_BRANCH_IDS - seen_ids)
    if missing_branch_ids:
        errors.append(f"required branch ids without entries: {', '.join(missing_branch_ids)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the machine-readable PFT14 reachable-state ledger.")
    parser.add_argument("ledger", nargs="?", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    path = args.ledger.resolve()
    errors = validate_ledger(load_ledger(path), root=ROOT)
    if errors:
        print(f"FAIL {path}")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"OK {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
