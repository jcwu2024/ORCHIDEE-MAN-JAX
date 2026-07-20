from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "audit_pft14_process_inventory_extensions.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_process_inventory_extensions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _current_documents():
    inventory = AUDIT.load_inventory()
    resolution = json.loads(AUDIT.DEFAULT_RESOLUTION.read_text(encoding="utf-8"))
    return inventory, resolution


def test_current_extension_inventory_is_source_and_resolution_valid():
    inventory, resolution = _current_documents()

    assert AUDIT.validate_inventory(inventory, root=ROOT, resolution=resolution) == []
    assert {entry["category"] for entry in inventory["entries"]} == AUDIT.ALLOWED_CATEGORIES


def test_required_scientific_initializers_cannot_be_swept_into_infrastructure():
    inventory, resolution = _current_documents()
    mutated = deepcopy(inventory)
    hydrol = next(entry for entry in mutated["entries"] if entry["fortran_subroutine"].lower() == "hydrol_init")
    hydrol["category"] = "valid_input_infrastructure"

    errors = AUDIT.validate_inventory(mutated, root=ROOT, resolution=resolution)

    assert any("hydrol_init must remain scientific_process" in error for error in errors)


def test_audit_requires_exact_fortran_procedure_span_and_owner_or_gap():
    inventory, resolution = _current_documents()
    mutated = deepcopy(inventory)
    readstart = next(entry for entry in mutated["entries"] if entry["fortran_subroutine"].lower() == "readstart")
    readstart["fortran_lines"] = "35-1747"
    readstart["jax_owner"] = None
    readstart["gap"] = None

    errors = AUDIT.validate_inventory(mutated, root=ROOT, resolution=resolution)

    assert any("recorded span (35, 1747) != declared span (34, 1747)" in error for error in errors)
    assert any("requires a JAX owner or an explicit gap" in error for error in errors)


def test_inventory_resolution_count_is_stable_across_open_and_closed_statuses():
    inventory, resolution = _current_documents()
    mutated = deepcopy(resolution)
    target = inventory["entries"][1]
    for row in mutated["rows"]:
        if (
            row["file"] == target["fortran_file"]
            and str(row["procedure"]).lower() == target["fortran_subroutine"].lower()
            and row["resolution"] == "process_inventory_open_gap"
        ):
            row["resolution"] = "process_inventory_classified"

    assert AUDIT.validate_inventory(inventory, root=ROOT, resolution=mutated) == []
