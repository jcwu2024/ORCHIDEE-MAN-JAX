"""Write compact portable reference and provenance from completed B evidence.

This script reads completed outputs; it does not execute scientific kernels.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/canopy"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def symbol(source: str, name: str) -> dict:
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )
    text = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno]) + "\n"
    return dict(
        start_line=node.lineno, end_line=node.end_lineno, sha256=hashlib.sha256(text.encode()).hexdigest(), text=text
    )


def main() -> None:
    reference = OUT / "reference/fortran_outputs.csv"
    rows = list(csv.DictReader(reference.open(newline="", encoding="utf-8")))
    selected = {
        "diffuco.gpp": 14,
        "diffuco.gsmean": 14,
        "diffuco.rveget": 14,
        "diffuco.rstruct": 14,
        "diffuco.cimean": 14,
        "diffuco.vbeta3": 14,
        "diffuco.vbeta2": 14,
        "enerbil.qsol_sat": 0,
        "enerbil.pdqsold": 0,
        "enerbil.temp_sol_new": 0,
        "enerbil.transpir": 14,
        "enerbil.transpot": 14,
        "enerbil.fluxlat": 0,
        "enerbil.qsurf": 0,
    }
    expected = {
        name: float(
            next(row["value"] for row in rows if row["field"] == name and int(row["i"]) == 1 and int(row["j"]) == j)
        )
        for name, j in selected.items()
    }
    fixture = dict(
        schema_version=1,
        case_name="dry_active",
        point_index=0,
        pft_fortran_index=14,
        original_fortran_csv=str(reference.relative_to(ROOT)).replace("\\", "/"),
        original_fortran_csv_sha256=sha(reference),
        source_spans=json.loads((OUT / "reference/source_spans.json").read_text()),
        input_fixture="scripts/dev/oracle_canopy_component_fixture.py::make_canopy_fixture(indices=(0,))",
        input_fixture_sha256=sha(ROOT / "scripts/dev/oracle_canopy_component_fixture.py"),
        reference_kind="original complete procedures in bounded component harness; scientific routines unmodified; IO stubs only; not diffuco_main or SECHIBA",
        rtol=1e-11,
        atol=1e-12,
        expected=expected,
    )
    save(ROOT / "tests/fixtures/canopy_local_component_reference.json", fixture)
    names = (
        "assemble_pft14_local_enerbil_precall_kwargs_from_first_step_precall",
        "pft14_trans_co2_parameter_inputs_from_run_def",
        "run_pft14_local_enerbil_precall_from_first_step_precall",
        "diffuco_lai_light_table",
    )
    stages = {}
    for revision in ("7cae3e02", "a6a99f8c612813ad8ba3520d0ccb8f83dd059420"):
        source = subprocess.run(
            ["git", "show", f"{revision}:jax_orchidee/sechiba/diffuco.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        stages[revision] = {name: symbol(source, name) for name in names}
    current = (ROOT / "jax_orchidee/sechiba/diffuco.py").read_text()
    stages["working_tree"] = {name: symbol(current, name) for name in names}
    old = stages["7cae3e02"]
    baseline = stages["a6a99f8c612813ad8ba3520d0ccb8f83dd059420"]
    origin = dict(
        classification="pre-existing optional explicit static-parameter JIT entry defect",
        old_and_baseline_assembler_equal=old[names[0]]["sha256"] == baseline[names[0]]["sha256"],
        old_parameter_getter_already_supplies_lai_light='"lai_light": _cached_diffuco_lai_light_table('
        in old[names[1]]["text"],
        paper_default_scope="Prepared context uses pft14_trans_co2_parameter_inputs_from_run_def, which already supplies lai_light. Compiled parameter adapters preserve the trans_co2 mapping. No inference that the default paper compiled route failed or that physical values changed.",
        counterexample="attempts/jit_legacy_layout_01/receipt_jit.json",
        repair="Construct the source-static 20-layer table in the actual first-step assembler only when explicit PFT parameters omit lai_light.",
        source_stages=stages,
    )
    save(OUT / "origin.json", origin)
    reused = []
    for family, assets in {
        "qsat_moisture": ["comparison.json", "fortran_qsatcalc.csv", "fortran_dev_qsatcalc.csv", "fortran_qsfrict.csv"],
        "diffuco_surface_exchange": ["comparison.json", "fortran_outputs.csv"],
        "surface_diffuco_co2_fvcb": ["comparison.json", "fortran_outputs.csv"],
        "surface_diffuco_mangrove_controls": ["comparison.json", "fortran_outputs.csv"],
        "diffuco_batch_c/main_local": ["comparison.json"],
        "surface_enerbil_main": ["comparison.json", "fortran_outputs.csv"],
    }.items():
        for asset in assets:
            path = ROOT / "outputs/reference_mode/micro_oracles" / family / asset
            record = dict(path=str(path.relative_to(ROOT)).replace("\\", "/"), exists=path.is_file(), rerun=False)
            if path.is_file():
                record["sha256"] = sha(path)
                if path.suffix == ".json":
                    record["recorded_status"] = json.loads(path.read_text())["status"]
            reused.append(record)
    save(OUT / "reused_reference_identity.json", reused)
    save(
        OUT / "graph_source_evidence.json",
        dict(
            tier="Verify; task-directed exact source fallback",
            project="E-ORCHIDEE-MAN-JAX",
            parent_successful_generation="2026-11-17T05:57:08Z",
            parent_coverage="metadata_changed; exact current source fallback required",
            child_transport="list_projects and check_index_coverage returned Transport closed; no child graph availability claimed",
            exact_source_files=[
                str(p.relative_to(ROOT)).replace("\\", "/")
                for p in (
                    ROOT / "jax_orchidee/sechiba/diffuco.py",
                    ROOT / "jax_orchidee/sechiba/enerbil.py",
                    ROOT / "jax_orchidee/sechiba/science_completion.py",
                    ROOT / "jax_orchidee/driver/orchestration.py",
                )
            ],
            fortran="Excluded from graph; extracted exact source procedures and direct source phase inspection.",
        ),
    )
    print(
        json.dumps(
            {
                "portable_reference_fields": len(expected),
                "old_equals_baseline": origin["old_and_baseline_assembler_equal"],
                "old_getter_supplies_light": origin["old_parameter_getter_already_supplies_lai_light"],
            }
        )
    )


if __name__ == "__main__":
    main()
