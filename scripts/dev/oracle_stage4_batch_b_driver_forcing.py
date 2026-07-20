"""Stage 4 Batch B gate for the five DIM2/forcing-info owner contracts.

This is deliberately a *non-certifying* runner until a harness can execute the
original DIM2 program fragments and ``forcing_info`` with its IOIPSL/NetCDF/MPI
boundaries.  It records the immutable source bytes, the complete static arm
set, and the focused JAX checks.  In particular, it must not manufacture a
passing Fortran oracle from the JAX implementation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
    extract_source_fragments,
)
from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
)


ROOT = Path(__file__).resolve().parents[2]
FAMILY = "stage4_batch_b_driver_forcing_formal"
OUTPUT = ROOT / "outputs" / "reference_mode" / "micro_oracles" / FAMILY
DIM2 = ROOT / "fortran_source" / "ORCHIDEE" / "src_driver" / "dim2_driver.f90"
READDIM2 = ROOT / "fortran_source" / "ORCHIDEE" / "src_driver" / "readdim2.f90"
CONTROLS_TEMPLATE = ROOT / "scripts/dev/oracle_stage4_batch_b_dim2_controls.f90.template"

OWNERS = {
    "pft14-owner-contract-8b2f034e7b0a": ["600", "639-645", "659-676"],
    "pft14-owner-contract-9d83d36c3d1b": ["361-401", "827-931", "1441-1445"],
    "pft14-owner-contract-2f2f83908ced": ["443-458", "569-572"],
    "pft14-owner-contract-f6d7f1a4e631": ["285-311", "379-390", "470-490", "724-806", "835-914", "995-1008", "1051", "1141-1158", "1403-1426"],
}


def _arm_ids(owner_id: str) -> list[str]:
    document = json.loads(
        (ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json").read_text(
            encoding="utf-8"
        )
    )
    for owner in document["owner_regions"]:
        if owner["owner_region_id"] == owner_id:
            return list(owner["arm_ids"])
    raise KeyError(owner_id)


def _run_tests() -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/unit/test_dim2_driver.py",
        "tests/unit/test_forcing_metadata.py",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def _dim2_controls_oracle(output_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Execute the unmodified DIM2 lines 620-696 with config transport stubs."""

    original = extract_source_fragments(DIM2, ["620-696"])
    with tempfile.TemporaryDirectory(prefix="orchidee_stage4_dim2_controls_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(
            CONTROLS_TEMPLATE.read_bytes().replace(b"! <DIM2_CONTROLS>", original.span_bytes)
        )
        executable = build / "oracle.exe"
        metadata = compile_fortran(source, executable, DEFAULT_COMPILER)
        path = output_dir / "dim2_controls_fortran.txt"
        subprocess.run(
            [str(executable), str(path.resolve())], cwd=build,
            env=compiler_environment(DEFAULT_COMPILER), check=True,
            text=True, capture_output=True,
        )
    rows = np.loadtxt(path, dtype=str)
    actual = np.asarray(
        [[int(row[1] == "T"), int(row[2] == "T"), int(row[3]), int(row[4] == "T")] for row in rows],
        dtype=np.int64,
    )
    from jax_orchidee.driver.dim2 import resolve_dim2_forcing_controls

    arguments = (
        dict(dt_force=7200.0, dt=1800.0, split=4),
        dict(dt_force=21600.0, dt=1800.0, split=12, inter_lin=True, spread_override=99, netrad_cons_override=False),
        dict(dt_force=21600.0, dt=21600.0, split=1, inter_lin=True, spread_override=2),
        dict(dt_force=3600.0, dt=1800.0, split=2, inter_lin=True),
    )
    expected = np.asarray(
        [[item.no_inter, item.inter_lin, item.nb_spread, item.netrad_cons]
         for item in (resolve_dim2_forcing_controls(**item) for item in arguments)],
        dtype=np.int64,
    )
    return metadata, [exact_comparison("dim2_forcing_controls", actual, expected)]


def run_oracle(output_dir: Path = OUTPUT, compiler: Path | None = None) -> dict[str, object]:
    """Run the non-certifying formal-gate preflight.

    ``compiler`` is accepted for unified-runner compatibility.  It is unused
    because compiling substituted IOIPSL/NetCDF/MPI calls would not be a real
    Fortran call segment and therefore cannot close these owners.
    """

    del compiler
    output_dir.mkdir(parents=True, exist_ok=True)
    dim2_spans = {
        owner_id: extract_source_fragments(DIM2, spans)
        for owner_id, spans in OWNERS.items()
    }
    forcing_info = extract_procedure_bytes(READDIM2, "forcing_info")
    tests = _run_tests()
    controls_build, controls_comparisons = _dim2_controls_oracle(output_dir)
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "status": "blocked",
        "owners": [
            {
                "owner_region_id": owner_id,
                "arm_ids": _arm_ids(owner_id),
                "covered_arm_ids": [],
                "uncovered_arm_ids": _arm_ids(owner_id),
            }
            for owner_id in (*OWNERS, "pft14-owner-contract-d6769359ae0a")
        ],
        "block": "Only DIM2 lines 620-696 have a compiled real-source microcase. Remaining closure requires real DIM2 fragments and forcing_info plus its source helpers, with thin fixture implementations for IOIPSL/NetCDF/MPI transport.",
    }
    provenance = {
        owner_id: {
            "source_file": str(DIM2.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": span.source_sha256,
            "span_sha256": span.span_sha256,
            "start_line": span.start_line,
            "end_line": span.end_line,
            "source_fragments": OWNERS[owner_id],
        }
        for owner_id, span in dim2_spans.items()
    }
    provenance["pft14-owner-contract-d6769359ae0a"] = {
        "source_file": str(READDIM2.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": forcing_info.source_sha256,
        "span_sha256": forcing_info.span_sha256,
        "start_line": forcing_info.start_line,
        "end_line": forcing_info.end_line,
        "procedure": "forcing_info",
    }
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, "source_provenance": provenance}, indent=2)
        + "\n",
        encoding="ascii",
    )
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )
    result = {
        "schema_version": 2,
        "family": FAMILY,
        "status": "blocked",
        "comparisons": [
            *controls_comparisons,
            {
                "name": "focused_jax_contract_tests",
                "passed": tests["passed"],
                "comparison": "pytest",
            }
        ],
        "source_provenance": provenance,
        "real_fortran_segments": {
            "pft14-owner-contract-8b2f034e7b0a": {
                "fortran_lines": "620-696",
                "span_sha256": extract_source_fragments(DIM2, ["620-696"]).span_sha256,
                "compiler": controls_build,
            }
        },
        "attempted_dependency_graph": {
            "forcing_info": {
                "real_source_helpers": [
                    "readdim2.f90::domain_size (2263-2341)",
                    "readdim2.f90::forcing_zoom (2034-2051)",
                    "readdim2.f90::forcing_landind (1899-1968)",
                    "readdim2.f90::forcing_vertical_ioipsl (2057-2257)",
                ],
                "transport_boundaries": [
                    "flininfo/flinopen/flinget_buffer/flinquery_var",
                    "ioget_calendar/ioconf_calendar/itau2date",
                    "bcast/grid_set_glo/grid_allocate_glo/Init_orchidee_data_para_driver/init_ioipsl_para",
                    "getin_p/ipslerr_p",
                ],
                "local_probe": "gfortran present; no local netcdf.mod, netcdff library, nf-config, or nc-config found",
            },
            "dim2_driver": {
                "compiled": ["620-696 forcing controls"],
                "remaining": ["restart/default state fragments", "time/restart chronology", "root/fixed dispatch", "dimension guards"],
            },
        },
        "block": coverage["block"],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    return result


def main() -> int:
    result = run_oracle()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
