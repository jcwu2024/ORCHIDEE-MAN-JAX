from __future__ import annotations
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from extract_fortran_micro_oracle import extract_procedure_bytes
from fortran_oracle_common import (
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FAMILY = "surface_enerbil_potential_temperature"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_surface_enerbil.f90.template"


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    span = extract_procedure_bytes(SOURCE, "enerbil_pottemp")
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_surface_enerbil_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(
            TEMPLATE.read_bytes().replace(b"! <ENERBIL_PROCEDURES>", span.span_bytes)
        )
        exe = build / "oracle.exe"
        metadata = compile_fortran(source, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    values = {}
    with csv_path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    f = {k: np.asarray(v) for k, v in values.items()}
    from jax_orchidee.sechiba.enerbil import enerbil_pottemp_pass_through

    jax = enerbil_pottemp_pass_through(
        q_sol_pot=np.array([0.001, 0.01, 0.1]),
        temp_sol_pot=np.array([250.0, 280.0, 310.0]),
    )
    j = {
        "q_sol_pot": np.asarray(jax.q_sol_pot),
        "temp_sol_pot": np.asarray(jax.temp_sol_pot),
    }
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "branch_cases": ["nonzero diagnostic state pass-through"],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", f, j, rtol=1e-12, atol=0.0
    )
    comparisons = [
        float_comparison(k, v, j[k], rtol=1e-12, atol=0.0) for k, v in f.items()
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "ledger_entries": ["enerbil.conditional.potential_temperature"],
            "span_sha256": {"enerbil_pottemp": span.span_sha256},
            "build": metadata,
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
