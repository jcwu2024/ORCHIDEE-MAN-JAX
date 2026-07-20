from __future__ import annotations
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from jax_orchidee.driver.geometry_gauss_jordan import (  # noqa: E402
    error_l1_passive,
    gauss_jordan_method,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa:E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "gauss_jordan_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/gauss_jordan_method.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_gauss_jordan_owner.f90.template"
PROCEDURES = {"gauss_jordan_method", "error_l1_passive"}


def compose(path):
    spans = {n: extract_procedure_bytes(SOURCE, n) for n in PROCEDURES}
    data = TEMPLATE.read_bytes()
    for n, s in spans.items():
        data = data.replace(f"! <{n.upper()}>".encode(), s.span_bytes)
    path.write_bytes(data)
    return {n: s.span_sha256 for n, s in spans.items()}


def _read(path):
    d = {}
    with path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            d.setdefault(row["field"], []).append(float(row["value"]))
    return {k: np.asarray(v) for k, v in d.items()}


def _jax():
    a = np.asarray([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    b = np.asarray([7.0, 3.0, 5.0])
    r = gauss_jordan_method(a, b)
    r2 = gauss_jordan_method(np.diag([2.0, 4.0]), np.asarray([6.0, 8.0]))
    cur = np.zeros((3, 2, 7))
    prev = np.zeros_like(cur)
    veg = np.asarray([[0.25, 0.75], [1.0, 0.0], [0.5, 0.5]])
    prev[:, 0, 6] = [100.0, 1e-10, 4.0]
    prev[:, 1, 6] = [20.0, 1e-10, 6.0]
    cur[:, 0, 6] = [101.0, 2e-10, 4.01]
    cur[:, 1, 6] = [19.0, 3e-10, 6.01]
    return {
        "solution": r.vector_b,
        "inverse": r.matrix_a.ravel(),
        "diag_solution": r2.vector_b,
        "passive_flag": error_l1_passive(cur, prev, veg, 0.2).astype(float),
    }


def run_oracle(output_dir, compiler=DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_gauss_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        exe = build / "oracle.exe"
        hashes = compose(source)
        meta = compile_fortran(source, exe, compiler)
        out = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(exe), str(out.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    f = _read(output_dir / "fortran_outputs.csv")
    j = _jax()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", f, j, rtol=1e-12, atol=1e-14
    )
    comp = [float_comparison(n, f[n], j[n], rtol=1e-12, atol=1e-14) for n in sorted(f)]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "full pivot",
                    "diagonal no-swap",
                    "relative and absolute passive error",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comp,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": hashes,
            "compiler": meta,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
