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
from jax_orchidee.driver.geometry_polygons import (  # noqa: E402
    polygones_area,
    polygones_lineintersect,
    polygones_pointinside,
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

FAMILY = "polygon_primitives"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/polygones.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_polygon_primitives.f90.template"
PROCEDURES = {"polygones_pointinside", "polygones_lineintersect", "polygones_area"}


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
    square = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    points = [(1.0, 1.0), (3.0, 1.0), (0.0, 1.0), (2.0, 1.0)]
    lines = [
        (np.asarray([[0.0, 0.0], [2.0, 2.0]]), np.asarray([[0.0, 2.0], [2.0, 0.0]])),
        (np.asarray([[0.0, 0.0], [1.0, 0.0]]), np.asarray([[0.0, 1.0], [1.0, 1.0]])),
        (np.asarray([[0.0, 0.0], [1.0, 1.0]]), np.asarray([[2.0, 0.0], [3.0, -1.0]])),
    ]
    results = [polygones_lineintersect(a, b) for a, b in lines]
    return {
        "point_flags": np.asarray(
            [polygones_pointinside(4, square, *p) for p in points], dtype=float
        ),
        "line_flags": np.asarray([r.intersection for r in results], dtype=float),
        "line_point": np.asarray([results[0].point_x, results[0].point_y]),
        "area": np.asarray([polygones_area(4, square, 3.0, 4.0)]),
    }


def run_oracle(output_dir, compiler=DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_polygon_") as td:
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
                    "inside/outside/asymmetric boundary",
                    "crossing/parallel/nonsegment lines",
                    "oriented area",
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
