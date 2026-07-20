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
from jax_orchidee.driver.geometry_haversine import (  # noqa: E402
    haversine_clockwise,
    haversine_laloarea,
    haversine_laloseglen,
    haversine_polyseglen,
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

FAMILY = "haversine_primitives"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/haversine.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_haversine_primitives.f90.template"
PROCEDURES = {
    "haversine_polyseglen",
    "haversine_laloseglen",
    "haversine_clockwise",
    "haversine_laloarea",
}
DEPENDENCIES = {
    "haversine_dtor",
    "haversine_rtod",
    "haversine_heading",
    "haversine_distance",
}


def compose(path):
    spans = {n: extract_procedure_bytes(SOURCE, n) for n in PROCEDURES | DEPENDENCIES}
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
    lon = np.asarray(
        [
            [0.0, 5.0, 10.0, 10.0, 10.0, 5.0, 0.0, 0.0],
            [5.0, 10.0, 10.0, 10.0, 5.0, 0.0, 0.0, 0.0],
        ]
    )
    lat = np.asarray(
        [
            [0.0, 0.0, 0.0, 5.0, 10.0, 10.0, 10.0, 5.0],
            [0.0, 0.0, 5.0, 10.0, 10.0, 10.0, 5.0, 0.0],
        ]
    )
    poly = haversine_polyseglen(lon, lat)
    lalo = haversine_laloseglen(lon, lat)
    return {
        "polyseg": poly.ravel(),
        "laloseg": lalo.ravel(),
        "clockwise": haversine_clockwise(4, np.arange(0.0, 360.0, 45.0), 0.0).astype(
            float
        ),
        "clockwise_wrap": haversine_clockwise(
            4, np.asarray([350.0, 10.0, 100.0, 170.0, 190.0, 250.0, 280.0, 320.0]), 0.0
        ).astype(float),
        "area": haversine_laloarea(lalo),
    }


def run_oracle(output_dir, compiler=DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_haversine_") as td:
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
        output_dir / "point_comparisons.csv", f, j, rtol=1e-12, atol=1e-9
    )
    comp = [float_comparison(n, f[n], j[n], rtol=1e-12, atol=1e-9) for n in sorted(f)]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "vertex-first regular polygon",
                    "midpoint-first regular polygon",
                    "clockwise wrap",
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
