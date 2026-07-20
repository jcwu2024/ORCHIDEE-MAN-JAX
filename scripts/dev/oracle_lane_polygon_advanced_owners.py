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
    polygones_cleanup,
    polygones_convexhull,
    polygones_crossing,
    polygones_extend,
    polygones_intersection,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "polygon_advanced_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/polygones.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_polygon_advanced_owners.f90.template"
PROCEDURES = {
    "polygones_extend",
    "polygones_intersection",
    "polygones_cleanup",
    "polygones_crossing",
    "polygones_convexhull",
}
DEPENDENCIES = {"polygones_pointinside", "polygones_lineintersect"}
DISCRETE_FIELDS = {
    "extend_axis_n",
    "extend_mixed_n",
    "intersection_inner_n",
    "intersection_reverse_n",
    "intersection_empty_n",
    "cleanup_n",
    "crossing_n",
    "hull_a_n",
    "hull_b_n",
}


def compose(path: Path) -> dict[str, str]:
    spans = {
        name: extract_procedure_bytes(SOURCE, name)
        for name in PROCEDURES | DEPENDENCIES
    }
    data = TEMPLATE.read_bytes()
    for name, span in spans.items():
        data = data.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(data)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _result_fields(name: str, result) -> dict[str, np.ndarray]:
    return {
        f"{name}_n": np.asarray([result.nvert]),
        f"{name}_xy": np.asarray(result.vertices).ravel(),
    }


def _jax() -> dict[str, np.ndarray]:
    square = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    inner = np.asarray([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    disjoint = np.asarray([[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0]])
    diagonal = np.asarray([[0.0, 0.0], [2.0, 2.0], [1.0, 3.0]])
    dirty = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0], [2.0, 0.0]]
    )
    cross_b = np.asarray([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]])
    hull_a = np.asarray(
        [
            [0, 0],
            [2, 0],
            [2, 2],
            [0, 2],
            [1, 1],
            [0.5, 0.5],
            [1.5, 0.5],
            [1, 1.5],
            [1, 1],
        ]
    )
    hull_b = np.asarray(
        [
            [3, 1],
            [-1, 2],
            [0, -1],
            [2, 0],
            [1, 3],
            [-2, 0],
            [3, 2],
            [0, 1],
            [2, -2],
            [1, 0.5],
        ]
    )
    fields: dict[str, np.ndarray] = {}
    for name, result in (
        ("extend_axis", polygones_extend(4, square, 2)),
        ("extend_mixed", polygones_extend(3, diagonal, 3)),
        ("intersection_inner", polygones_intersection(4, square, 4, inner)),
        ("intersection_reverse", polygones_intersection(4, inner, 4, square)),
        ("cleanup", polygones_cleanup(6, dirty)),
        ("crossing", polygones_crossing(4, square, 4, cross_b)),
        ("hull_a", polygones_convexhull(9, hull_a)),
        ("hull_b", polygones_convexhull(10, hull_b)),
    ):
        fields.update(_result_fields(name, result))
    fields["intersection_empty_n"] = np.asarray(
        [polygones_intersection(4, square, 4, disjoint).nvert]
    )
    return fields


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_polygon_advanced_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        compiler_meta = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str((output_dir / "fortran_outputs.csv").resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        exact_comparison(name, fortran[name].astype(np.int64), jax[name])
        if name in DISCRETE_FIELDS
        else float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "axis-aligned and diagonal edge extension",
                    "contained and disjoint polygon intersection",
                    "duplicate cleanup and touching-edge crossings",
                    "concave/interior-point convex hulls",
                    "deterministic convex-hull control-flow matrix",
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
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": hashes,
            "compiler": compiler_meta,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
