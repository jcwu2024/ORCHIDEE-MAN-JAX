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

from jax_orchidee.parameters.vertical_soil import vertical_soil_init  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
    write_result,
)


FAMILY = "vertical_soil_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/vertical_soil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_vertical_soil_owner.f90.template"
PROCEDURES = {"vertical_soil_init"}
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
VALID_CASES: dict[str, dict[str, float]] = {
    "default": {},
    "correct": {"DEPTH_CSTTHICK": 1.5},
    "geometry": {
        "DEPTH_MAX_T": 12.0,
        "DEPTH_MAX_H": 3.0,
        "DEPTH_TOPTHICK": 0.002,
        "DEPTH_CSTTHICK": 0.4,
        "DEPTH_GEOM": 4.0,
        "RATIO_GEOM_BELOW": 1.2,
    },
}
ERROR_CASES = {
    "bad_hydro": "zmaxh needs to be smaller than zmaxt",
    "bad_geom": "depth_geom needs to be larger than zmaxh",
}
DISCRETE = {"nslm", "ngrnd"}


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, "vertical_soil_init")
    source = TEMPLATE.read_bytes().replace(b"! <VERTICAL_SOIL_INIT>", span.span_bytes)
    if b"! <" in source:
        raise RuntimeError("unresolved vertical-soil procedure marker")
    path.write_bytes(source)
    return {"vertical_soil_init": span.span_sha256}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _jax(case: str) -> dict[str, np.ndarray]:
    result = vertical_soil_init(VALID_CASES[case])
    return {
        "nslm": np.asarray([result.nslm]),
        "ngrnd": np.asarray([result.ngrnd]),
        "znh": result.znh,
        "dnh": result.dnh,
        "dlh": result.dlh,
        "zlh": result.zlh,
        "znt": result.znt,
        "dlt": result.dlt,
        "zlt": result.zlt,
    }


def _run(
    executable: Path,
    output: Path,
    case: str,
    build: Path,
    compiler: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), str(output), case],
        cwd=build,
        env=compiler_environment(compiler),
        check=False,
        capture_output=True,
        text=True,
    )


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict[str, object]] = []
    point_rows: dict[str, np.ndarray] = {}
    expected_rows: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="orchidee_vertical_soil_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        compiler_meta = compile_fortran(source, executable, compiler)
        for case in VALID_CASES:
            output = build / f"{case}.csv"
            completed = _run(executable, output, case, build, compiler)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"vertical_soil_init {case} failed: {completed.stdout}{completed.stderr}"
                )
            fortran = _read(output)
            jax = _jax(case)
            for field in sorted(fortran):
                name = f"{case}.{field}"
                if field in DISCRETE:
                    comparisons.append(
                        exact_comparison(name, fortran[field].astype(np.int64), jax[field])
                    )
                else:
                    comparisons.append(
                        float_comparison(
                            name, fortran[field], jax[field], rtol=1e-12, atol=1e-14
                        )
                    )
                point_rows[name] = fortran[field]
                expected_rows[name] = jax[field]
        for case, message in ERROR_CASES.items():
            completed = _run(executable, build / f"{case}.csv", case, build, compiler)
            text = completed.stdout + completed.stderr
            comparisons.append(
                {
                    "name": f"{case}.fatal_contract",
                    "passed": completed.returncode != 0 and message.lower() in text.lower(),
                    "comparison": "exact_diagnostic_boundary",
                    "expected": message,
                    "actual": text.strip(),
                }
            )
    write_point_comparisons(
        output_dir / "point_comparisons.csv",
        point_rows,
        expected_rows,
        rtol=1e-12,
        atol=1e-14,
    )
    with (output_dir / "fortran_outputs.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("field", "flat_index", "value"))
        for name, values in sorted(point_rows.items()):
            for index, value in enumerate(np.asarray(values).ravel()):
                writer.writerow((name, index, f"{float(value):.17e}"))
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {"schema_version": 1, "valid_cases": VALID_CASES, "error_cases": ERROR_CASES},
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
            "tolerance": {
                "float_rtol": 1e-12,
                "float_atol": 1e-14,
                "discrete": "exact",
            },
        },
    )


if __name__ == "__main__":
    result = run_oracle(DEFAULT_OUTPUT_DIR)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
