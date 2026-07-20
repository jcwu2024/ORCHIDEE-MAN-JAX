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
from jax_orchidee.driver.forcing_domain_io import FlingetBuffer, flinget_buffer  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "readdim2_flinget_buffer"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_readdim2_flinget_buffer.f90.template"
PROCEDURES = {"flinget_buffer"}


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, "flinget_buffer")
    path.write_bytes(
        TEMPLATE.read_bytes().replace(b"! <FLINGET_BUFFER>", span.span_bytes)
    )
    return {"flinget_buffer": span.span_sha256}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _source(offset: float) -> np.ndarray:
    result = np.empty((2, 2, 4), dtype=np.float64)
    for time in range(4):
        for j in range(2):
            for i in range(2):
                result[i, j, time] = (
                    offset + 100.0 * (time + 1) + 10.0 * (i + 1) + j + 1
                )
    return result


def _jax_outputs() -> dict[str, np.ndarray]:
    buffer = FlingetBuffer({"A": _source(0.0), "B": _source(1000.0)}, nbuff=2)
    outputs = {
        "a_t1": flinget_buffer(buffer, "A", 2, 2, 1, 4, 1, 1).ravel(order="F"),
        "a_t2": flinget_buffer(buffer, "A", 2, 2, 1, 4, 2, 2).ravel(order="F"),
        "a_t3": flinget_buffer(buffer, "A", 2, 2, 1, 4, 3, 3).ravel(order="F"),
        "b_t1": flinget_buffer(buffer, "B", 2, 2, 1, 4, 1, 1).ravel(order="F"),
    }
    outputs["read_count"] = np.asarray([sum(buffer.read_count.values())])
    return outputs


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_readdim2_buffer_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [
                str(executable),
                str((output_dir / "fortran_outputs.csv").resolve()),
                "normal",
            ],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=0.0, atol=0.0
    )
    comparisons = [
        exact_comparison(name, values, jax[name]) for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "nbuff": 2,
                "calls": ["A:1", "A:2 cached", "A:3 refill", "B:1 new variable"],
                "coverage_only_case": "negative NBUFF fatal exit",
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
            "compiler": metadata,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
