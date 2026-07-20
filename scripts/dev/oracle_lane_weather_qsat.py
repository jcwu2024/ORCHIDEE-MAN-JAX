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
from jax_orchidee.driver.driver_forcing_completion import weathgen_qsat_2d  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "weather_qsat"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/weather.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_weather_qsat.f90.template"
PROCEDURES = {"weathgen_qsat_2d"}


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, "weathgen_qsat_2d")
    path.write_bytes(
        TEMPLATE.read_bytes().replace(b"! <WEATHGEN_QSAT_2D>", span.span_bytes)
    )
    return {"weathgen_qsat_2d": span.span_sha256}


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="ascii") as handle:
        return {
            "qsat": np.asarray([float(row["value"]) for row in csv.DictReader(handle)])
        }


def _jax_outputs() -> dict[str, np.ndarray]:
    temperature = np.asarray([220.0, 273.15, 273.16, 280.0, 373.15, 213.15]).reshape(
        (2, 3), order="F"
    )
    pressure = np.asarray(
        [100000.0, 100000.0, 100000.0, 500.0, 100000.0, 100000.0]
    ).reshape((2, 3), order="F")
    return {"qsat": weathgen_qsat_2d(temperature, pressure).ravel(order="F")}


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_weather_qsat_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str((output_dir / "fortran_outputs.csv").resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison("qsat", fortran["qsat"], jax["qsat"], rtol=1e-12, atol=1e-14)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": ["below/equal/above freezing", "pressure denominator guard"],
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
