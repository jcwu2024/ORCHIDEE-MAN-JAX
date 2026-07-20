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
from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    snow3lhold_explicit,
    snow3ltemp_explicit,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "snow_thermo_helpers"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_snow_thermo_helpers.f90.template"
PROCEDURES = {"snow3lhold_1d", "snow3ltemp_1d"}
DEPENDENCIES = ("snow3lscap_1d",)


def compose(path: Path) -> dict[str, str]:
    names = (*DEPENDENCIES, *sorted(PROCEDURES))
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in names}
    source = TEMPLATE.read_bytes()
    for name, span in spans.items():
        source = source.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _jax_outputs() -> dict[str, np.ndarray]:
    rho = np.asarray([50.0, 199.0, 749.0, 750.0, 900.0])
    dz = np.asarray([0.01, 0.02, 0.03, 0.04, 0.05])
    raw = np.asarray([280.0, 273.15, 250.0, 100.0, 50.0])
    heat = dz * (rho * 2106.0 * (raw - 273.15) - 333600.0 * rho)
    return {
        "hold": np.asarray(snow3lhold_explicit(rho, dz)),
        "temp": np.asarray(snow3ltemp_explicit(heat, rho, dz)),
    }


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_snow_thermo_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        compiler_metadata = compile_fortran(source, executable, compiler)
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
        float_comparison(name, values, jax[name], rtol=1e-12, atol=1e-14)
        for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "density below/at/above xrhosmax",
                    "temperature above freezing",
                    "temperature at/below 100 K reset",
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
            "compiler": compiler_metadata,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
