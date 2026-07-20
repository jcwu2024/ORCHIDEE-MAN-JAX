from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from build_qsat_micro_oracle import DEFAULT_COMPILER, ROOT, build_oracle

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "reference_mode" / "micro_oracles" / "qsat_moisture"
ATOL = 1.0e-24
RTOL = {"qsatcalc": 1.0e-12, "dev_qsatcalc": 1.0e-12, "qsfrict_init": 1.0e-12}
INPUTS = {
    "qsatcalc": [
        {"temperature_k": 99.25, "pressure_hpa": 1013.25, "coverage": "lower_clamped_index_extrapolation"},
        {"temperature_k": 101.0, "pressure_hpa": 1013.25, "coverage": "lower_valid_index"},
        {"temperature_k": 150.25, "pressure_hpa": 800.0, "coverage": "ice"},
        {"temperature_k": 250.25, "pressure_hpa": 950.0, "coverage": "ice"},
        {"temperature_k": 271.999, "pressure_hpa": 1000.0, "coverage": "ice_to_phase_edge"},
        {"temperature_k": 272.0, "pressure_hpa": 1000.0, "coverage": "ice_table_node"},
        {"temperature_k": 272.5, "pressure_hpa": 1000.0, "coverage": "phase_interval"},
        {"temperature_k": 272.999, "pressure_hpa": 1000.0, "coverage": "phase_interval_edge"},
        {"temperature_k": 273.0, "pressure_hpa": 1000.0, "coverage": "liquid_table_node"},
        {"temperature_k": 273.001, "pressure_hpa": 1000.0, "coverage": "liquid"},
        {"temperature_k": 280.25, "pressure_hpa": 1000.0, "coverage": "liquid_interpolation"},
        {"temperature_k": 295.75, "pressure_hpa": 990.0, "coverage": "liquid_interpolation"},
        {"temperature_k": 368.999, "pressure_hpa": 700.0, "coverage": "upper_valid_index"},
        {"temperature_k": 370.25, "pressure_hpa": 700.0, "coverage": "upper_clamped_index_extrapolation"},
    ],
    "dev_qsatcalc": [
        {"temperature_k": 99.25, "pressure_hpa": 1013.25, "coverage": "lower_clamped_derivative_index"},
        {"temperature_k": 100.5, "pressure_hpa": 1013.25, "coverage": "lower_valid_rounding_boundary"},
        {"temperature_k": 101.0, "pressure_hpa": 1013.25, "coverage": "lower_valid_index"},
        {"temperature_k": 250.25, "pressure_hpa": 950.0, "coverage": "ice_derivative"},
        {"temperature_k": 272.49, "pressure_hpa": 1000.0, "coverage": "ice_side_derivative"},
        {"temperature_k": 272.5, "pressure_hpa": 1000.0, "coverage": "phase_rounding_boundary"},
        {"temperature_k": 272.999, "pressure_hpa": 1000.0, "coverage": "phase_derivative"},
        {"temperature_k": 273.0, "pressure_hpa": 1000.0, "coverage": "liquid_phase_boundary"},
        {"temperature_k": 273.5, "pressure_hpa": 1000.0, "coverage": "liquid_rounding_boundary"},
        {"temperature_k": 280.25, "pressure_hpa": 1000.0, "coverage": "liquid_derivative"},
        {"temperature_k": 295.75, "pressure_hpa": 990.0, "coverage": "liquid_derivative"},
        {"temperature_k": 368.0, "pressure_hpa": 700.0, "coverage": "upper_table_index"},
        {"temperature_k": 368.499999, "pressure_hpa": 700.0, "coverage": "upper_valid_rounding_boundary"},
        {"temperature_k": 370.25, "pressure_hpa": 700.0, "coverage": "upper_clamped_derivative_index"},
    ],
    "qsfrict_indices": [100, 101, 272, 273, 369, 370],
}


def _write_fortran_input(path: Path) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for procedure in ("qsatcalc", "dev_qsatcalc"):
            rows = INPUTS[procedure]
            handle.write(f"{len(rows)}\n")
            for row in rows:
                handle.write(f"{row['temperature_k']:.17g} {row['pressure_hpa']:.17g}\n")


def _recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path, value_field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["temperature_k"]) for row in rows], dtype=np.float64),
        np.asarray([float(row["pressure_hpa"]) for row in rows], dtype=np.float64),
        np.asarray([float(row[value_field]) for row in rows], dtype=np.float64),
    )


def _write_jax_csv(path: Path, temperatures: np.ndarray, pressures: np.ndarray, field: str, values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["temperature_k", "pressure_hpa", field])
        for temperature, pressure, value in zip(temperatures, pressures, values, strict=True):
            writer.writerow([f"{temperature:.17e}", f"{pressure:.17e}", f"{value:.17e}"])


def _comparison(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    absolute = np.abs(actual - expected)
    denominator = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    relative = absolute / denominator
    return {
        "name": name,
        "count": int(actual.size),
        "rtol": RTOL[name],
        "atol": ATOL,
        "max_abs_error": float(np.max(absolute)),
        "max_rel_error": float(np.max(relative)),
        "max_error_index": int(np.argmax(absolute)),
        "passed": bool(np.allclose(actual, expected, rtol=RTOL[name], atol=ATOL)),
    }


def _write_point_comparisons(
    path: Path,
    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["procedure", "temperature_or_index", "pressure_hpa", "fortran", "jax", "abs_error", "rel_error", "passed"]
        )
        for name, coordinates, pressures, actual, expected in datasets:
            absolute = np.abs(actual - expected)
            relative = absolute / np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
            passed = np.isclose(actual, expected, rtol=RTOL[name], atol=ATOL)
            for values in zip(coordinates, pressures, actual, expected, absolute, relative, passed, strict=True):
                coordinate, pressure, fortran_value, jax_value, abs_error, rel_error, point_passed = values
                writer.writerow(
                    [
                        name,
                        f"{coordinate:.17e}",
                        "" if np.isnan(pressure) else f"{pressure:.17e}",
                        f"{fortran_value:.17e}",
                        f"{jax_value:.17e}",
                        f"{abs_error:.17e}",
                        f"{rel_error:.17e}",
                        str(bool(point_passed)).lower(),
                    ]
                )


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    from jax_orchidee.sechiba.enerbil import (
        _qsfrict_table,
        qsat_moisture_dev_qsatcalc,
        qsat_moisture_qsatcalc,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inputs.json").write_text(json.dumps(INPUTS, indent=2) + "\n", encoding="ascii")
    input_path = output_dir / "oracle_input.dat"
    _write_fortran_input(input_path)

    with tempfile.TemporaryDirectory(prefix="orchidee_qsat_micro_oracle_") as temporary:
        build_dir = Path(temporary)
        build = build_oracle(build_dir, compiler)
        executable = Path(str(build["executable"]))
        run_command = [str(executable), str(input_path.resolve()), str(output_dir.resolve())]
        run_env = os.environ.copy()
        run_env["PATH"] = str(compiler.parent) + os.pathsep + run_env.get("PATH", "")
        completed = subprocess.run(
            run_command, cwd=build_dir, env=run_env, text=True, capture_output=True, check=True
        )
        retained_build = {
            key: value for key, value in build.items() if key not in {"executable", "source"}
        }
        retained_build["compile_command"] = [
            "<gfortran>", *build["compile_command"][1:-3], "<generated_harness>", "-o", "<temporary_executable>"
        ]
        retained_build["run_command"] = [
            "<temporary_executable>",
            _recorded_path(input_path),
            _recorded_path(output_dir),
        ]
        retained_build["run_stdout"] = completed.stdout
        retained_build["run_stderr"] = completed.stderr

    q_temp, q_pres, q_fortran = _read_csv(output_dir / "fortran_qsatcalc.csv", "qsat")
    d_temp, d_pres, d_fortran = _read_csv(output_dir / "fortran_dev_qsatcalc.csv", "dev_qsat")
    q_jax = np.asarray(qsat_moisture_qsatcalc(q_temp, q_pres), dtype=np.float64)
    d_jax = np.asarray(qsat_moisture_dev_qsatcalc(d_temp, d_pres), dtype=np.float64)
    table_fortran = np.genfromtxt(
        output_dir / "fortran_qsfrict.csv", delimiter=",", names=True, dtype=None, encoding="ascii"
    )
    table_indices = np.asarray(table_fortran["temperature_index"], dtype=np.int32)
    table_values = np.asarray(table_fortran["qsfrict_hpa"], dtype=np.float64)
    table_jax = np.asarray(_qsfrict_table(), dtype=np.float64)[table_indices]

    _write_jax_csv(output_dir / "jax_qsatcalc.csv", q_temp, q_pres, "qsat", q_jax)
    _write_jax_csv(output_dir / "jax_dev_qsatcalc.csv", d_temp, d_pres, "dev_qsat", d_jax)
    with (output_dir / "jax_qsfrict.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["temperature_index", "qsfrict_hpa"])
        for index, value in zip(table_indices, table_jax, strict=True):
            writer.writerow([index, f"{value:.17e}"])

    comparisons = [
        _comparison("qsatcalc", q_fortran, q_jax),
        _comparison("dev_qsatcalc", d_fortran, d_jax),
        _comparison("qsfrict_init", table_values, table_jax),
    ]
    _write_point_comparisons(
        output_dir / "point_comparisons.csv",
        [
            ("qsatcalc", q_temp, q_pres, q_fortran, q_jax),
            ("dev_qsatcalc", d_temp, d_pres, d_fortran, d_jax),
            (
                "qsfrict_init",
                table_indices.astype(np.float64),
                np.full(table_indices.shape, np.nan),
                table_values,
                table_jax,
            ),
        ],
    )
    result = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(item["passed"] for item in comparisons) else "failed",
        "source_file": "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90",
        "jax_owner": "jax_orchidee.sechiba.enerbil",
        "build": retained_build,
        "tolerance": {"rtol_by_procedure": RTOL, "atol": ATOL, "comparison": "numpy.allclose"},
        "comparisons": comparisons,
        "assets": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    result["assets"] = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    if result["status"] != "passed":
        raise RuntimeError(f"Fortran/JAX comparison failed: {comparisons}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile, run, and compare the qsat Fortran micro-oracle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args(argv)
    try:
        result = run_oracle(args.output_dir.resolve(), args.compiler.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "comparisons": result["comparisons"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
