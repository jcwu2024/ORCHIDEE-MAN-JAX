from __future__ import annotations

import hashlib
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPILER = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
COMPILE_FLAGS = (
    "-std=f2008",
    "-fdefault-real-8",
    "-ffree-line-length-none",
    "-O0",
    "-fcheck=all",
    "-ffpe-trap=invalid,zero,overflow",
    "-Wall",
    "-Wextra",
)


def compiler_environment(compiler: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = (
        str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    )
    return environment


def compile_fortran(
    source: Path,
    executable: Path,
    compiler: Path = DEFAULT_COMPILER,
    *,
    extra_flags: tuple[str, ...] = (),
    base_flags: tuple[str, ...] = COMPILE_FLAGS,
) -> dict[str, object]:
    if not compiler.is_file():
        raise FileNotFoundError(f"GNU Fortran compiler not found: {compiler}")
    command = [
        str(compiler),
        *base_flags,
        *extra_flags,
        str(source),
        "-o",
        str(executable),
    ]
    completed = subprocess.run(
        command,
        cwd=source.parent,
        env=compiler_environment(compiler),
        text=True,
        capture_output=True,
        check=True,
    )
    version = subprocess.run(
        [str(compiler), "--version"],
        env=compiler_environment(compiler),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    return {
        "compiler": str(compiler),
        "compiler_version": version,
        "compile_flags": [*base_flags, *extra_flags],
        "compile_stdout": completed.stdout,
        "compile_stderr": completed.stderr,
        "compile_unit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def float_comparison(
    name: str, fortran: np.ndarray, jax: np.ndarray, *, rtol: float, atol: float
) -> dict[str, object]:
    fortran = np.asarray(fortran, dtype=np.float64)
    jax = np.asarray(jax, dtype=np.float64)
    if fortran.shape != jax.shape:
        return {
            "name": name,
            "passed": False,
            "reason": "shape_mismatch",
            "fortran_shape": list(fortran.shape),
            "jax_shape": list(jax.shape),
        }
    absolute = np.abs(fortran - jax)
    relative = absolute / np.maximum(np.abs(fortran), np.finfo(np.float64).tiny)
    ordered_fortran = fortran.view(np.int64)
    ordered_jax = jax.view(np.int64)
    ulp = np.abs(ordered_fortran - ordered_jax)
    return {
        "name": name,
        "count": int(fortran.size),
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": float(absolute.max(initial=0.0)),
        "max_rel_error": float(relative.max(initial=0.0)),
        "max_ulp_error": int(ulp.max(initial=0)),
        "passed": bool(np.allclose(fortran, jax, rtol=rtol, atol=atol)),
    }


def exact_comparison(
    name: str, fortran: np.ndarray, jax: np.ndarray
) -> dict[str, object]:
    fortran = np.asarray(fortran)
    jax = np.asarray(jax)
    return {
        "name": name,
        "count": int(fortran.size),
        "comparison": "exact",
        "passed": bool(fortran.shape == jax.shape and np.array_equal(fortran, jax)),
    }


def write_result(
    output_dir: Path,
    family: str,
    comparisons: Iterable[dict[str, object]],
    metadata: dict[str, object],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_list = list(comparisons)
    result = {
        "schema_version": 2,
        "family": family,
        "status": "passed"
        if comparison_list and all(item["passed"] for item in comparison_list)
        else "failed",
        "comparisons": comparison_list,
        **metadata,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    return result


def write_point_comparisons(
    path: Path,
    fortran: dict[str, np.ndarray],
    jax: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "field",
                "flat_index",
                "fortran",
                "jax",
                "abs_error",
                "rel_error",
                "passed",
            ]
        )
        for field, fortran_values in fortran.items():
            actual = np.asarray(fortran_values, dtype=np.float64).ravel()
            expected = np.asarray(jax[field], dtype=np.float64).ravel()
            if actual.shape != expected.shape:
                raise ValueError(
                    f"{field}: cannot write point comparison for different shapes"
                )
            absolute = np.abs(actual - expected)
            relative = absolute / np.maximum(np.abs(actual), np.finfo(np.float64).tiny)
            passed = np.isclose(actual, expected, rtol=rtol, atol=atol)
            for index, values in enumerate(
                zip(actual, expected, absolute, relative, passed, strict=True)
            ):
                fortran_value, jax_value, abs_error, rel_error, point_passed = values
                writer.writerow(
                    [
                        field,
                        index,
                        f"{fortran_value:.17e}",
                        f"{jax_value:.17e}",
                        f"{abs_error:.17e}",
                        f"{rel_error:.17e}",
                        str(bool(point_passed)).lower(),
                    ]
                )
