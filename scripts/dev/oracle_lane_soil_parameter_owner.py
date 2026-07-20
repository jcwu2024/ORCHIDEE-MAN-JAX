from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.parameters.soil import SoilParameters, config_soil_parameters  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_result,
)


FAMILY = "soil_parameter_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/constantes_soil.f90"
VAR_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_soil_parameter_owner.f90.template"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
PROCEDURE = "config_soil_parameters"
COMPILE_FLAGS = (
    "-std=legacy", "-fdefault-real-8", "-ffree-line-length-none", "-O0",
    "-fcheck=all", "-ffpe-trap=invalid,zero,overflow", "-Wall", "-Wextra",
)
COVERAGE_FLAGS = ("-std=legacy", "-fdefault-real-8", "-ffree-line-length-none", "-O0")
VALID_MODES = (
    "valid_gate_off", "valid_default", "valid_custom_cwrr",
    "valid_custom_choisnel", "valid_freeze", "valid_freeze_overrides",
    "valid_thermo_skip",
)
EXPECTED_BOUNDARIES = {
    "valid_gate_off": (13, 1),
    "valid_default": (22, 0),
    "valid_custom_cwrr": (22, 14),
    "valid_custom_choisnel": (27, 18),
    "valid_freeze": (23, 3),
    "valid_freeze_overrides": (23, 5),
    "valid_thermo_skip": (22, 0),
}
ERROR_MODES = {
    "error_dry_capa": "DRY_SOIL_HEAT_CAPACITY",
    "error_dry_cond": "DRY_SOIL_HEAT_COND",
    "error_wet_capa": "WET_SOIL_HEAT_CAPACITY",
    "error_wet_cond": "WET_SOIL_HEAT_COND",
    "error_snow_cond": "SNOW_HEAT_COND",
    "error_snow_dens": "SNOW_DENSITY",
    "error_nobio": "NOBIO_WATER_CAPAC_VOLUMETRI",
    "error_qsint": "SECHIBA_QSINT",
    "error_method": "SOIL_LAYERS_DISCRE_METHOD",
    "error_min": "CHOISNEL_DIFF_MIN",
    "error_max": "CHOISNEL_DIFF_MAX",
    "error_exp": "CHOISNEL_DIFF_EXP",
    "error_rsol": "CHOISNEL_RSOL_CSTE",
    "error_litter": "HCRIT_LITTER",
    "error_ecorr": "OK_ECORR cannot be activated",
}


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, PROCEDURE)
    source = TEMPLATE.read_bytes()
    source = source.replace(b"! <CONSTANTES_SOIL_VAR_MODULE>", VAR_SOURCE.read_bytes())
    source = source.replace(b"! <CONFIG_SOIL_PARAMETERS>", span.span_bytes)
    if b"! <" in source:
        raise RuntimeError("unresolved soil owner template marker")
    path.write_bytes(source)
    return {PROCEDURE: span.span_sha256}


def _initial() -> SoilParameters:
    return SoilParameters(
        so_discretization_method=9, read_reftemp=True, ok_freeze_thermix=True,
        ok_ecorr=True, poros=.9, fr_dt=9., ok_snowfact=True,
        ok_freeze_cwrr=True, ok_thermodynamical_freezing=True,
        check_cwrr=True, check_cwrr2=True,
    )


CUSTOM = {
    "DRY_SOIL_HEAT_CAPACITY": 1800001., "DRY_SOIL_HEAT_COND": .41,
    "WET_SOIL_HEAT_CAPACITY": 3030001., "WET_SOIL_HEAT_COND": 1.91,
    "SNOW_HEAT_COND": .31, "SNOW_DENSITY": 331.,
    "NOBIO_WATER_CAPAC_VOLUMETRI": 151., "SECHIBA_QSINT": .11,
    "SOIL_LAYERS_DISCRE_METHOD": 2, "CHOISNEL_DIFF_MIN": .002,
    "CHOISNEL_DIFF_MAX": .2, "CHOISNEL_DIFF_EXP": 1.6,
    "CHOISNEL_RSOL_CSTE": 34000., "HCRIT_LITTER": .09,
    "TAU_PEAT": 315360001., "Z_TAU": 1000001., "POROS": .42, "FR_DT": 2.1,
}


def _expected(mode: str) -> SoilParameters:
    overrides: dict[str, object] = {}
    if mode == "valid_gate_off":
        overrides = {"TAU_PEAT": 7.}
    elif mode.startswith("valid_custom"):
        overrides = dict(CUSTOM)
        if mode == "valid_custom_cwrr":
            overrides["SMCMAX_FAO"] = [.4, .5, .6]
    elif mode == "valid_freeze":
        overrides = {
            "OK_FREEZE": True, "READ_REFTEMP": True, "OK_FREEZE_THERMIX": True,
            "OK_ECORR": True, "OK_SNOWFACT": True, "OK_FREEZE_CWRR": True,
            "OK_THERMODYNAMICAL_FREEZING": True, "CHECK_CWRR": True,
            "CHECK_CWRR2": True,
        }
    elif mode == "valid_freeze_overrides":
        overrides = {
            "OK_FREEZE": True, "READ_REFTEMP": False, "OK_FREEZE_THERMIX": True,
            "OK_ECORR": False, "OK_SNOWFACT": False, "OK_FREEZE_CWRR": True,
            "OK_THERMODYNAMICAL_FREEZING": False, "CHECK_CWRR": False,
            "CHECK_CWRR2": False,
        }
    initial = _initial()
    if mode == "valid_thermo_skip":
        initial = SoilParameters(**{
            **{field.name: getattr(initial, field.name) for field in fields(SoilParameters)},
            "ok_thermodynamical_freezing": False,
        })
    return config_soil_parameters(
        overrides, ok_sechiba=mode != "valid_gate_off",
        impose_param=mode != "valid_gate_off",
        hydrol_cwrr=mode != "valid_custom_choisnel", initial=initial,
    )


def _read_output(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _expected_array(state: SoilParameters, name: str) -> np.ndarray:
    value = getattr(state, name)
    if isinstance(value, tuple):
        return np.asarray(value)
    if isinstance(value, bool):
        return np.asarray([int(value)])
    return np.asarray([value])


def _run(executable: Path, build: Path, mode: str, compiler: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), str(build / f"{mode}.csv"), mode], cwd=build,
        env=compiler_environment(compiler), capture_output=True, text=True, check=False,
    )


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="soilown_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        try:
            metadata = compile_fortran(source, executable, compiler, base_flags=COMPILE_FLAGS)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"soil owner Fortran compilation failed:\n{exc.stderr}") from exc
        for mode in VALID_MODES:
            completed = _run(executable, build, mode, compiler)
            comparisons.append({
                "name": f"{mode}.execution", "passed": completed.returncode == 0,
                "comparison": "exact_exit_code", "actual": completed.returncode, "expected": 0,
            })
            if completed.returncode != 0:
                continue
            output = _read_output(build / f"{mode}.csv")
            expected = _expected(mode)
            for field in fields(SoilParameters):
                name = field.name
                target = _expected_array(expected, name)
                actual = output[name]
                if target.dtype.kind in "biu":
                    passed = actual.shape == target.shape and np.array_equal(actual, target)
                    comparisons.append({
                        "name": f"{mode}.{name}", "passed": bool(passed),
                        "comparison": "exact", "actual": actual.tolist(), "expected": target.tolist(),
                    })
                else:
                    comparisons.append(float_comparison(
                        f"{mode}.{name}", actual, target, rtol=1e-13, atol=1e-15
                    ))
            expected_getin, expected_writeback = EXPECTED_BOUNDARIES[mode]
            comparisons.append({
                "name": f"{mode}.getin_boundary",
                "passed": int(output["getin_total"][0]) == expected_getin,
                "comparison": "exact", "actual": int(output["getin_total"][0]),
                "expected": expected_getin,
            })
            comparisons.append({
                "name": f"{mode}.writeback_boundary",
                "passed": int(output["writeback_total"][0]) == expected_writeback,
                "comparison": "exact", "actual": int(output["writeback_total"][0]),
                "expected": expected_writeback,
            })
            shutil.copyfile(build / f"{mode}.csv", output_dir / f"{mode}.csv")
        for mode, diagnostic in ERROR_MODES.items():
            completed = _run(executable, build, mode, compiler)
            combined = completed.stdout + completed.stderr
            comparisons.append({
                "name": f"{mode}.ipslerr_boundary",
                "passed": completed.returncode != 0 and diagnostic.lower() in combined.lower(),
                "comparison": "exact_diagnostic_boundary", "actual": combined.strip(),
                "expected": diagnostic,
            })
    (output_dir / "inputs.json").write_text(json.dumps({
        "schema_version": 1, "valid_modes": VALID_MODES, "error_modes": ERROR_MODES,
    }, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "constantes_soil_var_sha256": hashlib.sha256(VAR_SOURCE.read_bytes()).hexdigest(),
        "span_sha256": hashes, "compiler": metadata,
        "tolerance": {"float_rtol": 1e-13, "float_atol": 1e-15, "discrete": "exact"},
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run config_soil_parameters owner oracle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args(argv)
    try:
        result = run_oracle(args.output_dir.resolve(), args.compiler.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
