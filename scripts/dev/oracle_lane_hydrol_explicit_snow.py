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

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "hydrol_explicit_snow_nonzero"
LEDGER_ENTRY = "hydrol.conditional.explicit_snow_nonzero"
SNOW_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90"
QSAT_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_explicit_snow.f90.template"
SNOW_PROCEDURES = (
    "explicitsnow_main",
    "explicitsnow_grain",
    "explicitsnow_compactn",
    "explicitsnow_transf",
    "explicitsnow_fall",
    "explicitsnow_gone",
    "explicitsnow_melt_refrz",
    "explicitsnow_levels",
    "explicitsnow_profile",
)
THERMO_FUNCTIONS = (
    "snow3lgrain_0d",
    "snow3lgrain_1d",
    "snow3lscap_1d",
    "snow3lhold_1d",
    "snow3lheat_1d",
    "snow3ltemp_1d",
    "snow3lliq_1d",
)


def _compose(path: Path) -> dict[str, str]:
    snow = {
        name: extract_procedure_bytes(SNOW_SOURCE, name) for name in SNOW_PROCEDURES
    }
    thermo = {
        name: extract_procedure_bytes(QSAT_SOURCE, name) for name in THERMO_FUNCTIONS
    }
    source = TEMPLATE.read_bytes()
    source = source.replace(
        b"! <THERMO_FUNCTIONS>",
        b"\n\n".join(thermo[name].span_bytes for name in THERMO_FUNCTIONS),
    ).replace(
        b"! <SNOW_PROCEDURES>",
        b"\n\n".join(snow[name].span_bytes for name in SNOW_PROCEDURES),
    )
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in {**snow, **thermo}.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _inputs() -> dict[str, np.ndarray]:
    n = 10
    zeros = np.zeros(n)
    layers = np.full((n, 3), 0.02)
    layers[0] = 0.0
    layers[1] = [0.01, 0.02, 0.03]
    layers[2] = [0.02, 0.03, 0.05]
    layers[3:5] = [0.005, 0.01, 0.02]
    layers[5] = 4.0
    layers[6] = 3.5
    layers[7:] = 0.0
    rho = np.full((n, 3), 300.0)
    rho[0] = 50.0
    rho[1] = [150.0, 200.0, 250.0]
    rho[2] = [250.0, 300.0, 350.0]
    temp = np.full((n, 3), 268.0)
    temp[0] = 273.15
    temp[1] = [270.0, 269.0, 268.0]
    temp[2] = [272.0, 271.0, 270.0]
    temp[7] = 274.0
    temp[9] = 274.0
    snow_nobio = np.zeros((n, 1))
    snow_nobio[7:, 0] = [1.0, 3500.0, 20.0]
    snow_nobio_age = np.zeros((n, 1))
    snow_nobio_age[8:, 0] = 5.0

    return {
        "precip_rain": np.array([0, 0, 2, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        "precip_snow": np.array([0, 1, 3, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        "temp_air": np.array([275, 268, 271, 268, 268, 265, 265, 276, 265, 275], dtype=float),
        "pb": np.array([1000, 900, 850, 900, 900, 850, 850, 900, 900, 900], dtype=float),
        "u": np.array([0.1, 2, 5, 2, 2, 4, 4, 3, 2, 2], dtype=float),
        "v": np.array([0, 1, .5, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        "temp_sol_new": np.array([275, 270, 273, 268, 268, 265, 265, 278, 265, 275], dtype=float),
        "soilcap": np.array([4e5, 5e5, 6e5, 5e5, 5e5, 6e5, 6e5, 4e5, 5e5, 5e5]),
        "pgflux": np.array([0, -5, 20, 0, 0, -10000, -10000, 50, 0, 0], dtype=float),
        "frac_nobio": np.array([[0], [.1], [.2], [0], [.999999999], [.1], [.1], [.5], [.5], [.5]]),
        "totfrac_nobio": np.array([0, .1, .2, 0, .999999999, .1, .1, .5, .5, .5]),
        "gtemp": np.array([274, 269, 272, 267, 267, 264, 264, 277, 264, 274], dtype=float),
        "lambda_snow": np.ones(n),
        "cgrnd_snow": np.zeros((n, 3)),
        "dgrnd_snow": np.zeros((n, 3)),
        "vevapsno": np.array([0, .2, 1, 20, 1e12, 20, 20, 0, 0, 0], dtype=float),
        "snow_age": zeros.copy(),
        "snow_nobio_age": snow_nobio_age,
        "snow_nobio": snow_nobio,
        "snowrho": rho,
        "snowgrain": np.full((n, 3), 1e-4),
        "snowdz": layers,
        "snowtemp": temp,
        "snowheat": np.zeros((n, 3)),
        "snow": np.sum(rho * layers, axis=1),
        "temp_sol_add": zeros.copy(),
        "snowliq": np.zeros((n, 3)),
        "subsnownobio": np.zeros((n, 1)),
        "grndflux": zeros.copy(),
        "snowmelt": zeros.copy(),
        "soilflxresid": zeros.copy(),
        "subsinksoil": zeros.copy(),
    }


def _jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.hydrol import explicitsnow_main_step

    inputs = _inputs()
    inputs.pop("subsinksoil")
    result = explicitsnow_main_step(**inputs)
    fields = (
        "snow",
        "snowdz",
        "snowrho",
        "snowtemp",
        "snowheat",
        "snowliq",
        "snowgrain",
        "snow_age",
        "snow_nobio",
        "snow_nobio_age",
        "vevapsno",
        "subsnownobio",
        "subsinksoil",
        "grndflux",
        "snowmelt",
        "tot_melt",
        "temp_sol_add",
    )
    return {name: np.asarray(getattr(result, name)).ravel(order="C") for name in fields}


def _write_branch_coverage(output_dir: Path) -> None:
    assignments = {
        (222, "false"): ["zero snow"], (222, "true"): ["cold snowfall"],
        (261, "false"): ["zero snow"], (261, "true"): ["cold snowfall"],
        (265, "false"): ["zero snow"], (265, "true"): ["cold non-biological snow aging"],
        (275, "false"): ["cold snowfall"], (275, "true"): ["shallow-pack sublimation"],
        (276, "false"): ["fully non-biological sublimation boundary"], (276, "true"): ["shallow-pack sublimation"],
        (297, "false"): ["cold snowfall"], (297, "true"): ["extreme overflow above 1.2 maxmass"],
        (303, "false"): ["extreme overflow above 1.2 maxmass"], (303, "true"): ["cold snowfall"],
        (307, "true"): ["extreme overflow above 1.2 maxmass"],
        (329, "false"): ["cold snowfall"], (329, "true"): ["extreme overflow above 1.2 maxmass"],
        (330, "false"): ["moderate overflow above maxmass"], (330, "true"): ["extreme overflow above 1.2 maxmass"],
        (349, "false"): ["extreme overflow above 1.2 maxmass"], (355, "true"): ["extreme overflow above 1.2 maxmass"],
        (422, "false"): ["cold non-biological snow aging"], (422, "true"): ["warm non-biological snow melt"],
        (428, "false"): ["warm non-biological age boundary"], (428, "true"): ["warm non-biological snow melt"],
        (438, "false"): ["warm non-biological age boundary"], (438, "true"): ["cold non-biological snow aging"],
        (448, "false"): ["all cases"],
        (460, "false"): ["cold snowfall"], (460, "true"): ["zero snow"],
        (472, "false"): ["cold non-biological snow aging"], (472, "true"): ["zero snow"],
        (479, "false"): ["warm non-biological age boundary"], (479, "true"): ["cold non-biological snow aging"],
        (493, "false"): ["cold snowfall"], (493, "true"): ["zero snow"],
        (517, "false"): ["all cases"],
        (528, "false"): ["zero snow"], (528, "true"): ["cold snowfall"],
    }
    arms = [
        {"arm_id": f"fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90:{line}:"
         f"{'else_if' if line == 307 else 'if'}:{arm}", "case_ids": cases}
        for (line, arm), cases in assignments.items()
    ]
    document = {
        "schema_version": 1,
        "branch_complete": True,
        "missing_disposition": [],
        "missing_case_assignment": [],
        "ledger_entries": {LEDGER_ENTRY: arms},
    }
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="ascii"
    )


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_explicit_snow_oracle_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        hashes = _compose(source)
        exe = build / "oracle.exe"
        metadata = compile_fortran(source, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(csv_path)
    jax = _jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "zero snow",
                    "cold snowfall",
                    "rain-on-snow melt and sublimation",
                    "shallow-pack sublimation",
                    "fully non-biological sublimation boundary",
                    "extreme overflow above 1.2 maxmass",
                    "moderate overflow above maxmass",
                    "warm non-biological snow melt",
                    "cold non-biological snow aging",
                    "warm non-biological age boundary",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, value, jax[name], rtol=1e-12, atol=1e-14)
        for name, value in fortran.items()
    ]
    _write_branch_coverage(output_dir)
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SNOW_SOURCE.read_bytes()).hexdigest(),
            "span_sha256": hashes,
            "compiler": metadata,
            "path_evidence_ledger_entries": ["hydrol.conditional.explicit_snow_nonzero"],
            "verified_ledger_entries": [LEDGER_ENTRY],
        },
    )


if __name__ == "__main__":
    try:
        result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout, file=sys.stderr)
        raise
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
