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
from jax_orchidee.driver.forcing_domain_io import (  # noqa: E402
    ForcingDomainIOError,
    domain_size,
)
from jax_orchidee.driver.forcing_readers import forcing_grid, forcing_landind  # noqa: E402
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

FAMILY = "readdim2_geometry_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_readdim2_geometry.f90.template"
PROCEDURES = {"domain_size", "forcing_landind", "forcing_grid"}
DEPENDENCIES = ("forcing_zoom",)


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


def _domain_arrays() -> tuple[np.ndarray, np.ndarray]:
    lon_axis = np.asarray([-200.0, -150.0, -90.0, 0.0, 30.0, 170.0, 200.0])
    lat_axis = np.asarray([60.0, 30.0, 0.0, -30.0, -60.0])
    return np.broadcast_to(lon_axis[:, None], (7, 5)), np.broadcast_to(
        lat_axis[None, :], (7, 5)
    )


def _domain_outputs(
    prefix: str, limits: tuple[float, float, float, float]
) -> dict[str, np.ndarray]:
    lon, lat = _domain_arrays()
    result = domain_size(*limits, lon, lat)
    return {
        f"{prefix}_counts": np.asarray(
            [
                result.iim,
                result.jjm,
                result.iim_g_begin,
                result.iim_g_end,
                result.jjm_g_begin,
                result.jjm_g_end,
            ]
        ),
        f"{prefix}_iind": result.iind,
        f"{prefix}_jind": result.jind,
    }


def _jax_outputs() -> dict[str, np.ndarray]:
    outputs = {
        **_domain_outputs("normal", (-100.0, 10.0, 40.0, -40.0)),
        **_domain_outputs("dateline", (100.0, -100.0, 40.0, -40.0)),
    }
    lon_axis = -180.0 + 30.0 * np.arange(7)
    lat_axis = 80.0 - 20.0 * np.arange(5)
    lon_full = lon_axis[:, None] + 0.1 * np.arange(1, 6)[None, :]
    lat_full = lat_axis[None, :] + 0.01 * np.arange(1, 8)[:, None]
    grid_lon, grid_lat = forcing_grid(
        mode="interpol",
        iim=3,
        jjm=3,
        lon_full=lon_full,
        lat_full=lat_full,
        i_indices_one_based=[2, 4, 6],
        j_indices_one_based=[1, 3, 5],
    )
    outputs["grid_lon"] = grid_lon.ravel(order="F")
    outputs["grid_lat"] = grid_lat.ravel(order="F")
    for name, tair in {
        "celsius": np.asarray(
            [20.0, 150.0, 30.0, 140.0, 40.0, 130.0, 50.0, 120.0, 60.0]
        ).reshape((3, 3), order="F"),
        "kelvin": np.asarray(
            [280.0, 999999.0, 290.0, 510.0, 300.0, 520.0, 310.0, 530.0, 320.0]
        ).reshape((3, 3), order="F"),
    }.items():
        result = forcing_landind(tair)
        outputs[f"{name}_meta"] = np.asarray(
            [result.nbindex, result.i_test, result.j_test]
        )
        outputs[f"{name}_kindex"] = result.kindex
    try:
        domain_size(-100.0, 10.0, -20.0, 20.0, *_domain_arrays())
    except ForcingDomainIOError:
        outputs["invalid_lat_error"] = np.asarray([1])
    else:
        outputs["invalid_lat_error"] = np.asarray([0])
    return outputs


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_readdim2_geometry_") as td:
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
    integer_fields = {name for name in fortran if name not in {"grid_lon", "grid_lat"}}
    comparisons = [
        exact_comparison(name, values.astype(np.int64), jax[name].astype(np.int64))
        if name in integer_fields
        else float_comparison(name, values, jax[name], rtol=1e-12, atol=1e-14)
        for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "normal zoom",
                    "dateline zoom",
                    "invalid latitude",
                    "Celsius land mask",
                    "Kelvin land mask",
                    "interpol grid zoom",
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
            "compiler": metadata,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
