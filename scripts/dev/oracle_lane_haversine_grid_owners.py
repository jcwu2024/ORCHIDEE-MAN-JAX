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

from jax_orchidee.driver.geometry_haversine import (  # noqa: E402
    haversine_reglatlontoploy,
    haversine_regxytoploy,
    haversine_singlepointploy,
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

FAMILY = "haversine_grid_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/haversine.f90"
LLXY_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/module_llxy.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_haversine_grid_owners.f90.template"
PROCEDURES = {
    "haversine_reglatlontoploy",
    "haversine_regxytoploy",
    "haversine_singlepointploy",
}
DEPENDENCIES = {"haversine_radialdis", "haversine_dtor", "haversine_rtod"}
DISCRETE_SUFFIXES = ("_neighb", "_iorig", "_jorig")


def compose(path: Path) -> dict[str, str]:
    spans = {
        name: extract_procedure_bytes(SOURCE, name)
        for name in PROCEDURES | DEPENDENCIES
    }
    llxy = extract_procedure_bytes(LLXY_SOURCE, "ijll_latlon")
    data = TEMPLATE.read_bytes().replace(b"! <IJLL_LATLON>", llxy.span_bytes)
    for name, span in spans.items():
        data = data.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(data)
    return {
        **{name: span.span_sha256 for name, span in spans.items()},
        "module_llxy::ijll_latlon": llxy.span_sha256,
    }


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _fields(prefix: str, result) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_lonpoly": result.lonpoly.ravel(),
        f"{prefix}_latpoly": result.latpoly.ravel(),
        f"{prefix}_center": result.center.ravel(),
        f"{prefix}_neighb": result.neighb_loc.ravel(),
        f"{prefix}_iorig": result.iorig.ravel(),
        f"{prefix}_jorig": result.jorig.ravel(),
    }


def _jax() -> dict[str, np.ndarray]:
    lon = np.empty((3, 3), dtype=np.float64)
    lat = np.empty((3, 3), dtype=np.float64)
    for j in range(3):
        for i in range(3):
            lon[i, j] = 10.0 + 2.0 * i
            lat[i, j] = 40.0 + 1.5 * j
    indices = np.arange(1, 10)

    def ij_to_latlon(i: float, j: float) -> tuple[float, float]:
        return 30.0 + (j - 1.0) * 1.25, -20.0 + (i - 1.0) * 2.5

    fields: dict[str, np.ndarray] = {}
    fields.update(_fields("lalo_local", haversine_reglatlontoploy(lon, lat, indices)))
    fields.update(
        _fields(
            "lalo_global",
            haversine_reglatlontoploy(lon, lat, indices, global_grid=True),
        )
    )
    fields.update(
        _fields(
            "regxy",
            haversine_regxytoploy(
                lon,
                lat,
                indices,
                projection_code=6,
                ij_to_latlon=ij_to_latlon,
            ),
        )
    )
    fields.update(
        _fields(
            "single",
            haversine_singlepointploy(np.asarray([[10.0]]), np.asarray([[45.0]])),
        )
    )
    return fields


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_haversine_grid_") as temporary:
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
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-12
    )
    comparisons = [
        exact_comparison(name, fortran[name].astype(np.int64), jax[name])
        if name.endswith(DISCRETE_SUFFIXES)
        else float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-12)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "3x3 local regular lon/lat grid",
                    "3x3 globally wrapped regular lon/lat grid",
                    "3x3 projected grid using extracted LATLON projection formula",
                    "single-point fixed-area polygon",
                    "invalid projection fatal contract",
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
            "dependency_source_sha256": hashlib.sha256(
                LLXY_SOURCE.read_bytes()
            ).hexdigest(),
            "procedure_span_sha256": hashes,
            "compiler": compiler_meta,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
