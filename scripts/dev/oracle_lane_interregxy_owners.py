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

from jax_orchidee.driver.geometry_interregxy import (  # noqa: E402
    R_EARTH,
    UNDEFINED_INDEX,
    interregxy_aggr2d,
    interregxy_aggrve,
    interregxy_findpoints,
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

FAMILY = "interregxy_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interregxy.f90"
POLYGONES = ROOT / "fortran_source/ORCHIDEE/src_global/polygones.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_interregxy_owners.f90.template"
PROCEDURES = {"interregxy_aggr2d", "interregxy_aggrve", "interregxy_findpoints"}
POLYGON_CLOSURE = {
    "polygones_pointinside",
    "polygones_lineintersect",
    "polygones_extend",
    "polygones_intersection",
    "polygones_cleanup",
    "polygones_area",
    "polygones_crossing",
    "polygones_convexhull",
}
DISCRETE = {
    "aggr2d_ok",
    "aggr2d_ind",
    "aggrve_ok",
    "aggrve_ind",
    "find_nbpt",
    "find_sourcei",
    "find_sourcej",
}


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    spans.update(
        {name: extract_procedure_bytes(POLYGONES, name) for name in POLYGON_CLOSURE}
    )
    data = TEMPLATE.read_bytes()
    for name, span in spans.items():
        data = data.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    if b"! <" in data:
        raise ValueError("unreplaced extracted-procedure marker")
    path.write_bytes(data)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _jax() -> dict[str, np.ndarray]:
    lalo = np.asarray([[0.0, 0.0], [3.0, 3.0]])
    common = dict(
        nbpt=2,
        lalo=lalo,
        neighbours=np.zeros((2, 4), dtype=np.int32),
        resolution=np.ones((2, 2)),
        contfrac=np.ones(2),
        incmax=8,
        grid_toij=lambda x, y: (x + 0.1, y + 0.1),
        land_indices=np.asarray([[1, 1], [2, 2]]),
        dx_wrf=np.full((3, 3), 10.0),
        dy_wrf=np.full((3, 3), 20.0),
    )
    two = interregxy_aggr2d(
        iml=2,
        jml=2,
        lon_rel=np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        lat_rel=np.asarray([[1.0, 2.0], [1.0, 2.0]]),
        mask=np.ones((2, 2), dtype=np.int32),
        callsign="oracle",
        **common,
    )
    width = 0.8 * R_EARTH / (180.0 / (2.0 * np.pi))
    vec = interregxy_aggrve(
        iml=2,
        lon_rel=np.asarray([1.0, 2.0]),
        lat_rel=np.asarray([1.0, 2.0]),
        resol_lon=width,
        resol_lat=width,
        callsign="oracle",
        **common,
    )
    find = interregxy_findpoints(
        1,
        1,
        0.0,
        0.0,
        7,
        9,
        np.asarray([1.0, 0.5, 1.5, 1.5, 0.5]),
        np.asarray([1.0, 0.5, 0.5, 1.5, 1.5]),
        8,
        False,
        np.zeros((1, 1), dtype=np.int32),
        np.full((1, 1, 8), UNDEFINED_INDEX, dtype=np.int32),
        np.full((1, 1, 8), UNDEFINED_INDEX, dtype=np.int32),
        np.full((1, 1, 8), np.nan),
        dx_wrf=np.full((1, 1), 10.0),
        dy_wrf=np.full((1, 1), 20.0),
    )
    return {
        "aggr2d_ok": np.asarray([two.ok]),
        "aggr2d_area": two.areaoverlap.ravel(order="F"),
        "aggr2d_ind": np.where(two.areaoverlap[..., None] > 0, two.indinc, -999).ravel(
            order="F"
        ),
        "aggrve_ok": np.asarray([vec.ok]),
        "aggrve_area": vec.areaoverlap.ravel(order="F"),
        "aggrve_ind": np.where(vec.areaoverlap > 0, vec.indinc, -999).ravel(order="F"),
        "find_nbpt": find.nbpt.ravel(order="F"),
        "find_sourcei": np.where(
            find.nbpt[..., None] > np.arange(8), find.sourcei, -999
        ).ravel(order="F"),
        "find_sourcej": np.where(
            find.nbpt[..., None] > np.arange(8), find.sourcej, -999
        ).ravel(order="F"),
        "find_overlap": np.where(
            find.nbpt[..., None] > np.arange(8), find.overlap_area, -999.0
        ).ravel(order="F"),
    }


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_interregxy_") as temporary:
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
    fortran, jax = _read(output_dir / "fortran_outputs.csv"), _jax()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        exact_comparison(name, fortran[name].astype(np.int64), jax[name])
        if name in DISCRETE
        else float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "2-D source grid, both latitude order arms and border corrections",
                    "vector source grid with latitude-dependent longitude width",
                    "direct convex cell overlap with all findpoints inout state",
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
            "polygones_source_file_sha256": hashlib.sha256(
                POLYGONES.read_bytes()
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
