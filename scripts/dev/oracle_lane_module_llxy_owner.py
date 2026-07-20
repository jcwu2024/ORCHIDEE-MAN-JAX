from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.geometry_llxy import (  # noqa: E402
    HH,
    PROJ_ALBERS_NAD83,
    PROJ_CASSINI,
    PROJ_CYL,
    PROJ_GAUSS,
    PROJ_LATLON,
    PROJ_LC,
    PROJ_MERC,
    PROJ_PS,
    PROJ_PS_WGS84,
    PROJ_ROTLL,
    ProjectionInfo,
    ij_to_latlon,
    ijll_cassini,
    ijll_cyl,
    ijll_lc,
    ijll_merc,
    ijll_ps,
    ijll_rotlatlon,
    latlon_to_ij,
    llij_cassini,
    llij_cyl,
    llij_gauss,
    llij_lc,
    llij_merc,
    llij_rotlatlon,
    rotate_coords,
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

FAMILY = "module_llxy_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/module_llxy.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_module_llxy_owner.f90.template"
PROCEDURES = {
    "latlon_to_ij",
    "ij_to_latlon",
    "ijll_ps",
    "ijll_lc",
    "llij_lc",
    "llij_merc",
    "ijll_merc",
    "llij_cyl",
    "ijll_cyl",
    "llij_cassini",
    "ijll_cassini",
    "rotate_coords",
    "llij_rotlatlon",
    "ijll_rotlatlon",
    "llij_gauss",
}
DEPENDENCIES = {
    "llij_ps",
    "llij_ps_wgs84",
    "ijll_ps_wgs84",
    "llij_albers_nad83",
    "ijll_albers_nad83",
    "llij_latlon",
    "ijll_latlon",
    "wrf_error_fatal",
}
FATAL_MODES = (
    "uninit_forward",
    "invalid_forward",
    "uninit_inverse",
    "invalid_inverse",
    "gauss_missing",
)


def compose(path: Path) -> dict[str, str]:
    spans = {
        name: extract_procedure_bytes(SOURCE, name)
        for name in PROCEDURES | DEPENDENCIES
    }
    data = TEMPLATE.read_bytes()
    for name, span in spans.items():
        data = data.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(data)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(field_values) for name, field_values in values.items()}


def _projections() -> list[ProjectionInfo]:
    return [
        ProjectionInfo(
            PROJ_LATLON,
            init=True,
            lat1=-10.0,
            lon1=100.0,
            latinc=0.5,
            loninc=2.0,
            knowni=1.0,
            knownj=1.0,
        ),
        ProjectionInfo(
            PROJ_MERC,
            init=True,
            lon1=170.0,
            knowni=3.0,
            knownj=4.0,
            dlon=0.02,
            rsw=-2.5,
        ),
        ProjectionInfo(
            PROJ_PS,
            init=True,
            stdlon=10.0,
            truelat1=60.0,
            hemi=1.0,
            rebydx=100.0,
            polei=7.0,
            polej=8.0,
        ),
        ProjectionInfo(
            PROJ_PS_WGS84,
            init=True,
            stdlon=0.0,
            truelat1=60.0,
            hemi=1.0,
            dx=10000.0,
            knowni=1.0,
            knownj=1.0,
        ),
        ProjectionInfo(
            PROJ_ALBERS_NAD83,
            init=True,
            stdlon=0.0,
            hemi=1.0,
            dx=10000.0,
            knowni=1.0,
            knownj=1.0,
            rho0=100.0,
            nc=0.5,
            bigc=0.5,
        ),
        ProjectionInfo(
            PROJ_LC,
            init=True,
            stdlon=0.0,
            truelat1=30.0,
            truelat2=60.0,
            hemi=1.0,
            cone=0.7,
            rebydx=100.0,
        ),
        ProjectionInfo(
            PROJ_GAUSS,
            init=True,
            nlat=2,
            lon1=0.0,
            loninc=90.0,
            gauss_lat=np.asarray([80.0, 30.0, -30.0, -80.0]),
        ),
        ProjectionInfo(
            PROJ_CYL,
            init=True,
            lat1=-90.0,
            lon1=-180.0,
            latinc=1.0,
            loninc=1.0,
            knowni=1.0,
            knownj=1.0,
        ),
        ProjectionInfo(
            PROJ_CASSINI,
            init=True,
            lat1=-90.0,
            lon1=-180.0,
            lat0=45.0,
            lon0=10.0,
            stdlon=0.0,
            latinc=1.0,
            loninc=1.0,
            knowni=1.0,
            knownj=1.0,
        ),
        ProjectionInfo(
            PROJ_ROTLL,
            init=True,
            ixdim=9,
            jydim=9,
            stagger=HH,
            phi=40.0,
            lambda_=80.0,
            lat1=45.0,
            lon1=0.0,
        ),
    ]


def _jax_outputs() -> dict[str, np.ndarray]:
    p = _projections()
    output: dict[str, list[float]] = {}

    def pair(name: str, value: tuple[object, object]) -> None:
        output.setdefault(f"{name}_a", []).append(float(np.asarray(value[0])))
        output.setdefault(f"{name}_b", []).append(float(np.asarray(value[1])))

    for lat, lon in zip([-30.0, 0.0, 45.0], [179.0, -179.0, 160.0], strict=True):
        result = llij_merc(lat, lon, p[1])
        pair("merc_f", result)
        pair("merc_i", ijll_merc(*result, p[1]))
    for n in range(1, 6):
        result = llij_cyl(-91.0 + n, -541.0 + 180.0 * n, p[7])
        pair("cyl_f", result)
        pair("cyl_i", ijll_cyl(*result, p[7]))
    pair("ps_i", ijll_ps(p[2].polei, p[2].polej, p[2]))
    pair("ps_i", ijll_ps(p[2].polei + 3.0, p[2].polej + 4.0, p[2]))
    pair("ps_i", ijll_ps(p[2].polei + 3.0, p[2].polej - 4.0, p[2]))
    pair("lc_f", llij_lc(30.0, 190.0, p[5]))
    pair("lc_f", llij_lc(30.0, -190.0, p[5]))
    pair("lc_i", ijll_lc(0.0, 0.0, p[5]))
    pair("lc_i", ijll_lc(20.0, -10.0, p[5]))
    pair("lc_i", ijll_lc(20.0, 10.0, replace(p[5], truelat2=30.0)))
    result = llij_cassini(10.0, 20.0, p[8])
    pair("cass_f", result)
    pair("cass_i", ijll_cassini(*result, p[8]))
    unrotated = replace(p[8], lat0=90.0)
    result = llij_cassini(10.0, 20.0, unrotated)
    pair("cass_f", result)
    pair("cass_i", ijll_cassini(*result, unrotated))
    pair("rotate", rotate_coords(10.0, 20.0, 45.0, 10.0, 0.0))
    pair("rotate", rotate_coords(10.0, 20.0, 45.0, 10.0, 0.0, -1))
    pair("rotate", rotate_coords(10.0, 20.0, 45.0, 10.0, 0.0, 1))
    for lat in [90.0, 55.0, 0.0, -90.0]:
        pair("gauss", llij_gauss(lat, 90.0, p[6]))
    result = llij_rotlatlon(45.0, 0.0, p[9])
    pair("rot_f", result)
    pair("rot_i", ijll_rotlatlon(*result, p[9]))
    for projection in p:
        pair("dispatch_f", latlon_to_ij(projection, 10.0, 20.0))
    for projection in p:
        if projection.code != PROJ_GAUSS:
            pair("dispatch_i", ij_to_latlon(projection, 2.0, 3.0))
    return {name: np.asarray(values) for name, values in output.items()}


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_module_llxy_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        compiler_metadata = compile_fortran(source, executable, compiler)
        output_path = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(output_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        for mode in FATAL_MODES:
            fatal = subprocess.run(
                [str(executable), str(build / f"{mode}.csv"), mode],
                cwd=build,
                env=compiler_environment(compiler),
                check=False,
                capture_output=True,
                text=True,
            )
            if fatal.returncode == 0 or "ipslerr" not in fatal.stderr:
                raise RuntimeError(f"{mode} did not reach the source fatal exit")
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs()
    if set(fortran) != set(jax):
        raise RuntimeError(
            f"field mismatch: Fortran-only={sorted(set(fortran) - set(jax))}, "
            f"JAX-only={sorted(set(jax) - set(fortran))}"
        )
    rtol = 1e-12
    atol = 1e-11
    write_point_comparisons(
        output_dir / "point_comparisons.csv",
        fortran,
        jax,
        rtol=rtol,
        atol=atol,
    )
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=rtol, atol=atol)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "all forward dispatch codes",
                    "all inverse dispatch codes",
                    "longitude wrap thresholds",
                    "projection poles and hemispheric branches",
                    "Cassini rotated and unrotated",
                    "rotated HH and VV grid parity",
                    "Gaussian polar and interpolated latitude",
                ],
                "rtol": rtol,
                "atol": atol,
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
            "procedure_span_sha256": span_hashes,
            "compiler": compiler_metadata,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
