from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.forcing_metadata import forcing_info  # noqa: E402
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

FAMILY = "forcing_info_batch_b_formal"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
TEMPLATE = ROOT / "scripts/dev/forcing_info_batch_b_oracle.f90.template"
PROCEDURES = {"forcing_info"}
HELPERS = (
    "forcing_landind",
    "forcing_grid",
    "forcing_zoom",
    "forcing_vertical_ioipsl",
    "domain_size",
)


def compose(path: Path) -> dict[str, str]:
    source = TEMPLATE.read_bytes()
    hashes: dict[str, str] = {}
    for procedure in (*HELPERS, "forcing_info"):
        span = extract_procedure_bytes(SOURCE, procedure)
        source = source.replace(
            f"! <{procedure.upper()}>".encode("ascii"), span.span_bytes
        )
        hashes[procedure] = span.span_sha256
    path.write_bytes(source)
    return hashes


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values, dtype=np.float64) for name, values in fields.items()}


def _dataset(case: int) -> xr.Dataset:
    lon = np.asarray([-120.0, 0.0, 120.0])
    lat = np.asarray([60.0, 0.0, -60.0])
    nav_lon, nav_lat = np.meshgrid(lon, lat, indexing="xy")
    qair = np.empty((2, 3, 3), dtype=np.float64)
    for time in range(2):
        for j in range(3):
            for i in range(3):
                qair[time, j, i] = 100.0 + 10.0 * (i + 1) + (j + 1) + time
    dt_days = 30.0 if case in (4, 5) else (1.0 if case == 2 else 0.25)
    time = np.asarray([0.0, dt_days])
    data: dict[str, object] = {
        "nav_lon": (("lat", "lon"), nav_lon),
        "nav_lat": (("lat", "lon"), nav_lat),
        "time": (("time",), time),
        "Qair": (("time", "lat", "lon"), qair),
    }
    if case != 2:
        contfrac = np.ones((2, 3, 3), dtype=np.float64)
        contfrac[:, 0, 1] = 0.0
        data["contfrac"] = (("time", "lat", "lon"), contfrac)
    if case == 1:
        data["lev"] = (("lev",), np.asarray([1.0]))
        data["Height_Lev1"] = (("lev",), np.asarray([2.5]))
        data["Height_Levuv"] = (("lev",), np.asarray([12.5]))
    ds = xr.Dataset(data)
    ds["time"].attrs["units"] = "days since 2001-01-01"
    ds["time"].attrs["calendar"] = (
        "gregorian\x00" if case == 2 else ("XXXX\x00" if case == 4 else "gregorian")
    )
    return ds


def _expected_case(case: int) -> dict[str, np.ndarray]:
    ds = _dataset(case)
    result = forcing_info(
        ds,
        date0=10.0,
        allow_weathergen=case in (4, 5),
        dt_weathgen=1800.0,
        limit_west=-10.0 if case not in (4, 5) else -180.0,
        limit_east=120.0 if case not in (4, 5) else 180.0,
        limit_north=60.0 if case not in (4, 5) else 90.0,
        limit_south=-1.0 if case not in (4, 5) else -90.0,
        zonal_res=180.0,
        merid_res=90.0,
        mpi_size=1,
    )
    prefix = f"case{case}_"
    output: dict[str, np.ndarray] = {
        prefix + "meta": np.asarray(
            [
                result.iim,
                result.jjm,
                result.llm,
                result.tm,
                1,
                result.nbpoint,
            ],
            dtype=np.float64,
        ),
        prefix + "dt": np.asarray([result.dt_force]),
        prefix + "date0": np.asarray([result.date0]),
        prefix + "calendar_values": np.asarray([result.one_year, result.one_day]),
        prefix + "flags": np.asarray(
            [
                result.interpol,
                result.daily_interpol,
                result.weathergen,
                result.is_watchout,
                result.have_zaxis,
                result.vertical.zfixed,
                result.vertical.zsigma,
                result.vertical.zhybrid,
                result.vertical.zlevels,
                result.vertical.zheight,
                result.vertical.zsamelev_uv,
            ],
            dtype=np.int32,
        ),
        prefix + "vertical_values": np.asarray(
            [
                result.vertical.zlev_fixed or 0.0,
                result.vertical.zlevuv_fixed or 0.0,
                result.vertical.zhybrid_a or 0.0,
                result.vertical.zhybrid_b or 0.0,
                result.vertical.zhybriduv_a or 0.0,
                result.vertical.zhybriduv_b or 0.0,
            ]
        ),
        prefix + "index_g": result.land_index_fortran.astype(np.float64),
    }
    if result.interpol:
        output[prefix + "i_index"] = result.i_index_fortran.astype(np.float64)
        output[prefix + "j_index_g"] = result.j_index_fortran.astype(np.float64)
        if case == 1:
            raw = np.asarray(ds["contfrac"].isel(time=0).values).T
            output[prefix + "data_full"] = raw.ravel(order="F")
        elif case == 2:
            raw = np.asarray(ds["Qair"].isel(time=0).values).T
            output[prefix + "data_full"] = raw.ravel(order="F")
    if result.weathergen and case == 4:
        # forcing_grid in the extracted source uses merid_res for longitude
        # spacing and zonal_res for latitude spacing; retain that source order.
        weather_lon = np.empty((result.iim, result.jjm), dtype=np.float64)
        weather_lat = np.empty((result.iim, result.jjm), dtype=np.float64)
        for i in range(result.iim):
            weather_lon[i, :] = -180.0 + 90.0 / 2.0 + i * 360.0 / result.iim
        for j in range(result.jjm):
            weather_lat[:, j] = 90.0 - 180.0 / 2.0 - j * 180.0 / result.jjm
        output[prefix + "weather_lon"] = weather_lon.ravel(order="F")
        output[prefix + "weather_lat"] = weather_lat.ravel(order="F")
    return output


def _jax_outputs() -> dict[str, np.ndarray]:
    return {
        name: value
        for case in range(1, 6)
        for name, value in _expected_case(case).items()
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_forcing_info_batch_b_") as td:
        build = Path(td)
        source, executable = build / "oracle.f90", build / "oracle.exe"
        span_hashes = compose(source)
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
    if set(fortran) != set(jax):
        missing = sorted(set(jax) - set(fortran))
        extra = sorted(set(fortran) - set(jax))
        raise RuntimeError(f"output field mismatch; missing={missing}, extra={extra}")
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    integer_fields = {name for name in fortran if name.endswith(("meta", "flags", "index_g", "i_index", "j_index_g"))}
    comparisons = [
        exact_comparison(name, values.astype(np.int64), jax[name].astype(np.int64))
        if name in integer_fields
        else float_comparison(name, values, jax[name], rtol=1e-12, atol=1e-14)
        for name, values in sorted(fortran.items())
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "subdaily root with contfrac and height levels",
                    "daily zero-dt root without contfrac and embedded calendar NUL",
                    "subdaily non-root interpolation",
                    "monthly weather-generator root with missing calendar",
                    "monthly weather-generator non-root",
                ],
                "source_owner": "pft14-owner-contract-d6769359ae0a",
                "tolerances": {"rtol": 1e-12, "atol": 1e-14},
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
            "compiler": metadata,
            "verified_ledger_entries": [],
            "fortran_provenance": {
                "forcing_info": "fortran_source/ORCHIDEE/src_driver/readdim2.f90 lines 49-495",
                "forcing_landind": "fortran_source/ORCHIDEE/src_driver/readdim2.f90 lines 1899-1968",
                "forcing_grid": "fortran_source/ORCHIDEE/src_driver/readdim2.f90 lines 1971-2032",
                "forcing_zoom": "fortran_source/ORCHIDEE/src_driver/readdim2.f90 lines 2034-2051",
                "forcing_vertical_ioipsl": "fortran_source/ORCHIDEE/src_driver/readdim2.f90 lines 2057-2257",
                "domain_size": "fortran_source/ORCHIDEE/src_driver/readdim2.f90 lines 2263-2341",
            },
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
