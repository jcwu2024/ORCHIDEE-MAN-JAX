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

from jax_orchidee.driver.orchestration import prepare_paper_1961_driver_context  # noqa: E402
from jax_orchidee.driver.static import read_paper_thermosoil_refsoc_static_field  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "thermosoil_refsoc_overlap_actual"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpol_help.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_thermosoil_refsoc_overlap_actual.f90.template"
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
REFSOC = ROOT / "data/MICT_BIOE/Input/refSOC_NCSCD_linear_05deg_v5.nc"
TARGETS = {
    "069.0-119.0": ROOT
    / "outputs/paper_250919_materialized_run_defs/arg2_1.0/069.0-119.0/I7/S15_69.354_0.0033_0.1089_92.010/used_run.def",
    "319.0-057.0": ROOT
    / "outputs/paper_250919_materialized_run_defs/arg2_1.0/319.0-057.0/I5/S26_69.834_0.0019_0.3980_97.918/used_run.def",
}


def _span(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def _write_fortran_map(path: Path) -> None:
    with xr.open_dataset(REFSOC, decode_times=False) as dataset:
        lon_1d = np.asarray(dataset["longitude"].values, dtype=np.float64)
        lat_1d = np.asarray(dataset["latitude"].values, dtype=np.float64)
        lon, lat = np.meshgrid(lon_1d, lat_1d, indexing="xy")
        lon = np.asfortranarray(lon.T)
        lat = np.asfortranarray(lat.T)
        mask = np.asfortranarray((np.asarray(dataset["mask"].values).T > 0).astype(np.int32))
        refsoc = np.asfortranarray(
            np.asarray(dataset["soil_organic_carbon"].isel(time_counter=0).values, dtype=np.float64).transpose(2, 1, 0)
        )
    with path.open("wb") as handle:
        for value in (lon, lat, mask, refsoc):
            value.ravel(order="F").tofile(handle)


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _span(46, 466)
    fortran: dict[str, np.ndarray] = {}
    jax: dict[str, np.ndarray] = {}
    target_inputs: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="orchidee_refsoc_overlap_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        source.write_bytes(TEMPLATE.read_bytes().replace(b"! <AGGREGATE_2D>", aggregate))
        executable = build / "oracle.exe"
        compiler_meta = compile_fortran(source, executable, compiler, extra_flags=("-cpp",))
        map_path = build / "refsoc.bin"
        _write_fortran_map(map_path)
        for point_id, run_def in TARGETS.items():
            context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=run_def)
            domain = context.first_step_bundle.domain
            lalo = np.asarray(domain.lalo, dtype=np.float64).reshape((-1, 2))
            resolution = np.asarray(domain.resolution, dtype=np.float64).reshape((-1, 2))
            contfrac = np.asarray(domain.contfrac, dtype=np.float64).reshape((-1,))
            metadata_path = build / f"{point_id}.dat"
            metadata_path.write_text(
                f"{lalo[0, 0]:.17e} {lalo[0, 1]:.17e} "
                f"{resolution[0, 0]:.17e} {resolution[0, 1]:.17e} {contfrac[0]:.17e}\n",
                encoding="ascii",
            )
            csv_path = build / f"{point_id}.csv"
            subprocess.run(
                [str(executable), str(metadata_path), str(map_path), str(csv_path)],
                cwd=build,
                env=compiler_environment(compiler),
                check=True,
                capture_output=True,
                text=True,
            )
            point_fortran = _read_csv(csv_path)
            refsoc, overlap = read_paper_thermosoil_refsoc_static_field(
                CONFIG,
                lalo=lalo,
                resolution_m=resolution,
            )
            active = overlap["refsoc_sub_area"][0] > 0.0
            index = overlap["refsoc_sub_index"][0, active]
            area = overlap["refsoc_sub_area"][0, active]
            point_jax = {
                "refsoc": np.asarray(refsoc[0], dtype=np.float64),
                "count": np.asarray([len(area)], dtype=np.float64),
                "ip": index[:, 0].astype(np.float64),
                "jp": index[:, 1].astype(np.float64),
                "area": area,
            }
            for name, values in point_fortran.items():
                fortran[f"{point_id}.{name}"] = values
                jax[f"{point_id}.{name}"] = point_jax[name]
            target_inputs.append(
                {
                    "landpoint_id": point_id,
                    "lalo": lalo.tolist(),
                    "resolution_m": resolution.tolist(),
                    "contfrac": contfrac.tolist(),
                }
            )

    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, "targets": target_inputs}, indent=2) + "\n",
        encoding="ascii",
    )
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-12)
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-12)
        for name in fortran
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": {"aggregate_2d": hashlib.sha256(aggregate).hexdigest()},
            "compiler": compiler_meta,
            "scope": "actual 069/319 cold-start refSOC geometry and complete 32-layer profile",
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
