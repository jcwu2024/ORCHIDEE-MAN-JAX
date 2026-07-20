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
from jax_orchidee.driver.forcing_readers import (  # noqa: E402
    VerticalForcingOptions,
    forcing_just_read,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "readdim2_just_read"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_readdim2_just_read.f90.template"
PROCEDURES = {"forcing_just_read"}
DEPENDENCIES = ("forcing_zoom",)


def compose(path: Path) -> dict[str, str]:
    names = (*DEPENDENCIES, *PROCEDURES)
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


def _field(name: str) -> np.ndarray:
    result = np.empty((2, 2, 4), dtype=np.float64)
    for j in range(2):
        for i in range(2):
            values = {
                "Tair": 270.0 + i + 1 + 2 * (j + 1),
                "Tmin": 1e21 if (i, j) == (1, 1) else 265.0 + i + 1 + j + 1,
                "precip": 0.001 * (i + j + 2),
                "Rainf": 0.0001 * (i + 1),
                "Snowf": 0.00001 * (j + 1),
                "SWdown": 100.0 + 10 * (i + 1) + j + 1,
                "LWdown": 200.0 + 10 * (i + 1) + j + 1,
                "PSurf": 90000.0 + 100 * (i + 1) + j + 1,
                "Qair": 0.005 + 0.0001 * (i + 1),
                "Wind_N": 2.0 + i + 1,
                "Wind_E": -1.0 - (j + 1),
                "Wind": 3.0 + i + j + 2,
                "Levels": 20.0 + i + 1,
                "Levels_uv": 30.0 + j + 1,
                "levels": 40.0 + i + j + 2,
                "SWnet": 50.0 + i + j + 2,
                "Eair": 1000.0 + 10 * (i + 1) + j + 1,
                "petAcoef": 0.1 * (i + 1),
                "peqAcoef": 0.2 * (j + 1),
                "petBcoef": 0.3 * (i + 1),
                "peqBcoef": 0.4 * (j + 1),
                "cdrag": 0.005 * (i + 1),
                "ccanopy": 350.0 + i + j + 2,
            }
            result[i, j, :] = values[name]
    return result


SOURCE_FIELDS = (
    "Tair",
    "Tmin",
    "precip",
    "Rainf",
    "Snowf",
    "SWdown",
    "LWdown",
    "PSurf",
    "Qair",
    "Wind_N",
    "Wind_E",
    "Wind",
    "Levels",
    "Levels_uv",
    "levels",
    "SWnet",
    "Eair",
    "petAcoef",
    "peqAcoef",
    "petBcoef",
    "peqBcoef",
    "cdrag",
    "ccanopy",
)


def _case(
    prefix: str,
    vertical: VerticalForcingOptions,
    *,
    daily: bool,
    wind: bool,
    watch: bool,
) -> dict[str, np.ndarray]:
    packet = forcing_just_read(
        {name: _field(name) for name in SOURCE_FIELDS},
        vertical,
        ttm=4,
        itb=2,
        ite=2,
        daily_interpol=daily,
        wind_n_exists=wind,
        is_watchout=watch,
    )
    names = [
        "zlev",
        "zlev_uv",
        "swdown",
        "rainf",
        "tair",
        "u",
        "v",
        "qair",
        "pb",
        "lwdown",
    ]
    if not daily:
        names.append("snowf")
    if watch:
        names.extend(
            (
                "SWnet",
                "Eair",
                "petAcoef",
                "peqAcoef",
                "petBcoef",
                "peqBcoef",
                "cdrag",
                "ccanopy",
            )
        )
    return {
        f"{prefix}_{name}": np.asarray(getattr(packet, name)).ravel(order="F")
        for name in names
    }


def _jax_outputs() -> dict[str, np.ndarray]:
    return {
        **_case(
            "height",
            VerticalForcingOptions(zheight=True, zsamelev_uv=True, zlev_fixed=2.0),
            daily=False,
            wind=True,
            watch=False,
        ),
        **_case(
            "sigma",
            VerticalForcingOptions(
                zsigma=True,
                zsamelev_uv=False,
                zhybrid_a=100.0,
                zhybrid_b=0.9,
                zhybriduv_a=200.0,
                zhybriduv_b=0.8,
            ),
            daily=True,
            wind=True,
            watch=False,
        ),
        **_case(
            "levels_watch",
            VerticalForcingOptions(zlevels=True, zsamelev_uv=False),
            daily=False,
            wind=True,
            watch=True,
        ),
        **_case(
            "hybrid_scalar",
            VerticalForcingOptions(
                zhybrid=True,
                zsamelev_uv=False,
                zhybrid_a=500.0,
                zhybrid_b=0.7,
                zhybriduv_a=700.0,
                zhybriduv_b=0.6,
            ),
            daily=False,
            wind=False,
            watch=False,
        ),
        **_case(
            "height_distinct",
            VerticalForcingOptions(
                zheight=True,
                zsamelev_uv=False,
                zlev_fixed=2.0,
                zlevuv_fixed=10.0,
            ),
            daily=False,
            wind=True,
            watch=False,
        ),
    }


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_readdim2_just_read_") as td:
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
    comparisons = [
        float_comparison(name, values, jax[name], rtol=1e-12, atol=1e-14)
        for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "height",
                    "daily sigma missing-cell",
                    "levels WATCHOUT",
                    "hybrid scalar wind",
                    "height distinct UV",
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
