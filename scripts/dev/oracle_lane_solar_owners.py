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

from jax_orchidee.driver.domain import _fortran_time_zone  # noqa: E402
from jax_orchidee.driver.driver_forcing_completion import (  # noqa: E402
    downward_solar_flux,
    solarang,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "solar_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/solar.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_solar_owners.f90.template"
PROCEDURES = {"solarang", "time_zone", "downward_solar_flux"}
MODES = ("gregorian", "nongregorian")
LONGITUDES = np.asarray([-179.0, -172.0, -120.0, -7.0, 0.0, 7.0, 120.0, 172.0, 179.0])
LATITUDES = np.asarray([-80.0, -21.0, 0.0, 21.0, 80.0])
CLOUD = np.asarray([0.0, 0.01, 0.5, 0.9, 1.0])
SOLAR_JULIANS = (1.01, 80.41666666666667, 172.99, 266.25, 355.75, 365.0)
TIMEZONE_GMTS = (0.0, 12.0, 23.75)


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    source = TEMPLATE.read_bytes()
    for name, span in spans.items():
        source = source.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {field: np.asarray(field_values) for field, field_values in values.items()}


def _solar_expected() -> dict[str, np.ndarray]:
    lon = np.repeat(LONGITUDES[:, None], 3, axis=1)
    lat = np.repeat(np.asarray([-80.0, 0.0, 80.0])[None, :], len(LONGITUDES), axis=0)
    expected: dict[str, np.ndarray] = {}
    for index, julian in enumerate(SOLAR_JULIANS, start=1):
        expected[f"solarang_{index}"] = solarang(
            julian, 1.0, lon, lat, one_year=365.0
        ).ravel(order="F")
    for index, gmt in enumerate(TIMEZONE_GMTS, start=1):
        zone, lhour = _fortran_time_zone(lon, gmt=gmt)
        expected[f"timezone_{index}_zone"] = zone
        expected[f"timezone_{index}_lhour"] = lhour
    return expected


def _flux_expected(mode: str) -> dict[str, np.ndarray]:
    if mode == "gregorian":
        calendar = "gregorian"
        one_year = 365.0
        orbit = {}
    else:
        calendar = "360d"
        one_year = 360.0
        orbit = {"eccentricity": 0.0201, "perihelie": 95.25, "obliquity": 22.75}
    first = downward_solar_flux(
        LATITUDES,
        calendar_str=calendar,
        jday=1.5,
        rtime=12.0,
        cloud=CLOUD,
        nband=2,
        one_year=one_year,
        **orbit,
    )
    later = downward_solar_flux(
        LATITUDES,
        calendar_str="opposite-calendar",
        jday=172.75,
        rtime=0.0,
        cloud=CLOUD,
        nband=3,
        one_year=one_year,
        state=first.state,
    )
    return {
        f"{mode}_first_solad": first.solad.ravel(order="F"),
        f"{mode}_first_solai": first.solai.ravel(order="F"),
        f"{mode}_later_solad": later.solad.ravel(order="F"),
        f"{mode}_later_solai": later.solai.ravel(order="F"),
    }


def run_executable(
    executable: Path, build: Path, compiler: Path, output_dir: Path
) -> dict[str, np.ndarray]:
    combined: dict[str, np.ndarray] = {}
    for mode in MODES:
        output = output_dir / f"fortran_outputs_{mode}.csv"
        subprocess.run(
            [str(executable), str(output.resolve()), mode],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        combined.update(_read_csv(output))
    with (output_dir / "fortran_outputs.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["field", "flat_index", "value"])
        for field, values in combined.items():
            for index, value in enumerate(values):
                writer.writerow([field, index, f"{value:.17e}"])
    return combined


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_solar_owners_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        compiler_metadata = compile_fortran(source, executable, compiler)
        fortran = run_executable(executable, build, compiler, output_dir)

    expected = _solar_expected()
    for mode in MODES:
        expected.update(_flux_expected(mode))
    if set(fortran) != set(expected):
        missing = sorted(set(fortran) ^ set(expected))
        raise RuntimeError(f"solar Oracle field mismatch: {missing}")

    write_point_comparisons(
        output_dir / "point_comparisons.csv",
        fortran,
        expected,
        rtol=1e-12,
        atol=1e-13,
    )
    comparisons = []
    for field in sorted(fortran):
        if field.endswith("_zone"):
            comparisons.append(exact_comparison(field, fortran[field], expected[field]))
        else:
            comparisons.append(
                float_comparison(
                    field,
                    fortran[field],
                    expected[field],
                    rtol=1e-12,
                    atol=1e-13,
                )
            )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "solarang_julians": SOLAR_JULIANS,
                "timezone_gmt": TIMEZONE_GMTS,
                "longitudes": LONGITUDES.tolist(),
                "latitudes": LATITUDES.tolist(),
                "cloud": CLOUD.tolist(),
                "calendars": ["gregorian", "360d"],
                "wavebands": [2, 3],
                "save_calls": ["first", "later"],
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
