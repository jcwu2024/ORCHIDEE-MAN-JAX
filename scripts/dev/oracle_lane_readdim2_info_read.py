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
    ForcingDispatchContext,
    ForcingFields,
    ForcingGridState,
    ForcingOwnerTransition,
    forcing_read,
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

FAMILY = "readdim2_info_read_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_readdim2_info_read.f90.template"
PROCEDURES = {"forcing_read"}


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, "forcing_read")
    path.write_bytes(TEMPLATE.read_bytes().replace(b"! <FORCING_READ>", span.span_bytes))
    return {"forcing_read": span.span_sha256}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _fields(*, weather_itau: int | None = None) -> ForcingFields:
    i = np.arange(1, 3, dtype=np.float64)[:, None]
    j = np.arange(1, 3, dtype=np.float64)[None, :]
    if weather_itau is None:
        tair = 280.0 + i + j
        tair[-1, -1] = 2.0e20
        swdown = 100.0 + i + 10.0 * j
        rainf = 0.01 * np.broadcast_to(i, (2, 2))
        snowf = 0.02 * np.broadcast_to(j, (2, 2))
        u = np.broadcast_to(3.0 + i, (2, 2)).copy()
        v = np.broadcast_to(4.0 + j, (2, 2)).copy()
        qair = np.broadcast_to(0.005 * i, (2, 2)).copy()
        lwdown = 300.0 + i + j
        eair = np.full((2, 2), -77.0)
        watch = dict(
            SWnet=np.broadcast_to(50.0 + i, (2, 2)).copy(), Eair=eair,
            petAcoef=np.ones((2, 2)), peqAcoef=np.full((2, 2), 2.0),
            petBcoef=np.full((2, 2), 3.0), peqBcoef=np.full((2, 2), 4.0),
            cdrag=np.full((2, 2), 5.0), ccanopy=np.full((2, 2), 6.0),
        )
    else:
        tair = 270.0 + i + j
        swdown = 200.0 + weather_itau + np.broadcast_to(i, (2, 2))
        rainf = np.broadcast_to(0.03 * i, (2, 2)).copy()
        snowf = np.broadcast_to(0.04 * j, (2, 2)).copy()
        u = np.broadcast_to(7.0 + i, (2, 2)).copy()
        v = np.broadcast_to(8.0 + j, (2, 2)).copy()
        qair = np.broadcast_to(0.006 * i, (2, 2)).copy()
        lwdown = 310.0 + i + j
        watch = dict(Eair=np.full((2, 2), -15.0))
    return ForcingFields(
        1, 1, np.full((2, 2), 2.0), np.full((2, 2), 10.0), swdown,
        rainf, snowf, tair, u, v, qair, np.full((2, 2), 100000.0 if weather_itau is None else 99000.0),
        lwdown, **watch,
    )


def _interpol_owner(_: ForcingDispatchContext) -> ForcingOwnerTransition:
    ii = np.arange(1, 3)[:, None]
    jj = np.arange(1, 3)[None, :]
    neighbours = np.empty((2, 2, 8), dtype=np.int32)
    for component in range(8):
        neighbours[..., component] = 100 * ii + 10 * jj + component + 1
    resolution = np.empty((2, 2, 2))
    resolution[..., 0] = 1000.0 + ii
    resolution[..., 1] = 2000.0 + jj
    return ForcingOwnerTransition(
        _fields(),
        ForcingGridState(
            0.1 * ii + 0.2 * jj, neighbours, resolution,
            np.asarray([1, 4], dtype=np.int32), 2, True,
        ),
    )


def _weather_owner(context: ForcingDispatchContext) -> ForcingOwnerTransition:
    initial = context.itau_split == 0
    neighbours = np.full((2, 2, 8), -12, dtype=np.int32)
    resolution = np.full((2, 2, 2), -13.0)
    if initial:
        for component in range(8):
            neighbours[0, 0, component] = 11 + component
            neighbours[1, 1, component] = 21 + component
        resolution[0, 0, :] = [111.0, 222.0]
        resolution[1, 1, :] = [333.0, 444.0]
    return ForcingOwnerTransition(
        _fields(weather_itau=context.itauin),
        ForcingGridState(
            np.full((2, 2), -11.0), neighbours, resolution,
            np.asarray([1, 4], dtype=np.int32), 2, initial,
        ),
    )


def _case(prefix: str, *, interpol: bool, watchout: bool, context: ForcingDispatchContext) -> dict[str, np.ndarray]:
    result = forcing_read(
        context=context, interpol=interpol, weathergen=not interpol,
        interpol_owner=_interpol_owner, weathergen_owner=_weather_owner,
        is_watchout=watchout, return_complete=True,
    )
    return {
        f"{prefix}_tair": result.fields.tair.ravel(order="F"),
        f"{prefix}_eair": result.fields.Eair.ravel(order="F"),
        f"{prefix}_contfrac": result.grid.fcontfrac.ravel(order="F"),
        f"{prefix}_meta": np.asarray([result.grid.nbindex, *result.grid.kindex[:2]]),
        f"{prefix}_neighbours": result.grid.fneighbours.ravel(order="F"),
        f"{prefix}_resolution": result.grid.fresolution.ravel(order="F"),
        f"{prefix}_swdown": result.fields.swdown.ravel(order="F"),
        f"{prefix}_rainf": result.fields.rainf.ravel(order="F"),
    }


def _jax_outputs() -> dict[str, np.ndarray]:
    return {
        **_case("interp", interpol=True, watchout=False, context=ForcingDispatchContext(2, 7, 1, False)),
        **_case("watchout", interpol=True, watchout=True, context=ForcingDispatchContext(2, 7, 1, False)),
        **_case("weather_init", interpol=False, watchout=False, context=ForcingDispatchContext(0, 7, 0, True, np.full((2, 2), -11.0))),
        **_case("weather_later", interpol=False, watchout=False, context=ForcingDispatchContext(3, 7, 1, False)),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_readdim2_info_read_") as td:
        build = Path(td)
        source, executable = build / "oracle.f90", build / "oracle.exe"
        hashes = compose(source)
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str((output_dir / "fortran_outputs.csv").resolve())],
            cwd=build, env=compiler_environment(compiler), check=True,
            capture_output=True, text=True,
        )
    fortran, jax = _read(output_dir / "fortran_outputs.csv"), _jax_outputs()
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14)
    integers = {name for name in fortran if name.endswith(("_meta", "_neighbours"))}
    comparisons = [
        exact_comparison(name, values.astype(np.int64), jax[name].astype(np.int64))
        if name in integers else float_comparison(name, values, jax[name], rtol=1e-12, atol=1e-14)
        for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "cases": ["interpol", "watchout", "weather cold start", "weather later"]}, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": hashes, "compiler": metadata,
        "verified_ledger_entries": [],
    })


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
