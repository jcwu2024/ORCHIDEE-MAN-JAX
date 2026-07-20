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

from jax_orchidee.driver.geometry_grid import (  # noqa: E402
    grid_allocate_glo,
    grid_init,
    grid_set_glo,
    grid_stuff,
    grid_toij_1d,
    grid_toij_2d,
    grid_toij_scal,
    grid_topolylist,
)
from jax_orchidee.driver.geometry_llxy import (  # noqa: E402
    PROJ_LATLON,
    ProjectionInfo,
    ij_to_latlon,
    latlon_to_ij,
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
    write_result,
)

FAMILY = "grid_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/grid.f90"
LLXY_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/module_llxy.f90"
HAVERSINE_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/haversine.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_grid_owners.f90.template"
PROCEDURES = {
    "grid_init",
    "grid_set_glo",
    "grid_allocate_glo",
    "grid_stuff",
    "grid_topolylist",
    "grid_scatter",
    "grid_toij_scal",
    "grid_toij_1d",
    "grid_toij_2d",
}


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    generated = TEMPLATE.read_bytes()
    generated = generated.replace(b"! <MODULE_LLXY>", LLXY_SOURCE.read_bytes())
    generated = generated.replace(
        b"! <HAVERSINE_MODULE>", HAVERSINE_SOURCE.read_bytes()
    )
    for name, span in spans.items():
        generated = generated.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(generated)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read_outputs(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _regular_grid() -> tuple[np.ndarray, np.ndarray]:
    lon_axis = np.asarray([107.0, 109.0, 111.0])
    lat_axis = np.asarray([23.0, 21.0, 19.0])
    return np.repeat(lon_axis[:, None], 3, axis=1), np.repeat(
        lat_axis[None, :], 3, axis=0
    )


def _projection() -> ProjectionInfo:
    return ProjectionInfo(
        PROJ_LATLON,
        init=True,
        lat1=40.0,
        lon1=100.0,
        latinc=1.0,
        loninc=2.0,
        knowni=1.0,
        knownj=1.0,
    )


class _ProjectionAdapter:
    def __init__(self, projection: ProjectionInfo) -> None:
        self.projection = projection

    def latlon_to_ij(self, lat: float, lon: float) -> tuple[float, float]:
        i, j = latlon_to_ij(self.projection, lat, lon)
        return float(i), float(j)


def _flatten(value: object) -> np.ndarray:
    return np.asarray(value).ravel(order="F")


def _add_topology(fields: dict[str, np.ndarray], prefix: str, topology: object) -> None:
    fields[f"{prefix}_global"] = _flatten(int(topology.global_grid))
    for name in (
        "corners",
        "neighbours",
        "headings",
        "seglength",
        "area",
        "ilandindex",
        "jlandindex",
    ):
        output_name = {"ilandindex": "iland", "jlandindex": "jland"}.get(name, name)
        fields[f"{prefix}_{output_name}"] = _flatten(getattr(topology, name))


def _jax_outputs() -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}

    global_state = grid_allocate_glo(grid_set_glo(3, 2, 6), 4)
    state = grid_init(
        6, 4, "prefix-RegLonLat-grid", "forcing", global_state=global_state
    )
    fields.update(
        {
            "meta_reg_nseg": _flatten(state.nb_segments),
            "meta_reg_nneigh": _flatten(state.nb_neighbours),
            "meta_reg_global": _flatten(int(state.global_grid)),
            "meta_reg_lalo": _flatten(state.lalo),
            "meta_reg_neighbours": _flatten(state.neighbours),
            "meta_reg_iland": _flatten(state.ilandindex),
        }
    )
    state = grid_init(
        6,
        4,
        "RegLonLat",
        "forcing",
        global_state=global_state,
        isglobal=False,
        existing=state,
    )
    fields["meta_reg_present_global"] = _flatten(int(state.global_grid))
    fields["meta_optional_nbp"] = _flatten(
        grid_set_glo(3, 2, state=global_state).nbp_glo
    )
    preset = replace(grid_set_glo(2, 2, 4), nb_segments=4, nb_neighbours=8)
    allocated = grid_allocate_glo(preset, 4)
    fields["meta_allocate_existing_nseg"] = _flatten(allocated.nb_segments)
    fields["meta_allocate_neigh_size"] = _flatten(allocated.neighbours.shape[1])
    xy = grid_init(4, 4, "RegXY", "xy", global_state=allocated, isglobal=True)
    fields["meta_xy_global"] = _flatten(int(xy.global_grid))
    xy_default_global = grid_allocate_glo(grid_set_glo(2, 2, 4), 4)
    xy_default = grid_init(4, 4, "RegXY", "xy_default", global_state=xy_default_global)
    fields["meta_xy_default_global"] = _flatten(int(xy_default.global_grid))
    unstruct_global = grid_allocate_glo(grid_set_glo(1, 1, 1), 5)
    unstruct = grid_init(1, 5, "UnStruct", "mesh", global_state=unstruct_global)
    fields["meta_unstruct_nseg"] = _flatten(unstruct.nb_segments)
    fields["meta_unstruct_global"] = _flatten(int(unstruct.global_grid))
    unstruct = grid_init(
        1,
        5,
        "UnStruct",
        "mesh",
        global_state=unstruct_global,
        isglobal=False,
        existing=unstruct,
    )
    fields["meta_unstruct_present_global"] = _flatten(int(unstruct.global_grid))

    lon, lat = _regular_grid()
    index = np.asarray([1, 5, 9], dtype=np.int32)
    _add_topology(
        fields,
        "top_reg",
        grid_topolylist("RegLonLat", 4, 3, 3, 3, lon, lat, index),
    )
    _add_topology(
        fields,
        "top_single",
        grid_topolylist(
            "RegLonLat",
            4,
            1,
            1,
            1,
            np.asarray([[109.0]]),
            np.asarray([[21.0]]),
            np.asarray([1], dtype=np.int32),
        ),
    )
    projection = _projection()
    axis_i, axis_j = np.meshgrid(
        np.arange(1.0, 4.0), np.arange(1.0, 4.0), indexing="ij"
    )
    xy_lat, xy_lon = ij_to_latlon(projection, axis_i, axis_j)
    dx = np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]])
    dy = np.asarray([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    _add_topology(
        fields,
        "top_xy",
        grid_topolylist(
            "RegXY",
            4,
            3,
            3,
            3,
            xy_lon,
            xy_lat,
            index,
            projection=projection,
            ij_to_latlon=lambda i, j: ij_to_latlon(projection, i, j),
            dxwrf=dx,
            dywrf=dy,
        ),
    )

    def initialized() -> object:
        owner = grid_allocate_glo(grid_set_glo(3, 3, 3), 4)
        owner = replace(
            owner,
            index=index.copy(),
            lon=lon.copy(),
            lat=lat.copy(),
            lalo=np.column_stack(
                (lat.ravel(order="F")[index - 1], lon.ravel(order="F")[index - 1])
            ),
        )
        return grid_init(3, 4, "RegLonLat", "forcing", global_state=owner)

    for prefix, fractions, retained_indices in (
        ("stuff_present", np.asarray([1.0, 0.4, 0.0]), None),
        ("stuff_absent", np.asarray([0.2, 0.3, 0.4]), ([9, 8, 7], [6, 5, 4])),
    ):
        initial = initialized()
        initial = replace(
            initial,
            global_state=replace(initial.global_state, contfrac=fractions.copy()),
        )
        if retained_indices is not None:
            initial = replace(
                initial,
                ilandindex=np.asarray(retained_indices[0], dtype=np.int32),
                jlandindex=np.asarray(retained_indices[1], dtype=np.int32),
            )
        completed = grid_stuff(
            initial,
            3,
            3,
            3,
            lon,
            lat,
            index,
            None if retained_indices is not None else fractions,
        )
        for name in (
            "neighbours",
            "headings",
            "seglength",
            "corners",
            "area",
            "resolution",
            "contfrac",
            "ilandindex",
            "jlandindex",
        ):
            output_name = {"ilandindex": "iland", "jlandindex": "jland"}.get(name, name)
            fields[f"{prefix}_{output_name}"] = _flatten(getattr(completed, name))

    adapter = _ProjectionAdapter(projection)
    ri, rj = grid_toij_scal(108.0, 45.0, projection=adapter)
    fields["toij_scal_ri"] = _flatten(ri)
    fields["toij_scal_rj"] = _flatten(rj)
    ri, rj = grid_toij_1d(
        np.asarray([100.0, 104.0, 108.0]),
        np.asarray([40.0, 42.0, 45.0]),
        projection=adapter,
    )
    fields["toij_1d_ri"] = _flatten(ri)
    fields["toij_1d_rj"] = _flatten(rj)
    ri, rj = grid_toij_2d(
        np.asarray([[100.0, 104.0], [102.0, 106.0]]),
        np.asarray([[40.0, 42.0], [41.0, 43.0]]),
        projection=adapter,
    )
    fields["toij_2d_ri"] = _flatten(ri)
    fields["toij_2d_rj"] = _flatten(rj)
    return fields


def _is_discrete(name: str) -> bool:
    return (
        any(
            token in name
            for token in (
                "_global",
                "_nseg",
                "_nneigh",
                "_neigh_size",
                "neighbours",
                "_iland",
                "_jland",
            )
        )
        or name == "meta_optional_nbp"
    )


def _write_points(
    path: Path, fortran: dict[str, np.ndarray], jax: dict[str, np.ndarray]
) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "field",
                "flat_index",
                "fortran",
                "jax",
                "abs_error",
                "rel_error",
                "passed",
            ]
        )
        for field in sorted(fortran):
            actual = np.asarray(fortran[field]).ravel()
            expected = np.asarray(jax[field]).ravel()
            if actual.shape != expected.shape:
                raise ValueError(
                    f"{field}: shape mismatch {actual.shape} != {expected.shape}"
                )
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                absolute = abs(float(left) - float(right))
                relative = absolute / max(abs(float(left)), np.finfo(np.float64).tiny)
                passed = (
                    left == right
                    if _is_discrete(field)
                    else np.isclose(left, right, rtol=1e-12, atol=1e-9)
                )
                writer.writerow(
                    [
                        field,
                        index,
                        left,
                        right,
                        absolute,
                        relative,
                        str(bool(passed)).lower(),
                    ]
                )


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_grid_owners_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        compiler_metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str((output_dir / "fortran_outputs.csv").resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read_outputs(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs()
    if set(fortran) != set(jax):
        raise ValueError(
            f"field mismatch: missing_jax={sorted(set(fortran) - set(jax))}, "
            f"missing_fortran={sorted(set(jax) - set(fortran))}"
        )
    _write_points(output_dir / "point_comparisons.csv", fortran, jax)
    comparisons = [
        exact_comparison(name, fortran[name], jax[name])
        if _is_discrete(name)
        else float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-9)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "metadata allocation and retained allocation",
                    "RegLonLat multi-point and single-point topology",
                    "RegXY topology with real module_llxy projection",
                    "global longitude tests on both conventions",
                    "grid_stuff present/absent continent fraction and retained indices",
                    "scalar/rank-1/rank-2 grid_toij",
                ],
                "undefined_allocation_policy": (
                    "grid_allocate_glo values are undefined by Fortran; only allocation "
                    "shape and module metadata are compared before population"
                ),
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
            "scientific_dependency_sha256": {
                "module_llxy.f90": hashlib.sha256(LLXY_SOURCE.read_bytes()).hexdigest(),
                "haversine.f90": hashlib.sha256(
                    HAVERSINE_SOURCE.read_bytes()
                ).hexdigest(),
            },
            "compiler": compiler_metadata,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
