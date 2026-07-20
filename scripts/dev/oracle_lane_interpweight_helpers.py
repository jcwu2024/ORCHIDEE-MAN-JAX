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

from jax_orchidee.driver.interpolation import (  # noqa: E402
    interpweight_calc_resolution_in,
    interpweight_masking_input1d,
    interpweight_masking_input2d,
    interpweight_masking_input3d,
    interpweight_masking_input4d,
    interpweight_modifying_input1d,
    interpweight_modifying_input2d,
    interpweight_modifying_input3d,
    interpweight_modifying_input4d,
    interpweight_provide_fractions1d,
    interpweight_provide_fractions2d,
    interpweight_provide_fractions4d,
    interpweight_provide_interpolation2d,
    interpweight_valvecr,
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

FAMILY = "interpweight_helpers"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpweight.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_interpweight_helpers.f90.template"
PROCEDURES = {
    "interpweight_calc_resolution_in",
    "interpweight_modifying_input1d",
    "interpweight_modifying_input2d",
    "interpweight_modifying_input3d",
    "interpweight_modifying_input4d",
    "interpweight_masking_input1d",
    "interpweight_masking_input2d",
    "interpweight_masking_input3d",
    "interpweight_masking_input4d",
    "interpweight_provide_fractions1d",
    "interpweight_provide_fractions2d",
    "interpweight_provide_fractions4d",
    "interpweight_provide_interpolation2d",
    "interpweight_valvecr",
}


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    data = TEMPLATE.read_bytes()
    for name, span in spans.items():
        marker = f"! <{name.upper()}>".encode()
        if data.count(marker) != 1:
            raise RuntimeError(f"template marker missing or duplicated: {marker!r}")
        data = data.replace(marker, span.span_bytes)
    path.write_bytes(data)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _mask_expected() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    values1 = np.asarray([-1.0, 0.0, 2.0])
    initial1 = np.asarray([7, 8, 9])
    for mode, threshold in (("nomask", 0.0), ("mbelow", 0.0), ("mabove", 1.0)):
        _, mask = interpweight_masking_input1d(
            values1, initial1, mode, [threshold, 0.0, 0.0]
        )
        result[f"mask1_{mode.removeprefix('m')}"] = mask

    values2 = np.asarray([[-1.0, 0.0], [2.0, 4.0]])
    initial2 = np.zeros((2, 2), dtype=np.int32)
    for mode, threshold in (("nomask", 0.0), ("mbelow", 0.0), ("mabove", 1.0)):
        _, mask = interpweight_masking_input2d(
            values2, initial2, mode, [threshold, 0.0, 0.0]
        )
        result[f"mask2_{mode.removeprefix('m')}"] = mask.ravel()
    sum2, mask2 = interpweight_masking_input2d(
        np.asarray([[2.0, 2.0], [0.2, 0.2]]), initial2, "msumrange", [3.0, 1.0, 5.0]
    )
    result["mask2_sum_values"] = sum2.ravel()
    result["mask2_sum_mask"] = mask2.ravel()

    values3 = np.asarray([[[0.0, 3.0], [1.0, 1.0]], [[2.0, 0.0], [3.0, 1.0]]])
    for mode in ("mbelow", "mabove", "nomask"):
        _, mask = interpweight_masking_input3d(values3, initial2, mode, [1.0, 0.0, 0.0])
        result[f"mask3_{mode.removeprefix('m')}"] = mask.ravel()
    sum3, mask3 = interpweight_masking_input3d(
        np.asarray([[[1.0, 1.0], [2.0, 2.0]], [[0.2, 0.2], [3.0, 3.0]]]),
        initial2,
        "msumrange",
        [3.0, 1.0, 5.0],
    )
    result["mask3_sum_values"] = sum3.ravel()
    result["mask3_sum_mask"] = mask3.ravel()

    values4 = np.asarray(
        [
            -1.0,
            2.0,
            1.0,
            3.0,
            0.0,
            4.0,
            1.0,
            3.0,
            2.0,
            0.0,
            1.0,
            3.0,
            0.0,
            4.0,
            1.0,
            3.0,
        ]
    ).reshape((2, 2, 2, 2), order="F")
    for mode in ("mbelow", "mabove", "nomask"):
        _, mask = interpweight_masking_input4d(values4, initial2, mode, [1.0, 0.0, 0.0])
        result[f"mask4_{mode.removeprefix('m')}"] = mask.ravel()
    sum4_values = np.full((2, 2, 2, 2), 0.2)
    sum4_values[0, 0, :, 0] = [1.0, 1.0]
    sum4_values[0, 0, :, 1] = [2.0, 2.0]
    sum4_values[0, 1, :, :] = 3.0
    sum4, mask4 = interpweight_masking_input4d(
        sum4_values,
        initial2,
        "msumrange",
        [3.0, 1.0, 5.0],
    )
    result["mask4_sum_values"] = sum4.ravel()
    result["mask4_sum_mask"] = mask4.ravel()
    return result


def _jax_outputs() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    lon = np.repeat(np.asarray([0.0, 2.0, 6.0])[:, None], 3, axis=1)
    lat = np.repeat(np.asarray([10.0, 13.0, 19.0])[None, :], 3, axis=0)
    result["resolution"] = interpweight_calc_resolution_in(
        lon, lat, r_earth=1.0
    ).ravel()
    raw = np.asarray([-2.0, -1.0, 0.0, 3.0])
    result["modify1"] = interpweight_modifying_input1d(raw, -1.0)
    result["modify2"] = interpweight_modifying_input2d(
        raw.reshape(2, 2, order="F"), -1.0
    ).ravel()
    result["modify3"] = interpweight_modifying_input3d(
        raw.reshape(1, 2, 2, order="F"), -1.0
    ).ravel()
    result["modify4"] = interpweight_modifying_input4d(
        raw.reshape(1, 1, 2, 2, order="F"), -1.0
    ).ravel()
    result.update(_mask_expected())

    types = np.asarray([1.0, 2.0])
    area = np.asarray([[0.25, 0.75], [0.0, 0.0]])
    fraction1 = interpweight_provide_fractions1d(
        np.asarray([1.0, 2.0, 3.0]),
        area,
        np.asarray([[1, 2], [1, 1]]),
        types,
        vmin=1.0,
        vmax=3.0,
    )
    result["fraction1_values"] = fraction1.fractions.ravel()
    result["fraction1_availability"] = fraction1.availability
    fraction_indices = np.asarray([1, 1, 1, 1, 2, 1, 1, 1]).reshape(
        (2, 2, 2), order="F"
    )
    fraction2 = interpweight_provide_fractions2d(
        np.asarray([1.0, 2.0, 2.0, 1.0]).reshape((2, 2), order="F"),
        area,
        fraction_indices,
        types,
        vmin=1.0,
        vmax=3.0,
    )
    result["fraction2_values"] = fraction2.fractions.ravel()
    result["fraction2_availability"] = fraction2.availability
    values4 = np.asarray([2.0, 25.0, 0.0, 0.0, 4.0, -1.0, 0.0, 0.0] * 2).reshape(
        (2, 2, 2, 2), order="F"
    )
    fraction4 = interpweight_provide_fractions4d(
        values4,
        area,
        fraction_indices,
        types,
        vmin=np.asarray([1.0, 3.0]),
        vmax=np.asarray([1.0, 1.0]),
    )
    result["fraction4_values"] = fraction4.fractions.ravel()
    result["fraction4_availability"] = fraction4.availability

    values = np.asarray([[2.0, 8.0], [4.0, 16.0]])
    interp_area = np.asarray([[0.25, 0.75, 0.0], [0.0, 1.0, 0.0], [0.4, 0.0, 0.0]])
    indices = np.ones((3, 3, 2), dtype=np.int64)
    indices[0, 1] = [2, 1]
    indices[2, 0] = [2, 2]
    default = interpweight_provide_interpolation2d(
        values, interp_area, indices, defaultval=9.0, default_no_value=2.0
    )
    slope = interpweight_provide_interpolation2d(
        values,
        interp_area,
        indices,
        tint="slopecalc",
        defaultval=9.0,
        default_no_value=2.0,
    )
    result["interpolation_default"] = default.fractions
    result["interpolation_default_availability"] = default.availability
    result["interpolation_slope"] = slope.fractions
    result["interpolation_slope_availability"] = slope.availability

    mixed = np.asarray([0.0, 1.0, 1.0, 2.0])
    full = np.ones(3)
    for operation in ("eq", "ge", "le", "neq"):
        mixed_result = interpweight_valvecr(mixed, 1.0, operation)
        full_values = np.zeros(3) if operation == "neq" else full
        full_result = interpweight_valvecr(full_values, 1.0, operation)
        result[f"val_{operation}_mixed"] = np.concatenate(
            ([mixed_result.count], mixed_result.stored_positions)
        )
        result[f"val_{operation}_full"] = np.concatenate(
            ([full_result.count], full_result.stored_positions)
        )
    return {name: np.asarray(value).ravel() for name, value in result.items()}


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="orchidee_interpweight_helpers_"
    ) as temporary:
        build = Path(temporary)
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
    if set(fortran) != set(jax):
        raise RuntimeError(
            f"output fields differ: Fortran={sorted(fortran)}, JAX={sorted(jax)}"
        )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    discrete = {
        name for name in fortran if name.startswith("mask") and "values" not in name
    } | {name for name in fortran if name.startswith("val_")}
    comparisons = [
        exact_comparison(
            name, fortran[name].astype(np.int64), jax[name].astype(np.int64)
        )
        if name in discrete
        else float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "procedures": sorted(PROCEDURES),
                "cases": [
                    "edge/interior resolution",
                    "rank 1-4 clamp",
                    "rank 1-4 masking modes and threshold arms",
                    "fraction overlap/no-overlap/filtering",
                    "default/slopecalc interpolation",
                    "four ValVecR operations with partial/full matches",
                    "fatal and error arms executed by coverage runner",
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
