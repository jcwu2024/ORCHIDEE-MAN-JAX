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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fortran_oracle_common import compiler_environment  # noqa: E402

SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_soilcarbon_litter.f90.template"
COMPILER = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
SPANS = (
    ("deadleaf", 1314, 1364),
    ("control_moist_func", 1396, 1418),
    ("control_temp_func", 1451, 1511),
    ("control_moist_func_peat", 1516, 1569),
    ("control_moist_func_man", 1576, 1628),
    ("littercalc_leak", 1637, 2873),
    ("control_moist_func_moyano", 2904, 3037),
)


def _span(start: int, end: int) -> bytes:
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    return b"".join(lines[start - 1 : end])


def build_source() -> tuple[bytes, dict[str, str]]:
    hashes: dict[str, str] = {}
    chunks = []
    for name, start, end in SPANS:
        data = _span(start, end)
        if f"{name}".encode() not in data.lower():
            raise ValueError(f"bad span for {name}")
        hashes[name] = hashlib.sha256(data).hexdigest()
        chunks.append(data)
    template = TEMPLATE.read_bytes()
    return template.replace(b"!__ORIGINAL_PROCEDURES__", b"\n".join(chunks)), hashes


def compile_probe() -> dict[str, object]:
    source, hashes = build_source()
    with tempfile.TemporaryDirectory(prefix="orcjax_litter_oracle_") as tmp:
        path = Path(tmp) / "litter_oracle.f90"
        obj = Path(tmp) / "litter_oracle.o"
        path.write_bytes(source)
        completed = subprocess.run(
            [
                str(COMPILER),
                "-std=legacy",
                "-fdefault-real-8",
                "-ffree-line-length-none",
                "-O0",
                "-fcheck=all",
                "-c",
                str(path),
                "-o",
                str(obj),
            ],
            cwd=tmp,
            env=compiler_environment(COMPILER),
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"gfortran exited {completed.returncode}: stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
    return {
        "status": "compiled",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "span_sha256": hashes,
    }


SHAPES = (
    ("litter_above", (2, 2, 14, 1)),
    ("litter_below", (2, 2, 14, 32, 1)),
    ("lignin_struc_above", (2, 14)),
    ("lignin_struc_below", (2, 14, 32)),
    ("litterpart", (2, 14, 2)),
    ("dead_leaves", (2, 14, 2)),
    ("deadleaf_cover", (2,)),
    ("resp_hetero_litter", (2, 14, 2)),
    ("resp_hetero_flood", (2, 14)),
    ("soilcarbon_input", (2, 3, 14)),
    ("soilcarbon_input_doc", (2, 14, 32, 7, 1)),
    ("floodcarbon_input", (2, 14, 7, 1)),
    ("fuel_1hr", (2, 14, 2, 1)),
    ("fuel_10hr", (2, 14, 2, 1)),
    ("fuel_100hr", (2, 14, 2, 1)),
    ("fuel_1000hr", (2, 14, 2, 1)),
    ("litter_avail", (2, 2, 14)),
    ("litter_avail_frac", (2, 2, 14)),
    ("control_temp_below", (2, 32, 14, 14)),
    ("control_moist_above", (2, 14)),
    ("bulk_dens", (2,)),
    ("matrix_a", (2, 14, 7, 7)),
    ("vector_b", (2, 14, 7)),
    ("tsoil_cm", (2, 14, 3)),
    ("soilhum_cm", (2, 14, 3)),
)


def _read_fortran(path: Path) -> list[dict[str, np.ndarray]]:
    values = np.fromfile(path, dtype=np.float64)
    offset = 0
    calls = []
    for _ in range(3):
        call = {}
        for name, shape in SHAPES:
            size = int(np.prod(shape))
            call[name] = values[offset : offset + size].reshape(shape, order="F")
            offset += size
        calls.append(call)
    if offset != values.size:
        raise ValueError(
            f"unexpected Fortran output size {values.size}, expected {offset}"
        )
    return calls


def _initial_inputs() -> dict[str, np.ndarray]:
    turnover = np.zeros((2, 14, 12, 1))
    bm = np.zeros_like(turnover)
    bm[:, 13, 0, 0] = (0.03, 0.02)
    bm[:, 13, 5, 0] = (0.02, 0.015)
    turnover[:, 13, 0, 0] = (0.015, 0.01)
    turnover[:, 13, 5, 0] = (0.01, 0.008)
    litter_above = np.zeros((2, 2, 14, 1))
    litter_above[:, 0, 13, 0] = (4.0, 3.0)
    litter_above[:, 1, 13, 0] = (8.0, 6.0)
    litter_below = np.zeros((2, 2, 14, 32, 1))
    litter_below[:, 0, 13, :, 0] = np.asarray((0.08, 0.06))[:, None]
    litter_below[:, 1, 13, :, 0] = np.asarray((0.12, 0.09))[:, None]
    lignin_above = np.zeros((2, 14))
    lignin_above[:, 13] = 0.2
    lignin_below = np.zeros((2, 14, 32))
    lignin_below[:, 13, :] = 0.2
    litterpart = np.zeros((2, 14, 2))
    litterpart[:, 13, :] = 1.0
    fuel = []
    for _ in range(4):
        item = np.zeros((2, 14, 2, 1))
        item[:, 13, :, 0] = np.swapaxes(litter_above[:, :, 13, 0], 0, 1).T * 0.25
        fuel.append(item)
    return {
        "turnover": turnover,
        "bm": bm,
        "litter_above": litter_above,
        "litter_below": litter_below,
        "lignin_above": lignin_above,
        "lignin_below": lignin_below,
        "litterpart": litterpart,
        "dead": np.zeros((2, 14, 2)),
        "fuel": fuel,
    }


def _jax_calls() -> list[dict[str, np.ndarray]]:
    from jax_orchidee.stomate.carbon_kernels import (
        litter_availability_fraction_step,
        littercalc_aboveground_controls,
        littercalc_leak_core_with_controls,
    )

    inputs = _initial_inputs()
    z_soil = np.asarray(
        (0.0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.0)
    )
    soil_mc = np.full((2, 11, 3), 0.45)
    soil_mc[1] = 0.18
    veget = np.zeros((2, 14))
    veget[:, 13] = (0.7, 0.6)
    sla = np.full((2, 14), 0.02)
    fbact = np.full((2, 32, 14), 0.08)
    poor = np.asarray((0.0, 0.3))
    flood = np.asarray((0.0, 0.4))
    rprof = np.full((2, 14), 0.5)
    results = []
    bulk = np.asarray((1200.0, 1350.0))
    for call_index in range(3):
        do_slow = call_index == 2
        if do_slow:
            inputs["litter_above"] = np.zeros_like(inputs["litter_above"])
            inputs["litter_below"] = np.zeros_like(inputs["litter_below"])
            inputs["fuel"] = [np.zeros_like(item) for item in inputs["fuel"]]
            soil_mc = np.zeros_like(soil_mc)
            bulk = np.asarray((400.0, 1350.0))
        controls = littercalc_aboveground_controls(
            np.asarray((285.0, 268.0)),
            soil_mc,
            z_soil,
            np.asarray((0,) * 13 + (2,)),
            frozen_respiration_func=1,
        )
        result = littercalc_leak_core_with_controls(
            inputs["litter_above"],
            inputs["litter_below"],
            inputs["lignin_above"],
            inputs["lignin_below"],
            inputs["litterpart"],
            inputs["dead"],
            *inputs["fuel"],
            inputs["bm"],
            inputs["turnover"],
            rprof,
            z_soil,
            veget,
            sla,
            fbact,
            poor,
            flood,
            dt_days=1800.0 / 86400.0,
            control_temp_above=np.asarray(controls.control_temp_above),
            control_moist_above=np.asarray(controls.control_moist_above),
            soil_mc_top_by_pft=np.asarray(controls.soil_mc_top_by_pft),
            nslm=11,
            ndeep=32,
            sro_bottom=5,
        )
        current = {
            name: np.asarray(getattr(result, name))
            for name in (
                "litter_above",
                "litter_below",
                "lignin_struc_above",
                "lignin_struc_below",
                "litterpart",
                "dead_leaves",
                "deadleaf_cover",
                "resp_hetero_litter",
                "resp_hetero_flood",
                "soilcarbon_input_doc",
                "floodcarbon_input",
            )
        }
        current.update(
            {
                "fuel_1hr": np.asarray(result.fuel.fuel_1hr),
                "fuel_10hr": np.asarray(result.fuel.fuel_10hr),
                "fuel_100hr": np.asarray(result.fuel.fuel_100hr),
                "fuel_1000hr": np.asarray(result.fuel.fuel_1000hr),
                "soilcarbon_input": np.zeros((2, 3, 14)),
                "litter_avail": np.zeros((2, 2, 14)),
                "litter_avail_frac": np.asarray(
                    litter_availability_fraction_step(
                        litter_above_carbon=current["litter_above"][:, :, :, 0],
                        litter_not_avail=np.zeros((2, 2, 14)),
                        is_tree=np.asarray((False,) * 13 + (True,)),
                        natural=np.asarray((False,) * 13 + (True,)),
                        is_grassland_manag=np.zeros(14, dtype=bool),
                        is_grassland_grazed=np.zeros(14, dtype=bool),
                        is_grassland_cut=np.zeros(14, dtype=bool),
                        do_slow=do_slow,
                    )
                ),
                "control_temp_below": np.zeros((2, 32, 14, 14)),
                "control_moist_above": np.asarray(controls.control_moist_above),
                "bulk_dens": bulk,
                "matrix_a": np.zeros((2, 14, 7, 7)),
                "vector_b": np.zeros((2, 14, 7)),
                "tsoil_cm": np.zeros((2, 14, 3)),
                "soilhum_cm": np.zeros((2, 14, 3)),
            }
        )
        results.append(current)
        inputs.update(
            {
                "litter_above": current["litter_above"],
                "litter_below": current["litter_below"],
                "lignin_above": current["lignin_struc_above"],
                "lignin_below": current["lignin_struc_below"],
                "litterpart": current["litterpart"],
                "dead": current["dead_leaves"],
                "fuel": [
                    current["fuel_1hr"],
                    current["fuel_10hr"],
                    current["fuel_100hr"],
                    current["fuel_1000hr"],
                ],
                "turnover": np.zeros_like(inputs["turnover"]),
                "bm": np.zeros_like(inputs["bm"]),
            }
        )
    return results


def run_oracle(output_dir: Path, compiler: Path = COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source, hashes = build_source()
    with tempfile.TemporaryDirectory(prefix="orcjax_litter_oracle_") as tmp:
        src = Path(tmp) / "litter_oracle.f90"
        exe = Path(tmp) / "litter_oracle.exe"
        raw = Path(tmp) / "outputs.bin"
        src.write_bytes(source)
        subprocess.run(
            [
                str(compiler),
                "-std=legacy",
                "-cpp",
                "-fdefault-real-8",
                "-ffree-line-length-none",
                "-O0",
                "-fcheck=all",
                str(src),
                "-o",
                str(exe),
            ],
            cwd=tmp,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [str(exe), str(raw)],
            cwd=tmp,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
        )
        fortran_calls = _read_fortran(raw)
    jax_calls = _jax_calls()
    comparisons = []
    passed = True
    for index, (fortran, jax) in enumerate(
        zip(fortran_calls, jax_calls, strict=True), start=1
    ):
        for name, _ in SHAPES:
            diff = np.abs(fortran[name] - jax[name])
            scale = np.maximum(np.abs(fortran[name]), np.abs(jax[name]))
            ok = bool(np.all(diff <= 1e-14 + 1e-12 * scale))
            passed &= ok
            relative = np.zeros_like(diff)
            np.divide(diff, scale, out=relative, where=scale > 0)
            comparisons.append(
                {
                    "call": index,
                    "name": name,
                    "passed": ok,
                    "max_abs_error": float(diff.max(initial=0.0)),
                    "max_rel_error": float(relative.max(initial=0.0)),
                }
            )
    result = {
        "status": "passed" if passed else "failed",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "span_sha256": hashes,
        "comparisons": comparisons,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "npts": 2,
                "nvm": 14,
                "ndeep": 32,
                "calls": [
                    {
                        "id": "firstcall",
                        "points": ["dry-pft14", "cold-wet-flood-pft14"],
                    },
                    {
                        "id": "later-call",
                        "turnover": "zero",
                        "bm_to_litter": "zero",
                        "state": "call-1-writeback",
                    },
                    {
                        "id": "dry-zero-pool-daily-boundary",
                        "do_slow": True,
                        "soil_moisture": "zero",
                        "litter_and_fuel": "zero",
                        "bulk_density": [0.4, 1350.0],
                    },
                ],
                "module_state": {
                    "firstcall_litter": True,
                    "frozen_respiration_func": 1,
                    "moist_func_moyano": False,
                    "do_slow": [False, False, True],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "point_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("call", "name", "passed", "max_abs_error", "max_rel_error"),
        )
        writer.writeheader()
        writer.writerows(comparisons)
    if not passed:
        failed = [item for item in comparisons if not item["passed"]]
        raise AssertionError(f"littercalc_leak mismatches: {failed}")
    return result


if __name__ == "__main__":
    print(
        run_oracle(
            ROOT / "outputs/reference_mode/micro_oracles/stomate_littercalc_leak"
        )["status"]
    )
