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

from jax_orchidee.stomate.daily_inputs import (  # noqa: E402
    stomate_entry_local_prep_explicit,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "stomate_main_entry_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_main_entry_owner.f90.template"
BLOCK_START = 2917
BLOCK_END = 3033


def _block_bytes() -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[BLOCK_START - 1 : BLOCK_END])


def compose(path: Path) -> dict[str, str]:
    block = _block_bytes()
    path.write_bytes(TEMPLATE.read_bytes().replace(b"! <STOMATE_MAIN_ENTRY_BLOCK>", block))
    return {"stomate_main_entry_2917_3033": hashlib.sha256(block).hexdigest()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _inputs() -> dict[str, np.ndarray]:
    npts, nvm = 3, 14
    i = np.arange(1, npts + 1)[:, None]
    j = np.arange(1, nvm + 1)[None, :]
    i12 = np.arange(1, npts + 1)[:, None]
    j12 = np.arange(1, 13)[None, :]
    return {
        "precip_rain": np.asarray([1.0, 2.0, 3.0]),
        "precip_snow": np.asarray([0.5, 0.0, 1.0]),
        "totfrac_nobio": np.asarray([0.25, 1.0, 0.999999995]),
        "totfrac_nobio_new": np.asarray([0.5, 1.0, 0.999999995]),
        "veget": 0.001 * (100 * i + j),
        "veget_max": 0.002 * (100 * i + j),
        "vegetnew_firstday": 0.003 * (100 * i + j),
        "veget_max_new": 0.004 * (100 * i + j),
        "gpp": 0.005 * (100 * i + j),
        "glccNetLCC": 0.01 * (10 * i12 + j12),
        "glccSecondShift": 0.02 * (10 * i12 + j12),
        "glccPrimaryShift": 0.03 * (10 * i12 + j12),
        "harvest_matrix": 0.04 * (10 * i12 + j12),
    }


def _jax_case(*, first: bool, lcchange: bool) -> dict[str, np.ndarray]:
    inputs = _inputs()
    result = stomate_entry_local_prep_explicit(
        precip_rain=inputs["precip_rain"],
        precip_snow=inputs["precip_snow"],
        dt_sechiba=1800.0,
        veget=inputs["veget"],
        veget_max=inputs["veget_max"],
        totfrac_nobio=inputs["totfrac_nobio"],
        gpp=inputs["gpp"],
        glccNetLCC=inputs["glccNetLCC"],
        glccSecondShift=inputs["glccSecondShift"],
        glccPrimaryShift=inputs["glccPrimaryShift"],
        harvest_matrix=inputs["harvest_matrix"],
        date=1 if first else 2,
        vegetnew_firstday=inputs["vegetnew_firstday"],
        totfrac_nobio_new=inputs["totfrac_nobio_new"],
        veget_max_new=inputs["veget_max_new"],
        do_now_stomate_lcchange=lcchange,
        dyn_peat=False,
        update_peatfrac=False,
        min_sechiba=1e-8,
        min_stomate=1e-8,
    )
    return {
        "precip": np.asarray(result.precip),
        "veget_cov": np.asarray(result.veget_cov),
        "veget_cov_max": np.asarray(result.veget_cov_max),
        "gpp_d": np.asarray(result.gpp_d),
        "glccNetLCC": np.asarray(result.glccNetLCC),
        "glccSecondShift": np.asarray(result.glccSecondShift),
        "glccPrimaryShift": np.asarray(result.glccPrimaryShift),
        "harvest_matrix": np.asarray(result.harvest_matrix),
        "vegetnew_firstday": np.asarray(result.vegetnew_firstday),
        "veget_cov_max_new": (
            np.asarray(result.veget_cov_max_new)
            if result.veget_cov_max_new is not None
            else np.full((3, 14), -13.0)
        ),
    }


def _jax_values() -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for prefix, options in {
        "first_lc": {"first": True, "lcchange": True},
        "later_no_lc": {"first": False, "lcchange": False},
    }.items():
        for name, value in _jax_case(**options).items():
            values[f"{prefix}_{name}"] = np.asarray(value).ravel()
    return values


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_main_entry_") as temporary:
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
    fortran_values = _read(output_dir / "fortran_outputs.csv")
    jax_values = _jax_values()
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran_values, jax_values, rtol=1e-12, atol=1e-14)
    comparisons = [float_comparison(name, fortran_values[name], jax_values[name], rtol=1e-12, atol=1e-14) for name in sorted(fortran_values)]
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "cases": ["first day with LC change", "later day without LC change"]}, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {"source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "source_block_sha256": hashes, "source_block_lines": [BLOCK_START, BLOCK_END], "compiler": metadata, "verified_ledger_entries": []})


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
