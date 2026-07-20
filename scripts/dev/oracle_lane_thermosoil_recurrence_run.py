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

from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "thermosoil_cwrr_recurrence_defined_tail"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_thermosoil_recurrence_zimov.f90.template"


def _span(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _main_tail() -> bytes:
    # Preserve source bytes while omitting intervening history IO and the coef call,
    # neither of which is claimed by this component oracle.
    return _span(990, 993) + b"\n" + _span(1011, 1033)


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            key = f"case{row['case']}.{row['field']}"
            values.setdefault(key, []).append(float(row["value"]))
    return {key: np.asarray(value) for key, value in values.items()}


def _jax() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.thermosoil import add_heat_zimov, thermosoil_final_state

    npts, ngrnd, nvm = 2, 4, 14
    i, j, k = np.indices((npts, ngrnd, nvm))
    ptn0 = 260.0 + (i + 1) + 0.1 * (j + 1) + 0.01 * (k + 1)
    heat = (-1.0) ** ((i + 1) + (j + 1) + (k + 1)) * (0.02 + 0.001 * (j + 1))
    pcapa = 2.0e6 + 1000.0 * (i + 1) + 100.0 * (j + 1) + 10.0 * (k + 1)
    dlt = np.array([0.1, 0.3, 0.8, 1.7])
    mask = np.ones((npts, nvm), dtype=bool)
    mask[0, 3] = False
    mask[1, 13] = False
    veg = np.zeros((npts, nvm))
    veg[:, 0] = 0.2
    veg[:, 13] = 0.8
    pkappa = 0.5 + 0.01 * (i + 1) + 0.02 * (j + 1) + 0.001 * (k + 1)
    humidity = 0.1 * (i + 1) + 0.01 * (j + 1) + 0.001 * (k + 1)
    result: dict[str, np.ndarray] = {}
    for case, enabled in enumerate((False, True)):
        if enabled:
            heated = add_heat_zimov(
                ptn=ptn0, heat_zimov=heat, pcapa=pcapa, dlt=dlt,
                dt_sechiba=1800.0, veget_mask_2d=mask, veget_max_bg=veg,
            ).ptn
        else:
            heated = ptn0
        final = thermosoil_final_state(
            ptn=heated, pkappa=pkappa, veget_max=veg,
            shum_ngrnd_permalong=humidity,
        )
        prefix = f"case{case}."
        result[prefix + "ptn"] = np.asarray(heated).transpose(2, 1, 0).ravel()
        result[prefix + "ptn_pftmean"] = np.asarray(final.ptn_pftmean).T.ravel()
        result[prefix + "pkappa_pftmean"] = np.asarray(final.pkappa_pftmean).T.ravel()
        result[prefix + "gtemp"] = np.asarray(final.gtemp).ravel()
        result[prefix + "ptnlev1"] = np.asarray(final.ptnlev1).ravel()
    return result


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    add_heat = _span(3426, 3463)
    tail = _main_tail()
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_recurrence_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(
            TEMPLATE.read_bytes()
            .replace(b"! <ADD_HEAT_ZIMOV>", add_heat)
            .replace(b"! <MAIN_ZIMOV_AND_FINAL_STATE>", tail)
        )
        executable = build / "oracle.exe"
        compiler_meta = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str(csv_path.resolve())], cwd=build,
            env=compiler_environment(compiler), check=True, capture_output=True, text=True,
        )
    fortran = _read(csv_path)
    jax = _jax()
    (output_dir / "inputs.json").write_text(json.dumps({
        "schema_version": 1,
        "cases": [
            {"id": "ok_zimov_explicit_false", "meaning": "defined bypass arm"},
            {"id": "ok_zimov_explicit_true", "meaning": "real add_heat_Zimov callee"},
        ],
        "note": "Explicit harness control does not assign a meaning to thermosoil_main's uninitialized local SAVE ok_zimov.",
    }, indent=2) + "\n", encoding="ascii")
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14)
    comparisons = [float_comparison(key, fortran[key], jax[key], rtol=1e-12, atol=1e-14) for key in fortran]
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": {
            "add_heat_Zimov": hashlib.sha256(add_heat).hexdigest(),
            "thermosoil_main_defined_tail_fragments": hashlib.sha256(tail).hexdigest(),
        },
        "compiler": compiler_meta,
        "component_status": "verified" if all(item["passed"] for item in comparisons) else "failed",
        "ledger_status": "partial",
        "verified_ledger_entries": [],
        "fortran_undefined_contract": ["thermosoil_main.local_ok_zimov"],
    })


if __name__ == "__main__":
    try:
        result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout, file=sys.stderr)
        raise
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
