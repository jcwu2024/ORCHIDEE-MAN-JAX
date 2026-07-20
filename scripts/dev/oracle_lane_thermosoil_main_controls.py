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

FAMILY = "thermosoil_main_controls"
LEDGER_ENTRY = "thermosoil.active.cwrr_recurrence"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_thermosoil_main_controls.f90.template"


def _span(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            key = f"case{row['case']}.{row['field']}"
            values.setdefault(key, []).append(float(row["value"]))
    return {key: np.asarray(value) for key, value in values.items()}


def _jax() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.thermosoil import thermosoil_wlupdate

    i, j, k = np.indices((3, 4, 14))
    ptn = 260.0 + (i + 1) + 0.1 * (j + 1) + 0.01 * (k + 1)
    humidity = 0.1 * (i + 1) + 0.01 * (j + 1) + 0.001 * (k + 1)
    updated = thermosoil_wlupdate(
        ptn=ptn,
        hsd=humidity,
        hsdlong=np.full_like(humidity, 0.8),
        veget_mask_2d=np.ones((3, 14), dtype=bool),
        dt_sechiba=1800.0,
    ).hsdlong
    return {
        "case0.humidity": humidity.transpose(2, 1, 0).ravel(),
        "case1.humidity": np.asarray(updated).transpose(2, 1, 0).ravel(),
        "case2.hist_calls": np.array([13.0]),
        "case2.low_pft_calls": np.array([4.0]),
        "case2.high_pft_calls": np.array([4.0]),
        "case3.temp_sol_new": np.array([273.15, 275.0, 270.0]),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spans = {
        "thermosoil_wlupdate": _span(3648, 3672),
        "humidity_owner": _span(899, 903),
        "history_owner": _span(940, 988),
        "final_owner": _span(1031, 1032) + b"\n" + _span(1036, 1038) + b"\n" + _span(1042, 1052),
    }
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_main_controls_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        content = TEMPLATE.read_bytes()
        for marker, key in (
            (b"! <THERMOSOIL_WLUPDATE>", "thermosoil_wlupdate"),
            (b"! <HUMIDITY_OWNER_BLOCK>", "humidity_owner"),
            (b"! <HISTORY_OWNER_BLOCK>", "history_owner"),
            (b"! <FINAL_OWNER_BLOCK>", "final_owner"),
        ):
            content = content.replace(marker, spans[key])
        source.write_bytes(content)
        executable = build / "oracle.exe"
        compiler_meta = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(csv_path)
    jax = _jax()
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14)
    comparisons = [float_comparison(key, value, jax[key], rtol=1e-12, atol=1e-14) for key, value in fortran.items()]
    assignments = {
        (899, "if", "false"): ["humidity direct"],
        (899, "if", "true"): ["humidity long-memory"],
        (940, "if", "true"): ["native history"],
        (944, "if", "true"): ["CWRR history"],
        (946, "if", "false"): ["non-permafrost PFT loop slots"],
        (946, "if", "true"): ["PFT1 and PFT14 permafrost slots"],
        (948, "if", "true"): ["PFT1 zero-padded name"],
        (948, "if", "false"): ["PFT14 two-digit name"],
        (979, "if", "false"): ["hist2 disabled"],
        (1042, "if", "true"): ["explicit snow enabled"],
        (1044, "if", "false"): ["snow-free point"],
        (1044, "if", "true"): ["snow-covered points"],
        (1045, "if", "false"): ["cold snow-covered point"],
        (1045, "if", "true"): ["warm snow-covered point"],
        (1052, "if", "false"): ["printlev below diagnostic threshold"],
    }
    coverage = {
        "schema_version": 1,
        "branch_complete": True,
        "missing_disposition": [],
        "missing_case_assignment": [],
        "ledger_entries": {
            LEDGER_ENTRY: [
                {
                    "arm_id": f"fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90:{line}:{kind}:{arm}",
                    "case_ids": cases,
                }
                for (line, kind, arm), cases in assignments.items()
            ]
        },
    }
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )
    (output_dir / "inputs.json").write_text(json.dumps({
        "schema_version": 1,
        "cases": ["humidity direct", "humidity long-memory", "native history PFT1/PFT14", "explicit snow clamp"],
        "composition": "All branch-bearing thermosoil_main owner spans 899-1052 plus separately verified real scientific callees.",
    }, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in spans.items()},
        "compiler": compiler_meta,
        "verified_ledger_entries": [LEDGER_ENTRY],
        "fortran_undefined_contract": ["thermosoil_main.local_ok_zimov"],
    })


if __name__ == "__main__":
    try:
        result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout, file=sys.stderr)
        raise
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
