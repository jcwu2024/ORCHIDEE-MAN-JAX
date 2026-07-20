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
from jax_orchidee.sechiba.condveg import condveg_finalize_restart_packet  # noqa:E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa:E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "condveg_finalize_state_io"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/condveg.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_condveg_finalize_state_io.f90.template"
PROCEDURES = {"condveg_finalize"}


def compose(path):
    span = extract_procedure_bytes(SOURCE, "condveg_finalize")
    path.write_bytes(
        TEMPLATE.read_bytes().replace(b"! <CONDVEG_FINALIZE>", span.span_bytes)
    )
    return {"condveg_finalize": span.span_sha256}


def _read(path):
    d = {}
    with path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            d.setdefault(row["field"], []).append(float(row["value"]))
    return {k: np.asarray(v) for k, v in d.items()}


def _jax():
    roughp = np.asarray([[10 * i + 0.1 * j for j in range(1, 15)] for i in range(1, 3)])
    packet = condveg_finalize_restart_packet(
        z0m=[0.1, 0.2],
        z0h=[0.01, 0.02],
        roughheight=[1.0, 2.0],
        roughheight_pft=roughp,
        alb_bg_modis=True,
        soilalb_bg=np.asarray([[0.11, 0.12], [0.21, 0.22]]),
    )
    return {k: np.asarray(v).ravel() for k, v in packet.items()}


def run_oracle(output_dir, compiler=DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_condveg_finalize_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        exe = build / "oracle.exe"
        hashes = compose(source)
        meta = compile_fortran(source, exe, compiler)
        out = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(exe), str(out.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    f = _read(output_dir / "fortran_outputs.csv")
    j = _jax()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", f, j, rtol=0.0, atol=0.0
    )
    comp = [float_comparison(n, f[n], j[n], rtol=0.0, atol=0.0) for n in sorted(f)]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {"schema_version": 1, "alb_bg_modis": True, "fields": sorted(f)}, indent=2
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comp,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": hashes,
            "compiler": meta,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
