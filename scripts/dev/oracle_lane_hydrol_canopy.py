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

FAMILY = "hydrol_canopy_infiltration_freeze"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_canopy.f90.template"


def _procedure_bytes(start: int = 4992, end: int = 5117) -> bytes:
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    return b"".join(lines[start - 1 : end])


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(data) for name, data in values.items()}


def _jax_outputs() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.hydrol import hydrol_canop_interception

    n, nvm = 3, 14
    wet = np.zeros((n, nvm))
    wet[:, 13] = [0.0, 0.1, 0.4]
    vmax = np.zeros((n, nvm))
    vmax[:, 0] = 0.2
    vmax[:, 13] = [0.0, 0.7, 0.8]
    veg = vmax.copy()
    veg[1, 13] = 0.5
    qmax = np.zeros((n, nvm))
    qmax[:, 13] = [0.2, 0.5, 0.6]
    qs = np.zeros((n, nvm))
    qs[:, 13] = [0.0, 0.3, 0.9]
    throughfall = np.full(nvm, 0.1)
    throughfall[0] = 1.0
    throughfall[13] = 0.25
    result = hydrol_canop_interception(
        precip_rain=np.array([0.0, 2.0, 5.0]),
        vevapwet=wet,
        veget_max=vmax,
        veget=veg,
        qsintmax=qmax,
        qsintveg=qs,
        tot_melt=np.array([1.0, 0.5, 0.0]),
        vegtot=np.array([0.0, 0.7, 1.0]),
        throughfall_by_pft=throughfall,
    )
    outputs = {
        name: np.asarray(getattr(result, name)).ravel(order="C")
        for name in ("qsintveg", "precisol", "precip2canopy", "precip2ground")
    }
    outputs["canopy2ground"] = np.asarray(result.canopy2ground)[:, 1:].ravel(order="C")
    return outputs


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    span = _procedure_bytes()
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_hydrol_canopy_oracle_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(TEMPLATE.read_bytes().replace(b"! <HYDROL_CANOP>", span))
        executable = build / "oracle.exe"
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read_csv(csv_path)
    jax = _jax_outputs()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "zero vegetation/no rain",
                    "unsaturated PFT14 canopy plus melt",
                    "overflowing PFT14 canopy",
                ],
                "dimensions": {"npts": 3, "nvm": 14},
                "undefined_fortran_outputs": ["canopy2ground(:,1)"],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14)
        for name in fortran
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": hashlib.sha256(span).hexdigest(),
            "compiler": metadata,
            "verified_ledger_entries": ["hydrol.conditional.canopy_interception"]
            if all(item["passed"] for item in comparisons)
            else [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
