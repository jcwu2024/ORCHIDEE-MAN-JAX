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

from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    QZ_USDA,
    SMCMAX_USDA,
    thermosoil_cond,
    thermosoil_cond_nopft,
    thermosoil_cond_pft,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)


FAMILY = "thermosoil_cond_matrix"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_thermosoil_cond_matrix.f90.template"
PROCEDURES = {"thermosoil_cond", "thermosoil_cond_pft", "thermosoil_cond_nopft"}


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    generated = TEMPLATE.read_bytes()
    for name, span in spans.items():
        generated = generated.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(generated)
    return {name: span.span_sha256 for name, span in spans.items()}


def _inputs() -> dict[str, np.ndarray]:
    smc = np.empty((4, 4), dtype=np.float64)
    smc[:, 0] = 0.30
    smc[:, 1] = 0.0
    smc[:, 2] = 0.03
    smc[:, 3] = 0.01
    sh2o = smc.copy()
    sh2o[0, 0] = 0.10
    return {
        "njsc": np.asarray([1, 5, 4, 10], dtype=np.int32),
        "smc": smc,
        "sh2o": sh2o,
        "zx1": np.full((4, 4, 14), 0.20),
        "zx2": np.full((4, 4, 14), 0.80),
        "porosnet": np.full((4, 4, 14), 0.40),
    }


def _jax_outputs(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    common = {
        "njsc": inputs["njsc"],
        "smc": inputs["smc"],
        "sh2o": inputs["sh2o"],
        "qz": np.asarray(QZ_USDA),
        "smcmax": np.asarray(SMCMAX_USDA),
    }
    plain = thermosoil_cond(**common)
    pft = thermosoil_cond_pft(
        **common,
        zx1=inputs["zx1"],
        zx2=inputs["zx2"],
        porosnet=inputs["porosnet"],
    )
    nopft = thermosoil_cond_nopft(
        **common,
        zx1=inputs["zx1"][:, :, 0],
        zx2=inputs["zx2"][:, :, 0],
        porosnet=inputs["porosnet"][:, :, 0],
    )
    return {
        "cond": np.asarray(plain.cnd).ravel(order="C"),
        "cond_pft": np.asarray(pft.cnd).ravel(order="C"),
        "cond_nopft": np.asarray(nopft.cnd).ravel(order="C"),
    }


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_cond_matrix_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        compiler_meta = compile_fortran(source, executable, compiler)
        output_path = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(output_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    inputs = _inputs()
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs(inputs)
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "njsc": inputs["njsc"].tolist(),
                "smc": inputs["smc"].tolist(),
                "sh2o": inputs["sh2o"].tolist(),
                "organic_fraction": 0.20,
                "porosnet": 0.40,
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
            "compiler": compiler_meta,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
