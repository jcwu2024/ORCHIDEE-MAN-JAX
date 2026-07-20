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
from jax_orchidee.stomate.carbon_kernels import crown_step  # noqa:E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa:E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "lpj_crown_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/lpj_crown.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_lpj_crown_owner.f90.template"
PROCEDURES = {"crown"}


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, "crown")
    path.write_bytes(TEMPLATE.read_bytes().replace(b"! <CROWN>", span.span_bytes))
    return {"crown": span.span_sha256}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _jax_outputs() -> dict[str, np.ndarray]:
    present = np.zeros((2, 14), dtype=bool)
    present[0, 1] = present[1, 1] = True
    present[0, 13] = True
    wood = np.zeros((2, 14))
    wood[0, 1] = 20000.0
    veget = np.full((2, 14), 0.2)
    height = np.full((2, 14), 3.0)
    is_tree = np.zeros(14, dtype=bool)
    is_tree[1] = True
    result = crown_step(
        pft_present=present,
        ind=np.ones((2, 14)),
        biomass=np.zeros((2, 14, 12, 1)),
        woodmass_ind=wood,
        veget_max=veget,
        height=height,
        is_tree=is_tree,
        natural=np.ones(14, dtype=bool),
        maxdia=np.full(14, 0.5),
        pipe_tune1=100.0,
        pipe_tune2=40.0,
        pipe_tune3=0.5,
        pipe_density=2.0e5,
        pipe_tune_exp_coeff=1.6,
        min_stomate=0.0,
    )
    return {
        name: np.asarray(getattr(result, name)).ravel(order="C")
        for name in ("cn_ind", "height", "veget_max")
    }


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_lpj_crown_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        exe = build / "oracle.exe"
        hashes = compose(source)
        meta = compile_fortran(source, exe, compiler)
        subprocess.run(
            [str(exe), str((output_dir / "fortran_outputs.csv").resolve()), "normal"],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, values, jax[name], rtol=1e-12, atol=1e-14)
        for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": ["natural tree active/inactive", "PFT14 grass present/absent"],
                "coverage_only_case": "static vegetation coherence fatal",
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
            "compiler": meta,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
