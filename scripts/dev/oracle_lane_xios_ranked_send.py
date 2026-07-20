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

from jax_orchidee import parallel  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)


FAMILY = "xios_ranked_send"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parallel/xios_orchidee.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_xios_ranked_send.f90.template"
PROCEDURES = {f"xios_orchidee_send_field_r{rank}d" for rank in range(1, 6)}


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    generated = TEMPLATE.read_bytes()
    for name, span in spans.items():
        generated = generated.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(generated)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(value) for name, value in fields.items()}


def _jax() -> dict[str, np.ndarray]:
    outputs = {}
    for rank in range(1, 6):
        shape = (2,) * rank
        field = np.arange(1, np.prod(shape) + 1, dtype=np.float64).reshape(
            shape, order="F"
        )
        sent = []
        status = getattr(parallel, f"xios_orchidee_send_field_r{rank}d")(
            f"rank{rank}",
            field,
            nbp_mpi=2,
            xios_orchidee_ok=True,
            is_omp_root=True,
            gather_omp=lambda value: value.copy(),
            transport=lambda field_id, value: sent.append(value.copy()),
        )
        if not status.transported or status.gathered_shape != shape:
            raise RuntimeError(f"rank {rank} transport failed")
        outputs[f"rank{rank}"] = sent[0].ravel(order="C")
    return outputs


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_xios_send_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        meta = compile_fortran(source, executable, compiler)
        output = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(output.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=0.0, atol=0.0
    )
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=0.0, atol=0.0)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {"schema_version": 1, "ranks": [1, 2, 3, 4, 5], "nbp_mpi": 2}, indent=2
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
