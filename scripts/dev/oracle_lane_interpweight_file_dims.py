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
from jax_orchidee.driver.interpolation_file_dims import (  # noqa: E402
    InterpolationFileMetadataError,
    interpweight_get_var2dims_file,
    interpweight_get_var3dims_file,
    interpweight_get_var4dims_file,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "interpweight_file_dims"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/interpweight.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_interpweight_file_dims.f90.template"
PROCEDURES = {
    "interpweight_get_var2dims_file",
    "interpweight_get_var3dims_file",
    "interpweight_get_var4dims_file",
}


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    source = TEMPLATE.read_bytes()
    for name, span in spans.items():
        marker = f"! <{name.upper()}>".encode("ascii")
        source = source.replace(marker, span.span_bytes)
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[int]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(int(row["value"]))
    return {name: np.asarray(values, dtype=np.int64) for name, values in fields.items()}


def _metadata(rank: int) -> dict[str, object]:
    fortran_sizes = (7, 5, 3, 2)[:rank]
    names = tuple(f"dim{index}" for index in range(rank))
    file_sizes = tuple(reversed(fortran_sizes))
    return {
        "dimensions": dict(zip(names, file_sizes, strict=True)),
        "variables": {"field": {"dimensions": names}},
    }


def _error_flag(function, metadata: dict[str, object], varname: str) -> np.ndarray:
    try:
        function(metadata, varname)
    except InterpolationFileMetadataError:
        return np.asarray([1], dtype=np.int64)
    return np.asarray([0], dtype=np.int64)


def _jax_outputs() -> dict[str, np.ndarray]:
    functions = {
        2: interpweight_get_var2dims_file,
        3: interpweight_get_var3dims_file,
        4: interpweight_get_var4dims_file,
    }
    outputs: dict[str, np.ndarray] = {}
    for rank, function in functions.items():
        outputs[f"success{rank}"] = np.asarray(
            function(_metadata(rank), "field"), dtype=np.int64
        )
        outputs[f"mismatch{rank}_error"] = _error_flag(
            function, _metadata(rank + 1 if rank < 4 else 3), "field"
        )
        outputs[f"empty{rank}_error"] = _error_flag(function, _metadata(rank), " ")
    return outputs


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_interpweight_file_dims_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        compiler_metadata = compile_fortran(source, executable, compiler)
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
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=0.0, atol=0.0
    )
    comparisons = [
        exact_comparison(name, values, jax[name]) for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": ["success", "rank_mismatch", "empty_variable_name"],
                "fortran_dimension_sizes": [7, 5, 3, 2],
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
            "compiler": compiler_metadata,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
