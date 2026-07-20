from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)
from scripts.dev.oracle_lane_sechiba_setvar_common import (  # noqa: E402
    jax_outputs,
    read_csv,
)


FAMILY = "sechiba_setvar_serial"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba_io.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_sechiba_setvar_serial.f90.template"
PROCEDURES = {
    "i0setvar", "i10setvar", "i11setvar", "i20setvar", "i21setvar", "i22setvar",
    "r0setvar", "r10setvar", "r11setvar", "r20setvar", "r21setvar", "r22setvar", "r30setvar",
}


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    generated = TEMPLATE.read_bytes()
    for name, span in spans.items():
        generated = generated.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(generated)
    return {name: span.span_sha256 for name, span in spans.items()}


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_setvar_serial_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        compiler_meta = compile_fortran(source, executable, compiler)
        output_path = output_dir / "fortran_outputs.csv"
        subprocess.run([str(executable), str(output_path.resolve())], cwd=build,
                       env=compiler_environment(compiler), check=True, capture_output=True, text=True)
    fortran = read_csv(output_dir / "fortran_outputs.csv")
    jax = jax_outputs(parallel=False)
    if set(fortran) != set(jax):
        raise RuntimeError(f"setvar fields differ: Fortran={sorted(fortran)}, JAX={sorted(jax)}")
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=0.0, atol=0.0)
    comparisons = [float_comparison(name, fortran[name], jax[name], rtol=0.0, atol=0.0)
                   for name in sorted(fortran)]
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1,
        "cases": ["NO_KEYWORD assignment", "configuration assignment", "partially initialized preservation",
                  "rank-1 spread over first and second dimensions"]}, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": span_hashes, "compiler": compiler_meta,
        "verified_ledger_entries": [],
    })


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
