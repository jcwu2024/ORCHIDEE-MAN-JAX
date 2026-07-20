from __future__ import annotations

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

from jax_orchidee.sechiba import io_contracts as io  # noqa: E402
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
    IX,
    RX,
    MissingReader,
    OracleReader,
    jax_outputs,
    read_csv,
)


FAMILY = "sechiba_setvar_parallel"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba_io_p.f90"
SERIAL_TEMPLATE = ROOT / "scripts/dev/oracle_lane_sechiba_setvar_serial.f90.template"
BASE_NAMES = (
    "i0setvar", "i10setvar", "i11setvar", "i20setvar", "i21setvar", "i22setvar",
    "r0setvar", "r10setvar", "r11setvar", "r20setvar", "r21setvar", "r22setvar", "r30setvar",
)
PROCEDURES = {name + "_p" for name in BASE_NAMES}


def _parallel_template() -> bytes:
    generated = SERIAL_TEMPLATE.read_text(encoding="ascii")
    generated = generated.replace("sechiba_setvar_serial", "sechiba_setvar_parallel")
    generated = generated.replace("long_print_setvar", "long_print_setvar_p")
    generated = generated.replace(
        "  integer :: numout=6\n",
        "  integer :: numout=6, nbp_glo=3\n  logical :: is_root_prc=.true.\n",
    )
    generated = generated.replace(
        "  interface getin\n",
        "  interface getin\n",
    ).replace(
        "  end interface\ncontains\n",
        "  end interface\n"
        "  interface getin_p\n"
        "    module procedure getin_i0,getin_i1,getin_i2,getin_r0,getin_r1,getin_r2\n"
        "  end interface\n"
        "  interface gather\n    module procedure gather_i1\n  end interface\n"
        "  interface scatter\n    module procedure scatter_i1\n  end interface\n"
        "contains\n"
        "  subroutine gather_i1(local_value,global_value)\n"
        "    integer(i_std),intent(in)::local_value(:)\n"
        "    integer(i_std),intent(out)::global_value(:)\n"
        "    global_value=local_value\n  end subroutine\n"
        "  subroutine scatter_i1(local_value,global_value)\n"
        "    integer(i_std),intent(out)::local_value(:)\n"
        "    integer(i_std),intent(in)::global_value(:)\n"
        "    local_value=global_value\n  end subroutine\n",
        1,
    )
    for name in BASE_NAMES:
        generated = generated.replace(f"call {name}(", f"call {name}_p(")
    extra = (
        "  iv3=ix; call i11setvar_p(iv3,ix,'CFG_I1',ip3,.true.); "
        "call wi1(u,'i11.grid.cfg',iv3)\n"
        "  rm23=rx; call r22setvar_p(rm23,rx,'MISSING_R0',"
        "reshape((/1.,2.,3.,4.,5.,6./),(/2,3/))); call wr2(u,'r22.missing',rm23)\n"
    )
    generated = generated.replace("  close(u)\ncontains\n", extra + "  close(u)\ncontains\n")
    return generated.encode("ascii")


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    generated = _parallel_template()
    for base in BASE_NAMES:
        span = spans[base + "_p"]
        generated = generated.replace(f"! <{base.upper()}>".encode(), span.span_bytes)
    path.write_bytes(generated)
    return {name: span.span_sha256 for name, span in spans.items()}


def _jax_outputs() -> dict[str, np.ndarray]:
    outputs = jax_outputs(parallel=True)
    reader = OracleReader()
    grid = io.SerialGridParallelIO(reader, np.arange(1, 4, dtype=np.int32))
    outputs["i11.grid.cfg"] = np.asarray(
        io.i11setvar_p(
            np.full(3, IX, dtype=np.int32), IX, "CFG_I1",
            np.asarray([7, 8, 9], dtype=np.int32), config=reader,
            is_grid=True, grid_io=grid,
        )
    ).ravel(order="C")
    outputs["r22.missing"] = np.asarray(
        io.r22setvar_p(
            np.full((2, 3), RX), RX, "MISSING_R0",
            np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
            config=MissingReader(),
        )
    ).ravel(order="C")
    return outputs


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_setvar_parallel_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        compiler_meta = compile_fortran(source, executable, compiler)
        output_path = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(output_path.resolve())], cwd=build,
            env=compiler_environment(compiler), check=True, capture_output=True, text=True,
        )
    fortran = read_csv(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs()
    if set(fortran) != set(jax):
        raise RuntimeError(f"parallel setvar fields differ: Fortran={sorted(fortran)}, JAX={sorted(jax)}")
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=0.0, atol=0.0)
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=0.0, atol=0.0)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, "cases": [
            "serial-equivalent typed setvar matrix", "i11 present is_grid gather/root-read/scatter",
            "r22 present and absent scalar configuration", "rank mismatch STOP exits",
        ]}, indent=2) + "\n", encoding="ascii",
    )
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": span_hashes, "compiler": compiler_meta,
        "verified_ledger_entries": [],
    })


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
