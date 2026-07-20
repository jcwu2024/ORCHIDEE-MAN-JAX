"""Compile witness for the original ``sechiba_initialize`` restart-I/O arm."""

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

from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compiler_environment  # noqa: E402


SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/sechiba_batch_c"
ARMS = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:774:if:false",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:774:if:true",
)


def _block() -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[771:777])


def run(output: Path = OUTPUT, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    block = _block()
    unit = b"""program irrigation_owner
  implicit none
  logical :: do_fullirr
  integer :: rest_id, nbp_glo, kjit, ih2o, val_exp
  integer :: index_g(1)
  real(8), parameter :: zero=0.d0
  real(8) :: irrigation(1,1)
  do_fullirr=.true.; rest_id=1; nbp_glo=1; kjit=1; ih2o=1; val_exp=-9999
  irrigation=0.d0; index_g=1
! <ORIGINAL_BLOCK>
end program irrigation_owner
""".replace(b"! <ORIGINAL_BLOCK>", block.rstrip())
    with tempfile.TemporaryDirectory(prefix="orchidee_batch_c_irrigation_") as temporary:
        build = Path(temporary)
        source = build / "irrigation_owner.f90"
        executable = build / "irrigation_owner.exe"
        source.write_bytes(unit)
        command = [str(compiler), "-O0", "-fdefault-real-8", "-ffree-line-length-none", str(source), "-o", str(executable)]
        completed = subprocess.run(command, cwd=build, env=compiler_environment(compiler), capture_output=True, text=True)
    result = {
        "schema_version": 1,
        "arm_ids": list(ARMS),
        "original_block_lines": "772-777",
        "original_block_sha256": hashlib.sha256(block).hexdigest(),
        "compile_command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "status": "blocked" if completed.returncode else "unexpectedly_linked",
        "blocker": (
            "The original true arm calls restget_p to materialize restart variable irrigation; "
            "linking the byte-exact block fails without the IOIPSL restart runtime. Supplying a "
            "stub would invent restart values and cannot be used for a Fortran-vs-JAX state comparison."
        ),
    }
    (output / "sechiba_initialize_irrigation_compile_blocker.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "blocked" else 1)
