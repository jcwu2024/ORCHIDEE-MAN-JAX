"""Executable original-byte witness for ``sechiba_init`` lines 1983-1987."""

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
from scripts.dev.fortran_gcov import parse_gcov  # noqa: E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compiler_environment  # noqa: E402
from jax_orchidee.sechiba.initialize import SechibaInitDimensions, sechiba_init_state  # noqa: E402


OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/sechiba_batch_c"
SECHIBA = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
IOIPSL = ROOT / "fortran_source/ORCHIDEE/src_parallel/ioipsl_para.f90"
ARM_TRUE = "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:1983:if:true"


def _lines(path: Path, start: int, end: int) -> bytes:
    return b"".join(path.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _dimensions() -> SechibaInitDimensions:
    return SechibaInitDimensions(
        npts=1, nvm=14, nslm=3, nstm=6, nnobio=1, itimetide=4, nflow=3,
        nexp=2, nctext=3, ncarb=3, nparts=2, nelements=2, nlitt=2,
        ndeep=2, ndoc=2, npool=2, nleafages=4, nlai=3, ngrnd=4, nsnow=3,
        npco2=3,
    )


def _unit() -> bytes:
    branch = _lines(SECHIBA, 1983, 1987)
    ipslerr = extract_procedure_bytes(IOIPSL, "ipslerr_p").span_bytes
    return b"""module oracle_state
  implicit none
  integer :: orch_ilv_cur=0, orch_ilv_max=0, orch_ipslout=6
end module oracle_state
program oracle
  use oracle_state
  implicit none
  logical :: l_first_sechiba
  character(len=8) :: mode
  call get_command_argument(1, mode)
  l_first_sechiba = trim(mode) == 'first'
! <BRANCH>
  write(*,'(L1)') l_first_sechiba
contains
! <IPSLERR>
end program oracle
""".replace(b"! <BRANCH>", branch.rstrip()).replace(
        b"! <IPSLERR>", ipslerr.rstrip().replace(
            b"   IMPLICIT NONE", b"   USE oracle_state\n   IMPLICIT NONE", 1
        )
    )


def run(output: Path = OUTPUT, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_batch_c_init_first_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        source.write_bytes(_unit())
        command = [str(compiler), "-cpp", "--coverage", "-O0", "-fdefault-real-8", "-ffree-line-length-none", str(source), "-o", str(executable)]
        compiled = subprocess.run(command, cwd=build, env=compiler_environment(compiler), capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        first = subprocess.run([str(executable), "first"], cwd=build, env=compiler_environment(compiler), capture_output=True, text=True)
        repeated = subprocess.run([str(executable), "repeat"], cwd=build, env=compiler_environment(compiler), capture_output=True, text=True)
        gcov = subprocess.run([str(compiler.with_name("gcov.exe")), "-b", "-c", "oracle.gcno"], cwd=build, env=compiler_environment(compiler), capture_output=True, text=True)
        if gcov.returncode:
            raise RuntimeError(gcov.stderr)
        gcov_path = build / "oracle.f90.gcov"
        generated = gcov_path.read_text(encoding="utf-8")
        generated_source = source.read_bytes()
        branch_marker = b"IF (l_first_sechiba) THEN"
        branch_offset = generated_source.index(branch_marker)
        branch_line = generated_source[:branch_offset].count(b"\n") + 1
        branch_rows = parse_gcov(gcov_path).get(branch_line, [])

    jax = sechiba_init_state(l_first=True, dimensions=_dimensions())
    evidence = {
        "schema_version": 1,
        "arm_id": ARM_TRUE,
        "source_branch_sha256": hashlib.sha256(_lines(SECHIBA, 1983, 1987)).hexdigest(),
        "original_ipslerr_sha256": extract_procedure_bytes(IOIPSL, "ipslerr_p").span_sha256,
        "fortran_first_exit_code": first.returncode,
        "fortran_first_l_first": first.stdout.strip(),
        "jax_first_l_first": str(jax.l_first).lower(),
        "first_writeback_matches": first.returncode == 0 and first.stdout.strip() == ("T" if jax.l_first else "F"),
        "fortran_repeated_exit_code": repeated.returncode,
        "fortran_repeated_fatal": repeated.returncode == 1 and "Fatal error from ORCHIDEE" in repeated.stdout,
        "jax_repeated_fatal": False,
        "gcov_mapping": {
            "original_line": 1983,
            "generated_line": branch_line,
            "branches": branch_rows,
            "both_directions_taken": len(branch_rows) >= 2 and all(int(row["taken"]) > 0 for row in branch_rows[:2]),
        },
        "compiler_command": command,
    }
    # JAX raises before producing a value on the repeated-call path.
    try:
        sechiba_init_state(l_first=False, dimensions=_dimensions())
    except Exception as error:  # exact source-routed fatal boundary
        evidence["jax_repeated_fatal"] = type(error).__name__ == "SechibaInitializationError"
    evidence["passed"] = bool(evidence["first_writeback_matches"] and evidence["fortran_repeated_fatal"] and evidence["jax_repeated_fatal"] and evidence["gcov_mapping"]["both_directions_taken"])
    (output / "sechiba_init_first_call_witness.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    return evidence


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
