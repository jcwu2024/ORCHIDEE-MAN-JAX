"""Compile only HYDROL's two original F10.3 READ statements, never the model.

Produces a bounded, inspectable input-decoding reference. It certifies the
text-to-real boundary only, not hydrology, chronology or scientific accuracy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = root / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
    source_lines = source_path.read_text(encoding="latin1").splitlines()
    reads = [line.strip() for line in source_lines if line.strip().startswith(('READ(103,', 'READ(104,'))]
    if reads != ['READ(103,"(F10.3)") wtp_tide', 'READ(104,"(F10.3)") dwtp_tide']:
        raise RuntimeError("original HYDROL READ boundaries changed; review source before regenerating")
    pairs = []
    # Only eight actual records from each asset are read for the oracle.
    with (root / "data/forcing_VN/WTM_forcing_pos.txt").open() as pos, (root / "data/forcing_VN/WTM_forcing_diff.txt").open() as diff:
        for _ in range(8):
            pairs.append([next(pos).rstrip("\r\n"), next(diff).rstrip("\r\n")])
    pairs += [["     12345", "    -12345"], ["  1.234567", " -1.234567"],
              ["123.456789999", "-123.456789999"], ["1", "-1"],
              [" 1.234E+02", "     1D+02"], [" 12345-02", "  1 2 3 4"],
              ["", "          "], ["-0.0000000", "0.00000000"],
              ["1,ignored", ".12345678999"]]
    (output / "positive.txt").write_text("\n".join(p[0] for p in pairs) + "\n", encoding="ascii")
    (output / "differential.txt").write_text("\n".join(p[1] for p in pairs) + "\n", encoding="ascii")
    program = "\n".join([
        "program hydrol_tide_read_boundary", "  use iso_fortran_env, only: real64, int64", "  implicit none",
        "  real(real64) :: wtp_tide, dwtp_tide", "  integer :: i",
        '  OPEN(103,FILE="positive.txt",ACTION="READ")',
        '  OPEN(104,FILE="differential.txt",ACTION="READ")',
        f"  do i=1,{len(pairs)}", "    " + reads[0], "    " + reads[1],
        "    write(*,'(Z16.16,1X,Z16.16)') transfer(wtp_tide,0_int64), transfer(dwtp_tide,0_int64)",
        "  end do", "end program hydrol_tide_read_boundary", ""])
    program_path = output / "hydrol_tide_read_boundary.f90"
    program_path.write_text(program, encoding="ascii")
    compiler = shutil.which("gfortran")
    if compiler is None:
        raise RuntimeError("gfortran unavailable; no reference was generated")
    executable = output / "hydrol_tide_read_boundary.exe"
    compiled = subprocess.run([compiler, "-std=f2008", "-O0", str(program_path), "-o", str(executable)],
                              cwd=output, check=True, text=True, capture_output=True)
    native = subprocess.run([str(executable)], cwd=output, check=True, text=True, capture_output=True)
    (output / "compiler.txt").write_text(compiled.stdout + compiled.stderr, encoding="utf8")
    (output / "native-output.txt").write_text(native.stdout, encoding="ascii")
    rows = []
    for pair, result in zip(pairs, native.stdout.splitlines(), strict=True):
        bits = result.split()
        values = [np.asarray(int(b, 16), dtype=np.uint64).view(np.float64).item() for b in bits]
        rows.append(dict(records=pair, float64_bits=bits, values=values))
    result = dict(schema="hydrol-tide-f10.3-native-boundary-v1", scope="text input decoding only; no model execution",
                  source="fortran_source/ORCHIDEE/src_sechiba/hydrol.f90:5648,5654", compiler=compiler,
                  original_read_statements=reads, rows=rows)
    text = json.dumps(result, indent=2) + "\n"
    (output / "reference.json").write_text(text, encoding="utf8")
    if args.fixture_output:
        args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
        args.fixture_output.write_text(text, encoding="utf8")
    print(json.dumps(dict(output=str(output), pairs=len(rows), native_boundary_only=True)))


if __name__ == "__main__":
    main()
