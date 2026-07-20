from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_soilcarbon_owner_harness.f90.template"
COMPILER = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
SPANS = (("soilcarbon_leak", 641, 2416), ("altcalc_DOC", 2438, 2578), ("cryoturbate_DOC_POC", 2599, 3369))


def span_bytes(start: int, end: int) -> bytes:
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    return b"".join(lines[start - 1 : end])


def build_source() -> tuple[bytes, dict[str, str]]:
    chunks: list[bytes] = []
    hashes: dict[str, str] = {}
    for name, start, end in SPANS:
        chunk = span_bytes(start, end)
        if f"subroutine {name}".lower().encode() not in chunk.lower():
            raise ValueError(f"bad source span for {name}")
        hashes[name] = hashlib.sha256(chunk).hexdigest()
        chunks.append(chunk)
    return TEMPLATE.read_bytes().replace(b"!__ORIGINAL_PROCEDURES__", b"\n".join(chunks)), hashes


def compile_probe() -> dict[str, object]:
    source, hashes = build_source()
    with tempfile.TemporaryDirectory(prefix="orcjax_soilcarbon_owner_") as tmp:
        src = Path(tmp) / "soilcarbon_owner.f90"
        obj = Path(tmp) / "soilcarbon_owner.exe"
        src.write_bytes(source)
        result = subprocess.run(
            [str(COMPILER), "-std=legacy", "-cpp", "-fdefault-real-8", "-ffree-line-length-none", "-O0", "-fcheck=all", str(src), "-o", str(obj)],
            cwd=tmp,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            sys.stderr.write(result.stderr)
            raise RuntimeError(f"gfortran exited {result.returncode}")
        raw = Path(tmp) / "outputs.bin"
        run = subprocess.run([str(obj), str(raw)], cwd=tmp, text=True, capture_output=True)
        if run.returncode:
            sys.stderr.write(run.stdout + run.stderr)
            raise RuntimeError(f"oracle executable exited {run.returncode}")
        size = raw.stat().st_size
    return {"status": "ran", "output_bytes": size, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "span_sha256": hashes}


if __name__ == "__main__":
    print(compile_probe())
