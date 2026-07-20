from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_PATH = ROOT / "scripts" / "dev" / "extract_fortran_micro_oracle.py"
TEMPLATE_PATH = ROOT / "scripts" / "dev" / "qsat_micro_oracle_harness.f90.template"
DEFAULT_COMPILER = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
PROCEDURE_ORDER = ("qsfrict_init", "qsatcalc", "dev_qsatcalc")
MARKER = b"! <ORIGINAL_PROCEDURES>"
COMPILE_FLAGS = (
    "-std=f2008",
    # GNU equivalent of the paper executable's Intel ``-i4 -r8`` kind policy.
    "-fdefault-real-8",
    "-O0",
    "-fcheck=all",
    "-ffpe-trap=invalid,zero,overflow",
    "-Wall",
    "-Wextra",
)


def _load_extractor():
    spec = importlib.util.spec_from_file_location("extract_fortran_micro_oracle", EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extractor: {EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compose_harness(build_dir: Path) -> tuple[Path, dict[str, str]]:
    extractor = _load_extractor()
    manifest = extractor.load_manifest()
    extracted = extractor.validate_and_extract(manifest, root=ROOT)
    by_name = {entry["procedure"]: span for entry, span in extracted}
    missing = set(PROCEDURE_ORDER) - set(by_name)
    if missing:
        raise RuntimeError(f"manifest is missing procedures: {sorted(missing)}")

    template = TEMPLATE_PATH.read_bytes()
    if template.count(MARKER) != 1:
        raise RuntimeError("harness template must contain exactly one procedure marker")
    procedure_bytes = b"\n\n".join(by_name[name].span_bytes.rstrip(b"\r\n") for name in PROCEDURE_ORDER)
    compile_unit = template.replace(MARKER, procedure_bytes)
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "qsat_micro_oracle_harness.f90"
    source_path.write_bytes(compile_unit)
    hashes = {
        "compile_unit_sha256": hashlib.sha256(compile_unit).hexdigest(),
        "source_sha256": next(iter(by_name.values())).source_sha256,
        **{f"{name}_span_sha256": by_name[name].span_sha256 for name in PROCEDURE_ORDER},
    }
    return source_path, hashes


def build_oracle(build_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    if not compiler.is_file():
        raise FileNotFoundError(f"GNU Fortran compiler not found: {compiler}")
    source_path, hashes = compose_harness(build_dir)
    executable = build_dir / "qsat_micro_oracle.exe"
    command = [str(compiler), *COMPILE_FLAGS, str(source_path), "-o", str(executable)]
    compiler_env = os.environ.copy()
    compiler_env["PATH"] = str(compiler.parent) + os.pathsep + compiler_env.get("PATH", "")
    completed = subprocess.run(
        command, cwd=build_dir, env=compiler_env, text=True, capture_output=True, check=True
    )
    version = subprocess.run(
        [str(compiler), "--version"],
        env=compiler_env,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    return {
        "compiler": str(compiler),
        "compiler_version": version,
        "compile_command": command,
        "compile_stdout": completed.stdout,
        "compile_stderr": completed.stderr,
        "executable": str(executable),
        "source": str(source_path),
        "hashes": hashes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the source-extracted qsat GNU Fortran oracle.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args(argv)
    try:
        metadata = build_oracle(args.build_dir.resolve(), args.compiler.resolve())
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
