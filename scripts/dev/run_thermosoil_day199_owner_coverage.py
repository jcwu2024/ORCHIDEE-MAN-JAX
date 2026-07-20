from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    ROOT,
    compiler_environment,
)
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa: E402
from oracle_thermosoil_day199_recurrence_actual import (  # noqa: E402
    FAMILY,
    FORTRAN_SPANS,
    NSTEPS,
    SOURCE,
    _sha256,
    _write_values,
    compose_fortran_source,
)


PROCEDURES = {
    "thermosoil_wlupdate",
}
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def _verified_cached_comparison(output_dir: Path, compiler: Path) -> dict[str, object]:
    del compiler
    comparison_path = output_dir / "comparison.json"
    inputs_path = output_dir / "production_inputs.npz"
    if not comparison_path.is_file() or not inputs_path.is_file():
        raise FileNotFoundError("day199 comparison or production input cache is missing")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("status") != "passed" or comparison.get("first_mismatch") is not None:
        raise RuntimeError("cached day199 numerical Oracle is not passed")
    if comparison.get("source_file_sha256") != _sha256(SOURCE):
        raise RuntimeError("cached day199 source hash drifted")
    expected_spans = comparison.get("procedure_span_sha256", {})
    source_lines = SOURCE.read_bytes().splitlines(keepends=True)
    for name, (start, end) in FORTRAN_SPANS.items():
        span = b"".join(source_lines[start - 1 : end])
        if expected_spans.get(name) != hashlib.sha256(span).hexdigest():
            raise RuntimeError(f"cached day199 span hash drifted: {name}")
    return comparison


def _write_cached_input(npz_path: Path, comparison: dict[str, object], path: Path) -> None:
    external_order = tuple(str(name) for name in comparison["external_field_order"])
    with np.load(npz_path, allow_pickle=False) as cached, path.open("wb") as handle:
        base_names = [name for name in cached.files if not name.startswith("external.")]
        for name in base_names:
            _write_values(handle, np.asfortranarray(cached[name]))
        for step in range(NSTEPS):
            for name in external_order:
                _write_values(handle, np.asfortranarray(cached[f"external.{name}"][step]))


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, object]:
    comparison = _verified_cached_comparison(output_dir, compiler)
    inputs_path = output_dir / "production_inputs.npz"

    def execute(executable: Path, build: Path, compiler_path: Path) -> None:
        input_path = build / "production_inputs.bin"
        _write_cached_input(inputs_path, comparison, input_path)
        subprocess.run(
            [
                str(executable),
                str(input_path),
                str(build / "fortran_outputs.bin"),
                str(build / "fortran_precoef_outputs.bin"),
            ],
            cwd=build,
            env=compiler_environment(compiler_path),
            check=True,
            capture_output=True,
            text=True,
        )

    return run_extracted_owner_coverage(
        family=FAMILY,
        source_file=SOURCE,
        procedures=PROCEDURES,
        compose=compose_fortran_source,
        run_numerical_oracle=_verified_cached_comparison,
        output_dir=output_dir,
        compiler=compiler,
        gcov=gcov,
        execute_harness=execute,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run day199 THERMOSOIL gcov owner evidence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--gcov", type=Path, default=GCOV)
    args = parser.parse_args(argv)
    try:
        result = run_coverage(
            output_dir=args.output_dir.resolve(),
            compiler=args.compiler.resolve(),
            gcov=args.gcov.resolve(),
        )
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "covered_arms": result["branch_coverage"]["covered_arm_count"],
                "owner_regions": len(result["owner_evidence"]["records"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
