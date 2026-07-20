from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    ROOT,
    compiler_environment,
)
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage  # noqa: E402
from oracle_lane_thermosoil_recurrence_coef import (  # noqa: E402
    FAMILY,
    SOURCE,
    TEMPLATE,
    _span,
    run_oracle,
)


PROCEDURES = {
    "thermosoil_coef",
}
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY


def _compose(path: Path) -> None:
    spans = {
        "thermosoil_cond_pft": _span(2007, 2127),
        "thermosoil_cond_nopft": _span(2129, 2231),
        "thermosoil_cond": _span(1883, 1972),
        "thermosoil_getdiff": _span(2566, 2863),
        "thermosoil_getdiff_old_thermix_with_snow": _span(2884, 3031),
        "thermosoil_coef": _span(1386, 1729),
    }
    generated = TEMPLATE.read_bytes()
    for name, span in spans.items():
        generated = generated.replace(f"! <{name.upper()}>".encode(), span)
    path.write_bytes(generated)


def _execute(executable: Path, build: Path, compiler: Path) -> None:
    for mode in ("explicit", "no_explicit"):
        subprocess.run(
            [str(executable), str(build / f"{mode}.csv"), mode],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )


def run_coverage(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compiler: Path = DEFAULT_COMPILER,
    gcov: Path = GCOV,
) -> dict[str, object]:
    return run_extracted_owner_coverage(
        family=FAMILY,
        source_file=SOURCE,
        procedures=PROCEDURES,
        compose=_compose,
        run_numerical_oracle=run_oracle,
        output_dir=output_dir,
        compiler=compiler,
        gcov=gcov,
        execute_harness=_execute,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run THERMOSOIL coefficient gcov evidence.")
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
