from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fortran_oracle_common import DEFAULT_COMPILER, ROOT
from fortran_owner_coverage import GCOV, run_extracted_owner_coverage
from fortran_oracle_common import compiler_environment
from oracle_stage4_batch_c_condveg import FAMILY, PROCEDURES, SOURCE, _compose, run_oracle

DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY

FATAL_SOIL_CLASS = "fortran_source/ORCHIDEE/src_sechiba/condveg.f90:1091:if:false"


def _execute_cases(executable: Path, build: Path, compiler: Path) -> None:
    environment = compiler_environment(compiler)
    subprocess.run([str(executable), str(build / "fortran_outputs.csv")], cwd=build,
                   env=environment, check=True, capture_output=True, text=True)
    fatal = subprocess.run([str(executable), str(build / "fatal.csv"), "fatal"], cwd=build,
                           env=environment, capture_output=True, text=True)
    if fatal.returncode == 0 or "fatal owner boundary" not in (fatal.stdout + fatal.stderr):
        raise RuntimeError("invalid soil-class case did not reach ipslerr_p fatal boundary")


def run_coverage(output_dir: Path = DEFAULT_OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER, gcov: Path = GCOV):
    return run_extracted_owner_coverage(
        family=FAMILY, source_file=SOURCE, procedures=set(PROCEDURES), compose=_compose,
        run_numerical_oracle=run_oracle, output_dir=output_dir, compiler=compiler, gcov=gcov,
        condition_terminal_lines={227: 229},
        arm_execution_lines={
            # The scalar ALL loops are emitted over physical lines 227-229, so GCOV
            # has no single source-level edge. These are unique process-boundary
            # witnesses from deterministic restart-missing and restart-present calls.
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90:227:if:true": 232,
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90:227:if:false": 306,
        },
        arm_branch_indices={
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90:1277:if:true": 0,
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90:1093:if:true": 36,
            "fortran_source/ORCHIDEE/src_sechiba/condveg.f90:1112:if:false": 0,
        },
        fatal_boundary_witness_lines={FATAL_SOIL_CLASS: 1105},
        execute_harness=_execute_cases,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = run_coverage(args.output_dir.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    print(json.dumps({"status": "passed", "covered_arms": result["branch_coverage"]["covered_arm_count"]}, indent=2))
