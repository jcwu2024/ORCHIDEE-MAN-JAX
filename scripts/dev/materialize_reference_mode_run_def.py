from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.run_def_materialization import (  # noqa: E402
    JOB_PROTOCOL_OVERRIDES,
    materialize_run_def_values,
    read_run_def_values,
    static_run_def_values,
    write_materialized_run_def,
)


DEFAULT_TRACE_USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
DEFAULT_REFERENCE_OVERRIDE_RUN_DEF = (
    ROOT
    / "fortran_run_scripts"
    / "paper_250919"
    / "sen_reference_arg2_1.0_001.0-071.0"
    / "I10"
    / "S2_63.206_0.0876_0.2019_50.658"
    / "run.def_63.206_0.0876_0.2019_50.658"
)
DEFAULT_REFERENCE_FINAL_RUN_DEF = (
    ROOT
    / "reference"
    / "case_001_071"
    / "OUT"
    / "orc_calibrate_250919_sen"
    / "arg2_1.0"
    / "001.0-071.0"
    / "I10"
    / "S2_63.206_0.0876_0.2019_50.658"
    / "run.def"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "reference_mode" / "used_run.def"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a complete local reference-mode run.def by applying "
            "the paper sensitivity run.def overrides on top of the audited "
            "1961 trace used_run.def defaults."
        )
    )
    parser.add_argument("--base-used-run-def", type=Path, default=DEFAULT_TRACE_USED_RUN_DEF)
    parser.add_argument("--reference-override-run-def", type=Path, default=DEFAULT_REFERENCE_OVERRIDE_RUN_DEF)
    parser.add_argument("--reference-final-run-def", type=Path, default=DEFAULT_REFERENCE_FINAL_RUN_DEF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base = read_run_def_values(args.base_used_run_def)
    override = read_run_def_values(args.reference_override_run_def)
    final = static_run_def_values(read_run_def_values(args.reference_final_run_def))
    merged = materialize_run_def_values(base, override, final)

    header = [
        "# Materialized local reference-mode run.def for ORCHIDEE-MAN-JAX development.",
        f"# Base materialized defaults: {args.base_used_run_def}",
        f"# Paper sensitivity overrides: {args.reference_override_run_def}",
        f"# Archived final reference run.def overrides except year-dynamic keys: {args.reference_final_run_def}",
        "# Job protocol overrides: Job0_bio/Job_001.0-071.0.sh remplace lines 62-77, 159-163, 182, and 288-290.",
        "# This file is generated under outputs/ and is not Fortran source truth.",
    ]
    write_materialized_run_def(merged, args.output, header_lines=header)
    print(
        {
            "output": str(args.output),
            "base_keys": len(base),
            "override_keys": len(override),
            "reference_final_static_keys": len(final),
            "job_protocol_override_keys": len(JOB_PROTOCOL_OVERRIDES),
            "merged_keys": len(merged),
            "overridden_keys": sorted(key for key in override if key in base)[:20],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
