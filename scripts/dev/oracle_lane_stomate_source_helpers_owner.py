from __future__ import annotations

import csv
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

from jax_orchidee.stomate.initialize import stomate_var_init_carbon_owner  # noqa: E402
from jax_orchidee.stomate.source_helpers import sort_ascending_owner  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "stomate_source_helpers_owner"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_source_helpers_owner.f90.template"
VMAX_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_vmax.f90"
DEADLEAF_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90"
PROCEDURES = {"sort_ascending", "stomate_var_init"}


def compose(path: Path) -> dict[str, str]:
    spans = {
        "sort_ascending": extract_procedure_bytes(SOURCE, "sort_ascending"),
        "stomate_var_init": extract_procedure_bytes(SOURCE, "stomate_var_init"),
        "vmax": extract_procedure_bytes(VMAX_SOURCE, "vmax"),
        "deadleaf": extract_procedure_bytes(DEADLEAF_SOURCE, "deadleaf"),
    }
    source = TEMPLATE.read_bytes()
    for name, span in spans.items():
        source = source.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _jax_values() -> dict[str, np.ndarray]:
    values = np.asarray(
        [
            [4.0, -2.0, 4.0, 0.0, 9.0, -2.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    npts, nvm, nleafages = 3, 14, 4
    veget_cov_max = np.zeros((npts, nvm))
    leaf_age = np.zeros((npts, nvm, nleafages))
    leaf_frac = np.zeros_like(leaf_age)
    dead_leaves = np.zeros((npts, nvm, 2))
    sla_calc = np.full((npts, nvm), 0.02)
    for i in range(npts):
        for j in range(nvm):
            veget_cov_max[i, j] = 0.005 * ((i + 1) + (j + 1))
            dead_leaves[i, j, 0] = 0.2 * ((i + 1) + (j + 1))
            dead_leaves[i, j, 1] = 0.1 * (2 * (i + 1) + (j + 1))
            for m in range(nleafages):
                leaf_age[i, j, m] = 10.0 * (m + 1)
                leaf_frac[i, j, m] = 0.25
    common = {
        "leaf_age": leaf_age,
        "leaf_frac": leaf_frac,
        "dead_leaves": dead_leaves,
        "veget_cov_max": veget_cov_max,
        "sla_calc": sla_calc,
        "val_exp": 999999.0,
    }
    preserved = stomate_var_init_carbon_owner(
        assim_param=np.full((npts, nvm, 1), 17.0),
        **common,
    )
    initialized = stomate_var_init_carbon_owner(
        assim_param=np.full((npts, nvm, 1), 999999.0),
        vmax_inputs={
            "ivcmax": 0,
            "vcmax25": np.full(nvm, 50.0),
            "n_limfert": np.ones((npts, nvm)),
            "leaf_timecst": np.full(nvm, 100.0),
            "leafagecrit": np.full(nvm, 200.0),
            "pheno_type": np.zeros(nvm, dtype=int),
            "leaf_tab": np.zeros(nvm, dtype=int),
            "ok_laidev": np.zeros(nvm, dtype=bool),
            "ok_dgvm": False,
            "ok_nlim": False,
            "min_stomate": 0.0,
        },
        **common,
    )
    return {
        "sorted": np.asarray(sort_ascending_owner(values)).ravel(),
        "deadleaf_cover": np.asarray(initialized.deadleaf_cover).ravel(),
        "assim_preserved": np.asarray(preserved.assim_param).ravel(),
        "assim_initialized": np.asarray(initialized.assim_param).ravel(),
    }


def run_oracle(
    output_dir: Path,
    compiler: Path = DEFAULT_COMPILER,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_helpers_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        compiler_metadata = compile_fortran(source, executable, compiler)
        fortran_output = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(fortran_output.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )

    fortran_values = _read(output_dir / "fortran_outputs.csv")
    jax_values = _jax_values()
    write_point_comparisons(
        output_dir / "point_comparisons.csv",
        fortran_values,
        jax_values,
        rtol=0.0,
        atol=0.0,
    )
    comparisons = [
        float_comparison(
            name,
            fortran_values[name],
            jax_values[name],
            rtol=0.0,
            atol=0.0,
        )
        for name in sorted(fortran_values)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "unsorted values with duplicate minima",
                    "descending values",
                    "already ascending values",
                    "all values equal",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "dependency_source_sha256": {
                "stomate_vmax.f90": hashlib.sha256(VMAX_SOURCE.read_bytes()).hexdigest(),
                "stomate_litter.f90": hashlib.sha256(DEADLEAF_SOURCE.read_bytes()).hexdigest(),
            },
            "procedure_span_sha256": hashes,
            "compiler": compiler_metadata,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
