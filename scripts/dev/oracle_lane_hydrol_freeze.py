from __future__ import annotations
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "hydrol_freeze_hydraulics"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_freeze.f90.template"


def _branch_coverage():
    resolution = json.loads((ROOT / "outputs/reference_mode/pft14_control_flow_resolution.json").read_text(encoding="utf-8"))
    rules = yaml.safe_load((ROOT / "docs/source_audits/pft14_arm_dispositions_sechiba.yaml").read_text(encoding="utf-8"))["rules"]
    dispositions = {item["branch_id"]: item["decisions"] for item in rules}
    arms = []
    missing = []
    for row in resolution["rows"]:
        if "hydrol.active.freeze_hydraulics" not in row.get("process_ledger_ids", []):
            continue
        decisions = dispositions.get(row["control_flow_id"])
        if decisions is None:
            missing.append(row["control_flow_id"])
            continue
        for side, decision in decisions.items():
            if decision["disposition"] in {"pft14_active", "pft14_conditional"}:
                arms.append({
                    "arm_id": f"{row['control_flow_id']}:{str(side).lower()}",
                    "input_cases": ["thermodynamic", "linear", "mineral", "peat", "warm", "transition", "frozen", "below_residual"],
                    "source_condition": row["source"],
                })
    return {
        "schema_version": 1,
        "ledger_entry": "hydrol.active.freeze_hydraulics",
        "arms": arms,
        "missing_disposition": missing,
        "missing_case_assignment": [],
        "branch_complete": not missing and all(item["input_cases"] for item in arms),
    }


def _span():
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[8331:8460])


def _read(path):
    values = {}
    with path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {k: np.asarray(v) for k, v in values.items()}


def _jax():
    from jax_orchidee.sechiba.hydrol import hydrol_soil_froz_profile

    temp = np.array(
        [
            [275.0, 273.15, 272.5, 270.0],
            [274.15, 273.6, 272.3, 270.0],
            [272.0, 273.15, 274.15, 270.0],
        ]
    )
    mc = np.full((3, 4, 6), 0.3)
    mc[0, 1, :] = 0.045000001
    mc[1, :, :] = 0.35
    mc[2, :, :] = 0.2
    thermo = hydrol_soil_froz_profile(
        temp_hydro=temp,
        mc=mc,
        njsc=np.array([1, 4, 10]),
        dh_mm=np.array([20.0, 40.0, 80.0, 160.0]),
        peat_hydro=True,
    )
    linear = hydrol_soil_froz_profile(
        temp_hydro=temp,
        mc=mc,
        njsc=np.array([1, 4, 10]),
        dh_mm=np.array([20.0, 40.0, 80.0, 160.0]),
        peat_hydro=True,
        ok_thermodynamical_freezing=False,
    )
    return {
        "profil_thermodynamic": np.asarray(thermo).ravel(order="C"),
        "profil_linear": np.asarray(linear).ravel(order="C"),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    span = _span()
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_hydrol_freeze_oracle_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(TEMPLATE.read_bytes().replace(b"! <HYDROL_SOIL_FROZ>", span))
        exe = build / "oracle.exe"
        metadata = compile_fortran(source, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    f = _read(csv_path)
    j = _jax()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "warm",
                    "transition",
                    "fully frozen",
                    "below residual",
                    "mineral tiles",
                    "peat tiles 4-6",
                    "thermodynamic freezing enabled and disabled",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", f, j, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, f[name], j[name], rtol=1e-12, atol=1e-14)
        for name in f
    ]
    coverage = _branch_coverage()
    (output_dir / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    branch_complete = all(item["passed"] for item in comparisons) and coverage["branch_complete"]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": hashlib.sha256(span).hexdigest(),
            "compiler": metadata,
            "branch_coverage_asset": str((output_dir / "branch_coverage.json").relative_to(ROOT)),
            "branch_complete_ledger_entries": ["hydrol.active.freeze_hydraulics"] if branch_complete else [],
            "path_evidence_ledger_entries": ["hydrol.active.freeze_hydraulics"],
            "verified_ledger_entries": (
                ["hydrol.active.freeze_hydraulics"]
                if coverage["branch_complete"] and all(item["passed"] for item in comparisons)
                else []
            ),
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
