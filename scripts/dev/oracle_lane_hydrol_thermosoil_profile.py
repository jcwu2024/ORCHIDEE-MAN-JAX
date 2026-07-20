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

FAMILY = "thermosoil_explicit_snow_profile"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_thermosoil_profile.f90.template"
LEDGER_ENTRY = "thermosoil.conditional.explicit_snow_thermal_profile"


def _branch_coverage() -> dict[str, object]:
    resolution = json.loads(
        (ROOT / "outputs/reference_mode/pft14_control_flow_resolution.json").read_text(
            encoding="utf-8"
        )
    )
    rules = yaml.safe_load(
        (ROOT / "docs/source_audits/pft14_arm_dispositions_sechiba.yaml").read_text(
            encoding="utf-8"
        )
    )["rules"]
    dispositions = {item["branch_id"]: item["decisions"] for item in rules}
    line_cases = {
        1806: {"true": ["active_pfts"], "false": ["inactive_pft14"]},
        1807: {"true": ["explicit_snow"], "false": ["no_explicit_snow"]},
        1817: {"false": ["paper_static_non_laidev"]},
        1830: {"true": ["active_pfts"], "false": ["inactive_pft14"]},
        1849: {"false": ["printlev_zero"]},
    }
    arms = []
    missing = []
    for row in resolution["rows"]:
        if LEDGER_ENTRY not in row.get("process_ledger_ids", []):
            continue
        decisions = dispositions.get(row["control_flow_id"])
        if decisions is None:
            missing.append(row["control_flow_id"])
            continue
        for side, decision in decisions.items():
            if decision["disposition"] not in {
                "pft14_active",
                "pft14_conditional",
                "paper_static_active",
            }:
                continue
            side_name = str(side).lower()
            cases = line_cases.get(int(row["line"]), {}).get(side_name, [])
            arms.append(
                {
                    "arm_id": f"{row['control_flow_id']}:{side_name}",
                    "input_cases": cases,
                    "source_condition": row["source"],
                }
            )
    missing_cases = [item["arm_id"] for item in arms if not item["input_cases"]]
    return {
        "schema_version": 1,
        "ledger_entry": LEDGER_ENTRY,
        "arms": arms,
        "missing_disposition": missing,
        "missing_case_assignment": missing_cases,
        "branch_complete": not missing and not missing_cases,
    }


def _span(a, b):
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[a - 1 : b])


def _read(path):
    v = {}
    with path.open(newline="", encoding="ascii") as h:
        for r in csv.DictReader(h):
            v.setdefault(r["field"], []).append(float(r["value"]))
    return {k: np.asarray(x) for k, x in v.items()}


def _jax():
    from jax_orchidee.sechiba.thermosoil import (
        thermosoil_profile_explicit_snow,
        thermosoil_profile_no_explicit_snow,
    )

    ts = np.array([280.0, 275.0, 268.0])
    v = np.zeros((3, 14))
    v[:, 0] = 0.2
    v[:, 13] = [0.0, 0.7, 0.8]
    fsv = np.array([0.0, 0.5, 1.0])
    fsn = np.zeros((3, 2))
    fsn[1, :] = 0.2
    tnb = np.array([0.0, 0.1, 0.2])
    snow = np.full((3, 3), 270.0)
    snow[:, 2] = [271.0, 269.0, 265.0]
    mask = np.ones((3, 14), bool)
    mask[0, 13] = False
    r = thermosoil_profile_explicit_snow(
        ptn=np.full((3, 4, 14), 260.0),
        cgrnd=np.full((3, 3, 14), 5.0),
        dgrnd=np.full((3, 3, 14), 0.8),
        cgrnd_snow=np.full((3, 3), 10.0),
        dgrnd_snow=np.full((3, 3), 0.9),
        temp_sol_new=ts,
        snowtemp=snow,
        veget_max=v,
        frac_snow_veg=fsv,
        frac_snow_nobio=fsn,
        totfrac_nobio=tnb,
        veget_mask_2d=mask,
        nslm=4,
    )
    legacy = thermosoil_profile_no_explicit_snow(
        ptn=np.full((3, 4, 14), 260.0),
        cgrnd=np.full((3, 3, 14), 5.0),
        dgrnd=np.full((3, 3, 14), 0.8),
        temp_sol_new=ts,
        temp_sol_new_pft=np.broadcast_to(ts[:, None], (3, 14)),
        veget_max=v,
        ok_laidev=np.zeros(14, dtype=bool),
        lambda_thermal=1.0,
        veget_mask_2d=mask,
        nslm=4,
    )
    return {
        "ptn": np.asarray(r.ptn).ravel(),
        "stempdiag": np.asarray(r.stempdiag).ravel(),
        "noexplicit_ptn": np.asarray(legacy.ptn).ravel(),
        "noexplicit_stempdiag": np.asarray(legacy.stempdiag).ravel(),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    p = _span(1761, 1851)
    d = _span(3484, 3513)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(
        prefix="orchidee_thermosoil_profile_oracle_"
    ) as td:
        b = Path(td)
        s = b / "oracle.f90"
        s.write_bytes(
            TEMPLATE.read_bytes()
            .replace(b"! <THERMOSOIL_PROFILE>", p)
            .replace(b"! <THERMOSOIL_DIAGLEV>", d)
        )
        e = b / "oracle.exe"
        meta = compile_fortran(s, e, compiler)
        subprocess.run(
            [str(e), str(csv_path.resolve())],
            cwd=b,
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
                    "no snow",
                    "mixed vegetation/nobio snow",
                    "full vegetation snow",
                    "inactive PFT14 mask",
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
    c = [float_comparison(n, f[n], j[n], rtol=1e-12, atol=1e-14) for n in f]
    coverage = _branch_coverage()
    coverage_path = output_dir / "branch_coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    passed = all(item["passed"] for item in c)
    return write_result(
        output_dir,
        FAMILY,
        c,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": {
                "thermosoil_profile": hashlib.sha256(p).hexdigest(),
                "thermosoil_diaglev": hashlib.sha256(d).hexdigest(),
            },
            "compiler": meta,
            "branch_coverage_asset": str(coverage_path.relative_to(ROOT)),
            "branch_complete_ledger_entries": [LEDGER_ENTRY]
            if passed and coverage["branch_complete"]
            else [],
            "path_evidence_ledger_entries": [
                "thermosoil.conditional.explicit_snow_thermal_profile"
            ],
            "verified_ledger_entries": [LEDGER_ENTRY]
            if passed and coverage["branch_complete"]
            else [],
        },
    )


if __name__ == "__main__":
    try:
        r = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout, file=sys.stderr)
        raise
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["status"] == "passed" else 1)
