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

FAMILY = "hydrol_infiltration"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_infilt.f90.template"
LEDGER_ENTRY = "hydrol.conditional.infiltration_and_excess_routing"


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
        7475: {"false": ["mineral_freeze"]},
        7491: {"true": ["peat_freeze"], "false": ["mineral_freeze"]},
        7514: {"true": ["peat_freeze"], "false": ["mineral_freeze"]},
        7523: {"true": ["mineral_freeze", "peat_freeze"], "false": ["mineral_nofreeze"]},
        7524: {"true": ["mineral_freeze"], "false": ["mineral_freeze", "peat_freeze"]},
        7536: {"true": ["peat_freeze"], "false": ["mineral_freeze"]},
        7542: {"true": ["all_positive_flux"], "false": ["all_completed_flux"]},
        7545: {"true": ["small_final_substep"], "false": ["large_full_substep"]},
        7570: {"false": ["mineral_freeze"]},
        7583: {"false": ["all_valid_flux"]},
    }
    arms: list[dict[str, object]] = []
    missing: list[str] = []
    for row in resolution["rows"]:
        if LEDGER_ENTRY not in row.get("process_ledger_ids", []):
            continue
        decisions = dispositions.get(row["control_flow_id"])
        if decisions is None:
            missing.append(row["control_flow_id"])
            continue
        for side, decision in decisions.items():
            if decision["disposition"] not in {"pft14_active", "pft14_conditional"}:
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


def _span():
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[7431:7590])


def _read(path):
    values = {}
    with path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {k: np.asarray(v) for k, v in values.items()}


def _jax():
    from jax_orchidee.sechiba.hydrol import hydrol_soil_infilt_explicit

    nj = np.array([1, 4, 10])
    sat = np.array([0.43, 0.45, 0.38])
    dz = np.array([10.0, 30.0, 70.0, 150.0])
    flux = np.array([0.2, 3.0, 20.0])
    k = np.full((3, 4), 0.08)
    root = np.ones((3, 4))
    temp = np.full((3, 4), 275.0)
    temp[:, 1] = 272.0
    mc = np.full((3, 4), 0.2)
    mc[1, 0] = 0.449
    mc[2, 1] = 0.37
    kfact = np.ones((4, 12))
    kfact[0, :] = 0.8
    kfact[1, :] = 0.5
    ks = np.full(12, 0.12)
    mineral = hydrol_soil_infilt_explicit(
        mc=mc,
        flux_infilt=flux,
        k=k,
        dz_mm=dz,
        dt_days=1800.0 / 86400.0,
        mcs=sat,
        ks=ks,
        kfact=kfact,
        kfact_root=root,
        njsc=nj,
        soil_tile_index=1,
        ok_freeze_cwrr=True,
        temp_hydro=temp,
    )
    peat = hydrol_soil_infilt_explicit(
        mc=mc,
        flux_infilt=flux,
        k=k,
        dz_mm=dz,
        dt_days=1800.0 / 86400.0,
        mcs=sat,
        ks=ks,
        kfact=kfact,
        kfact_root=root,
        njsc=nj,
        soil_tile_index=4,
        peat_hydro=True,
        mcs_peat=0.9,
        ks_peat=0.25,
        kfact_peat=np.array([0.9, 0.7, 0.4, 0.2]),
        ok_freeze_cwrr=True,
        temp_hydro=temp,
    )
    nofreeze = hydrol_soil_infilt_explicit(
        mc=mc,
        flux_infilt=flux,
        k=k,
        dz_mm=dz,
        dt_days=1800.0 / 86400.0,
        mcs=sat,
        ks=ks,
        kfact=kfact,
        kfact_root=root,
        njsc=nj,
        soil_tile_index=1,
        ok_freeze_cwrr=False,
        temp_hydro=temp,
    )
    return {
        "mineral_mc": np.asarray(mineral.mc).ravel(),
        "mineral_q": np.asarray(mineral.qinfilt),
        "mineral_ru": np.asarray(mineral.ru_infilt),
        "peat_mc": np.asarray(peat.mc).ravel(),
        "peat_q": np.asarray(peat.qinfilt),
        "peat_ru": np.asarray(peat.ru_infilt),
        "nofreeze_mc": np.asarray(nofreeze.mc).ravel(),
        "nofreeze_q": np.asarray(nofreeze.qinfilt),
        "nofreeze_ru": np.asarray(nofreeze.ru_infilt),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    span = _span()
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_hydrol_infilt_oracle_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(
            TEMPLATE.read_bytes().replace(b"! <HYDROL_SOIL_INFILT>", span)
        )
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
                    "small flux",
                    "near-saturated top",
                    "large excess runoff",
                    "frozen layer",
                    "mineral tile",
                    "peat tile",
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
    comparisons = [float_comparison(n, f[n], j[n], rtol=1e-12, atol=1e-14) for n in f]
    coverage = _branch_coverage()
    coverage_path = output_dir / "branch_coverage.json"
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    passed = all(item["passed"] for item in comparisons)
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": hashlib.sha256(span).hexdigest(),
            "compiler": metadata,
            "branch_coverage_asset": str(coverage_path.relative_to(ROOT)),
            "branch_complete_ledger_entries": [LEDGER_ENTRY]
            if passed and coverage["branch_complete"]
            else [],
            "path_evidence_ledger_entries": [
                "hydrol.conditional.infiltration_and_excess_routing"
            ],
            "verified_ledger_entries": [LEDGER_ENTRY]
            if passed and coverage["branch_complete"]
            else [],
        },
    )


if __name__ == "__main__":
    r = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["status"] == "passed" else 1)
