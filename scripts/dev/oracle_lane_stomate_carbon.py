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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)
from stomate_oracle_arm_coverage import audit_document  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILY = "stomate_constraints"
FRAGMENT = ROOT / "docs/source_audits/oracle_families/stomate_carbon.yaml"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_constraints.f90.template"
GAP_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_gap.f90.template"
VMAX_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_vmax.f90.template"
PRESCRIBE_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_prescribe.f90.template"
TURNOVER_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_turnover.f90.template"
ALLOCATION_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_allocation.f90.template"
NPP_TEMPLATE = ROOT / "scripts/dev/oracle_lane_stomate_npp.f90.template"


def _compile_legacy(
    source: Path, executable: Path, compiler: Path
) -> dict[str, object]:
    flags = [
        "-std=legacy",
        "-fdefault-real-8",
        "-ffree-line-length-none",
        "-O0",
        "-fcheck=all",
        "-ffpe-trap=invalid,zero,overflow",
        "-Wall",
        "-Wextra",
    ]
    completed = subprocess.run(
        [str(compiler), *flags, str(source), "-o", str(executable)],
        cwd=source.parent,
        env=compiler_environment(compiler),
        text=True,
        capture_output=True,
        check=True,
    )
    version = subprocess.run(
        [str(compiler), "--version"],
        env=compiler_environment(compiler),
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    return {
        "compiler": str(compiler),
        "compiler_version": version,
        "compile_flags": flags,
        "compile_stdout": completed.stdout,
        "compile_stderr": completed.stderr,
        "compile_unit_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _document() -> dict:
    return yaml.safe_load(FRAGMENT.read_text(encoding="utf-8"))


def validate_fragment() -> dict:
    document = _document()
    if document.get("schema_version") != 2 or not isinstance(
        document.get("families"), list
    ):
        raise ValueError("STOMATE fragment must have schema_version 2 and families")
    for family in document.get("families", []) + document.get("pending_families", []):
        for key in (
            "id",
            "runner",
            "ledger_entries",
            "procedures",
            "input_asset",
            "comparison_asset",
            "output_contract",
            "branch_cases",
        ):
            if key not in family:
                raise ValueError(f"{family.get('id')}: missing {key}")
        covered: dict[str, set[str]] = {
            entry: set() for entry in family["ledger_entries"]
        }
        for case in family["branch_cases"]:
            for key in (
                "id",
                "ledger_entry",
                "fortran_lines",
                "condition",
                "expected_arm",
                "input_case",
                "control_flow_arm_ids",
            ):
                if key not in case:
                    raise ValueError(f"{family['id']} branch case missing {key}")
            covered[case["ledger_entry"]].update(case["control_flow_arm_ids"])
        for entry in family["ledger_entries"]:
            if not any(
                case["ledger_entry"] == entry for case in family["branch_cases"]
            ):
                raise ValueError(f"{family['id']}:{entry}: no branch case")
    audit_document(document, ROOT)
    return document


def _procedure_bytes(path: Path, start: int, end: int) -> bytes:
    raw = path.read_bytes().splitlines(keepends=True)
    return b"".join(raw[start - 1 : end])


def compose_prescribe_oracle(destination: Path) -> None:
    """Compose the byte-exact prescribe harness used by parity and GCOV lanes."""

    source_file = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_prescribe.f90"
    span = _procedure_bytes(source_file, 90, 348)
    destination.write_bytes(
        PRESCRIBE_TEMPLATE.read_bytes().replace(b"! <PRESCRIBE_PROCEDURE>", span)
    )


def _jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import constraints_step

    nvm = 14
    natural = np.zeros(nvm, bool)
    tree = np.zeros(nvm, bool)
    peat = np.zeros(nvm, bool)
    natural[13] = True
    tree[13] = True
    wg = np.zeros((3, nvm))
    wg[1, 13] = 2000.0
    results = []
    for tmin14, tcm14 in ((1.0e20, 275.0), (265.0, 1.0e20)):
        tmin = np.full(nvm, 1.0e20)
        tcm = np.full(nvm, 1.0e20)
        tmin[13], tcm[13] = tmin14, tcm14
        regenerate = np.full((3, nvm), 0.5)
        regenerate[0, 13] = 1.0e-4
        results.append(
            constraints_step(
                t2m_month=np.array([270.0, 285.0, 280.0]),
                t2m_min_daily=np.array([260.0, 270.0, 275.0]),
                when_growthinit=wg,
                adapted=np.full((3, nvm), 0.25),
                regenerate=regenerate,
                tseason=np.array([285.0, 275.0, 281.0]),
                natural=natural,
                is_tree=tree,
                is_peat=peat,
                pheno_is_none=np.ones(nvm, bool),
                tmin_crit=tmin,
                tcm_crit=tcm,
                agriculture=False,
                undef=1.0e20,
            )
        )
    return {
        "adapted": np.concatenate(
            [np.asarray(result.adapted).ravel(order="F") for result in results]
        ),
        "regenerate": np.concatenate(
            [np.asarray(result.regenerate).ravel(order="F") for result in results]
        ),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = ROOT / "fortran_source/ORCHIDEE/src_stomate/lpj_constraints.f90"
    span = _procedure_bytes(source_file, 99, 246)
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    span_hash = hashlib.sha256(span).hexdigest()
    template = TEMPLATE.read_bytes().replace(b"! <CONSTRAINTS_PROCEDURE>", span)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_lane_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(template)
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
    values: dict[str, list[float]] = {}
    with csv_path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    f = {k: np.asarray(v) for k, v in values.items()}
    j = _jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "PFT1 inactive mask",
                    "PFT14 defined/undef climate parameters",
                    "grow, vernalization, and regenerate-min threshold sides",
                    "first/later SAVE call",
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
    comparisons = [float_comparison(k, f[k], j[k], rtol=1e-12, atol=1e-14) for k in f]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "ledger_entries": ["stomate.active.constraints"],
            "source_sha256": source_hash,
            "span_sha256": {"constraints": span_hash},
            "build": metadata,
        },
    )


def _gap_jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import gap_mortality_step

    npts, nvm, nparts = 3, 14, 12
    natural = np.zeros(nvm, bool)
    tree = np.zeros(nvm, bool)
    natural[13] = tree[13] = True
    present = np.zeros((npts, nvm), bool)
    present[:2, 13] = True
    biomass = np.zeros((npts, nvm, nparts, 1))
    biomass[0, 13, 0, 0] = 365.0
    biomass[0, 13, 5, 0] = 73.0
    biomass[1, 13, 0, 0] = 100.0
    common = dict(
        npp_longterm=np.full((npts, nvm), 20.0),
        turnover_longterm=np.zeros_like(biomass),
        lm_lastyearmax=np.full((npts, nvm), 50.0),
        pft_present=present,
        t2m_min_daily=np.full(npts, 280.0),
        tmin_spring_time=np.zeros((npts, nvm)),
        sla_calc=np.full((npts, nvm), 0.02),
        natural=natural,
        is_tree=tree,
        pasture=np.zeros(nvm, bool),
        availability_fact=np.full(nvm, 0.14),
        residence_time=np.full(nvm, 30.0),
        tmin_crit=np.full(nvm, -9999.0),
        leaf_tab=np.zeros(nvm, int),
        pheno_type=np.zeros(nvm, int),
        dt_days=2.0,
        lpj_gap_const_mort=True,
        ok_dgvm=False,
    )
    first = gap_mortality_step(
        **common,
        biomass=biomass,
        ind=np.ones((npts, nvm)),
        bm_to_litter=np.zeros_like(biomass),
    )
    result = gap_mortality_step(
        **common,
        biomass=np.asarray(first.biomass),
        ind=np.asarray(first.ind),
        bm_to_litter=np.asarray(first.bm_to_litter),
    )
    return {
        "mortality": np.asarray(result.mortality).ravel(order="F"),
        "ind": np.asarray(result.ind).ravel(order="F"),
        "biomass": np.asarray(result.biomass).transpose(1, 0, 2, 3).ravel(),
        "bm_to_litter": np.asarray(result.bm_to_litter).transpose(1, 0, 2, 3).ravel(),
    }


def run_gap_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = ROOT / "fortran_source/ORCHIDEE/src_stomate/lpj_gap.f90"
    span = _procedure_bytes(source_file, 117, 364)
    source = GAP_TEMPLATE.read_bytes().replace(b"! <GAP_PROCEDURE>", span)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_gap_") as td:
        build = Path(td)
        unit = build / "oracle.f90"
        unit.write_bytes(source)
        exe = build / "oracle.exe"
        metadata = compile_fortran(unit, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    values: dict[str, list[float]] = {}
    with csv_path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    fortran = {key: np.asarray(value) for key, value in values.items()}
    jax = _gap_jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "PFT1 inactive mask",
                    "PFT14 constant mortality",
                    "PFT14 present/absent mask",
                    "first/later SAVE call",
                    "biomass/litter writeback",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(key, fortran[key], jax[key], rtol=1e-12, atol=1e-14)
        for key in fortran
    ]
    return write_result(
        output_dir,
        "stomate_gap_mortality",
        comparisons,
        {
            "ledger_entries": ["stomate.active.gap_mortality"],
            "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "span_sha256": {"gap": hashlib.sha256(span).hexdigest()},
            "build": metadata,
        },
    )


def _vmax_jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import vmax_step

    npts, nvm, nleaf = 3, 14, 4
    age = np.zeros((npts, nvm, nleaf))
    frac = np.zeros_like(age)
    age[0, 13] = [5.0, 40.0, 120.0, 220.0]
    frac[0, 13] = [0.6, 0.25, 0.1, 0.05]
    age[1, 13] = [0.0, 20.0, 80.0, 180.0]
    frac[1, 13] = [0.0, 0.5, 0.3, 0.2]
    nlim = np.ones((npts, nvm))
    nlim[:, 13] = [0.8, 1.1, 1.0]
    laidev = np.zeros(nvm, bool)
    laidev[13] = False
    first = vmax_step(
        leaf_age=age,
        leaf_frac=frac,
        vcmax25=np.full(nvm, 50.0),
        n_limfert=nlim,
        leaf_timecst=np.full(nvm, 100.0),
        leafagecrit=np.full(nvm, 200.0),
        pheno_type=np.zeros(nvm, int),
        leaf_tab=np.zeros(nvm, int),
        ok_laidev=laidev,
        dt_days=1.0,
    )
    result = vmax_step(
        leaf_age=np.asarray(first.leaf_age),
        leaf_frac=np.asarray(first.leaf_frac),
        vcmax25=np.full(nvm, 50.0),
        n_limfert=nlim,
        leaf_timecst=np.full(nvm, 100.0),
        leafagecrit=np.full(nvm, 200.0),
        pheno_type=np.zeros(nvm, int),
        leaf_tab=np.zeros(nvm, int),
        ok_laidev=laidev,
        dt_days=1.0,
    )
    return {
        "vcmax": np.asarray(result.vcmax).ravel(order="F"),
        "leaf_age": np.asarray(result.leaf_age).transpose(1, 0, 2).ravel(),
        "leaf_frac": np.asarray(result.leaf_frac).transpose(1, 0, 2).ravel(),
    }


def run_vmax_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_vmax.f90"
    span = _procedure_bytes(source_file, 105, 363)
    source = VMAX_TEMPLATE.read_bytes().replace(b"! <VMAX_PROCEDURE>", span)
    csv_path = output_dir / "fortran_outputs_vmax.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_vmax_") as td:
        build = Path(td)
        unit = build / "oracle.f90"
        unit.write_bytes(source)
        exe = build / "oracle.exe"
        metadata = compile_fortran(unit, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    values: dict[str, list[float]] = {}
    with csv_path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    fortran = {key: np.asarray(value) for key, value in values.items()}
    jax = _vmax_jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "PFT1 inactive mask",
                    "PFT14 leaf-age class transfer",
                    "PFT14 zero/nonzero leaf fractions",
                    "first/later SAVE call",
                    "paper ordinary vcmax branch",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons_vmax.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(key, fortran[key], jax[key], rtol=1e-12, atol=1e-14)
        for key in fortran
    ]
    return write_result(
        output_dir,
        "stomate_final_lai_vmax",
        comparisons,
        {
            "ledger_entries": ["stomate.active.final_lai_vmax"],
            "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "span_sha256": {"vmax": hashlib.sha256(span).hexdigest()},
            "build": metadata,
        },
    )


def _prescribe_jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import prescribe_step

    npts, nvm, nparts, nleaf = 3, 14, 12, 4
    veget = np.zeros((npts, nvm))
    veget[:, 13] = [0.8, 0.4, 0.0]
    natural = np.zeros(nvm, bool)
    natural[13] = True
    tree = np.zeros(nvm, bool)
    tree[13] = True
    sapling = np.zeros((nvm, nparts, 1))
    sapling[:, :, 0] = np.arange(1.0, nparts + 1.0)
    common = dict(
        veget_max=veget,
        dt_days=1.0,
        pft_present=np.zeros((npts, nvm), bool),
        everywhere=np.zeros((npts, nvm)),
        when_growthinit=np.zeros((npts, nvm)),
        biomass=np.zeros((npts, nvm, nparts, 1)),
        leaf_frac=np.zeros((npts, nvm, nleaf)),
        ind=np.zeros((npts, nvm)),
        cn_ind=np.zeros((npts, nvm)),
        co2_to_bm=np.zeros((npts, nvm)),
        natural=natural,
        pasture=np.zeros(nvm, bool),
        is_tree=tree,
        bm_sapl=sapling,
        maxdia=np.full(nvm, 0.5),
        pheno_is_none=np.ones(nvm, bool),
        bm_sapl_rescale=40.0,
    )
    result = {}
    for prefix, restart_none in (("cold", True), ("restart", False)):
        value = prescribe_step(
            **common, firstcall=True, stomate_restart_none=restart_none
        )
        result[f"{prefix}_present"] = np.asarray(value.pft_present, float).ravel(
            order="F"
        )
        result[f"{prefix}_everywhere"] = np.asarray(value.everywhere).ravel(order="F")
        result[f"{prefix}_ind"] = np.asarray(value.ind).ravel(order="F")
        result[f"{prefix}_cn"] = np.asarray(value.cn_ind).ravel(order="F")
        result[f"{prefix}_co2"] = np.asarray(value.co2_to_bm).ravel(order="F")
        result[f"{prefix}_biomass"] = (
            np.asarray(value.biomass).transpose(1, 0, 2, 3).ravel()
        )
        result[f"{prefix}_leaf_frac"] = (
            np.asarray(value.leaf_frac).transpose(1, 0, 2).ravel()
        )

    for prefix, is_tree_pft14 in (("tree", True), ("grass", False)):
        matrix_veget = np.zeros((npts, nvm))
        matrix_veget[:, 13] = (
            [0.8, 0.4, 0.0] if is_tree_pft14 else [0.8, 0.0, 0.4]
        )
        matrix_tree = np.zeros(nvm, bool)
        matrix_tree[13] = is_tree_pft14
        matrix_biomass = np.zeros((npts, nvm, nparts, 1))
        if is_tree_pft14:
            matrix_biomass[0, 13, [1, 2, 3, 4], 0] = 1.0e4
            matrix_biomass[1, 13, [1, 2, 3, 4], 0] = 1.0e3
        value = prescribe_step(
            **{
                **common,
                "veget_max": matrix_veget,
                "biomass": matrix_biomass,
                "is_tree": matrix_tree,
            },
            firstcall=False,
            stomate_restart_none=False,
        )
        result[f"{prefix}_present"] = np.asarray(value.pft_present, float).ravel(order="F")
        result[f"{prefix}_everywhere"] = np.asarray(value.everywhere).ravel(order="F")
        result[f"{prefix}_ind"] = np.asarray(value.ind).ravel(order="F")
        result[f"{prefix}_cn"] = np.asarray(value.cn_ind).ravel(order="F")
        result[f"{prefix}_co2"] = np.asarray(value.co2_to_bm).ravel(order="F")
        result[f"{prefix}_biomass"] = np.asarray(value.biomass).transpose(1, 0, 2, 3).ravel()
        result[f"{prefix}_leaf_frac"] = np.asarray(value.leaf_frac).transpose(1, 0, 2).ravel()
    return result


def run_prescribe_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_prescribe.f90"
    span = _procedure_bytes(source_file, 90, 348)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_prescribe_") as td:
        build = Path(td)
        unit = build / "oracle.f90"
        compose_prescribe_oracle(unit)
        exe = build / "oracle.exe"
        metadata = compile_fortran(unit, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    values: dict[str, list[float]] = {}
    with csv_path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    fortran = {k: np.asarray(v) for k, v in values.items()}
    jax = _prescribe_jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "cold-start NONE gate",
                    "restart-backed gate",
                    "PFT1 mask",
                    "PFT14 full state writeback",
                    "PFT14 zero-vegetation mask",
                    "PFT14 tree crown/density high-wood, low-wood, and zero-cover arms",
                    "PFT14 grass crown and density WHERE true/false arms",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(k, fortran[k], jax[k], rtol=1e-12, atol=1e-14) for k in fortran
    ]
    return write_result(
        output_dir,
        "stomate_prescribe_restart_gate",
        comparisons,
        {
            "ledger_entries": ["stomate.active.prescribe_restart_gate"],
            "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "span_sha256": {"prescribe": hashlib.sha256(span).hexdigest()},
            "build": metadata,
        },
    )


def _turnover_jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import SENESCENCE_NONE, turnover_step

    npts, nvm, nparts, nleaf = 3, 14, 12, 4
    present = np.zeros((npts, nvm), bool)
    present[:, 13] = True
    biomass = np.zeros((npts, nvm, nparts, 1))
    biomass[:2, 13, 0, 0] = 100.0
    biomass[:2, 13, 5, 0] = 40.0
    biomass[:2, 13, 6, 0] = 90.0
    biomass[:2, 13, 1, 0] = 730.0
    biomass[0, 13, 3, 0] = 10.0
    lage = np.zeros((npts, nvm, nleaf))
    lfrac = np.zeros_like(lage)
    lage[:2, 13, 0] = [60.0, 10.0]
    lfrac[:2, 13, 0] = 1.0
    tree = np.zeros(nvm, bool)
    tree[13] = True
    stype = np.zeros(nvm, int)
    stype[13] = SENESCENCE_NONE
    common = dict(
        pft_present=present,
        herbivores=np.zeros((npts, nvm)),
        maxmoiavail_lastyear=np.ones((npts, nvm)),
        minmoiavail_lastyear=np.zeros((npts, nvm)),
        moiavail_week=np.where(np.arange(nvm)[None, :] == 13, 0.1, 0.5).repeat(
            npts, axis=0
        ),
        t2m_longterm=np.full(npts, 293.15),
        t2m_month=np.full(npts, 293.15),
        t2m_week=np.full(npts, 293.15),
        veget_max=present.astype(float),
        gdd_from_growthinit=np.zeros((npts, nvm)),
        sla_calc=np.full((npts, nvm), 0.02),
        senescence_type=stype,
        is_tree=tree,
        natural=np.ones(nvm, bool),
        is_grassland_manag=np.zeros(nvm, bool),
        ok_laidev=np.zeros(nvm, bool),
        min_leaf_age_for_senescence=np.full(nvm, 30.0),
        gdd_senescence=np.full(nvm, 100.0),
        senescence_temp=np.zeros((nvm, 3)),
        hum_frac=np.full(nvm, 0.5),
        senescence_hum=np.full(nvm, 0.2),
        nosenescence_hum=np.full(nvm, 0.8),
        max_turnover_time=np.full(nvm, 80.0),
        min_turnover_time=np.full(nvm, 10.0),
        leaffall=np.full(nvm, 10.0),
        lai_max=np.full(nvm, 4.0),
        leafagecrit=np.full(nvm, 100.0),
        lai_initmin=np.full(nvm, 0.3),
        tau_fruit=np.full(nvm, 90.0),
        tau_sap=np.full(nvm, 730.0),
    )
    first = turnover_step(
        **common,
        leaf_age=lage,
        leaf_frac=lfrac,
        age=np.full((npts, nvm), 10.0),
        lai=np.zeros((npts, nvm)),
        biomass=biomass,
        turnover_time=np.full((npts, nvm), 80.0),
        nrec=np.zeros((npts, nvm), int),
    )
    result = turnover_step(
        **common,
        leaf_age=np.asarray(first.leaf_age),
        leaf_frac=np.asarray(first.leaf_frac),
        age=np.asarray(first.age),
        lai=np.asarray(first.lai),
        biomass=np.asarray(first.biomass),
        turnover_time=np.asarray(first.turnover_time),
        nrec=np.zeros((npts, nvm), int),
    )
    return {
        "senescence": np.asarray(result.senescence, float).ravel(order="F"),
        "age": np.asarray(result.age).ravel(order="F"),
        "lai": np.asarray(result.lai).ravel(order="F"),
        "turnover_time": np.asarray(result.turnover_time).ravel(order="F"),
        "c_export": np.asarray(result.c_export).ravel(order="F"),
        "biomass": np.asarray(result.biomass).transpose(1, 0, 2, 3).ravel(),
        "turnover": np.asarray(result.turnover).transpose(1, 0, 2, 3).ravel(),
        "leaf_age": np.asarray(result.leaf_age).transpose(1, 0, 2).ravel(),
        "leaf_frac": np.asarray(result.leaf_frac).transpose(1, 0, 2).ravel(),
    }


def run_turnover_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    sf = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_turnover.f90"
    span = _procedure_bytes(sf, 169, 947)
    source = TURNOVER_TEMPLATE.read_bytes().replace(b"! <TURNOVER_PROCEDURE>", span)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_turnover_") as td:
        b = Path(td)
        u = b / "oracle.f90"
        u.write_bytes(source)
        exe = b / "oracle.exe"
        metadata = compile_fortran(u, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=b,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    values = {}
    with csv_path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            values.setdefault(row["field"], []).append(float(row["value"]))
    f = {k: np.asarray(v) for k, v in values.items()}
    j = _turnover_jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "PFT1 mask",
                    "PFT14 dry senescence",
                    "leaf age fruit sapwood writeback",
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
    comparisons = [float_comparison(k, f[k], j[k], rtol=1e-12, atol=1e-14) for k in f]
    return write_result(
        output_dir,
        "stomate_turnover",
        comparisons,
        {
            "ledger_entries": ["stomate.active.turnover"],
            "source_sha256": hashlib.sha256(sf.read_bytes()).hexdigest(),
            "span_sha256": {"turn": hashlib.sha256(span).hexdigest()},
            "build": metadata,
        },
    )


def _allocation_jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import allocation_step

    npts, nvm, nparts, nleaf = 5, 14, 12, 4
    bio = np.zeros((npts, nvm, nparts, 1))
    bio[:, 13, 0, 0] = [10.0, 0.0, 10.0, 10.0, 10.0]
    bio[:, 13, 5, 0] = 5.0
    bio[:, 13, 7, 0] = [100.0, 0.0, 100.0, 100.0, 2000.0]
    la = np.zeros((npts, nvm, nleaf))
    la[:, 13, :] = [20.0, 30.0, 40.0, 50.0]
    lf = np.zeros_like(la)
    lf[:, 13, :] = 0.25
    lai = np.zeros((npts, nvm))
    lai[:, 13] = [0.4, 0.0, 20.0, 0.4, 0.4]
    veg = np.zeros((npts, nvm))
    veg[:, 13] = 1.0
    natural = np.zeros(nvm, bool)
    pasture = np.zeros(nvm, bool)
    tree = np.zeros(nvm, bool)
    crop = np.zeros(nvm, bool)
    natural[13] = tree[13] = True
    senescence = np.zeros((npts, nvm), bool)
    senescence[3, 13] = True
    common = dict(
        lai=lai,
        veget_max=veg,
        senescence=senescence,
        when_growthinit=np.full((npts, nvm), 5.0),
        moiavail_week=np.full((npts, nvm), 0.7),
        tsoil_month=np.tile([283.15, 293.15], (npts, 1)),
        soilhum_month=np.tile([0.4, 0.8], (npts, 1)),
        age=np.zeros((npts, nvm)),
        z_soil=np.array([0.0, 1.0, 2.0]),
        sla_calc=np.full((npts, nvm), 0.02),
        natural=natural,
        pasture=pasture,
        is_tree=tree,
        ok_LAIdev=crop,
        r0=np.full(nvm, 0.35),
        s0=np.full(nvm, 0.35),
        ext_coeff=np.full(nvm, 0.5),
        lai_max=np.full(nvm, 12.0),
        lai_max_to_happy=np.full(nvm, 0.5),
        tau_leafinit=np.full(nvm, 10.0),
        alloc_min=np.full(nvm, 0.2),
        alloc_max=np.full(nvm, 0.8),
        demi_alloc=np.full(nvm, 100.0),
        alloc_agr_st=np.where(np.arange(nvm) == 13, 0.25, 0.0),
        alloc_agr_pn=np.where(np.arange(nvm) == 13, 0.05, 0.0),
        when_growthinit_cut=np.full((npts, nvm), 100.0),
        is_grassland_manag=np.zeros(nvm, bool),
        ecureuil=np.zeros(nvm),
    )
    first = allocation_step(**common, biomass=bio, leaf_age=la, leaf_frac=lf)
    result = allocation_step(
        **common,
        biomass=np.asarray(first.biomass),
        leaf_age=np.asarray(first.leaf_age),
        leaf_frac=np.asarray(first.leaf_frac),
    )
    return {
        "biomass": np.asarray(result.biomass).transpose(1, 0, 2, 3).ravel(),
        "f_alloc": np.asarray(result.f_alloc).transpose(1, 0, 2).ravel(),
        "leaf_age": np.asarray(result.leaf_age).transpose(1, 0, 2).ravel(),
        "leaf_frac": np.asarray(result.leaf_frac).transpose(1, 0, 2).ravel(),
    }


def run_allocation_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    sf = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_alloc.f90"
    span = _procedure_bytes(sf, 146, 834)
    source = ALLOCATION_TEMPLATE.read_bytes().replace(b"! <ALLOC_PROCEDURE>", span)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_alloc_") as td:
        b = Path(td)
        u = b / "oracle.f90"
        u.write_bytes(source)
        exe = b / "oracle.exe"
        metadata = _compile_legacy(u, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=b,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    values = {}
    with csv_path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            values.setdefault(row["field"], []).append(float(row["value"]))
    f = {k: np.asarray(v) for k, v in values.items()}
    j = _allocation_jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "PFT1 mask",
                    "PFT14 reserve translocation",
                    "water/light/nitrogen allocation",
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
    comparisons = [float_comparison(k, f[k], j[k], rtol=1e-12, atol=1e-14) for k in f]
    return write_result(
        output_dir,
        "stomate_allocation",
        comparisons,
        {
            "ledger_entries": ["stomate.active.allocation"],
            "source_sha256": hashlib.sha256(sf.read_bytes()).hexdigest(),
            "span_sha256": {"alloc": hashlib.sha256(span).hexdigest()},
            "build": metadata,
        },
    )


def _npp_jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import (
        npp_closed_update,
        npp_leaf_age_sla_age_update,
    )

    npts, nvm, nparts = 5, 14, 12
    present = np.zeros((npts, nvm), bool)
    present[:4, 13] = True
    bio = np.zeros((npts, nvm, nparts, 1))
    bio[:, 13, 0, 0] = 100.0
    bio[1, 13, 0, 0] = 0.0
    bio[3, 13, 5, 0] = -1.0
    fa = np.zeros((npts, nvm, nparts))
    fa[:, 13, 0] = 0.6
    fa[:, 13, 5] = 0.4
    rmp = np.zeros_like(fa)
    rmp[:, 13, 0] = [0.7, 0.7, 10.0, 0.7, 0.7]
    rmp[:, 13, 5] = [0.3, 0.3, 0.0, 0.3, 0.3]
    gpp = np.zeros((npts, nvm))
    gpp[:, 13] = [10.0, 0.0, 10.0, 0.0, 10.0]
    leaf_age = np.full((npts, nvm, 4), 20.0)
    leaf_frac = np.full((npts, nvm, 4), 0.25)
    age_state = np.full((npts, nvm), 10.0)
    sla_calc = np.full((npts, nvm), 0.02)
    sla_calc[3, 13] = 0.05
    sla_age1 = np.full((npts, nvm), 0.02)
    biomass = bio
    for _ in range(2):
        closed = npp_closed_update(
            biomass,
            gpp,
            fa,
            rmp,
            present,
            np.full(nvm, 0.25),
            dt_days=1.0,
            tax_max=0.8,
        )
        age = npp_leaf_age_sla_age_update(
            closed.biomass,
            closed.biomass_before_alloc,
            closed.bm_alloc,
            leaf_age,
            leaf_frac,
            age_state,
            present,
            np.ones(nvm, bool),
            sla_age1,
            sla_calc,
            np.full(nvm, 0.03),
            np.full(nvm, 0.01),
            dt_days=1.0,
        )
        biomass = np.asarray(closed.biomass)
        leaf_age = np.asarray(age.leaf_age)
        leaf_frac = np.asarray(age.leaf_frac)
        age_state = np.asarray(age.age)
        sla_calc = np.asarray(age.sla_calc)
        sla_age1 = np.asarray(age.sla_age1)
    return {
        "resp_maint": np.asarray(closed.resp_maint).ravel(order="F"),
        "resp_growth": np.asarray(closed.resp_growth).ravel(order="F"),
        "npp": np.asarray(closed.npp).ravel(order="F"),
        "age": np.asarray(age.age).ravel(order="F"),
        "sla_calc": np.asarray(age.sla_calc).ravel(order="F"),
        "sla_age1": np.asarray(age.sla_age1).ravel(order="F"),
        "biomass": np.asarray(closed.biomass).transpose(1, 0, 2, 3).ravel(),
        "bm_alloc": np.asarray(closed.bm_alloc).transpose(1, 0, 2, 3).ravel(),
        "leaf_age": np.asarray(age.leaf_age).transpose(1, 0, 2).ravel(),
        "leaf_frac": np.asarray(age.leaf_frac).transpose(1, 0, 2).ravel(),
    }


def run_npp_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict:
    validate_fragment()
    output_dir.mkdir(parents=True, exist_ok=True)
    nf = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_npp.f90"
    cf = ROOT / "fortran_source/ORCHIDEE/src_sticslai/crop_alloc.f90"
    nspan = _procedure_bytes(nf, 116, 725)
    cspan = _procedure_bytes(cf, 18, 384)
    source = (
        NPP_TEMPLATE.read_bytes()
        .replace(b"! <CROP_BMALLOC_PROCEDURE>", cspan)
        .replace(b"! <NPP_PROCEDURE>", nspan)
    )
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_npp_") as td:
        b = Path(td)
        u = b / "oracle.f90"
        u.write_bytes(source)
        exe = b / "oracle.exe"
        metadata = _compile_legacy(u, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=b,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    values = {}
    with csv_path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            values.setdefault(row["field"], []).append(float(row["value"]))
    f = {k: np.asarray(v) for k, v in values.items()}
    j = _npp_jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "PFT1 mask",
                    "PFT14 non-crop GPP-maintenance-growth",
                    "biomass leaf SLA age writeback",
                ],
                "compiled_callee": "crop_alloc::crop_bmalloc",
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", f, j, rtol=1e-12, atol=1e-14
    )
    comparisons = [float_comparison(k, f[k], j[k], rtol=1e-12, atol=1e-14) for k in f]
    return write_result(
        output_dir,
        "stomate_npp_growth",
        comparisons,
        {
            "ledger_entries": ["stomate.active.npp_growth"],
            "source_sha256": {
                "npp_calc": hashlib.sha256(nf.read_bytes()).hexdigest(),
                "crop_bmalloc": hashlib.sha256(cf.read_bytes()).hexdigest(),
            },
            "span_sha256": {
                "npp_calc": hashlib.sha256(nspan).hexdigest(),
                "crop_bmalloc": hashlib.sha256(cspan).hexdigest(),
            },
            "build": metadata,
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
