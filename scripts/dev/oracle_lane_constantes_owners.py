from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.parameters.constantes import (  # noqa: E402
    activate_sub_models,
    config_sechiba_parameters,
    config_stomate_parameters,
    veget_config,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_gcov import locate_generated_span, parse_gcov, select_arm_branch  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_result,
)

FAMILY = "constantes_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/constantes.f90"
VAR_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/constantes_var.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_constantes_owners.f90.template"
OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
CONTRACT_CLASSES = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
OWNER_IDS = {
    "pft14-owner-contract-83361683fc19",
    "pft14-owner-contract-970163ebd0f0",
    "pft14-owner-contract-40b5b8f14d8b",
    "pft14-owner-contract-474d2e53e54a",
}
PROCEDURES = (
    "activate_sub_models",
    "veget_config",
    "config_sechiba_parameters",
    "config_stomate_parameters",
)
CASES = {
    "activate_sub_models": ("stomate_off", "base", "fire_off", "age_on", "lpj_conflict"),
    "veget_config": ("base", "impveg", "map_format"),
    "config_sechiba_parameters": ("base", "impaze", "watstress", "overrides"),
    "config_stomate_parameters": ("base", "ch4_on", "ch4_on_overrides"),
}
COMPILE_FLAGS = (
    "-std=legacy", "-fdefault-real-8", "-ffree-line-length-none", "-O0",
    "-fcheck=all", "-ffpe-trap=invalid,zero,overflow",
)
COVERAGE_FLAGS = ("-std=legacy", "-fdefault-real-8", "-ffree-line-length-none", "-O0")
GCOV = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
ALIASES = {
    "HERBIVORES": "OK_HERBIVORES", "FIRE_DISABLE": "DISABLE_FIRE", "TF_DOC": "OK_TF_DOC",
    "GLUC_USE_AGE_CLASS": "USE_AGE_CLASS", "GLUC_NAGEC_TREE": "NAGEC_TREE",
    "GLUC_NAGEC_HERB": "NAGEC_HERB", "GLUC_ALLOW_FORESTRY_HARVEST": "ALLOW_FORESTRY_HARVEST",
    "GLUC_SINGLE_AGE_CLASS": "SINGLEAGECLASS", "GLUC_USE_BOUND_SPA": "USE_BOUND_SPA",
    "LAI_MAP": "READ_LAI", "VEGET_YEAR": "VEGET_YEAR_ORIG", "CONDVEG_Z0": "Z0_SCAL",
    "ROUGHHEIGHT": "ROUGHHEIGHT_SCAL", "CONDVEG_ALBVIS": "ALBEDO_SCAL_IVIS",
    "CONDVEG_ALBNIR": "ALBEDO_SCAL_INIR", "CONDVEG_EMIS": "EMIS_SCAL",
    "PERCENT_RESIDUAL": "PRC_RESIDUAL",
}


def compose(path: Path) -> dict[str, dict[str, Any]]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    payload = TEMPLATE.read_bytes().replace(b"! <CONSTANTES_VAR_MODULE>", VAR_SOURCE.read_bytes())
    for name, span in spans.items():
        payload = payload.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    unresolved = re.findall(rb"! <[A-Z0-9_]+>", payload)
    if unresolved:
        raise RuntimeError(f"unresolved template markers: {unresolved}")
    path.write_bytes(payload)
    return {
        name: {"start_line": span.start_line, "end_line": span.end_line, "span_sha256": span.span_sha256}
        for name, span in spans.items()
    }


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def _typed(value: str, kind: str) -> Any:
    if kind == "logical":
        return value.strip().upper() == "T"
    if kind == "integer":
        return int(value)
    if kind == "real":
        return float(value.replace("D", "E").replace("d", "e"))
    if kind == "real_array":
        return [float(item.replace("D", "E").replace("d", "e")) for item in value.split(";")]
    return value.strip()


def _jax_result(procedure: str, mode: str, rows: list[dict[str, str]]):
    calls = [row for row in rows if not row["key"].startswith("__") and row["before"] != "derived"]
    defaults = {ALIASES.get(row["key"].upper(), row["key"]): _typed(row["before"], row["kind"]) for row in calls}
    overrides = {
        row["key"]: _typed(row["after"], row["kind"])
        for row in calls
        if row["after"] != row["before"]
    }
    if procedure == "activate_sub_models":
        return activate_sub_models(
            defaults, overrides, ok_stomate="stomate_off" not in mode,
            ok_dgvm="dgvm_on" in mode or "lpj_conflict" in mode,
        )
    if procedure == "veget_config":
        return veget_config(defaults, overrides)
    if procedure == "config_sechiba_parameters":
        return config_sechiba_parameters(defaults, overrides)
    return config_stomate_parameters(defaults, overrides, ch4_calcul="ch4_on" in mode)


def _compare_case(procedure: str, mode: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = _jax_result(procedure, mode, rows)
    calls = [row for row in rows if not row["key"].startswith("__") and row["before"] != "derived"]
    comparisons: list[dict[str, Any]] = [{
        "name": f"{procedure}.{mode}.read_order",
        "passed": tuple(row["key"].upper() for row in calls) == result.read_order,
        "comparison": "exact",
        "actual": [row["key"].upper() for row in calls],
        "expected": list(result.read_order),
    }]
    for row in calls:
        key = row["key"].upper()
        expected = result.values[ALIASES.get(key, key)]
        actual = _typed(row["after"], row["kind"])
        name = f"{procedure}.{mode}.{row['sequence']}.{key}"
        if row["kind"] in {"real", "real_array"}:
            comparisons.append(float_comparison(name, np.atleast_1d(actual), np.atleast_1d(expected), rtol=1e-12, atol=1e-14))
        else:
            comparisons.append({"name": name, "passed": actual == expected, "comparison": "exact", "actual": actual, "expected": expected})
    derived = next((row for row in rows if row["key"] == "SNEIGE"), None)
    if derived is not None:
        comparisons.append(float_comparison(
            f"{procedure}.{mode}.SNEIGE", np.asarray([_typed(derived["after"], "real")]),
            np.asarray([result.values["SNEIGE"]]), rtol=1e-12, atol=1e-14,
        ))
    warning = next(row for row in rows if row["key"] == "__WARNING_COUNT__")
    comparisons.append({
        "name": f"{procedure}.{mode}.warnings", "comparison": "exact",
        "actual": int(warning["after"]), "expected": len(result.warnings),
        "passed": int(warning["after"]) == len(result.warnings),
    })
    return comparisons


def _run(executable: Path, build: Path, procedure: str, mode: str, compiler: Path, name: str) -> list[dict[str, str]]:
    output = build / f"{name}.csv"
    subprocess.run(
        [str(executable), str(output), procedure, mode], cwd=build,
        env=compiler_environment(compiler), check=True, capture_output=True, text=True,
    )
    return _read(output)


def run_oracle(output_dir: Path = OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="orchidee_constantes_owners_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        metadata = compile_fortran(source, executable, compiler, base_flags=COMPILE_FLAGS)
        for procedure, modes in CASES.items():
            for mode in modes:
                rows = _run(executable, build, procedure, mode, compiler, f"{procedure}_{mode}")
                comparisons.extend(_compare_case(procedure, mode, rows))
                shutil.copyfile(build / f"{procedure}_{mode}.csv", output_dir / f"{procedure}_{mode}.csv")
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "cases": CASES}, indent=2) + "\n", encoding="ascii")
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "constantes_var_sha256": hashlib.sha256(VAR_SOURCE.read_bytes()).hexdigest(),
        "span_sha256": hashes, "compiler": metadata,
        "tolerance": {"float_rtol": 1e-12, "float_atol": 1e-14, "discrete": "exact"},
    })


def _owners() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = json.loads(CONTRACT_CLASSES.read_text(encoding="utf-8"))
    owners = {row["base_region_id"]: row for row in contract["owner_regions"] if row["owner_region_id"] in OWNER_IDS}
    arms = [row for row in contract["arm_classifications"] if row.get("base_region_id") in owners]
    return owners, arms


def run_coverage(output_dir: Path = OUTPUT_DIR, compiler: Path = DEFAULT_COMPILER, gcov: Path = GCOV) -> dict[str, Any]:
    comparison = run_oracle(output_dir, compiler)
    if comparison["status"] != "passed":
        raise RuntimeError("constantes numerical comparison did not pass")
    owners, arms = _owners()
    with tempfile.TemporaryDirectory(prefix="orchidee_constantes_gcov_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        spans = compose(source)
        compile_fortran(source, executable, compiler, base_flags=COVERAGE_FLAGS, extra_flags=("--coverage",))
        for procedure, modes in CASES.items():
            for index, mode in enumerate(modes):
                _run(executable, build, procedure, mode, compiler, f"coverage_{procedure}_{index}")
        notes = next(build.glob("*.gcno"))
        subprocess.run([str(gcov), "-b", "-c", notes.name], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
        branches = parse_gcov(build / f"{source.name}.gcov")
        arm_records = []
        for arm in arms:
            procedure = owners[arm["base_region_id"]]["fortran_procedure"]
            span = extract_procedure_bytes(SOURCE, procedure)
            generated_start = locate_generated_span(source.read_bytes(), span.span_bytes)
            generated_line = generated_start + int(arm["line"]) - span.start_line
            rows = branches.get(generated_line, [])
            branch = select_arm_branch(str(arm["arm_id"]), rows)
            arm_records.append({
                "arm_id": arm["arm_id"], "base_region_id": arm["base_region_id"], "procedure": procedure,
                "original_line": arm["line"], "generated_line": generated_line,
                "gcov_branch_index": -1 if branch is None else branch["branch_index"],
                "gcov_taken": 0 if branch is None else branch["taken"], "raw_gcov_branches": rows,
                "procedure_span_sha256": spans[procedure]["span_sha256"], "passed": bool(branch and branch["taken"] > 0),
            })
    covered = {row["arm_id"] for row in arm_records if row["passed"]}
    required = {row["arm_id"] for row in arms}
    coverage = {
        "schema_version": 1, "family": FAMILY, "branch_complete": covered == required,
        "required_arm_count": len(required), "covered_arm_count": len(covered),
        "missing_arm_ids": sorted(required - covered), "arms": arm_records,
        "compiler": str(compiler), "gcov": str(gcov),
    }
    (output_dir / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arm in arms:
        by_region[arm["base_region_id"]].append(arm)
    records = []
    for region_id, owner in sorted(owners.items()):
        ids = {row["arm_id"] for row in by_region[region_id]}
        records.append({
            "owner_region_id": owner["owner_region_id"], "base_region_id": region_id,
            "fortran_procedure": owner["fortran_procedure"], "jax_owners": owner["jax_owners"],
            "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json",
            "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
            "required_arm_ids": sorted(ids), "gcov_arm_ids": sorted(ids & covered),
            "covered_arm_ids": sorted(ids & covered), "missing_arm_ids": sorted(ids - covered),
            "passed": ids <= covered,
        })
    evidence = {"schema_version": 1, "family": FAMILY, "complete": all(row["passed"] for row in records), "records": records}
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    if not coverage["branch_complete"]:
        raise RuntimeError(f"constantes owner coverage incomplete: {coverage['missing_arm_ids']}")
    return {"branch_coverage": coverage, "owner_evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args(argv)
    result = run_coverage(args.output_dir.resolve(), args.compiler.resolve())
    print(json.dumps({"status": "passed", "covered_arms": result["branch_coverage"]["covered_arm_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
