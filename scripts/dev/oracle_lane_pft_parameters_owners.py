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
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.parameters import (  # noqa: E402
    apply_configured_pft_constraints,
    build_age_class_layout,
    crop_rotation_sowing_keys,
    derive_pft_physiology_labels,
    resolve_age_class_config,
    resolve_pft_parameter_main,
    resolve_pft_parameter_section_switches,
    select_humcste_reference,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_result,
)


FAMILY = "pft_parameters_owners"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/pft_parameters.f90"
PFT_VAR_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/pft_parameters_var.f90"
MTC_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_parameters/constantes_mtc.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_pft_parameters_owners.f90.template"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
PROCEDURES = {
    "pft_parameters_main",
    "pft_parameters_init",
    "pft_parameters_alloc",
    "config_pft_parameters",
    "config_sechiba_pft_parameters",
    "config_stomate_pft_parameters",
}
OWNER_PROCEDURES = PROCEDURES - {"pft_parameters_alloc"}
PFT_COMPILE_FLAGS = (
    "-std=legacy",
    "-fdefault-real-8",
    "-ffree-line-length-none",
    "-O0",
    "-fcheck=all",
    "-ffpe-trap=invalid,zero,overflow",
    "-Wall",
    "-Wextra",
)
PFT_COVERAGE_FLAGS = (
    "-std=legacy",
    "-fdefault-real-8",
    "-ffree-line-length-none",
    "-O0",
)

VALID_MODES = (
    "main_default",
    "main_custom",
    "init_z2",
    "init_z4",
    "init_zother",
    "init_sections_off",
    "init_offline",
    "config_base",
    "config_nstm_low",
    "config_nstm_bad",
    "config_crop2",
    "config_crop3",
    "config_age_warning_bound_managed",
    "config_age_multi_bound_off_managed",
    "sechiba_bvoc_on",
    "sechiba_bvoc_off",
    "stomate_age_off",
    "stomate_tree_multi",
    "stomate_herb_multi",
)
ERROR_MODES = {
    "main_undef": "array PFT_TO_MTC is empty",
    "main_outside": "metaclass chosen does not exist",
    "main_first_not_bare": "first pft has to be the bare soil",
    "main_duplicate_bare": "only pft_to_mtc(1) has to be the bare soil",
    "config_age_no_managed": "Age classes are used but none",
    "stomate_missing_group": "Could not find a start index",
    "stomate_tree_mismatch": "real number of age class for trees",
}


def _strip_fortran_comment(line: str) -> str:
    quote: str | None = None
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "!" and quote is None:
            return line[:index]
    return line


def _declaration_statements(source: str) -> list[str]:
    statements: list[str] = []
    pending: list[str] = []
    for line in source.splitlines():
        code = _strip_fortran_comment(line).rstrip()
        if not pending and "::" not in code:
            continue
        if not pending and "::" in code:
            pending = [line]
        elif pending:
            pending.append(line)
        if pending and not code.endswith("&"):
            statements.append("\n".join(pending))
            pending = []
    if pending:
        raise RuntimeError("unterminated constantes_mtc declaration")
    return statements


def _declared_name(statement: str) -> str | None:
    code = " ".join(_strip_fortran_comment(line) for line in statement.splitlines())
    match = re.search(r"::\s*([A-Za-z][A-Za-z0-9_]*)", code)
    return None if match is None else match.group(1).lower()


def _typed_mtc_name(statement: str) -> str:
    literals = re.findall(r"'([^']*)'", statement)
    if len(literals) != 18:
        raise RuntimeError(f"expected 18 MTC_name literals, found {len(literals)}")
    values = [value[:34].ljust(34).replace("'", "''") for value in literals]
    rows = ["  character(len=34), parameter :: MTC_name(nvmc) = [character(len=34) :: &"]
    for index, value in enumerate(values):
        suffix = ", &" if index < len(values) - 1 else "]"
        rows.append(f"    '{value}'{suffix}")
    return "\n".join(rows)


def _referenced_mtc_declarations(procedure_text: str) -> bytes:
    code = "\n".join(_strip_fortran_comment(line) for line in procedure_text.splitlines())
    required = {word.lower() for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", code)}
    statements = _declaration_statements(MTC_SOURCE.read_text(encoding="utf-8"))
    by_name = {
        name: statement
        for statement in statements
        if (name := _declared_name(statement)) is not None
    }
    selected: set[str] = set()
    frontier = required & by_name.keys()
    while frontier:
        selected.update(frontier)
        dependency_words: set[str] = set()
        for name in frontier:
            dependency_words.update(
                word.lower()
                for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", by_name[name])
            )
        frontier = (dependency_words & by_name.keys()) - selected
    missing = sorted(
        name
        for name in required
        if name.endswith("_mtc") and name != "pft_to_mtc" and name not in by_name
    )
    if missing:
        raise RuntimeError(f"constantes_mtc declarations not found: {missing}")
    rendered: list[str] = []
    for statement in statements:
        name = _declared_name(statement)
        if name not in selected:
            continue
        rendered.append(_typed_mtc_name(statement) if name == "mtc_name" else statement)
    return ("\n\n".join(rendered) + "\n").encode("utf-8")


def _mtc_statement(name: str) -> str:
    for statement in _declaration_statements(MTC_SOURCE.read_text(encoding="utf-8")):
        if _declared_name(statement) == name.lower():
            return statement
    raise RuntimeError(f"constantes_mtc declaration not found: {name}")


def _mtc_rhs(name: str) -> str:
    statement = "\n".join(
        _strip_fortran_comment(line) for line in _mtc_statement(name).splitlines()
    )
    return statement.split("=", 1)[1]


def _numeric_mtc_array(name: str) -> np.ndarray:
    rhs = _mtc_rhs(name)
    body = rhs.split("(/", 1)[1].split("/)", 1)[0]
    body = re.sub(r"_[A-Za-z][A-Za-z0-9_]*", "", body)
    values = [float(item.strip().replace("D", "E").replace("d", "e")) for item in body.replace("&", "").split(",") if item.strip()]
    return np.asarray(values, dtype=np.float64)


def _integer_mtc_array(name: str) -> np.ndarray:
    return _numeric_mtc_array(name).astype(np.int32)


def _logical_mtc_array(name: str) -> np.ndarray:
    rhs = _mtc_rhs(name).upper()
    return np.asarray(re.findall(r"\.(TRUE|FALSE)\.", rhs), dtype=str) == "TRUE"


def _character_mtc_array(name: str) -> tuple[str, ...]:
    rhs = _mtc_rhs(name)
    return tuple(value.strip() for value in re.findall(r"'([^']*)'", rhs))


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    procedure_text = "\n".join(
        span.span_bytes.decode("utf-8", errors="strict") for span in spans.values()
    )
    source = TEMPLATE.read_bytes()
    source = source.replace(b"! <PFT_PARAMETERS_VAR_MODULE>", PFT_VAR_SOURCE.read_bytes())
    source = source.replace(
        b"! <CONSTANTES_MTC_REFERENCED_DECLARATIONS>",
        _referenced_mtc_declarations(procedure_text),
    )
    for name, span in spans.items():
        source = source.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    unresolved = re.findall(rb"! <[A-Z0-9_]+>", source)
    if unresolved:
        raise RuntimeError(f"unresolved template markers: {unresolved}")
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read_output(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _run_mode(executable: Path, build: Path, mode: str, compiler: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), str(build / f"{mode}.csv"), mode],
        cwd=build,
        env=compiler_environment(compiler),
        capture_output=True,
        text=True,
        check=False,
    )


def _exact(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return {
        "name": name,
        "passed": bool(actual.shape == expected.shape and np.array_equal(actual, expected)),
        "comparison": "exact",
        "actual": actual.tolist(),
        "expected": expected.tolist(),
    }


def _valid_case_comparisons(mode: str, output: dict[str, np.ndarray]) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    if mode == "main_default":
        expected = resolve_pft_parameter_main(first_call=True, nvm=18, nvmc=18)
        comparisons.extend([
            _exact(f"{mode}.mapping", output["mapping"], expected),
            _exact(f"{mode}.reentry", output["first"], np.asarray([0])),
            _exact(f"{mode}.writebacks", output["writeback_total"], np.asarray([0])),
        ])
    elif mode == "main_custom":
        expected = resolve_pft_parameter_main(
            first_call=True, nvm=2, nvmc=18, pft_to_mtc=[1, 14]
        )
        comparisons.append(_exact(f"{mode}.mapping", output["mapping"], expected))
        comparisons.append(_exact(f"{mode}.writebacks", output["writeback_total"], np.asarray([1])))
    if mode.startswith("init_"):
        mapping = np.arange(1, 19)
        depth = 4.0 if mode == "init_z4" else (3.0 if mode == "init_zother" else 2.0)
        expected_hum = select_humcste_reference(
            depth, mapping, _numeric_mtc_array("humcste_ref2m"),
            _numeric_mtc_array("humcste_ref4m"),
        )
        comparisons.append(float_comparison(
            f"{mode}.humcste", output["humcste"], expected_hum, rtol=1e-12, atol=1e-14
        ))
        labels = derive_pft_physiology_labels(
            _integer_mtc_array("leaf_tab_mtc"), _character_mtc_array("pheno_model_mtc")
        )
        comparisons.extend([
            _exact(f"{mode}.is_tree", output["is_tree"], labels.is_tree),
            _exact(f"{mode}.is_deciduous", output["is_deciduous"], labels.is_deciduous),
            _exact(f"{mode}.is_evergreen", output["is_evergreen"], labels.is_evergreen),
            _exact(f"{mode}.is_needleleaf", output["is_needleleaf"], labels.is_needleleaf),
        ])
        switches = resolve_pft_parameter_section_switches(
            ok_sechiba="sections_off" not in mode,
            hydrol_cwrr="offline" in mode,
            offline_mode="offline" in mode,
            ok_stomate="sections_off" not in mode,
            ok_bvoc="sections_off" not in mode,
        )
        if switches.zero_offline_cwrr_throughfall:
            comparisons.append(_exact(
                f"{mode}.throughfall_zero", output["throughfall"], np.zeros(18)
            ))
    if mode.startswith("config_"):
        managed = np.zeros(18, dtype=bool)
        if "managed" in mode:
            managed[9] = True
        nstm = 4 if "nstm_low" in mode else (21 if "nstm_bad" in mode else 6)
        initial_soil = _integer_mtc_array("pref_soil_veg_mtc")
        if "nstm_low" in mode:
            initial_soil[13] = 6
        expected_natural, expected_soil = apply_configured_pft_constraints(
            _logical_mtc_array("natural_mtc"),
            managed,
            initial_soil,
            nstm=nstm,
            use_age_class="age_" in mode,
        )
        comparisons.extend([
            _exact(f"{mode}.natural", output["natural"], expected_natural),
            _exact(f"{mode}.pref_soil", output["pref_soil"], expected_soil),
        ])
        labels = derive_pft_physiology_labels(
            _integer_mtc_array("leaf_tab_mtc"), _character_mtc_array("pheno_model_mtc")
        )
        comparisons.extend([
            _exact(f"{mode}.is_tree", output["is_tree"], labels.is_tree),
            _exact(f"{mode}.is_deciduous", output["is_deciduous"], labels.is_deciduous),
            _exact(f"{mode}.is_evergreen", output["is_evergreen"], labels.is_evergreen),
            _exact(f"{mode}.is_needleleaf", output["is_needleleaf"], labels.is_needleleaf),
        ])
        expected_keys = crop_rotation_sowing_keys(3 if "crop3" in mode else (2 if "crop2" in mode else 1))
        comparisons.extend([
            _exact(f"{mode}.iplt1", output["hit_iplt1"], np.asarray([int("SP_IPLT1" in expected_keys)])),
            _exact(f"{mode}.iplt2", output["hit_iplt2"], np.asarray([int("SP_IPLT2" in expected_keys)])),
        ])
        if "age_warning" in mode:
            config = resolve_age_class_config(18, use_age_class=True, use_bound_spa=False)
            comparisons.append(_exact(
                f"{mode}.age_bound_read",
                output["hit_age_bound"],
                np.asarray([int(config.read_age_class_bound)]),
            ))
        if "age_multi" in mode:
            config = resolve_age_class_config(
                18, use_age_class=True, nvmap=2, agec_group=[1] + [2] * 17,
                use_bound_spa=True,
            )
            comparisons.append(_exact(
                f"{mode}.age_bound_skip",
                output["hit_age_bound"],
                np.asarray([int(config.read_age_class_bound)]),
            ))
    if mode == "stomate_tree_multi":
        layout = build_age_class_layout(
            [1, 2, 2, 2, 2], nvmap=2, is_tree=[False, True, True, True, True],
            nagec_tree=4, nagec_herb=1,
        )
        comparisons.extend([
            _exact(f"{mode}.start", output["start_index"][:2], layout.start_index + 1),
            _exact(f"{mode}.count", output["nagec_pft"][:2], layout.nagec_pft),
        ])
    if mode == "stomate_herb_multi":
        layout = build_age_class_layout(
            [1, 2, 2], nvmap=2, is_tree=[False, False, False],
            nagec_tree=1, nagec_herb=2,
        )
        comparisons.extend([
            _exact(f"{mode}.start", output["start_index"][:2], layout.start_index + 1),
            _exact(f"{mode}.count", output["nagec_pft"][:2], layout.nagec_pft),
        ])
    if mode.startswith("sechiba_"):
        expected = int(mode == "sechiba_bvoc_on")
        comparisons.append(_exact(
            f"{mode}.bvoc_read", output["hit_iso_activity"], np.asarray([expected])
        ))
    return comparisons


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="orchidee_pft_parameters_owners_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        metadata = compile_fortran(
            source, executable, compiler, base_flags=PFT_COMPILE_FLAGS
        )
        for mode in VALID_MODES:
            completed = _run_mode(executable, build, mode, compiler)
            comparisons.append({
                "name": f"{mode}.execution",
                "passed": completed.returncode == 0,
                "comparison": "exact_exit_code",
                "actual": completed.returncode,
                "expected": 0,
            })
            if completed.returncode == 0:
                output = _read_output(build / f"{mode}.csv")
                comparisons.extend(_valid_case_comparisons(mode, output))
                shutil.copyfile(build / f"{mode}.csv", output_dir / f"{mode}.csv")
        for mode, message in ERROR_MODES.items():
            completed = _run_mode(executable, build, mode, compiler)
            combined = completed.stdout + completed.stderr
            comparisons.append({
                "name": f"{mode}.expected_error",
                "passed": message.lower() in combined.lower(),
                "comparison": "exact_diagnostic_boundary",
                "expected": message,
                "actual": combined.strip(),
            })
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {"schema_version": 1, "valid_modes": VALID_MODES, "error_modes": ERROR_MODES},
            indent=2,
        ) + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "pft_parameters_var_sha256": hashlib.sha256(PFT_VAR_SOURCE.read_bytes()).hexdigest(),
            "constantes_mtc_sha256": hashlib.sha256(MTC_SOURCE.read_bytes()).hexdigest(),
            "span_sha256": span_hashes,
            "compiler": metadata,
            "tolerance": {"float_rtol": 1e-12, "float_atol": 1e-14, "discrete": "exact"},
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pft_parameters owner micro-oracle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    args = parser.parse_args(argv)
    try:
        result = run_oracle(args.output_dir.resolve(), args.compiler.resolve())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
