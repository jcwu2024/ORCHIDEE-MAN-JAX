"""Original-fragment Oracle for the 29-arm ``sechiba_main`` fixed tail owner."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.main import (  # noqa: E402
    sechiba_main_crop_rotation_transition,
    sechiba_main_post_output_transition,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_gcov import (  # noqa: E402
    locate_generated_span,
    parse_gcov,
    select_arm_branch,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    exact_comparison,
    float_comparison,
    write_point_comparisons,
)


FAMILY = "sechiba_batch_c"
OWNER_ID = "pft14-owner-contract-beb73c2fad63"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
SECHIBA = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
HYDROL = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
THERMOSOIL = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
SLOWPROC = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_stage4_batch_c_main_fixed_tail.f90.template"
RTOL = 1.0e-12
ATOL = 1.0e-14

ROTATION_LINES = (1303, 1410)
CONTROL_LINES = (1760, 1785)
GCOV_ARMS = tuple(
    f"fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:{line}:{kind}:{arm}"
    for line, kind, arm in (
        (1312, "if", "false"), (1312, "if", "true"),
        (1328, "if", "false"), (1328, "if", "true"),
        (1334, "where", "false"), (1334, "where", "true"),
        (1352, "if", "false"),
        (1371, "if", "false"), (1371, "if", "true"),
        (1373, "if", "false"), (1373, "if", "true"),
        (1768, "if", "false"), (1768, "if", "true"),
        (1769, "if", "false"), (1769, "if", "true"),
        (1780, "if", "false"), (1780, "if", "true"),
    )
)
FATAL_ARM = "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:1352:if:true"
NEW_ARMS = tuple(sorted((*GCOV_ARMS, FATAL_ARM)))

CASE_MATRIX = (
    {
        "case_id": 1,
        "kind": "rotation",
        "ok_rotate": True,
        "f_rot_sech": [True, False],
        "rot_cmd": [[501413, 0], [0, 0]],
        "expected": "valid integrated rotation and one inactive point",
    },
    {
        "case_id": 2,
        "kind": "rotation",
        "ok_rotate": False,
        "f_rot_sech": [True, False],
        "expected": "fixed paper no-rotation dispatch",
    },
    {
        "case_id": 11,
        "kind": "post_output",
        "done_stomate_lcchange": False,
        "dyn_peat": False,
        "use_age_class": False,
        "ldrestart_write": False,
    },
    {
        "case_id": 12,
        "kind": "post_output",
        "done_stomate_lcchange": True,
        "dyn_peat": False,
        "use_age_class": False,
        "ldrestart_write": True,
    },
    {
        "case_id": 13,
        "kind": "post_output",
        "done_stomate_lcchange": True,
        "dyn_peat": False,
        "use_age_class": True,
        "ldrestart_write": False,
    },
    {
        "case_id": 14,
        "kind": "post_output",
        "done_stomate_lcchange": True,
        "dyn_peat": True,
        "use_age_class": False,
        "ldrestart_write": False,
    },
    {
        "case_id": 99,
        "kind": "fatal",
        "ok_rotate": True,
        "totfrac_nobio": 0.1,
        "expected_stop": "sechiba_rotation: sum of soiltile not equal to 1",
    },
)


def _source_fragment(path: Path, start: int, end: int) -> bytes:
    return b"".join(path.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compose(path: Path) -> dict[str, object]:
    rotation = _source_fragment(SECHIBA, *ROTATION_LINES)
    controls = _source_fragment(SECHIBA, *CONTROL_LINES)
    procedures = (
        (SECHIBA, "sechiba_end"),
        (SECHIBA, "sechiba_get_cmd"),
        (HYDROL, "hydrol_rotation_update"),
        (THERMOSOIL, "thermosoil_rotation_update"),
        (SLOWPROC, "slowproc_veget"),
        (SLOWPROC, "slowproc_checkveget"),
        (SLOWPROC, "slowproc_change_frac"),
    )
    spans = {(source, name): extract_procedure_bytes(source, name) for source, name in procedures}
    unit = TEMPLATE.read_bytes()
    unit = unit.replace(b"! <ROTATION_FRAGMENT>", rotation.rstrip())
    unit = unit.replace(b"! <CONTROL_FRAGMENT>", controls.rstrip())
    unit = unit.replace(
        b"! <EXACT_PROCEDURES>",
        b"\n\n".join(spans[item].span_bytes for item in procedures),
    )
    path.write_bytes(unit)
    return {
        "rotation_fragment": {
            "source": SECHIBA.relative_to(ROOT).as_posix(),
            "lines": list(ROTATION_LINES),
            "sha256": _sha256(rotation),
        },
        "control_fragment": {
            "source": SECHIBA.relative_to(ROOT).as_posix(),
            "lines": list(CONTROL_LINES),
            "sha256": _sha256(controls),
        },
        "procedures": {
            name: {
                "source": source.relative_to(ROOT).as_posix(),
                "lines": [spans[(source, name)].start_line, spans[(source, name)].end_line],
                "sha256": spans[(source, name)].span_sha256,
            }
            for source, name in procedures
        },
    }


def _rotation_seed() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    npts, nvm, nstm = 2, 14, 3
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[:, 0] = 0.2
    veget_max[0, 12:14] = [0.3, 0.3]
    veget_max[1, 13] = 0.6
    lai = np.empty((npts, nvm), dtype=np.float64)
    co2_flux = np.empty((npts, nvm), dtype=np.float64)
    temp_sol_new_pft = np.empty((npts, nvm), dtype=np.float64)
    soilcap_pft = np.empty((npts, nvm), dtype=np.float64)
    soilflx_pft = np.empty((npts, nvm), dtype=np.float64)
    ptn = np.empty((npts, 2, nvm), dtype=np.float64)
    cgrnd = np.empty((npts, 1, nvm), dtype=np.float64)
    dgrnd = np.empty((npts, 1, nvm), dtype=np.float64)
    for point in range(npts):
        for pft in range(nvm):
            lai[point, pft] = 0.25 + 0.01 * (pft + 1) + 0.02 * (point + 1)
            co2_flux[point, pft] = (10 * (point + 1) + pft + 1) / 8.0
            temp_sol_new_pft[point, pft] = 270.0 + 2 * (point + 1) + (pft + 1) / 4.0
            soilcap_pft[point, pft] = 10.0 + (point + pft + 2) / 8.0
            soilflx_pft[point, pft] = 20.0 + (2 * (point + 1) + pft + 1) / 16.0
            for layer in range(2):
                ptn[point, layer, pft] = (
                    260.0 + 2 * (point + 1) + 3 * (layer + 1) + (pft + 1) / 8.0
                )
            cgrnd[point, 0, pft] = 1.0 + (point + pft + 2) / 16.0
            dgrnd[point, 0, pft] = 2.0 + (2 * (point + 1) + pft + 1) / 16.0
    extinction = np.full(nvm, 0.5)
    veget = veget_max * (1.0 - np.exp(-lai * extinction[None, :]))
    veget[:, 0] = veget_max[:, 0]
    soiltile = np.asarray([[0.25, 0.375, 0.375], [0.25, 0.0, 0.75]])
    qsintveg = np.zeros((npts, nvm), dtype=np.float64)
    qsintveg[0, [0, 12, 13]] = [0.001, 0.013, 0.014]
    qsintveg[1, [0, 13]] = [0.001, 0.014]
    mc = np.empty((npts, 2, nstm), dtype=np.float64)
    for point in range(npts):
        for layer in range(2):
            for tile in range(nstm):
                mc[point, layer, tile] = (
                    1.0 + ((point + 1) + 2 * (layer + 1) + 3 * (tile + 1)) / 16.0
                )
    water = np.empty((npts, nstm), dtype=np.float64)
    for point in range(npts):
        for tile in range(nstm):
            water[point, tile] = ((point + 1) + 2 * (tile + 1)) / 32.0
    state = {
        "veget_max": veget_max,
        "veget": veget,
        "lai": lai,
        "totfrac_nobio": np.full(npts, 0.2),
        "soiltile": soiltile,
        "f_rot_sech": np.asarray([True, False]),
        "rot_cmd": np.asarray([[501413, 0], [0, 0]], dtype=np.int32),
        "qsintveg": qsintveg,
        "mc": mc,
        "water2infilt": water,
        "tmc": np.ones((npts, nstm)),
        "humtot": np.ones(npts),
        "resdist": soiltile.copy(),
        "ptn": ptn,
        "cgrnd": cgrnd,
        "dgrnd": dgrnd,
        "temp_sol_new_pft": temp_sol_new_pft,
        "soilcap_pft": soilcap_pft,
        "soilflx_pft": soilflx_pft,
    }
    inputs = {
        "ok_laidev": np.asarray([False] + [True] * 13),
        "pref_soil_veg": np.asarray([1] * 12 + [2, 3]),
        "ext_coeff": extinction,
        "dz": np.asarray([0.5, 1.0]),
    }
    extras = {
        "co2_flux": co2_flux,
        "temp_sol_new": np.asarray([280.0, 281.0]),
    }
    return state | extras, inputs


def _rotation_jax_cases() -> dict[int, dict[str, np.ndarray]]:
    state, inputs = _rotation_seed()
    rotation_state = {name: value for name, value in state.items() if name not in {"co2_flux", "temp_sol_new"}}
    result = sechiba_main_crop_rotation_transition(state=rotation_state, **inputs).state
    netco2 = np.zeros(2)
    for pft in range(1, 14):
        netco2 += state["co2_flux"][:, pft] * state["veget_max"][:, pft]

    names = (
        "veget_max", "veget", "qsintveg", "temp_sol_new_pft", "soilcap_pft",
        "soilflx_pft", "soiltile", "water2infilt", "tmc", "resdist", "mc",
        "ptn", "cgrnd", "dgrnd", "f_rot_sech", "humtot", "rot_cmd",
    )
    case1 = {name: np.asarray(result[name]).ravel(order="C") for name in names}
    case1["netco2flux"] = netco2
    case1["temp_sol"] = state["temp_sol_new"]
    case1["temp_sol_pft"] = np.asarray(result["temp_sol_new_pft"]).ravel(order="C")

    case2 = {name: np.asarray(rotation_state[name]).ravel(order="C") for name in names}
    case2["netco2flux"] = netco2
    case2["temp_sol"] = state["temp_sol_new"]
    case2["temp_sol_pft"] = np.asarray(rotation_state["temp_sol_new_pft"]).ravel(order="C")
    return {1: case1, 2: case2}


def _control_seed() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    npts, nvm = 2, 14
    veget_max = np.zeros((npts, nvm))
    veget_max[:, 0] = 0.2
    veget_max[:, 13] = 0.6
    lai = np.empty((npts, nvm))
    for point in range(npts):
        for pft in range(nvm):
            lai[point, pft] = 0.25 + 0.01 * (pft + 1) + 0.02 * (point + 1)
    extinction = np.full(nvm, 0.5)
    veget = veget_max * (1.0 - np.exp(-lai * extinction[None, :]))
    veget[:, 0] = veget_max[:, 0]
    state = {
        "veget_max": veget_max,
        "veget": veget,
        "frac_nobio": np.asarray([[0.2, 0.0], [0.2, 0.0]]),
        "totfrac_nobio": np.full(npts, 0.2),
        "tot_bare_soil": veget_max[:, 0] + np.sum(veget_max[:, 1:] - veget[:, 1:], axis=1),
        "soiltile": np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "lai": lai,
        "biomass": np.full((npts, nvm, 1, 1), 3.0),
        "litter_above": np.full((npts, 1, nvm, 1), 4.0),
        "litter_below": np.full((npts, 1, nvm, 1, 1), 5.0),
        "carbon_32l": np.full((npts, 1, nvm, 1), 6.0),
        "DOC": np.full((npts, nvm, 1, 1, 1, 1), 7.0),
        "netco2flux": np.ones(npts),
    }
    new = np.zeros((npts, nvm))
    new[:, 0] = 0.3
    new[:, 13] = 0.5
    lcc_inputs = {
        "agri_peat": False,
        "veget_max_new": new,
        "frac_nobio_new": np.asarray([[0.2, 0.0], [0.2, 0.0]]),
        "pref_soil_veg": np.ones(nvm, dtype=np.int32),
        "ext_coeff_vegetfrac": extinction,
        "nstm": 3,
    }
    return state, lcc_inputs


def _control_jax_cases() -> dict[int, dict[str, np.ndarray]]:
    cases: dict[int, dict[str, np.ndarray]] = {}
    fields = (
        "veget_max", "veget", "frac_nobio", "totfrac_nobio", "tot_bare_soil",
        "soiltile", "biomass", "litter_above", "litter_below", "carbon_32l",
        "DOC", "netco2flux",
    )
    for spec in CASE_MATRIX:
        if spec["kind"] != "post_output":
            continue
        state, lcc_inputs = _control_seed()
        finalize_calls = 0

        def finalize_owner(current):
            nonlocal finalize_calls
            finalize_calls += 1
            updated = dict(current)
            updated["netco2flux"] = np.asarray(current["netco2flux"]) + 10.0
            return SimpleNamespace(state=updated)

        result = sechiba_main_post_output_transition(
            state=state,
            done_stomate_lcchange=bool(spec["done_stomate_lcchange"]),
            dyn_peat=bool(spec["dyn_peat"]),
            use_age_class=bool(spec["use_age_class"]),
            ldrestart_write=bool(spec["ldrestart_write"]),
            lcc_inputs=lcc_inputs,
            finalize_owner=finalize_owner,
        )
        values = {name: np.asarray(result.state[name]).ravel(order="C") for name in fields}
        values["done_stomate_lcchange"] = np.asarray([0.0])
        values["finalize_calls"] = np.asarray([float(finalize_calls)])
        cases[int(spec["case_id"])] = values
    return cases


def _read_outputs(path: Path) -> dict[int, dict[str, np.ndarray]]:
    records: dict[int, dict[str, list[float]]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            case = int(row["case_id"])
            records.setdefault(case, {}).setdefault(row["field"], []).append(float(row["value"]))
    return {
        case: {field: np.asarray(values) for field, values in fields.items()}
        for case, fields in records.items()
    }


def _arm_line(arm_id: str) -> int:
    return int(arm_id.rsplit(":", 2)[0].rsplit(":", 1)[1])


def run(output: Path = OUTPUT, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_sechiba_main_tail_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        source_metadata = compose(source)
        shutil.copyfile(source, output / "sechiba_main_fixed_tail_oracle.f90")
        compile_metadata = compile_fortran(
            source,
            executable,
            compiler,
            extra_flags=("--coverage",),
        )
        normal = subprocess.run(
            [str(executable), "0"],
            cwd=build,
            env=compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=True,
        )
        fatal = subprocess.run(
            [str(executable), "1"],
            cwd=build,
            env=compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=False,
        )
        shutil.copyfile(build / "fortran_outputs.csv", output / "sechiba_main_fixed_tail_fortran.csv")
        gcov_exe = compiler.with_name("gcov.exe")
        gcov_run = subprocess.run(
            [str(gcov_exe), "-b", "-c", "oracle.gcno"],
            cwd=build,
            env=compiler_environment(compiler),
            text=True,
            capture_output=True,
            check=True,
        )
        gcov_path = build / "oracle.f90.gcov"
        shutil.copyfile(gcov_path, output / "sechiba_main_fixed_tail_oracle.f90.gcov")
        gcov = parse_gcov(gcov_path)
        rotation_start = locate_generated_span(source.read_bytes(), _source_fragment(SECHIBA, *ROTATION_LINES))
        control_start = locate_generated_span(source.read_bytes(), _source_fragment(SECHIBA, *CONTROL_LINES))

    fatal_text = fatal.stdout + fatal.stderr
    fatal_passed = CASE_MATRIX[-1]["expected_stop"] in fatal_text
    (output / "sechiba_main_fixed_tail_fatal.txt").write_text(fatal_text, encoding="ascii", errors="replace")
    fatal_witness = {
        "arm_id": FATAL_ARM,
        "case_id": 99,
        "returncode": fatal.returncode,
        "expected_stop": CASE_MATRIX[-1]["expected_stop"],
        "matched": fatal_passed,
    }
    (output / "sechiba_main_fixed_tail_fatal_witness.json").write_text(
        json.dumps(fatal_witness, indent=2) + "\n", encoding="ascii"
    )

    mappings = []
    for arm_id in (*GCOV_ARMS, FATAL_ARM):
        line = _arm_line(arm_id)
        if ROTATION_LINES[0] <= line <= ROTATION_LINES[1]:
            generated_line = rotation_start + line - ROTATION_LINES[0]
        else:
            generated_line = control_start + line - CONTROL_LINES[0]
        selected = select_arm_branch(arm_id, gcov.get(generated_line, []))
        mappings.append({
            "arm_id": arm_id,
            "original_line": line,
            "generated_line": generated_line,
            "selected_branch": selected,
            "support": "fatal_witness" if arm_id == FATAL_ARM else "gcov",
            "passed": bool(selected and selected["taken"] > 0) and (fatal_passed if arm_id == FATAL_ARM else True),
        })

    fortran = _read_outputs(output / "sechiba_main_fixed_tail_fortran.csv")
    jax = {**_rotation_jax_cases(), **_control_jax_cases()}
    comparisons = []
    fortran_points: dict[str, np.ndarray] = {}
    jax_points: dict[str, np.ndarray] = {}
    exact_fields = {
        "f_rot_sech", "rot_cmd", "done_stomate_lcchange", "finalize_calls",
    }
    for case_id in sorted(fortran):
        if set(fortran[case_id]) != set(jax[case_id]):
            raise RuntimeError(
                f"case {case_id} field mismatch: Fortran={sorted(fortran[case_id])}, JAX={sorted(jax[case_id])}"
            )
        for field in sorted(fortran[case_id]):
            name = f"case{case_id}.{field}"
            expected = fortran[case_id][field]
            actual = jax[case_id][field]
            comparison = (
                exact_comparison(name, expected, actual)
                if field in exact_fields
                else float_comparison(name, expected, actual, rtol=RTOL, atol=ATOL)
            )
            comparisons.append(comparison)
            fortran_points[name] = expected
            jax_points[name] = actual
    write_point_comparisons(
        output / "sechiba_main_fixed_tail_point_comparisons.csv",
        fortran_points,
        jax_points,
        rtol=RTOL,
        atol=ATOL,
    )
    comparison_passed = bool(comparisons) and all(item["passed"] for item in comparisons)
    coverage_passed = all(item["passed"] for item in mappings)
    report = {
        "schema_version": 2,
        "family": FAMILY,
        "owner_region_id": OWNER_ID,
        "status": "passed" if comparison_passed and coverage_passed else "failed",
        "case_matrix": list(CASE_MATRIX),
        "comparisons": comparisons,
        "comparison_passed": comparison_passed,
        "required_new_arm_ids": list(NEW_ARMS),
        "gcov_arm_ids": list(GCOV_ARMS),
        "fatal_witness_arm_ids": [FATAL_ARM],
        "passed_arm_ids": sorted(item["arm_id"] for item in mappings if item["passed"]),
        "gcov_mapping": mappings,
        "fatal_witness": fatal_witness,
        "source_fragments": source_metadata,
        "harness_sha256": _sha256((output / "sechiba_main_fixed_tail_oracle.f90").read_bytes()),
        "compiler": compile_metadata,
        "gcov_stdout": gcov_run.stdout,
        "normal_stdout": normal.stdout,
        "normal_stderr": normal.stderr,
        "tolerance_policy": {
            "rtol": RTOL,
            "atol": ATOL,
            "source": "existing hydrol_state_updates owner Oracle policy",
            "changed": False,
        },
    }
    (output / "sechiba_main_fixed_tail_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii"
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "status": result["status"],
        "comparisons": len(result["comparisons"]),
        "arms": len(result["passed_arm_ids"]),
    }, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
