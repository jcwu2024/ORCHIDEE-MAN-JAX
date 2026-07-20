"""Stage 4 Batch C owner oracle: ENERBIL fusion/initialization and hydro_subgrid.

The generated Fortran program uses byte-extracted owner procedures.  The only
stubs are ORCHIDEE restart/metadata transport calls; their deterministic
missing-restart behaviour is explicit in the harness.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from extract_fortran_micro_oracle import extract_procedure_bytes
from fortran_oracle_common import DEFAULT_COMPILER, ROOT, compile_fortran, compiler_environment, float_comparison, write_point_comparisons, write_result
from fortran_gcov import locate_generated_span, parse_gcov, parse_gcov_line_counts, select_arm_branch

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILY = "enerbil_hydro_subgrid_batch_c"
ENERBIL = ROOT / "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90"
HYDRO = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydro_subgrid.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_stage4_batch_c_enerbil_hydro.f90.template"


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(ENERBIL, name) for name in ("enerbil_initialize", "enerbil_fusion", "enerbil_t2mdiag")}
    hydro = extract_procedure_bytes(HYDRO, "hydro_subgrid_main")
    data = TEMPLATE.read_bytes().replace(b"! <ENERBIL>", b"\n\n".join(spans[n].span_bytes for n in spans)).replace(b"! <HYDRO>", hydro.span_bytes)
    path.write_bytes(data)
    return {**{name: item.span_sha256 for name, item in spans.items()}, "hydro_subgrid_main": hydro.span_sha256}


def read_values(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    for row in path.read_text(encoding="ascii").splitlines()[1:]:
        field, value = row.split(",")
        values.setdefault(field, []).append(float(re.sub(r"(?<=\\d)([+-]\\d{3})$", r"E\\1", value)))
    return {name: np.asarray(value) for name, value in values.items()}


def jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.enerbil import enerbil_cold_start_surface_state, enerbil_fusion_step
    from jax_orchidee.sechiba.hydrol_thermosoil_completion import hydro_subgrid_main
    cold = enerbil_cold_start_surface_state(qair=np.array([.003, .007]), nvm=14)
    fusion = enerbil_fusion_step(tot_melt=np.array([.01, .02]), soilcap=np.array([5e5, 4e5]), soilcap_pft=np.full((2, 14), 5e5), snowdz=np.array([[.01, .01, .01], [0., 0., 0.]]), temp_sol_new=np.array([275., 270.]), temp_sol_new_pft=np.full((2, 14), 276.), ok_laidev=np.array([False] * 13 + [True]), ok_explicitsnow=True)
    table = np.tile(np.linspace(.001, 1., 1000), (2, 1))
    table[1] = 0.
    hydro = hydro_subgrid_main(tab_fsat=table, tab_wtop=table, humtot=np.array([100., 0.]), profil_froz_hydro=np.vstack((np.full(9, .2), np.zeros(9))), tab_fwet=table, tab_wtop_wet=table, pd_top=1., ruu_ch=np.array([1., 1.]), ti_min=np.zeros(2), ti_max=np.array([1., -99.99]), pas=np.ones(2), dz=np.ones(9))
    return {
        "init_temp_sol": np.asarray(cold.temp_sol), "init_temp_sol_pft14": np.asarray(cold.temp_sol_pft)[:, 13], "init_temp_sol_new": np.asarray(cold.temp_sol_new), "init_qsurf": np.asarray(cold.qsurf), "init_evapot": np.asarray(cold.evapot), "init_evapot_corr": np.asarray(cold.evapot_corr), "init_tsol_rad": np.asarray(cold.tsol_rad), "init_vevapp": np.zeros(2), "init_fluxlat": np.zeros(2), "init_fluxsens": np.zeros(2), "init_temp_sol_pot": np.asarray(cold.temp_sol_pot), "init_q_sol_pot": np.asarray(cold.q_sol_pot),
        "fusion_temp_sol": np.asarray(fusion.temp_sol_new), "fusion_temp_sol_pft14": np.asarray(fusion.temp_sol_new_pft)[:, 13], "fusion": np.asarray(fusion.fusion),
        "fsat": hydro.fsat, "fwet": hydro.fwet, "fwt1": hydro.fwt1, "fwt2": hydro.fwt2, "fwt3": hydro.fwt3, "fwt4": hydro.fwt4,
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_batch_c_") as temporary:
        source = Path(temporary) / "oracle.f90"
        hashes = compose(source)
        exe = Path(temporary) / "oracle.exe"
        try:
            build = compile_fortran(source, exe, compiler)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr) from exc
        subprocess.run([str(exe), str(csv_path.resolve())], cwd=temporary, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
    fortran, jax = read_values(csv_path), jax_values()
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14)
    result = write_result(output_dir, FAMILY, [float_comparison(name, value, jax[name], rtol=1e-12, atol=1e-14) for name, value in fortran.items()], {"ledger_entries": ["enerbil.active.fusion", "enerbil.active.initialize", "hydrol.active.subgrid"], "span_sha256": hashes, "build": build})
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "cases": ["restart-missing cold state", "explicit-snow fusion writeback", "TOPMODEL active and empty-table state"]}, indent=2) + "\n", encoding="ascii")
    return result


def run_coverage(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    """Map every executable canonical arm to the extracted-owner gcov listing."""
    comparison = run_oracle(output_dir, compiler)
    if comparison["status"] != "passed":
        raise RuntimeError("numerical comparison failed")
    contracts = json.loads((ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json").read_text(encoding="utf-8"))
    proofs = json.loads((ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json").read_text(encoding="utf-8"))
    owner_ids = {"pft14-owner-contract-975df6e7ea38", "pft14-owner-contract-b80ae3c9991f", "pft14-owner-contract-c601713a214a"}
    owners = {x["base_region_id"]: x for x in contracts["owner_regions"] if x["owner_region_id"] in owner_ids}
    arms = [x for x in contracts["arm_classifications"] if x.get("base_region_id") in owners]
    proof_ids = {x["arm_id"] for x in proofs["records"] if x.get("passed")}
    with tempfile.TemporaryDirectory(prefix="orchidee_batch_c_gcov_") as temporary:
        build = Path(temporary); source = build / "oracle.f90"; compose(source)
        exe = build / "oracle.exe"; compile_fortran(source, exe, compiler, extra_flags=("--coverage",))
        subprocess.run([str(exe), str((build / "values.csv").resolve())], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
        notes = next(build.glob("*.gcno")); gcov = Path(r"C:\msys64\ucrt64\bin\gcov.exe")
        subprocess.run([str(gcov), "-b", "-c", notes.name], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
        branch_rows = parse_gcov(build / "oracle.f90.gcov"); line_counts = parse_gcov_line_counts(build / "oracle.f90.gcov"); raw = source.read_bytes()
        spans = {"enerbil_fusion": extract_procedure_bytes(ENERBIL, "enerbil_fusion"), "enerbil_initialize": extract_procedure_bytes(ENERBIL, "enerbil_initialize"), "hydro_subgrid_main": extract_procedure_bytes(HYDRO, "hydro_subgrid_main")}
        mappings = {name: (locate_generated_span(raw, span.span_bytes), span.start_line, span.span_sha256) for name, span in spans.items()}
        records = []
        for arm in arms:
            proc = arm["fortran_procedure"]; start, original, sha = mappings[proc]; original_map_line = {172: 173, 181: 182}.get(int(arm["line"]), int(arm["line"])); generated = start + original_map_line - original
            if arm["arm_id"].endswith("elsewhere:fallthrough"):
                hit = line_counts.get(generated, 0) > 0; detail = {"gcov_line_count": line_counts.get(generated, 0)}
            else:
                branch = select_arm_branch(arm["arm_id"], branch_rows.get(generated, [])); hit = bool(branch and branch["taken"] > 0); detail = {"gcov_branch": branch, "raw_gcov_branches": branch_rows.get(generated, [])}
            records.append({"arm_id": arm["arm_id"], "procedure": proc, "original_line": arm["line"], "condition_terminal_original_line": original_map_line, "generated_line": generated, "procedure_span_sha256": sha, "passed": hit, **detail})
    (output_dir / "branch_coverage.json").write_text(json.dumps({"schema_version": 1, "family": FAMILY, "mapping": "GCOV branches are mapped by exact extracted procedure byte spans; ELSEWHERE is mapped by its executed source line.", "arms": records, "branch_complete": all(x["passed"] or x["arm_id"] in proof_ids for x in records)}, indent=2) + "\n", encoding="ascii")
    evidence = []
    for region, owner in owners.items():
        required = {x["arm_id"] for x in arms if x["base_region_id"] == region}; gcov_ids = {x["arm_id"] for x in records if x["arm_id"] in required and x["passed"]}; source_ids = required & proof_ids; covered = gcov_ids | source_ids
        evidence.append({"owner_region_id": owner["owner_region_id"], "base_region_id": region, "fortran_procedure": owner["fortran_procedure"], "jax_owners": owner["jax_owners"], "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/comparison.json", "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json", "required_arm_ids": sorted(required), "gcov_arm_ids": sorted(gcov_ids), "source_proof_arm_ids": sorted(source_ids), "covered_arm_ids": sorted(covered), "missing_arm_ids": sorted(required-covered), "passed": required == covered})
    document = {"schema_version": 1, "family": FAMILY, "complete": all(x["passed"] for x in evidence), "records": evidence}
    (output_dir / "owner_region_evidence.json").write_text(json.dumps(document, indent=2) + "\n", encoding="ascii")
    if not document["complete"]:
        raise RuntimeError("missing arms: " + str([x["missing_arm_ids"] for x in evidence if not x["passed"]]))
    return document


if __name__ == "__main__":
    result = run_coverage(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(not result["complete"])
