"""Five finite source-owner comparisons and one real compiled closure.

Uses verbatim original Fortran procedures through the existing extractor.
Only driver inputs/case selection and extra output statements change.
This is not an independent full hydrol_main reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev import oracle_lane_hydrol_soil_owner as source_owner
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compile_fortran, compiler_environment, float_comparison
from scripts.dev.oracle_hydrol_component_owner import CASE_IDS, make_owner_inputs, run_owner_case, run_owner_diagnostics

DEFAULT_OUT = ROOT / "outputs/reviews/main-physics-20261117/hydrol-diffuco-state-closure/hydrol"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_reference(output_dir):
    """Original var_init+soil with zero melt, cold case18 and extra returns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hydrol_component_reference_") as temporary:
        directory = Path(temporary)
        path = directory / "reference.f90"
        hashes = source_owner.compose(path)
        # Latin-1 round-trip preserves every original source byte and newline.
        text = path.read_bytes().decode("latin1")
        selected = ",".join(str(value) for value in CASE_IDS)
        text = text.replace(f"do case_id=1,{source_owner.MAX_OWNER_CASE_ID}",
            f"do case_id=1,19\n   if (.not. any(case_id == (/{selected}/))) cycle")
        text = text.replace("tot_melt=0.05_r_std", "tot_melt=0._r_std")
        # The inherited generic extractor declares these module scalars but
        # never initializes them. Bind the frozen contract, not compiler BSS.
        text = text.replace("    call reset_state()", "    call reset_state()\n"
            "    froz_frac_corr=1._r_std; smtot_corr=2._r_std; max_froz_hydro=1._r_std")
        text = text.replace("mcl=mc; temp_hydro(1,1:2)=271.5_r_std",
            "mcl=mc; temp_hydro=271.5_r_std; stempdiag=271.5_r_std")
        text = text.replace("    call emit3(unit_out,case_id,'mc',mc)",
            "    call emit3(unit_out,case_id,'profil_froz_hydro_ns',profil_froz_hydro_ns)\n"
            "    call emit2(unit_out,case_id,'evap_bare_lim_ns',evap_bare_lim_ns)\n"
            "    call emit1(unit_out,case_id,'drysoil_frac',drysoil_frac)\n"
            "    call emit2(unit_out,case_id,'temp_hydro',temp_hydro)\n"
            "    call emit3(unit_out,case_id,'mc',mc)")
        path.write_bytes(text.encode("latin1"))
        preserved = {name:source_owner.extract_procedure_bytes(source_owner.SOURCE,name).span_bytes in path.read_bytes()
            for name in source_owner.PROCEDURES}
        assert all(preserved.values()), "original scientific procedures changed"
        shutil.copyfile(path, output_dir / "reference.f90")
        compiled = compile_fortran(path, directory / "reference.exe")
        started = time.perf_counter()
        process = subprocess.run([str(directory / "reference.exe")], cwd=directory,
            env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True)
        compiled.update(run_elapsed_s=time.perf_counter()-started,
            run_exit=process.returncode, run_stdout=process.stdout, run_stderr=process.stderr,
            extracted_procedure_sha256=hashes,verbatim_procedures_retained=preserved,
            source_sha256=_sha(source_owner.SOURCE))
        (output_dir / "reference_metadata.json").write_text(json.dumps(compiled,indent=2)+"\n")
        if process.returncode:
            raise RuntimeError(f"Fortran reference failed: {process.returncode}: {process.stderr}")
        shutil.copyfile(directory / "fortran_outputs.txt", output_dir / "fortran_outputs.csv")
    return source_owner._read_fortran(output_dir / "fortran_outputs.csv")


def closure_fields(result, initialized):
    soil, diagnostics = result.module.soil, result.diagnostics
    return dict(mc=soil.mc,mcl=soil.mcl,tmc=soil.tmc,ru_ns=soil.ru_ns,dr_ns=soil.dr_ns,
        water2infilt=soil.water2infilt,runoff2peat=soil.runoff2peat,
        tmc_litter=diagnostics.tmc_litter,tmc_litter_wilt=initialized.tmc_litter_wilt,
        tmc_litter_res=initialized.tmc_litter_res,humrel=diagnostics.humrel,
        vegstress=diagnostics.vegstress,nroot=diagnostics.nroot,
        evap_bare_lim=diagnostics.evap_bare_lim,
        evap_bare_lim_ns=diagnostics.evap_bare_lim_ns,
        profil_froz_hydro_ns=soil.profil_froz_hydro_ns,
        drysoil_frac=diagnostics.drysoil_frac,temp_hydro=result.module_inputs["temp_hydro"])


def independent_phase_checks(result):
    """Reuse C17's literal source NumPy matrix, never production references."""
    # Existing accepted independent formulas, loaded as helpers only.
    import importlib.util
    path = ROOT / "tests/unit/test_stage25_hydrol_boundary_fixes.py"
    spec = importlib.util.spec_from_file_location("hydrol_c17_source_formulas", path)
    formulas = importlib.util.module_from_spec(spec); spec.loader.exec_module(formulas)
    values = result.module_inputs
    soil = result.module.soil
    mc = np.asarray(soil.mc)
    freeze = values["ok_freeze_cwrr"]
    residual, saturation = float(values["mcr"][0]), float(values["mcs"][0])
    controls = {key:values[key] for key in ("fr_dt","froz_frac_corr","smtot_corr","max_froz_hydro")}
    texture = int(values["njsc"][0])-1
    tables = values["mineral_tables"]
    fresh = formulas._source_profile(mc, np.asarray(values["temp_hydro"]),np.asarray(values["dh_mm"]),
        residual,saturation,float(np.asarray(tables.nvan)[texture]) if hasattr(tables,"nvan") else 1.89,
        float(np.asarray(tables.avan)[texture]) if hasattr(tables,"avan") else .0075,
        values["ok_thermodynamical_freezing"],controls) if freeze else np.zeros_like(mc)
    comparisons = [float_comparison("fresh_profile", fresh,np.asarray(soil.profil_froz_hydro_ns),rtol=1e-12,atol=1e-14)]
    for tile, normal in enumerate(soil.tile_results):
        if float(values["soiltile"][0,tile]) <= 0:
            continue
        normal_mc = np.asarray(normal.solve.mc_final)
        comparisons.append(float_comparison(f"tile{tile+1}.normal_restore",normal_mc,mc[:,:,tile],rtol=0,atol=0))
        normal_mcl = np.minimum(normal_mc,residual+(1-np.asarray(values["profil_froz"])[:,:,tile])*(normal_mc-residual))
        comparisons.append(float_comparison(f"tile{tile+1}.normal_liquid_restore",normal_mcl,np.asarray(soil.mcl)[:,:,tile],rtol=1e-12,atol=1e-14))
        expected_tmc = np.array([formulas._water(normal_mc[0],np.asarray(values["dz_mm"]))+float(soil.water2infilt[0,tile])])
        comparisons.append(float_comparison(f"tile{tile+1}.normal_tmc",expected_tmc,np.asarray(soil.tmc)[:,tile],rtol=1e-12,atol=1e-12))
    # The accepted source matrix formula uses tile-specific hydraulic tables.
    # Mineral fixture shares one texture, and tile1 is active in cases1/5/18.
    if float(values["soiltile"][0,0]) > 0:
        inputs = {key:np.asarray(value) if hasattr(value,"shape") else value for key,value in values.items()}
        # Source6938-6939 multiplies the potential demand by AINT(frac_bare
        # +1-epsilon). C17's pure matrix helper assumes that factor is one.
        factor = np.trunc(np.asarray(result.module.vegupd.frac_bare_ns)[:,0]+1-1e-8)
        inputs["evapot_corr"] = np.asarray(values["evapot_corr"])*factor
        expected = formulas._source_dummy(mc[0,:,0],fresh[0,:,0],tables,inputs)
        alt = soil.evap_bare_limit_alt
        comparisons += [float_comparison("dummy_rhs",expected["rhs"],np.asarray(alt.saved.rhs)[0,:,0],rtol=1e-10,atol=1e-11),
            float_comparison("dummy_first",expected["first"],np.asarray(alt.first.mcl)[0,:,0],rtol=1e-10,atol=1e-11),
            float_comparison("dummy_second",expected["second"],np.asarray(alt.second.mcl)[0,:,0],rtol=1e-10,atol=1e-11),
            float_comparison("dummy_total",expected["mc_after"],np.asarray(alt.mc_after)[0,:,0],rtol=1e-10,atol=1e-11)]
    return comparisons


def independent_topmodel(result):
    diagnostics = run_owner_diagnostics(result,topmodel_new=True)
    mc = np.asarray(result.module.soil.mc)
    temperature = np.asarray(result.module_inputs["temp_hydro"])
    dh = np.asarray(result.module_inputs["dh_mm"])
    saturation = np.asarray(result.module_inputs["mcs"])
    tiles = np.asarray(result.module_inputs["soiltile"])
    expected = np.zeros(mc.shape[0])
    for point in range(mc.shape[0]):
        unfrozen = 0
        for layer in range(mc.shape[1]):
            if temperature[point,layer] < 273.15:
                break
            unfrozen += 1
        if unfrozen <= 6:
            expected[point] = 0.
            continue
        depth = sum(dh[:unfrozen])
        mean = 0.
        for tile in range(mc.shape[2]):
            water = sum(mc[point,layer,tile]/saturation[point]*dh[layer] for layer in range(unfrozen))/1000-depth/1000
            if water > 0:
                water *= saturation[point]
            mean += tiles[point,tile]*water
        expected[point] = max(0.,min(.7,(1+.8*np.exp(-2.*(mean-.1)))**(-1/.8)))
    return float_comparison("topmodel_independent_chain.fwet_new",expected,np.asarray(diagnostics.fwet_new),rtol=1e-12,atol=1e-14)


def run(output_dir=DEFAULT_OUT, *, jit_case=None, case_ids=None, reuse_reference=False):
    output_dir.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend() == "cpu", jax.default_backend()
    assert jax.config.jax_enable_x64, "requires actual JAX x64"
    started = time.perf_counter()
    references = source_owner._read_fortran(output_dir/"fortran_outputs.csv") if reuse_reference else build_reference(output_dir)
    cases, all_comparisons = [], []
    for case_id in ((jit_case,) if jit_case else (case_ids or CASE_IDS)):
        before = time.perf_counter()
        values = make_owner_inputs(case_id)
        result = run_owner_case(case_id,use_jit=jit_case is not None)
        fields = closure_fields(result,values["initialized"])
        np.savez_compressed(output_dir/f"{'jit' if jit_case else 'eager'}_case{case_id}.npz",
            **{name:np.asarray(value) for name,value in fields.items()},
            dummy_rhs=np.asarray(result.module.soil.evap_bare_limit_alt.saved.rhs),
            dummy_first=np.asarray(result.module.soil.evap_bare_limit_alt.first.mcl),
            dummy_second=np.asarray(result.module.soil.evap_bare_limit_alt.second.mcl),
            incoming_profile=np.asarray(result.module_inputs["profil_froz"]))
        comparisons = []
        for name, expected in references[case_id].items():
            comparisons.append(float_comparison(name,expected,np.asarray(fields[name]).ravel(order="C"),rtol=1e-12,atol=1e-14))
        phases = independent_phase_checks(result)
        topmodel = independent_topmodel(result)
        case = dict(case_id=case_id,jit=jit_case is not None,elapsed_s=time.perf_counter()-before,
            contract=values["contract"],source_fields=comparisons,phase_checks=phases,
            independent_topmodel=topmodel,snow_zero=bool(np.all(np.asarray(result.snow_state.snow)==0)),
            wrapper_fwet_none=result.diagnostics.fwet_new is None)
        cases.append(case)
        all_comparisons += comparisons+phases+[topmodel]
        print(json.dumps(dict(case=case_id,jit=jit_case is not None,elapsed_s=case["elapsed_s"],
            passed=sum(x["passed"] for x in comparisons+phases+[topmodel]),
            failed=[x for x in comparisons+phases+[topmodel] if not x["passed"]])),flush=True)
    result = dict(backend=jax.default_backend(),x64=jax.config.jax_enable_x64,
        elapsed_s=time.perf_counter()-started,cases=cases,
        count=len(all_comparisons),passed=sum(x["passed"] for x in all_comparisons),
        failed=sum(not x["passed"] for x in all_comparisons),
        identities={str(path.relative_to(ROOT)).replace("\\","/"):_sha(path) for path in
            [ROOT/"jax_orchidee/sechiba/hydrol.py",Path(__file__),ROOT/"scripts/dev/oracle_hydrol_component_owner.py",
             ROOT/"tests/unit/test_stage25_hydrol_boundary_fixes.py",source_owner.SOURCE]},
        limitation="Partial owner certification: independent original var_init+soil chain; full hydrol_main is not independently compiled.")
    name = "jit_comparison.json" if jit_case else (f"case{case_ids[0]}_comparison.json" if case_ids else "eager_comparison.json")
    (output_dir/name).write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--jit-case",type=int,choices=CASE_IDS)
    parser.add_argument("--case",type=int,choices=CASE_IDS)
    parser.add_argument("--reuse-reference",action="store_true")
    args=parser.parse_args()
    result=run(args.output_dir,jit_case=args.jit_case,case_ids=(args.case,) if args.case else None,reuse_reference=args.reuse_reference)
    print(json.dumps({key:result[key] for key in ("backend","x64","elapsed_s","count","passed","failed")}))
    raise SystemExit(bool(result["failed"]))
