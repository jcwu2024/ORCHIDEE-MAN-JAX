"""Source-extracted, phase-separated full THERMOSOIL component oracle.

The input seam is the saved original-Fortran initialization output of the C50
oracle. This executes neither a model timestep nor its initialization again.
The original profile, energy, humlev, wetness, getdiff and coef routines are
compiled unchanged. The current production owner selects its actual eager or
module-level compiled delegate. No equivalent test wrapper replaces a kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import oracle_thermosoil_initialization as initial  # noqa: E402
from fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS, DEFAULT_COMPILER, compiler_environment, float_comparison,
    write_point_comparisons,
)

SAVED = ROOT / "outputs/reviews/main-physics-20261116/thermal-initialization-oracle"
CASES = (
    ("explicit_complete", "attempt03/complete_no_snow", True),
    ("explicit_thin_partial", "attempt03/complete_thin_partial", True),
    ("explicit_thin_whole", "attempt03/complete_thin_whole", True),
    ("explicit_cold", "attempt03/cold_all_coefficients_missing", True),
    ("explicit_sparse", "sparse-before/complete_sparse_vegetation", True),
    ("bucket_no_snow", "attempt03/complete_no_snow", False),
    ("bucket_thin", "attempt03/complete_thin_whole", False),
    ("bucket_thick", "attempt03/complete_thick", False),
)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path) as raw:
        return {name: raw[name].copy() for name in raw.files}


def saved_input(relative, explicit):
    directory = SAVED / relative
    v, reference = read_npz(directory / "inputs.npz"), read_npz(directory / "fortran.npz")
    # All state inputs come from the saved original Fortran, not saved JAX.
    result = {name: v[name] for name in (
        "QZ", "SMCMAX", "mcs", "so_capa_dry_ns", "zlt", "dlt", "dz1", "refSOC", "znt",
        "organic_layer_thick", "soilc_total", "snow", "snowrho", "snowtemp", "snowdz", "pb",
        "veget_max", "frac_snow_veg", "frac_snow_nobio", "totfrac_nobio", "njsc", "scalars", "ok_lai_i",
    )}
    result["dz5"] = (v["zlt"][:-1] - v["znt"][:-1]) * v["dz1"]
    result.update({name: reference[f"moisture.{name}"] for name in initial.MOISTURE_FIELDS})
    result.update({name: reference[f"initial.{name}"] for name in initial.COEFFICIENTS})
    result.update({name: reference[f"initial.{name}"] for name in ("pcapa_en", "temp_sol_beg")})
    result["ptn"] = reference["initial.ptn_beg"]
    result["shum_ngrnd_permalong"] = v["shum_ngrnd_permalong"]
    result["temp_sol_new"] = v["post_temp_sol"]
    result["temp_sol_new_pft"] = v["temp_sol_new_pft"]
    # The source flag is active for this bounded component, in both snow modes.
    result["controls"] = np.asarray([int(explicit), 1], dtype=np.int32)
    return result, {"saved_reference_directory": str(directory),
        "saved_inputs_sha256": digest(directory / "inputs.npz"),
        "saved_Fortran_npz_sha256": digest(directory / "fortran.npz"),
        "saved_Fortran_binary_sha256": digest(directory / "outputs.bin"),
        "initialization": "saved C50 original Fortran explicit-snow initialization; no new initialization execution",
        "no_explicit_snow_boundary": "same declared restored coefficient/material seam, not a no-explicit-snow initialization oracle" if not explicit else None}


def compile_unit(inputs):
    initial.SPANS.clear()
    header = (ROOT / "scripts/dev/oracle_thermosoil_cold_coef_actual.f90.template").read_text().split("contains", 1)[0]
    header = header.replace("thermosoil_cold_coef_actual_oracle", "thermal_component_oracle")
    declarations = """
  real(r_std), parameter :: so_cond_wet=1.89_r_std
  real(r_std) :: znt(ngrnd), dz5(ngrnd-1), ptn_beg(kjpindex,ngrnd,nvm),temp_sol_beg(kjpindex)
  real(r_std) :: shum_ngrnd_perma(kjpindex,ngrnd,nvm),surfheat_incr(kjpindex),coldcont_incr(kjpindex)
  real(r_std) :: cgrnd(kjpindex,ngrnd-1,nvm),dgrnd(kjpindex,ngrnd-1,nvm)
  logical :: satsoil=.false.
contains
  subroutine ipslerr_p(level,a,b,c,d)
    integer,intent(in)::level
    character(*),intent(in)::a,b,c,d
    if(level>=3) then
      write(*,*) a,b,c,d
      error stop 'source ipslerr_p'
    endif
  end subroutine
"""
    routines = "\n".join(initial.routine(initial.THERMAL, name) for name in (
        "thermosoil_cond_pft", "thermosoil_cond_nopft", "thermosoil_cond", "thermosoil_getdiff",
        "thermosoil_getdiff_old_thermix_with_snow", "thermosoil_coef", "thermosoil_humlev",
        "thermosoil_diaglev", "thermosoil_profile", "thermosoil_energy", "thermosoil_wlupdate"))
    program = """
end module
program thermal_component_driver
  use thermal_component_oracle
  implicit none
  integer::iu,ou,m,jg,ji
  real(r_std)::thawed_humidity(1),stempdiag(1,11),profil_froz(1,32,14),pcappa_supp(1,32,14)
  real(r_std)::veget_max_bg(1,14),pkappa_pftmean(1,32),gtemp(1),ptnlev1(1)
  real(r_std)::deephum_prof(1,32,14),deeptemp_prof(1,32,14)
  character(1024)::input_path,output_path
"""
    module_names = {"QZ", "SMCMAX", "mcs", "so_capa_dry_ns", "zlt", "dlt", "dz1", "refSOC", "znt", "dz5",
        "ptn", "pcapa_en", "temp_sol_beg", "cgrnd", "dgrnd", "shum_ngrnd_permalong"}
    for name, array in inputs.items():
        if name not in module_names:
            kind = "integer(i_std)" if array.dtype.kind in "iu" else "real(r_std)"
            program += f"  {kind}::{name}({','.join(map(str, array.shape))})\n"
    program += """
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  open(newunit=iu,file=trim(input_path),access='stream',form='unformatted',status='old',action='read')
"""
    program += "".join(f"  read(iu) {name}\n" for name in inputs)
    program += """
  close(iu)
  dt_sechiba=scalars(1);lambda=scalars(2);phigeoth=scalars(3);psnowdzmin=scalars(4)
  use_toporganiclayer_tempdiff=.false.;use_soilc_tempdiff=.true.;use_refSOC=.true.
  ok_freeze_thermix=.true.;ok_explicitsnow=controls(1)/=0;ok_LAIdev=ok_lai_i/=0;veget_mask_2d=.true.
  veget_max_bg(:,2:nvm)=veget_max(:,2:nvm)
  veget_max_bg(:,1)=MAX((un-SUM(veget_max(:,2:nvm),2)),zero)
  thawed_humidity=1.
  open(newunit=ou,file=trim(output_path),access='stream',form='unformatted',status='replace',action='write')
  call thermosoil_humlev(1,shumdiag_perma,thawed_humidity,mc_layh,mcl_layh,tmc_layh,mc_layh_pft,mcl_layh_pft,tmc_layh_pft)
  if(controls(2)/=0) then
    call thermosoil_wlupdate(1,ptn,shum_ngrnd_perma,shum_ngrnd_permalong)
  else
    shum_ngrnd_permalong=shum_ngrnd_perma
  endif
"""
    layout = []

    def dump(phase, fields):
        nonlocal program
        for name, expression, shape in fields:
            program += f"  write(ou) {expression}\n"
            layout.append((f"{phase}.{name}", shape))

    dump("humlev", [(name, name, (1,32,14) if name.endswith("_pft") or name=="shum_ngrnd_perma" else (1,32)) for name in initial.HUMLEV_FIELDS])
    dump("wetness", [("hsdlong", "shum_ngrnd_permalong", (1,32,14))])
    program += initial.span(initial.THERMAL, 906, 909, "main_profile_call")
    program += initial.span(initial.THERMAL, 913, 913, "main_energy_call")
    dump("profile", [("ptn","ptn",(1,32,14)),("stempdiag","stempdiag",(1,11))])
    dump("energy", [(name,name,(1,32,14) if name=="ptn_beg" else (1,)) for name in ("surfheat_incr","coldcont_incr","ptn_beg","temp_sol_beg")])
    program += initial.span(initial.THERMAL, 914, 920, "main_precoef_pftmean")
    program += initial.span(initial.THERMAL, 1002, 1032, "main_coef_and_final_state")
    dump("getdiff", [(name,name,(1,32,14)) for name in ("pcapa","pcapa_en","pkappa")])
    # Explicit-only values are defined by getdiff. The old thermix branch does
    # not write these fields; conditional output avoids claiming defined tails.
    program += "  if(ok_explicitsnow) then\n"
    dump("getdiff", [(name,name,(1,32,14)) for name in ("profil_froz","pcappa_supp")])
    dump("getdiff", [(name,name,(1,3)) for name in ("pcapa_snow","pkappa_snow")])
    program += "  endif\n"
    dump("coef", [(name,name,shape) for name,shape in initial.COEFFICIENTS.items()])
    dump("final", [(name,name,(1,32) if name.endswith("pftmean") else (1,32,14) if name.endswith("prof") else (1,)) for name in ("ptn_pftmean","pkappa_pftmean","gtemp","ptnlev1","deephum_prof","deeptemp_prof")])
    program += "  close(ou)\nend program\n"
    return header + declarations + routines + "\n" + program, layout


def owner_kwargs(v):
    return dict(moisture=NS(**{name:v[name] for name in initial.MOISTURE_FIELDS}),
        thermosoil_restart=NS(ptn=v["ptn"],refsoc=v["refSOC"],gtemp=np.asarray([123.]),
            **{name:v[name] for name in ("cgrnd","dgrnd","cgrnd_snow","dgrnd_snow","lambda_snow","pcapa_en","temp_sol_beg","shum_ngrnd_permalong")},
            veget_mask_2d=np.ones((1,14),bool)),
        enerbil_payload={name:v[name] for name in ("temp_sol_new","temp_sol_new_pft","soilcap")},
        condveg_result=NS(frac_snow_veg=v["frac_snow_veg"],frac_snow_nobio=v["frac_snow_nobio"]),
        **{name:v[name] for name in ("snowrho","snowtemp","snowdz","snow","njsc","veget_max","pb","totfrac_nobio","dlt","dz1","zlt","znt","dz5")},
        dt_sechiba=float(v["scalars"][0]),ok_laidev=v["ok_lai_i"].astype(bool),
        ok_explicitsnow=bool(v["controls"][0]),ok_shum_ngrnd_permalong=bool(v["controls"][1]))


def flatten_result(module, explicit):
    result = {f"humlev.{name}":getattr(module.humlev,name) for name in initial.HUMLEV_FIELDS}
    result.update({"wetness.hsdlong":module.final.deephum_prof})
    result.update({f"profile.{name}":getattr(module.profile,name) for name in ("ptn","stempdiag")})
    result.update({f"energy.{name}":getattr(module.energy,name) for name in ("surfheat_incr","coldcont_incr","ptn_beg","temp_sol_beg")})
    material=("pcapa","pcapa_en","pkappa", "profil_froz","pcappa_supp","pcapa_snow","pkappa_snow") if explicit else ("pcapa","pcapa_en","pkappa")
    result.update({f"getdiff.{name}":getattr(module.getdiff,name) for name in material})
    result.update({f"coef.{name}":getattr(module.coef.soil if name in ("cgrnd","dgrnd","soilcap_pft","soilflx_pft") else module.coef,name) for name in initial.COEFFICIENTS})
    result.update({f"final.{name}":getattr(module.final,name) for name in ("ptn_pftmean","pkappa_pftmean","gtemp","ptnlev1","deephum_prof","deeptemp_prof")})
    return {name:np.asarray(value) for name,value in result.items()}


def run_owner(v, use_jit):
    from jax_orchidee.sechiba import thermosoil as th
    selected = ("_thermosoil_explicit_step_jit" if bool(v["controls"][0]) else "_thermosoil_no_explicit_snow_step_jit") if use_jit else ("thermosoil_explicit_step" if bool(v["controls"][0]) else "thermosoil_no_explicit_snow_step")
    actual=getattr(th,selected)
    calls=[]

    def observe(**kwargs):
        calls.append({"ptn_shape":list(kwargs["ptn"].shape),"pcapa_en_previous_present":kwargs["pcapa_en_previous"] is not None})
        return actual(**kwargs)

    started=time.perf_counter()
    with patch.object(th,selected,observe):
        result=th.run_thermosoil_first_step_module(**owner_kwargs(v),use_jit=use_jit)
        assert result.ok
        arrays=flatten_result(result.module,bool(v["controls"][0]))
    assert len(calls)==1
    return result,arrays,{"selected_delegate":selected,"observed_calls":calls,"actual_production_owner":True,
        "full_component":True,"use_jit":use_jit,"elapsed_seconds":time.perf_counter()-started}


def compare(reference, actual):
    assert reference.keys()==actual.keys(), (reference.keys()-actual.keys(),actual.keys()-reference.keys())
    return [float_comparison(name,reference[name],actual[name],rtol=1.e-12,atol=1.e-12) for name in reference]


def run(output, reuse_reference=None, names=None):
    if output.exists():
        raise FileExistsError(f"New attempt directory required: {output}")
    output.mkdir(parents=True)
    (output/"runner.py").write_bytes(Path(__file__).read_bytes())
    write(output/"invocation.json",{"argv":sys.argv,"cwd":str(Path.cwd()),"python":sys.executable,
        "production_sha256":digest(ROOT/"jax_orchidee/sechiba/thermosoil.py"),"runner_sha256":digest(Path(__file__))})
    started=time.perf_counter()
    chosen=[c for c in CASES if names is None or c[0] in names]
    v,_=saved_input(chosen[0][1],chosen[0][2])
    source,layout=compile_unit(v)
    sourcepath=output/"oracle.f90"
    sourcepath.write_text(source,encoding="utf-8",newline="\n")
    write(output/"source-spans.json",initial.SPANS)
    write(output/"output-layout.json",layout)
    executable=output/"oracle.exe"
    if reuse_reference is not None:
        if digest(reuse_reference/"oracle.f90")!=digest(sourcepath):
            raise ValueError("Source reference changed")
        write(output/"compile.json",{"new_compilation":False,"reuse_reference":str(reuse_reference),"source_sha256":digest(sourcepath)})
    else:
        command=[str(DEFAULT_COMPILER),*COMPILE_FLAGS,str(sourcepath.resolve()),"-o",str(executable.resolve())]
        start=time.perf_counter()
        completed=subprocess.run(command,cwd=output,env=compiler_environment(DEFAULT_COMPILER),text=True,capture_output=True,timeout=60)
        (output/"compile.stdout").write_text(completed.stdout,encoding="utf-8")
        (output/"compile.stderr").write_text(completed.stderr,encoding="utf-8")
        write(output/"compile.json",{"new_compilation":True,"command":command,"returncode":completed.returncode,
            "elapsed_seconds":time.perf_counter()-start,"source_sha256":digest(sourcepath),"compiler_sha256":digest(DEFAULT_COMPILER)})
        completed.check_returncode()
    summaries=[]
    import jax
    for name,relative,explicit in chosen:
        directory=output/name
        directory.mkdir()
        v,origin=saved_input(relative,explicit)
        write(directory/"input-origin.json",origin)
        np.savez(directory/"inputs.npz",**v)
        binary=directory/"inputs.bin"
        with binary.open("wb") as handle:
            for array in v.values():
                handle.write(np.asarray(array,dtype="<i4" if array.dtype.kind in "iu" else "<f8").ravel(order="F").tobytes())
        write(directory/"input-layout.json",[{"field":k,"shape":list(a.shape),"dtype":str(a.dtype),"sha256":hashlib.sha256(a.tobytes()).hexdigest()} for k,a in v.items()])
        expected_layout=[(n,s) for n,s in layout if explicit or n not in ("getdiff.profil_froz","getdiff.pcappa_supp","getdiff.pcapa_snow","getdiff.pkappa_snow")]
        if reuse_reference is not None:
            old=reuse_reference/name
            if digest(old/"inputs.bin")!=digest(binary):
                raise ValueError(f"Input changed for reference reuse: {name}")
            reference=read_npz(old/"fortran.npz")
            execution={"new_Fortran_process":False,"reused_from":str(old),"binary_sha256":digest(old/"outputs.bin")}
        else:
            out=directory/"outputs.bin"
            command=[str(executable.resolve()),str(binary.resolve()),str(out.resolve())]
            start=time.perf_counter()
            completed=subprocess.run(command,cwd=directory,env=compiler_environment(DEFAULT_COMPILER),text=True,capture_output=True,timeout=20)
            (directory/"fortran.stdout").write_text(completed.stdout,encoding="utf-8")
            (directory/"fortran.stderr").write_text(completed.stderr,encoding="utf-8")
            execution={"new_Fortran_process":True,"command":command,"returncode":completed.returncode,"elapsed_seconds":time.perf_counter()-start,"executable_sha256":digest(executable)}
            write(directory/"execution.json",execution)
            if completed.returncode:
                summaries.append({"name":name,"status":"source_rejected","execution":execution})
                continue
            raw=np.fromfile(out,dtype="<f8")
            reference={}
            offset=0
            for field,shape in expected_layout:
                size=int(np.prod(shape))
                reference[field]=raw[offset:offset+size].reshape(shape,order="F")
                offset+=size
            assert offset==raw.size
        write(directory/"execution.json",execution)
        np.savez(directory/"fortran.npz",**reference)
        runtimes=[]
        for use_jit in (False,True):
            label="jit" if use_jit else "eager"
            try:
                owner,actual,observed=run_owner(v,use_jit)
                np.savez(directory/f"{label}.npz",**actual)
                np.savez(directory/f"{label}-downstream.npz",**{k:np.asarray(x) for k,x in owner.downstream_payload.items()})
                checks=compare(reference,actual)
                write_point_comparisons(directory/f"{label}-points.csv",reference,actual,rtol=1.e-12,atol=1.e-12)
                record={**observed,"status":"passed" if all(c["passed"] for c in checks) else "failed","comparisons":checks}
            except Exception as exc:
                record={"status":"runtime_error","use_jit":use_jit,"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}
            write(directory/f"{label}.json",record)
            runtimes.append(record)
            print(json.dumps({"case":name,"runtime":label,"status":record["status"],"failed":[c["name"] for c in record.get("comparisons",[]) if not c["passed"]],"message":record.get("message")}),flush=True)
        summaries.append({"name":name,"explicit_snow":explicit,"status":"passed" if all(r["status"]=="passed" for r in runtimes) else "failed","runtime_results":runtimes})
    result={"status":"passed" if all(s["status"]=="passed" for s in summaries) else "failed",
        "elapsed_seconds":time.perf_counter()-started,"case_count":len(summaries),"cases":summaries,
        "jax_version":jax.__version__,"jax_x64":bool(jax.config.jax_enable_x64),"devices":[str(d) for d in jax.devices()],
        "new_Fortran_processes":len(chosen) if reuse_reference is None else 0,"new_compilations":int(reuse_reference is None),
        "rtol":1.e-12,"atol":1.e-12,"full_model_steps":0,"numerical_acceptance":False}
    write(output/"comparison.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reuse-reference",type=Path)
    parser.add_argument("--cases",nargs="+",choices=[c[0] for c in CASES])
    args=parser.parse_args()
    attempt=args.output.resolve()
    try:
        report=run(attempt,args.reuse_reference.resolve() if args.reuse_reference else None,args.cases)
    except FileExistsError:
        raise
    except Exception as exc:
        if attempt.is_dir():
            write(attempt/"failure.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        raise
    raise SystemExit(0 if report["status"]=="passed" else 1)
