"""Phase06 source-extracted forcing and output-layer micro references.

This runner never calls an old runner's main or writes old oracle outputs.
Only --build-reference compiles original local Fortran spans into TEMP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/forcing-modelout"
FIXTURE = ROOT / "tests/fixtures/phase06_forcing_modelout.json"
STATIC_FIXTURE = ROOT / "tests/fixtures/phase06_salinity_tide.json"
CONSUMER_FIXTURE = ROOT / "tests/fixtures/phase06_forcing_consumer.json"
NEAREST_FIXTURE = ROOT / "tests/fixtures/phase06_static_nearest.json"
COMPILER = Path("C:/msys64/ucrt64/bin/gfortran.exe")
FORCING = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
LPJ = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90"
PAPER = ROOT / "fortran_run_scripts/paper_250919/c2.4_Model_run_functions_sensitivity.py"
BASE_FIELDS = ("Tair", "PSurf", "Qair", "Wind_N", "Wind_E", "Rainf", "Snowf", "SWdown", "LWdown")
WATCH_FIELDS = ("SWnet", "Eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy")
FIELD_MAP = dict(zip(
    ("tair", "pb", "qair", "u", "v", "rainf", "snowf", "swdown", "lwdown"), BASE_FIELDS, strict=True,
))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": sha(path)}


def span(path, first, last):
    return b"".join(Path(path).read_bytes().splitlines(keepends=True)[first - 1:last])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compile_run(work, sources, output, *arguments):
    exe = work / (output.stem + ".exe")
    command = [str(COMPILER), "-cpp", "-fdefault-real-8", "-ffree-line-length-none",
               "-ffunction-sections", "-Wl,--gc-sections", *map(str, sources), "-o", str(exe)]
    env = {**os.environ, "PATH": str(COMPILER.parent) + os.pathsep + os.environ.get("PATH", "")}
    build = subprocess.run(command, cwd=work, env=env, text=True, capture_output=True, check=True)
    run = subprocess.run([str(exe), str(output), *arguments], cwd=work, env=env,
                         text=True, capture_output=True, check=True)
    version = subprocess.run([str(COMPILER), "--version"], env=env, text=True,
                             capture_output=True, check=True).stdout.splitlines()[0]
    return {"command": command, "compiler_version": version, "stdout": build.stdout,
            "stderr": build.stderr, "run_stdout": run.stdout, "run_stderr": run.stderr,
            "compile_units": [{"name": path.name, "sha256": sha(path)} for path in sources]}


def saved_forcing(path):
    result = {}
    with path.open(encoding="ascii", newline="") as handle:
        for line, (name, step, value) in enumerate(csv.reader(handle), 1):
            group = result.setdefault(int(step), {}).setdefault(name, {"values": [], "csv_lines": []})
            group["values"].append(float(value))
            group["csv_lines"].append(line)
    return result


def build_forcing(work, artifacts):
    runner_path = ROOT / "scripts/dev/oracle_lane_driver_forcing_owner_run.py"
    old = import_file(runner_path, "phase06_old_forcing_extractor")
    generated = work / "readdim2_extracted.f90"
    generated.write_text(old._generated_source(), encoding="utf-8")
    solar = work / "solar_extracted.f90"
    solar.write_text(old._generated_solar_source(), encoding="utf-8")
    stubs = ROOT / "scripts/dev/oracle_lane_driver_forcing_owner_stubs.f90"
    harness_path = ROOT / "scripts/dev/oracle_lane_driver_forcing_owner_harness.f90"
    harness = harness_path.read_text(encoding="utf-8")
    new_harness = work / "forcing_phase06.f90"
    overrides = """  if(trim(scenario)=='hourly_split2'.or.trim(scenario)=='hourly_split2_netrad') then
    split=2;spread=1;dt_force=3600.;nsteps=4
    conserve=trim(scenario)=='hourly_split2_netrad'
  endif
"""
    new_harness.write_text(harness.replace("  kindex=0; nbindex=0", overrides + "  kindex=0; nbindex=0"), encoding="utf-8")
    builds = []
    cases = []
    old_dir = ROOT / "outputs/reference_mode/micro_oracles/driver_forcing_owner"
    for scenario in ("standard", "netrad", "watchout", "high_standard", "hourly"):
        path = old_dir / f"fortran_outputs_{scenario}.csv"
        expected = saved_forcing(path)
        split, spread, cadence = (1, 1, 3600.0) if scenario == "hourly" else (4, 2, 21600.0)
        cases.append({"id": scenario, "split": split, "spread": spread, "dt_force": cadence,
                      "netrad_cons": scenario == "netrad", "watchout": scenario == "watchout",
                      "high_solar": scenario == "high_standard", "reference": binding(path),
                      "expected": {str(k): v for k, v in expected.items() if k > 0}})
    for scenario in ("hourly_split2", "hourly_split2_netrad"):
        output = artifacts / f"fortran_outputs_{scenario}.csv"
        builds.append(compile_run(work, [stubs, solar, generated, new_harness], output, scenario))
        cases.append({"id": scenario, "split": 2, "spread": 1, "dt_force": 3600.0,
                      "netrad_cons": scenario.endswith("netrad"), "watchout": False, "high_solar": False,
                      "reference": binding(output),
                      "expected": {str(k): v for k, v in saved_forcing(output).items() if k > 0}})
    input_program = work / "forcing_inputs.f90"
    quoted = ",".join("'" + name + "'" for name in (*BASE_FIELDS, "levels", *WATCH_FIELDS))
    input_program.write_text(f"""program inputs
use readdim2
implicit none
integer::u0,t,k,i,j
real::value(2,2)
character(len=16)::names(18)=[character(len=16)::{quoted}]
character(len=1024)::path
call get_command_argument(1,path)
open(newunit=u0,file=trim(path),status='replace')
do k=1,size(names)
 do t=1,24
  call oracle_flinget(trim(names(k)),t,value)
  do j=1,2
   do i=1,2
    write(u0,'(A,4(",",I0),",",ES26.17E3)')trim(names(k)),t,i,j,1,value(i,j)
   enddo
  enddo
 enddo
enddo
close(u0)
end program
""", encoding="utf-8")
    raw_path = artifacts / "fortran_raw_inputs.csv"
    builds.append(compile_run(work, [stubs, solar, generated, input_program], raw_path))
    inputs = {name: np.zeros((24, 4, 1)) for name in (*BASE_FIELDS, "levels", *WATCH_FIELDS)}
    with raw_path.open(encoding="ascii", newline="") as handle:
        for name, record, i, j, _, value in csv.reader(handle):
            inputs[name][int(record) - 1, (int(j) - 1) * 2 + int(i) - 1, 0] = float(value)
    for path in (generated, solar, new_harness, input_program):
        (artifacts / path.name).write_bytes(path.read_bytes())
    return {"inputs": {name: value.tolist() for name, value in inputs.items()}, "cases": cases,
            "source": [binding(FORCING), binding(runner_path), binding(harness_path), binding(stubs)],
            "old_reference_limit": "Existing original-Fortran outputs reused. Old build receipt lacks compiler-version/source/compile-unit hashes; current extractor identities do not retroactively establish old build bytes.",
            "new_builds": builds}


def build_modelout(work, artifacts):
    template_path = ROOT / "scripts/dev/oracle_lane_driver_modelout.f90.template"
    template = template_path.read_text(encoding="utf-8")
    template = template.replace("nelements=1", "nelements=2").replace("integer::unit,i,j", "integer::unit,i,j,p,e,record")
    begin = template.index("  biomass=0.")
    end = template.index("  open(newunit=unit", begin)
    template = template[:begin] + template[end:]
    start = """  do record=1,3
  do e=1,nelements;do p=1,nparts;do j=1,nvm;do i=1,npts
    biomass(i,j,p,e)=10000*record+1000*i+100*j+p+e/8.
  enddo;enddo;enddo;enddo
  do j=1,nvm;do i=1,npts
    gpp_daily(i,j)=1.5*record+.25*i+.03125*j
    npp_daily(i,j)=-2.+.5*record+.125*i+.0625*j
  enddo;enddo
  ind=0.;cn_ind=0.;woodmass_ind=0.;moiavail_week=0.;tot_live_biomass=0.
"""
    source = work / "modelout_phase06.f90"
    template = template.replace("! <MODELOUT_XIOS_SPAN>", start + span(LPJ, 1677, 1699).decode() + "\n  enddo")
    template = template.replace("write(unit,'(A,\",\",ES26.17E3)') trim(name),value(ii,jj)",
                                "write(unit,'(A,3(\",\",I0),\",\",ES26.17E3)') trim(name),record,ii,jj,value(ii,jj)")
    source.write_text(template, encoding="utf-8")
    output = artifacts / "fortran_modelout_sends.csv"
    build = compile_run(work, [source], output)
    expected = {}
    with output.open(encoding="ascii", newline="") as handle:
        for name, record, point, pft, value in csv.reader(handle):
            array = expected.setdefault(name, np.zeros((3, 2, 14)))
            array[int(record) - 1, int(point) - 1, int(pft) - 1] = float(value)
    records = []
    for record in range(1, 4):
        point, pft, part, element = np.indices((2, 14, 12, 2)) + 1
        biomass = 10000 * record + 1000 * point + 100 * pft + part + element / 8.0
        point, pft = np.indices((2, 14)) + 1
        records.append({"biomass": biomass.tolist(), "gpp_daily": (1.5 * record + .25 * point + .03125 * pft).tolist(),
                        "npp_daily": (-2 + .5 * record + .125 * point + .0625 * pft).tolist()})
    # Execute exact original Python expressions on legal small history arrays.
    import pandas as pd

    paper = PAPER.read_text(encoding="utf-8").splitlines(keepends=True)
    selection_source = textwrap.dedent("".join(paper[59:76]))
    conversion_source = textwrap.dedent("".join(paper[114:118]))
    history = {("NPP" if name == "NPP_STOMATE" else name): np.mean(values, axis=0).T[None, :, None, :]
               for name, values in expected.items()}
    namespace = {"dt_nc": history}
    exec(compile(selection_source, str(PAPER) + ":60-76", "exec"), {}, namespace)
    table = pd.DataFrame({"AGB_model": [namespace["dt_AGB"]], "BGB_model": [namespace["dt_BGB"]],
                          "GPP_model": [namespace["dt_GPP"]], "NPP_model": [namespace["dt_NPP"]]})
    conversion = {"dt_obs_grid": table, **{name: [0, 1] for name in ("p_std_AGB", "p_std_BGB", "p_std_GPP", "p_std_NPP")}}
    exec(compile(conversion_source, str(PAPER) + ":115-118", "exec"), {}, conversion)
    (artifacts / "paper_selection.py.txt").write_text(selection_source, encoding="utf-8")
    (artifacts / "paper_conversion.py.txt").write_text(conversion_source, encoding="utf-8")
    (artifacts / source.name).write_bytes(source.read_bytes())
    return {"records": records, "expected_sends": {name: value.tolist() for name, value in expected.items()},
            "expected_finite_mean_modelout_point0": {name: float(table[name][0]) for name in table.columns},
            "source": [binding(LPJ), binding(PAPER), binding(template_path)], "build": build,
            "phase": "Three explicit output-layer send records only; no process timestep/day/year or natural annual integration.",
            "units": {"mass_inputs": "gC/m^2", "AGB_BGB_factor": 0.02,
                      "AGB_BGB_limit": "exact paper expression;0.02 includes paper carbon-to-biomass/area convention, no independently inferred new conversion",
                      "GPP": "gC/day/m^2; paper forwards without factor", "NPP": "gC/day/(m^2 tot); paper forwards without factor"}}


def cache_from_fixture(fixture, case):
    from jax_orchidee.driver.domain import PaperPointForcingCache

    inputs = {name: np.asarray(value, dtype=np.float64) for name, value in fixture["forcing"]["inputs"].items()}
    if case["high_solar"]:
        inputs["SWdown"] = np.full_like(inputs["SWdown"], 10000.0)
    values = {name: inputs[name] for name in BASE_FIELDS}
    values.update(Areas=np.ones((4, 1)), contfrac=np.ones((4, 1)), Height_Lev1=2.0, Height_Levuv=10.0)
    if case["watchout"]:
        values.update(is_watchout=True, dt_force=case["dt_force"], zlev=inputs["levels"], zlevuv=inputs["levels"],
                      **{name: inputs[name] for name in WATCH_FIELDS if name not in {"SWnet", "Eair"}},
                      swnet=inputs["SWnet"], eair=inputs["Eair"])
    return PaperPointForcingCache(**values)


def read_named_csv(path):
    values = {}
    with path.open(encoding="ascii", newline="") as handle:
        for row in csv.reader(handle):
            if row[0] == "field":
                continue
            values.setdefault(row[0], []).append(float(row[1]))
    return values


def build_consumer_reference(work, artifacts):
    from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    qsat = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
    diffuco = ROOT / "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90"
    dim2 = ROOT / "fortran_source/ORCHIDEE/src_driver/dim2_driver.f90"
    intersurf = ROOT / "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90"
    qtemplate = ROOT / "scripts/dev/qsat_micro_oracle_harness.f90.template"
    header = qtemplate.read_bytes().split(b"program qsat_micro_oracle")[0]
    procedures = b"\n".join(extract_procedure_bytes(qsat, name).span_bytes for name in ("qsfrict_init", "qsatcalc", "dev_qsatcalc"))
    header = header.replace(b"! <ORIGINAL_PROCEDURES>", procedures)
    drag = """module drag
use defprec
implicit none
real(r_std),parameter::min_wind=.1,un=1.
real(r_std)::wind(4)
contains
subroutine source_wind(u,v)
real(r_std)::u(4),v(4)
""" + span(diffuco, 625, 625).decode() + "\nend subroutine\n" + span(diffuco, 3312, 3353).decode() + "\nend module\n"
    program = """program boundary
use qsat_moisture
use drag
implicit none
integer::unit,ik,i,j,sample
integer,parameter::nbindex=4
integer::ilandindex(4)=[1,2,1,2],jlandindex(4)=[1,1,2,2]
real::pb(2,2),for_psurf(2,2),u(2,2),v(2,2),for_u(2,2),for_v(2,2),zlev_vec(2,2),zlevuv_vec(2,2),z0(2,2)
real::precip_rain(2,2),precip_snow(2,2),zprecip_rain(4),zprecip_snow(4),xrdt,temperature(4),qsatt(4),qc(4),raero(4)
character(len=1024)::path
call get_command_argument(1,path)
open(newunit=unit,file=trim(path),status='replace')
"""
    samples = []

    def assignment(name, values):
        flat = np.asarray(values).reshape(-1)
        expression = ",".join(f"{v:.17e}" for v in flat)
        return f"{name}=reshape([{expression}],shape({name}))\n"

    for case in fixture["forcing"]["cases"]:
        if case["id"] not in {"standard", "watchout", "hourly_split2", "hourly_split2_netrad"}:
            continue
        steps = [1, 2, 5, 8] if case["split"] == 4 else [1, 3]
        for step in steps:
            values = case["expected"][str(step)]
            for roughness in (.1, .25):
                samples.append({"case_id": case["id"], "step": step, "roughness": roughness,
                                "dt_sechiba": case["dt_force"] / case["split"], "watchout": case["watchout"]})
                program += f"sample={len(samples)}\n"
                for name, field in {"pb": "pb", "u": "u", "v": "v", "zlev_vec": "zlev", "zlevuv_vec": "zlevuv",
                                    "precip_rain": "rainf", "precip_snow": "snowf", "temperature": "tair"}.items():
                    program += assignment(name, values[field]["values"])
                program += f"z0={roughness:.17e}\nxrdt={samples[-1]['dt_sechiba']:.17e}\n"
                program += span(dim2, 853, 853).decode() + span(dim2, 1014, 1021).decode()
                program += "do ik=1,4\ni=ilandindex(ik);j=jlandindex(ik)\n" + span(intersurf, 585, 586).decode() + "enddo\n"
                program += "call qsatcalc(4,temperature,reshape(for_psurf,[4]),qsatt)\n"
                for name, value in {"pb": "reshape(for_psurf,[4])", "u": "reshape(for_u,[4])", "v": "reshape(for_v,[4])",
                                    "precip_rain": "zprecip_rain", "precip_snow": "zprecip_snow", "qsatt": "qsatt"}.items():
                    program += f"call emit('{name}',{value})\n"
                if case["watchout"]:
                    program += assignment("qc", values["cdrag"]["values"])
                    program += "call source_wind(reshape(for_u,[4]),reshape(for_v,[4]))\ncall diffuco_raerod(4,reshape(for_u,[4]),reshape(for_v,[4]),qc,raero)\ncall emit('raero',raero)\n"
    program += """close(unit)
contains
subroutine emit(name,value)
character(len=*),intent(in)::name
real,intent(in)::value(:)
integer::point
do point=1,size(value)
 write(unit,'(I0,",",A,",",I0,",",ES26.17E3)')sample,trim(name),point,value(point)
enddo
end subroutine
end program
"""
    source = work / "forcing_consumer_phase06.f90"
    source.write_bytes(header + drag.encode() + program.encode())
    output = artifacts / "fortran_forcing_consumer.csv"
    build = compile_run(work, [source], output)
    with output.open(encoding="ascii", newline="") as handle:
        for sample, name, _, value in csv.reader(handle):
            samples[int(sample) - 1].setdefault("expected", {}).setdefault(name, []).append(float(value))
    (artifacts / source.name).write_bytes(source.read_bytes())
    return {"schema": "phase06-forcing-consumer-reference-v1", "source": [binding(qsat), binding(diffuco), binding(dim2), binding(intersurf), binding(qtemplate)],
            "build": build, "cases": samples,
            "limits": "Original pressure/wind and gathered precipitation amounts, complete qsatcalc and raerod only. Surface temperature is a declared equal-to-air component input. No physical timestep, photosynthesis, energy solve or atmospheric producer identity."}


def build_static_reference(work, artifacts, *, nearest=False):
    slow = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
    diffuco = ROOT / "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90"
    # The original bbox and nearest bodies execute on explicit raw file values.
    annual_decl = """integer::ilast,ip,iprog,idi,ii,ib,inear
integer::n_origpopden(nbpt)
real::popd(nbpt),popden_orig(ijml),mask(ijml),lon_ful(ijml),lat_ful(ijml)
real::lon_low(nbpt),lon_up(nbpt),lat_low(nbpt),lat_up(nbpt),lalo(nbpt,2)
real::domaine_lon_min,domaine_lon_max,domaine_lat_min,domaine_lat_max
logical::found
character(len=20)::field_name='salinity'
"""
    tide_decl = """integer::ilast,ip,iprog,idi,ii,ib,inear,j,t
integer::n_origlightn(nbpt)
real::proxydata(nbpt,tml),lightn_orig(ijml,tml),mask(ijml),lon_ful(ijml),lat_ful(ijml)
real::lon_low(nbpt),lon_up(nbpt),lat_low(nbpt),lat_up(nbpt),lalo(nbpt,2)
real::domaine_lon_min,domaine_lon_max,domaine_lat_min,domaine_lat_max
logical::found
character(len=20)::field_name='tide'
real::a(6)=[-1.2,-.5,-.1,0.,.2,.8],b(6)=[-2.,-.8,-.2,.1,.5,1.5]
"""
    common_geometry = """lat_ful=0.;lalo=reshape([0.,0.,0.,10.],[2,2])
lon_ful=[0.,1.,10.];mask=1.;lat_low=-.5;lat_up=.5
"""
    program = """program bbox
implicit none
integer,parameter::i_std=4,r_std=8,nbpt=2,ijml=3,tml=2920
real,parameter::pi=3.1415926535897932384626433832795
integer::numout=6,printlev_loc=1,unit
character(len=1024)::path
call get_command_argument(1,path)
open(newunit=unit,file=trim(path),status='replace')
call annual_case();call tide_case(.false.);call tide_case(.true.);close(unit)
contains
subroutine annual_case()
""" + annual_decl + common_geometry + """popden_orig=[2.,-1.,4.]
lalo=reshape([0.,0.,0.,1.],[2,2])
lon_low=[-.5,.5];lon_up=[.5,1.5]
domaine_lon_min=-.5;domaine_lon_max=1.5;domaine_lat_min=-.5;domaine_lat_max=.5
""" + span(slow, 6393, 6396).decode() + span(slow, 6445, 6552).decode() + """
call emit1('salinity',popd)
call emit1('salinity_count',real(n_origpopden))
end subroutine
subroutine tide_case(mask_case)
logical,intent(in)::mask_case
""" + tide_decl + common_geometry + """do t=1,tml
 lightn_orig(1,t)=a(min(t,6));lightn_orig(2,t)=.125;lightn_orig(3,t)=b(min(t,6))
enddo
lon_low=[-.5,9.5];lon_up=[.5,10.5]
if(mask_case)then
 lightn_orig=0.
 lightn_orig(:,1)=[2.,4.,8.];lightn_orig(:,2)=[-9100.,40.,80.]
 lightn_orig(:,3:)=1.
 lon_low=[-2.,8.];lon_up=[2.,12.]
endif
domaine_lon_min=minval(lon_low);domaine_lon_max=maxval(lon_up)
domaine_lat_min=-.5;domaine_lat_max=.5
""" + span(slow, 6166, 6279).decode() + """
if(mask_case)then
 call emit1('tide_mask_first',proxydata(:,1));call emit1('tide_mask_second',proxydata(:,2))
 call emit1('tide_mask_count',real(n_origlightn))
else
 call emit2('tide_full',proxydata)
 call emit1('tide_count',real(n_origlightn))
endif
end subroutine
""" + span(slow, 4209, 4263).decode() + """
subroutine ipslerr_p(level,routine,message,a,b)
integer,intent(in)::level
character(len=*),intent(in)::routine,message,a,b
error stop 'slowproc_nearest failure'
end subroutine
subroutine emit1(name,value)
character(len=*),intent(in)::name
real,intent(in)::value(:)
integer::i
do i=1,size(value)
 write(unit,'(A,",",ES26.17E3)')trim(name),value(i)
enddo
end subroutine
subroutine emit2(name,value)
character(len=*),intent(in)::name
real,intent(in)::value(:,:)
integer::i,j
do j=1,size(value,2)
 do i=1,size(value,1)
  write(unit,'(A,",",ES26.17E3)')trim(name),value(i,j)
 enddo
enddo
end subroutine
end program
"""
    if nearest:
        program = program.replace("lon_ful=[0.,1.,10.]", "lon_ful=[-2.,10.,5.]")
    bbox_source = work / "static_bbox_phase06.f90"
    bbox_source.write_text(program, encoding="utf-8")
    bbox_output = artifacts / "fortran_static_bbox.csv"
    build = compile_run(work, [bbox_source], bbox_output)
    bbox = read_named_csv(bbox_output)
    tide = np.asarray(bbox["tide_full"]).reshape((2, 2920), order="F")
    template = ROOT / "scripts/dev/oracle_lane_surface_mangrove.f90.template"
    original = template.read_text(encoding="utf-8")
    original = original.replace("itimetide=1006", "itimetide=2920")
    original = original.replace("! <MANGROVE_SEGMENT>", span(diffuco, 445, 623).decode())
    original = original.replace("sal=(/5.,35./)", "sal=(/2.,0./)")
    # Both target bboxes selected the complete source vectors used by the
    # original mangrove harness; compare that transport identity before use.
    if nearest:
        np.testing.assert_array_equal(tide[0, :6], [-1.2, -.5, -.1, 0., .2, .8])
        np.testing.assert_array_equal(tide[1], np.full(2920, .125))
        original = original.replace("sal=(/2.,0./)", "sal=(/2.,2./)")
        original = original.replace("tide(2,1:6)=(/-2.,-.8,-.2,.1,.5,1.5/)", "tide(2,1:6)=.125")
    else:
        np.testing.assert_array_equal(tide[:, :6], [[-1.2, -.5, -.1, 0., .2, .8], [-2., -.8, -.2, .1, .5, 1.5]])
    control_source = work / "static_controls_phase06.f90"
    control_source.write_text(original, encoding="utf-8")
    control_output = artifacts / "fortran_static_controls.csv"
    control_build = compile_run(work, [control_source], control_output)
    controls = read_named_csv(control_output)
    raw = np.empty((2920, 1, 3))
    raw[:, 0, 0] = tide[0]
    raw[:, 0, 1] = .125
    raw[:, 0, 2] = np.asarray([-2., -.8, -.2, .1, .5, 1.5])[np.minimum(np.arange(2920), 5)]
    samples = [0, 1, 2, 3, 4, 5, 999, 1005, 2919]
    sampled = {}
    for name in ("n_soil_inudate", "frac_root_inudate", "frac_root_ventilate", "frac_root_anoxia"):
        sampled[name] = np.asarray(controls[name]).reshape((2, 2920), order="F")[:, samples].tolist()
    for path in (bbox_source, control_source):
        (artifacts / path.name).write_bytes(path.read_bytes())
    return {
        "schema": "phase06-static-control-reference-v1", "source": [binding(slow), binding(diffuco), binding(template)],
        "builds": [build, control_build], "itimetide": 2920,
        "nearest_fallback_case": nearest,
        "salinity": {"raw": [[[2., -1., 4.]]], "lon": [-2., 10., 5.] if nearest else [0., 1., 10.], "lat": [0.],
                     "lalo": [[0., 0.], [0., 1.]], "bbox_width_degrees": 1.,
                     "expected": bbox["salinity"], "counts": bbox["salinity_count"]},
        "tide": {"raw": raw.tolist(), "lon": [-2., 10., 5.] if nearest else [0., 1., 10.], "lat": [0.],
                 "lalo": [[0., 0.], [0., 10.]], "bbox_width_degrees": 1.,
                 "expected": tide.tolist(), "counts": bbox["tide_count"]},
        "tide_mask": {"lalo": [[0., 0.], [0., 10.]], "bbox_width_degrees": 4.,
                      "first": bbox["tide_mask_first"], "second": bbox["tide_mask_second"], "counts": bbox["tide_mask_count"]},
        "controls": {"control_salinity": controls["control_salinity"], "control_inudate": controls["control_inudate"],
                     "agb_st": controls["agb_st"], "agb_pn": controls["agb_pn"],
                     "h_st": controls["h_st"], "h_pn": controls["h_pn"],
                     "sample_indices": samples, "samples": sampled,
                     "rprof_pft14": [.35, .8], "diaglev": [.1, .4, 1.],
                     "min": .5, "agb_agr_ven_all_st": 200., "agb_agr_ven_all_pn": 40., "h_agr_max_st": 2., "h_agr_max_pn": .4},
        "limits": "Original slowproc bbox/nearest/preprocessing and original DIFFUCO firstcall/control segment only. ITIMETIDE2920 explicit, no astronomical tide/historical producer identity or whole C62 replay.",
    }


def actual_forcing(fixture, case, model_tstep):
    from jax_orchidee.driver.domain import _forcing_model_step_from_cached_point

    return _forcing_model_step_from_cached_point(
        cache_from_fixture(fixture, case), model_tstep=model_tstep, split=case["split"], nb_spread=case["spread"],
        dt_force=case["dt_force"], netrad_cons=case["netrad_cons"],
        lon=np.array([[-30., 30.], [0., 60.]]), lat=np.array([[-20., 30.], [10., 60.]]),
        land_indices_key=((0, 0), (1, 0), (0, 1), (1, 1)),
    )


def verify(output):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    results = []
    for case in fixture["forcing"]["cases"]:
        for step, expected in case["expected"].items():
            actual = actual_forcing(fixture, case, int(step) - 1)
            for field, attr in {"tair": "temp_air", "pb": "pb", "qair": "qair", "u": "u", "v": "v",
                                "rainf": "precip_rain", "snowf": "precip_snow", "swdown": "swdown", "lwdown": "lwdown"}.items():
                value = np.asarray(getattr(actual, attr)).reshape(-1)
                reference = np.asarray(expected[field]["values"])
                results.append({"case": case["id"], "step": int(step), "field": field,
                                "passed": bool(np.allclose(value, reference, rtol=1e-12, atol=1e-14)),
                                "actual": value.tolist(), "expected": reference.tolist()})
    receipt = {"schema": "phase06-forcing-current-comparison-v1", "comparisons": results,
               "failed": sum(not row["passed"] for row in results),
               "production": binding(ROOT / "jax_orchidee/driver/domain.py"),
               "fixture": binding(FIXTURE), "model_steps_or_days_executed": 0,
               "limits": "Real host interpolation component; no full model execution. Source phase counted as forcing substeps only."}
    write_json(output / "forcing-comparison.json", receipt)
    print(json.dumps({"comparisons": len(results), "failed": receipt["failed"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--build-static-reference", action="store_true")
    parser.add_argument("--build-consumer-reference", action="store_true")
    parser.add_argument("--build-nearest-reference", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts/dev"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.build_reference:
        if FIXTURE.exists():
            raise FileExistsError("Frozen fixture exists; use a new version rather than overwriting reference")
        artifacts = args.output / "reference"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="orchidee_phase06_forcing_modelout_") as raw:
            work = Path(raw)
            forcing = build_forcing(work, artifacts)
            modelout = build_modelout(work, artifacts)
        write_json(FIXTURE, {"schema": "phase06-forcing-modelout-reference-v1", "forcing": forcing, "modelout": modelout})
    if args.build_static_reference:
        if STATIC_FIXTURE.exists():
            raise FileExistsError("Frozen static fixture exists; use a new version")
        artifacts = args.output / "reference"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="orchidee_phase06_static_") as raw:
            static = build_static_reference(Path(raw), artifacts)
        write_json(STATIC_FIXTURE, static)
    if args.build_consumer_reference:
        if CONSUMER_FIXTURE.exists():
            raise FileExistsError("Frozen consumer fixture exists; use a new version")
        artifacts = args.output / "reference"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="orchidee_phase06_consumer_") as raw:
            consumer = build_consumer_reference(Path(raw), artifacts)
        write_json(CONSUMER_FIXTURE, consumer)
    if args.build_nearest_reference:
        if NEAREST_FIXTURE.exists():
            raise FileExistsError("Frozen nearest fixture exists; use a new version")
        artifacts = args.output / "reference/nearest"
        artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="orchidee_phase06_nearest_") as raw:
            nearest = build_static_reference(Path(raw), artifacts, nearest=True)
        write_json(NEAREST_FIXTURE, nearest)
    verify(args.output)


if __name__ == "__main__":
    main()
