"""Bounded current-source snow maxmass/domain reference (not a trajectory).

Build extracts the unchanged current snow procedures, instruments read-only
phase snapshots, and adds an exact main325-372/386-402 fragment wrapper.
Verify consumes the saved original-Fortran outputs without recompiling them.
An existing build or verification destination is always rejected untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER, compile_fortran, compiler_environment, float_comparison,
)
from scripts.dev.oracle_lane_hydrol_explicit_snow import (  # noqa: E402
    SNOW_SOURCE, QSAT_SOURCE, TEMPLATE, _compose,
)

LF = 2.8345e6 - 2.5008e6
TP = 273.15
RTOL, ATOL = 1e-12, 1e-14


def cases() -> list[dict]:
    """Every case declares its input domain; none is a natural-paper trigger."""
    result = []

    def helper(name, masses, removal, expected_loc, *, rho=400., liquid=False):
        m = np.array(masses, dtype=float)
        k = 3. if m.sum() > 3600. else 1.
        result.append(dict(
            id=name, mode="helper", domain="mathematical_positive_density_maxmass_domain",
            rho=[rho] * 3, dz=(m / rho).tolist(), temperature=[268., 270., TP],
            liquid=([.001, .001, .001] if liquid else [0., 0., 0.]),
            soilcap=removal * LF / k, expected_loc=expected_loc,
            vevapsno=0., snowfall=0., profile_temperature=268.,
        ))

    helper("below_mass_threshold", [2800., 100., 99.], 50., 0)
    helper("mass_threshold_equal", [2800., 100., 100.], 50., 0)
    helper("moderate_zero_capacity", [3200., 200., 100.], 0., 3)
    helper("moderate_bottom_cap", [3200., 200., 100.], 50., 3)
    helper("bottom_interface_equal", [3200., 200., 100.], 100., 2)
    helper("middle_positive_removal", [3200., 200., 100.], 200., 2)
    helper("middle_interface_equal", [3200., 200., 100.], 300., 1)
    helper("top_positive_removal", [3200., 200., 100.], 400., 1)
    helper("excess_limited", [3200., 200., 100.], 1000., 1)
    helper("upper_mass_below_3000", [2799., 200., 100.], 1000., 3)
    helper("upper_mass_equal_3000", [2800., 200., 100.], 1000., 2)
    helper("upper_mass_above_3000", [2801., 200., 100.], 1000., 2)
    helper("empty_bottom_equal_zero", [3200., 200., 0.], 0., 2)
    helper("two_empty_bottom_layers_equal_zero", [3200., 0., 0.], 0., 1)
    helper("wet_cross_layer_return_heat", [3200., 200., 100.], 400., 1, liquid=True)
    helper("extreme_factor_threshold_equal", [3200., 200., 200.], 1., 3)
    # Same capacity as previous case: source >3600 switches factor1 to factor3.
    helper("extreme_factor_threshold_above", [3200., 200., np.nextafter(200., np.inf)], 3., 3)
    result[-1]["dz"][-1] = (np.nextafter(3600., np.inf) - 3400.) / 400.
    result[-1]["soilcap"] = LF
    assert sum(np.array(result[-1]["rho"]) * result[-1]["dz"]) > 3600.
    result.append(dict(
        id="deep_regular_remap", mode="path",
        domain="source_consumable_component_restart_and_inputs_not_natural_path_proof",
        rho=[300.] * 3, dz=[.05, .5, 10.], temperature=[268.] * 3,
        liquid=[0.] * 3, soilcap=2e6, vevapsno=0., snowfall=0., profile_temperature=268.,
    ))
    result.append(dict(
        id="zero_layer_deep_uniform_transfer_crossing", mode="path",
        domain="source_consumable_ordinary_density_high_capacity_not_natural_path_proof",
        rho=[600.] * 3, dz=[0., 5., 5.], temperature=[268.] * 3,
        liquid=[0.] * 3, soilcap=800. * LF, vevapsno=0., snowfall=0., profile_temperature=268.,
    ))
    result.append(dict(
        id="negative_sublimation_crossing", mode="path",
        domain="source_consumable_ordinary_density_extreme_deposition_not_natural_path_proof",
        rho=[300.] * 3, dz=[.01] * 3, temperature=[268.] * 3,
        liquid=[0.] * 3, soilcap=2e6, vevapsno=-3005., snowfall=0., profile_temperature=268.,
    ))
    result.append(dict(
        id="snowfall_warm_phase_and_drainage", mode="path",
        domain="bounded_synthetic_component_path_not_natural_path_proof",
        rho=[300.] * 3, dz=[.05, .5, 10.], temperature=[TP] * 3,
        liquid=[0.] * 3, soilcap=2e6, vevapsno=.2, snowfall=1., profile_temperature=TP + 8.,
    ))
    return result


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real(value):
    return f"{float(value):.17e}_r_std"


def _vector(values):
    return "(/" + ",".join(_real(v) for v in values) + "/)"


CAPTURE = r'''
 subroutine capture_phase(phase,kjpindex,snowrho,snowdz,snowtemp,snowliq,snowheat,snowgrain,grndflux)
 character(*),intent(in)::phase
 integer,intent(in)::kjpindex
 real(r_std),intent(in)::snowrho(kjpindex,nsnow),snowdz(kjpindex,nsnow),snowtemp(kjpindex,nsnow),snowliq(kjpindex,nsnow),snowheat(kjpindex,nsnow),snowgrain(kjpindex,nsnow),grndflux(kjpindex)
 integer::j
 call emit(phase,'mass',0,sum(snowrho(1,:)*snowdz(1,:)))
 call emit(phase,'grndflux',0,grndflux(1))
 do j=1,nsnow
 call emit(phase,'snowrho',j,snowrho(1,j));call emit(phase,'snowdz',j,snowdz(1,j))
 call emit(phase,'snowtemp',j,snowtemp(1,j));call emit(phase,'snowliq',j,snowliq(1,j))
 call emit(phase,'snowheat',j,snowheat(1,j));call emit(phase,'snowgrain',j,snowgrain(1,j))
 enddo
 end subroutine
 subroutine emit(phase,field,j,value)
 character(*),intent(in)::phase,field
 integer,intent(in)::j
 real(r_std),intent(in)::value
 write(10,'(i0,a,a,a,a,a,i0,a,es25.17)')capture_case,',',phase,',',field,',',j,',',value
 end subroutine
'''


def _source(output):
    base = output / "source-composition.f90"
    spans = _compose(base)
    module = base.read_text(encoding="ascii").split("program explicit_snow_oracle", 1)[0]
    template = TEMPLATE.read_text(encoding="ascii")
    assert "chalfu0=chalsu0-chalev0" in template
    assert "chalsu0=2.8345e6_r_std,chalev0=2.5008e6_r_std" in template
    module = module.replace("integer::printlev=0", "integer::printlev=0,capture_case=0", 1)
    boundaries = {
        "fall": "    ! 2.2 calculate the new snow discretization",
        "transfer": "    ! 2.4 Diagonize water portion",
        "compact": "    !2.6 Calculate the snow temperature profile",
        "profile": "    !2.7 Test whether snow is existed",
        "gone": "    !2.8 Calculate snow melt/refreezing processes",
        "melt": "    ! 2.9 Snow sublimation changing snow thickness",
        "sublimation": "! 2.9.1 Snowmelt_from_maxmass",
        "maxmass": "    !2.10 Calculate snow grain size",
        "return_heat": "    !3. on land ice",
    }
    for phase, marker in boundaries.items():
        assert module.count(marker) == 1, (phase, marker)
        hook = f"    CALL capture_phase('{phase}',kjpindex,snowrho,snowdz,snowtemp,snowliq,snowheat,snowgrain,grndflux)\n"
        module = module.replace(marker, hook + marker, 1)
    lines = SNOW_SOURCE.read_text(encoding="ascii").splitlines(keepends=True)
    fragments = "".join(lines[324:372]) + "".join(lines[385:402])
    assert "DO ji=1,kjpindex" in fragments and "snow3lheat_1d" in fragments
    wrapper = r'''
 subroutine maxmass_fragment(soilcap,snowrho,snowdz,snowtemp,snowliq,snowheat,snowgrain,grndflux,snow,removal,location)
 integer,parameter::kjpindex=1
 real(r_std),intent(in)::soilcap(kjpindex),snowrho(kjpindex,nsnow),snowtemp(kjpindex,nsnow),snowgrain(kjpindex,nsnow)
 real(r_std),intent(inout)::snowdz(kjpindex,nsnow),snowliq(kjpindex,nsnow),snowheat(kjpindex,nsnow),grndflux(kjpindex)
 real(r_std),intent(out)::snow(kjpindex),removal(kjpindex)
 integer,intent(out)::location
 integer::ji,jj,locjj
 real(r_std)::snow_d1k,snowmelt_from_maxmass(kjpindex),wsnowdz(kjpindex,nsnow),smass(kjpindex,nsnow),smassc(kjpindex),snow_remove(kjpindex),zsnowmelt_xs(kjpindex),zsnowdz(kjpindex),zliqheatxs(kjpindex)
 snowmelt_from_maxmass=0._r_std;locjj=0
'''
    wrapper += fragments + "\n removal=snowmelt_from_maxmass;location=locjj\n end subroutine\n"
    module = module.rsplit("end module", 1)[0] + CAPTURE + wrapper + "end module\n"
    program = r'''
program snow_domain_oracle
 use defprec;use explicit_snow_lane;implicit none
 integer,parameter::n=1
 integer::loc,j
 real(r_std)::rain(n),fall(n),tair(n),pb(n),u(n),v(n),ts(n),cap(n),pg(n),frac(n,nnobio),totnb(n),gtemp(n),lambda(n),cs(n,nsnow),ds(n,nsnow)
 real(r_std)::evap(n),age(n),nbage(n,nnobio),nb(n,nnobio),rho(n,nsnow),grain(n,nsnow),dz(n,nsnow),st(n,nsnow),heat(n,nsnow),snow(n),tadd(n),liq(n,nsnow),subnb(n,nnobio),gflux(n),melt(n),totm(n),resid(n),sink(n),removal(n)
 character(1024)::path
 call get_command_argument(1,path);open(10,file=trim(path),status='new');write(10,'(a)')'case,phase,field,j,value'
'''
    data = cases()
    for i, c in enumerate(data):
        program += f"\n capture_case={i}\n"
        program += " rain=0;fall=0;tair=268;pb=900;u=1;v=0;ts=268;cap=2.e6;pg=0;frac=0;totnb=0;gtemp=268;lambda=0;cs=268;ds=0\n"
        program += " evap=0;age=0;nbage=0;nb=0;grain=1.e-4;tadd=0;subnb=0;gflux=0;melt=0;totm=0;resid=0;sink=0\n"
        for name, key in (("rho", "rho"), ("dz", "dz"), ("st", "temperature"), ("liq", "liquid")):
            program += f" {name}(1,:)={_vector(c[key])}\n"
        program += f" cap={_real(c['soilcap'])};evap={_real(c['vevapsno'])};fall={_real(c['snowfall'])}\n"
        program += f" ts={_real(c['profile_temperature'])};cs={_real(c['profile_temperature'])}\n"
        program += " snow=sum(rho*dz,2);heat(1,:)=snow3lheat_1d(liq(1,:),rho(1,:),dz(1,:),st(1,:))\n"
        program += " call capture_phase('entry',n,rho,dz,st,liq,heat,grain,gflux)\n"
        if c["mode"] == "helper":
            program += " call maxmass_fragment(cap,rho,dz,st,liq,heat,grain,gflux,snow,removal,loc)\n"
            program += " call emit('helper','removal',0,removal(1));call emit('helper','location',0,real(loc,r_std))\n"
            program += " call capture_phase('helper',n,rho,dz,st,liq,heat,grain,gflux)\n"
        else:
            program += " call explicitsnow_main(n,rain,fall,tair,pb,u,v,ts,cap,pg,frac,totnb,gtemp,lambda,cs,ds,evap,age,nbage,nb,rho,grain,dz,st,heat,snow,tadd,liq,subnb,gflux,melt,totm,resid,sink)\n"
            program += " call capture_phase('return',n,rho,dz,st,liq,heat,grain,gflux)\n"
            for field, name in (("snow", "snow"), ("snowmelt", "melt"), ("tot_melt", "totm"), ("vevapsno", "evap"), ("subsinksoil", "sink"), ("temp_sol_add", "tadd")):
                program += f" call emit('return','{field}',0,{name}(1))\n"
    program += "close(10)\nend program\n"
    target = output / "oracle.f90"
    target.write_text(module + program, encoding="ascii")
    return spans


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        _dump(output / "inputs.json", cases())
        spans = _source(output)
        metadata = compile_fortran(output / "oracle.f90", output / "oracle.exe")
        _dump(output / "compile.json", metadata)
        result = subprocess.run(
            [str((output / "oracle.exe").resolve()), str((output / "fortran.csv").resolve())],
            cwd=output, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True,
        )
        _dump(output / "execution.json", dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        result.check_returncode()
        _dump(output / "identity.json", dict(
            baseline=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_files={str(p.relative_to(ROOT)): _sha(p) for p in (SNOW_SOURCE, QSAT_SOURCE, TEMPLATE, Path(__file__))},
            extracted_procedures=spans,
            exact_fragments={"maxmass": [325, 372], "return_heat": [386, 402]},
            instrumentation="Read-only phase captures; original procedure arithmetic unchanged. Wrapper initializes locals and returns location; original inactive location is undefined and represented by0 in wrapper only.",
            snow_chalfu0=LF, soil_lhf=333600., old_oracle="not overwritten or regenerated",
            graph="Verify exact-source fallback; parent list_projects Transport closed, generation/coverage unavailable",
        ))
        return dict(status="fortran_passed", cases=len(cases()), output=str(output))
    except Exception as exc:
        _dump(output / "failure.json", dict(type=type(exc).__name__, message=str(exc), stderr=getattr(exc, "stderr", None)))
        raise


def _read(output):
    records = {}
    with (output / "fortran.csv").open(encoding="ascii", newline="") as stream:
        for r in csv.DictReader(stream):
            key = (int(r["case"]), r["phase"], r["field"])
            records.setdefault(key, []).append(float(r["value"]))
    return {key: np.array(value) for key, value in records.items()}


def _path_inputs(c):
    z = lambda: np.zeros(1)
    lay = lambda v: np.full((1, 3), v)
    rho, dz, temp, liq = (np.array([c[k]]) for k in ("rho", "dz", "temperature", "liquid"))
    # Match the source equation's declared order (qsat_moisture823-824).
    heat = dz * (rho * 2106. * (temp - TP) - LF * rho) + LF * 1000. * liq
    return dict(
        precip_rain=z(), precip_snow=np.array([c["snowfall"]]), temp_air=np.array([268.]),
        pb=np.array([900.]), u=np.ones(1), v=z(), temp_sol_new=np.array([c["profile_temperature"]]),
        soilcap=np.array([c["soilcap"]]), pgflux=z(), frac_nobio=np.zeros((1, 1)), totfrac_nobio=z(),
        gtemp=np.array([268.]), lambda_snow=z(), cgrnd_snow=lay(c["profile_temperature"]), dgrnd_snow=lay(0.),
        vevapsno=np.array([c["vevapsno"]]), snow_age=z(), snow_nobio_age=np.zeros((1, 1)),
        snow_nobio=np.zeros((1, 1)), snowrho=rho, snowgrain=lay(1e-4), snowdz=dz,
        snowtemp=temp, snowheat=heat, snow=np.sum(rho*dz, axis=1), temp_sol_add=z(),
        snowliq=liq, subsnownobio=np.zeros((1, 1)), grndflux=z(), snowmelt=z(), soilflxresid=z(),
    )


def make_path_inputs(case_id="deep_regular_remap"):
    """Reusable exact complete snow-owner inputs for adjacent consumer checks."""
    return _path_inputs(next(c for c in cases() if c["id"] == case_id and c["mode"] == "path"))


def path_inputs_from_reference(reference_dir, case_id="deep_regular_remap"):
    """Use exact emitted source entry heat, including its floating-point order."""
    output = Path(reference_dir)
    saved = json.loads((output / "inputs.json").read_text(encoding="utf-8"))
    i, c = next((i, c) for i, c in enumerate(saved) if c["id"] == case_id and c["mode"] == "path")
    values = _path_inputs(c)
    values["snowheat"] = _read(output)[(i, "entry", "snowheat")].reshape(1, 3)
    return values


def snow_owner_arrays(values):
    """Expose actual hydrology snow-owner arrays without provenance strings."""
    from jax_orchidee.sechiba import hydrol
    owner = hydrol.hydrol_explicit_snow_step(**values)
    fields = ("snowrho", "snowdz", "snowtemp", "snowliq", "snowheat", "snowgrain", "grndflux", "snow", "snowmelt", "tot_melt", "vevapsno", "subsinksoil", "temp_sol_add")
    return {field: getattr(owner.snow_state if field in owner.snow_state._fields else owner, field) for field in fields}


def _path_phases(c, snow, inputs=None):
    """Capture actual production helpers in main order; final result is real adapter."""
    a = dict(_path_inputs(c) if inputs is None else inputs)
    phases = {}
    def capture(name):
        phases[name] = {k: np.asarray(a[k]).ravel() for k in (
            "snowrho", "snowdz", "snowtemp", "snowliq", "snowheat", "snowgrain", "grndflux")}
        phases[name]["mass"] = np.sum(np.asarray(a["snowrho"]) * np.asarray(a["snowdz"]), axis=1)
    def apply(function, keys, updated, **kwargs):
        r = function(**{k: a[k] for k in keys.split()}, **kwargs)
        for k in updated.split():
            a[k] = getattr(r, k)
        return r
    capture("entry")
    apply(snow.explicitsnow_fall_step, "precip_snow temp_air u v totfrac_nobio snowrho snowdz snowheat snowgrain snowtemp", "snowrho snowdz snowheat snowgrain")
    capture("fall")
    old = a["snowdz"]
    a["snowdz"] = snow.explicitsnow_levels_step(np.sum(np.asarray(old), axis=1)).snowdz
    apply(snow.explicitsnow_transf_step, "snowdz snowrho snowheat snowgrain", "snowrho snowdz snowheat snowgrain", snowdz_old=old)
    capture("transfer")
    a["snowtemp"] = snow.snow3ltemp_explicit(a["snowheat"], a["snowrho"], a["snowdz"])
    a["snowliq"] = snow.snow3lliq_explicit(a["snowheat"], a["snowrho"], a["snowdz"], a["snowtemp"])
    apply(snow.explicitsnow_compactn_step, "snowtemp snowrho snowdz", "snowrho snowdz")
    a["snowheat"] = snow.snow3lheat_explicit(a["snowliq"], a["snowrho"], a["snowdz"], a["snowtemp"])
    capture("compact")
    apply(snow.explicitsnow_profile_step, "cgrnd_snow dgrnd_snow lambda_snow temp_sol_new snowtemp snowdz temp_sol_add", "snowtemp temp_sol_add")
    capture("profile")
    apply(snow.explicitsnow_gone_step, "pgflux snowheat snowtemp snowdz snowrho snowliq grndflux snowmelt soilflxresid", "snowtemp snowdz snowrho snowliq grndflux snowmelt")
    capture("gone")
    apply(snow.explicitsnow_melt_refrz_step, "precip_rain pgflux soilcap snowtemp snowdz snowrho snowliq snowmelt grndflux temp_air soilflxresid", "snowtemp snowdz snowrho snowliq snowmelt grndflux")
    capture("melt")
    apply(snow.explicitsnow_sublimation_step, "vevapsno frac_nobio totfrac_nobio snowrho snowdz snowliq snowtemp", "snow snowdz snowliq snowtemp vevapsno subsnownobio subsinksoil")
    capture("sublimation")
    maxmass = apply(snow.explicitsnow_maxmass_step, "soilcap snowrho snowdz", "snow snowdz")
    capture("maxmass")
    apply(snow.explicitsnow_grain_step, "snowliq snowdz gtemp snowtemp pb snowgrain", "snowgrain")
    apply(snow.explicitsnow_liquid_heat_excess_step, "snowliq snowdz snowrho grndflux", "snowliq grndflux")
    a["snowheat"] = snow.snow3lheat_explicit(a["snowliq"], a["snowrho"], a["snowdz"], a["snowtemp"])
    capture("return_heat")
    return phases, maxmass


def verify(output, result_dir, *, jit_main=False):
    result_dir.mkdir(parents=True, exist_ok=False)
    import jax
    jax.config.update("jax_enable_x64", True)
    from jax_orchidee.sechiba import hydrol as snow
    source = _read(output)
    comparisons, invariants, paths = [], [], []
    all_cases = json.loads((output / "inputs.json").read_text(encoding="utf-8"))
    helper = [c for c in all_cases if c["mode"] == "helper"]
    rho = np.array([c["rho"] for c in helper]); dz = np.array([c["dz"] for c in helper])
    cap = np.array([c["soilcap"] for c in helper])
    temperature = np.array([c["temperature"] for c in helper]); liquid = np.array([c["liquid"] for c in helper])
    for compiled in (False, True):
        def component(c, r, d, t, q):
            m = snow.explicitsnow_maxmass_step(soilcap=c, snowrho=r, snowdz=d)
            h = snow.explicitsnow_liquid_heat_excess_step(snowliq=q, snowdz=m.snowdz, snowrho=r, grndflux=np.zeros(c.shape))
            heat = snow.snow3lheat_explicit(h.snowliq, r, m.snowdz, t)
            return m.snow, m.snowdz, m.snowmelt_from_maxmass, h.snowliq, h.grndflux, heat
        actual = (jax.jit(component) if compiled else component)(cap, rho, dz, temperature, liquid)
        actual = [np.asarray(v) for v in actual]
        fields = ("mass", "snowdz", "removal", "snowliq", "grndflux", "snowheat")
        for i, c in enumerate(helper):
            for field, value in zip(fields, actual):
                comparisons.append(float_comparison(f"{c['id']}/{field}/{'jit' if compiled else 'eager'}", source[(i, "helper", field)], np.atleast_1d(value[i]), rtol=RTOL, atol=ATOL))
            if not compiled:
                mass0 = float(np.sum(rho[i] * dz[i])); mass1 = float(actual[0][i]); removal = float(actual[2][i])
                loc = int(source[(i, "helper", "location")][0])
                invariants.append(dict(case=c["id"], source_loc=loc, expected_loc=c["expected_loc"], mass_before=mass0, mass_after=mass1, removal=removal,
                    mass_balance_residual=mass0-mass1-removal, finite=all(np.all(np.isfinite(v[i])) for v in actual),
                    nonnegative_depth=bool(np.all(actual[1][i] >= 0.)), passed=bool(loc == c["expected_loc"] and abs(mass0-mass1-removal) < 1e-10 and all(np.all(np.isfinite(v[i])) for v in actual))))
    adapter = snow_owner_arrays
    compiled_adapter = jax.jit(adapter) if jit_main else None
    for i, c in enumerate(all_cases):
        if c["mode"] != "path":
            continue
        inputs = path_inputs_from_reference(output, c["id"])
        phases, maxmass = _path_phases(c, snow, inputs)
        for phase, fields in phases.items():
            for field, values in fields.items():
                comparisons.append(float_comparison(f"{c['id']}/{phase}/{field}/helpers", source[(i, phase, field)], values, rtol=RTOL, atol=ATOL))
        result = adapter(inputs)
        for compiled in ([False, True] if jit_main else [False]):
            if compiled:
                # Compile one complete snow component; no model timestep is called.
                result = compiled_adapter(inputs)
            for (case_idx, phase, field), expected in source.items():
                if case_idx != i or phase != "return":
                    continue
                field_name = "snow" if field == "mass" else field
                comparisons.append(float_comparison(f"{c['id']}/return/{field}/{'whole_snow_jit' if compiled else 'actual_adapter'}", expected, np.asarray(result[field_name]).ravel(), rtol=RTOL, atol=ATOL))
        pre = phases["sublimation"]
        masses = pre["snowrho"] * pre["snowdz"]
        r = float(np.asarray(maxmass.snowmelt_from_maxmass)[0])
        location = 3
        if masses[2] <= r <= masses[2]+masses[1]: location = 2
        if masses[2]+masses[1] <= r <= masses.sum(): location = 1
        paths.append(dict(case=c["id"], domain=c["domain"], upper_mass_before_maxmass=float(masses[:2].sum()), bottom_mass_before_maxmass=float(masses[2]), removal=r, source_condition_location=location,
            transfer_depth=phases["transfer"]["snowdz"].tolist(), transfer_density=phases["transfer"]["snowrho"].tolist(), phase_masses={p: float(v["mass"][0]) for p,v in phases.items()}))
    passed = all(c["passed"] for c in comparisons) and all(c["passed"] for c in invariants)
    _dump(result_dir / "jax-snapshots.json", paths)
    _dump(result_dir / "verification.json", dict(status="passed" if passed else "failed", rtol=RTOL, atol=ATOL,
        helper_cases=len(helper), path_cases=len(paths), comparisons=comparisons, invariants=invariants,
        full_snow_jit=jit_main, paths=paths, jax_version=jax.__version__, backend=jax.default_backend(), x64=jax.config.x64_enabled,
        verifier_sha256=_sha(Path(__file__)), production_hydrol_sha256=_sha(ROOT / "jax_orchidee/sechiba/hydrol.py"),
        entry_heat="Exact emitted original-source entry heat consumed by both helper sequence and actual adapter; no algebraically reordered NumPy initialization.",
        model_steps=0, corrected_source_oracle=True, old_oracle_regenerated=False,
        numerical_acceptance=False, natural_reachability="domain-unproven"))
    return dict(status="passed" if passed else "failed", comparisons=len(comparisons), failed=[v["name"] for v in comparisons if not v["passed"]], output=str(result_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jit-main", action="store_true")
    parser.add_argument("--verification-name", default="verification01")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode in ("build", "run"):
        print(json.dumps(build(output), indent=2))
    if args.mode in ("verify", "run"):
        result_dir = output / args.verification_name
        if result_dir.parent != output or result_dir.name in ("", ".", ".."):
            raise ValueError("verification-name must be one directory name")
        try:
            result = verify(output, result_dir, jit_main=args.jit_main)
        except FileExistsError:
            raise
        except Exception:
            _dump(result_dir / "failure.json", dict(traceback=traceback.format_exc()))
            raise
        print(json.dumps(result, indent=2))
        return result["status"] != "passed"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
