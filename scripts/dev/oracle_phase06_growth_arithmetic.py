"""New complete growth tuples and read-only original-Fortran arm observations.

Fortran truth: stomate_alloc146-834, stomate_turnover169-947 and
stomate_vmax105-363, composed in original stomate_lpj caller order. The
Phase05 composer supplies extraction/glue only; its six old cases are not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_phase05_growth_conditions as base  # noqa: E402
from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
)

PACKAGE = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/growth-arithmetic"
FIXTURE = ROOT / "tests/fixtures/phase06_growth_arithmetic_reference.json"
FLAGS = ("-std=legacy", "-fdefault-real-8", "-fdefault-double-8", "-ffree-line-length-none", "-O0", "-fcheck=all", "-ffpe-trap=invalid,zero,overflow", "-Wall", "-Wextra")


def inputs(case):
    """Source PFT14/MTC2 parameters, independently declared new finite states."""
    x = base.chain_inputs()
    pr, al, po = (x[k] for k in ("prescribe_inputs", "alloc_inputs", "post_npp_inputs"))
    pr["veget_max"][:, 13] = 1.0
    pr["pft_present"][:, 13] = True
    pr["everywhere"][:, 13] = 1.0
    pr["when_growthinit"][:, 13] = 100.0
    pr["biomass"][:] = 0.0
    pr["biomass"][:, 13, 5, 0] = 60.0
    pr["biomass"][:, 13, 7, 0] = 100.0
    pr["leaf_frac"][:, 13] = 0.25
    al["senescence"][:] = False
    al["r0"][:] = 0.30
    al["s0"][:] = 0.25
    po["senescence"][:] = False
    po["leafagecrit"][:] = 730.0
    po["vmax_leafagecrit"][:] = 730.0
    po["leaf_timecst"][:] = 182.5
    po["resp_maint_part"][:] = 0.0
    po["gpp_daily"][:, 13] = 10.0
    x["allocation_kwargs"]["max_l_to_lsr"] = 0.6
    if case in {"environment_partition", "partition_lower", "water_limited"}:
        lai = np.asarray([0.2, 1.0, 4.5, 8.0, 10.0, 15.0])
        temperature = np.asarray([220.0, 250.0, 283.15, 298.15, 318.15, 283.15])
        humidity = np.asarray([0.0, 0.49, 0.75, 1.0, 1.0, 1.0])
        water = np.asarray([0.0, 0.1, 0.5, 1.0, 1.0, 0.7])
        ages = np.asarray([0.0, 2.0, 100.0, 365.0, 730.0, 6000.0])
        if case == "partition_lower":
            al["r0"][:] = 0.40
        if case == "water_limited":
            water = np.asarray([0.0, 0.1, 0.2, 0.25, 0.2, 0.7])
    elif case in {"age_efficiency", "partition_upper", "source_default_upper", "fraction_floor"}:
        lai = np.asarray([0.0, 0.4, 2.0, 4.0, 6.0, 8.0])
        temperature = np.full(6, 298.15)
        humidity = np.full(6, 1.0)
        water = np.full(6, 1.0)
        ages = np.asarray([0.0, 0.0, 100.0, 500.0, 900.0, 6000.0])
        pr["biomass"][0, 13] = 0.0
        pr["leaf_frac"][0, 13] = 0.0
        po["gpp_daily"][0, 13] = 0.0
        if case in {"partition_upper", "source_default_upper"}:
            al["r0"][:] = 0.10
            al["s0"][:] = 0.10
        if case == "source_default_upper":
            x["allocation_kwargs"].pop("max_l_to_lsr")
        if case == "fraction_floor":
            po["leafagecrit"][:] = 2.0
            po["vmax_leafagecrit"][:] = 2.0
            po["leaf_timecst"][:] = 0.5
    else:
        raise ValueError(case)
    pr["biomass"][:, 13, 0, 0] = lai / 0.02
    al["lai"][:, 13] = lai
    al["leaf_age"][:, 13] = ages[:, None]
    x["phenology_inputs"]["leaf_age"] = al["leaf_age"].copy()
    al["tsoil_month"][:] = temperature[:, None]
    al["soilhum_month"][:] = humidity[:, None]
    al["moiavail_week"][:, 13] = water
    po["moiavail_week"][:, 13] = water
    return x


OBSERVER = """
module p06_observer
implicit none
integer::obsunit
contains
subroutine obsopen(filename)
character(*),intent(in)::filename
open(newunit=obsunit,file=trim(filename),status='replace')
write(obsunit,'(A)')'field,value'
end subroutine
subroutine obsclose()
close(obsunit)
end subroutine
subroutine observe(name,x,leafclass,valid)
character(*),intent(in)::name
real(kind(1d0)),intent(in)::x(:)
integer,intent(in),optional::leafclass
logical,intent(in),optional::valid(:)
integer::i
character(len=32)::suffix
suffix=''
if(present(leafclass))write(suffix,'(I0)')leafclass
do i=1,size(x)
if(present(valid))then
 if(.not.valid(i))then
  write(obsunit,'(A,",",ES26.17)')trim(name)//trim(suffix),0d0
  cycle
 endif
endif
write(obsunit,'(A,",",ES26.17)')trim(name)//trim(suffix),x(i)
enddo
end subroutine
end module
"""


def observed_source():
    """Insert read-only vector writes outside WHERE; no physics assignment changes."""
    source = base.compose_chain_source().replace("\r", "")
    source = source.replace("R0=.35d0;S0=.35d0", "R0=.30d0;S0=.25d0")
    source = source.replace("leafagecrit=100d0", "leafagecrit=730d0")
    source = source.replace("leaf_timecst=100d0", "leaf_timecst=182.5d0")
    for module in ("p05_allocation_alloc_oracle", "p05_turnover_turnover_oracle", "p05_vmax_stomate_vmax_oracle"):
        source = source.replace("module " + module + "\n", "module " + module + "\nuse p06_observer,only:observe\n", 1)
    fields = ["lai_around(:,j)", "moiavail_week(:,j)", "h_nitrogen", "t_nitrogen", "limit_L", "limit_W", "limit_N_hum", "limit_N_temp", "limit_N", "limit_WorN", "limit", "RtoLSR", "StoLSR", "LtoLSR"]
    calls = "if(j==14)then\ncall observe('alloc.valid',merge(1d0,0d0,biomass(:,j,ileaf,icarbon)>min_stomate))\ncall observe('alloc.r0',spread(R0(j),1,npts))\ncall observe('alloc.s0',spread(S0(j),1,npts))\ncall observe('alloc.max_leaf',spread(max_LtoLSR,1,npts))\n" + "\n".join(f"call observe('alloc.{name.split('(')[0]}',{name},valid=biomass(:,j,ileaf,icarbon)>min_stomate)" for name in fields) + "\nendif\n"
    anchor = "    !       IF (j.EQ.10) WRITE(numout,*) 'zd toLSR 2 '"
    assert source.count(anchor) == 1
    source = source.replace(anchor, calls + anchor)
    anchor = "              dturnover(:) = biomass(:,j,ileaf,icarbon) * leaf_frac(:,j,m) * turnover_rate(:)"
    assert source.count(anchor) == 1
    source = source.replace(anchor, "if(j==14)then\ncall observe('turn.age',leaf_age(:,j,m),m)\ncall observe('turn.crit',leaf_age_crit(:,j),m)\ncall observe('turn.rate',turnover_rate,m)\nendif\n" + anchor)
    anchor = "          leaf_frac(:,j,m) = MAX( zero, leaf_frac(:,j,m) )"
    assert source.count(anchor) == 1
    source = source.replace(anchor, "if(j==14)call observe('vmax.fraction.before',leaf_frac(:,j,m),m)\n" + anchor + "\nif(j==14)call observe('vmax.fraction.after',leaf_frac(:,j,m),m)")
    anchor = "                  ( leafage_old - leafage_lastmax ) ) )"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + "\nif(j==14)then\ncall observe('vmax.rel_age',rel_age,m)\ncall observe('vmax.efficiency',leaf_efficiency,m)\nendif")
    source = source.replace("program main\n", "program main\nuse p06_observer,only:obsopen,obsclose\n", 1)
    source = source.replace("use p06_observer,only:obsopen,obsclose\n", "use p06_observer,only:obsopen,obsclose\nuse p05_allocation_pft_parameters,only:p06_r0=>R0,p06_s0=>S0\nuse p05_turnover_pft_parameters,only:p06_crit_turn=>leafagecrit\nuse p05_vmax_pft_parameters,only:p06_crit_vmax=>leafagecrit,p06_tau=>leaf_timecst\n", 1)
    source = source.replace("use p05_allocation_pft_parameters,only:", "use p05_allocation_constantes,only:p06_maxleaf=>max_LtoLSR\nuse p05_allocation_pft_parameters,only:", 1)
    source = source.replace("character(len=1024)::input_file,out", "character(len=1024)::input_file,out,obsfile")
    source = source.replace("real(r_std)::veg(n,v)", "real(r_std)::p06_input_r(v),p06_input_s(v),p06_input_crit(v),p06_input_tau(v),p06_input_maxleaf\nreal(r_std)::veg(n,v)")
    source = source.replace("close(inp)", "read(inp,*)p06_input_r\nread(inp,*)p06_input_s\nread(inp,*)p06_input_crit\nread(inp,*)p06_input_tau\nread(inp,*)p06_input_maxleaf\nclose(inp)", 1)
    source = source.replace("call setup_vmax(cold,firstcall,agri)", "call setup_vmax(cold,firstcall,agri)\np06_r0=p06_input_r;p06_s0=p06_input_s\np06_crit_turn=p06_input_crit;p06_crit_vmax=p06_input_crit;p06_tau=p06_input_tau\np06_maxleaf=p06_input_maxleaf")
    source = source.replace("call get_command_argument(1,input_file)", "call get_command_argument(3,obsfile);call obsopen(obsfile)\ncall get_command_argument(1,input_file)")
    source = source.replace("close(u)\ncontains", "close(u)\ncall obsclose()\ncontains")
    return OBSERVER + source


def write_input(path, x):
    base.write_fortran_input(path, x)
    arrays = (x["alloc_inputs"]["r0"], x["alloc_inputs"]["s0"], x["post_npp_inputs"]["leafagecrit"], x["post_npp_inputs"]["leaf_timecst"])
    with path.open("a", encoding="ascii") as stream:
        stream.write("\n".join(" ".join(format(float(v), ".17e") for v in array) for array in arrays) + "\n")
        stream.write(format(x["allocation_kwargs"].get("max_l_to_lsr", .5), ".17e") + "\n")


def current(x, compiled=False):
    from jax_orchidee.stomate import integration

    function = integration.stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_static_jit if compiled else integration.stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit
    return function(**x)


def actual_vmax_consumer(x, result):
    """Consume the returned turnover leaf arrays in the real next component."""
    from jax_orchidee.stomate.carbon_kernels import vmax_step

    p = x["post_npp_inputs"]
    return vmax_step(leaf_age=result.post_npp.turnover.leaf_age, leaf_frac=result.post_npp.turnover.leaf_frac,
        vcmax25=p["vcmax25"], n_limfert=p["n_limfert"], leaf_timecst=p["leaf_timecst"], leafagecrit=p["vmax_leafagecrit"],
        pheno_type=p["vmax_pheno_type"], leaf_tab=p["vmax_leaf_tab"], ok_laidev=p["vmax_ok_laidev"],
        dt_days=1.0, ok_dgvm=False, ok_nlim=False, min_stomate=p["min_stomate"])


def identities():
    out = base.source_identities()
    for name in ("constantes_mtc.f90", "constantes_var.f90", "constantes.f90", "pft_parameters.f90"):
        path = ROOT / "fortran_source/ORCHIDEE/src_parameters" / name
        out[name] = {"file": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90"
    out["leaf_timecst_source"] = {"file": path.relative_to(ROOT).as_posix(), "span": [574, 574], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return out


def arm_evidence(observations):
    """Classify actual original internal operands, never endpoint inversion."""
    rows = []
    for case, obs in observations.items():
        a = {k: np.asarray(v) for k, v in obs.items()}
        def add(arm, source, mask, operands):
            if source.startswith("stomate_alloc"):
                mask = mask & (a["alloc.valid"] > 0)
            cells = np.flatnonzero(mask).tolist()
            if cells:
                rows.append({"arm": arm, "source": source, "case": case, "cells_zero_based": cells, "observed_operands": operands})
        water = a["alloc.limit_W"]
        water_raw = a["alloc.moiavail_week"]
        hum = a["alloc.h_nitrogen"]
        temp_raw = 2.0 ** ((a["alloc.t_nitrogen"] - 273.15 - 25.0) / 10.0)
        add("water.lower", "stomate_alloc:676", water_raw < 0.1, {"raw": water_raw.tolist(), "bounded": water.tolist()})
        add("water.interior", "stomate_alloc:676", (water > 0.1) & (water < 1), {"bounded": water.tolist()})
        add("water.upper_equality", "stomate_alloc:676", water == 1.0, {"bounded": water.tolist()})
        add("humidity.lower", "stomate_alloc:687", hum < 0.5, {"raw": hum.tolist(), "bounded": a["alloc.limit_N_hum"].tolist()})
        add("humidity.interior", "stomate_alloc:687", (hum > 0.5) & (hum < 1), {"raw": hum.tolist()})
        add("humidity.upper_equality", "stomate_alloc:687", hum >= 1, {"raw": hum.tolist()})
        for label, mask in (("lower", temp_raw < .1), ("interior", (temp_raw > .1) & (temp_raw < 1)), ("upper", temp_raw > 1)):
            add("temperature." + label, "stomate_alloc:691-692", mask, {"raw_from_observed_temperature": temp_raw.tolist(), "bounded": a["alloc.limit_N_temp"].tolist()})
        nraw = a["alloc.limit_N_hum"] * a["alloc.limit_N_temp"]
        add("nitrogen.lower", "stomate_alloc:695", nraw < .1, {"raw": nraw.tolist(), "bounded": a["alloc.limit_N"].tolist()})
        add("nitrogen.interior", "stomate_alloc:695", (nraw > .1) & (nraw < 1), {"raw": nraw.tolist()})
        add("nitrogen.upper_equality", "stomate_alloc:695", nraw == 1, {"raw": nraw.tolist(), "bounded": a["alloc.limit_N"].tolist()})
        light, below = a["alloc.limit_L"], a["alloc.limit_WorN"]
        around = a["alloc.lai_around"]
        raw_light = np.exp(-.5 * around)
        add("light.inner_floor", "stomate_alloc:663-670", (around < 10) & (raw_light < .1), {"lai_around": around.tolist(), "raw_with_declared_ext_coeff": raw_light.tolist(), "bounded": light.tolist()})
        add("light.interior", "stomate_alloc:663-670", (around < 10) & (raw_light > .1), {"lai_around": around.tolist(), "bounded": light.tolist()})
        add("light.outer_high_lai", "stomate_alloc:663-670", around >= 10, {"lai_around": around.tolist(), "bounded": light.tolist()})
        add("water_or_nitrogen.water", "stomate_alloc:698", water < a["alloc.limit_N"], {"water": water.tolist(), "nitrogen": a["alloc.limit_N"].tolist()})
        add("water_or_nitrogen.nitrogen", "stomate_alloc:698", water > a["alloc.limit_N"], {"water": water.tolist(), "nitrogen": a["alloc.limit_N"].tolist()})
        add("total_limit.light", "stomate_alloc:702", light < below, {"light": light.tolist(), "belowground": below.tolist()})
        add("total_limit.belowground", "stomate_alloc:702", light > below, {"light": light.tolist(), "belowground": below.tolist()})
        denominator = np.where(a["alloc.valid"] > 0, light + 2 * below, 1.0)
        root_raw = 3 * a["alloc.r0"] * light / denominator
        leaf_raw = 1 - np.maximum(.15, root_raw) - a["alloc.StoLSR"]
        add("partition.root_floor", "stomate_alloc:712-714", root_raw < .15, {"raw_from_observed_limits": root_raw.tolist()})
        add("partition.root_interior", "stomate_alloc:712-714", root_raw > .15, {"raw_from_observed_limits": root_raw.tolist()})
        add("partition.leaf_lower", "stomate_alloc:722-723", leaf_raw < .2, {"raw_from_observed_limits": leaf_raw.tolist(), "bounded": a["alloc.LtoLSR"].tolist()})
        add("partition.leaf_interior", "stomate_alloc:722-723", (leaf_raw > .2) & (leaf_raw < a["alloc.max_leaf"]), {"raw_from_observed_limits": leaf_raw.tolist()})
        add("partition.leaf_upper", "stomate_alloc:722-723", leaf_raw > a["alloc.max_leaf"], {"raw_from_observed_limits": leaf_raw.tolist(), "bounded": a["alloc.LtoLSR"].tolist(), "configured_maximum": a["alloc.max_leaf"].tolist()})
        for m in range(1, 5):
            age, crit = a[f"turn.age{m}"], a[f"turn.crit{m}"]
            rate_raw = (age / crit) ** 4 / crit
            add("turnover.age_cap", "stomate_turnover:606-612", (age > crit / 2) & (rate_raw > .99), {"class": m, "age": age.tolist(), "critical_age": crit.tolist(), "rate": a[f"turn.rate{m}"].tolist()})
            add("turnover.below_cap", "stomate_turnover:606-612", (age > crit / 2) & (rate_raw < .99), {"class": m, "rate": a[f"turn.rate{m}"].tolist()})
            f = a[f"vmax.fraction.before{m}"]
            add("vmax.fraction_zero", "stomate_vmax:285", f == 0, {"class": m, "before": f.tolist(), "after": a[f"vmax.fraction.after{m}"].tolist()})
            add("vmax.fraction_positive", "stomate_vmax:285", f > 0, {"class": m, "before": f.tolist()})
            add("vmax.fraction_negative", "stomate_vmax:285", f < 0, {"class": m, "before": f.tolist(), "after": a[f"vmax.fraction.after{m}"].tolist()})
            rel, efficiency = a[f"vmax.rel_age{m}"], a[f"vmax.efficiency{m}"]
            for label, mask in (("rising", (rel > 0) & (rel < .03)), ("plateau", (rel > .03) & (rel < .5)), ("declining", (rel > .5) & (rel < 1)), ("floor", rel > 1)):
                add("vmax.efficiency_" + label, "stomate_vmax:341-346", mask, {"class": m, "relative_age": rel.tolist(), "efficiency": efficiency.tolist()})
    return rows


def run(output_dir=PACKAGE, *, write_fixture=False, compiler=DEFAULT_COMPILER):
    output_dir = Path(output_dir)
    checks = output_dir / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    source = observed_source()
    (checks / "observed-original.f90").write_text(source, encoding="utf-8")
    fixture = {"schema_version": 1, "source_identities": identities(), "rtol": 1e-12, "atol": 1e-14, "cases": {}, "observations": {}}
    comparison = []
    actuals = {}
    production_snapshot = {name: (ROOT / "jax_orchidee/stomate" / name).read_bytes() for name in ("carbon_kernels.py", "integration.py")}
    with tempfile.TemporaryDirectory(prefix="orchidee_p06_growth_") as td:
        folder = Path(td)
        unit, exe = folder / "original.f90", folder / "original.exe"
        unit.write_text(source, encoding="utf-8")
        build = compile_fortran(unit, exe, compiler, base_flags=FLAGS)
        (checks / "compile.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
        initial_cases = ("environment_partition", "age_efficiency", "partition_upper", "partition_lower", "fraction_floor", "water_limited", "source_default_upper")
        pending = [(case, inputs(case)) for case in initial_cases]
        while pending:
            case, x = pending.pop(0)
            inp, csvp, obsp = (checks / f"{case}-{kind}" for kind in ("inputs.txt", "outputs.csv", "observer.csv"))
            write_input(inp, x)
            executed = subprocess.run([str(exe), str(inp.resolve()), str(csvp.resolve()), str(obsp.resolve())], cwd=folder, env=compiler_environment(compiler), text=True, capture_output=True)
            (checks / f"{case}-execution.txt").write_text(executed.stdout + "\n" + executed.stderr, encoding="utf-8")
            executed.check_returncode()
            original = base.read_csv(csvp)
            fixture["observations"][case] = {k: v.tolist() for k, v in base.read_csv(obsp).items()}
            fixture["cases"][case] = {"inputs": base.encode(x), "expected": {k: {"shape": list(base.reference_shape(k)), "values": v.tolist()} for k, v in original.items()}}
            for compiled in (False, True):
                result = current(x, compiled)
                fields = base.chain_fields(result)
                actuals[f"{case}.{compiled}"] = base.encode(fields)
                for name, values in original.items():
                    comparison.append(float_comparison(f"{case}.{compiled}.{name}", values, fields[name].ravel(order="F"), rtol=1e-12, atol=1e-14))
                consumer = actual_vmax_consumer(x, result)
                for name in ("leaf_age", "leaf_frac", "vcmax"):
                    comparison.append(float_comparison(f"{case}.{compiled}.actual_vmax_consumer.{name}", original["vmax." + name], np.asarray(getattr(consumer, name)).ravel(order="F"), rtol=1e-12, atol=1e-14))
            if not case.endswith("_reentry"):
                independent_carry = base.carry_from_fields(x, {k: v.reshape(base.reference_shape(k), order="F") for k, v in original.items()})
                pending.append((case + "_reentry", independent_carry))
                actual_carry = base.carry_inputs(x, current(x, True))
                for group in actual_carry:
                    for name, value in actual_carry[group].items():
                        if isinstance(value, np.ndarray):
                            comparison.append(float_comparison(f"{case}.carry.{group}.{name}", independent_carry[group][name], value, rtol=1e-12, atol=1e-14))
                next_result = base.chain_fields(current(actual_carry, True))
                fixture["cases"][case]["actual_next_component_fields"] = base.encode(next_result)
        for case in initial_cases:
            actual_next = base.decode(fixture["cases"][case].pop("actual_next_component_fields"))
            for name, ref in fixture["cases"][case + "_reentry"]["expected"].items():
                comparison.append(float_comparison(case + ".actual_carry_consumed." + name, ref["values"], actual_next[name].ravel(order="F"), rtol=1e-12, atol=1e-14))
    arms = arm_evidence(fixture["observations"])
    report = {"status": "passed" if all(row["passed"] for row in comparison) else "failed", "comparisons": comparison, "arm_ids": sorted({row["arm"] for row in arms})}
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "arm-evidence.json").write_text(json.dumps(arms, indent=2) + "\n", encoding="utf-8")
    (checks / "current-full-fields.json").write_text(json.dumps(actuals, separators=(",", ":")) + "\n", encoding="ascii")
    if report["status"] == "failed":
        before = checks / "source-default-before"
        before.mkdir(exist_ok=True)
        for name, content in production_snapshot.items():
            (before / name).write_bytes(content)
        for name in ("comparison.json", "arm-evidence.json"):
            (before / name).write_bytes((output_dir / name).read_bytes())
        (before / "reference.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="ascii")
        (before / "current-full-fields.json").write_bytes((checks / "current-full-fields.json").read_bytes())
    if write_fixture:
        FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="ascii")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    report = run(write_fixture=args.write_fixture)
    failed = [row for row in report["comparisons"] if not row["passed"]]
    print(json.dumps({"status": report["status"], "comparisons": len(report["comparisons"]), "arms": report["arm_ids"], "failed_count": len(failed), "first_failures": failed[:3]}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)
