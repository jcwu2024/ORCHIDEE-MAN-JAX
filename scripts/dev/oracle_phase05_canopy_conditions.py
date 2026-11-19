"""PB01 phase05 finite original-source conditions; no coupled model step.

The phase01 composer is reused read-only. Every new input, generated original
program, compiler receipt, and output is kept under the phase05 canopy packet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from scripts.dev import oracle_physics_branch_canopy as pb
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes
from scripts.dev.fortran_oracle_common import COMPILE_FLAGS, DEFAULT_COMPILER, compiler_environment, float_comparison

ROOT = pb.ROOT
OUTPUT = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/canopy"
FIXTURE = ROOT / "tests/fixtures/phase05_canopy_reference.json"
RTOL, ATOL = 1e-11, 1e-12
CAPS = {"stilt_zero": (0.0, 0.4), "pneumatophore_zero": (1.5, 0.0), "both_zero": (0.0, 0.0)}
ALBEDOS = {"modis": (True, False), "wet_dry": (False, True), "mean": (False, False)}
LOW_LAI = np.array([0.0, 1e-8, 1.01e-8, 2e-8, 1e-7, 1e-6, 1e-4, 0.1])
PREF_SOIL_VEG = np.array([1,2,2,2,2,2,2,2,2,3,3,3,3,4], dtype=np.int32)


def make_control_inputs(cap="stilt_zero"):
    x = pb.make_control_inputs()
    # Positive stilt biomass avoids LOG10(0); cap-zero is the intended IF arm.
    x["h_agr_max_st"], x["h_agr_max_pn"] = CAPS[cap]
    x["tide_height"][:] = [-3.0, -0.05, 0.0, 0.1, 0.4, 2.0]
    return x


def make_albedo_inputs():
    x = pb.make_condveg_inputs()
    # Bracket fraction_veg > min_sechiba using representable source operands.
    x["totfrac_nobio"] = np.array([1.0, 1.0-5e-9, 1.0-1e-8, 1.0-2e-8, 0.0, 0.1, 0.1, 0.1])
    x["frac_nobio"] = x["totfrac_nobio"][:, None]
    fraction = 1.0 - x["totfrac_nobio"]
    x["veget_max"][:] = 0.0
    x["veget_max"][:, 0] = fraction * 0.2
    x["veget_max"][:, 13] = fraction * 0.8
    x["veget"][:] = x["veget_max"]
    x["veget"][:, 13] *= 0.5
    x["tot_bare_soil"] = x["veget_max"][:, 0] + x["veget_max"][:, 13] - x["veget"][:, 13]
    x["lai"][:] = 0.0
    x["lai"][:, 13] = 0.1
    x["height"][:, 13] = 4.0
    x["snow"][:] = [0.0, 1e-8, 1.0, 5.0, 10.0, 30.0, 10.0, 1.0]
    x["snow_nobio"][:, 0] = x["snow"]
    x["snow_age"][:] = [0.0, 0.0, 1.0, 5.0, 10.0, 100.0, 1e4, 20.0]
    x["snow_nobio_age"][:, 0] = x["snow_age"]
    return x


def make_low_lai_inputs():
    x = pb.make_condveg_inputs()
    x["lai"][:] = 0.0
    x["lai"][:, 13] = LOW_LAI
    x["height"][:, 13] = 4.0
    x["snow"][:] = 0.0
    x["snow_nobio"][:] = 0.0
    x["snowdz"][:] = 0.0
    x["snow_age"][:] = 0.0
    x["snow_nobio_age"][:] = 0.0
    x["veget_max"][:] = 0.0
    x["veget_max"][:, 0] = 0.2
    x["veget_max"][:, 13] = 0.7
    # The producer replaces these initial fields. Keep the independently
    # diagnostic high-fc tuple internally consistent, including point7.
    x["veget"][:] = 0.0
    x["veget"][:, 0] = 0.2
    x["veget"][:, 13] = 0.6
    x["tot_bare_soil"][:] = 0.3
    return x


def producer_inputs(x):
    return dict(lai=x["lai"], frac_nobio=x["frac_nobio"], veget_max=x["veget_max"],
                pref_soil_veg=PREF_SOIL_VEG, ext_coeff_vegetfrac=np.ones(14), nstm=6,
                min_vegfrac=1e-6, min_sechiba=1e-8, ok_dgvm=False)


def condveg_values(x, *, use_jit=False, kind="modis"):
    from jax_orchidee.sechiba.condveg import _condveg_main_minimal_jit, condveg_main_minimal
    kwargs = dict(x)
    kwargs.update(alb_bg_modis=ALBEDOS[kind][0], alb_bare_model=ALBEDOS[kind][1],
                  soilalb_wet=x["soilalb_bg"]*0.7, soilalb_dry=x["soilalb_bg"]*1.2,
                  soilalb_moy=x["soilalb_bg"]*0.95)
    result = (_condveg_main_minimal_jit if use_jit else condveg_main_minimal)(**kwargs)
    values = {k: np.asarray(getattr(result, k)).ravel() for k in result._fields}
    # Original dynamic roughness does not assign the PFT1 diagnostic.
    values["roughheight_pft"] = values["roughheight_pft"].reshape(8, 14)[:, 1:].ravel()
    return values


def run_low_lai(*, use_jit=False):
    import jax

    from jax_orchidee.sechiba.slowproc import slowproc_surface_update_explicit
    x = make_low_lai_inputs()
    numeric_fields = ("frac_nobio", "veget_max", "veget", "totfrac_nobio", "soiltile")
    def numeric_update(**inputs):
        update = slowproc_surface_update_explicit(**inputs)
        return {**{k:getattr(update.vegetation,k) for k in numeric_fields}, "tot_bare_soil":update.tot_bare_soil}
    fn = jax.jit(numeric_update, static_argnames=("nstm", "ok_dgvm")) if use_jit else numeric_update
    update = fn(**producer_inputs(x))
    for k in ("veget", "veget_max", "frac_nobio", "totfrac_nobio"):
        x[k] = np.asarray(update[k])
    x["tot_bare_soil"] = np.asarray(update["tot_bare_soil"])
    values = condveg_values(x, use_jit=use_jit)
    values.update({"producer."+k: np.asarray(update[k]).ravel() for k in numeric_fields})
    values["producer.tot_bare_soil"] = x["tot_bare_soil"].ravel()
    return values


def actual_values(*, use_jit=False):
    result = {"root_"+k: pb.run_controls(make_control_inputs(k), use_jit=use_jit) for k in CAPS}
    result.update({"albedo_"+k: condveg_values(make_albedo_inputs(), use_jit=use_jit, kind=k) for k in ALBEDOS})
    result["low_lai_predecessor"] = run_low_lai(use_jit=use_jit)
    # The actual helper and literal reference both return the PFT14 slice.
    return result


def composed(x, kind="modis", predecessor=False):
    previous = pb.make_condveg_inputs
    try:
        pb.make_condveg_inputs = lambda: x
        source, spans = pb.compose_condveg()
    finally:
        pb.make_condveg_inputs = previous
    modis, moisture = ALBEDOS[kind]
    source = source.replace("alb_bg_modis=.true., alb_bare_model=.false.",
                            f"alb_bg_modis={'.true.' if modis else '.false.'}, alb_bare_model={'.true.' if moisture else '.false.'}")
    marker = "call condveg_main(1,n"
    source = source.replace(marker, "soilalb_wet=soilalb_bg*.7d0;soilalb_dry=soilalb_bg*1.2d0;soilalb_moy=soilalb_bg*.95d0\n"+marker, 1)
    if predecessor:
        path = ROOT / "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90"
        span = extract_procedure_bytes(path, "slowproc_veget")
        spans["slowproc_veget"] = dict(start_line=span.start_line, end_line=span.end_line,
                                      sha256=span.span_sha256, source_sha256=span.source_sha256)
        source = source.replace("integer :: printlev=0,numout=6", "integer,parameter::nstm=6,nparts=12,nelements=1,nlitt=2,ndeep=11,ncarb=3,ndoc=2,npool=3\n"
                                "integer::pref_soil_veg(nvm)=(/1,2,2,2,2,2,2,2,2,3,3,3,3,4/),printlev_loc=0\nlogical::ok_dgvm=.false.\n"
                                "real(r_std)::ext_coeff_vegetfrac(nvm)=1d0\nreal(r_std),parameter::min_vegfrac=1d-6\ninteger :: printlev=0,numout=6")
        source = source.replace("  subroutine xios_orchidee_send_field", span.span_bytes.decode()+"\n  subroutine xios_orchidee_send_field", 1)
        decl = ("real(r_std)::soiltile(n,nstm),biomass(n,nvm,nparts,nelements),litter_above(n,nlitt,nvm,nelements),"
                "litter_below(n,nlitt,nvm,ndeep,nelements),carbon_32l(n,ncarb,nvm,ndeep),DOC(n,nvm,ndeep,ndoc,npool,nelements)\n"
                "logical::f_rot_sech(n)\n")
        source = source.replace("real(r_std)::lalo(n,2)", decl+"real(r_std)::lalo(n,2)")
        # Source SLOWPROC main1116–1121 follows its actual full veget procedure.
        producer = ("biomass=0d0;litter_above=0d0;litter_below=0d0;carbon_32l=0d0;DOC=0d0;f_rot_sech=.false.\n"
                    "call slowproc_veget(n,f_rot_sech,lai,frac_nobio,totfrac_nobio,veget_max,veget,soiltile,biomass,litter_above,litter_below,carbon_32l,DOC)\n")
        lines = path.read_bytes().splitlines(keepends=True)
        block = b"".join(lines[1115:1122])
        spans["slowproc_tot_bare_soil_block"] = dict(start_line=1116, end_line=1122,
              sha256=hashlib.sha256(block).hexdigest(), source_sha256=pb.sha(path))
        # Replace original ji/kjpindex boundary names only, preserving arithmetic.
        producer += block.decode().replace("kjpindex", "n").replace("ji", "i").replace("jv", "j")
        source = source.replace(marker, producer+"\n"+marker, 1)
        emit = []
        for name, shape in (("veget",(8,14)), ("veget_max",(8,14)), ("frac_nobio",(8,1)),
                            ("totfrac_nobio",(8,)), ("soiltile",(8,6)), ("tot_bare_soil",(8,))):
            expression = f"{name}(i)" if len(shape)==1 else f"{name}(i,j)"
            emit.append(pb.emit("producer."+name, expression, shape))
        source = source.replace("close(10)\nend program", "\n".join(emit)+"\nclose(10)\nend program")
    return source, spans


def build(source, target, *, may_fail=False):
    target.mkdir(parents=True, exist_ok=True)
    (target/"oracle.f90").write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="orchidee_phase05_canopy_") as temp:
        directory = Path(temp)
        src, exe = directory/"oracle.f90", directory/"oracle.exe"
        src.write_text(source, encoding="utf-8")
        env = compiler_environment(DEFAULT_COMPILER)
        command = [str(DEFAULT_COMPILER), *COMPILE_FLAGS, "-cpp", str(src), "-o", str(exe)]
        result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True)
        pb.save(target/"compile.json", dict(command=command, returncode=result.returncode, stdout=result.stdout,
                                           stderr=result.stderr, source_sha256=pb.sha(src)))
        if result.returncode:
            raise RuntimeError(f"compile failed {target}: {result.stderr[-5000:]}")
        command = [str(exe), str(target/"fortran_outputs.csv")]
        result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True)
        pb.save(target/"run.json", dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        if result.returncode and not may_fail:
            raise RuntimeError(f"run failed {target}: {result.stderr}")
        return result.returncode


def reference():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for k in CAPS:
        previous = pb.make_control_inputs
        try:
            pb.make_control_inputs = lambda k=k: make_control_inputs_original(k, previous)
            source, spans = pb.compose_controls()
        finally:
            pb.make_control_inputs = previous
        target = OUTPUT/("reference_root_"+k)
        build(source, target)
        pb.save(target/"source_spans.json", spans)
    for k in ALBEDOS:
        source, spans = composed(make_albedo_inputs(), kind=k)
        target = OUTPUT/("reference_albedo_"+k)
        build(source, target)
        pb.save(target/"source_spans.json", spans)
    source, spans = composed(make_low_lai_inputs(), predecessor=True)
    target = OUTPUT/"reference_low_lai_predecessor"
    build(source, target)
    pb.save(target/"source_spans.json", spans)
    groups = ["root_"+k for k in CAPS]+["albedo_"+k for k in ALBEDOS]+["low_lai_predecessor"]
    expected, identities = {}, {}
    for group in groups:
        target = OUTPUT/("reference_"+group)
        fields = pb.old.read_csv(target/"fortran_outputs.csv")
        if group.startswith("albedo_") or group=="low_lai_predecessor":
            fields["roughheight_pft"] = fields["roughheight_pft"].reshape(8,14)[:,1:].ravel()
        expected[group] = fields
        identities[group] = dict(csv_sha256=pb.sha(target/"fortran_outputs.csv"), spans=json.loads((target/"source_spans.json").read_text()))
    inputs = dict(roots={k:make_control_inputs(k) for k in CAPS}, albedo=make_albedo_inputs(),
                  low_lai=make_low_lai_inputs(), producer=producer_inputs(make_low_lai_inputs()), albedo_switches=ALBEDOS)
    pb.save(OUTPUT/"input_tuples.json", inputs)
    pb.save(FIXTURE, dict(schema_version=1, rtol=RTOL, atol=ATOL, inputs=inputs,
                         input_tuples_sha256=pb.sha(OUTPUT/"input_tuples.json"), expected=expected, reference_identity=identities))


def make_control_inputs_original(k, original):
    x = original()
    x["h_agr_max_st"], x["h_agr_max_pn"] = CAPS[k]
    x["tide_height"][:] = [-3.0, -0.05, 0.0, 0.1, 0.4, 2.0]
    return x


def diagnostics():
    source, spans = composed(make_albedo_inputs())
    source = source.replace("fixed_snow_albedo=undef_sechiba", "fixed_snow_albedo=.7d0")
    fixed = {}
    fixed["unmodified"] = build(source, OUTPUT/"diagnostic_fixed_snow_unmodified", may_fail=True)
    anchor = "    IF (ABS(fixed_snow_albedo - undef_sechiba)"
    assert source.count(anchor)==1
    # Deliberate local scratch initializations diagnose definition/use only.
    # Neither instrumented output is an original-source numerical reference.
    for value in (0,1):
        variant = source.replace(anchor, f"    fraction_veg={value}.d0\n"+anchor)
        fixed[str(value)] = build(variant, OUTPUT/f"diagnostic_fixed_snow_scratch_{value}")
    a = pb.old.read_csv(OUTPUT/"diagnostic_fixed_snow_scratch_0/fortran_outputs.csv")
    b = pb.old.read_csv(OUTPUT/"diagnostic_fixed_snow_scratch_1/fortran_outputs.csv")
    difference = {k:float(np.max(np.abs(a[k]-b[k]))) for k in ("albedo","albedo_snow")}
    assert all(v>0 for v in difference.values())
    x = make_low_lai_inputs()
    # The original arbitrary high-fc tiny-LAI tuple remains a source trap;
    # do not replace it with the legal predecessor in the failure record.
    x["lai"][:,13] = 1.01e-8
    x["veget"][:,13] = .6
    x["veget_max"][:,13] = .7
    source, _ = composed(x)
    trap = build(source, OUTPUT/"diagnostic_arbitrary_low_lai", may_fail=True)
    assert trap != 0
    pb.save(OUTPUT/"diagnostics.json", dict(fixed_snow=dict(returncodes=fixed, scratch_max_abs_difference=difference,
          status="original undefined local; numerical equivalence unavailable", spans=spans),
          arbitrary_low_lai=dict(returncode=trap, status="original invalid/zero/overflow trap retained")))


def certify():
    expected = json.loads(FIXTURE.read_text())["expected"]
    report = {}
    for use_jit in (False,True):
        mode = "jit" if use_jit else "eager"
        actual = actual_values(use_jit=use_jit)
        np.savez(OUTPUT/f"actual_{mode}.npz", **{g+"."+k:v for g,fs in actual.items() for k,v in fs.items()})
        comparisons = [float_comparison(g+"."+k, np.asarray(v), actual[g][k], rtol=RTOL, atol=ATOL)
                       for g,fs in expected.items() for k,v in fs.items()]
        report[mode] = dict(passed=all(c["passed"] for c in comparisons), comparisons=comparisons)
    pb.save(OUTPUT/"comparison.json", report)
    assert all(v["passed"] for v in report.values())


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reference","diagnostics","certify"))
    args = parser.parse_args()
    globals()[args.action]()
