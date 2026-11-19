"""Phase05 PB02: actual joint predecessor and remaining flux conditions.

Original Fortran procedures and the Milly observer fragment are retained
verbatim. No SECHIBA step, trajectory, thermal producer or external PBL is run.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.dev import oracle_canopy_component_certify as canopy  # noqa: E402
from scripts.dev import oracle_physics_branch_energy as phase01  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compile_fortran, compiler_environment  # noqa: E402
from scripts.dev.oracle_canopy_component_fixture import make_canopy_fixture, run_canopy_components  # noqa: E402

OUTPUT = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase05-canopy-energy-carbon/energy"
FIXTURE = ROOT / "tests/fixtures/phase05_energy_reference.json"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90"
JOINT_NAMES = ("paper_dry_wind_floor", "paper_wet", "paper_warm_dew", "paper_frozen_dew")
FLUX_NAMES = (
    "snow_warm_positive_cap",
    "snow_exact_freezing",
    "snow_cold",
    "zero_snow",
    "zero_heat_capacity",
    "negative_heat_capacity",
    "zero_sensible_difference",
    "dew_no_potential",
)
TOLERANCE = dict(rtol=1e-12, atol=1e-14)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def save(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def joint_fixture(data=None):
    """New paper LAIdev=false local predecessor; no natural-state claim."""
    base = make_canopy_fixture(indices=(0, 1, 7, 8))
    payload = {k: np.asarray(v).copy() for k, v in base.diffuco_precall.payload.items()}
    payload["u"][0] = payload["v"][0] = 0.0
    # Source-shaped density is a declared input to the original DIFFUCO chain.
    # It is not reconstructed from an observed downstream endpoint.
    payload["rau"] = 100.0 * payload["pb"] / (287.05 * payload["temp_air"])
    static = {**base.static_kwargs, "ok_laidev": np.zeros(14, dtype=bool)}
    driver = {**base.driver, "u": payload["u"], "v": payload["v"]}
    if data is not None:
        payload = {k: np.asarray(v) for k, v in data["payload"].items()}
        driver = {k: np.asarray(v) for k, v in data["driver"].items()}
        static = {
            **data["static"],
            "ok_laidev": np.asarray(data["static"]["ok_laidev"], dtype=bool),
            "rstruct_const": np.asarray(data["static"]["rstruct_const"]),
        }
        albedo = {k: np.asarray(v) for k, v in data["albedo"].items()}
        thermal = {k: np.asarray(v) for k, v in data["thermal"].items()}
    else:
        albedo, thermal = base.albedo, base.thermal
    return base._replace(
        case_names=JOINT_NAMES,
        diffuco_precall=base.diffuco_precall._replace(payload=payload),
        static_kwargs=static,
        driver=driver,
        albedo=albedo,
        thermal=thermal,
    )


def joint_inputs(fixture):
    return jsonable(
        dict(
            payload=fixture.diffuco_precall.payload,
            driver=fixture.driver,
            static=fixture.static_kwargs,
            albedo=fixture.albedo,
            thermal=fixture.thermal,
        )
    )


def run_joint(data=None, *, use_jit=False):
    return run_canopy_components(joint_fixture(data), diffuco_use_jit=use_jit, enerbil_use_jit=use_jit)


def scientific_spans():
    return {
        name: extract_procedure_bytes(source, name)
        for source, names in (
            (canopy.DIFFUCO, canopy.DIFF_PROCS),
            (canopy.ENERBIL, canopy.ENER_PROCS),
            (canopy.QSAT, canopy.Q_PROCS),
        )
        for name in names
    }


def source_records(spans):
    return [
        dict(
            procedure=n,
            path=str(
                (
                    canopy.DIFFUCO if n in canopy.DIFF_PROCS else canopy.QSAT if n in canopy.Q_PROCS else SOURCE
                ).relative_to(ROOT)
            ).replace("\\", "/"),
            start=s.start_line,
            end=s.end_line,
            source_sha256=s.source_sha256,
            span_sha256=s.span_sha256,
        )
        for n, s in spans.items()
    ]


def compose_joint(path):
    # Reuse declarations/scaffolding; the new fixture changes the paper PFT
    # flag, wind and density. Never write the previous accepted artifact.
    factory = canopy.make_canopy_fixture
    canopy.make_canopy_fixture = joint_fixture
    try:
        canopy.compose(path)
    finally:
        canopy.make_canopy_fixture = factory
    text = path.read_text(encoding="utf-8")
    text = text.replace("ok_LAIdev=.false.;ok_LAIdev(14)=.true.;", "ok_LAIdev=.false.;ok_LAIdev(14)=.false.;")
    extras = []
    for name, expression, matrix in (
        ("pgflux", "pgflux", False),
        ("temp_sol_add", "temp_sol_add", False),
        ("t2mdiag", "t2m", False),
        ("qsol_sat_new", "qsol_sat_new", False),
        ("qsol_sat_new_pft", "qsol_sat_new_pft", True),
        ("psnew", "psnew", False),
        ("lwup", "lwup", False),
        ("lwnet", "lwnet", False),
        ("flux_netrad", "netrad", False),
        ("fluxsubli", "fluxsubli", False),
        ("tsol_rad", "tsol_rad", False),
    ):
        extras += canopy.output_lines("enerbil." + name, expression, matrix)
    prefix, suffix = text.rsplit("close(10)", 1)
    text = prefix + "\n".join(extras) + "\nclose(10)" + suffix
    # No science substitution. Restore each scientific procedure's exact raw
    # source bytes after the older text scaffold's newline normalization.
    path.write_text(text, encoding="utf-8")
    unit = path.read_bytes()
    spans = scientific_spans()
    for name, span in spans.items():
        generated = extract_procedure_bytes(path, name)
        unit = unit.replace(generated.span_bytes, span.span_bytes)
    path.write_bytes(unit)
    return source_records(spans)


def make_flux_inputs(*, tiny=False, explicit=True):
    n = 4 if tiny else 8

    def grid(value):
        return np.full(n, value, dtype=np.float64)

    tsn = np.array([274.0, 273.15, 272.0, 274.0, 274.0, 274.0, 280.0, 280.0])[:n]
    x = dict(
        emis=grid(0.98),
        temp_sol=grid(272.0),
        rau=grid(1.2),
        u=grid(1.0),
        v=grid(0.3),
        q_cdrag=grid(0.01),
        vbeta=grid(0.5),
        valpha=grid(1.0),
        vbeta1=grid(0.4),
        vbeta5=grid(0.1),
        qair=grid(0.003),
        epot_air=grid(278.0 * 1004.675),
        psnew=tsn * 1004.675,
        qsol_sat_new=grid(0.008),
        temp_sol_new=tsn,
        lwdown=grid(340.0),
        swnet=grid(150.0),
        dt_sechiba=900.0,
        pb=grid(1000.0),
        precip_rain=grid(0.02),
        snowdz=np.tile([0.01, 0.02, 0.03], (n, 1)),
        temp_air=grid(279.0),
        pgflux=np.arange(n, dtype=float) + 21.0,
        soilcap=grid(4.0e5),
        ok_explicitsnow=explicit,
    )
    if not tiny:
        x["snowdz"][3] = 0.0
        x["soilcap"][4:6] = [0.0, -1.0]
        x["epot_air"][6] = x["psnew"][6]
        x["qair"][7] = 0.05
    else:
        # Direct source-local diagnostic domain only: these are deliberately
        # not a sechiba density/QSAT/CONDVEG joint tuple. They exercise the
        # source's retained tiny-denominator behavior, not natural reachability.
        x["emis"][:] = 1.0e-20
        x["rau"][:] = 1.0e-20
        x["q_cdrag"][:] = 0.001
        x["vbeta"][:] = 0.0
        x["vbeta1"][:] = 0.0
        x["vbeta5"][:] = 0.0
        x["qair"][:] = 0.003
        x["snowdz"][:] = 0.0
        x["soilcap"][:] = 4.0e5
        x["temp_sol_new"][:] = 281.0
        x["psnew"][:] = 281.0 * 1004.675
        x["emis"][1] = 0.98  # non-tiny neighboring denominator
        x["qair"][2] = 0.02  # outer active condition false
        x["epot_air"][3] = x["psnew"][3]  # exact zero sensible difference
    return x


LOCAL_FIELDS = (
    "emis",
    "temp_sol",
    "rau",
    "u",
    "v",
    "q_cdrag",
    "vbeta",
    "valpha",
    "vbeta1",
    "vbeta5",
    "qair",
    "epot_air",
    "psnew",
    "qsol_sat_new",
    "temp_sol_new",
    "lwdown",
    "swnet",
    "dt_sechiba",
)
SNOW_FIELDS = (
    "emis",
    "temp_sol",
    "temp_sol_new",
    "lwdown",
    "swnet",
    "rau",
    "u",
    "v",
    "q_cdrag",
    "vbeta",
    "valpha",
    "vbeta1",
    "vbeta5",
    "qair",
    "epot_air",
    "qsol_sat_new",
    "pb",
    "precip_rain",
    "snowdz",
    "temp_air",
    "pgflux",
    "soilcap",
    "dt_sechiba",
)
CORR_FIELDS = ("emis", "rau", "u", "v", "q_cdrag", "epot_air", "psnew", "pb")


def run_flux(inputs, *, use_jit=False):
    from jax import jit

    from jax_orchidee.sechiba.enerbil import enerbil_flux_source_routed

    flag = inputs["ok_explicitsnow"]
    data = {k: np.asarray(v) for k, v in inputs.items() if k != "ok_explicitsnow"}

    def calculate(values):
        return enerbil_flux_source_routed(
            local_inputs={k: values[k] for k in LOCAL_FIELDS},
            snow_inputs={**{k: values[k] for k in SNOW_FIELDS}, "ok_explicitsnow": flag},
            correction_inputs={k: values[k] for k in CORR_FIELDS},
        )

    return (jit(calculate) if use_jit else calculate)(data)


def flux_values(result):
    result = dict(result._asdict())
    values = {**result["flux"]._asdict(), **result["correction"]._asdict(), "tair": result["tair"]}
    snow = result.get("snow", result.get("explicit_snow"))
    values.update(pgflux=snow.pgflux, temp_sol_add=snow.temp_sol_add)
    return {k: np.asarray(v) for k, v in values.items()}


def milly_block():
    lines = SOURCE.read_bytes().splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if b"warning_correction(:)=.FALSE." in line)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip().upper() == b"ENDDO")
    return b"".join(lines[start:end]), dict(
        path=str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        start=start + 1,
        end=end,
        source_sha256=sha(SOURCE),
        span_sha256=hashlib.sha256(b"".join(lines[start:end])).hexdigest(),
    )


def compose_flux(path, inputs):
    n = len(inputs["emis"])
    template = (ROOT / "scripts/dev/oracle_lane_surface_enerbil_main.f90.template").read_bytes()
    header = template.split(b"program oracle_main", 1)[0]
    spans = {
        name: extract_procedure_bytes(source, name)
        for source, names in ((SOURCE, phase01.E_PROCS), (phase01.QSAT, phase01.Q_PROCS))
        for name in names
    }
    for marker, names in ((b"! <ENERBIL_PROCEDURES>", phase01.E_PROCS), (b"! <QSAT_PROCEDURES>", phase01.Q_PROCS)):
        header = header.replace(marker, b"\n\n".join(spans[name].span_bytes for name in names))
    decl = """program oracle_main
 use enerbil_oracle
 implicit none
 integer,parameter::n=NN
 integer::i
 character(1024)::outpath
 real(r_std)::emis(n),temp_sol(n),rau(n),u(n),v(n),q_cdrag(n),vbeta(n),valpha(n),vbeta1(n),vbeta5(n)
 real(r_std)::qair(n),epot_air(n),temp_sol_new(n),lwdown(n),swnet(n),pb(n),precip_rain(n),snowdz(n,nsnow),temp_air(n),pgflux(n),soilcap(n)
 real(r_std)::qsurf(n),fs(n),fl(n),vevapp(n),trad(n),evapot(n),ecorr(n),tadd(n),correction_out(n),warning_out(n)
 allocate(psnew(n),qsol_sat_new(n),lwup(n),lwnet(n),netrad(n),fluxsubli(n),qsat_air(n),tair(n))
""".replace("NN", str(n))
    program = decl
    for key, value in inputs.items():
        if key == "ok_explicitsnow":
            program += "ok_explicitsnow=" + (".true." if value else ".false.") + "\n"
        elif key == "dt_sechiba":
            program += f"dt_sechiba={value}\n"
        else:
            program += phase01.assignment(key, value)
    program += "call enerbil_flux(n,emis,temp_sol,rau,u,v,q_cdrag,vbeta,valpha,vbeta1,vbeta5,qair,epot_air,psnew,qsurf,fs,fl,fluxsubli,vevapp,temp_sol_new,lwdown,swnet,lwup,lwnet,pb,trad,netrad,evapot,ecorr,precip_rain,snowdz,temp_air,pgflux,soilcap,tadd)\n"
    program += (
        "call observe_milly(n,emis,rau,u,v,q_cdrag,evapot,psnew,epot_air,vevapp,tair,pb,correction_out,warning_out)\n"
    )
    program += "call get_command_argument(1,outpath);open(10,file=trim(outpath),status='replace');write(10,'(a)')'field,i,j,value'\n"
    fields = dict(
        qsurf="qsurf",
        lwup="lwup",
        lwnet="lwnet",
        tsol_rad="trad",
        netrad="netrad",
        vevapp="vevapp",
        fluxlat="fl",
        fluxsubli="fluxsubli",
        fluxsens="fs",
        evapot="evapot",
        evapot_corr="ecorr",
        pgflux="pgflux",
        temp_sol_add="tadd",
        tair="tair",
        correction="correction_out",
        warning="warning_out",
    )
    for name, expr in fields.items():
        program += "\n".join(canopy.output_lines(name, expr, False)) + "\n"
    program += "close(10)\ncontains\nsubroutine observe_milly(kjpindex,emis,rau,u,v,q_cdrag,evapot,psnew,epot_air,vevapp,tair,pb,correction_out,warning_out)\n"
    program += "integer,intent(in)::kjpindex\ninteger::ji\nreal(r_std)::emis(kjpindex),rau(kjpindex),u(kjpindex),v(kjpindex),q_cdrag(kjpindex),evapot(kjpindex),psnew(kjpindex),epot_air(kjpindex),vevapp(kjpindex),tair(kjpindex),pb(kjpindex),grad_qsat(kjpindex),evapot_corr(kjpindex),correction_out(kjpindex),warning_out(kjpindex),speed,qc,correction\nlogical::warning_correction(kjpindex)\ncall dev_qsatcalc(kjpindex,tair,pb,grad_qsat)\n"
    block, metadata = milly_block()
    unit = (
        header
        + program.encode()
        + block
        + b"\ncorrection_out(ji)=correction\nwarning_out(ji)=merge(1._r_std,0._r_std,warning_correction(ji))\nENDDO\nend subroutine\nend program\n"
    )
    path.write_bytes(unit)
    return metadata


def execute_reference(kind, compose, inputs=None):
    target = OUTPUT / "reference" / kind
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase05_energy_") as td:
        src, exe = Path(td) / "oracle.f90", Path(td) / "oracle.exe"
        metadata = compose(src) if inputs is None else compose(src, inputs)
        (target / "oracle.f90").write_bytes(src.read_bytes())
        build = compile_fortran(src, exe, extra_flags=("-cpp",))
        build["command"] = [str(DEFAULT_COMPILER), *build["compile_flags"], str(src), "-o", str(exe)]
        save(target / "compile.json", build)
        command = [str(exe), str((target / "fortran_outputs.csv").resolve())]
        ran = subprocess.run(
            command, cwd=td, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True
        )
        save(
            target / "run.json", dict(command=command, returncode=ran.returncode, stdout=ran.stdout, stderr=ran.stderr)
        )
        ran.check_returncode()
    return canopy.read_csv(target / "fortran_outputs.csv"), metadata


def normalize_joint(values, *, reference):
    data = {k: np.asarray(v).copy() for k, v in values.items()}
    if reference:
        data.pop("diffuco.qsintmax")
        data["diffuco.cim"] = data["diffuco.cim"].reshape(4, 14)[:, 13]
    for name in (
        "diffuco.q_cdrag_pft",
        "enerbil.qsol_sat_pft",
        "enerbil.pdqsold_pft",
        "enerbil.temp_sol_new_pft",
        "enerbil.qsol_sat_new_pft",
    ):
        data[name] = data[name].reshape(4, 14)[:, 1:].ravel()
    return data


def current_joint_values(components):
    values = canopy.current_values(components)
    for name in (
        "pgflux",
        "temp_sol_add",
        "t2mdiag",
        "qsol_sat_new",
        "qsol_sat_new_pft",
        "psnew",
        "lwup",
        "lwnet",
        "fluxsubli",
        "tsol_rad",
    ):
        values["enerbil." + name] = np.asarray(components.enerbil.payload[name]).ravel()
    values["enerbil.flux_netrad"] = np.asarray(components.enerbil.step.flux.netrad).ravel()
    return values


def comparison(case, mode, name, expected, actual, *, atol=1e-14, rtol=1e-12):
    y, x = np.asarray(expected), np.asarray(actual)
    return dict(
        case=case,
        mode=mode,
        field=name,
        expected_shape=list(y.shape),
        actual_shape=list(x.shape),
        values=int(y.size),
        rtol=rtol,
        atol=atol,
        passed=bool(x.shape == y.shape and np.all(np.isfinite(x)) and np.allclose(x, y, rtol=rtol, atol=atol)),
        max_abs=float(np.max(np.abs(x - y))) if x.shape == y.shape else None,
    )


def certify(reference):
    from jax_orchidee.sechiba.enerbil import C_STEFAN, CP_AIR, qsat_moisture_dev_qsatcalc

    checks = []
    snapshots = {}
    for mode in ("eager", "jit"):
        joint = run_joint(reference["joint"]["inputs"], use_jit=mode == "jit")
        actual = normalize_joint(current_joint_values(joint), reference=False)
        for key, expected in reference["joint"]["expected"].items():
            tol = dict(rtol=1e-11, atol=1e-12) if key.startswith("diffuco.") else TOLERANCE
            checks.append(comparison("joint", mode, key, expected, actual[key], **tol))
        snapshots[mode] = {"joint": {k: v.tolist() for k, v in actual.items()}}
        payload, step = joint.enerbil.payload, joint.enerbil.step
        tair = np.asarray(step.surftemp.epot_air_new) / CP_AIR
        grad = np.asarray(qsat_moisture_dev_qsatcalc(tair, payload["pb"]))
        potential = np.asarray(step.flux.evapot)
        amount = np.asarray(step.flux.vevapp)
        ratio = np.divide(amount, potential, out=np.zeros_like(amount), where=potential > 0.0)
        active = (potential > 0.0) & (np.asarray(step.surftemp.psnew) != np.asarray(step.surftemp.epot_air_new))
        radiation = 4.0 * np.asarray(payload["emis"]) * C_STEFAN * tair**3
        snapshots[mode]["joint_invariant_observations"] = dict(
            scope="Actual current production intermediates for a bounded source inequality; not a new independent numerical oracle",
            tair_k=tair.tolist(),
            grad_qsat=grad.tolist(),
            evapot_positive=active.tolist(),
            vevapp_over_evapot_on_positive_potential=ratio.tolist(),
            radiation_lower_bound=radiation.tolist(),
            min_radiation_lower_bound=float(np.min(radiation[active])),
            valpha=np.asarray(payload["valpha"]).tolist(),
            all_active_nonnegative_terms=bool(np.all(grad[active] >= 0.0) and np.all(ratio[active] >= 0.0)),
        )
        for case in reference["flux"]:
            result = run_flux(case["inputs"], use_jit=mode == "jit")
            actual = flux_values(result)
            # Warning is an original diagnostic, not a public JAX output.
            # The actual numerical correction is compared directly, ensuring
            # that a tiny branch cannot pass merely due to small flux scale.
            for key, expected in case["expected"].items():
                if key == "warning":
                    continue
                tol = (
                    dict(rtol=1e-12, atol=1e-30)
                    if case["name"] == "tiny_milly_local"
                    and key in ("correction", "evapot", "vevapp", "evapot_corr", "fluxsens", "fluxlat", "fluxsubli")
                    else TOLERANCE
                )
                checks.append(comparison(case["name"], mode, key, expected, actual[key], **tol))
            snapshots[mode][case["name"]] = {k: v.tolist() for k, v in actual.items()}
    return checks, snapshots


def build_reference():
    joint = joint_fixture()
    data, spans = execute_reference("joint", compose_joint)
    original = dict(
        schema="phase05-energy-original-source-v1",
        joint=dict(case_names=JOINT_NAMES, inputs=joint_inputs(joint), expected=normalize_joint(data, reference=True)),
        flux=[],
        source=spans,
    )
    for name, inputs in (
        ("snow_conjuncts", make_flux_inputs()),
        ("snow_disabled", make_flux_inputs(explicit=False)),
        ("tiny_milly_local", make_flux_inputs(tiny=True)),
    ):
        expected, block = execute_reference(name, compose_flux, inputs)
        original["flux"].append(
            dict(
                name=name,
                case_names=FLUX_NAMES
                if not name.startswith("tiny")
                else ["tiny_true_warning", "positive_neighbor", "zero_evapot", "zero_sensible_difference"],
                inputs=inputs,
                expected=expected,
            )
        )
        original["milly_observer_source"] = block
    save(FIXTURE, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.reference:
        build_reference()
    reference = json.loads(FIXTURE.read_text())
    checks, values = certify(reference)
    save(
        OUTPUT / "checks.json",
        dict(passed=all(c["passed"] for c in checks), checks=checks, fixture_sha256=sha(FIXTURE)),
    )
    save(OUTPUT / "actual-values.json", values)
    failed = [c for c in checks if not c["passed"]]
    print(
        json.dumps(dict(checks=len(checks), numerical_values=sum(c["values"] for c in checks), failed=failed), indent=2)
    )
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
