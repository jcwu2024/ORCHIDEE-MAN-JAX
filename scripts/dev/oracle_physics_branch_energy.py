"""PB-02 same-input original-Fortran energy/QSAT component certification.

No model integration or external atmospheric producer is executed. Scientific
procedures are extracted byte for byte; only diagnostic IO is stubbed.
"""

from __future__ import annotations

import argparse
import csv
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
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import DEFAULT_COMPILER, compile_fortran, compiler_environment  # noqa: E402

OUTPUT = ROOT / "outputs/reviews/main-physics-20261118/physics-branch-closeout/phase01-canopy-energy/energy"
FIXTURE = ROOT / "tests/fixtures/physics_branch_energy_reference.json"
ENER = ROOT / "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90"
QSAT = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
E_PROCS = (
    "enerbil_main",
    "enerbil_begin",
    "enerbil_surftemp",
    "enerbil_pottemp",
    "enerbil_flux",
    "enerbil_evapveg",
    "enerbil_t2mdiag",
    "enerbil_fusion",
)
Q_PROCS = ("qsatcalc", "dev_qsatcalc", "qsfrict_init")
RTOL, ATOL = 1e-12, 1e-14
NAMES = [
    "dry_zero_latent",
    "dew_negative_evap",
    "cold_snow",
    "warm_snow_rain",
    "flooded",
    "wind_floor",
    "zero_pft_cover",
    "radiative_cooling",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_inputs(dt=900.0, explicit=True, laidev=False):
    n, nv = 8, 14
    ts = np.array([285.0, 278.0, 263.0, 274.0, 291.0, 286.0, 280.0, 281.0])
    ta = np.array([284.0, 279.0, 261.0, 280.0, 290.0, 285.0, 279.0, 279.0])
    qair = np.array([0.005, 0.020, 0.001, 0.003, 0.010, 0.006, 0.005, 0.004])
    vb = np.array([0.0, 0.5, 0.3, 0.5, 0.6, 0.5, 0.4, 0.4])
    qc = np.array([0.009, 0.012, 0.01, 0.011, 0.013, 0.012, 0.01, 0.009])
    cap = np.array([5e5, 4e5, 3e5, 4e5, 6e5, 5e5, 4e5, 5e5])

    def pft(x):
        return np.repeat(np.asarray(x)[:, None], nv, axis=1)

    vm = np.zeros((n, nv))
    vm[:, 13] = 0.8
    vm[6, 13] = 0.0
    b2 = np.zeros_like(vm)
    b3 = b2.copy()
    b3p = b2.copy()
    b4p = b2.copy()
    b2[:, 13] = 0.05
    b3[:, 13] = 0.25
    b3p[:, 13] = 0.35
    b4p[:, 13] = 0.15
    b2[0] = 0.0
    b3[0] = 0.0
    b3p[0] = 0.0
    b4p[0] = 0.0
    # No covered PFT means all source beta components are zero for that PFT.
    b2[6] = 0.0
    b3[6] = 0.0
    b3p[6] = 0.0
    b4p[6] = 0.0
    sd = np.zeros((n, 3))
    sd[2] = [0.02, 0.03, 0.04]
    sd[3] = [0.01, 0.02, 0.03]
    tsp = pft(ts)
    tsp[:, 13] += 1.25
    ok = [False] * nv
    ok[13] = laidev
    return dict(
        temp_sol=ts,
        temp_sol_pft=tsp,
        lwdown=np.array([310.0, 340.0, 220.0, 350.0, 360.0, 330.0, 310.0, 180.0]),
        swnet=np.array([130.0, 50.0, 20.0, 240.0, 210.0, 140.0, 100.0, 0.0]),
        pb=np.array([1013.0, 990.0, 950.0, 980.0, 1005.0, 1000.0, 1010.0, 990.0]),
        emis=np.array([1.0, 0.98, 0.97, 0.99, 0.96, 0.98, 1.0, 0.98]),
        ok_laidev=ok,
        epot_air=ta * 1004.675,
        petAcoef=np.array([0.0, 0.01, -0.02, 0.03, 0.02, -0.01, 0.0, 0.01]),
        petBcoef=ta * 1004.675,
        qair=qair,
        peqAcoef=np.array([0.0, 0.1, -0.1, 0.2, 0.15, -0.05, 0.0, 0.05]),
        peqBcoef=qair,
        soilflx=np.array([8.0, -3.0, 5.0, 9.0, -6.0, 7.0, 4.0, -9.0]),
        soilflx_pft=pft([8.0, -3.0, 5.0, 9.0, -6.0, 7.0, 4.0, -9.0]),
        rau=np.array([1.2, 1.15, 1.25, 1.2, 1.1, 1.2, 1.2, 1.2]),
        u=np.array([2.0, 1.0, 0.1, 3.0, 2.0, 0.0, 0.06, 1.0]),
        v=np.array([1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.08, 0.3]),
        q_cdrag=qc,
        q_cdrag_pft=pft(qc) * 1.1,
        vbeta=vb,
        vbeta_pft=pft(vb) * vm,
        valpha=np.array([1.0, 0.8, 1.0, 0.9, 0.85, 1.0, 1.0, 0.9]),
        vbeta1=np.array([0.0, 0.0, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0]),
        vbeta2=b2,
        vbeta3=b3,
        vbeta3pot=b3p,
        vbeta4=np.array([0.0, 0.2, 0.1, 0.2, 0.15, 0.2, 0.2, 0.2]),
        vbeta4_pft=b4p,
        vbeta5=np.array([0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0]),
        soilcap=cap,
        soilcap_pft=pft(cap) * 1.2,
        veget_max=vm,
        dt_sechiba=dt,
        q_sol_pot=np.linspace(0.01, 0.017, n),
        temp_sol_pot=ts - 1.0,
        precip_rain=np.array([0.0, 0.0, 0.001, 0.02, 0.01, 0.0, 0.0, 0.0]),
        snowdz=sd,
        temp_air=ta,
        pgflux=np.arange(n, dtype=float) + 11.0,
        ok_explicitsnow=explicit,
    )


def json_inputs(inputs):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}


def run_production(inputs, mode="eager"):
    """Return the actual production component; static flags match its existing JIT."""
    from jax_orchidee.sechiba import enerbil

    args = {k: (tuple(v) if k == "ok_laidev" else v) for k, v in inputs.items()}
    kernel = enerbil.enerbil_explicit_local_step if mode == "eager" else enerbil._enerbil_explicit_local_step_jit
    return kernel(**args)


def values(result):
    fields = {}
    for name, value in result.begin._asdict().items():
        array = np.asarray(value)
        fields["begin_" + name] = array[:, 13] if array.ndim == 2 else array
    for group in ("surftemp", "flux", "evapveg_grid", "evapveg_pft", "explicit_snow", "pottemp"):
        item = getattr(result, group)
        for name, value in item._asdict().items():
            if name in (
                "dtheta",
                "dtheta_pft",
                "psnew_pft",
                "sensfl",
                "sensfl_pft",
                "larsub",
                "larsub_pft",
                "lareva",
                "lareva_pft",
                "phpsnow",
            ):
                continue
            array = np.asarray(value)
            fields[name] = array[:, 13] if array.ndim == 2 else array
    fields["evapot_corr"] = np.asarray(result.evapot_corr.evapot_corr)
    fields["t2mdiag"] = np.asarray(result.t2mdiag)
    return fields


def assignment(name, array):
    a = np.asarray(array)
    vals = ",".join(f"{float(v):.17e}".replace("e", "d") for v in a.ravel(order="F"))
    return f"{name}=reshape((/{vals}/),shape({name}))\n"


ALIASES = dict(
    temp_sol="ts",
    temp_sol_pft="tsp",
    temp_air="tair_in",
    epot_air="epot",
    petAcoef="peta",
    petBcoef="petb",
    peqAcoef="peqa",
    peqBcoef="peqb",
    vbeta="vb",
    vbeta_pft="vbp",
    valpha="va",
    vbeta1="vb1",
    vbeta2="vb2",
    vbeta3="vb3",
    vbeta3pot="vb3p",
    vbeta4="vb4",
    vbeta4_pft="vb4p",
    vbeta5="vb5",
    soilflx_pft="soilflxp",
    soilcap_pft="soilcapp",
    q_cdrag="qc",
    q_cdrag_pft="qcp",
    veget_max="vmax",
    precip_rain="rain",
    snowdz="sdz",
    pgflux="pg",
)
OUTMAP = dict(
    temp_sol_new="tsn",
    temp_sol_new_pft="tsnp(:,14)",
    qsol_sat_new="qsol_sat_new",
    qsol_sat_new_pft="qsol_sat_new_pft(:,14)",
    psnew="psnew",
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
    vevapsno="vsno",
    vevapnu="vnu",
    vevapflo="vflo",
    vevapnu_pft="vnup(:,14)",
    vevapwet="vwet(:,14)",
    transpir="tr(:,14)",
    transpot="trp(:,14)",
    pgflux="pg",
    temp_sol_add="tadd",
    q_sol_pot="q_sol_pot",
    temp_sol_pot="temp_sol_pot",
    t2mdiag="t2",
)


def compose(cases):
    template = (ROOT / "scripts/dev/oracle_lane_surface_enerbil_main.f90.template").read_bytes()
    spans = {
        name: extract_procedure_bytes(src, name) for src, names in ((ENER, E_PROCS), (QSAT, Q_PROCS)) for name in names
    }
    header = template.split(b"program oracle_main", 1)[0]
    for marker, names in ((b"! <ENERBIL_PROCEDURES>", E_PROCS), (b"! <QSAT_PROCEDURES>", Q_PROCS)):
        header = header.replace(marker, b"\n\n".join(spans[n].span_bytes for n in names))
    original_program = template.split(b"program oracle_main", 1)[1].decode()
    declarations = original_program.split(" index=(/1,2/)", 1)[0].replace("n=2", "n=8")
    declarations += "\n real(r_std)::qa_out(n),ep_out(n),melt(n),fus(n),qtemps(24),qpres(24),qvals(24),dvals(24)\n"
    program = "program oracle_main" + declarations
    program += "\n call get_command_argument(1,path);open(10,file=trim(path),status='replace');write(10,'(a)')'case,field,index,value'\n"
    # Only harness declarations/allocations are reused from the older driver.
    program += " allocate(psold" + original_program.split(" allocate(psold", 1)[1].split(" q_sol_pot=", 1)[0]
    program += "\nindex=1;indexveg=1;zlev=2.;hum=1.\n"
    for case in cases:
        data, label = case["inputs"], case["name"]
        for k, v in data.items():
            if k == "ok_laidev":
                program += "ok_LAIdev=.false.\n" + ("ok_LAIdev(14)=.true.\n" if v[13] else "")
            elif k in ("dt_sechiba", "ok_explicitsnow"):
                program += f"{k}=" + ((".true." if v else ".false.") if isinstance(v, bool) else str(v)) + "\n"
            else:
                program += assignment(ALIASES.get(k, k), v)
        program += "evapot=0.;ecorr=0.;trad=ts;vevapp=0.\n"
        # The original main keeps qair_new/epot_air_new local. Observe them by
        # an independent begin+surface invocation BEFORE main, which repeats
        # begin and overwrites all workspace. Never solve after flux has
        # overwritten module netrad with its new-time diagnostic.
        program += "call enerbil_begin(n,ts,tsp,lwdown,swnet,pb,psold,psold_pft,qsol_sat,qsol_sat_pft,pdqsold,pdqsold_pft,netrad,netrad_pft,emis)\n"
        for name in (
            "psold",
            "psold_pft",
            "qsol_sat",
            "qsol_sat_pft",
            "pdqsold",
            "pdqsold_pft",
            "lwabs",
            "netrad",
            "netrad_pft",
        ):
            expr = name + "(:,14)" if name.endswith("_pft") else name
            program += f"call emit('{label}','begin_{name}',{expr})\n"
        program += "call enerbil_surftemp(n,zlev,emis,epot,peta,petb,qair,peqa,peqb,soilflx,soilflxp,rau,u,v,qc,qcp,vb,vbp,va,vb1,vb5,soilcap,soilcapp,lwdown,swnet,psnew,qsol_sat_new,qsol_sat_new_pft,tsn,tsnp,qa_out,ep_out,vmax)\n"
        maincall = original_program.split(" call enerbil_main", 1)[1].split(" call get_command_argument", 1)[0]
        program += " call enerbil_main" + maincall
        for name, expr in {**OUTMAP, "qair_new": "qa_out", "epot_air_new": "ep_out"}.items():
            program += f"call emit('{label}','{name}',{expr})\n"
        # Fusion is a separate post-HYDROL boundary. Explicit branch does not
        # assign fusion in Fortran; retain sentinel rather than certify it.
        program += assignment("melt", [0.0, 0.001, 0.02, 0.1, 0.0, 0.03, 0.04, 0.01])
        program += "fus=-999.;call enerbil_fusion(n,melt,soilcap,soilcapp,sdz,tsn,tsnp,fus)\n"
        for name, expr in {
            "fusion_temp_sol_new": "tsn",
            "fusion_temp_sol_new_pft": "tsnp(:,14)",
            **({} if data["ok_explicitsnow"] else {"fusion": "fus"}),
        }.items():
            program += f"call emit('{label}','{name}',{expr})\n"
    temps = [
        99.25,
        100.0,
        100.499999,
        100.5,
        101.0,
        150.25,
        250.25,
        271.999,
        272.0,
        272.49,
        272.5,
        272.999,
        273.0,
        273.001,
        273.49,
        273.5,
        280.25,
        295.75,
        368.0,
        368.499999,
        368.5,
        368.999,
        369.0,
        370.25,
    ]
    program += assignment("qtemps", temps) + assignment("qpres", np.linspace(700.0, 1013.25, len(temps)))
    program += "call qsatcalc(24,qtemps,qpres,qvals);call dev_qsatcalc(24,qtemps,qpres,dvals)\ncall emit('qsat','qsat',qvals);call emit('qsat','derivative',dvals)\nl_qsat_first=.true.;call dev_qsatcalc(24,qtemps,qpres,dvals);call emit('qsat','derivative_firstcall',dvals)\nclose(10)\ncontains\n"
    program += "subroutine emit(label,field,a)\ncharacter(*),intent(in)::label,field\nreal(r_std),intent(in)::a(:)\ninteger::k\ndo k=1,size(a)\nwrite(10,'(a,a,a,a,i0,a,es26.17)')label,',',field,',',k,',',a(k)\nenddo\nend subroutine\nend program\n"
    return header + program.encode(), spans, temps


def certify(fixture):
    from jax import jit

    from jax_orchidee.sechiba import enerbil as e

    checks = []
    for case in fixture["cases"]:
        for mode in ("eager", "jit"):
            result = run_production(case["inputs"], mode)
            actual = values(result)
            # Return only numerical fields from fusion (the public record also
            # contains source strings, which are intentionally outside JIT).
            args = dict(
                tot_melt=fixture["melt_amount"],
                soilcap=case["inputs"]["soilcap"],
                soilcap_pft=case["inputs"]["soilcap_pft"],
                snowdz=case["inputs"]["snowdz"],
                temp_sol_new=result.surftemp.temp_sol_new,
                temp_sol_new_pft=result.surftemp.temp_sol_new_pft,
                ok_laidev=case["inputs"]["ok_laidev"],
                ok_explicitsnow=case["inputs"]["ok_explicitsnow"],
                dt_sechiba=case["inputs"]["dt_sechiba"],
            )

            def fusion(t, tp):
                out = e.enerbil_fusion_step(**{**args, "temp_sol_new": t, "temp_sol_new_pft": tp})
                return out.temp_sol_new, out.temp_sol_new_pft, out.fusion

            fn = fusion if mode == "eager" else jit(fusion)
            a, ap, f = fn(args["temp_sol_new"], args["temp_sol_new_pft"])
            actual.update(fusion_temp_sol_new=np.asarray(a), fusion_temp_sol_new_pft=np.asarray(ap)[:, 13])
            if not case["inputs"]["ok_explicitsnow"]:
                actual["fusion"] = np.asarray(f)
            for name, expected in case["expected"].items():
                x, y = np.asarray(actual[name]), np.asarray(expected)
                checks.append(
                    dict(
                        case=case["name"],
                        mode=mode,
                        field=name,
                        count=y.size,
                        passed=bool(np.allclose(x, y, rtol=RTOL, atol=ATOL)),
                        max_abs=float(np.max(np.abs(x - y))),
                    )
                )
    q = fixture["qsat"]
    for name, fn in (
        ("qsat", e.qsat_moisture_qsatcalc),
        ("derivative", e.qsat_moisture_dev_qsatcalc),
        ("derivative_firstcall", e.qsat_moisture_dev_qsatcalc),
    ):
        for mode in ("eager", "jit"):
            x = np.asarray((fn if mode == "eager" else jit(fn))(np.array(q["temp"]), np.array(q["pres"])))
            y = np.asarray(q["expected"][name])
            checks.append(
                dict(
                    case="qsat",
                    mode=mode,
                    field=name,
                    count=y.size,
                    passed=bool(np.allclose(x, y, rtol=1e-12, atol=1e-24)),
                    max_abs=float(np.max(np.abs(x - y))),
                )
            )
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.reference:
        cases = [
            dict(name=f"dt{int(dt)}-explicit{int(ex)}-laidev{int(lai)}", inputs=json_inputs(make_inputs(dt, ex, lai)))
            for dt, ex, lai in (
                (900.0, True, False),
                (2700.0, True, False),
                (900.0, False, False),
                (900.0, True, True),
                (900.0, False, True),
            )
        ]
        unit, spans, temps = compose(cases)
        with tempfile.TemporaryDirectory(prefix="pb02_energy_") as td:
            src = Path(td) / "energy.f90"
            exe = Path(td) / "energy.exe"
            src.write_bytes(unit)
            (OUTPUT / "reference.f90").write_bytes(unit)
            build = compile_fortran(src, exe)
            raw = OUTPUT / "reference.csv"
            run = subprocess.run(
                [str(exe), str(raw.resolve())],
                cwd=td,
                env=compiler_environment(DEFAULT_COMPILER),
                capture_output=True,
                text=True,
                check=True,
            )
        save(OUTPUT / "build.json", build)
        (OUTPUT / "reference-stdout.txt").write_text(run.stdout, encoding="utf-8")
        table = {}
        for row in csv.DictReader(raw.open()):
            table.setdefault(row["case"], {}).setdefault(row["field"], []).append(float(row["value"]))
        for case in cases:
            case["expected"] = table[case["name"]]
        fixture = dict(
            schema="physics-branch-energy-original-fortran-v1",
            case_domain="synthetic conditional local-component input tuples; independent points, no DIFFUCO comb consistency or natural-trajectory/PBL producer claim",
            point_names=NAMES,
            cases=cases,
            melt_amount=[0.0, 0.001, 0.02, 0.1, 0.0, 0.03, 0.04, 0.01],
            qsat=dict(temp=temps, pres=np.linspace(700.0, 1013.25, 24).tolist(), expected=table["qsat"]),
            source=[
                dict(
                    procedure=k,
                    path=str((QSAT if k in Q_PROCS else ENER).relative_to(ROOT)).replace("\\", "/"),
                    start=v.start_line,
                    end=v.end_line,
                    source_sha256=v.source_sha256,
                    span_sha256=v.span_sha256,
                )
                for k, v in spans.items()
            ],
            build=build,
            tolerance=dict(rtol=RTOL, atol=ATOL, qsat_atol=1e-24),
        )
        save(FIXTURE, fixture)
    fixture = json.loads(FIXTURE.read_text())
    checks = certify(fixture)
    save(
        OUTPUT / "checks.json",
        dict(
            passed=all(x["passed"] for x in checks),
            checks=checks,
            fixture_sha256=sha(FIXTURE),
            production_sha256=sha(ROOT / "jax_orchidee/sechiba/enerbil.py"),
        ),
    )
    failed = [x for x in checks if not x["passed"]]
    print(
        json.dumps(
            dict(checks=len(checks), numerical_values=sum(x["count"] for x in checks), failures=failed), indent=2
        )
    )
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
