"""Bounded PB02 producer admission: coefficients, retained restart and QSAT.

No time integration, natural history, or whole THERMOSOIL component is run.
Ordinary checks use the portable frozen reference; --reference needs original
sources and the existing local compiler and writes only this phase06 lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

from scripts.dev import oracle_phase05_energy_conditions as energy  # noqa: E402
from scripts.dev import oracle_thermosoil_cold_coef_actual as cold  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
)

OUT = (
    ROOT
    / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/energy-predecessors"
)
FIXTURE = ROOT / "tests/fixtures/phase06_energy_predecessors_reference.json"
THERMAL = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
QSAT = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
COEFFICIENTS = {
    "soilcap": (1,),
    "soilcap_pft": (1, 14),
    "soilflx": (1,),
    "soilflx_pft": (1, 14),
    "cgrnd": (1, 31, 14),
    "dgrnd": (1, 31, 14),
    "cgrnd_snow": (1, 3),
    "dgrnd_snow": (1, 3),
    "lambda_snow": (1,),
}
THERMAL_TOL = dict(rtol=1e-12, atol=1e-12)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    energy.save(path, value)


def encode(value):
    a = np.asarray(value)
    result = dict(shape=list(a.shape), dtype=str(a.dtype))
    if np.all(a == a.flat[0]):
        result["fill"] = a.flat[0].item()
    else:
        result["data"] = a.tolist()
    return result


def decode(value):
    if "fill" in value:
        return np.full(value["shape"], value["fill"], dtype=value["dtype"])
    if "pft_broadcast" in value:
        return np.broadcast_to(
            np.asarray(value["pft_broadcast"], dtype=value["dtype"])[:, :, None], value["shape"]
        ).copy()
    if "indices" in value:
        return np.asarray(value["values"], dtype=value["dtype"])[value["indices"]].reshape(value["shape"])
    return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])


def decode_dict(value):
    return {k: decode(v) for k, v in value.items()}


def source_record(path, first, last, meaning):
    raw = Path(path).read_bytes()
    span = b"".join(raw.splitlines(keepends=True)[first - 1 : last])
    return dict(
        path=Path(path).relative_to(ROOT).as_posix(),
        first=first,
        last=last,
        source_sha256=sha(path),
        span_sha256=hashlib.sha256(span).hexdigest(),
        meaning=meaning,
    )


def execute(name, source, inputs=None):
    directory = OUT / "reference" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "oracle.f90").write_bytes(source)
    with tempfile.TemporaryDirectory(prefix="phase06_energy_") as temporary:
        folder = Path(temporary)
        src, exe = folder / "oracle.f90", folder / "oracle.exe"
        src.write_bytes(source)
        compiled = compile_fortran(src, exe)
        save(directory / "compile.json", compiled)
        output = directory / "outputs.bin"
        arguments = [str(exe)]
        if inputs is not None:
            binary = directory / "inputs.bin"
            with binary.open("wb") as handle:
                for a in inputs.values():
                    handle.write(
                        np.asarray(a, dtype="<i4" if a.dtype.kind in "iub" else "<f8").ravel(order="F").tobytes()
                    )
            arguments.append(str(binary.resolve()))
        arguments.append(str(output.resolve()))
        process = subprocess.run(
            arguments,
            cwd=folder,
            env=compiler_environment(DEFAULT_COMPILER),
            capture_output=True,
            text=True,
            timeout=30,
        )
        save(
            directory / "run.json",
            dict(command=arguments, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr),
        )
        process.check_returncode()
    return np.fromfile(output, dtype="<f8")


def deep_inputs():
    """Reuse only complete frozen original geometry/moisture, declare new state."""
    fixture = json.loads((ROOT / "tests/fixtures/physics_branch_thermal_reference.json").read_text())
    base = fixture["cases"][0]
    v = decode_dict(base["inputs"])
    expected = decode_dict(base["expected"])
    names = ("QZ", "SMCMAX", "mcs", "so_capa_dry_ns", "zlt", "dlt", "dz1", "refSOC")
    result = {k: v[k] for k in names}
    for name in ("mc_layt", "mcl_layt", "tmc_layt", "mc_layt_pft", "mcl_layt_pft", "tmc_layt_pft"):
        result[name] = expected["humlev." + name]
    result.update(
        ptn=np.full((1, 32, 14), 268.0),
        ptn_pftmean=np.full((1, 32), 268.0),
        shum_ngrnd_permalong=np.full((1, 32, 14), 0.5),
        njsc=v["njsc"],
        temp_sol_new=np.array([268.0]),
        temp_sol_new_pft=np.full((1, 14), 268.0),
        snow=np.array([6000.0]),
        organic_layer_thick=np.zeros(1),
        soilc_total=np.zeros((1, 32, 14)),
        veget_max=np.eye(1, 14, 13) * 0.8,
        snowdz=np.array([[0.05, 0.5, 9.45]]),
        snowrho=np.full((1, 3), 600.0),
        snowtemp=np.full((1, 3), 268.0),
        pb=np.array([950.0]),
        frac_snow_veg=np.ones(1),
        frac_snow_nobio=np.zeros((1, 1)),
        totfrac_nobio=np.array([0.2]),
        lambda_snow=np.zeros(1),
        cgrnd_snow=np.zeros((1, 3)),
        dgrnd_snow=np.zeros((1, 3)),
        ok_lai_i=np.zeros(14, dtype=np.int32),
        mask_i=np.ones((1, 14), dtype=np.int32),
        controls=np.array([0, 1, 1, 1, 1, 0], dtype=np.int32),
        scalars=v["scalars"],
    )
    return result


def coefficient_source():
    """Complete original getdiff/coef, with read-only local observers."""
    unit = cold.TEMPLATE.read_bytes()
    spans = {}
    for name in (
        "thermosoil_cond_pft",
        "thermosoil_cond_nopft",
        "thermosoil_cond",
        "thermosoil_getdiff",
        "thermosoil_coef",
    ):
        raw = extract_procedure_bytes(THERMAL, name)
        spans[name] = raw
        body = raw.span_bytes
        assignments = ()
        if name == "thermosoil_getdiff":
            assignments = (b"  oracle_porosity=poros_net", b"  oracle_zx1=zx1", b"  oracle_zx2=zx2")
        if name == "thermosoil_coef":
            assignments = tuple(
                f"  oracle_{field}={field}".encode()
                for field in (
                    "soilcap_nosnow",
                    "soilflx_nosnow",
                    "soilcap_pft_nosnow",
                    "soilflx_pft_nosnow",
                    "zdz1",
                    "zdz2",
                    "cgrnd_soil",
                    "dgrnd_soil",
                    "zdz1_soil",
                    "zdz2_soil",
                    "snowcap",
                    "snowflx",
                    "dz1_snow",
                    "dz2_snow",
                    "zdz1_snow",
                    "zdz2_snow",
                )
            )
        if assignments:
            body = cold._instrument(body, f"  END SUBROUTINE {name}".encode(), assignments)
        unit = unit.replace(f"! <{name.upper()}>".encode(), body)
    # Exact original CONDVEG producer for both cover fractions, not arbitrary
    # independently chosen cover weights. Only this call is added to driver.
    condpath = ROOT / "fortran_source/ORCHIDEE/src_sechiba/condveg.f90"
    cond = extract_procedure_bytes(condpath, "condveg_frac_snow")
    unit = unit.replace(
        b"end module thermosoil_cold_coef_actual_oracle",
        cond.span_bytes + b"\nend module thermosoil_cold_coef_actual_oracle",
    )
    unit = unit.replace(b"sn_dens=330._r_std, sn_capa=", b"snowcri_alb=1.5_r_std, sn_dens=330._r_std, sn_capa=")
    call = b"  call condveg_frac_snow(1,snow,reshape([0._r_std],[1,1]),snowrho,snowdz,frac_snow_veg,frac_snow_nobio)\n"
    anchor = b"  call thermosoil_getdiff(kjpindex,ptn,njsc"
    assert unit.count(anchor) == 1
    unit = unit.replace(anchor, call + anchor)
    return unit, [source_record(THERMAL, s.start_line, s.end_line, name) for name, s in spans.items()] + [
        source_record(condpath, cond.start_line, cond.end_line, "actual snow-cover producer")
    ]


def run_coefficient(v, *, use_jit=False):
    import jax
    from jax.experimental import checkify

    from jax_orchidee.sechiba import condveg
    from jax_orchidee.sechiba import thermosoil as th

    def component(data):
        fractions = condveg.condveg_frac_snow(
            snow=data["snow"],
            snow_nobio=np.zeros((1, 1)),
            snowrho=data["snowrho"],
            snowdz=data["snowdz"],
            ok_explicitsnow=True,
        )
        material = th.thermosoil_getdiff_explicit(
            ptn=data["ptn"],
            njsc=data["njsc"],
            veget_max=data["veget_max"],
            shum_ngrnd_permalong=data["shum_ngrnd_permalong"],
            mc_layt=data["mc_layt"],
            tmc_layt_pft=data["tmc_layt_pft"],
            snowrho=data["snowrho"],
            snowtemp=data["snowtemp"],
            pb=data["pb"],
            dlt=data["dlt"],
            zx1=np.zeros((1, 32, 14)),
            ok_laidev=data["ok_lai_i"].astype(bool),
            veget_mask_2d=data["mask_i"].astype(bool),
            nslm=11,
        )
        coef = th.thermosoil_coef_explicit_snow(
            ptn=data["ptn"],
            ptn_pftmean=data["ptn_pftmean"],
            pcapa=material.pcapa,
            pkappa=material.pkappa,
            pcapa_snow=material.pcapa_snow,
            pkappa_snow=material.pkappa_snow,
            snowdz=data["snowdz"],
            snowtemp=data["snowtemp"],
            temp_sol_new=data["temp_sol_new"],
            temp_sol_new_pft=data["temp_sol_new_pft"],
            veget_max=data["veget_max"],
            ok_laidev=data["ok_lai_i"].astype(bool),
            dlt=data["dlt"],
            dz1=data["dz1"],
            zlt=data["zlt"],
            dt_sechiba=data["scalars"][0],
            lambda_thermal=data["scalars"][1],
            phigeoth=data["scalars"][2],
            psnowdzmin=data["scalars"][3],
            frac_snow_veg=fractions.frac_snow_veg,
            frac_snow_nobio=fractions.frac_snow_nobio,
            totfrac_nobio=data["totfrac_nobio"],
            veget_mask_2d=data["mask_i"].astype(bool),
        )
        return material, coef, fractions

    if use_jit:
        error, result = jax.jit(checkify.checkify(component))(v)
        checkify.check_error(error)
        return result
    return component(v)


def coefficient_values(result):
    material, coef, fractions = result
    values = {"getdiff." + k: a for k, a in material._asdict().items() if k != "provenance"}
    values.update({"coef.soil." + k: a for k, a in coef.soil._asdict().items()})
    values.update({"coef." + k: a for k, a in coef._asdict().items() if k != "soil"})
    values.update({"fraction." + k: a for k, a in fractions._asdict().items()})
    return {k: np.asarray(v) for k, v in values.items()}


def restart_source():
    """Exact original nine-read decision fragment; only native IO is injected."""
    lines = THERMAL.read_bytes().splitlines(keepends=True)
    fragment = b"".join(lines[664:714])
    module = """module retained_oracle
use iso_fortran_env,only:real64
implicit none
integer,parameter::r_std=real64,nvm=14,ngrnd=32,nsnow=3
real(r_std),parameter::val_exp=999999._r_std
interface restget_p
module procedure read1,read2,read3
end interface
"""
    for k, shape in COEFFICIENTS.items():
        module += f"real(r_std)::rest_{k}({','.join(map(str, shape))})\n"
    module += "contains\nsubroutine ioconf_setatt_p(a,b)\ncharacter(*)::a,b\nend subroutine\n"
    for rank in (1, 2, 3):
        module += f"""subroutine read{rank}(rid,name,nbp,n1,n2,kjit,flag,a,gather,nbpg,idx)
integer::rid,nbp,n1,n2,kjit,nbpg,idx(:)
logical::flag
character(*)::name,gather
real(r_std)::a({",".join(":" for _ in range(rank))})
select case(name)
"""
        for k, shape in COEFFICIENTS.items():
            if len(shape) == rank:
                module += f"case('{k}');a=rest_{k}\n"
        module += "case default;error stop 'unknown restart variable'\nend select\nend subroutine\n"
    module += "end module\nprogram retained\nuse retained_oracle\nimplicit none\ninteger::iu,ou,rest_id=0,nbp_glo=1,kjit=0,index_g(1)=[1]\nlogical::calculate_coef\ncharacter(1024)::ip,op\n"
    for k, shape in COEFFICIENTS.items():
        module += f"real(r_std)::{k}({','.join(map(str, shape))})\n"
    module += "call get_command_argument(1,ip);call get_command_argument(2,op)\nopen(newunit=iu,file=trim(ip),access='stream',form='unformatted',status='old')\n"
    module += "".join(f"read(iu)rest_{k}\n" for k in COEFFICIENTS)
    module += "close(iu)\n"
    end = "\nopen(newunit=ou,file=trim(op),access='stream',form='unformatted',status='replace')\nwrite(ou)merge(1._r_std,0._r_std,calculate_coef)\n"
    end += "".join(f"write(ou){k}\n" for k in COEFFICIENTS)
    return module.encode() + fragment + end.encode() + b"close(ou)\nend program\n"


def run_retained(inputs, coefficients):
    """Actual public initializer, not a synthetic result shell."""
    from jax_orchidee.sechiba import thermosoil as th

    v = inputs
    original = json.loads((ROOT / "tests/fixtures/physics_branch_thermal_reference.json").read_text())
    # Only portable frozen inputs are used; no source or ignored output lookup.
    base = decode_dict(original["cases"][0]["inputs"])
    restart = {**coefficients, "ptn": v["ptn"], "refSOC": v["refSOC"], "shum_ngrnd_prmlng": v["shum_ngrnd_permalong"]}
    kwargs = {
        k: v[k]
        for k in (
            "temp_sol_new_pft",
            "snowdz",
            "snowrho",
            "snowtemp",
            "njsc",
            "pb",
            "dlt",
            "dz1",
            "zlt",
            "frac_snow_veg",
            "frac_snow_nobio",
            "totfrac_nobio",
        )
    }
    kwargs.update(
        moisture=SimpleNamespace(
            **{
                k: base[k]
                for k in (
                    "shumdiag_perma",
                    "mc_layh",
                    "mcl_layh",
                    "tmc_layh",
                    "mc_layh_pft",
                    "mcl_layh_pft",
                    "tmc_layh_pft",
                )
            }
        ),
        znt=base["znt"],
        dz5=base["dz5"],
        dt_sechiba=float(v["scalars"][0]),
        ok_laidev=np.zeros(14, bool),
    )
    return th.thermosoil_initialize(
        ngrnd=32,
        nscm=12,
        temp_sol_new=v["temp_sol_new"],
        veget_max=v["veget_max"],
        coefficient_inputs=kwargs,
        restart_fields=restart,
    )


def qsat_inputs():
    bins = np.arange(101.0, 369.0)
    temp = np.ravel(np.column_stack((bins - 0.5, bins + 0.5 - 1e-9)))
    return dict(temp=temp, pressure=np.full(temp.shape, 1000.0))


def qsat_source(inputs):
    head = """module qsat_oracle
use iso_fortran_env,only:real64
implicit none
integer,parameter::r_std=real64,i_std=kind(1),min_temp=100,max_temp=370
real(r_std),parameter::zero=0._r_std,un=1._r_std,deux=2._r_std,undemi=.5_r_std,msmlr_h2o=18.02e-3_r_std,msmlr_air=28.964e-3_r_std
real(r_std)::qsfrict(max_temp)
logical::l_qsat_first=.true.,diag_qsat=.false.
integer::numout=6,printlev=0
contains
subroutine ipslerr_p(level,a,b,c,d)
integer::level
character(*)::a,b,c,d
if(level>=3) error stop 'original qsat admission'
end subroutine
"""
    bodies = b"\n".join(extract_procedure_bytes(QSAT, k).span_bytes for k in ("qsfrict_init", "dev_qsatcalc"))
    n = len(inputs["temp"])
    tail = f"end module\nprogram qsat\nuse qsat_oracle\nimplicit none\ninteger::iu,ou\nreal(r_std)::temp({n}),pressure({n}),gradient({n})\ncharacter(1024)::ip,op\ncall get_command_argument(1,ip);call get_command_argument(2,op)\nopen(newunit=iu,file=trim(ip),access='stream',form='unformatted',status='old')\nread(iu)temp;read(iu)pressure;close(iu)\ncall dev_qsatcalc({n},temp,pressure,gradient)\nopen(newunit=ou,file=trim(op),access='stream',form='unformatted',status='replace')\nwrite(ou)qsfrict;write(ou)gradient;close(ou)\nend program\n"
    return head.encode() + bodies + b"\n" + tail.encode()


def parse_layout(raw, layout):
    result, offset = {}, 0
    for name, shape in layout:
        count = int(np.prod(shape))
        result[name] = raw[offset : offset + count].reshape(shape, order="F")
        offset += count
    assert offset == raw.size
    return result


def build_reference():
    v = deep_inputs()
    source, sources = coefficient_source()
    outputs = parse_layout(execute("deep_getdiff_coef", source, v), cold.OUTPUT_LAYOUT)
    fractions = {"fraction.frac_snow_veg": np.ones(1), "fraction.frac_snow_nobio": np.zeros((1, 1))}
    outputs.update(fractions)
    retained = []
    for label, capacity in (("zero", 0.0), ("negative", -1.0), ("high", 266960000.0)):
        coeff = {
            k: outputs["coef.soil." + k] if "coef.soil." + k in outputs else outputs["coef." + k] for k in COEFFICIENTS
        }
        coeff = {k: a.copy() for k, a in coeff.items()}
        coeff["soilcap"][:] = capacity
        raw = execute("retained_" + label, restart_source(), coeff)
        assert raw[0] == 0.0
        expected = parse_layout(raw[1:], COEFFICIENTS.items())
        retained.append(
            dict(
                name=label,
                inputs={k: encode(a) for k, a in coeff.items()},
                expected={k: encode(a) for k, a in expected.items()},
                calculate_coef=False,
            )
        )
    qinput = qsat_inputs()
    raw = execute("qsat_domain", qsat_source(qinput), qinput)
    flux = {k: (np.asarray(a)[:1].copy() if np.asarray(a).ndim else a) for k, a in energy.make_flux_inputs().items()}
    flux["soilcap"] = outputs["coef.soilcap"].copy()
    flux["snowdz"] = v["snowdz"].copy()
    # New same-input original flux consumer, with actual original capacity.
    directory = OUT / "reference/flux_with_generated_capacity"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase06_energy_flux_") as td:
        src = Path(td) / "oracle.f90"
        energy.compose_flux(src, flux)
        source = src.read_bytes()
    # Existing flux writer uses CSV, so preserve its independent execution.
    (directory / "oracle.f90").write_bytes(source)
    with tempfile.TemporaryDirectory(prefix="phase06_energy_flux_") as td:
        src, exe = Path(td) / "oracle.f90", Path(td) / "oracle.exe"
        src.write_bytes(source)
        save(directory / "compile.json", compile_fortran(src, exe, extra_flags=("-cpp",)))
        command = [str(exe), str((directory / "outputs.csv").resolve())]
        process = subprocess.run(
            command, cwd=td, env=compiler_environment(DEFAULT_COMPILER), capture_output=True, text=True
        )
        save(
            directory / "run.json",
            dict(command=command, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr),
        )
        process.check_returncode()
    expected_flux = energy.canopy.read_csv(directory / "outputs.csv")
    reference = dict(
        schema="phase06-energy-predecessors-v1",
        scope="Finite independent component/predecessor proof; no natural history",
        sources=sources,
        coefficient=dict(
            inputs={k: encode(a) for k, a in v.items()}, expected={k: encode(a) for k, a in outputs.items()}
        ),
        retained=retained,
        qsat=dict(
            inputs={k: encode(a) for k, a in qinput.items()}, table=encode(raw[:370]), gradient=encode(raw[370:])
        ),
        flux=dict(inputs=energy.jsonable(flux), expected=energy.jsonable(expected_flux)),
        reused_fixtures={
            k: sha(ROOT / k)
            for k in (
                "tests/fixtures/physics_branch_thermal_reference.json",
                "tests/fixtures/phase05_energy_reference.json",
                "tests/fixtures/snow_maxmass_domain_current_source.json",
            )
        },
    )
    reference["sources"] += [
        source_record(THERMAL, 665, 714, "complete retained restart read/decision"),
        source_record(THERMAL, 716, 742, "var-init followed by conditional coefficient overwrite"),
        source_record(QSAT, 309, 410, "complete dev_qsatcalc"),
        source_record(QSAT, 547, 589, "complete qsfrict_init"),
        source_record(
            ROOT / "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90",
            56,
            88,
            "zero-depth cold default or externally supplied restart",
        ),
    ]
    save(FIXTURE, reference)


def build_adjacent_reference():
    """Add one new coefficient case from preserved original snow return.

    The first original snow call is never rerun. Its complete layer return is
    used directly, while material/thermal inputs remain explicitly declared.
    """
    reference = json.loads(FIXTURE.read_text())
    v = decode_dict(reference["coefficient"]["inputs"])
    previous = json.loads((ROOT / "tests/fixtures/physics_branch_thermal_reference.json").read_text())[
        "snow_predecessor"
    ]
    original = previous["calls"][0]["expected"]
    for name in ("snowdz", "snowrho", "snowtemp"):
        v[name] = decode(original[name]).reshape(1, 3)
    assert np.array_equal(v["snowdz"].ravel(), previous["calls"][1]["entry_snowdz"])
    v["snow"] = np.sum(v["snowrho"] * v["snowdz"], axis=1)
    source, _ = coefficient_source()
    expected = parse_layout(execute("adjacent_getdiff_coef", source, v), cold.OUTPUT_LAYOUT)
    expected.update({"fraction.frac_snow_veg": np.ones(1), "fraction.frac_snow_nobio": np.zeros((1, 1))})
    reference["adjacent_coefficient"] = dict(
        inputs={k: encode(a) for k, a in v.items()},
        expected={k: encode(a) for k, a in expected.items()},
        producer=dict(
            fixture="tests/fixtures/physics_branch_thermal_reference.json",
            call=0,
            consumer_call=1,
            fields=["snowdz", "snowrho", "snowtemp"],
            class_of_evidence="actual original snow predecessor with synthetic thermal/material state; not natural history",
            old_independent_capacity=266960000.0,
            snow_source_sha256=previous["source_sha256"],
        ),
    )
    save(FIXTURE, reference)


def certify(reference):
    from jax import jit

    from jax_orchidee.sechiba.enerbil import _qsfrict_table, qsat_moisture_dev_qsatcalc
    from jax_orchidee.sechiba.hydrol import explicitsnow_initialize_zero_state

    rows, observations = [], {}
    v = decode_dict(reference["coefficient"]["inputs"])
    expected = decode_dict(reference["coefficient"]["expected"])
    for mode in ("eager", "jit"):
        result = run_coefficient(v, use_jit=mode == "jit")
        actual = coefficient_values(result)
        for k, value in expected.items():
            original, current = value.ravel(order="F"), actual[k].ravel(order="F")
            rows.append(
                dict(
                    case="deep_coefficient",
                    mode=mode,
                    failed_indices=np.flatnonzero(~np.isclose(original, current, **THERMAL_TOL)).tolist(),
                    **float_comparison(k, original, current, **THERMAL_TOL),
                )
            )
        flux = {**reference["flux"]["inputs"], "soilcap": np.asarray(result[1].soilcap)}
        values = energy.flux_values(energy.run_flux(flux, use_jit=mode == "jit"))
        for k, value in reference["flux"]["expected"].items():
            if k != "warning":
                rows.append(
                    dict(
                        case="generated_capacity_flux",
                        mode=mode,
                        **float_comparison(k, np.asarray(value), values[k].ravel(), **energy.TOLERANCE),
                    )
                )
        qinput = decode_dict(reference["qsat"]["inputs"])
        function = jit(qsat_moisture_dev_qsatcalc) if mode == "jit" else qsat_moisture_dev_qsatcalc
        gradient = function(qinput["temp"], qinput["pressure"])
        rows.append(
            dict(
                case="qsat_domain",
                mode=mode,
                **float_comparison("gradient", decode(reference["qsat"]["gradient"]), gradient, **energy.TOLERANCE),
            )
        )
        observations[mode] = dict(
            generated_capacity=np.asarray(result[1].soilcap).tolist(),
            min_dgrnd=float(np.min(result[1].soil.dgrnd)),
            max_dgrnd=float(np.max(result[1].soil.dgrnd)),
            min_pcapa=float(np.min(result[0].pcapa)),
            min_pkappa=float(np.min(result[0].pkappa)),
            min_snow_pcapa=float(np.min(result[0].pcapa_snow)),
            min_snow_pkappa=float(np.min(result[0].pkappa_snow)),
            min_gradient=float(np.min(gradient)),
            max_gradient=float(np.max(gradient)),
            snow_mass=float(np.sum(v["snowrho"] * v["snowdz"])),
            snow_depth=float(np.sum(v["snowdz"])),
            snow_cap_upper=float(
                v["scalars"][0] * (np.asarray(result[1].zdz2_snow)[0, 0] + np.asarray(result[1].zdz1_snow)[0, 0])
            ),
            full_coefficient_values={k: encode(a) for k, a in actual.items()},
        )
        if "adjacent_coefficient" in reference:
            adjacent_inputs = decode_dict(reference["adjacent_coefficient"]["inputs"])
            adjacent = coefficient_values(run_coefficient(adjacent_inputs, use_jit=mode == "jit"))
            for k, original in decode_dict(reference["adjacent_coefficient"]["expected"]).items():
                current = adjacent[k].ravel(order="F")
                original = original.ravel(order="F")
                rows.append(
                    dict(
                        case="adjacent_coefficient",
                        mode=mode,
                        failed_indices=np.flatnonzero(~np.isclose(original, current, **THERMAL_TOL)).tolist(),
                        **float_comparison(k, original, current, **THERMAL_TOL),
                    )
                )
            observations[mode]["adjacent"] = dict(
                generated_capacity=adjacent["coef.soilcap"].tolist(),
                snow_depth=float(np.sum(adjacent_inputs["snowdz"])),
                snow_mass=float(np.sum(adjacent_inputs["snowdz"] * adjacent_inputs["snowrho"])),
                snow_cap_upper=float(
                    adjacent_inputs["scalars"][0]
                    * (adjacent["coef.zdz2_snow"][0, 0] + adjacent["coef.zdz1_snow"][0, 0])
                ),
                full_coefficient_values={k: encode(a) for k, a in adjacent.items()},
            )
    for case in reference["retained"]:
        actual = run_retained(v, decode_dict(case["inputs"]))
        assert not actual.calculate_coef
        for k, expected_value in decode_dict(case["expected"]).items():
            rows.append(
                dict(
                    case="retained_" + case["name"],
                    mode="public_initialize",
                    **float_comparison(k, expected_value, getattr(actual, k), rtol=0.0, atol=0.0),
                )
            )
    # Original table is indexed1:370; production keeps an unused index0.
    table = np.asarray(_qsfrict_table())[1:]
    rows.append(
        dict(
            case="qsat_domain",
            mode="table",
            **float_comparison("qsfrict", decode(reference["qsat"]["table"]), table, **energy.TOLERANCE),
        )
    )
    state = explicitsnow_initialize_zero_state(kjpindex=1)
    observations["cold_snow"] = dict(
        depth=float(np.sum(state.snowdz)), mass=float(np.sum(state.snowrho * state.snowdz))
    )
    return rows, observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--adjacent-reference", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reference:
        build_reference()
    if args.adjacent_reference:
        build_adjacent_reference()
    reference = json.loads(FIXTURE.read_text())
    rows, observations = certify(reference)
    save(OUT / "checks.json", dict(checks=rows, passed=all(r["passed"] for r in rows), fixture_sha256=sha(FIXTURE)))
    save(OUT / "observations.json", observations)
    failed = [r for r in rows if not r["passed"]]
    print(
        json.dumps(
            dict(
                checks=len(rows),
                failed=failed,
                observations={
                    mode: {k: a for k, a in values.items() if k not in ("full_coefficient_values", "adjacent")}
                    for mode, values in observations.items()
                },
            ),
            indent=2,
        )
    )
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
