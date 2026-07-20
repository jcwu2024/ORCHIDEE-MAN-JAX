from __future__ import annotations
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from fortran_oracle_common import (
    DEFAULT_COMPILER,
    ROOT,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEMPLATE = ROOT / "scripts/dev/diffuco_surface_exchange_oracle_harness.f90.template"
EXTRACTOR = ROOT / "scripts/dev/extract_fortran_micro_oracle.py"
FAMILY = "diffuco_surface_exchange"


def _extracts():
    spec = importlib.util.spec_from_file_location("diff_oracle_extractor", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {
        entry["procedure"]: span
        for entry, span in module.validate_and_extract(
            module.load_manifest(), root=ROOT
        )
    }


def _compose(path):
    s = _extracts()
    q = b"\n\n".join(s[n].span_bytes for n in ("qsfrict_init", "qsatcalc"))
    d = b"\n\n".join(
        s[n].span_bytes
        for n in (
            "diffuco_aero",
            "diffuco_snow",
            "diffuco_inter",
            "diffuco_bare",
            "diffuco_comb",
        )
    )
    path.write_bytes(
        TEMPLATE.read_bytes()
        .replace(b"! <QSAT_PROCEDURES>", q)
        .replace(b"! <DIFFUCO_PROCEDURES>", d)
    )
    return {
        n: s[n].span_sha256
        for n in (
            "diffuco_aero",
            "diffuco_snow",
            "diffuco_inter",
            "diffuco_bare",
            "diffuco_comb",
            "qsatcalc",
        )
    }


def _read(path):
    values = {}
    with path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {k: np.asarray(v) for k, v in values.items()}


def _jax():
    from jax_orchidee.sechiba.diffuco import (
        diffuco_aero_explicit,
        diffuco_snow_beta_explicit,
        diffuco_inter_explicit,
        diffuco_bare_cwrr_beta,
        diffuco_bare_non_cwrr_beta,
        diffuco_comb_explicit,
    )
    from jax_orchidee.sechiba.enerbil import qsat_moisture_qsatcalc

    n, nvm = 3, 14
    u = np.array([0.02, 2.0, 4.0])
    v = np.array([0.0, 1.0, 0.0])
    z = np.full(n, 10.0)
    rh = np.ones(n)
    rhp = np.full((n, nvm), 1.2)
    ts = np.array([285.0, 290.0, 265.0])
    ta = np.array([290.0, 285.0, 270.0])
    tsp = np.repeat(ts[:, None], nvm, 1)
    tsp[:, 13] = ts + 1.0
    qair = np.array([0.005, 0.1, 0.1])
    snow = np.array([0.0, 2.0, 2.0])
    mask = np.zeros(nvm, bool)
    mask[13] = True
    aero = diffuco_aero_explicit(
        u=u,
        v=v,
        zlev=z,
        z0h=np.full(n, 0.1),
        z0m=np.full(n, 0.15),
        roughheight=rh,
        roughheight_pft=rhp,
        temp_sol=ts,
        temp_sol_pft=tsp,
        temp_air=ta,
        qsurf=np.full(n, 0.006),
        qair=qair,
        snow=snow,
        ok_laidev=mask,
    )
    aero_snowfact = diffuco_aero_explicit(
        u=u,
        v=v,
        zlev=z,
        z0h=np.full(n, 0.1),
        z0m=np.full(n, 0.15),
        roughheight=rh,
        roughheight_pft=rhp,
        temp_sol=ts,
        temp_sol_pft=tsp,
        temp_air=ta,
        qsurf=np.full(n, 0.006),
        qair=qair,
        snow=snow,
        ok_laidev=mask,
        ok_snowfact=True,
        rough_dyn=False,
    )
    qc = np.asarray(aero.q_cdrag)
    qcp = np.asarray(aero.q_cdrag_pft)
    qs = np.asarray(qsat_moisture_qsatcalc(ts, np.full(n, 1000.0)))
    rau = np.array([1.2, 1.1, 1.3])
    frac = np.full((n, 2), 0.05)
    sn = diffuco_snow_beta_explicit(
        qair=qair,
        qsatt=qs,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=qc,
        snow=snow,
        frac_nobio=frac,
        totfrac_nobio=np.full(n, 0.1),
        snow_nobio=np.full((n, 2), 0.02),
        frac_snow_veg=np.array([0.0, 0.8, 1.0]),
        frac_snow_nobio=np.full((n, 2), 0.5),
        dt_sechiba=1800.0,
    )
    hum = np.full((n, nvm), 0.7)
    veg = np.zeros((n, nvm))
    veg[:, 13] = [0.4, 0.7, 0.8]
    veg[0, 1:3] = 0.2
    qsv = np.zeros((n, nvm))
    qsv[:, 13] = [1e-8, 0.2, 0.01]
    qsv[0, 1:3] = [0.1, 1e-8]
    qsm = np.full((n, nvm), 0.3)
    qsm[0, 1] = 0.0
    hum[0, 2] = 0.0
    rstruct = np.full((n, nvm), 50.0)
    inter = diffuco_inter_explicit(
        qair=qair,
        qsatt=qs,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=qc,
        q_cdrag_pft=qcp,
        humrel=hum,
        veget=veg,
        qsintveg=qsv,
        qsintmax=qsm,
        rstruct=rstruct,
        ok_laidev=mask,
        dt_sechiba=1800.0,
    )
    v3 = np.zeros((n, nvm))
    v3[:, 13] = 0.1
    vmax = veg.copy()
    vmax[:, 0] = 0.2
    vmax[:, 13] = 0.8
    bare = diffuco_bare_cwrr_beta(np.array([0.2, 0.4, 0.6]), inter.vbeta2, v3, vmax)
    bare_non_cwrr = diffuco_bare_non_cwrr_beta(
        u=u,
        v=v,
        q_cdrag=qc,
        rsol=np.full(n, 100.0),
        tot_bare_soil=np.array([0.0, 0.2, 0.2]),
        nvm=nvm,
    )
    qsm2 = np.zeros((n, nvm))
    qsm2[:, 13] = 0.3
    comb = diffuco_comb_explicit(
        humrel=hum,
        qair=qair,
        temp_air=ta,
        qsatt=qs,
        veget=veg,
        veget_max=vmax,
        lai=np.ones((n, nvm)),
        tot_bare_soil=np.full(n, 0.2),
        vbeta1=sn.vbeta1,
        vbeta2=inter.vbeta2,
        vbeta3=v3,
        vbeta4=bare.vbeta4,
        vbeta4_pft=bare.vbeta4_pft,
        qsintmax=qsm2,
        dew_veg_poly_coeff=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    qsm3 = np.zeros((n, nvm))
    qsm3[:, 13] = 0.3
    qsm3[1, 12] = 0.3
    lai3 = np.ones((n, nvm))
    lai3[1, 12] = 0.0
    lai3[1, 13] = 2.0
    comb2 = diffuco_comb_explicit(
        humrel=hum,
        qair=qair,
        temp_air=ta,
        qsatt=qs,
        veget=veg,
        veget_max=vmax,
        lai=lai3,
        tot_bare_soil=np.array([0.2, 0.0, 0.2]),
        vbeta1=sn.vbeta1,
        vbeta2=inter.vbeta2,
        vbeta3=v3,
        vbeta4=bare.vbeta4,
        vbeta4_pft=bare.vbeta4_pft,
        qsintmax=qsm3,
        dew_veg_poly_coeff=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    def flat(value):
        return np.asarray(value).ravel(order="C")

    return {
        "aero_qc": qc,
        "aero_qcp": qcp[:, 1:].ravel(),
        "aero_snowfact_qc": np.asarray(aero_snowfact.q_cdrag),
        "snow_vb1": np.asarray(sn.vbeta1),
        "inter_vb2": flat(inter.vbeta2),
        "inter_vb23": flat(inter.vbeta23),
        "bare_v4": np.asarray(bare.vbeta4),
        "bare_v4p": flat(bare.vbeta4_pft),
        "bare_non_cwrr_v4": np.asarray(bare_non_cwrr.vbeta4),
        "comb_alpha": np.asarray(comb.valpha),
        "comb_beta": np.asarray(comb.vbeta),
        "comb_vb1": np.asarray(comb.vbeta1),
        "comb_vb4": np.asarray(comb.vbeta4),
        "comb_betap": flat(comb.vbeta_pft),
        "comb_vb2": flat(comb.vbeta2),
        "comb_vb3": flat(comb.vbeta3),
        "comb_hum": flat(comb.humrel),
        "comb2_alpha": np.asarray(comb2.valpha),
        "comb2_beta": np.asarray(comb2.vbeta),
        "comb2_vb1": np.asarray(comb2.vbeta1),
        "comb2_vb4": np.asarray(comb2.vbeta4),
        "comb2_betap": flat(comb2.vbeta_pft),
        "comb2_vb2": flat(comb2.vbeta2),
        "comb2_vb3": flat(comb2.vbeta3),
        "comb2_hum": flat(comb2.humrel),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_diffuco_oracle_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        hashes = _compose(source)
        exe = build / "oracle.exe"
        metadata = compile_fortran(source, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    f = _read(csv_path)
    j = _jax()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "npts": 3,
                "nvm": 14,
                "cases": [
                    "minimum-wind stable dry surface",
                    "unstable warm-dew snow-covered surface",
                    "freezing-dew snow-covered surface",
                ],
                "coverage": [
                    "PFT1 and PFT14",
                    "ok_LAIdev false and true",
                    "snow supply limitation",
                    "wet canopy storage limitation",
                    "CWRR bare soil",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", f, j, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, f[name], j[name], rtol=1e-12, atol=1e-14) for name in f
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "ledger_entries": [
                "diffuco.active.aerodynamic_boundary",
                "diffuco.conditional.snow_sublimation",
                "diffuco.conditional.canopy_interception",
                "diffuco.active.cwrr_bare_soil",
                "diffuco.conditional.dew_freezing_combination",
            ],
            "span_sha256": hashes,
            "build": metadata,
        },
    )
