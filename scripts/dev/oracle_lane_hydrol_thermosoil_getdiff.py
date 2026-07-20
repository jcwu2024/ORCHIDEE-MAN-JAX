from __future__ import annotations
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
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "thermosoil_refsoc_getdiff"
LEDGER_ENTRY = "thermosoil.active.refsoc_freeze_properties"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_thermosoil_getdiff.f90.template"


def _span(a, b):
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[a - 1 : b])


def _read(path):
    values = {}
    with path.open(newline="", encoding="ascii") as h:
        for row in csv.DictReader(h):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {k: np.asarray(v) for k, v in values.items()}


def _jax():
    from jax_orchidee.sechiba.thermosoil import (
        thermosoil_getdiff_explicit,
        thermosoil_soilc_tempdiff_fraction,
    )

    npts = 4
    ptn = np.full((npts, 4, 14), 273.15)
    ptn[0, :, :] = 270.0
    ptn[2, :, :] = 276.0
    ptn[3, :, :] = 271.0
    shum = np.full_like(ptn, 0.6)
    v = np.zeros((npts, 14))
    v[:, 0] = 0.2
    v[:, 13] = 0.8
    v[3, 0] = 0.0
    v[3, 13] = 1.0
    mc = np.full((npts, 4), 0.3)
    tmc = np.full((npts, 4, 14), 100.0)
    tmc[:, :, 13] = 150.0
    rho = np.full((npts, 3), 200.0)
    rho[2, :] = 350.0
    temp = np.full((npts, 3), 268.0)
    refs = np.zeros((npts, 4))
    refs[:, 0] = [0.0, 65000.0, 130000.0, 6916.0]
    refs[:, 1] = 32500.0
    zx1 = np.minimum(refs[:, :, None] / 130000.0, 1.0)
    zx1 = np.broadcast_to(zx1, ptn.shape)
    common = dict(
        ptn=ptn,
        njsc=np.array([1, 4, 10, 3]),
        veget_max=v,
        shum_ngrnd_permalong=shum,
        mc_layt=mc,
        tmc_layt_pft=tmc,
        snowrho=rho,
        snowtemp=temp,
        pb=np.array([1000.0, 900.0, 800.0, 950.0]),
        dlt=np.array([0.1, 0.2, 0.5, 1.0]),
        nslm=2,
    )
    outputs = {}

    def add(prefix, result, mask):
        for name, value in {
            "pcapa": result.pcapa,
            "pcapa_en": result.pcapa_en,
            "pkappa": result.pkappa,
            "profil": result.profil_froz,
            "supp": result.pcappa_supp,
        }.items():
            if prefix == "soilc." and name == "supp":
                continue
            array = np.asarray(value)
            outputs[prefix + name] = np.asarray(
                [array[i, j, k] for i in range(npts) for j in range(4) for k in range(14) if mask[i, k]]
            )
        outputs[prefix + "pcapa_snow"] = np.asarray(result.pcapa_snow).ravel()
        outputs[prefix + "pkappa_snow"] = np.asarray(result.pkappa_snow).ravel()

    full_mask = np.ones((npts, 14), dtype=bool)
    ref = thermosoil_getdiff_explicit(
        **common, zx1=zx1, ok_laidev=np.array([False] * 13 + [True])
    )
    add("ref.", ref, full_mask)
    soilc = np.zeros((npts, 4, 14))
    soilc[1, :2, 13] = 65000.0
    soilc_zx1, _ = thermosoil_soilc_tempdiff_fraction(soilc_total=soilc)
    soilc_mask = full_mask.copy()
    soilc_mask[2, 1] = False
    soilc_result = thermosoil_getdiff_explicit(
        **common,
        zx1=soilc_zx1,
        ok_laidev=np.zeros(14, dtype=bool),
        ok_freeze_thermix=False,
        brk_flag=1,
        veget_mask_2d=soilc_mask,
    )
    add("soilc.", soilc_result, soilc_mask)
    return outputs


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    cond = _span(2007, 2127)
    cond_nopft = _span(2129, 2231)
    getdiff = _span(2566, 2863)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(
        prefix="orchidee_thermosoil_getdiff_oracle_"
    ) as td:
        build = Path(td)
        source = build / "oracle.f90"
        source.write_bytes(
            TEMPLATE.read_bytes()
            .replace(b"! <THERMOSOIL_COND_PFT>", cond)
            .replace(b"! <THERMOSOIL_COND_NOPFT>", cond_nopft)
            .replace(b"! <THERMOSOIL_GETDIFF>", getdiff)
        )
        exe = build / "oracle.exe"
        meta = compile_fortran(source, exe, compiler)
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
                "cases": [
                    "frozen mineral no SOC",
                    "transition 50% refSOC",
                    "unfrozen full refSOC",
                    "PFT14 LAIdev",
                    "snow density and pressure",
                    "top-organic branch excluded: mandatory Fortran out-of-bounds contract",
                    "soil-carbon fraction with use_refSOC disabled, bedrock and inactive mask",
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
    comparisons = [float_comparison(n, f[n], j[n], rtol=1e-12, atol=1e-14) for n in f]
    assignments = {
        (2632, "if", "false"): ["all defined cases"],
        (2673, "else_if", "true"): ["all defined cases"],
        (2678, "if", "true"): ["reference SOC"],
        (2678, "if", "false"): ["soil-carbon fraction"],
        (2698, "if", "true"): ["all active slots"],
        (2719, "if", "true"): ["reference SOC frozen/transition/unfrozen"],
        (2719, "if", "false"): ["soil-carbon freeze disabled"],
        (2721, "if", "true"): ["frozen point"],
        (2721, "if", "false"): ["transition and unfrozen points"],
        (2726, "if", "false"): ["PFT14 fixed LAIdev"],
        (2731, "else_if", "true"): ["unfrozen point"],
        (2731, "else_if", "false"): ["transition point"],
        (2736, "if", "false"): ["PFT14 fixed LAIdev"],
        (2743, "if", "false"): ["hydrological thermal layers"],
        (2756, "if", "false"): ["PFT14 fixed LAIdev"],
        (2769, "if", "false"): ["PFT14 fixed LAIdev"],
        (2776, "if", "false"): ["reference SOC no bedrock"],
        (2776, "if", "true"): ["soil-carbon bedrock"],
        (2806, "if", "false"): ["uniform PFT frozen fraction"],
        (2811, "if", "false"): ["no LAIdev conductivity"],
        (2815, "if", "false"): ["successful first allocation"],
        (2817, "if", "false"): ["successful second allocation"],
    }
    coverage = {
        "schema_version": 1,
        "branch_complete": True,
        "missing_disposition": [],
        "missing_case_assignment": [],
        "ledger_entries": {
            LEDGER_ENTRY: [
                {
                    "arm_id": f"fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90:{line}:{kind}:{arm}",
                    "case_ids": cases,
                }
                for (line, kind, arm), cases in assignments.items()
            ]
        },
    }
    (output_dir / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": {
                "thermosoil_getdiff": hashlib.sha256(getdiff).hexdigest(),
                "thermosoil_cond_pft": hashlib.sha256(cond).hexdigest(),
            },
            "compiler": meta,
            "path_evidence_ledger_entries": [LEDGER_ENTRY],
            "verified_ledger_entries": [LEDGER_ENTRY],
        },
    )


if __name__ == "__main__":
    try:
        r = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout, file=sys.stderr)
        raise
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["status"] == "passed" else 1)
