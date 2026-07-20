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

from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    SN_CAPA,
    add_heat_zimov,
    thermosoil_energy_diagnostics,
    thermosoil_getdiff_thinsnow,
    thermosoil_humlev,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)


FAMILY = "thermosoil_aux_transitions"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_thermosoil_aux_transitions.f90.template"
PROCEDURES = {
    "thermosoil_humlev",
    "thermosoil_energy",
    "add_heat_zimov",
    "thermosoil_getdiff_thinsnow",
}
NPTS = 5
NVM = 14
NGRND = 6
NSLM = 4


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    generated = TEMPLATE.read_bytes()
    for name, span in spans.items():
        generated = generated.replace(f"! <{name.upper()}>".encode(), span.span_bytes)
    path.write_bytes(generated)
    return {name: span.span_sha256 for name, span in spans.items()}


def _inputs() -> dict[str, np.ndarray]:
    znt = np.asarray([0.01, 0.06, 0.20, 0.60, 1.20, 2.0])
    zlt = np.asarray([0.007, 0.10, 0.30, 0.80, 1.50, 3.0])
    dz5 = np.asarray([0.25, 0.40, 0.60, 0.70, 0.80])
    dlt = np.asarray([0.02, 0.08, 0.20, 0.50, 0.70, 1.50])
    indices = np.indices((NPTS, NGRND, NVM))
    ii, jj, kk = (axis + 1 for axis in indices)
    ptn = 273.15 * np.ones((NPTS, NGRND, NVM))
    ptn[:, :, 0] = 270.0
    ptn[:, :, 1] = 276.0
    shum = 0.05 + 0.01 * ii + 0.002 * jj + 0.001 * kk
    snowdz = np.zeros((NPTS, 3))
    snowdz[:, 0] = [0.0, 0.005, 0.009, 0.020, 0.010]

    i2, j2 = np.indices((NPTS, NSLM))
    i2, j2 = i2 + 1, j2 + 1
    shumdiag = 0.1 * i2 + 0.01 * j2
    mc_h = 0.15 + 0.01 * i2 + 0.02 * j2
    mcl_h = 0.10 + 0.008 * i2 + 0.015 * j2
    tmc_h = 10.0 * i2 + j2
    pft_index = np.arange(1, NVM + 1)[None, None, :]
    mc_hp = mc_h[:, :, None] + 0.001 * pft_index
    mcl_hp = mcl_h[:, :, None] + 0.0005 * pft_index
    tmc_hp = tmc_h[:, :, None] + 0.1 * pft_index
    mc_hp[0, 0, 0] = 0.0
    mcl_hp[0, 0, 0] = 0.0

    base_ptn = 265.0 + 0.1 * ii + 0.01 * jj + 0.001 * kk
    heat = 0.01 * ii + 0.001 * jj + 0.0001 * kk
    heat_pcapa = 1.5e6 + 1000.0 * jj + 10.0 * kk
    veget = np.zeros((NPTS, NVM))
    veget[:, 0] = 0.2
    veget[:, 13] = 0.8
    return {
        "znt": znt,
        "zlt": zlt,
        "dz5": dz5,
        "dlt": dlt,
        "thin_ptn": ptn,
        "thin_shum": shum,
        "snowdz": snowdz,
        "shumdiag": shumdiag,
        "mc_h": mc_h,
        "mcl_h": mcl_h,
        "tmc_h": tmc_h,
        "mc_hp": mc_hp,
        "mcl_hp": mcl_hp,
        "tmc_hp": tmc_hp,
        "zimov_ptn": base_ptn,
        "heat": heat,
        "heat_pcapa": heat_pcapa,
        "veget": veget,
    }


def _jax_outputs(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    shape = (NPTS, NGRND, NVM)
    mask = np.ones((NPTS, NVM), dtype=bool)
    thin = thermosoil_getdiff_thinsnow(
        ptn=inputs["thin_ptn"],
        shum_ngrnd_permalong=inputs["thin_shum"],
        snowdz=inputs["snowdz"],
        pcapa=np.full(shape, 111.0),
        pcapa_en=np.full(shape, 222.0),
        pkappa=np.full(shape, 0.9),
        profil_froz=np.full(shape, 0.25),
        veget_mask_2d=mask,
        zlt=inputs["zlt"],
    )

    energy_ptn = np.empty(shape)
    ii, jj, kk = (axis + 1 for axis in np.indices(shape))
    energy_ptn[:] = 260.0 + 0.1 * ii + 0.01 * jj + 0.001 * kk
    energy_pcapa = np.full(shape, 1.0e6)
    energy_pcapa[0, 0, :] = 1.0e5
    energy_pcapa[2, 0, :] = SN_CAPA
    energy = thermosoil_energy_diagnostics(
        temp_sol_new=np.asarray([271.0, 269.0, 273.0, 275.0, 272.0]),
        temp_sol_beg=np.asarray([270.0, 271.0, 272.0, 273.0, 274.0]),
        soilcap=np.asarray([100.0, 200.0, 300.0, 400.0, 500.0]),
        pcapa_en=energy_pcapa,
        veget_max=inputs["veget"],
        ptn=energy_ptn,
        sn_capa=SN_CAPA,
    )

    humlev = thermosoil_humlev(
        shumdiag_perma=inputs["shumdiag"],
        mc_layh=inputs["mc_h"],
        mcl_layh=inputs["mcl_h"],
        tmc_layh=inputs["tmc_h"],
        mc_layh_pft=inputs["mc_hp"],
        mcl_layh_pft=inputs["mcl_hp"],
        tmc_layh_pft=inputs["tmc_hp"],
        znt=inputs["znt"],
        zlt=inputs["zlt"],
        dz5=inputs["dz5"],
        veget_mask_2d=mask,
        satsoil=False,
    )
    zimov = add_heat_zimov(
        ptn=inputs["zimov_ptn"],
        heat_zimov=inputs["heat"],
        pcapa=inputs["heat_pcapa"],
        dlt=inputs["dlt"],
        dt_sechiba=1800.0,
        veget_mask_2d=mask,
        veget_max_bg=inputs["veget"],
    )
    outputs = {
        "thin.pcapa": thin.pcapa,
        "thin.pcapa_en": thin.pcapa_en,
        "thin.pkappa": thin.pkappa,
        "thin.profil_froz": thin.profil_froz,
        "energy.surfheat_incr": energy.surfheat_incr,
        "energy.coldcont_incr": energy.coldcont_incr,
        "energy.ptn_beg": energy.ptn_beg,
        "energy.temp_sol_beg": energy.temp_sol_beg,
        "humlev.mc_layt": humlev.mc_layt,
        "humlev.mcl_layt": humlev.mcl_layt,
        "humlev.tmc_layt": humlev.tmc_layt,
        "humlev.mc_layt_pft": humlev.mc_layt_pft,
        "humlev.mcl_layt_pft": humlev.mcl_layt_pft,
        "humlev.tmc_layt_pft": humlev.tmc_layt_pft,
        "humlev.shum_ngrnd_perma": humlev.shum_ngrnd_perma,
        "zimov.ptn": zimov.ptn,
        "zimov.ptn_pftmean": zimov.ptn_pftmean,
    }
    return {name: np.asarray(value).ravel(order="C") for name, value in outputs.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_aux_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        span_hashes = compose(source)
        compiler_meta = compile_fortran(source, executable, compiler)
        output_path = output_dir / "fortran_outputs.csv"
        subprocess.run(
            [str(executable), str(output_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    inputs = _inputs()
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs(inputs)
    if set(fortran) != set(jax):
        raise RuntimeError(f"output fields differ: Fortran={sorted(fortran)}, JAX={sorted(jax)}")
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14)
        for name in sorted(fortran)
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "no snow, mixed thin snow, full first-level thin snow, and non-thin snow",
                    "frozen, transition, and unfrozen first-level temperatures",
                    "snow-like and soil-like energy capacity threshold sides",
                    "first, second, interior, last hydrology levels and deep thermal carry",
                    "PFT1/PFT14 weighted Zimov heat state writeback",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "procedure_span_sha256": span_hashes,
            "compiler": compiler_meta,
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
