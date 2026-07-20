from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from extract_fortran_micro_oracle import extract_procedure_bytes
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

FAMILY = "surface_enerbil_main"
ENERBIL = ROOT / "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90"
QSAT = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_surface_enerbil_main.f90.template"
ENERBIL_PROCEDURES = (
    "enerbil_main",
    "enerbil_begin",
    "enerbil_surftemp",
    "enerbil_pottemp",
    "enerbil_flux",
    "enerbil_evapveg",
    "enerbil_t2mdiag",
)
QSAT_PROCEDURES = ("qsatcalc", "dev_qsatcalc", "qsfrict_init")


def _compose(path: Path) -> dict[str, str]:
    enerbil = {
        name: extract_procedure_bytes(ENERBIL, name) for name in ENERBIL_PROCEDURES
    }
    qsat = {name: extract_procedure_bytes(QSAT, name) for name in QSAT_PROCEDURES}
    source = TEMPLATE.read_bytes()
    source = source.replace(
        b"! <ENERBIL_PROCEDURES>",
        b"\n\n".join(enerbil[name].span_bytes for name in ENERBIL_PROCEDURES),
    ).replace(
        b"! <QSAT_PROCEDURES>",
        b"\n\n".join(qsat[name].span_bytes for name in QSAT_PROCEDURES),
    )
    path.write_bytes(source)
    return {
        **{name: span.span_sha256 for name, span in enerbil.items()},
        **{name: span.span_sha256 for name, span in qsat.items()},
    }


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items) for name, items in values.items()}


def _jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.enerbil import enerbil_explicit_local_step

    nvm = 14
    temp_sol = np.array([284.0, 274.0])
    temp_sol_pft = np.repeat(temp_sol[:, None], nvm, axis=1)
    vbeta = np.array([0.4, 0.6])
    q_cdrag = np.array([0.01, 0.012])
    shape = (2, nvm)
    vbeta2 = np.zeros(shape)
    vbeta3 = np.zeros(shape)
    vbeta3pot = np.zeros(shape)
    vbeta4p = np.zeros(shape)
    vbeta2[:, 13] = 0.05
    vbeta3[:, 13] = 0.25
    vbeta3pot[:, 13] = 0.35
    vbeta4p[:, 13] = 0.15
    veget_max = np.zeros(shape)
    veget_max[:, 13] = 0.8
    ok_laidev = np.zeros(nvm, dtype=bool)
    ok_laidev[13] = True
    result = enerbil_explicit_local_step(
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        lwdown=np.array([300.0, 340.0]),
        swnet=np.array([120.0, 180.0]),
        pb=np.array([1013.0, 980.0]),
        emis=np.full(2, 0.98),
        ok_laidev=ok_laidev,
        epot_air=np.array([285.0, 278.0]) * 1004.675,
        petAcoef=np.full(2, 0.01),
        petBcoef=np.array([285.0, 278.0]) * 1004.675,
        qair=np.array([0.006, 0.003]),
        peqAcoef=np.full(2, 0.1),
        peqBcoef=np.array([0.006, 0.003]),
        soilflx=np.array([8.0, 5.0]),
        soilflx_pft=np.repeat(np.array([[8.0], [5.0]]), nvm, axis=1),
        rau=np.array([1.2, 1.15]),
        u=np.array([2.0, 0.05]),
        v=np.array([1.0, 0.02]),
        q_cdrag=q_cdrag,
        q_cdrag_pft=np.repeat(q_cdrag[:, None], nvm, axis=1),
        vbeta=vbeta,
        vbeta_pft=np.repeat(vbeta[:, None], nvm, axis=1),
        valpha=np.ones(2),
        vbeta1=np.array([0.0, 0.3]),
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta3pot=vbeta3pot,
        vbeta4=np.array([0.25, 0.15]),
        vbeta4_pft=vbeta4p,
        vbeta5=np.zeros(2),
        soilcap=np.array([5e5, 4e5]),
        soilcap_pft=np.repeat(np.array([[5e5], [4e5]]), nvm, axis=1),
        veget_max=veget_max,
        dt_sechiba=1800.0,
        q_sol_pot=np.array([0.01, 0.02]),
        temp_sol_pot=np.array([280.0, 270.0]),
        precip_rain=np.array([0.0, 0.001]),
        snowdz=np.array([[0.0, 0.0, 0.0], [0.02, 0.03, 0.04]]),
        temp_air=np.array([285.0, 278.0]),
        pgflux=np.zeros(2),
        ok_explicitsnow=True,
    )
    snow = result.explicit_snow
    return {
        "temp_sol_new": np.asarray(result.surftemp.temp_sol_new),
        "temp_sol_new_pft14": np.asarray(result.surftemp.temp_sol_new_pft)[:, 13],
        "qsurf": np.asarray(result.flux.qsurf),
        "evapot": np.asarray(result.flux.evapot),
        "evapot_corr": np.asarray(result.evapot_corr.evapot_corr),
        "fluxsens": np.asarray(result.flux.fluxsens),
        "fluxlat": np.asarray(result.flux.fluxlat),
        "vevapp": np.asarray(result.flux.vevapp),
        "vevapnu": np.asarray(result.evapveg_grid.vevapnu),
        "vevapsno": np.asarray(result.evapveg_grid.vevapsno),
        "vevapflo": np.asarray(result.evapveg_grid.vevapflo),
        "transpir14": np.asarray(result.evapveg_pft.transpir)[:, 13],
        "transpot14": np.asarray(result.evapveg_pft.transpot)[:, 13],
        "vevapwet14": np.asarray(result.evapveg_pft.vevapwet)[:, 13],
        "vevapnu_pft14": np.asarray(result.evapveg_pft.vevapnu_pft)[:, 13],
        "pgflux": np.asarray(snow.pgflux),
        "temp_sol_add": np.asarray(snow.temp_sol_add),
        "history_call_count": np.asarray([11.0]),
    }


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_enerbil_main_") as td:
        source = Path(td) / "oracle.f90"
        hashes = _compose(source)
        exe = Path(td) / "oracle.exe"
        metadata = compile_fortran(source, exe, compiler)
        subprocess.run(
            [str(exe), str(csv_path.resolve())],
            cwd=td,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(csv_path)
    jax = _jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {"schema_version": 1, "cases": ["warm no-snow", "explicit-snow PFT14"]},
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, value, jax[name], rtol=1e-12, atol=1e-14)
        for name, value in fortran.items()
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "ledger_entries": ["enerbil.active.surface_energy"],
            "span_sha256": hashes,
            "build": metadata,
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
