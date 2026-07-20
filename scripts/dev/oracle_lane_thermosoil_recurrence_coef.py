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

FAMILY = "thermosoil_cwrr_recurrence_coef"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_thermosoil_recurrence_coef.f90.template"


def _span(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _read(path: Path, prefix: str = "", exclude: frozenset[str] = frozenset()) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            if row["field"] in exclude:
                continue
            values.setdefault(prefix + row["field"], []).append(float(row["value"]))
    return {key: np.asarray(value) for key, value in values.items()}


def _jax() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.thermosoil import (
        thermosoil_coef_explicit_snow,
        thermosoil_coef_no_explicit_snow,
        thermosoil_getdiff_explicit,
        thermosoil_getdiff_old_thermix_with_snow,
    )

    ptn = np.full((3, 4, 14), 273.15)
    ptn[0] = 270.0
    ptn[2] = 276.0
    shum = np.full_like(ptn, 0.6)
    veget = np.zeros((3, 14))
    veget[:, 0] = 0.2
    veget[:, 13] = 0.8
    mask = np.ones((3, 14), dtype=bool)
    mc = np.full((3, 4), 0.3)
    tmc = np.full((3, 4, 14), 100.0)
    tmc[:, :, 13] = 150.0
    snowrho = np.full((3, 3), 200.0)
    snowrho[2] = 350.0
    snowtemp = np.full((3, 3), 268.0)
    refsoc = np.zeros((3, 4))
    refsoc[:, 0] = [0.0, 65000.0, 130000.0]
    refsoc[:, 1] = 32500.0
    zx1 = np.broadcast_to(np.minimum(refsoc[:, :, None] / 130000.0, 1.0), ptn.shape)
    getdiff = thermosoil_getdiff_explicit(
        ptn=ptn,
        njsc=np.array([1, 4, 10]),
        veget_max=veget,
        shum_ngrnd_permalong=shum,
        mc_layt=mc,
        tmc_layt_pft=tmc,
        snowrho=snowrho,
        snowtemp=snowtemp,
        pb=np.array([1000.0, 900.0, 800.0]),
        dlt=np.array([0.1, 0.2, 0.5, 1.0]),
        zx1=zx1,
        ok_laidev=np.array([False] * 13 + [True]),
    )
    temp = np.array([271.0, 274.0, 278.0])
    temp_pft = np.broadcast_to(temp[:, None], (3, 14)).copy()
    temp_pft[:, 13] = temp + 2.0
    snowdz = np.full((3, 3), 0.02)
    snowdz[0, 0] = 0.0
    snowdz[1] = [0.01, 0.03, 0.06]
    frac_nobio = np.zeros((3, 2))
    frac_nobio[1] = 0.2
    ptn_mean = np.sum(ptn * veget[:, None, :], axis=2)
    coef = thermosoil_coef_explicit_snow(
        ptn=ptn,
        ptn_pftmean=ptn_mean,
        pcapa=getdiff.pcapa,
        pkappa=getdiff.pkappa,
        pcapa_snow=getdiff.pcapa_snow,
        pkappa_snow=getdiff.pkappa_snow,
        snowdz=snowdz,
        snowtemp=snowtemp,
        temp_sol_new=temp,
        temp_sol_new_pft=temp_pft,
        veget_max=veget,
        ok_laidev=np.array([False] * 13 + [True]),
        dlt=np.array([0.1, 0.2, 0.5, 1.0]),
        dz1=np.array([20.0, 8.0, 3.0]),
        zlt=np.array([0.05, 0.15, 0.4, 1.0]),
        dt_sechiba=1800.0,
        lambda_thermal=0.5,
        frac_snow_veg=np.array([0.0, 0.5, 1.0]),
        frac_snow_nobio=frac_nobio,
        totfrac_nobio=np.array([0.0, 0.1, 0.2]),
        psnowdzmin=0.0001,
        phigeoth=0.05,
        veget_mask_2d=mask,
    )
    explicit = {
        "pcapa": np.asarray(getdiff.pcapa).ravel(),
        "pcapa_en": np.asarray(getdiff.pcapa_en).ravel(),
        "pkappa": np.asarray(getdiff.pkappa).ravel(),
        "profil": np.asarray(getdiff.profil_froz).ravel(),
        "supp": np.asarray(getdiff.pcappa_supp).ravel(),
        "cgrnd": np.asarray(coef.soil.cgrnd).ravel(),
        "dgrnd": np.asarray(coef.soil.dgrnd).ravel(),
        "soilcap_pft": np.asarray(coef.soil.soilcap_pft).ravel(),
        "soilflx_pft": np.asarray(coef.soil.soilflx_pft).ravel(),
        "pcapa_snow": np.asarray(getdiff.pcapa_snow).ravel(),
        "pkappa_snow": np.asarray(getdiff.pkappa_snow).ravel(),
        "cgrnd_snow": np.asarray(coef.cgrnd_snow).ravel(),
        "dgrnd_snow": np.asarray(coef.dgrnd_snow).ravel(),
        "soilcap": np.asarray(coef.soilcap).ravel(),
        "soilflx": np.asarray(coef.soilflx).ravel(),
        "lambda_snow": np.asarray(coef.lambda_snow).ravel(),
    }
    old = thermosoil_getdiff_old_thermix_with_snow(
        snow=np.array([0.0, 10.0, 50.0]),
        njsc=np.array([1, 4, 10]),
        mc_layt=mc,
        mcl_layt=np.full((3, 4), 0.25),
        tmc_layt=np.full((3, 4), 100.0),
        mc_layt_pft=np.broadcast_to(mc[:, :, None], (3, 4, 14)),
        mcl_layt_pft=np.full((3, 4, 14), 0.25),
        tmc_layt_pft=tmc,
        dlt=np.array([0.1, 0.2, 0.5, 1.0]),
        zlt=np.array([0.05, 0.15, 0.4, 1.0]),
        ok_laidev=np.array([False] * 13 + [True]),
    )
    no_explicit = thermosoil_coef_no_explicit_snow(
        ptn=ptn, pcapa=old.pcapa, pkappa=old.pkappa,
        temp_sol_new=temp, temp_sol_new_pft=temp_pft, veget_max=veget,
        ok_laidev=np.array([False] * 13 + [True]),
        dlt=np.array([0.1, 0.2, 0.5, 1.0]), dz1=np.array([20.0, 8.0, 3.0]),
        dt_sechiba=1800.0, lambda_thermal=0.5,
        cgrnd_snow_template=np.zeros((3, 3)), dgrnd_snow_template=np.zeros((3, 3)),
        phigeoth=0.05, veget_mask_2d=mask,
    )
    old_defined = {
        "pcapa": np.asarray(old.pcapa).ravel(), "pcapa_en": np.asarray(old.pcapa_en).ravel(),
        "pkappa": np.asarray(old.pkappa).ravel(), "cgrnd": np.asarray(no_explicit.soil.cgrnd).ravel(),
        "dgrnd": np.asarray(no_explicit.soil.dgrnd).ravel(), "soilcap_pft": np.asarray(no_explicit.soil.soilcap_pft).ravel(),
        "soilflx_pft": np.asarray(no_explicit.soil.soilflx_pft).ravel(), "cgrnd_snow": np.asarray(no_explicit.cgrnd_snow).ravel(),
        "dgrnd_snow": np.asarray(no_explicit.dgrnd_snow).ravel(), "soilcap": np.asarray(no_explicit.soilcap).ravel(),
        "soilflx": np.asarray(no_explicit.soilflx).ravel(), "lambda_snow": np.asarray(no_explicit.lambda_snow).ravel(),
    }
    return {
        **{"explicit." + key: value for key, value in explicit.items()},
        **{"no_explicit." + key: value for key, value in old_defined.items()},
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spans = {
        "thermosoil_cond_pft": _span(2007, 2127),
        "thermosoil_cond_nopft": _span(2129, 2231),
        "thermosoil_cond": _span(1883, 1972),
        "thermosoil_getdiff": _span(2566, 2863),
        "thermosoil_getdiff_old_thermix_with_snow": _span(2884, 3031),
        "thermosoil_coef": _span(1386, 1729),
    }
    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_coef_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        generated = TEMPLATE.read_bytes()
        for marker, span in spans.items():
            generated = generated.replace(f"! <{marker.upper()}>".encode(), span)
        source.write_bytes(generated)
        executable = build / "oracle.exe"
        compiler_meta = compile_fortran(source, executable, compiler)
        csv_path = output_dir / "fortran_outputs.csv"
        old_csv_path = output_dir / "fortran_outputs_no_explicit.csv"
        subprocess.run([str(executable), str(csv_path.resolve()), "explicit"], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
        subprocess.run([str(executable), str(old_csv_path.resolve()), "no_explicit"], cwd=build, env=compiler_environment(compiler), check=True, capture_output=True, text=True)
    fortran = _read(output_dir / "fortran_outputs.csv", "explicit.")
    old_fortran = _read(
        output_dir / "fortran_outputs_no_explicit.csv",
        "no_explicit.",
        frozenset({"profil", "supp", "pcapa_snow", "pkappa_snow"}),
    )
    fortran.update(old_fortran)
    jax = _jax()
    (output_dir / "inputs.json").write_text(json.dumps({"schema_version": 1, "cases": ["explicit snow zero/mixed/full cover", "non-explicit old thermix snow depths", "PFT14 LAIdev", "frozen/transition/unfrozen refSOC"]}, indent=2) + "\n", encoding="ascii")
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14)
    comparisons = [float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14) for name in fortran]
    return write_result(output_dir, FAMILY, comparisons, {
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "procedure_span_sha256": {name: hashlib.sha256(span).hexdigest() for name, span in spans.items()},
        "compiler": compiler_meta,
        "defined_output_status": "verified" if all(item["passed"] for item in comparisons) else "failed",
        "verified_ledger_entries": [],
    })


if __name__ == "__main__":
    try:
        result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        raise
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
