from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

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

FAMILY = "surface_condveg_processes"
FRAGMENT = ROOT / "docs/source_audits/oracle_families/surface_energy_condveg.yaml"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/condveg.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_surface_condveg.f90.template"
PROCEDURES = (
    "condveg_frac_snow",
    "condveg_albedo",
    "condveg_z0cdrag",
    "condveg_z0cdrag_dyn",
    "condveg_main",
)


def _compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    body = b"\n\n".join(spans[name].span_bytes for name in PROCEDURES)
    path.write_bytes(TEMPLATE.read_bytes().replace(b"! <CONDVEG_PROCEDURES>", body))
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def _jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.sechiba.condveg import (
        condveg_albedo_explicit,
        condveg_frac_snow,
        condveg_z0cdrag_dyn,
    )

    n, nvm = 3, 14
    snow = np.array([0.0, 5.0, 20.0])
    snow_nobio = np.array([[0.0], [2.0], [20.0]])
    snowrho = np.array(
        [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0], [200.0, 300.0, 400.0]]
    )
    snowdz = np.zeros((n, 3))
    snowdz[1] = [0.01, 0.02, 0.03]
    snowdz[2] = [0.1, 0.2, 0.3]
    fractions = condveg_frac_snow(
        snow=snow,
        snow_nobio=snow_nobio,
        snowrho=snowrho,
        snowdz=snowdz,
        ok_explicitsnow=True,
        snowcri_alb=10.0,
        sn_dens=330.0,
    )
    veget = np.zeros((n, nvm))
    veget_max = np.zeros_like(veget)
    veget[:, 0] = veget_max[:, 0] = 0.2
    veget[:, 13] = [0.1, 0.4, 0.7]
    veget_max[:, 13] = 0.7
    frac_nobio = np.array([[0.0], [0.1], [0.2]])
    snow_age = np.array([0.0, 5.0, 20.0])
    soilalb_bg = np.array([[0.12, 0.24], [0.18, 0.32], [0.24, 0.40]])
    pft = np.arange(1, nvm + 1, dtype=np.float64)
    veget[0, :] = 0.0
    veget_max[0, :] = 0.0
    frac_nobio[0, 0] = 1.0
    albedo = condveg_albedo_explicit(
        veget=veget,
        veget_max=veget_max,
        drysoil_frac=np.array([0.0, 0.5, 1.0]),
        frac_nobio=frac_nobio,
        totfrac_nobio=frac_nobio[:, 0],
        snow=snow,
        snow_age=snow_age,
        snow_nobio=snow_nobio,
        snow_nobio_age=snow_age[:, None],
        tot_bare_soil=np.array([0.2, 0.2, 0.1]),
        frac_snow_veg=fractions.frac_snow_veg,
        frac_snow_nobio=fractions.frac_snow_nobio,
        alb_bg_modis=True,
        soilalb_bg=soilalb_bg,
        alb_leaf_vis=0.08 + 0.001 * pft,
        alb_leaf_nir=0.25 + 0.002 * pft,
        snowa_aged_vis=np.full(nvm, 0.5),
        snowa_aged_nir=np.full(nvm, 0.3),
        snowa_dec_vis=np.full(nvm, 0.4),
        snowa_dec_nir=np.full(nvm, 0.3),
        fixed_snow_albedo=1e20,
        undef_sechiba=1e20,
        tcst_snowa=10.0,
        alb_ice=(0.6, 0.2),
    )
    mean_albedo = condveg_albedo_explicit(
        veget=veget,
        veget_max=veget_max,
        drysoil_frac=np.array([0.0, 0.5, 1.0]),
        frac_nobio=frac_nobio,
        totfrac_nobio=frac_nobio[:, 0],
        snow=snow,
        snow_age=snow_age,
        snow_nobio=snow_nobio,
        snow_nobio_age=snow_age[:, None],
        tot_bare_soil=np.array([0.2, 0.2, 0.1]),
        frac_snow_veg=fractions.frac_snow_veg,
        frac_snow_nobio=fractions.frac_snow_nobio,
        alb_bg_modis=False,
        soilalb_bg=soilalb_bg,
        alb_leaf_vis=0.08 + 0.001 * pft,
        alb_leaf_nir=0.25 + 0.002 * pft,
        snowa_aged_vis=np.full(nvm, 0.5),
        snowa_aged_nir=np.full(nvm, 0.3),
        snowa_dec_vis=np.full(nvm, 0.4),
        snowa_dec_nir=np.full(nvm, 0.3),
        fixed_snow_albedo=1e20,
        undef_sechiba=1e20,
        tcst_snowa=10.0,
        alb_bare_model=False,
        soilalb_moy=soilalb_bg,
        alb_ice=(0.6, 0.2),
    )
    height = np.ones((n, nvm))
    height[:, 13] = [5.0, 10.0, 20.0]
    lai = np.zeros((n, nvm))
    lai[:, 13] = [0.0, 2.0, 5.0]
    rough = condveg_z0cdrag_dyn(
        veget=veget,
        veget_max=veget_max,
        frac_nobio=frac_nobio,
        totfrac_nobio=frac_nobio[:, 0],
        zlev=np.array([20.0, 30.0, 40.0]),
        height=height,
        temp_air=np.array([260.0, 280.0, 300.0]),
        pb=np.array([800.0, 950.0, 1013.25]),
        u=np.array([0.0, 2.0, 5.0]),
        v=np.array([0.0, 1.0, 0.0]),
        lai=lai,
        frac_snow_veg=fractions.frac_snow_veg,
    )
    return {
        "snow_veg": np.asarray(fractions.frac_snow_veg),
        "snow_nobio": np.asarray(fractions.frac_snow_nobio).ravel(order="F"),
        "albedo": np.asarray(albedo.albedo).ravel(order="F"),
        "albedo_snow": np.asarray(albedo.albedo_snow).ravel(order="F"),
        "alb_bare": np.asarray(albedo.alb_bare).ravel(order="F"),
        "alb_veget": np.asarray(albedo.alb_veget).ravel(order="F"),
        "mean_albedo": np.asarray(mean_albedo.albedo).ravel(order="F"),
        "mean_albedo_snow": np.asarray(mean_albedo.albedo_snow).ravel(order="F"),
        "mean_alb_bare": np.asarray(mean_albedo.alb_bare).ravel(order="F"),
        "mean_alb_veget": np.asarray(mean_albedo.alb_veget).ravel(order="F"),
        "z0m": np.asarray(rough.z0m),
        "z0h": np.asarray(rough.z0h),
        "roughheight": np.asarray(rough.roughheight),
        "roughheight_pft14": np.asarray(rough.roughheight_pft)[:, 13],
        "emis": np.full(n, 0.98),
    }


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_surface_condveg_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        hashes = _compose(source)
        executable = build / "oracle.exe"
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str(csv_path.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran, jax = _read(csv_path), _jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "branch_cases": [
                    "explicit snow zero/nonzero",
                    "MODIS background albedo with aged snow",
                    "dynamic roughness zero/nonzero LAI and minimum wind",
                ],
            },
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
            "ledger_entries": next(
                family["ledger_entries"]
                for family in yaml.safe_load(FRAGMENT.read_text(encoding="utf-8"))[
                    "families"
                ]
                if family["id"] == FAMILY
            ),
            "span_sha256": hashes,
            "build": metadata,
        },
    )


if __name__ == "__main__":
    result = run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
