from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    hydrol_alma_step,
    hydrol_bucket_snow_step,
    hydrol_waterbal_step,
)
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)
from scripts.dev.oracle_lane_hydrol_soil_owner import (  # noqa: E402
    _allocation_and_reset_source,
    _module_state,
)

SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_hydrol_residual_owners.f90.template"
PROCEDURES = ("hydrol_snow", "hydrol_waterbal", "hydrol_alma")


def compose(path: Path) -> dict[str, str]:
    spans = {name: extract_procedure_bytes(SOURCE, name) for name in PROCEDURES}
    source = TEMPLATE.read_bytes()
    source = source.replace(b"! <MODULE_STATE>", _module_state())
    source = source.replace(b"! <STATE_SETUP>", _allocation_and_reset_source())
    source = source.replace(
        b"! <HYDROL_PROCEDURES>",
        b"\n\n".join(spans[name].span_bytes for name in PROCEDURES),
    )
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in spans.items()}


def _snow_case() -> dict[str, np.ndarray]:
    pr = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    ps = np.array([0, 0.2, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    temp = np.array(
        [272, 274, 274, 274, 270, 274, 270, 274, 270, 274, 270, 274], dtype=float
    )
    frac = np.zeros((12, 1))
    frac[[1, 2, 5, 6, 7], 0] = 0.2
    total_frac = frac[:, 0].copy()
    total_frac[3] = 1.0
    evap = np.array(
        [2, 0.1, 0.1, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float
    )
    snow = np.array([0, 2, 2, 0.001, 2, 4001, 0, 2, 2, 2, 0, 2], dtype=float)
    snow_nb = np.zeros((12, 1))
    snow_nb[[1, 5, 7, 8], 0] = [2, 4001, 0.01, 2]
    result = hydrol_bucket_snow_step(
        precip_rain=pr,
        precip_snow=ps,
        temp_sol_new=temp,
        soilcap=np.full(12, 1e6),
        frac_nobio=frac,
        totfrac_nobio=total_frac,
        vevapsno=evap,
        snow=snow,
        snow_age=np.arange(1, 13, dtype=float),
        snow_nobio=snow_nb,
        snow_nobio_age=np.full((12, 1), 3.0),
    )
    return {
        name: np.asarray(getattr(result, name)).ravel()
        for name in (
            "snow",
            "snow_age",
            "snow_nobio",
            "snow_nobio_age",
            "vevapsno",
            "tot_melt",
            "snowmelt",
            "snowdepth",
            "subsnowveg",
            "subsnownobio",
            "subsinksoil",
            "icemelt",
        )
    }


def _alma_cases() -> dict[int, dict[str, np.ndarray]]:
    # Fortran RESHAPE fills the first index fastest.
    q = np.arange(1, 169, dtype=float).reshape((12, 14), order="F") / 100.0
    hum = np.arange(11, 23, dtype=float)
    snow = np.arange(1, 13, dtype=float)
    snow_nb = np.full((12, 1), 0.5)
    init = hydrol_alma_step(
        qsintveg=q,
        humtot=hum,
        snow=snow,
        snow_nobio=snow_nb,
        tot_watveg_beg=np.zeros(12),
        tot_watsoil_beg=np.zeros(12),
        snow_beg=np.zeros(12),
        mx_eau_var=np.where(np.arange(12) % 2 == 0, 100.0, 0.0),
        lstep_init=True,
    )
    later = hydrol_alma_step(
        qsintveg=q + 0.01,
        humtot=hum + 1.0,
        snow=snow + 0.2,
        snow_nobio=snow_nb + 0.1,
        tot_watveg_beg=np.asarray(init.tot_watveg_beg),
        tot_watsoil_beg=np.asarray(init.tot_watsoil_beg),
        snow_beg=np.asarray(init.snow_beg),
        mx_eau_var=np.where(np.arange(12) % 2 == 0, 100.0, 0.0),
        lstep_init=False,
    )
    return {
        2: {
            name: np.asarray(getattr(init, name)).ravel()
            for name in ("tot_watveg_beg", "tot_watsoil_beg", "snow_beg")
        },
        3: {
            name: np.asarray(getattr(later, name)).ravel()
            for name in (
                "tot_watveg_beg",
                "tot_watsoil_beg",
                "snow_beg",
                "delintercept",
                "delsoilmoist",
                "delswe",
                "soilwet",
            )
        },
    }


def _waterbal_case() -> dict[str, np.ndarray]:
    q = np.zeros((12, 14))
    q[:, 13] = 0.2
    z = np.zeros(12)
    result = hydrol_waterbal_step(
        tot_water_beg=np.full(12, 10.5),
        vegtot=np.full(12, 0.8),
        totfrac_nobio=np.full(12, 0.2),
        qsintveg=q,
        humtot=np.full(12, 10.0),
        snow=np.full(12, 0.3),
        snow_nobio=np.zeros((12, 1)),
        precip_rain=z,
        precip_snow=z,
        returnflow=z,
        reinfiltration=z,
        irrigation=z,
        vevapwet=np.zeros_like(q),
        transpir=np.zeros_like(q),
        vevapnu=z,
        vevapsno=z,
        vevapflo=z,
        floodout=z,
        runoff=z,
        drainage=z,
    )
    return {
        name: np.asarray(getattr(result, name)).ravel()
        for name in ("tot_water_end", "tot_water_beg", "tot_flux")
    }


def _read(path: Path) -> dict[int, dict[str, np.ndarray]]:
    rows: dict[int, dict[str, list[float]]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(int(row["case_id"]), {}).setdefault(
                row["field"], []
            ).append(float(row["value"]))
    return {
        case: {field: np.asarray(values) for field, values in fields.items()}
        for case, fields in rows.items()
    }


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_hydrol_residual_") as td:
        source = Path(td) / "oracle.f90"
        executable = Path(td) / "oracle.exe"
        hashes = compose(source)
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable)],
            cwd=td,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        shutil.copyfile(
            Path(td) / "fortran_outputs.csv", output_dir / "fortran_outputs.csv"
        )
    expected = {1: _snow_case(), **_alma_cases(), 4: _waterbal_case()}
    actual = _read(output_dir / "fortran_outputs.csv")
    comparisons = []
    flat_f = {}
    flat_j = {}
    for case, fields in actual.items():
        for field, values in fields.items():
            name = f"case{case}.{field}"
            flat_f[name] = values
            flat_j[name] = expected[case][field]
            comparisons.append(
                float_comparison(name, values, flat_j[name], rtol=1e-12, atol=1e-14)
            )
    write_point_comparisons(
        output_dir / "point_comparisons.csv", flat_f, flat_j, rtol=1e-12, atol=1e-14
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "snow_branch_matrix",
                    "alma_init",
                    "alma_later",
                    "waterbal_balanced",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        family="hydrol_residual_owners",
        comparisons=comparisons,
        metadata={
            **metadata,
            "source_spans": hashes,
            "tolerance_policy": {"rtol": 1e-12, "atol": 1e-14},
        },
    )
