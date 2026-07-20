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


TEMPLATE = (
    ROOT / "scripts" / "dev" / "stomate_daily_maintenance_oracle_harness.f90.template"
)
EXTRACTOR = ROOT / "scripts" / "dev" / "extract_fortran_micro_oracle.py"
FAMILY = "stomate_daily_maintenance"


def _extracts():
    spec = importlib.util.spec_from_file_location("oracle_extractor", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    document = module.load_manifest()
    return {
        entry["procedure"]: span
        for entry, span in module.validate_and_extract(document, root=ROOT)
        if entry.get("family_id") == FAMILY
    }


def _compose(path: Path) -> dict[str, str]:
    spans = _extracts()
    daily = b"\n\n".join(
        spans[name].span_bytes
        for name in ("stomate_accu_r1d", "stomate_accu_r2d", "stomate_accu_r3d")
    )
    source = (
        TEMPLATE.read_bytes()
        .replace(b"! <DAILY_PROCEDURES>", daily)
        .replace(b"! <MAINTENANCE_PROCEDURE>", spans["maint_respiration"].span_bytes)
    )
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in spans.items()}


def _read(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}


def _jax_values() -> dict[str, np.ndarray]:
    from jax_orchidee.stomate.carbon_kernels import maintenance_respiration
    from jax_orchidee.stomate.daily import stomate_accumulate_daily

    in1 = np.array([1.25, -0.5])
    out1 = np.array([0.1, -0.2])
    r1 = np.asarray(
        stomate_accumulate_daily(
            out1, in1, False, dt_sechiba=1800.0, dt_stomate=86400.0
        )
    )
    in2 = np.arange(1.0, 7.0).reshape((2, 3), order="F")
    in2 = (in2 - 3.5) / 4.0
    out2 = np.full((2, 3), 0.25)
    r2 = np.asarray(
        stomate_accumulate_daily(out2, in2, True, dt_sechiba=1800.0, dt_stomate=86400.0)
    ).ravel(order="F")
    in3 = np.arange(1.0, 9.0).reshape((2, 2, 2), order="F")
    in3 = (in3 - 4.5) / 3.0
    out3 = np.zeros_like(in3)
    for step in range(48):
        out3 = np.asarray(
            stomate_accumulate_daily(
                out3, in3, step == 47, dt_sechiba=1800.0, dt_stomate=86400.0
            )
        )
    npts, nvm, nparts = 2, 14, 12
    biomass = np.zeros((npts, nvm, nparts, 1))
    for i in range(npts):
        for k in range(nparts):
            biomass[i, 13, k, 0] = (i + 1) * (k + 1) * 2.0
    biomass[1, 13, 0, 0] = 0.0
    coeff = np.zeros((nvm, nparts))
    for j in range(1, nvm):
        coeff[j, :] = 0.0005 * np.arange(1, nparts + 1)
    slope = np.zeros((nvm, 3))
    slope[:, 0] = 0.01
    rprof = np.full((npts, nvm), 0.5)
    rprof[:, 13] = [0.35, 0.8]
    result = maintenance_respiration(
        biomass,
        np.array([280.0, 268.0]),
        np.array([285.0, 260.0]),
        np.array([[279.0, 277.0, 275.0], [269.0, 271.0, 273.0]]),
        np.array([0.0, 0.1, 0.4, 1.0]),
        rprof,
        np.full((npts, nvm), 0.02),
        coeff,
        slope,
        np.full(nvm, 0.5),
        np.arange(nvm) == 13,
        dt_sechiba_days=1800.0 / 86400.0,
        min_stomate=1e-8,
        maint_resp_min_vmax=0.3,
        maint_resp_coeff=1.4,
    )
    return {
        "accu_r1": r1,
        "accu_r2": r2,
        "accu_r3_48": out3.ravel(order="F"),
        "maint_lai": np.asarray(result.lai).ravel(order="F"),
        "maint_resp": np.asarray(result.resp_maint_part).transpose(1, 0, 2).ravel(),
    }


def run_oracle(
    output_dir: Path, compiler: Path = DEFAULT_COMPILER
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "fortran_outputs.csv"
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_oracle_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        hashes = _compose(source)
        executable = build / "oracle.exe"
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str(output_csv.resolve())],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(output_csv)
    jax = _jax_values()
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "npts": 2,
                "nvm": 14,
                "nparts": 12,
                "daily_cases": [
                    "r1d non-slow",
                    "r2d slow mean",
                    "r3d 48-step boundary",
                ],
                "maintenance_cases": [
                    "bare soil zero",
                    "PFT14 tree",
                    "zero and nonzero leaf biomass",
                    "distinct root profiles and soil temperatures",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    write_point_comparisons(
        output_dir / "point_comparisons.csv",
        fortran,
        jax,
        rtol=1e-12,
        atol=1e-14,
    )
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=1e-12, atol=1e-14)
        for name in fortran
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "ledger_entries": [
                "stomate.active.daily_accumulation",
                "stomate.active.maintenance_respiration",
            ],
            "span_sha256": hashes,
            "build": metadata,
        },
    )
