from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.forcing_metadata import forcing_vertical_ioipsl  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)

FAMILY = "readdim2_vertical"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_lane_readdim2_vertical.f90.template"
PROCEDURES = {"forcing_vertical_ioipsl"}


def compose(path: Path) -> dict[str, str]:
    span = extract_procedure_bytes(SOURCE, "forcing_vertical_ioipsl")
    path.write_bytes(
        TEMPLATE.read_bytes().replace(b"! <FORCING_VERTICAL_IOIPSL>", span.span_bytes)
    )
    return {"forcing_vertical_ioipsl": span.span_sha256}


def _read(path: Path) -> dict[str, np.ndarray]:
    fields: dict[str, list[float]] = {}
    with path.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            fields.setdefault(row["field"], []).append(float(row["value"]))
    return {name: np.asarray(values) for name, values in fields.items()}


def _dataset(**values: float) -> xr.Dataset:
    return xr.Dataset({name: xr.DataArray(value) for name, value in values.items()})


def _jax_case(
    prefix: str,
    source: xr.Dataset | None,
    *,
    force_id: int = 1,
    height_lev1: float | None = None,
    height_levw: float | None = None,
) -> dict[str, np.ndarray]:
    result = forcing_vertical_ioipsl(
        source,
        force_id=force_id,
        height_lev1=height_lev1,
        height_levw=height_levw,
    )
    sentinel = -777.0
    fields = {
        "zfixed": result.zfixed,
        "zsigma": result.zsigma,
        "zhybrid": result.zhybrid,
        "zlevels": result.zlevels,
        "zheight": result.zheight,
        "zsamelev_uv": result.zsamelev_uv,
        "zlev_fixed": sentinel if result.zlev_fixed is None else result.zlev_fixed,
        "zlevuv_fixed": sentinel if result.zlevuv_fixed is None else result.zlevuv_fixed,
        "zhybrid_a": sentinel if result.zhybrid_a is None else result.zhybrid_a,
        "zhybrid_b": sentinel if result.zhybrid_b is None else result.zhybrid_b,
        "zhybriduv_a": sentinel if result.zhybriduv_a is None else result.zhybriduv_a,
        "zhybriduv_b": sentinel if result.zhybriduv_b is None else result.zhybriduv_b,
    }
    return {
        f"{prefix}_{name}": np.asarray([float(value)], dtype=np.float64)
        for name, value in fields.items()
    }


def _jax_outputs() -> dict[str, np.ndarray]:
    return {
        **_jax_case("sigma_same", _dataset(Sigma=0.91)),
        **_jax_case("sigma_uv", _dataset(Sigma=0.91, Sigma_uv=0.82)),
        **_jax_case("hybrid_same", _dataset(HybSigA=12.0, HybSigB=0.5)),
        **_jax_case(
            "hybrid_uv",
            _dataset(HybSigA=12.0, HybSigB=0.5, HybSigA_uv=22.0, HybSigB_uv=0.4),
        ),
        **_jax_case("levels_same", _dataset(Levels=30.0)),
        **_jax_case("levels_uv", _dataset(Levels=30.0, Levels_uv=40.0)),
        **_jax_case("height_same", _dataset(Height_Lev1=3.0)),
        **_jax_case(
            "height_uv", _dataset(Height_Lev1=3.0, Height_Levuv=12.0)
        ),
        **_jax_case("legacy", _dataset(lev=4.0)),
        **_jax_case("fallback_positive", xr.Dataset()),
        **_jax_case(
            "fallback_same",
            None,
            force_id=-1,
            height_lev1=5.0,
            height_levw=5.0,
        ),
        **_jax_case(
            "fallback_uv",
            None,
            force_id=0,
            height_lev1=6.0,
            height_levw=11.0,
        ),
    }


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="orchidee_readdim2_vertical_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        executable = build / "oracle.exe"
        hashes = compose(source)
        metadata = compile_fortran(source, executable, compiler)
        subprocess.run(
            [str(executable), str((output_dir / "fortran_outputs.csv").resolve()), "valid"],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
    fortran = _read(output_dir / "fortran_outputs.csv")
    jax = _jax_outputs()
    write_point_comparisons(
        output_dir / "point_comparisons.csv", fortran, jax, rtol=1e-12, atol=1e-14
    )
    comparisons = [
        float_comparison(name, values, jax[name], rtol=1e-12, atol=1e-14)
        for name, values in fortran.items()
    ]
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    "sigma same/separate UV",
                    "hybrid same/separate UV",
                    "levels same/separate UV",
                    "fixed height same/separate UV",
                    "legacy lev",
                    "missing metadata fallback for positive, zero, and negative force_id",
                ],
                "coverage_only_fatal_cases": ["missing HybSigB", "missing HybSigB_uv"],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    result = write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_span_sha256": hashes,
            "template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
            "compiler": metadata,
        },
    )
    if result["status"] != "passed":
        raise RuntimeError("readdim2 vertical Oracle mismatch")
    return result


if __name__ == "__main__":
    run_oracle(ROOT / "outputs/reference_mode/micro_oracles" / FAMILY)
