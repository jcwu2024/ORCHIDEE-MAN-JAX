from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
)
from oracle_lane_stomate_carbon import (  # noqa: E402
    VMAX_TEMPLATE,
    _procedure_bytes,
)

from research.daily_coarse_graining.physical_parameter_gradients import (  # noqa: E402
    classify_physical_gradient_estimates,
)

jax.config.update("jax_enable_x64", True)

DEFAULT_OUTPUT = (
    ROOT
    / "outputs/research/daily_coarse_graining/gate_d1_physical_parameter_gradients"
    / "selected_fortran_finite_differences.json"
)
MAINTENANCE_TEMPLATE = (
    ROOT / "scripts/dev/stomate_daily_maintenance_oracle_harness.f90.template"
)


@dataclass(frozen=True)
class FortranCase:
    pair_id: str
    parameter_value: float
    step: float
    executable: Path
    compiler: Path
    target: str


def _run_fortran(case: FortranCase, value: float, output: Path) -> float:
    subprocess.run(
        [str(case.executable), str(output), f"{value:.17g}"],
        cwd=case.executable.parent,
        env=compiler_environment(case.compiler),
        check=True,
        capture_output=True,
        text=True,
    )
    with output.open(newline="", encoding="ascii") as handle:
        rows = tuple(csv.DictReader(handle))
    if case.target == "vmax_pft14_point1":
        values = [float(row["value"]) for row in rows if row["field"] == "vcmax"]
        return values[13 * 3]
    if case.target == "maintenance_pft14_point1_sum":
        return sum(
            float(row["value"])
            for row in rows
            if row["field"] == "maint_resp"
            and int(row["i"]) == 1
            and int(row["j"]) == 14
        )
    raise KeyError(case.target)


def _compose_maintenance(path: Path) -> dict[str, str]:
    sources = {
        "stomate_accu_r1d": (
            ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90",
            9341,
            9363,
        ),
        "stomate_accu_r2d": (
            ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90",
            9365,
            9387,
        ),
        "stomate_accu_r3d": (
            ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90",
            9389,
            9411,
        ),
        "maint_respiration": (
            ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_resp.f90",
            122,
            389,
        ),
    }
    spans = {
        name: _procedure_bytes(source, start, end)
        for name, (source, start, end) in sources.items()
    }
    daily = b"\n\n".join(
        spans[name]
        for name in ("stomate_accu_r1d", "stomate_accu_r2d", "stomate_accu_r3d")
    )
    path.write_bytes(
        MAINTENANCE_TEMPLATE.read_bytes()
        .replace(b"! <DAILY_PROCEDURES>", daily)
        .replace(b"! <MAINTENANCE_PROCEDURE>", spans["maint_respiration"])
    )
    return {name: hashlib.sha256(span).hexdigest() for name, span in spans.items()}


def _vmax_objective(value):
    from jax_orchidee.stomate.carbon_kernels import vmax_step

    age = jnp.zeros((3, 14, 4), dtype=jnp.float64)
    frac = jnp.zeros_like(age)
    age = age.at[0, 13, :].set(jnp.asarray([5.0, 40.0, 120.0, 220.0]))
    frac = frac.at[0, 13, :].set(jnp.asarray([0.6, 0.25, 0.1, 0.05]))
    age = age.at[1, 13, :].set(jnp.asarray([0.0, 20.0, 80.0, 180.0]))
    frac = frac.at[1, 13, :].set(jnp.asarray([0.0, 0.5, 0.3, 0.2]))
    nlim = jnp.ones((3, 14), dtype=jnp.float64).at[:, 13].set(
        jnp.asarray([0.8, 1.1, 1.0])
    )
    vcmax25 = jnp.full(14, 50.0, dtype=jnp.float64).at[13].set(value)
    common = {
        "vcmax25": vcmax25,
        "n_limfert": nlim,
        "leaf_timecst": jnp.full(14, 100.0),
        "leafagecrit": jnp.full(14, 200.0),
        "pheno_type": jnp.zeros(14, dtype=jnp.int32),
        "leaf_tab": jnp.zeros(14, dtype=jnp.int32),
        "ok_laidev": jnp.zeros(14, dtype=bool),
        "dt_days": 1.0,
    }
    first = vmax_step(leaf_age=age, leaf_frac=frac, **common)
    second = vmax_step(leaf_age=first.leaf_age, leaf_frac=first.leaf_frac, **common)
    return second.vcmax[0, 13]


def _maintenance_objective(value):
    from jax_orchidee.stomate.carbon_kernels import maintenance_respiration

    npts, nvm, nparts = 2, 14, 12
    biomass = jnp.zeros((npts, nvm, nparts, 1), dtype=jnp.float64)
    for i in range(npts):
        biomass = biomass.at[i, 13, :, 0].set(
            (i + 1) * 2.0 * jnp.arange(1, nparts + 1, dtype=jnp.float64)
        )
    biomass = biomass.at[1, 13, 0, 0].set(0.0)
    coeff = jnp.zeros((nvm, nparts), dtype=jnp.float64)
    coeff = coeff.at[1:, :].set(
        0.0005 * jnp.arange(1, nparts + 1, dtype=jnp.float64)[None, :]
    )
    slope = jnp.zeros((nvm, 3), dtype=jnp.float64).at[:, 0].set(0.01)
    slope = slope.at[13, 0].set(value)
    rprof = jnp.full((npts, nvm), 0.5, dtype=jnp.float64)
    rprof = rprof.at[:, 13].set(jnp.asarray([0.35, 0.8]))
    result = maintenance_respiration(
        biomass,
        jnp.asarray([280.0, 268.0]),
        jnp.asarray([285.0, 260.0]),
        jnp.asarray([[279.0, 277.0, 275.0], [269.0, 271.0, 273.0]]),
        jnp.asarray([0.0, 0.1, 0.4, 1.0]),
        rprof,
        jnp.full((npts, nvm), 0.02),
        coeff,
        slope,
        jnp.full(nvm, 0.5),
        jnp.arange(nvm) == 13,
        dt_sechiba_days=1800.0 / 86400.0,
        min_stomate=1e-8,
        maint_resp_min_vmax=0.3,
        maint_resp_coeff=1.4,
    )
    return jnp.sum(result.resp_maint_part[0, 13, :])


def _estimates(case: FortranCase, objective, work: Path) -> dict[str, object]:
    value = jnp.asarray(case.parameter_value, dtype=jnp.float64)
    primal = float(objective(value))
    forward = float(jax.jacfwd(objective)(value))
    reverse = float(jax.jacrev(objective)(value))
    evaluations = {}
    for label, candidate in (
        ("baseline", case.parameter_value),
        ("plus_h", case.parameter_value + case.step),
        ("minus_h", case.parameter_value - case.step),
        ("plus_h2", case.parameter_value + case.step / 2.0),
        ("minus_h2", case.parameter_value - case.step / 2.0),
    ):
        evaluations[label] = _run_fortran(case, candidate, work / f"{case.pair_id}.{label}.csv")
    fd = (evaluations["plus_h"] - evaluations["minus_h"]) / (2.0 * case.step)
    fd_half = (evaluations["plus_h2"] - evaluations["minus_h2"]) / case.step
    comparison = classify_physical_gradient_estimates(
        pair_id=case.pair_id,
        pair_class="smooth_active",
        parameter_value=case.parameter_value,
        finite_difference_step=case.step,
        primal_value=primal,
        forward_ad=forward,
        reverse_ad=reverse,
        central_difference=fd,
        central_difference_half_step=fd_half,
    )
    primal_scale = max(abs(primal), abs(evaluations["baseline"]), 1.0e-12)
    primal_relative_error = abs(primal - evaluations["baseline"]) / primal_scale
    return {
        **comparison.as_dict(),
        "fortran_primal": evaluations["baseline"],
        "fortran_evaluations": evaluations,
        "fortran_jax_primal_relative_error": primal_relative_error,
        "fortran_jax_primal_passed": primal_relative_error <= 1.0e-12,
        "passed": comparison.passed and primal_relative_error <= 1.0e-12,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify selected Gate-D1 JAX AD against source-extracted Fortran finite differences"
    )
    parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    vmax_source = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_vmax.f90"
    with tempfile.TemporaryDirectory(prefix="orchidee_gate_d1_fortran_fd_") as temporary:
        work = Path(temporary)
        vmax_span = _procedure_bytes(vmax_source, 105, 363)
        vmax_unit = work / "vmax_oracle.f90"
        vmax_unit.write_bytes(
            VMAX_TEMPLATE.read_bytes().replace(b"! <VMAX_PROCEDURE>", vmax_span)
        )
        vmax_executable = work / "vmax_oracle.exe"
        vmax_build = compile_fortran(vmax_unit, vmax_executable, args.compiler)

        maintenance_unit = work / "maintenance_oracle.f90"
        maintenance_hashes = _compose_maintenance(maintenance_unit)
        maintenance_executable = work / "maintenance_oracle.exe"
        maintenance_build = compile_fortran(
            maintenance_unit, maintenance_executable, args.compiler
        )

        cases = (
            (
                FortranCase(
                    "fortran.vmax__vcmax25",
                    50.0,
                    5.0e-4,
                    vmax_executable,
                    args.compiler,
                    "vmax_pft14_point1",
                ),
                _vmax_objective,
            ),
            (
                FortranCase(
                    "fortran.maintenance__maint_resp_slope_c",
                    0.01,
                    1.0e-6,
                    maintenance_executable,
                    args.compiler,
                    "maintenance_pft14_point1_sum",
                ),
                _maintenance_objective,
            ),
        )
        results = [_estimates(case, objective, work) for case, objective in cases]

    report = {
        "schema_version": "gate_d1_selected_fortran_finite_differences_v1",
        "status": "passed" if all(item["passed"] for item in results) else "failed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy": {
            "active_relative_error_max": 0.01,
            "fortran_jax_primal_relative_error_max": 1.0e-12,
            "finite_difference": "centered h/h2 with Richardson extrapolation",
        },
        "cases": results,
        "provenance": {
            "vmax_source": str(vmax_source.relative_to(ROOT)).replace("\\", "/"),
            "vmax_source_sha256": hashlib.sha256(vmax_source.read_bytes()).hexdigest(),
            "vmax_span_sha256": hashlib.sha256(vmax_span).hexdigest(),
            "maintenance_procedure_hashes": maintenance_hashes,
            "vmax_build": vmax_build,
            "maintenance_build": maintenance_build,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
