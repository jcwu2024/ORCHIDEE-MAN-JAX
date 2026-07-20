from __future__ import annotations

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

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_cold_start_first_step_coverage,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.sechiba.thermosoil import SN_CAPA  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_point_comparisons,
    write_result,
)


FAMILY = "thermosoil_first_transition_actual"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_thermosoil_first_transition_actual.f90.template"
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
RUN_DEF = (
    ROOT
    / "outputs/paper_250919_materialized_run_defs/arg2_1.0/319.0-057.0"
    / "I5/S26_69.834_0.0019_0.3980_97.918/used_run.def"
)
COLD_EVIDENCE = ROOT / "outputs/reference_mode/micro_oracles/thermosoil_cold_coef_actual"
LANDPOINT_ID = "319.0-057.0"
EXPECTED_DIMS = {"npts": 1, "ngrnd": 32, "nslm": 11, "nvm": 14, "nsnow": 3, "ndeep": 32, "nnobio": 1}
RTOL = 1.0e-12
ATOL = 1.0e-12


def _span(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _fortran_array(value: object, dtype: np.dtype | type = np.float64) -> np.ndarray:
    return np.asfortranarray(np.asarray(value, dtype=dtype))


def _write_stream(path: Path, ordered: tuple[tuple[str, np.ndarray], ...]) -> None:
    with path.open("wb") as handle:
        for _, values in ordered:
            values.ravel(order="F").tofile(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_transition() -> tuple[
    tuple[tuple[str, np.ndarray], ...],
    dict[str, np.ndarray],
    dict[str, object],
]:
    cold_comparison_path = COLD_EVIDENCE / "comparison.json"
    cold_inputs_path = COLD_EVIDENCE / "production_inputs.npz"
    if not cold_comparison_path.is_file() or not cold_inputs_path.is_file():
        raise FileNotFoundError("passed thermosoil_cold_coef_actual evidence is required")
    cold_comparison = json.loads(cold_comparison_path.read_text(encoding="ascii"))
    if cold_comparison.get("status") != "passed":
        raise RuntimeError("thermosoil_cold_coef_actual evidence is not passed")

    context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=RUN_DEF)
    coverage = paper_1961_driver_cold_start_first_step_coverage(CONFIG, prepared_context=context)
    closure = coverage.initialized_payloads["thermosoil_first_step_module"]
    cold = coverage.initialized_payloads["thermosoil_cold_start_coef"]
    if not closure.ok or not cold.ok:
        raise RuntimeError(
            f"production closure is incomplete: transition={closure.missing_inputs}, cold={cold.missing_inputs}"
        )
    module = closure.module
    boundary = closure.boundary_payload

    ptn = np.asarray(boundary["ptn"], dtype=np.float64)
    dims = {
        "npts": ptn.shape[0],
        "ngrnd": ptn.shape[1],
        "nvm": ptn.shape[2],
        "nslm": np.asarray(module.profile.stempdiag).shape[1],
        "nsnow": np.asarray(boundary["snowtemp"]).shape[1],
        "ndeep": np.asarray(module.final.deeptemp_prof).shape[1],
        "nnobio": np.asarray(boundary["frac_snow_nobio"]).shape[1],
    }
    if dims != EXPECTED_DIMS:
        raise ValueError(f"unexpected production dimensions: {dims}")

    with np.load(cold_inputs_path) as cold_inputs:
        reused_checks = {
            "ptn": np.array_equal(cold_inputs["ptn"], ptn),
            "snowtemp": np.array_equal(cold_inputs["snowtemp"], np.asarray(boundary["snowtemp"])),
            "veget_max": np.array_equal(cold_inputs["veget_max"], np.asarray(boundary["veget_max"])),
        }
        staged_temperature_checks = {
            "temp_sol_new_changed_after_enerbil": not np.array_equal(
                cold_inputs["temp_sol_new"], np.asarray(boundary["temp_sol_new"])
            ),
            "temp_sol_new_pft_changed_after_enerbil": not np.array_equal(
                cold_inputs["temp_sol_new_pft"], np.asarray(boundary["temp_sol_new_pft"])
            ),
        }
    if not all(reused_checks.values()):
        raise ValueError(f"production first transition drifted from cold evidence inputs: {reused_checks}")

    cold_reuse_checks = {
        "cgrnd": np.array_equal(np.asarray(boundary["cgrnd"]), np.asarray(cold.coef.soil.cgrnd)),
        "dgrnd": np.array_equal(np.asarray(boundary["dgrnd"]), np.asarray(cold.coef.soil.dgrnd)),
        "cgrnd_snow": np.array_equal(np.asarray(boundary["cgrnd_snow"]), np.asarray(cold.coef.cgrnd_snow)),
        "dgrnd_snow": np.array_equal(np.asarray(boundary["dgrnd_snow"]), np.asarray(cold.coef.dgrnd_snow)),
        "soilcap": np.array_equal(np.asarray(boundary["soilcap_initial"]), np.asarray(cold.coef.soilcap)),
        "pcapa_en": np.array_equal(np.asarray(boundary["pcapa_en_previous"]), np.asarray(cold.getdiff.pcapa_en)),
    }
    if not all(cold_reuse_checks.values()):
        raise ValueError(f"first transition does not reuse the passed cold coefficients: {cold_reuse_checks}")

    ordered = (
        ("ptn_initial", _fortran_array(boundary["ptn"])),
        ("cgrnd_previous", _fortran_array(boundary["cgrnd"])),
        ("dgrnd_previous", _fortran_array(boundary["dgrnd"])),
        ("cgrnd_snow_previous", _fortran_array(boundary["cgrnd_snow"])),
        ("dgrnd_snow_previous", _fortran_array(boundary["dgrnd_snow"])),
        ("temp_sol_new", _fortran_array(boundary["temp_sol_new"])),
        ("temp_sol_new_pft", _fortran_array(boundary["temp_sol_new_pft"])),
        ("snowtemp", _fortran_array(boundary["snowtemp"])),
        ("veget_max", _fortran_array(boundary["veget_max"])),
        ("frac_snow_veg", _fortran_array(boundary["frac_snow_veg"])),
        ("frac_snow_nobio", _fortran_array(boundary["frac_snow_nobio"])),
        ("totfrac_nobio", _fortran_array(boundary["totfrac_nobio"])),
        ("shum_ngrnd_perma", _fortran_array(module.humlev.shum_ngrnd_perma)),
        ("shum_ngrnd_permalong_previous", _fortran_array(boundary["shum_ngrnd_permalong_previous"])),
        ("pcapa_en_previous", _fortran_array(boundary["pcapa_en_previous"])),
        ("soilcap_initial", _fortran_array(boundary["soilcap_initial"])),
        ("temp_sol_beg_previous", _fortran_array(boundary["temp_sol_beg"])),
        # Final-state reduction consumes same-step pkappa; no getdiff formula is repeated here.
        ("pkappa_same_step", _fortran_array(module.getdiff.pkappa)),
        ("veget_mask_2d", _fortran_array(boundary["veget_mask_2d"], np.int32)),
        (
            "scalars",
            _fortran_array(
                [float(boundary["dt_sechiba"]), float(boundary["lambda_thermal"]), float(SN_CAPA)]
            ),
        ),
    )

    jax = {
        "wlupdate.shum_ngrnd_permalong": np.asarray(module.final.deephum_prof).ravel(order="F"),
        "profile.ptn": np.asarray(module.profile.ptn).ravel(order="F"),
        "profile.stempdiag": np.asarray(module.profile.stempdiag).ravel(order="F"),
        "energy.surfheat_incr": np.asarray(module.energy.surfheat_incr).ravel(order="F"),
        "energy.coldcont_incr": np.asarray(module.energy.coldcont_incr).ravel(order="F"),
        "energy.ptn_beg": np.asarray(module.energy.ptn_beg).ravel(order="F"),
        "energy.temp_sol_beg": np.asarray(module.energy.temp_sol_beg).ravel(order="F"),
        "final.ptn_pftmean": np.asarray(module.final.ptn_pftmean).ravel(order="F"),
        "final.pkappa_pftmean": np.asarray(module.final.pkappa_pftmean).ravel(order="F"),
        "final.gtemp": np.asarray(module.final.gtemp).ravel(order="F"),
        "final.ptnlev1": np.asarray(module.final.ptnlev1).ravel(order="F"),
        "writeback.deephum_prof": np.asarray(module.final.deephum_prof).ravel(order="F"),
        "writeback.deeptemp_prof": np.asarray(module.final.deeptemp_prof).ravel(order="F"),
    }
    metadata = {
        "landpoint_id": LANDPOINT_ID,
        "dimensions": dims,
        "run_def": str(RUN_DEF),
        "coverage_mode": coverage.mode,
        "coverage_ready": coverage.ready_for_cold_start_first_step,
        "input_fields": [name for name, _ in ordered],
        "cold_evidence": str(COLD_EVIDENCE.relative_to(ROOT)),
        "cold_evidence_sha256": {
            "comparison.json": _sha256(cold_comparison_path),
            "production_inputs.npz": _sha256(cold_inputs_path),
        },
        "cold_input_reuse_checks": reused_checks,
        "source_order_temperature_checks": staged_temperature_checks,
        "cold_coefficient_reuse_checks": cold_reuse_checks,
    }
    return ordered, jax, metadata


OUTPUT_LAYOUT = (
    ("wlupdate.shum_ngrnd_permalong", (1, 32, 14)),
    ("profile.ptn", (1, 32, 14)),
    ("profile.stempdiag", (1, 11)),
    ("energy.surfheat_incr", (1,)),
    ("energy.coldcont_incr", (1,)),
    ("energy.ptn_beg", (1, 32, 14)),
    ("energy.temp_sol_beg", (1,)),
    ("final.ptn_pftmean", (1, 32)),
    ("final.pkappa_pftmean", (1, 32)),
    ("final.gtemp", (1,)),
    ("final.ptnlev1", (1,)),
    ("writeback.deephum_prof", (1, 32, 14)),
    ("writeback.deeptemp_prof", (1, 32, 14)),
)


def _read_outputs(path: Path) -> dict[str, np.ndarray]:
    raw = np.fromfile(path, dtype=np.float64)
    offset = 0
    values: dict[str, np.ndarray] = {}
    for name, shape in OUTPUT_LAYOUT:
        count = int(np.prod(shape))
        values[name] = raw[offset : offset + count]
        offset += count
    if offset != raw.size:
        raise ValueError(f"Fortran output size mismatch: consumed {offset}, found {raw.size}")
    return values


def run_oracle(output_dir: Path, compiler: Path = DEFAULT_COMPILER) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_inputs, jax, input_metadata = _production_transition()
    np.savez(output_dir / "production_inputs.npz", **{name: value for name, value in ordered_inputs})
    (output_dir / "inputs.json").write_text(
        json.dumps({"schema_version": 1, **input_metadata}, indent=2) + "\n", encoding="ascii"
    )

    spans = {
        "thermosoil_wlupdate": _span(3648, 3672),
        "thermosoil_profile": _span(1761, 1851),
        "thermosoil_diaglev": _span(3484, 3513),
        "thermosoil_energy": _span(2420, 2462),
        "thermosoil_main_final_state": _span(1011, 1032),
    }
    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_first_transition_actual_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        generated = TEMPLATE.read_bytes()
        for marker, span in spans.items():
            generated = generated.replace(f"! <{marker.upper()}>".encode(), span)
        source.write_bytes(generated)
        executable = build / "oracle.exe"
        compiler_meta = compile_fortran(source, executable, compiler)
        input_path = build / "production_inputs.bin"
        output_path = build / "fortran_outputs.bin"
        _write_stream(input_path, ordered_inputs)
        subprocess.run(
            [str(executable), str(input_path), str(output_path)],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        fortran = _read_outputs(output_path)

    if set(fortran) != set(jax):
        raise ValueError(f"comparison field mismatch: fortran={sorted(fortran)}, jax={sorted(jax)}")
    write_point_comparisons(output_dir / "point_comparisons.csv", fortran, jax, rtol=RTOL, atol=ATOL)
    comparisons = [
        float_comparison(name, fortran[name], jax[name], rtol=RTOL, atol=ATOL)
        for name, _ in OUTPUT_LAYOUT
    ]
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "source_file_sha256": _sha256(SOURCE),
            "procedure_span_sha256": {name: hashlib.sha256(span).hexdigest() for name, span in spans.items()},
            "compiler": compiler_meta,
            "scope": "point 319 production cold-start first THERMOSOIL transition before the next-step recurrence",
            "source_order": ["thermosoil_wlupdate", "thermosoil_profile", "thermosoil_diaglev", "thermosoil_energy", "thermosoil_main_final_state"],
            "reused_results": {
                "previous_coefficients": "passed thermosoil_cold_coef_actual production closure",
                "same_step_pkappa_for_final_reduction": "JAX production thermosoil_first_step_module.getdiff.pkappa",
            },
            "verified_ledger_entries": [],
        },
    )


if __name__ == "__main__":
    target = ROOT / "outputs/reference_mode/micro_oracles/thermosoil_first_transition_actual"
    try:
        result = run_oracle(target)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        raise
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
