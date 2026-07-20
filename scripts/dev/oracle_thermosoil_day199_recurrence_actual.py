from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    DriverPreviousStepStatePacket,
    _paper_1961_later_day_half_hour_transition,
    _paper_1961_later_day_transition_inputs,
    driver_previous_step_state_from_next_step_hydrol_scaffold,
    paper_1961_next_step_hydrol_precall_scaffold,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.driver.init import parse_run_def_bool  # noqa: E402
from jax_orchidee.sechiba.thermosoil import (  # noqa: E402
    PHIGEOTH,
    PSNOWDZMIN,
    QZ_USDA,
    SMCMAX_USDA,
    SO_CAPA_DRY_NS_USDA,
    ThermosoilRecurrenceState,
    thermosoil_getdiff_thinsnow,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_result,
)


FAMILY = "thermosoil_day199_recurrence_actual"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_thermosoil_day199_recurrence_actual.f90.template"
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
RUN_DEF = (
    ROOT
    / "outputs/paper_250919_materialized_run_defs/arg2_1.0/319.0-057.0"
    / "I5/S26_69.834_0.0019_0.3980_97.918/used_run.def"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/reference_mode/diagnose_319_day198_current_20260714/319.0-057.0/checkpoints"
    / "paper_driver_1961_year_end_state.pkl"
)
START_TSTEP = 9504
NSTEPS = 48
END_TSTEP = START_TSTEP + NSTEPS - 1
RTOL = 1.0e-12
ATOL = 1.0e-12

FORTRAN_SPANS = {
    "thermosoil_cond_pft": (2007, 2127),
    "thermosoil_cond_nopft": (2129, 2231),
    "thermosoil_cond": (1883, 1972),
    "thermosoil_getdiff": (2566, 2863),
    "thermosoil_coef": (1386, 1729),
    "thermosoil_wlupdate": (3648, 3672),
    "thermosoil_profile": (1761, 1851),
    "thermosoil_diaglev": (3484, 3513),
    "thermosoil_energy": (2420, 2462),
    "thermosoil_getdiff_thinsnow": (3692, 3788),
    "thermosoil_main_precoef_pftmean": (914, 919),
    "thermosoil_main_final_state": (1011, 1032),
}


def _span(start: int, end: int) -> bytes:
    return b"".join(SOURCE.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_fortran_source(path: Path) -> dict[str, str]:
    spans = {name: _span(*line_span) for name, line_span in FORTRAN_SPANS.items()}
    generated = TEMPLATE.read_bytes()
    for marker, span in spans.items():
        generated = generated.replace(f"! <{marker.upper()}>".encode(), span)
    path.write_bytes(generated)
    return {name: hashlib.sha256(span).hexdigest() for name, span in spans.items()}


def _array(value: object, dtype: np.dtype | type = np.float64) -> np.ndarray:
    return np.asfortranarray(np.asarray(value, dtype=dtype))


def _write_values(handle, values: np.ndarray) -> None:
    values.ravel(order="F").tofile(handle)


def _checkpoint_payload(
    checkpoint: Path,
) -> tuple[dict[str, object], DriverPreviousStepStatePacket]:
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), DriverPreviousStepStatePacket):
        raise ValueError("checkpoint is not a DriverPreviousStepStatePacket cache")
    state = payload["state"]
    if state.tstep != START_TSTEP - 1:
        raise ValueError(f"checkpoint tstep is {state.tstep}, expected {START_TSTEP - 1}")
    return payload, state


def _validate_checkpoint(
    checkpoint: Path, state: DriverPreviousStepStatePacket
) -> dict[str, object]:
    therm = state.fields_by_component["thermosoil_previous_step_state"]
    hydrol = state.fields_by_component["hydrol_previous_step_state"]
    slow = state.fields_by_component["slowproc_stomate_previous_step_state"]
    required = {"pcapa_en", "temp_sol_beg", "shum_ngrnd_permalong"}
    missing_thermosoil = sorted(required - set(therm))
    required_hydrol = {"snow", "snowdz", "snowrho", "snowtemp", "snowliq", "snowheat"}
    missing_hydrol = sorted(required_hydrol - set(hydrol))
    if missing_thermosoil or missing_hydrol or "date" not in slow:
        raise ValueError(
            "checkpoint is not current strict production state: "
            f"thermosoil={missing_thermosoil}, hydrol={missing_hydrol}, "
            f"slowproc_date={'date' in slow}"
        )
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_tstep": state.tstep,
        "thermosoil_fields": sorted(therm),
        "hydrol_explicit_snow_fields": sorted(required_hydrol),
        "strict_carry_complete": True,
        "bridge": None,
    }


def _step_outputs(step, next_state, thin) -> dict[str, np.ndarray]:
    return {
        "wlupdate.shum_ngrnd_permalong": np.asarray(step.final.deephum_prof),
        "profile.ptn": np.asarray(step.profile.ptn),
        "profile.stempdiag": np.asarray(step.profile.stempdiag),
        "energy.surfheat_incr": np.asarray(step.energy.surfheat_incr),
        "energy.coldcont_incr": np.asarray(step.energy.coldcont_incr),
        "energy.ptn_beg": np.asarray(step.energy.ptn_beg),
        "energy.temp_sol_beg": np.asarray(step.energy.temp_sol_beg),
        "final.ptn_pftmean": np.asarray(step.final.ptn_pftmean),
        "final.pkappa_pftmean": np.asarray(step.final.pkappa_pftmean),
        "final.gtemp": np.asarray(step.final.gtemp),
        "final.ptnlev1": np.asarray(step.final.ptnlev1),
        "writeback.deephum_prof": np.asarray(step.final.deephum_prof),
        "writeback.deeptemp_prof": np.asarray(step.final.deeptemp_prof),
        "carry.cgrnd": np.asarray(next_state.cgrnd),
        "carry.dgrnd": np.asarray(next_state.dgrnd),
        "carry.cgrnd_snow": np.asarray(next_state.cgrnd_snow),
        "carry.dgrnd_snow": np.asarray(next_state.dgrnd_snow),
        "carry.lambda_snow": np.asarray(next_state.lambda_snow),
        "carry.pcapa_en": np.asarray(next_state.pcapa_en),
        "carry.soilcap": np.asarray(next_state.soilcap),
        "thin_shadow.pcapa_layer1": np.asarray(thin.pcapa[:, 0, :]),
        "thin_shadow.pcapa_en_layer1": np.asarray(thin.pcapa_en[:, 0, :]),
        "thin_shadow.pkappa_layer1": np.asarray(thin.pkappa[:, 0, :]),
        "thin_shadow.profil_froz_layer1": np.asarray(thin.profil_froz[:, 0, :]),
    }


def _production_window(context, initial_state):
    production_state = initial_state
    slow = initial_state.fields_by_component["slowproc_stomate_previous_step_state"]
    hydrol_initial = initial_state.fields_by_component["hydrol_previous_step_state"]
    external_steps: list[dict[str, np.ndarray]] = []
    expected_steps: list[dict[str, np.ndarray]] = []
    thresholds: list[dict[str, object]] = []
    production_replay_exact = True

    for tstep in range(START_TSTEP, END_TSTEP + 1):
        scaffold = paper_1961_next_step_hydrol_precall_scaffold(
            CONFIG,
            previous_state=production_state,
            year=1961,
            tstep=tstep,
            used_run_def_path=RUN_DEF,
            prepared_context=context,
            module_jit=True,
            diffuco_local_jit=True,
        )
        if not scaffold.ready_for_next_state:
            raise RuntimeError(
                f"production external-input generation stopped at tstep {tstep}: {scaffold.missing_previous_state_fields}"
            )
        closure = scaffold.thermosoil_module
        boundary = closure.boundary_payload
        canonical = closure.module
        next_state = ThermosoilRecurrenceState(
            ptn=canonical.profile.ptn,
            cgrnd=canonical.coef.soil.cgrnd,
            dgrnd=canonical.coef.soil.dgrnd,
            cgrnd_snow=canonical.coef.cgrnd_snow,
            dgrnd_snow=canonical.coef.dgrnd_snow,
            lambda_snow=canonical.coef.lambda_snow,
            pcapa_en=canonical.getdiff.pcapa_en,
            temp_sol_beg=canonical.energy.temp_sol_beg,
            soilcap=canonical.coef.soilcap,
        )

        snowdz = np.asarray(boundary["snowdz"], dtype=np.float64)
        snow_h = float(np.sum(snowdz))
        thin_active = 0.0 < snow_h <= 0.01
        positive_below_floor = bool(np.any((snowdz > 0.0) & (snowdz < PSNOWDZMIN)))
        profil_froz = np.asarray(canonical.getdiff.profil_froz, dtype=np.float64)
        profil_froz_pft_spread = float(
            np.max(np.max(profil_froz, axis=2) - np.min(profil_froz, axis=2))
        )
        thin = thermosoil_getdiff_thinsnow(
            ptn=canonical.profile.ptn,
            shum_ngrnd_permalong=canonical.final.deephum_prof,
            snowdz=snowdz,
            pcapa=canonical.getdiff.pcapa,
            pcapa_en=canonical.getdiff.pcapa_en,
            pkappa=canonical.getdiff.pkappa,
            profil_froz=canonical.getdiff.profil_froz,
            veget_mask_2d=boundary["veget_mask_2d"],
            zlt=context.cwrr_grid.zlt,
        )
        external_steps.append(
            {
                "mc_layt": np.asarray(canonical.humlev.mc_layt),
                "mcl_layt": np.asarray(canonical.humlev.mcl_layt),
                "tmc_layt": np.asarray(canonical.humlev.tmc_layt),
                "mc_layt_pft": np.asarray(canonical.humlev.mc_layt_pft),
                "mcl_layt_pft": np.asarray(canonical.humlev.mcl_layt_pft),
                "tmc_layt_pft": np.asarray(canonical.humlev.tmc_layt_pft),
                "shum_ngrnd_perma": np.asarray(canonical.humlev.shum_ngrnd_perma),
                "temp_sol_new": np.asarray(boundary["temp_sol_new"]),
                "temp_sol_new_pft": np.asarray(boundary["temp_sol_new_pft"]),
                "snow": np.asarray(scaffold.hydrol_module.snow_state.snow),
                "organic_layer_thick": np.asarray(slow["depth_organic_soil"]),
                "soilc_total": np.asarray(slow["soilc_total"]),
                "veget_max": np.asarray(boundary["veget_max"]),
                "snowdz": snowdz,
                "snowrho": np.asarray(boundary["snowrho"]),
                "snowtemp": np.asarray(boundary["snowtemp"]),
                "pb": np.asarray(boundary["pb"]),
                "frac_snow_veg": np.asarray(boundary["frac_snow_veg"]),
                "frac_snow_nobio": np.asarray(boundary["frac_snow_nobio"]),
                "totfrac_nobio": np.asarray(boundary["totfrac_nobio"]),
                "njsc": np.asarray(hydrol_initial["njsc"], dtype=np.int32),
            }
        )
        expected_steps.append(_step_outputs(canonical, next_state, thin))
        thresholds.append(
            {
                "tstep": tstep,
                "day199_step": tstep - START_TSTEP + 1,
                "snowdz": snowdz.ravel().tolist(),
                "snow_depth_sum": snow_h,
                "snow_present": snow_h > 0.0,
                "thin_snow_predicate_0_lt_sum_le_0p01": thin_active,
                "positive_layer_below_psnowdzmin": positive_below_floor,
                "profil_froz_pft_spread_max": profil_froz_pft_spread,
                "fortran_profil_froz_guard_would_abort": profil_froz_pft_spread > 0.0,
                "psnowdzmin": float(PSNOWDZMIN),
                "production_coef_thin_snow_call_active": False,
                "shadow_thin_snow_executed": True,
            }
        )
        production_state = driver_previous_step_state_from_next_step_hydrol_scaffold(
            scaffold, ok_laidev=context.ok_laidev
        )
    return external_steps, expected_steps, thresholds, production_replay_exact, production_state


def _compiled_physical_window(context, checkpoint_state):
    day_inputs = _paper_1961_later_day_transition_inputs(
        context=context,
        current_state=checkpoint_state,
        year=1961,
        start=START_TSTEP,
        steps_per_stomate=NSTEPS,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    strict_transition = _paper_1961_later_day_half_hour_transition(
        CONFIG,
        context=context,
        initial_state=checkpoint_state,
        day_inputs=day_inputs,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        module_jit=True,
        diffuco_local_jit=True,
        use_fast_state_loop=False,
        use_compiled_sechiba_day=False,
    )
    compiled_transition = _paper_1961_later_day_half_hour_transition(
        CONFIG,
        context=context,
        initial_state=checkpoint_state,
        day_inputs=day_inputs,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        module_jit=True,
        diffuco_local_jit=True,
        use_fast_state_loop=False,
        use_compiled_sechiba_day=True,
    )
    return compiled_transition, strict_transition, {
        "ok": compiled_transition.ok,
        "completed_entry_payloads": len(compiled_transition.completed_entry_payloads),
        "stopped_at_tstep": compiled_transition.stopped_at_tstep,
        "state_gaps": [
            {"component": gap.component, "fields": list(gap.fields)}
            for gap in compiled_transition.state_gaps
        ],
        "current_state_tstep": getattr(compiled_transition.current_state, "tstep", None),
        "strict_compact_ok": strict_transition.ok,
        "strict_compact_completed_entry_payloads": len(strict_transition.completed_entry_payloads),
    }


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
    ("carry.cgrnd", (1, 31, 14)),
    ("carry.dgrnd", (1, 31, 14)),
    ("carry.cgrnd_snow", (1, 3)),
    ("carry.dgrnd_snow", (1, 3)),
    ("carry.lambda_snow", (1,)),
    ("carry.pcapa_en", (1, 32, 14)),
    ("carry.soilcap", (1,)),
    ("thin_shadow.pcapa_layer1", (1, 14)),
    ("thin_shadow.pcapa_en_layer1", (1, 14)),
    ("thin_shadow.pkappa_layer1", (1, 14)),
    ("thin_shadow.profil_froz_layer1", (1, 14)),
)
PRECOEF_LAYOUT = OUTPUT_LAYOUT[:7]


def _static_and_initial_inputs(context, state):
    therm = state.fields_by_component["thermosoil_previous_step_state"]
    npts, _, nvm = np.asarray(therm["ptn"]).shape
    run_def = context.run_def_values
    controls = np.asarray(
        [
            int(parse_run_def_bool(run_def.get("USE_TOPORGANICLAYER_TEMPDIFF", "FALSE"))),
            int(parse_run_def_bool(run_def.get("USE_SOILC_TEMPDIFF", "FALSE"))),
            int(parse_run_def_bool(run_def.get("use_refSOC", "FALSE"))),
            int(parse_run_def_bool(run_def.get("OK_FREEZE_THERMIX", "TRUE"))),
            int(parse_run_def_bool(run_def.get("OK_EXPLICITSNOW", "TRUE"))),
            int(float(run_def.get("BEDROCK_FLAG", "0"))),
        ],
        dtype=np.int32,
    )
    ok_lai = np.asarray(context.ok_laidev[:nvm], dtype=np.int32)
    mask = np.ones((npts, nvm), dtype=np.int32)
    static = (
        ("QZ", _array(QZ_USDA)),
        ("SMCMAX", _array(SMCMAX_USDA)),
        ("mcs", _array(SMCMAX_USDA)),
        ("so_capa_dry_ns", _array(SO_CAPA_DRY_NS_USDA)),
        ("zlt", _array(context.cwrr_grid.zlt)),
        ("dlt", _array(context.cwrr_grid.dlt)),
        ("dz1", _array(context.cwrr_grid.dz1)),
        ("refSOC", _array(therm["refSOC"])),
        ("ok_laidev", _array(ok_lai, np.int32)),
        ("veget_mask_2d", _array(mask, np.int32)),
        ("controls", _array(controls, np.int32)),
        (
            "scalars",
            _array(
                [
                    float(context.dt_sechiba),
                    float(context.cwrr_grid.znt[0] * context.cwrr_grid.dz1[0]),
                    float(PHIGEOTH),
                    float(PSNOWDZMIN),
                ]
            ),
        ),
    )
    initial = (
        ("initial.ptn", _array(therm["ptn"])),
        ("initial.cgrnd", _array(therm["cgrnd"])),
        ("initial.dgrnd", _array(therm["dgrnd"])),
        ("initial.cgrnd_snow", _array(therm["cgrnd_snow"])),
        ("initial.dgrnd_snow", _array(therm["dgrnd_snow"])),
        ("initial.lambda_snow", _array(therm["lambda_snow"])),
        ("initial.pcapa_en", _array(therm["pcapa_en"])),
        ("initial.temp_sol_beg", _array(therm["temp_sol_beg"])),
        ("initial.soilcap", _array(therm["soilcap"])),
        ("initial.shum_ngrnd_permalong", _array(therm["shum_ngrnd_permalong"])),
    )
    return static, initial


def _write_input(path, static, initial, external_steps):
    field_order = tuple(external_steps[0])
    with path.open("wb") as handle:
        for _, value in (*static, *initial):
            _write_values(handle, value)
        for step in external_steps:
            for name in field_order:
                dtype = np.int32 if name == "njsc" else np.float64
                _write_values(handle, _array(step[name], dtype))
    return field_order


def _read_outputs(path, nsteps=NSTEPS, layout=OUTPUT_LAYOUT):
    raw = np.fromfile(path, dtype=np.float64)
    offset = 0
    steps: list[dict[str, np.ndarray]] = []
    for _ in range(nsteps):
        values = {}
        for name, shape in layout:
            count = int(np.prod(shape))
            values[name] = raw[offset : offset + count]
            offset += count
        steps.append(values)
    if offset != raw.size:
        raise ValueError(f"Fortran output size mismatch: consumed {offset}, found {raw.size}")
    return steps


def _comparisons(fortran_steps, jax_steps, output_dir, layout=OUTPUT_LAYOUT):
    aggregate = []
    step_rows = []
    first_mismatch = None
    if fortran_steps:
        for name, _ in layout:
            fortran = np.concatenate([step[name].ravel(order="F") for step in fortran_steps])
            jax = np.concatenate([step[name].ravel(order="F") for step in jax_steps])
            aggregate.append(float_comparison(name, fortran, jax, rtol=RTOL, atol=ATOL))
    for index, (fortran_step, jax_step) in enumerate(zip(fortran_steps, jax_steps, strict=True)):
        tstep = START_TSTEP + index
        for name, _ in layout:
            comparison = float_comparison(
                name,
                fortran_step[name],
                jax_step[name].ravel(order="F"),
                rtol=RTOL,
                atol=ATOL,
            )
            step_rows.append({"tstep": tstep, "day199_step": index + 1, **comparison})
            if not comparison["passed"] and first_mismatch is None:
                first_mismatch = {"tstep": tstep, "day199_step": index + 1, "field": name, **comparison}
    with (output_dir / "step_comparisons.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tstep",
                "day199_step",
                "name",
                "count",
                "rtol",
                "atol",
                "max_abs_error",
                "max_rel_error",
                "max_ulp_error",
                "passed",
            ],
        )
        writer.writeheader()
        writer.writerows(step_rows)
    return aggregate, first_mismatch


def _first_step(rows, key):
    match = next((row for row in rows if row[key]), None)
    return None if match is None else {"tstep": match["tstep"], "day199_step": match["day199_step"]}


def _compiled_state_comparisons(compiled_state, retained_state, *, prefix):
    comparisons = []
    components = (
        "diffuco_previous_step_state",
        "enerbil_previous_step_state",
        "hydrol_previous_step_state",
        "thermosoil_previous_step_state",
        "sechiba_finalize_state",
        "driver_previous_step_state",
    )
    for component in components:
        compiled_fields = compiled_state.fields_by_component.get(component, {})
        retained_fields = retained_state.fields_by_component.get(component, {})
        for name in sorted(set(compiled_fields) | set(retained_fields)):
            label = f"{prefix}.{component}.{name}"
            if name not in compiled_fields or name not in retained_fields:
                comparisons.append(
                    {
                        "name": label,
                        "count": 0,
                        "rtol": 0.0,
                        "atol": 0.0,
                        "passed": False,
                        "reason": "field missing from one transition path",
                    }
                )
                continue
            compiled = np.asarray(compiled_fields[name])
            retained = np.asarray(retained_fields[name])
            if compiled.dtype.kind not in "biufc" or retained.dtype.kind not in "biufc":
                continue
            if compiled.shape != retained.shape:
                comparisons.append(
                    {
                        "name": label,
                        "count": int(compiled.size),
                        "rtol": 0.0,
                        "atol": 0.0,
                        "passed": False,
                        "reason": f"shape mismatch {compiled.shape} != {retained.shape}",
                    }
                )
            elif compiled.dtype.kind in "biu" and retained.dtype.kind in "biu":
                comparisons.append(
                    {
                        "name": label,
                        "count": int(compiled.size),
                        "rtol": 0.0,
                        "atol": 0.0,
                        "max_abs_error": int(
                            np.max(np.abs(compiled.astype(np.int64) - retained.astype(np.int64)), initial=0)
                        ),
                        "max_rel_error": 0.0,
                        "max_ulp_error": 0,
                        "passed": bool(np.array_equal(compiled, retained)),
                    }
                )
            else:
                compiled_nan = np.isnan(compiled)
                retained_nan = np.isnan(retained)
                if not np.array_equal(compiled_nan, retained_nan):
                    comparisons.append(
                        {
                            "name": label,
                            "count": int(compiled.size),
                            "rtol": RTOL,
                            "atol": ATOL,
                            "passed": False,
                            "reason": "NaN mask mismatch",
                        }
                    )
                else:
                    comparisons.append(
                        float_comparison(
                            label,
                            np.where(compiled_nan, 0.0, compiled),
                            np.where(retained_nan, 0.0, retained),
                            rtol=RTOL,
                            atol=ATOL,
                        )
                    )
    return comparisons


def run_oracle(
    output_dir: Path,
    compiler: Path = DEFAULT_COMPILER,
    checkpoint: Path = DEFAULT_CHECKPOINT,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint.resolve()
    checkpoint_payload, checkpoint_state = _checkpoint_payload(checkpoint)
    context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=RUN_DEF)
    compiled_transition, strict_compact_transition, compiled_physical_window = _compiled_physical_window(
        context, checkpoint_state
    )
    if not compiled_transition.ok or len(compiled_transition.completed_entry_payloads) != NSTEPS:
        raise RuntimeError(f"compiled physical window did not close: {compiled_physical_window}")
    compatibility = _validate_checkpoint(checkpoint, checkpoint_state)
    external_steps, jax_steps, thresholds, replay_exact, retained_final_state = _production_window(
        context, checkpoint_state
    )
    compiled_therm = compiled_transition.current_state.fields_by_component[
        "thermosoil_previous_step_state"
    ]
    retained_therm = retained_final_state.fields_by_component[
        "thermosoil_previous_step_state"
    ]
    compiled_physical_window["retained_boundary_chain_thermosoil_final_bit_exact"] = all(
        name in retained_therm
        and np.array_equal(np.asarray(value), np.asarray(retained_therm[name]))
        for name, value in compiled_therm.items()
    )
    compiled_vs_compact = _compiled_state_comparisons(
        compiled_transition.current_state,
        strict_compact_transition.current_state,
        prefix="compiled_vs_compact",
    )
    compact_vs_scaffold = _compiled_state_comparisons(
        strict_compact_transition.current_state,
        retained_final_state,
        prefix="compact_vs_scaffold",
    )
    compiled_state_comparisons = (*compiled_vs_compact, *compact_vs_scaffold)
    compiled_physical_window["compiled_vs_compact_comparisons"] = len(
        compiled_vs_compact
    )
    compiled_physical_window["compiled_vs_compact_all_passed"] = all(
        comparison["passed"] for comparison in compiled_vs_compact
    )
    compiled_physical_window["compact_vs_scaffold_comparisons"] = len(
        compact_vs_scaffold
    )
    compiled_physical_window["compact_vs_scaffold_all_passed"] = all(
        comparison["passed"] for comparison in compact_vs_scaffold
    )
    compiled_gpp = np.stack(
        [np.asarray(payload["gpp"]) for payload in compiled_transition.completed_entry_payloads]
    )
    compact_gpp = np.stack(
        [np.asarray(payload["gpp"]) for payload in strict_compact_transition.completed_entry_payloads]
    )
    compiled_state_comparisons = (
        *compiled_state_comparisons,
        float_comparison(
            "compiled_vs_compact.entry_gpp", compiled_gpp, compact_gpp, rtol=RTOL, atol=ATOL
        ),
    )
    static, initial = _static_and_initial_inputs(context, checkpoint_state)

    npz_values = {name: value for name, value in (*static, *initial)}
    for name in external_steps[0]:
        npz_values[f"external.{name}"] = np.stack([step[name] for step in external_steps])
    np.savez(output_dir / "production_inputs.npz", **npz_values)
    (output_dir / "checkpoint_compatibility.json").write_text(
        json.dumps({"schema_version": 1, **compatibility}, indent=2) + "\n", encoding="ascii"
    )
    threshold_summary = {
        "schema_version": 1,
        "window": {"start_tstep": START_TSTEP, "end_tstep": END_TSTEP, "steps": NSTEPS},
        "call_site_semantics": {
            "thermosoil_var_init_lines_1319_1320": "thin-snow call active only during initialization",
            "thermosoil_coef_lines_1503_1504": "thin-snow call commented; production recurrence carry is not overwritten",
            "shadow_call": "original thermosoil_getdiff_thinsnow executes for diagnostics and is restored before carry",
        },
        "first_snow_present": _first_step(thresholds, "snow_present"),
        "first_thin_snow_predicate": _first_step(thresholds, "thin_snow_predicate_0_lt_sum_le_0p01"),
        "first_positive_layer_below_psnowdzmin": _first_step(thresholds, "positive_layer_below_psnowdzmin"),
        "first_fortran_profil_froz_guard_abort": _first_step(
            thresholds, "fortran_profil_froz_guard_would_abort"
        ),
        "steps": thresholds,
    }
    (output_dir / "snow_thresholds.json").write_text(
        json.dumps(threshold_summary, indent=2) + "\n", encoding="ascii"
    )

    with tempfile.TemporaryDirectory(prefix="orchidee_thermosoil_day199_recurrence_actual_") as temporary:
        build = Path(temporary)
        source = build / "oracle.f90"
        span_hashes = compose_fortran_source(source)
        executable = build / "oracle.exe"
        compiler_meta = compile_fortran(source, executable, compiler)
        input_path = build / "production_inputs.bin"
        output_path = build / "fortran_outputs.bin"
        precoef_path = build / "fortran_precoef_outputs.bin"
        external_field_order = _write_input(input_path, static, initial, external_steps)
        completed = subprocess.run(
            [str(executable), str(input_path), str(output_path), str(precoef_path)],
            cwd=build,
            env=compiler_environment(compiler),
            check=False,
            capture_output=True,
            text=True,
        )
        values_per_step = sum(int(np.prod(shape)) for _, shape in OUTPUT_LAYOUT)
        output_values = output_path.stat().st_size // np.dtype(np.float64).itemsize
        if output_values % values_per_step:
            raise ValueError(
                f"partial Fortran output is not step-aligned: {output_values} values, {values_per_step} per step"
            )
        completed_fortran_steps = output_values // values_per_step
        precoef_values_per_step = sum(int(np.prod(shape)) for _, shape in PRECOEF_LAYOUT)
        precoef_values = precoef_path.stat().st_size // np.dtype(np.float64).itemsize
        if precoef_values % precoef_values_per_step:
            raise ValueError("partial pre-coef Fortran output is not step-aligned")
        completed_precoef_steps = precoef_values // precoef_values_per_step
        if completed.returncode == 0 and completed_fortran_steps != NSTEPS:
            raise ValueError(
                f"successful Fortran process wrote {completed_fortran_steps} of {NSTEPS} steps"
            )
        fortran_steps = _read_outputs(output_path, completed_fortran_steps)
        fortran_precoef_steps = _read_outputs(
            precoef_path, completed_precoef_steps, PRECOEF_LAYOUT
        )

    if completed.returncode == 0:
        stale_abort = output_dir / "fortran_abort.json"
        if stale_abort.exists():
            stale_abort.unlink()
        comparisons, first_mismatch = _comparisons(
            fortran_steps, jax_steps[:completed_fortran_steps], output_dir
        )
    else:
        comparisons, first_mismatch = _comparisons(
            fortran_precoef_steps,
            jax_steps[:completed_precoef_steps],
            output_dir,
            PRECOEF_LAYOUT,
        )
    comparisons.extend(compiled_state_comparisons)
    fortran_abort = None
    if completed.returncode != 0:
        abort_tstep = START_TSTEP + completed_fortran_steps
        fortran_abort = {
            "returncode": completed.returncode,
            "completed_steps": completed_fortran_steps,
            "completed_precoef_steps": completed_precoef_steps,
            "abort_tstep": abort_tstep,
            "day199_step": completed_fortran_steps + 1,
            "field": "getdiff.profil_froz_pft_equality",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        (output_dir / "fortran_abort.json").write_text(
            json.dumps({"schema_version": 1, **fortran_abort}, indent=2) + "\n",
            encoding="ascii",
        )
        abort_comparison = {
            "name": "source_abort.getdiff.profil_froz_pft_equality",
            "count": 1,
            "rtol": RTOL,
            "atol": ATOL,
            "passed": False,
            "reason": "original Fortran ipslerr_p(3) before coefficient/writeback completion",
            "tstep": abort_tstep,
            "day199_step": completed_fortran_steps + 1,
        }
        comparisons.append(abort_comparison)
        if first_mismatch is None:
            first_mismatch = dict(abort_comparison)
    metadata = {
        "compiler": compiler_meta,
        "source_file_sha256": _sha256(SOURCE),
        "procedure_span_sha256": span_hashes,
        "checkpoint_sha256": compatibility["checkpoint_sha256"],
        "checkpoint_metadata": {
            "days_per_year": checkpoint_payload.get("days_per_year"),
            "end_year": checkpoint_payload.get("end_year"),
            "tstep": checkpoint_state.tstep,
        },
        "window": {"year": 1961, "day": 199, "start_tstep": START_TSTEP, "end_tstep": END_TSTEP, "steps": NSTEPS},
        "compiled_physical_window": compiled_physical_window,
        "external_field_order": list(external_field_order),
        "production_jax_carry_chained_from_module_outputs": replay_exact,
        "first_mismatch": first_mismatch,
        "fortran_abort": fortran_abort,
        "snow_threshold_summary": {
            key: threshold_summary[key]
            for key in (
                "first_snow_present",
                "first_thin_snow_predicate",
                "first_positive_layer_below_psnowdzmin",
                "first_fortran_profil_froz_guard_abort",
            )
        },
        "thin_snow_scope": "shadow original-procedure diagnostic only; thermosoil_coef production call site is commented",
        "verified_ledger_entries": [],
    }
    return write_result(output_dir, FAMILY, comparisons, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/reference_mode/micro_oracles/thermosoil_day199_recurrence_actual",
    )
    args = parser.parse_args()
    try:
        result = run_oracle(args.output_dir, checkpoint=args.checkpoint)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        raise
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
