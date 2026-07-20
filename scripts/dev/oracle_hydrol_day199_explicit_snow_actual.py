from __future__ import annotations

import argparse
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
    driver_previous_step_state_from_next_step_hydrol_scaffold,
    paper_1961_next_step_hydrol_precall_scaffold,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.sechiba.hydrol import hydrol_explicit_snow_step  # noqa: E402
from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_result,
)


FAMILY = "hydrol_day199_explicit_snow_actual"
SNOW_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/explicitsnow.f90"
QSAT_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/qsat_moisture.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_hydrol_day199_explicit_snow_actual.f90.template"
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
SNOW_PROCEDURES = (
    "explicitsnow_main",
    "explicitsnow_grain",
    "explicitsnow_compactn",
    "explicitsnow_transf",
    "explicitsnow_fall",
    "explicitsnow_gone",
    "explicitsnow_melt_refrz",
    "explicitsnow_levels",
    "explicitsnow_profile",
)
THERMO_FUNCTIONS = (
    "snow3lgrain_0d",
    "snow3lgrain_1d",
    "snow3lscap_1d",
    "snow3lhold_1d",
    "snow3lheat_1d",
    "snow3ltemp_1d",
    "snow3lliq_1d",
)
INPUT_FIELDS = (
    "precip_rain",
    "precip_snow",
    "temp_air",
    "pb",
    "u",
    "v",
    "temp_sol_new",
    "soilcap",
    "pgflux",
    "frac_nobio",
    "totfrac_nobio",
    "gtemp",
    "lambda_snow",
    "cgrnd_snow",
    "dgrnd_snow",
    "vevapsno",
    "snow_age",
    "snow_nobio_age",
    "snow_nobio",
    "snowrho",
    "snowgrain",
    "snowdz",
    "snowtemp",
    "snowheat",
    "snow",
    "temp_sol_add",
    "snowliq",
    "subsnownobio",
    "grndflux",
    "snowmelt",
    "soilflxresid",
)
OUTPUT_FIELDS = (
    "snow",
    "snowdz",
    "snowrho",
    "snowtemp",
    "snowheat",
    "snowliq",
    "snowgrain",
    "snow_age",
    "snow_nobio",
    "snow_nobio_age",
    "vevapsno",
    "subsnownobio",
    "subsinksoil",
    "grndflux",
    "snowmelt",
    "tot_melt",
    "temp_sol_add",
)
OUTPUT_SHAPES = {
    "snow": (NSTEPS,),
    "snowdz": (NSTEPS, 3),
    "snowrho": (NSTEPS, 3),
    "snowtemp": (NSTEPS, 3),
    "snowheat": (NSTEPS, 3),
    "snowliq": (NSTEPS, 3),
    "snowgrain": (NSTEPS, 3),
    "snow_age": (NSTEPS,),
    "snow_nobio": (NSTEPS, 1),
    "snow_nobio_age": (NSTEPS, 1),
    "vevapsno": (NSTEPS,),
    "subsnownobio": (NSTEPS, 1),
    "subsinksoil": (NSTEPS,),
    "grndflux": (NSTEPS,),
    "snowmelt": (NSTEPS,),
    "tot_melt": (NSTEPS,),
    "temp_sol_add": (NSTEPS,),
}


def _load_checkpoint(path: Path) -> DriverPreviousStepStatePacket:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, DriverPreviousStepStatePacket) or state.tstep != START_TSTEP - 1:
        raise ValueError("checkpoint must be a current Day198 DriverPreviousStepStatePacket")
    return state


def _snow_inputs(scaffold) -> dict[str, np.ndarray]:
    module = scaffold.hydrol_module
    values = dict(scaffold.hydrol_precall.payload)
    values.update(module.module_inputs)
    if values.get("totfrac_nobio") is None:
        values["totfrac_nobio"] = 1.0 - np.asarray(values["vegtot"], dtype=np.float64)
    forcing = scaffold.enerbil.payload
    direct = {
        "temp_air": forcing.temp_air,
        "pb": forcing.pb,
        "u": forcing.u,
        "v": forcing.v,
    }
    result = {}
    for name in INPUT_FIELDS:
        value = direct.get(name, values.get(name))
        if value is None:
            raise ValueError(f"production HYDROL boundary lacks {name}")
        result[name] = np.asarray(value, dtype=np.float64)
    return result


def _owner_outputs(result) -> dict[str, np.ndarray]:
    snow = result.snow_state
    return {
        "snow": np.asarray(snow.snow),
        "snowdz": np.asarray(snow.snowdz),
        "snowrho": np.asarray(snow.snowrho),
        "snowtemp": np.asarray(snow.snowtemp),
        "snowheat": np.asarray(snow.snowheat),
        "snowliq": np.asarray(result.snowliq),
        "snowgrain": np.asarray(snow.snowgrain),
        "snow_age": np.asarray(snow.snow_age),
        "snow_nobio": np.asarray(snow.snow_nobio),
        "snow_nobio_age": np.asarray(snow.snow_nobio_age),
        "vevapsno": np.asarray(result.vevapsno),
        "subsnownobio": np.asarray(result.subsnownobio),
        "subsinksoil": np.asarray(result.subsinksoil),
        "grndflux": np.asarray(result.grndflux),
        "snowmelt": np.asarray(result.snowmelt),
        "tot_melt": np.asarray(snow.tot_melt),
        "temp_sol_add": np.asarray(result.temp_sol_add),
    }


def _production_outputs(module, owner) -> dict[str, np.ndarray]:
    outputs = _owner_outputs(module.snow_step or owner)
    snow = module.snow_state
    outputs.update(
        {
            "snow": np.asarray(snow.snow),
            "snowdz": np.asarray(snow.snowdz),
            "snowrho": np.asarray(snow.snowrho),
            "snowtemp": np.asarray(snow.snowtemp),
            "snowheat": np.asarray(snow.snowheat),
            "snowgrain": np.asarray(snow.snowgrain),
            "snow_age": np.asarray(snow.snow_age),
            "snow_nobio": np.asarray(snow.snow_nobio),
            "snow_nobio_age": np.asarray(snow.snow_nobio_age),
            "tot_melt": np.asarray(snow.tot_melt),
        }
    )
    return outputs


def _window(checkpoint: Path):
    context = prepare_paper_1961_driver_context(CONFIG, used_run_def_path=RUN_DEF)
    state = _load_checkpoint(checkpoint)
    input_rows = []
    output_rows = []
    production_rows = []
    full_owner_steps = []
    for tstep in range(START_TSTEP, START_TSTEP + NSTEPS):
        scaffold = paper_1961_next_step_hydrol_precall_scaffold(
            CONFIG,
            previous_state=state,
            year=1961,
            tstep=tstep,
            used_run_def_path=RUN_DEF,
            prepared_context=context,
            module_jit=True,
            diffuco_local_jit=True,
        )
        if not scaffold.ready_for_next_state:
            raise RuntimeError(f"production scaffold stopped at {tstep}")
        inputs = _snow_inputs(scaffold)
        owner = hydrol_explicit_snow_step(
            **inputs, dt_sechiba=context.dt_sechiba, one_day=86400.0
        )
        input_rows.append(inputs)
        output_rows.append(_owner_outputs(owner))
        production_rows.append(_production_outputs(scaffold.hydrol_module, owner))
        full_owner_steps.append(scaffold.hydrol_module.snow_step is not None)
        state = driver_previous_step_state_from_next_step_hydrol_scaffold(
            scaffold, ok_laidev=context.ok_laidev
        )
    inputs = {name: np.concatenate([row[name] for row in input_rows], axis=0) for name in INPUT_FIELDS}
    outputs = {name: np.concatenate([row[name] for row in output_rows], axis=0) for name in OUTPUT_FIELDS}
    production = {
        name: np.concatenate([row[name] for row in production_rows], axis=0)
        for name in OUTPUT_FIELDS
    }
    return inputs, outputs, production, full_owner_steps


def _compose(path: Path) -> dict[str, str]:
    snow = {name: extract_procedure_bytes(SNOW_SOURCE, name) for name in SNOW_PROCEDURES}
    thermo = {name: extract_procedure_bytes(QSAT_SOURCE, name) for name in THERMO_FUNCTIONS}
    source = TEMPLATE.read_bytes()
    source = source.replace(
        b"! <THERMO_FUNCTIONS>", b"\n\n".join(thermo[name].span_bytes for name in THERMO_FUNCTIONS)
    ).replace(
        b"! <SNOW_PROCEDURES>", b"\n\n".join(snow[name].span_bytes for name in SNOW_PROCEDURES)
    )
    path.write_bytes(source)
    return {name: span.span_sha256 for name, span in {**snow, **thermo}.items()}


def _write_input(path: Path, inputs: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        for name in INPUT_FIELDS:
            np.asfortranarray(inputs[name]).ravel(order="F").tofile(handle)


def _read_output(path: Path) -> dict[str, np.ndarray]:
    raw = np.fromfile(path, dtype=np.float64)
    offset = 0
    outputs = {}
    for name in OUTPUT_FIELDS:
        shape = OUTPUT_SHAPES[name]
        count = int(np.prod(shape))
        outputs[name] = raw[offset : offset + count].reshape(shape, order="F")
        offset += count
    if offset != raw.size:
        raise ValueError(f"Fortran output has {raw.size} values; expected {offset}")
    return outputs


def run_oracle(output_dir: Path, checkpoint: Path, compiler: Path = DEFAULT_COMPILER):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint.resolve()
    inputs, jax_outputs, production_outputs, full_owner_steps = _window(checkpoint)
    np.savez(output_dir / "production_inputs.npz", **inputs)
    with tempfile.TemporaryDirectory(prefix="orchidee_hydrol_day199_snow_actual_") as td:
        build = Path(td)
        source = build / "oracle.f90"
        hashes = _compose(source)
        executable = build / "oracle.exe"
        compiler_meta = compile_fortran(source, executable, compiler)
        input_path = build / "inputs.bin"
        output_path = build / "outputs.bin"
        _write_input(input_path, inputs)
        subprocess.run(
            [str(executable), str(input_path), str(output_path)],
            cwd=build,
            env=compiler_environment(compiler),
            check=True,
            capture_output=True,
            text=True,
        )
        fortran_outputs = _read_output(output_path)
    comparisons = [
        float_comparison(name, fortran_outputs[name], jax_outputs[name], rtol=1e-12, atol=1e-12)
        for name in OUTPUT_FIELDS
    ]
    production_comparisons = [
        float_comparison(
            f"production.{name}", production_outputs[name], jax_outputs[name], rtol=0.0, atol=0.0
        )
        for name in OUTPUT_FIELDS
    ]
    comparisons.extend(production_comparisons)
    first_full = next((i for i, active in enumerate(full_owner_steps) if active), None)
    metadata = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "window": {"year": 1961, "day": 199, "start_tstep": START_TSTEP, "steps": NSTEPS},
        "first_production_full_owner_tstep": None if first_full is None else START_TSTEP + first_full,
        "production_full_owner_steps": sum(full_owner_steps),
        "source_file_sha256": hashlib.sha256(SNOW_SOURCE.read_bytes()).hexdigest(),
        "span_sha256": hashes,
        "compiler": compiler_meta,
        "verified_ledger_entries": [],
    }
    return write_result(output_dir, FAMILY, comparisons, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/reference_mode/micro_oracles" / FAMILY,
    )
    args = parser.parse_args()
    result = run_oracle(args.output_dir, args.checkpoint)
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
