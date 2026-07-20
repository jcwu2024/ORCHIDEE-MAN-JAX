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
    paper_1961_driver_later_day_scaffold,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.driver.reference_layout import (  # noqa: E402
    resolve_paper_landpoint_reference,
)
from scripts.dev.extract_fortran_micro_oracle import (  # noqa: E402
    extract_procedure_bytes,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compile_fortran,
    compiler_environment,
    float_comparison,
    write_result,
)


FAMILY = "stomate_day199_maintenance_actual"
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
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_resp.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_stomate_day199_maintenance_actual.f90.template"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
START_TSTEP = 9504
NSTEPS = 48
NVM = 14
NPARTS = 12
NSLM = 11
RTOL = 1.0e-12
ATOL = 1.0e-12


def _load_checkpoint(path: Path) -> DriverPreviousStepStatePacket:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, DriverPreviousStepStatePacket) or state.tstep != START_TSTEP - 1:
        raise ValueError("checkpoint must be a current Day198 DriverPreviousStepStatePacket")
    return state


def _repeat_first_axis(value, count: int = NSTEPS) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[0] != 1:
        raise ValueError(f"expected one-point production input, got {array.shape}")
    return np.repeat(array, count, axis=0)


def _actual_case(checkpoint: Path):
    previous = _load_checkpoint(checkpoint)
    reference = resolve_paper_landpoint_reference(ROOT, "319.0-057.0")
    if reference.output_dir is None:
        raise FileNotFoundError("Point319 reference run directory is unavailable")
    context = prepare_paper_1961_driver_context(
        CONFIG,
        used_run_def_path=RUN_DEF,
        reference_run_dir=reference.output_dir,
    )
    day = paper_1961_driver_later_day_scaffold(
        CONFIG,
        previous_state=previous,
        year=1961,
        start_tstep=START_TSTEP,
        used_run_def_path=RUN_DEF,
        reference_run_dir=reference.output_dir,
        prepared_context=context,
        retain_stomate_step_results=True,
    )
    if not day.ready_for_day_end_state or day.daily_process_fold is None:
        raise RuntimeError(f"Day199 production scaffold is incomplete: {day.missing_components}")
    if len(day.completed_entry_payloads) != NSTEPS:
        raise ValueError("Day199 production scaffold did not retain 48 entry payloads")
    maintenance = day.daily_process_fold.maintenance
    if len(maintenance.step_results) != NSTEPS:
        raise ValueError("Day199 production fold did not retain 48 maintenance results")

    slow = previous.fields_by_component["slowproc_stomate_previous_step_state"]
    boundary = context.stomate_boundary
    static = context.stomate_static
    if boundary is None or static is None:
        raise ValueError("prepared production context lacks STOMATE static/boundary inputs")
    z_soil = np.asarray(boundary.kwargs["z_soil"], dtype=np.float64)
    if z_soil.shape != (NSLM + 1,) or z_soil[0] != 0.0:
        raise ValueError(f"unexpected production z_soil shape/content: {z_soil.shape}")

    inputs = {
        "dt_sechiba": np.asarray(context.runtime.dt_sechiba, dtype=np.float64),
        "maint_resp_min_vmax": np.asarray(0.3, dtype=np.float64),
        "maint_resp_coeff": np.asarray(1.4, dtype=np.float64),
        "diaglev": z_soil[1:],
        "is_tree": np.asarray(static.parameters.is_tree, dtype=np.int32),
        "ok_laidev": np.asarray(context.ok_laidev, dtype=np.int32),
        "ext_coeff": np.asarray(static.parameters.ext_coeff, dtype=np.float64),
        "maint_resp_slope": np.asarray(static.parameters.maint_resp_slope, dtype=np.float64),
        "coeff_maint_zero": np.asarray(static.parameters.coeff_maint_zero, dtype=np.float64),
        "t2m": np.concatenate(
            [np.asarray(payload["t2m"], dtype=np.float64) for payload in day.completed_entry_payloads]
        ),
        "t2m_longterm": np.repeat(np.asarray(slow["t2m_longterm"], dtype=np.float64), NSTEPS),
        "stempdiag": np.concatenate(
            [np.asarray(payload["stempdiag"], dtype=np.float64) for payload in day.completed_entry_payloads], axis=0
        ),
        "height": _repeat_first_axis(slow["height"]),
        "veget_max": _repeat_first_axis(slow["veget_max"]),
        "rprof": _repeat_first_axis(boundary.kwargs["rprof"]),
        "biomass": _repeat_first_axis(slow["biomass"]),
        "sla_calc": _repeat_first_axis(slow["sla_calc"]),
        "initial_resp": np.asarray(slow["resp_maint_part"], dtype=np.float64)[0],
    }
    owner_resp = np.stack(
        [np.asarray(result.resp_maint_part, dtype=np.float64)[0] for result in maintenance.step_results]
    )
    owner_lai = np.stack(
        [np.asarray(result.lai, dtype=np.float64)[0] for result in maintenance.step_results]
    )
    expected = {
        "lai": owner_lai,
        "resp_maint_part_radia": owner_resp,
        "resp_maint_radia": owner_resp.sum(axis=2),
        "resp_maint_part_after_accum": np.asarray(maintenance.resp_maint_part, dtype=np.float64)[0],
    }
    return inputs, expected, day


def _write_array(handle, value) -> None:
    array = np.asarray(value)
    handle.write(np.ascontiguousarray(array.ravel(order="F")).tobytes())


def _write_inputs(path: Path, values: dict[str, np.ndarray]) -> str:
    order = (
        "dt_sechiba",
        "maint_resp_min_vmax",
        "maint_resp_coeff",
        "diaglev",
        "is_tree",
        "ok_laidev",
        "ext_coeff",
        "maint_resp_slope",
        "coeff_maint_zero",
        "t2m",
        "t2m_longterm",
        "stempdiag",
        "height",
        "veget_max",
        "rprof",
        "biomass",
        "sla_calc",
        "initial_resp",
    )
    with path.open("wb") as handle:
        for name in order:
            _write_array(handle, values[name])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compile_unit() -> tuple[bytes, dict[str, object]]:
    extracted = extract_procedure_bytes(SOURCE, "maint_respiration")
    procedure = extracted.span_bytes
    template = TEMPLATE.read_bytes()
    unit = template.replace(b"! <MAINTENANCE_PROCEDURE>", procedure)
    if b"! <" in unit:
        raise ValueError("unresolved placeholder in maintenance actual Oracle")
    return unit, {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_procedure": (
            f"stomate_resp.f90::maint_respiration lines "
            f"{extracted.start_line}-{extracted.end_line}"
        ),
        "source_procedure_sha256": extracted.span_sha256,
        "harness_template_sha256": hashlib.sha256(template).hexdigest(),
        "compile_unit_sha256": hashlib.sha256(unit).hexdigest(),
    }


def _compile_and_run(work: Path, input_path: Path):
    source = work / "oracle.f90"
    executable = work / "oracle.exe"
    output_path = work / "fortran_outputs.bin"
    unit, source_metadata = _compile_unit()
    source.write_bytes(unit)
    build = compile_fortran(source, executable, DEFAULT_COMPILER)
    executed = subprocess.run(
        [str(executable), str(input_path), str(output_path)],
        cwd=work,
        env=compiler_environment(DEFAULT_COMPILER),
        capture_output=True,
        text=True,
    )
    if executed.returncode:
        raise RuntimeError(
            "Fortran maintenance actual Oracle failed:\n"
            f"stdout:\n{executed.stdout}\n"
            f"stderr:\n{executed.stderr}"
        )
    values = np.fromfile(output_path, dtype=np.float64)
    sizes = (NSTEPS * NVM, NSTEPS * NVM * NPARTS, NSTEPS * NVM, NVM * NPARTS)
    if values.size != sum(sizes):
        raise ValueError(f"unexpected Fortran output size {values.size}, expected {sum(sizes)}")
    offsets = np.cumsum((0, *sizes))
    outputs = {
        "lai": values[offsets[0] : offsets[1]].reshape((NSTEPS, NVM), order="F"),
        "resp_maint_part_radia": values[offsets[1] : offsets[2]].reshape(
            (NSTEPS, NVM, NPARTS), order="F"
        ),
        "resp_maint_radia": values[offsets[2] : offsets[3]].reshape((NSTEPS, NVM), order="F"),
        "resp_maint_part_after_accum": values[offsets[3] : offsets[4]].reshape(
            (NVM, NPARTS), order="F"
        ),
    }
    return outputs, {**source_metadata, "build": build, "run_stderr": executed.stderr}


def run(checkpoint: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, expected, day = _actual_case(checkpoint)
    with tempfile.TemporaryDirectory(prefix="orchidee_stomate_day199_actual_") as temporary:
        work = Path(temporary)
        input_path = work / "inputs.bin"
        input_sha256 = _write_inputs(input_path, inputs)
        actual, metadata = _compile_and_run(work, input_path)

    comparisons = [
        float_comparison(name, actual[name], expected[name], rtol=RTOL, atol=ATOL)
        for name in actual
    ]
    production_writeback = np.asarray(
        day.daily_process_fold.daily_fields["resp_maint_part"], dtype=np.float64
    )[0]
    comparisons.append(
        float_comparison(
            "production_daily_fields_writeback",
            production_writeback,
            actual["resp_maint_part_after_accum"],
            rtol=RTOL,
            atol=ATOL,
        )
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "landpoint_id": "319.0-057.0",
                "year": 1961,
                "day_index": 199,
                "start_tstep": START_TSTEP,
                "steps": NSTEPS,
                "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "binary_input_sha256": input_sha256,
                "snow_active_steps": [
                    START_TSTEP + index
                    for index, payload in enumerate(day.completed_entry_payloads)
                    if float(np.asarray(payload["snow"]).reshape(-1)[0]) > 0.0
                ],
                "input_shapes": {name: list(np.asarray(value).shape) for name, value in inputs.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    return write_result(
        output_dir,
        FAMILY,
        comparisons,
        {
            "landpoint_id": "319.0-057.0",
            "year": 1961,
            "day_index": 199,
            "production_boundary": "48 actual STOMATE maintenance calls and day accumulation",
            "source_metadata": metadata,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run actual Point319 Day199 maintenance Fortran Oracle")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.checkpoint.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
