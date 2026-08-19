"""Compare runtime thermal reconstruction inputs with the true v6 endpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_mechanistic_output_v2_attainability import (
    _build_attainability_head,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_strong_baseline_one_point_qualification import (
    _git_head,
    _prepare_point_year,
    _sha256_file,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

SOURCE_ATOL = 1.0e-10
SOURCE_RTOL = 1.0e-12
THERMAL_SOURCE_NAMES = (
    "ptn",
    "shum_ngrnd_permalong",
    "ref_soc",
    "mc",
    "mcl",
    "veget_max",
    "frac_nobio",
    "lai",
    "temp_sol",
    "temp_sol_pft",
    "snow",
    "snow_nobio",
    "snowrho",
    "snowtemp",
    "snowdz",
    "pressure_hpa",
    "njsc",
    "hydrol_zz_mm",
    "hydrol_dz_mm",
    "pref_soil_veg",
    "ext_coeff_vegetfrac",
    "model_interval_seconds",
)
THERMAL_COEFFICIENT_TARGETS = {
    "cgrnd": ("thermosoil_previous_step_state", "cgrnd", 2),
    "dgrnd": ("thermosoil_previous_step_state", "dgrnd", 2),
    "cgrnd_snow": ("thermosoil_previous_step_state", "cgrnd_snow", None),
    "dgrnd_snow": ("thermosoil_previous_step_state", "dgrnd_snow", None),
    "lambda_snow": ("thermosoil_previous_step_state", "lambda_snow", None),
    "pcapa_en": ("thermosoil_previous_step_state", "pcapa_en", 2),
    "soilcap": ("thermosoil_previous_step_state", "soilcap", None),
    "soilcap_pft": ("enerbil_previous_step_state", "soilcap_pft", 1),
    "soilflx": ("thermosoil_previous_step_state", "soilflx", None),
    "soilflx_pft": ("enerbil_previous_step_state", "soilflx_pft", 1),
}


def _compact(value: Any, indices: np.ndarray, axis: int | None) -> np.ndarray:
    array = np.asarray(value)
    return array if axis is None else np.take(array, indices, axis=axis)


def _error_summary(actual: Any, target: Any) -> dict[str, Any]:
    left = np.asarray(actual)
    right = np.asarray(target)
    if left.shape != right.shape:
        raise ValueError(f"thermal source shape drift: {left.shape} != {right.shape}")
    left_numeric = np.asarray(left, dtype=np.float64)
    right_numeric = np.asarray(right, dtype=np.float64)
    defined = np.isfinite(right_numeric) & (np.abs(right_numeric) < 0.5e20)
    if not np.any(defined):
        raise ValueError("thermal source target has no defined values")
    difference = np.where(defined, np.abs(left_numeric - right_numeric), 0.0)
    tolerance = SOURCE_ATOL + SOURCE_RTOL * np.abs(right_numeric)
    flat_index = int(np.argmax(difference))
    index = tuple(int(value) for value in np.unravel_index(flat_index, difference.shape))
    unequal = defined & (left_numeric != right_numeric)
    failed = defined & (difference > tolerance)
    return {
        "shape": list(left.shape),
        "defined_value_count": int(np.count_nonzero(defined)),
        "bitwise_unequal_value_count": int(np.count_nonzero(unequal)),
        "tolerance_failure_count": int(np.count_nonzero(failed)),
        "maximum_absolute_error": float(difference[index]),
        "maximum_tolerance_ratio": float(np.max(np.where(defined, difference / tolerance, 0.0))),
        "maximum_error_index": list(index),
        "actual_at_maximum_error": float(left_numeric[index]),
        "target_at_maximum_error": float(right_numeric[index]),
        "bitwise_exact": bool(not np.any(unequal)),
        "within_tolerance": bool(not np.any(failed)),
    }


def _raw_static(sample: Mapping[str, Any], contract: Any, name: str) -> np.ndarray:
    leaf = next(
        leaf
        for leaf in contract.static_conditions.landpoint_static_leaves
        if leaf.name == name
    )
    return np.asarray(sample["landpoint_static"])[leaf.start : leaf.stop].reshape(leaf.shape)


def _teacher_sources_and_coefficients(
    runtime: Mapping[str, Any],
    row: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from research.daily_coarse_graining.daily_markov_contract import (
        reconstruct_fast_day_target,
        reconstruct_state_fields,
    )

    sample = runtime["ordinary_parent"].sample(row)
    contract = runtime["contract"]
    start = reconstruct_state_fields(
        np.asarray(sample["state"], dtype=np.float64),
        sample["discrete_state"],
        contract,
    )
    endpoint = reconstruct_fast_day_target(
        np.asarray(sample["fast_day_target"], dtype=np.float64),
        contract.fast_day_target_leaves,
        template_fields=start,
    )
    fields = endpoint.fields_by_component
    active = np.asarray(
        runtime["adapter"].input_assembler.active_pft_indices,
        dtype=np.int32,
    )
    thermal = fields["thermosoil_previous_step_state"]
    hydrol = fields["hydrol_previous_step_state"]
    diffuco = fields["diffuco_previous_step_state"]
    enerbil = fields["enerbil_previous_step_state"]
    forcing_index = {
        name: index for index, name in enumerate(contract.native_forcing.fields)
    }
    sources = {
        "ptn": _compact(thermal["ptn"], active, 2),
        "shum_ngrnd_permalong": _compact(
            thermal["shum_ngrnd_permalong"], active, 2
        ),
        "ref_soc": np.asarray(thermal["refSOC"]),
        "mc": np.asarray(hydrol["mc"]),
        "mcl": np.asarray(hydrol["mcl"]),
        "veget_max": _compact(diffuco["veget_max"], active, 1),
        "frac_nobio": np.asarray(diffuco["frac_nobio"]),
        "lai": _compact(diffuco["lai"], active, 1),
        "temp_sol": np.asarray(enerbil["temp_sol"]),
        "temp_sol_pft": _compact(diffuco["temp_sol_pft"], active, 1),
        "snow": np.asarray(hydrol["snow"]),
        "snow_nobio": np.asarray(hydrol["snow_nobio"]),
        "snowrho": np.asarray(hydrol["snowrho"]),
        "snowtemp": np.asarray(hydrol["snowtemp"]),
        "snowdz": np.asarray(hydrol["snowdz"]),
        "pressure_hpa": np.asarray(
            [sample["forcing_native"][-1, forcing_index["PSurf"]] / 100.0]
        ),
        "njsc": np.asarray(_raw_static(sample, contract, "njsc"), dtype=np.int32).reshape(-1),
        "hydrol_zz_mm": _raw_static(sample, contract, "hydrol_zz_mm"),
        "hydrol_dz_mm": _raw_static(sample, contract, "hydrol_dz_mm"),
        "pref_soil_veg": _compact(
            _raw_static(sample, contract, "pref_soil_veg"), active, 0
        ),
        "ext_coeff_vegetfrac": _compact(
            _raw_static(sample, contract, "ext_coeff_vegetfrac"), active, 0
        ),
        "model_interval_seconds": np.asarray(
            contract.native_forcing.model_interval_seconds,
            dtype=np.float64,
        ),
    }
    coefficients = {
        name: _compact(fields[component][field], active, axis)
        for name, (component, field, axis) in THERMAL_COEFFICIENT_TARGETS.items()
    }
    return sources, coefficients


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.doc_fixed_point_iterations < 1:
        raise ValueError("DOC fixed-point iterations must be positive")
    started = time.perf_counter()
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]

    from research.daily_coarse_graining.daily_operator.mechanistic_output_attainability import (
        materialize_attainability_oracle,
    )

    head = _build_attainability_head(runtime)
    dataset = _prepare_point_year(runtime)
    rows = tuple(dict.fromkeys(int(row) for row in args.row))
    if not rows:
        raise ValueError("at least one source-alignment row is required")
    sample_count = int(dataset["sample_count"])
    if any(row < 0 or row >= sample_count for row in rows):
        raise ValueError(f"source-alignment row must be in [0, {sample_count})")
    inputs_all = jax.device_put(dataset["assembled_inputs"])
    targets_all = jax.device_put(dataset["targets"])

    coefficient_output_keys = {
        name: (
            f"endpoint.fast_state.enerbil_previous_step_state.{name}"
            if name in {"soilcap_pft", "soilflx_pft"}
            else f"endpoint.fast_state.thermosoil_previous_step_state.{name}"
        )
        for name in THERMAL_COEFFICIENT_TARGETS
    }

    @jax.jit
    def one_day(inputs: Any, targets: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        oracle = materialize_attainability_oracle(
            head,
            targets,
            inputs,
            doc_fixed_point_iterations=args.doc_fixed_point_iterations,
        )
        fields, _ok_leak = head.reconstructor.updated_fast_owner_fields(
            oracle.prediction,
            oracle.fast_update,
            inputs,
        )
        sources = head.reconstructor.thermal_coefficient_inputs(fields, inputs)
        coefficients = {
            name: oracle.prediction.named_outputs[key]
            for name, key in coefficient_output_keys.items()
        }
        return sources, coefficients

    records = []
    for row in rows:
        inputs = jax.tree_util.tree_map(lambda value: value[row], inputs_all)
        targets = jax.tree_util.tree_map(lambda value: value[row], targets_all)
        actual_sources, actual_coefficients = one_day(inputs, targets)
        jax.block_until_ready((actual_sources, actual_coefficients))
        actual_sources = jax.device_get(actual_sources)
        actual_coefficients = jax.device_get(actual_coefficients)
        teacher_sources, teacher_coefficients = _teacher_sources_and_coefficients(
            runtime,
            row,
        )
        source_records = {
            name: _error_summary(actual_sources[name], teacher_sources[name])
            for name in THERMAL_SOURCE_NAMES
        }
        coefficient_records = {
            name: _error_summary(actual_coefficients[name], teacher_coefficients[name])
            for name in THERMAL_COEFFICIENT_TARGETS
        }
        records.append(
            {
                "row": row,
                "day_index": int(dataset["day_indices"][row]),
                "sources": source_records,
                "coefficients": coefficient_records,
            }
        )

    source_summary = {
        name: {
            "bitwise_unequal_value_count": sum(
                record["sources"][name]["bitwise_unequal_value_count"]
                for record in records
            ),
            "tolerance_failure_count": sum(
                record["sources"][name]["tolerance_failure_count"]
                for record in records
            ),
            "maximum_absolute_error": max(
                record["sources"][name]["maximum_absolute_error"]
                for record in records
            ),
            "maximum_tolerance_ratio": max(
                record["sources"][name]["maximum_tolerance_ratio"]
                for record in records
            ),
        }
        for name in THERMAL_SOURCE_NAMES
    }
    coefficient_summary = {
        name: {
            "tolerance_failure_count": sum(
                record["coefficients"][name]["tolerance_failure_count"]
                for record in records
            ),
            "maximum_absolute_error": max(
                record["coefficients"][name]["maximum_absolute_error"]
                for record in records
            ),
            "maximum_tolerance_ratio": max(
                record["coefficients"][name]["maximum_tolerance_ratio"]
                for record in records
            ),
        }
        for name in THERMAL_COEFFICIENT_TARGETS
    }
    bitwise_mismatched_sources = [
        name
        for name in THERMAL_SOURCE_NAMES
        if source_summary[name]["bitwise_unequal_value_count"] > 0
    ]
    tolerance_failed_sources = [
        name
        for name in THERMAL_SOURCE_NAMES
        if source_summary[name]["tolerance_failure_count"] > 0
    ]
    failed_coefficients = [
        name
        for name in THERMAL_COEFFICIENT_TARGETS
        if coefficient_summary[name]["tolerance_failure_count"] > 0
    ]
    report = {
        "schema_version": "mechanistic_output_v2_thermal_source_alignment_v1",
        "status": "completed",
        "identity": {
            "git_head": _git_head(),
            "dataset_manifest_sha256": _sha256_file(args.dataset_manifest),
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_used": False,
            "neural_body_executed": False,
            "optimizer_updates": 0,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "selection": {
            "row_count": len(rows),
            "rows": list(rows),
            "day_indices": [int(dataset["day_indices"][row]) for row in rows],
        },
        "summary": {
            "bitwise_mismatched_sources": bitwise_mismatched_sources,
            "tolerance_failed_sources": tolerance_failed_sources,
            "failed_coefficients": failed_coefficients,
        },
        "sources": source_summary,
        "coefficients": coefficient_summary,
        "records": records,
        "device": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in runtime["devices"]],
            "memory": [_memory_report(device) for device in runtime["devices"]],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--row", action="append", type=int, required=True)
    parser.add_argument("--doc-fixed-point-iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
