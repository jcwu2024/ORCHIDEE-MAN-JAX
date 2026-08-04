from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.orchestration import (  # noqa: E402
    paper_1961_driver_later_day_runtime_result,
    paper_1961_driver_multiday_modelout_lite_run,
    prepare_paper_1961_driver_context,
)
from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402
from research.daily_coarse_graining.daily_flux_capture import (  # noqa: E402
    write_daily_flux_capture,
)


def _children(value: Any):
    if isinstance(value, Mapping):
        return value.items()
    if is_dataclass(value) and not isinstance(value, type):
        return ((field.name, getattr(value, field.name)) for field in fields(value))
    if hasattr(value, "_asdict"):
        return value._asdict().items()
    return None


def _compare(actual: Any, expected: Any, path: str = "root") -> list[dict[str, Any]]:
    actual_children = _children(actual)
    expected_children = _children(expected)
    if expected_children is not None:
        if actual_children is None:
            return [{"path": path, "reason": "structure"}]
        actual_map = dict(actual_children)
        expected_map = dict(expected_children)
        if actual_map.keys() != expected_map.keys():
            return [{"path": path, "reason": "keys"}]
        mismatches: list[dict[str, Any]] = []
        for name in expected_map:
            mismatches.extend(_compare(actual_map[name], expected_map[name], f"{path}.{name}"))
        return mismatches
    if expected is None or isinstance(expected, (str, bytes)):
        return [] if actual == expected else [{"path": path, "reason": "value"}]
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape or actual_array.dtype != expected_array.dtype:
        return [{"path": path, "reason": "array_contract"}]
    equal = (
        np.array_equal(actual_array, expected_array, equal_nan=True)
        if actual_array.dtype.kind in "fc"
        else np.array_equal(actual_array, expected_array)
    )
    if not equal:
        if actual_array.dtype.kind not in "biufc":
            return [{"path": path, "reason": "value"}]
        delta = np.abs(actual_array.astype(np.float64) - expected_array.astype(np.float64))
        return [{"path": path, "reason": "numeric", "max_abs": float(np.nanmax(delta))}]
    return []


def _block(value: Any) -> None:
    children = _children(value)
    if children is not None:
        for _, child in children:
            _block(child)
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify daily-only Teacher flux capture.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs/reference_mode/used_run.def")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "manifests/coarse_graining/daily_flux_label_inventory_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/research/daily_coarse_graining/gate_c2_daily_flux_capture",
    )
    args = parser.parse_args()

    configure_jax_compilation_cache(ROOT)
    seed = paper_1961_driver_multiday_modelout_lite_run(
        args.config,
        ndays=1,
        year=1961,
        used_run_def_path=args.run_def,
        root=ROOT,
        module_jit=True,
        diffuco_local_jit=True,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    if not seed.ready_for_requested_days or seed.last_day_end_state is None:
        raise RuntimeError(f"day-1 seed failed: {seed.missing_components}")
    context = prepare_paper_1961_driver_context(
        args.config,
        used_run_def_path=args.run_def,
    )
    common = dict(
        previous_state=seed.last_day_end_state,
        day_index=2,
        year=1961,
        start_tstep=48,
        used_run_def_path=args.run_def,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
        maintenance_local_jit=True,
        retain_stomate_step_results=False,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
        use_compiled_sechiba_day=True,
    )
    ordinary = paper_1961_driver_later_day_runtime_result(
        args.config,
        **common,
        capture_daily_flux_labels=False,
    )
    captured = paper_1961_driver_later_day_runtime_result(
        args.config,
        **common,
        capture_daily_flux_labels=True,
    )
    _block(ordinary)
    _block(captured)
    if not ordinary.ready_for_day_end_state or not captured.ready_for_day_end_state:
        raise RuntimeError(
            f"day-2 transition failed: ordinary={ordinary.missing_components}, captured={captured.missing_components}"
        )
    if ordinary.daily_flux_labels is not None or captured.daily_flux_labels is None:
        raise RuntimeError("capture switch did not preserve default-off semantics")
    mismatches = _compare(captured.daily_modelout, ordinary.daily_modelout, "daily_modelout")
    mismatches.extend(_compare(captured.day_end_state, ordinary.day_end_state, "day_end_state"))
    if mismatches:
        raise RuntimeError(f"capture changed ordinary scientific outputs: {mismatches[:10]}")
    report = write_daily_flux_capture(
        captured.daily_flux_labels,
        output_dir=args.output,
        inventory_path=args.inventory,
        metadata={
            "year": 1961,
            "day_index": 2,
            "start_tstep": 48,
            "capture_default_off": True,
            "ordinary_scientific_outputs_exact": True,
            "jax_version": jax.__version__,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
