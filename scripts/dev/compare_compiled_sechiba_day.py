from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.runtime import configure_jax_compilation_cache  # noqa: E402

configure_jax_compilation_cache(ROOT)

from jax_orchidee.driver.orchestration import paper_1961_driver_multiday_modelout_lite_run  # noqa: E402


def _numeric_leaves(value):
    for leaf in jax.tree_util.tree_leaves(value):
        array = np.asarray(leaf)
        if array.dtype.kind in "biufc":
            yield array


def _difference(actual, expected) -> tuple[float, float]:
    max_abs = 0.0
    max_rel = 0.0
    for actual_leaf, expected_leaf in zip(_numeric_leaves(actual), _numeric_leaves(expected), strict=True):
        if actual_leaf.dtype.kind == "b":
            delta = np.not_equal(actual_leaf, expected_leaf).astype(np.float64)
            scale = np.ones_like(delta)
        else:
            actual_numeric = actual_leaf.astype(np.float64, copy=False)
            expected_numeric = expected_leaf.astype(np.float64, copy=False)
            delta = np.abs(actual_numeric - expected_numeric)
            scale = np.maximum(np.abs(expected_numeric), np.finfo(np.float64).tiny)
        max_abs = max(max_abs, float(np.max(delta, initial=0.0)))
        max_rel = max(max_rel, float(np.max(delta / scale, initial=0.0)))
    return max_abs, max_rel


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare strict and compiled SECHIBA day transitions.")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--year", type=int, default=1961)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "orchidee_man_250919.yaml")
    parser.add_argument("--run-def", type=Path, default=ROOT / "outputs" / "reference_mode" / "used_run.def")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--use-numpy-accumulator", choices=("on", "off"), default="off")
    args = parser.parse_args()

    common = dict(
        year=args.year,
        ndays=args.days,
        used_run_def_path=args.run_def,
        root=ROOT,
        module_jit=True,
        diffuco_local_jit=True,
        compact_later_days=True,
        use_static_jit_daily_carbon=True,
        prebuild_day_payloads=True,
    )
    strict = paper_1961_driver_multiday_modelout_lite_run(
        args.config,
        **common,
        use_compiled_sechiba_day=False,
        use_numpy_accumulator=False,
    )
    compiled = paper_1961_driver_multiday_modelout_lite_run(
        args.config,
        **common,
        use_compiled_sechiba_day=True,
        use_numpy_accumulator=args.use_numpy_accumulator == "on",
    )

    field_differences = {}
    ok = strict.ready_for_requested_days and compiled.ready_for_requested_days
    for strict_day, compiled_day in zip(strict.daily_modelout, compiled.daily_modelout, strict=True):
        for name, expected in strict_day.modelout_fields.items():
            actual = compiled_day.modelout_fields[name]
            max_abs, max_rel = _difference(actual, expected)
            current = field_differences.setdefault(name, {"max_abs": 0.0, "max_rel": 0.0})
            current["max_abs"] = max(current["max_abs"], max_abs)
            current["max_rel"] = max(current["max_rel"], max_rel)
            ok = ok and np.allclose(np.asarray(actual), np.asarray(expected), rtol=args.rtol, atol=args.atol, equal_nan=True)

    state_differences = {}
    strict_state = strict.last_day_end_state
    compiled_state = compiled.last_day_end_state
    for component, expected_fields in strict_state.fields_by_component.items():
        actual_fields = compiled_state.fields_by_component[component]
        for name, expected in expected_fields.items():
            max_abs, max_rel = _difference(actual_fields[name], expected)
            if max_abs:
                state_differences[f"{component}:{name}"] = {"max_abs": max_abs, "max_rel": max_rel}
            for actual_leaf, expected_leaf in zip(
                _numeric_leaves(actual_fields[name]),
                _numeric_leaves(expected),
                strict=True,
            ):
                ok = ok and np.allclose(actual_leaf, expected_leaf, rtol=args.rtol, atol=args.atol, equal_nan=True)

    report = {
        "ok": bool(ok),
        "days": args.days,
        "year": args.year,
        "atol": args.atol,
        "rtol": args.rtol,
        "use_numpy_accumulator": args.use_numpy_accumulator,
        "field_differences": field_differences,
        "state_differences": state_differences,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
