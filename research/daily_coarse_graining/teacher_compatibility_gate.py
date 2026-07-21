"""Capture and compare Teacher daily-boundary fingerprints across JAX runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.replay_ceiling import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_DEF,
    _load_state_cache,
)
from research.daily_coarse_graining.supervised_learnability_pilot import (
    _capture_days,
    _is_float,
    _teacher_target_vector,
    build_boundary_vector_spec,
)
from research.daily_coarse_graining.synthetic_operator_cost import (
    COARSE_STATE_COMPONENTS,
)

ROOT = Path(__file__).resolve().parents[2]
TEACHER_COMMIT = "7333b46c0b38650fb6c9250582876137831657b8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _artifact_paths(prefix: Path) -> tuple[Path, Path]:
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return prefix.with_suffix(".npz"), prefix.with_suffix(".json")


def _leaf_metadata(spec) -> list[dict[str, Any]]:
    return [
        {
            "family": leaf.family,
            "component": leaf.component,
            "name": leaf.name,
            "shape": list(leaf.shape),
            "start": leaf.start,
            "stop": leaf.stop,
        }
        for leaf in spec.leaves
    ]


def _discrete_arrays(records) -> dict[str, np.ndarray]:
    keys: dict[str, list[np.ndarray]] = {}
    for record in records:
        fields = record.half_hour_transition.current_state.fields_by_component
        for component in COARSE_STATE_COMPONENTS:
            for name, value in fields[component].items():
                if _is_float(value):
                    continue
                key = f"discrete__{component}__{name}"
                keys.setdefault(key, []).append(np.asarray(value))
    return {key: np.stack(values) for key, values in keys.items()}


def capture_fingerprint(
    *,
    state_cache: Path,
    config: Path,
    run_def: Path,
    reference_run_dir: Path | None,
    year: int,
    days: int,
    output_prefix: Path,
) -> dict[str, Any]:
    cache = _load_state_cache(state_cache)
    context = teacher.prepare_paper_1961_driver_context(
        config,
        used_run_def_path=run_def,
        reference_run_dir=reference_run_dir,
    )
    restart_state = context.first_step_restart_state
    restart_paths = {
        "driver_start.nc": restart_state.driver_start,
        "sechiba_start.nc": restart_state.sechiba_start,
        "stomate_start.nc": restart_state.stomate_input,
        "stomate_restart.nc": (
            context.first_step_stomate_boundary.stomate_files.restart
        ),
    }
    initial_state = teacher.rebase_driver_state_for_year_start(cache["state"])
    _, _, records, _ = _capture_days(
        config_path=config,
        context=context,
        previous_state=initial_state,
        year=year,
        start_day=1,
        days=days,
    )
    spec = build_boundary_vector_spec(records[0])
    arrays = {
        "continuous": np.stack(
            [_teacher_target_vector(record, spec) for record in records]
        ),
        **_discrete_arrays(records),
    }
    npz_path, metadata_path = _artifact_paths(output_prefix)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)
    metadata = {
        "schema_version": "teacher_daily_boundary_fingerprint_v1",
        "teacher_commit": TEACHER_COMMIT,
        "experiment_git_head": _git_head(),
        "jax": {
            "version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "x64_enabled": bool(jax.config.x64_enabled),
        },
        "platform": platform.platform(),
        "capture": {
            "year": year,
            "days": days,
            "state_cache_sha256": _sha256(state_cache),
            "config_sha256": _sha256(config),
            "run_def_sha256": _sha256(run_def),
            "reference_restart_sha256": {
                name: _sha256(path) for name, path in restart_paths.items()
            },
            "continuous_shape": list(arrays["continuous"].shape),
            "continuous_dtype": str(arrays["continuous"].dtype),
            "leaves": _leaf_metadata(spec),
            "discrete": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
                if key != "continuous"
            },
        },
        "npz": str(npz_path),
        "npz_sha256": _sha256(npz_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _array_metrics(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    same_shape = expected.shape == actual.shape
    if not same_shape:
        return {
            "passed": False,
            "same_shape": False,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
        }
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    finite = np.isfinite(expected) & np.isfinite(actual)
    if np.any(finite):
        difference = np.abs(actual[finite] - expected[finite])
        denominator = np.maximum(np.abs(expected[finite]), np.finfo(np.float64).tiny)
        max_abs = float(np.max(difference))
        max_rel = float(np.max(difference / denominator))
    else:
        max_abs = 0.0
        max_rel = 0.0
    return {
        "passed": bool(np.allclose(expected, actual, atol=atol, rtol=rtol, equal_nan=True)),
        "same_shape": True,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "nonfinite_pattern_equal": bool(
            np.array_equal(np.isnan(expected), np.isnan(actual))
            and np.array_equal(np.isposinf(expected), np.isposinf(actual))
            and np.array_equal(np.isneginf(expected), np.isneginf(actual))
        ),
    }


def compare_fingerprints(
    expected_prefix: Path,
    actual_prefix: Path,
    *,
    output: Path,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected_npz, expected_json = _artifact_paths(expected_prefix)
    actual_npz, actual_json = _artifact_paths(actual_prefix)
    expected_meta = json.loads(expected_json.read_text(encoding="utf-8"))
    actual_meta = json.loads(actual_json.read_text(encoding="utf-8"))
    contract_fields = (
        "teacher_commit",
        "experiment_git_head",
    )
    contract = {
        name: expected_meta[name] == actual_meta[name] for name in contract_fields
    }
    expected_capture = expected_meta["capture"]
    actual_capture = actual_meta["capture"]
    for name in (
        "year",
        "days",
        "state_cache_sha256",
        "config_sha256",
        "run_def_sha256",
        "reference_restart_sha256",
        "leaves",
    ):
        contract[f"capture.{name}"] = expected_capture[name] == actual_capture[name]

    continuous_leaves = []
    discrete = {}
    with np.load(expected_npz) as expected_data, np.load(actual_npz) as actual_data:
        expected_continuous = expected_data["continuous"]
        actual_continuous = actual_data["continuous"]
        for leaf in expected_capture["leaves"]:
            metrics = _array_metrics(
                expected_continuous[:, leaf["start"] : leaf["stop"]],
                actual_continuous[:, leaf["start"] : leaf["stop"]],
                atol=atol,
                rtol=rtol,
            )
            continuous_leaves.append({**leaf, **metrics})
        expected_keys = set(expected_data.files) - {"continuous"}
        actual_keys = set(actual_data.files) - {"continuous"}
        contract["discrete_keys"] = expected_keys == actual_keys
        for key in sorted(expected_keys & actual_keys):
            equal = bool(np.array_equal(expected_data[key], actual_data[key]))
            discrete[key] = {"passed": equal, "exact": equal}

    failed_leaves = [
        f"{leaf['family']}:{leaf['component']}:{leaf['name']}"
        for leaf in continuous_leaves
        if not leaf["passed"]
    ]
    passed = bool(
        all(contract.values())
        and not failed_leaves
        and all(item["passed"] for item in discrete.values())
    )
    report = {
        "schema_version": "teacher_daily_boundary_compatibility_v1",
        "decision": "compatible" if passed else "incompatible",
        "passed": passed,
        "tolerances": {"atol": atol, "rtol": rtol, "discrete": "exact"},
        "expected_runtime": expected_meta["jax"],
        "actual_runtime": actual_meta["jax"],
        "contract": contract,
        "continuous": {
            "leaf_count": len(continuous_leaves),
            "failed_count": len(failed_leaves),
            "failed_leaves": failed_leaves,
            "max_abs_error": max(
                (leaf.get("max_abs_error", 0.0) for leaf in continuous_leaves),
                default=0.0,
            ),
            "max_rel_error": max(
                (leaf.get("max_rel_error", 0.0) for leaf in continuous_leaves),
                default=0.0,
            ),
            "leaves": continuous_leaves,
        },
        "discrete": discrete,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--state-cache", type=Path, required=True)
    capture.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    capture.add_argument("--run-def", type=Path, default=DEFAULT_RUN_DEF)
    capture.add_argument("--reference-run-dir", type=Path)
    capture.add_argument("--year", type=int, default=1961)
    capture.add_argument("--days", type=int, default=1)
    capture.add_argument("--output-prefix", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--expected-prefix", type=Path, required=True)
    compare.add_argument("--actual-prefix", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--atol", type=float, default=1.0e-10)
    compare.add_argument("--rtol", type=float, default=1.0e-10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture":
        result = capture_fingerprint(
            state_cache=args.state_cache,
            config=args.config,
            run_def=args.run_def,
            reference_run_dir=args.reference_run_dir,
            year=args.year,
            days=args.days,
            output_prefix=args.output_prefix,
        )
        print(json.dumps({"jax": result["jax"], "npz": result["npz"]}, indent=2))
        return 0
    result = compare_fingerprints(
        args.expected_prefix,
        args.actual_prefix,
        output=args.output,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "failed_leaves": result["continuous"]["failed_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
