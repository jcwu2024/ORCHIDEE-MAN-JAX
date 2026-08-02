from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs" / "teacher_branch_parity.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "branch_parity"
DEFAULT_WORKTREES = ROOT / "runtime" / "worktrees"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_git(*args: str, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_revision(revision: str) -> str:
    return _run_git("rev-parse", f"{revision}^{{commit}}")


def _ensure_worktree(path: Path, commit: str) -> None:
    if path.exists():
        actual = _run_git("rev-parse", "HEAD", cwd=path)
        if actual != commit:
            raise RuntimeError(f"worktree {path} is at {actual}, expected {commit}")
        if _run_git("status", "--porcelain", cwd=path):
            raise RuntimeError(f"worktree {path} is not clean")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_git("worktree", "add", "--detach", str(path), commit)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _block_until_ready(value: object) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    jax.effects_barrier()


def _materialize_run_def(reference: object, output: Path) -> Path:
    from jax_orchidee.driver.run_def_materialization import (
        materialize_case_run_def_values,
        write_materialized_run_def,
    )

    if reference.run_def is None or reference.used_run_def is None:
        raise FileNotFoundError(
            f"{reference.landpoint_id} needs run.def and z1/used_run.def for branch parity"
        )
    values = materialize_case_run_def_values(
        base_used_run_def=reference.used_run_def,
        case_run_def=reference.run_def,
        base_is_fortran_used_truth=True,
    )
    return write_materialized_run_def(
        values,
        output,
        header_lines=(
            "# Deterministic branch-parity runtime run.def.",
            f"# Source: {reference.used_run_def}",
        ),
    )


def _daily_output_from_cold(first: object) -> object:
    from jax_orchidee.driver.orchestration import _driver_modelout_from_outputs

    return _driver_modelout_from_outputs(
        first.stomate_outputs,
        day_index=1,
        start_tstep=0,
        steps_per_stomate=48,
    )


def _daily_output_from_reference(first: object) -> object:
    from jax_orchidee.driver.orchestration import _driver_day_modelout

    return _driver_day_modelout(first, day_index=1)


def _advance_sequence(
    *,
    scenario: str,
    config_path: Path,
    run_def: Path,
    reference_run_dir: Path,
    asset_root: Path,
    year: int,
    days: int,
    runtime_flags: Mapping[str, object],
    previous_year_end_state: object | None = None,
) -> tuple[list[object], list[object], object]:
    from jax_orchidee.driver.orchestration import (
        paper_1961_driver_cold_start_day_scaffold,
        paper_1961_driver_day_scaffold,
        paper_1961_driver_later_day_runtime_result,
        paper_1961_driver_restart_year_start_day_result,
        prepare_paper_1961_driver_context,
    )

    context = prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=run_def,
        reference_run_dir=reference_run_dir,
    )
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    if steps_per_day != 48:
        raise RuntimeError(f"expected 48 SECHIBA steps/day, got {steps_per_day}")

    if scenario == "cold":
        first = paper_1961_driver_cold_start_day_scaffold(
            config_path,
            year=year,
            used_run_def_path=run_def,
            reference_run_dir=reference_run_dir,
            prepared_context=context,
            module_jit=bool(runtime_flags["module_jit"]),
            diffuco_local_jit=bool(runtime_flags["diffuco_local_jit"]),
        )
        output = _daily_output_from_cold(first)
        state = first.first_day_end_state
        ready = first.ready_for_first_day_end_state
        missing = first.missing_components
    elif scenario == "reference_start":
        first = paper_1961_driver_day_scaffold(
            config_path,
            year=year,
            start_tstep=0,
            used_run_def_path=run_def,
            reference_run_dir=reference_run_dir,
            root=asset_root,
            prepared_context=context,
            module_jit=bool(runtime_flags["module_jit"]),
            diffuco_local_jit=bool(runtime_flags["diffuco_local_jit"]),
            retain_stomate_step_results=False,
            single_pass_daily_fold=False,
            use_static_jit_daily_carbon=bool(runtime_flags["use_static_jit_daily_carbon"]),
        )
        output = _daily_output_from_reference(first)
        state = first.first_day_end_state
        ready = first.ready_for_first_day_end_state
        missing = first.missing_components
    elif scenario == "restart":
        first = paper_1961_driver_restart_year_start_day_result(
            config_path,
            previous_year_end_state=previous_year_end_state,
            year=year,
            used_run_def_path=run_def,
            prepared_context=context,
            retain_stomate_step_results=False,
            **runtime_flags,
        )
        output = first.daily_modelout
        state = first.day_end_state
        ready = first.ready_for_day_end_state
        missing = first.missing_components
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    if not ready or output is None or state is None:
        raise RuntimeError(f"{scenario} day 1 incomplete: {missing}")
    outputs = [output]
    states = [state]
    for day_index in range(2, days + 1):
        day = paper_1961_driver_later_day_runtime_result(
            config_path,
            previous_state=state,
            day_index=day_index,
            year=year,
            start_tstep=(day_index - 1) * steps_per_day,
            used_run_def_path=run_def,
            prepared_context=context,
            retain_stomate_step_results=False,
            **runtime_flags,
        )
        if not day.ready_for_day_end_state or day.daily_modelout is None or day.day_end_state is None:
            raise RuntimeError(f"{scenario} day {day_index} incomplete: {day.missing_components}")
        output = day.daily_modelout
        state = day.day_end_state
        outputs.append(output)
        states.append(state)
    return outputs, states, state


def _collect_snapshot(
    path: str,
    value: object,
    arrays: dict[str, np.ndarray],
    scalars: dict[str, object],
) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _collect_snapshot(f"{path}/{key}", value[key], arrays, scalars)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect_snapshot(f"{path}/{field.name}", getattr(value, field.name), arrays, scalars)
        return
    if hasattr(value, "_asdict"):
        for key, item in value._asdict().items():
            _collect_snapshot(f"{path}/{key}", item, arrays, scalars)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _collect_snapshot(f"{path}/{index:04d}", item, arrays, scalars)
        return
    if value is None or isinstance(value, (str, bool, int, float)):
        scalars[path] = value
        return
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise TypeError(f"unsupported object leaf at {path}: {type(value)!r}")
    arrays[path] = array


def _write_snapshot(output_dir: Path, payload: Mapping[str, object], metadata: Mapping[str, object]) -> None:
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, object] = {}
    _collect_snapshot("root", payload, arrays, scalars)
    ordered = sorted(arrays)
    keys = {path: f"array_{index:06d}" for index, path in enumerate(ordered)}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "arrays.npz", **{keys[path]: arrays[path] for path in ordered})
    _write_json(
        output_dir / "schema.json",
        {
            "metadata": dict(metadata),
            "scalars": scalars,
            "arrays": {
                path: {
                    "key": keys[path],
                    "dtype": str(arrays[path].dtype),
                    "shape": list(arrays[path].shape),
                }
                for path in ordered
            },
        },
    )


def _case_input_identities(
    *, config_path: Path, run_def: Path, reference: object, years: Sequence[int], data_root: Path
) -> list[dict[str, object]]:
    paths = [config_path, run_def, reference.run_def, reference.used_run_def]
    for name in (
        "driver_start",
        "sechiba_start",
        "stomate_start",
        "driver_restart",
        "sechiba_restart",
        "stomate_restart",
    ):
        paths.append(getattr(reference, name))
    paths.extend(data_root / "forcing" / f"cruncep_twodeg_{year}.nc" for year in years)
    unique = sorted({Path(path).resolve() for path in paths if path is not None})
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing parity inputs: {missing}")
    return [_file_identity(path) for path in unique]


def _worker(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve()
    asset_root = args.asset_root.resolve()
    output_root = args.output.resolve()
    os.environ["ORCHIDEE_REPO_ROOT"] = str(source_root)
    os.environ["ORCHIDEE_DATA_ROOT"] = str(asset_root / "data")
    os.environ["ORCHIDEE_REFERENCE_ROOT"] = str(asset_root / "reference")
    os.environ["ORCHIDEE_OUTPUT_ROOT"] = str(output_root / "runtime")
    sys.path.insert(0, str(source_root))

    import jax
    import jaxlib

    from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference
    from jax_orchidee.runtime import configure_jax_compilation_cache

    configure_jax_compilation_cache(source_root)
    manifest = _read_json(args.manifest)
    runtime_flags = dict(manifest["runtime"])
    cases = [case for case in manifest["cases"] if args.group in {"all", case["group"]}]
    if not cases:
        raise ValueError(f"manifest has no cases in group {args.group}")
    config_path = source_root / "configs" / "orchidee_man_250919.yaml"
    branch_summary: dict[str, object] = {
        "branch_id": args.branch_id,
        "commit": _run_git("rev-parse", "HEAD", cwd=source_root),
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "cases": {},
    }
    for case in cases:
        started = time.perf_counter()
        reference = resolve_paper_landpoint_reference(asset_root, case["landpoint_id"])
        if reference.output_dir is None:
            raise FileNotFoundError(f"missing reference package for {case['landpoint_id']}")
        case_dir = output_root / "branch_runs" / case["id"] / args.branch_id
        schema_path = case_dir / "schema.json"
        if args.resume and schema_path.is_file() and (case_dir / "arrays.npz").is_file():
            existing = _read_json(schema_path).get("metadata", {})
            if (
                existing.get("commit") == branch_summary["commit"]
                and existing.get("case") == case
                and existing.get("runtime_flags") == runtime_flags
            ):
                branch_summary["cases"][case["id"]] = existing
                print(json.dumps({"branch": args.branch_id, "case": case["id"], "resumed": True}))
                continue
        run_def = _materialize_run_def(reference, case_dir / "used_run.def")
        source_state = None
        source_payload = None
        years = [int(case["year"])]
        if case["scenario"] == "restart":
            source_year = int(case["source_year"])
            source_outputs, source_states, source_state = _advance_sequence(
                scenario=case["source_scenario"],
                config_path=config_path,
                run_def=run_def,
                reference_run_dir=reference.output_dir,
                asset_root=asset_root,
                year=source_year,
                days=int(case["source_days"]),
                runtime_flags=runtime_flags,
            )
            source_payload = {
                "last_modelout": source_outputs[-1],
                "last_state": source_states[-1],
            }
            years.append(source_year)
        daily_outputs, daily_states, final_state = _advance_sequence(
            scenario=case["scenario"],
            config_path=config_path,
            run_def=run_def,
            reference_run_dir=reference.output_dir,
            asset_root=asset_root,
            year=int(case["year"]),
            days=int(case["days"]),
            runtime_flags=runtime_flags,
            previous_year_end_state=source_state,
        )
        _block_until_ready((daily_outputs, daily_states, final_state))
        payload = {
            "source": source_payload,
            "days": {
                f"day_{index:04d}": {"modelout": output, "state": state}
                for index, (output, state) in enumerate(zip(daily_outputs, daily_states, strict=True), start=1)
            },
            "final_state": final_state,
        }
        inputs = _case_input_identities(
            config_path=config_path,
            run_def=run_def,
            reference=reference,
            years=years,
            data_root=asset_root / "data",
        )
        metadata = {
            "case": case,
            "branch_id": args.branch_id,
            "commit": branch_summary["commit"],
            "runtime_flags": runtime_flags,
            "input_files": inputs,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_snapshot(case_dir, payload, metadata)
        branch_summary["cases"][case["id"]] = metadata
        print(json.dumps({"branch": args.branch_id, "case": case["id"], "elapsed_seconds": metadata["elapsed_seconds"]}))
    _write_json(output_root / f"worker_{args.branch_id}.json", branch_summary)
    return 0


def _load_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    schema = _read_json(path / "schema.json")
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[item["key"]]) for name, item in schema["arrays"].items()}
    return schema, arrays


def _ulp_distance(left: np.ndarray, right: np.ndarray) -> int:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left64) & np.isfinite(right64)
    if not np.any(finite):
        return 0
    left_bits = left64[finite].view(np.uint64)
    right_bits = right64[finite].view(np.uint64)
    sign = np.uint64(1 << 63)
    left_ordered = np.where(left_bits & sign, ~left_bits, left_bits | sign)
    right_ordered = np.where(right_bits & sign, ~right_bits, right_bits | sign)
    distance = np.where(
        left_ordered >= right_ordered,
        left_ordered - right_ordered,
        right_ordered - left_ordered,
    )
    return int(distance.max(initial=np.uint64(0)))


def _day_from_path(path: str) -> int | None:
    match = re.search(r"/day_(\d{4})/", path)
    return int(match.group(1)) if match else None


def _compare_case(
    *, case: Mapping[str, object], baseline_dir: Path, candidate_dir: Path, atol: float, rtol: float
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline_schema, baseline_arrays = _load_snapshot(baseline_dir)
    candidate_schema, candidate_arrays = _load_snapshot(candidate_dir)
    rows: list[dict[str, object]] = []
    scalar_equal = baseline_schema["scalars"] == candidate_schema["scalars"]
    all_paths = sorted(set(baseline_arrays) | set(candidate_arrays))
    for path in all_paths:
        left = baseline_arrays.get(path)
        right = candidate_arrays.get(path)
        row: dict[str, object] = {
            "case_id": case["id"],
            "landpoint_id": case["landpoint_id"],
            "day": _day_from_path(path),
            "path": path,
        }
        if left is None or right is None:
            row.update(status="missing", max_abs_error="", max_rel_error="", max_ulp_error="")
        elif left.shape != right.shape or left.dtype.kind != right.dtype.kind:
            row.update(status="schema_mismatch", max_abs_error="", max_rel_error="", max_ulp_error="")
        elif left.dtype.kind not in "fc":
            row.update(
                status="passed" if np.array_equal(left, right) else "discrete_mismatch",
                max_abs_error="",
                max_rel_error="",
                max_ulp_error="",
            )
        else:
            left64 = left.astype(np.float64, copy=False)
            right64 = right.astype(np.float64, copy=False)
            same_nonfinite = (np.isnan(left64) & np.isnan(right64)) | (left64 == right64)
            absolute = np.where(same_nonfinite, 0.0, np.abs(left64 - right64))
            relative = absolute / np.maximum(np.abs(left64), np.finfo(np.float64).tiny)
            close = np.allclose(left64, right64, atol=atol, rtol=rtol, equal_nan=True)
            row.update(
                status="passed" if close else "float_mismatch",
                max_abs_error=float(absolute.max(initial=0.0)),
                max_rel_error=float(relative.max(initial=0.0)),
                max_ulp_error=_ulp_distance(left64, right64),
            )
        rows.append(row)
    failures = [row for row in rows if row["status"] != "passed"]
    def content_identities(schema: Mapping[str, object]) -> list[tuple[int, str]]:
        return sorted(
            (int(item["size"]), str(item["sha256"]))
            for item in schema["metadata"]["input_files"]
        )

    input_equal = content_identities(baseline_schema) == content_identities(candidate_schema)
    first_days = [row["day"] for row in failures if row["day"] is not None]
    return {
        "case_id": case["id"],
        "passed": not failures and scalar_equal and input_equal,
        "field_count": len(rows),
        "failure_count": len(failures),
        "first_differing_day": min(first_days) if first_days else None,
        "first_differing_field": failures[0]["path"] if failures else None,
        "scalar_metadata_equal": scalar_equal,
        "input_identities_equal": input_equal,
    }, rows


def _apply_disposition(
    *,
    summary: dict[str, object],
    rows: Sequence[Mapping[str, object]],
    dispositions: Sequence[Mapping[str, object]],
    asset_root: Path,
) -> dict[str, object]:
    summary = dict(summary)
    summary["raw_passed"] = bool(summary["passed"])
    summary["accepted"] = bool(summary["passed"])
    summary["disposition_id"] = None
    if summary["passed"]:
        return summary
    matching = [item for item in dispositions if item["case_id"] == summary["case_id"]]
    if len(matching) != 1:
        return summary
    disposition = matching[0]
    failures = [row for row in rows if row["status"] != "passed"]
    first_day = int(disposition["first_differing_day"])
    first_failures = [row for row in failures if row["day"] == first_day]
    expected_fields = set(disposition["first_day_fields"])
    observed_fields = {str(row["path"]) for row in first_failures}
    continuous_only = all(row["status"] == "float_mismatch" for row in failures)
    first_day_bounded = all(
        float(row["max_abs_error"]) <= float(disposition["first_day_max_abs_error"])
        for row in first_failures
    )
    oracle = disposition["oracle"]
    oracle_path = asset_root / str(oracle["path"])
    oracle_valid = False
    if oracle_path.is_file() and _sha256(oracle_path).lower() == str(oracle["sha256"]).lower():
        oracle_valid = _read_json(oracle_path).get("status") == oracle["required_status"]
    accepted = (
        summary["first_differing_day"] == first_day
        and observed_fields == expected_fields
        and continuous_only
        and first_day_bounded
        and oracle_valid
    )
    summary.update(
        accepted=accepted,
        disposition_id=disposition["id"] if accepted else None,
        disposition_checks={
            "first_day_matches": summary["first_differing_day"] == first_day,
            "first_day_fields_match": observed_fields == expected_fields,
            "continuous_only": continuous_only,
            "first_day_bounded": first_day_bounded,
            "oracle_valid": oracle_valid,
            "oracle_path": str(oracle_path),
        },
    )
    return summary


def _write_report(output: Path, result: Mapping[str, object]) -> None:
    lines = [
        "# Teacher Branch Parity",
        "",
        f"Status: **{str(result['status']).upper()}**",
        "",
        f"Baseline: `{result['baseline']['commit']}`",
        f"Candidate: `{result['candidate']['commit']}`",
        f"Cases: {result['passed_cases']}/{result['case_count']} passed",
        "",
        "| Case | Status | First differing day | First differing field |",
        "| --- | --- | ---: | --- |",
    ]
    for case in result["cases"]:
        case_status = (
            "passed"
            if case["raw_passed"]
            else f"accepted: {case['disposition_id']}"
            if case["accepted"]
            else "failed"
        )
        lines.append(
            f"| {case['case_id']} | {case_status} | "
            f"{case['first_differing_day'] or ''} | {case['first_differing_field'] or ''} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coordinator(args: argparse.Namespace) -> int:
    manifest_template = _read_json(args.manifest)
    baseline = dict(manifest_template["baseline"])
    candidate = dict(manifest_template["candidate"])
    baseline["commit"] = _resolve_revision(str(baseline["revision"]))
    candidate["commit"] = _resolve_revision(str(candidate["revision"]))
    output = args.output.resolve()
    worktree_root = args.worktree_root.resolve()
    worktrees = {
        baseline["id"]: worktree_root
        / f"teacher-parity-{baseline['id']}-{baseline['commit'][:12]}",
        candidate["id"]: worktree_root
        / f"teacher-parity-{candidate['id']}-{candidate['commit'][:12]}",
    }
    for branch in (baseline, candidate):
        _ensure_worktree(worktrees[branch["id"]], branch["commit"])

    cases = [case for case in manifest_template["cases"] if args.group in {"all", case["group"]}]
    generated_manifest = {
        **manifest_template,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_group": args.group,
        "baseline": baseline,
        "candidate": candidate,
        "cases": cases,
        "asset_root": str(ROOT),
        "runner_sha256": _sha256(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    generated_manifest_path = output / "manifest.json"
    _write_json(generated_manifest_path, generated_manifest)
    for branch in (baseline, candidate):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--manifest",
            str(generated_manifest_path),
            "--source-root",
            str(worktrees[branch["id"]]),
            "--asset-root",
            str(ROOT),
            "--output",
            str(output),
            "--branch-id",
            str(branch["id"]),
            "--group",
            args.group,
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=worktrees[branch["id"]], check=True)

    tolerances = manifest_template["tolerances"]
    summaries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for case in cases:
        summary, case_rows = _compare_case(
            case=case,
            baseline_dir=output / "branch_runs" / case["id"] / baseline["id"],
            candidate_dir=output / "branch_runs" / case["id"] / candidate["id"],
            atol=float(tolerances["atol"]),
            rtol=float(tolerances["rtol"]),
        )
        summary = _apply_disposition(
            summary=summary,
            rows=case_rows,
            dispositions=manifest_template.get("dispositions", ()),
            asset_root=ROOT,
        )
        summaries.append(summary)
        rows.extend(case_rows)
    with (output / "field_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "landpoint_id",
                "day",
                "path",
                "status",
                "max_abs_error",
                "max_rel_error",
                "max_ulp_error",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(bool(case["accepted"]) for case in summaries)
    raw_passed = sum(bool(case["raw_passed"]) for case in summaries)
    result = {
        "status": "passed" if passed == len(summaries) else "failed",
        "baseline": baseline,
        "candidate": candidate,
        "tolerances": tolerances,
        "case_count": len(summaries),
        "passed_cases": passed,
        "raw_passed_cases": raw_passed,
        "cases": summaries,
    }
    _write_json(output / "comparison.json", result)
    _write_report(output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare canonical Teacher outputs across two Git commits.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREES)
    parser.add_argument("--group", choices=("smoke", "promotion", "all"), default="smoke")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse a completed snapshot only when commit, case, and runtime flags match",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--asset-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--branch-id", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if args.source_root is None or args.asset_root is None or args.branch_id is None:
            raise ValueError("worker mode requires source-root, asset-root, and branch-id")
        return _worker(args)
    return _coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
