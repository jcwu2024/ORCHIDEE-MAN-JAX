"""CPU-only independent validation of sealed lifecycle arrays and source receipts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.dev import global_lifecycle_science as science  # noqa: E402
from scripts.dev.global_server_reference import _evidence_mismatch  # noqa: E402
from scripts.dev.probe_global_batch_cache import atomic_json  # noqa: E402


def flatten(tree, path=""):
    """The native output contract uses builtin tuple/list/dict trees of arrays."""
    if isinstance(tree, dict):
        return [(p, a) for key in sorted(tree) for p, a in flatten(tree[key], path + f"[{key!r}]")]
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        if tree._fields != science.ModeloutResult._fields or type(tree).__name__ != "ModeloutResult":
            raise ValueError("unsupported native NamedTuple schema")
        return [
            (p, a) for name, value in zip(tree._fields, tree, strict=True) for p, a in flatten(value, path + "." + name)
        ]
    if type(tree) in (tuple, list):
        return [(p, a) for index, value in enumerate(tree) for p, a in flatten(value, path + f"[{index}]")]
    if tree is None:
        return []
    if not isinstance(tree, (np.ndarray, np.generic)):
        raise ValueError(f"saved-array native leaf is not NumPy: {path}")
    return [(path, np.asarray(tree))]


def structure(tree):
    if isinstance(tree, dict):
        return ("dict", tuple((key, structure(tree[key])) for key in sorted(tree)))
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return (
            "ModeloutResult",
            tuple((name, structure(value)) for name, value in zip(tree._fields, tree, strict=True)),
        )
    if type(tree) in (tuple, list):
        return (type(tree).__name__, tuple(structure(value) for value in tree))
    return "none" if tree is None else "leaf"


def map_arrays(fn, tree):
    if isinstance(tree, dict):
        return {key: map_arrays(fn, value) for key, value in tree.items()}
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return type(tree)(*(map_arrays(fn, value) for value in tree))
    if type(tree) in (tuple, list):
        return type(tree)(map_arrays(fn, value) for value in tree)
    return None if tree is None else fn(np.asarray(tree))


def compare_tree(actual, reference, *, sentinel_paths, exact_float=False):
    if structure(actual) != structure(reference):
        return {"passed": False, "failures": [{"reason": "schema"}]}
    failures, allowed_sentinels, max_abs, max_ratio = [], [], 0.0, 0.0
    left, right = flatten(actual), flatten(reference)
    for (path, a), (other_path, r) in zip(left, right, strict=True):
        if path != other_path or a.shape != r.shape or a.dtype != r.dtype:
            failures.append(
                {
                    "path": path,
                    "reason": "shape_or_dtype",
                    "actual": [list(a.shape), str(a.dtype)],
                    "reference": [list(r.shape), str(r.dtype)],
                }
            )
            continue
        if a.dtype.kind in "biu":
            if not np.array_equal(a, r):
                failures.append({"path": path, "reason": "discrete"})
            continue
        if a.dtype.kind not in "fc":
            raise ValueError(f"unsupported native numeric dtype: {path}/{a.dtype}")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
            contract = sentinel_paths.get(path)
            allowed = np.zeros(a.shape, bool)
            if contract and a.ndim in (2, 3) and a.shape[-2:] == tuple(contract["inner_shape"]):
                allowed[..., contract["bare_soil_index"]] = True
            matched = np.isnan(a) & np.isnan(r) & allowed
            unexpected = ((~np.isfinite(a)) | (~np.isfinite(r))) & ~matched
            if np.any(unexpected):
                failures.append(
                    {
                        "path": path,
                        "reason": "nonfinite",
                        "actual_count": int((~np.isfinite(a)).sum()),
                        "reference_count": int((~np.isfinite(r)).sum()),
                        "unexpected_coordinates": np.argwhere(unexpected).tolist(),
                    }
                )
                continue
            allowed_sentinels.append({"path": path, "count": int(matched.sum()), "contract": contract})
            a, r = np.where(matched, 0.0, a), np.where(matched, 0.0, r)
        delta = np.abs(a - r)
        ratio = float(np.max(delta / (1e-8 + 1e-10 * np.abs(r)), initial=0))
        max_abs, max_ratio = max(max_abs, float(delta.max(initial=0))), max(max_ratio, ratio)
        if ratio > 1 or (exact_float and np.any(delta != 0)):
            failures.append({"path": path, "reason": "tolerance", "max_bound_ratio": ratio})
    return {
        "passed": not failures,
        "compared_leaves": len(left),
        "max_abs": max_abs,
        "max_bound_ratio": max_ratio,
        "failures": failures,
        "atol": 1e-8,
        "rtol": 1e-10,
        "nonfinite_policy": "fail_except_exact_source_backed_bare_soil_roughheight_nan_mask",
        "allowed_sentinels": allowed_sentinels,
    }


def require_comparison(actual, expected, sentinels, *, name, exact=False):
    record = compare_tree(actual, expected, sentinel_paths=sentinels, exact_float=exact)
    if record["passed"] is not True:
        raise ValueError(f"independent saved-array comparison failed: {name}: {record}")
    return record


def native_schema(outputs, reference, lanes):
    candidate, original = flatten(outputs), flatten(reference)
    if (
        structure(outputs) != structure(reference)
        or len(candidate) != 760
        or len(original) != 760
        or any(
            a.shape != (lanes, *r.shape[1:]) or a.dtype != r.dtype or r.shape[0] != 64
            for (_, a), (_, r) in zip(candidate, original, strict=True)
        )
    ):
        raise ValueError("trusted FULL64 full760 native output shape/dtype/mask schema mismatch")
    return [{"shape": list(a.shape), "dtype": str(a.dtype)} for _, a in candidate]


def compare_reference_lanes(outputs, reference, sentinel_paths):
    """Vectorize across lanes, preserving the original per-lane comparator records."""
    output_leaves = flatten(outputs)
    records = [
        {
            "passed": True,
            "compared_leaves": len(output_leaves),
            "max_abs": 0.0,
            "max_bound_ratio": 0.0,
            "failures": [],
            "atol": 1e-8,
            "rtol": 1e-10,
            "nonfinite_policy": "fail_except_exact_source_backed_bare_soil_roughheight_nan_mask",
            "allowed_sentinels": [],
        }
        for _ in range(2048)
    ]
    maximum_abs, maximum_ratio = np.zeros(2048), np.zeros(2048)
    sources = np.arange(2048) % 64
    for (path, a), (_, original) in zip(output_leaves, flatten(reference), strict=True):
        r = original[sources]
        axis = tuple(range(1, a.ndim))
        if a.dtype.kind in "biu":
            mismatch = np.any(a != r, axis=axis)
            if np.any(mismatch):
                raise ValueError(
                    f"independent saved-array discrete lane mask failed: {path}/{np.flatnonzero(mismatch)[0]}"
                )
            continue
        if a.dtype.kind not in "fc":
            raise ValueError(f"unsupported native dtype: {path}")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
            allowed = np.zeros(a.shape, bool)
            contract = sentinel_paths.get(path)
            if contract and a.ndim in (3, 4) and a.shape[-2:] == tuple(contract["inner_shape"]):
                allowed[..., contract["bare_soil_index"]] = True
            matched = np.isnan(a) & np.isnan(r) & allowed
            unexpected = ((~np.isfinite(a)) | (~np.isfinite(r))) & ~matched
            if np.any(unexpected):
                raise ValueError(f"independent saved-array nonfinite lane failed: {path}")
            for index, count in enumerate(np.sum(matched, axis=axis)):
                if count:
                    records[index]["allowed_sentinels"].append(
                        {"path": path, "count": int(count), "contract": contract}
                    )
            a, r = np.where(matched, 0.0, a), np.where(matched, 0.0, r)
        delta = np.abs(a - r)
        maximum_abs = np.maximum(maximum_abs, np.max(delta, axis=axis))
        maximum_ratio = np.maximum(maximum_ratio, np.max(delta / (1e-8 + 1e-10 * np.abs(r)), axis=axis))
    if np.any(maximum_ratio > 1):
        raise ValueError(
            f"independent saved-array FULL64 tolerance failed at lane {np.flatnonzero(maximum_ratio > 1)[0]}"
        )
    for index, record in enumerate(records):
        record.update(max_abs=float(maximum_abs[index]), max_bound_ratio=float(maximum_ratio[index]))
    return records


def validate_metrics(measurements):
    from scripts.dev.benchmark_global_gpu_capacity import lane_metrics

    if set(measurements) != {"core", "wrapper"}:
        raise ValueError("two complete lifecycle timing boundaries required")
    for metric in measurements.values():
        if metric != {**metric, **lane_metrics(metric.get("samples_seconds", []), 2048)}:
            raise ValueError("five synchronized sample timing summaries mismatch")


def validate_memory(proof, report, identity, outputs):
    from scripts.dev.benchmark_global_gpu_capacity import stable_resource_identity

    actual, previous = (
        proof.get("executable_memory_analysis", {}),
        proof.get("admission_compiler_memory_records", {}).get("1024", {}),
    )
    for analysis in (actual, previous):
        if (
            analysis.get("available") is not True
            or type(analysis.get("temp_size_in_bytes")) is not int
            or analysis["temp_size_in_bytes"] < 0
        ):
            raise ValueError("actual and admitted nonnegative compiler temporary evidence required")
    preflight, updated = proof.get("memory_preflight", {}), proof.get("postcompile_memory_budget_validation", {})
    resident, output_bytes = (
        preflight.get("resident_2048_input_bytes"),
        sum(array.nbytes for _, array in flatten(outputs)),
    )
    if type(resident) is not int or resident <= 0 or preflight.get("full_2048_output_bytes") != output_bytes:
        raise ValueError("actual logical root/output byte evidence mismatch")
    budget = preflight.get("physical_gpu", {})
    selected = identity["run_binding"]["physical_gpu"]
    estimated = 2 * resident + 3 * output_bytes + max(actual["temp_size_in_bytes"], previous["temp_size_in_bytes"])
    if (
        any(budget.get(key) != value for key, value in selected.items())
        or updated.get("original_preallocation_physical_gpu_budget") != budget
        or updated.get("estimated_device_bytes") != estimated
        or updated.get("required_physical_slack_bytes") != 2 * 1024**3
        or updated.get("resident_2048_input_bytes") != resident
        or updated.get("full_2048_output_bytes") != output_bytes
        or updated.get("actual_executable_memory_analysis") != actual
        or updated.get("permitted") is not True
        or type(budget.get("free_bytes")) is not int
        or budget["free_bytes"] < estimated + 2 * 1024**3
        or any(
            updated.get("postcompile_physical_gpu_snapshot", {}).get(key) != value for key, value in selected.items()
        )
    ):
        raise ValueError("original preallocation2R+3O+scaledtemp+2GiB guard changed or failed")
    if (
        stable_resource_identity(proof.get("server_resources", {})) != identity["run_binding"]["server_resources"]
        or stable_resource_identity(report.get("server_resources", {})) != identity["run_binding"]["server_resources"]
    ):
        raise ValueError("actual resource allocation evidence mismatch")
    events = proof.get("memory_observations", [])
    expected = [("core_roots_release_before", 1), ("core_roots_release_after", 1)]
    expected += [(f"wrapper_sample_{event}", index) for index in range(1, 6) for event in ("start", "end")]
    if [(event.get("event"), event.get("sample_index")) for event in events] != expected or report.get(
        "lifecycle_memory_observations"
    ) != events:
        raise ValueError("complete outside-timer lifecycle observation event order required")
    release = identity["measurement_strategy"] == "phased_release"
    for index, event in enumerate(events):
        owned = not release or index == 0
        if (
            event.get("logical_root_owned") is not owned
            or event.get("logical_root_bytes") != (resident if owned else 0)
            or event.get("peak_unavailable") is not True
            or type(event.get("host_RSS_bytes")) is not int
            or event["host_RSS_bytes"] <= 0
            or any(event.get("physical_gpu", {}).get(key) != value for key, value in selected.items())
        ):
            raise ValueError("lifecycle logical ownership/host-RSS/UUID snapshot evidence mismatch")


def validate_stage_receipt(
    directory, *, strategy, expected_code_sha, common_identity=None, reference_dir=None, acceptance_path=None
):
    directory = Path(directory).resolve()
    manifest_path, report_path = directory / "stage_manifest.json", directory / "report.json"
    manifest, report = json.loads(manifest_path.read_text()), json.loads(report_path.read_text())
    identity = manifest["identity"]
    science.validate_identity(identity, strategy=strategy, expected_code_sha=expected_code_sha)
    checkout_source = science.source_identity(ROOT, expected_code_sha=expected_code_sha, require_clean=True)
    for key, value in checkout_source.items():
        if identity.get(key) != value:
            raise ValueError(f"sealed identity differs from actual immutable scientific checkout: {key}")
    common = science.common_identity(identity)
    if common_identity is not None and common != common_identity:
        raise ValueError("lifecycle common runtime/source/order/UUID/allocator/workload identity mismatch")
    stage = manifest.get("stages", {}).get(science.STAGE_NAME, {})
    path = directory / f"stage_{science.STAGE_NAME}.pkl"
    sha, count = science.digest_file(path)
    if (
        stage.get("status") != "complete"
        or stage.get("path") != path.name
        or stage.get("sha256") != sha
        or stage.get("checkpoint_io", {}).get("bytes") != count
    ):
        raise ValueError("lifecycle sealed checkpoint status/path/hash/bytes mismatch")
    envelope = science.load_array_pickle(path.read_bytes())
    if (
        envelope.get("identity") != identity
        or envelope.get("name") != science.STAGE_NAME
        or envelope.get("status") != "complete"
    ):
        raise ValueError("lifecycle checkpoint envelope/identity mismatch")
    record, value = envelope["record"], envelope["value"]
    for key in record:
        if _evidence_mismatch(record[key], stage.get(key), f"record.{key}") is not None:
            raise ValueError(f"lifecycle checkpoint/manifest proof differs: {key}")
    if (
        report.get("status") != "completed_lifecycle_diagnostic"
        or report.get("completed_shapes") != [1024]
        or report.get("identity") != identity
        or report.get("stages", {}).get(science.STAGE_NAME) != stage
        or _evidence_mismatch(
            record.get("execution_origin", {}).get("environment"),
            report.get("environment"),
            "record.execution_origin.environment",
        )
        is not None
    ):
        raise ValueError("lifecycle sealed report/manifest/origin mismatch")
    origin = record.get("execution_origin", {})
    epoch = origin.get("process_start_epoch")
    if (
        origin.get("available") is not True
        or type(origin.get("pid")) is not int
        or origin["pid"] <= 0
        or type(epoch) not in (int, float)
        or not math.isfinite(epoch)
        or epoch <= 0
        or origin.get("environment", {}).get("git_head") != expected_code_sha
    ):
        raise ValueError("lifecycle actual code/process origin mismatch")
    environment = origin["environment"]
    if any(environment.get(key) != value for key, value in identity["runtime"].items()) or any(
        environment.get("env", {}).get(key) != value for key, value in identity["compiler_environment"].items()
    ):
        raise ValueError("complete actual runtime/compiler origin differs from common identity")
    if any(
        environment.get("physical_gpu", {}).get(key) != value
        for key, value in identity["run_binding"]["physical_gpu"].items()
    ):
        raise ValueError("actual origin GPU identity mismatch")
    expected_samples = [
        (f"capacity_B1024_{boundary}_{suffix}", index)
        for boundary in ("core", "wrapper")
        for index in range(1, 6)
        for suffix in ("start", "complete")
    ]
    actual_samples = [
        (phase["phase"], phase.get("sample_index"))
        for phase in report.get("phases", [])
        if phase.get("phase") in {name for name, _ in expected_samples}
    ]
    if actual_samples != expected_samples:
        raise ValueError("actual saved sample event order is not five_core_then_five_wrapper")
    gates = record.get("gates", [])
    names = [f"capacity_B1024_lane_{i}_source_{i % 64}_GPU_reference" for i in range(2048)]
    names += [f"capacity_B1024_{name}" for name in ("frozen_reset_repeat", "permutation", "replacement_isolation")]
    if (
        record.get("measurement_status") != "complete"
        or record.get("strict_numerical_passed") is not True
        or len(gates) != 2051
        or [gate.get("name") for gate in gates] != names
        or any(
            gate.get("passed") is not True or gate.get("atol") != 1e-8 or gate.get("rtol") != 1e-10 for gate in gates
        )
    ):
        raise ValueError("exact lifecycle2051 gate names/order/fixed tolerances required")
    proof = record.get("provenance", {})
    for key, expected in {
        "total_lanes": 2048,
        "lane_days": 28672,
        "unique_real_points": 64,
        "chunk_size": 1024,
        "chunks_per_collection": 2,
        "blocks": [[2, 8], [9, 15]],
        "half_hours_per_day": 48,
        "full_outputs": True,
        "full_two_block_workload": True,
        "process_strategy": science.PROCESS_STRATEGY,
        **science.measurement_fields(strategy),
    }.items():
        if proof.get(key) != expected:
            raise ValueError(f"lifecycle workload/provenance mismatch: {key}")
    validate_metrics(value["measurements"])
    if value["measurements"] != record.get("timing"):
        raise ValueError("lifecycle measured arrays/stage timing differs")
    if reference_dir is None:
        raise ValueError("independent validation requires explicit frozen FULL64 reference directory")
    reference, reference_origin = science.load_comparison_reference(reference_dir, identity)
    sentinels = reference_origin["sentinel_paths"]
    schema = native_schema(value["outputs"], reference, 2048)
    if proof.get("full_output_native_schema") != schema:
        raise ValueError("recorded native schema differs from actual arrays/trusted reference")
    validate_memory(proof, report, identity, value["outputs"])
    arrays = value.get("validation_arrays", {})
    if set(arrays) != {"initial_output", "frozen_reset_output", "permutation_output", "replacement_output", "handoffs"}:
        raise ValueError("complete actual initial/reset/permutation/replacement/continuity arrays required")
    initial = arrays["initial_output"]
    native_schema(initial, reference, 2048)
    rechecked = []
    for index, result in enumerate(compare_reference_lanes(initial, reference, sentinels)):
        for key in result:
            if _evidence_mismatch(result[key], gates[index].get(key), f"gate.{key}") is not None:
                raise ValueError(
                    f"saved GPU-reference gate differs from independently measured arrays: {names[index]}/{key}"
                )
        rechecked.append({"name": names[index], **result})
    expectations = [
        initial,
        map_arrays(lambda a: a[np.arange(1023, -1, -1)], initial),
        map_arrays(lambda a: a[np.array(list(range(1023)) + [0])], initial),
    ]
    for offset, (name, key, expected) in enumerate(
        zip(
            names[2048:], ("frozen_reset_output", "permutation_output", "replacement_output"), expectations, strict=True
        ),
        start=2048,
    ):
        result = require_comparison(arrays[key], expected, sentinels, name=name)
        if any(_evidence_mismatch(result[k], gates[offset].get(k), f"gate.{k}") is not None for k in result):
            raise ValueError(f"saved isolation/reset gate differs from actual arrays: {name}")
        rechecked.append({"name": name, **result})
    require_comparison(
        value["outputs"], arrays["frozen_reset_output"], sentinels, name="saved_final_output", exact=True
    )
    binding = proof.get("continuous_state_binding", {})
    if binding != science.run_two_binding(ROOT):
        raise ValueError("continuous source-bound run_two identity required")
    handoffs = arrays["handoffs"]
    if not isinstance(handoffs, list) or len(handoffs) != 2:
        raise ValueError("actual continuous handoff arrays required for both initial chunks")
    state_sentinels = {
        path.removeprefix("[0][0]"): contract for path, contract in sentinels.items() if path.startswith("[0][0]")
    }
    for index, handoff in enumerate(handoffs):
        if set(handoff) != {"first_output_state", "second_input_state"}:
            raise ValueError("continuous handoff record field mismatch")
        expected_state = map_arrays(lambda a: a[index * 1024 : (index + 1) * 1024], initial[0][0])
        require_comparison(
            handoff["first_output_state"], expected_state, state_sentinels, name="first_block_output_state", exact=True
        )
        require_comparison(
            handoff["second_input_state"],
            handoff["first_output_state"],
            state_sentinels,
            name="continuous_second_input_state",
            exact=True,
        )
    acceptance_path = Path(acceptance_path) if acceptance_path else directory / "array_acceptance.json"
    acceptance = {
        "contract": "independent-lifecycle-saved-array-validation-v1",
        "passed": True,
        "checkpoint_sha256": sha,
        "checkpoint_bytes": count,
        "identity": identity,
        "reference": reference_origin,
        "native_leaves": 760,
        "FULL64_lane_comparisons": 2048,
        "gate_count": len(rechecked),
        "gates": rechecked,
        "continuous_handoffs": 2,
        "continuous_handoff_exact": True,
        "schema_dtype_masks": "actual arrays compared against frozen FULL64",
        "peak_VRAM": {"available": False, "scope": "discrete snapshots only"},
        "scientific_acceptance": False,
    }
    atomic_json(acceptance_path, acceptance)
    return {
        "identity": identity,
        "common_identity": common,
        "stage": stage,
        "path": str(path),
        "sha256": sha,
        "bytes": count,
        "execution_origin": origin,
        "array_acceptance_path": str(acceptance_path),
        "array_acceptance_sha256": science.digest_file(acceptance_path)[0],
        "manifest_sha256": science.digest_file(manifest_path)[0],
        "report_sha256": science.digest_file(report_path)[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--measurement-strategy", choices=science.STRATEGIES, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--acceptance-path", type=Path)
    args = parser.parse_args()
    receipt = validate_stage_receipt(
        args.directory,
        strategy=args.measurement_strategy,
        expected_code_sha=args.expected_code_sha,
        reference_dir=args.reference_dir,
        acceptance_path=args.acceptance_path,
    )
    print(json.dumps(receipt, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
