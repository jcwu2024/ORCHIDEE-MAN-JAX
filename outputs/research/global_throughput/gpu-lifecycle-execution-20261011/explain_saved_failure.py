"""Read-only NumPy explanation of job15769234's saved initial arrays.

No model imports, execution, JAX, JIT, GPU, tolerance changes or cause claims.
Writes complete failing-coordinate detail as compact gzip JSONL, with exact
repeat grouping only after comparing coordinates and failing values bytewise.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import platform
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np

try:
    import resource
except ModuleNotFoundError:  # Windows local help/sanity; target Linux has resource.
    resource = None

CODE_PIN = "3b4ae35e695f85480ecb72461e4ba93f4534bbd8"
FAILED_SHA = "698d3dece201424778d9cf4b0f10c1ac3b1802a1ce35b7d231f73b751641ddc5"
FAILED_BYTES = 880948565
REFERENCE_SHA = "a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6"
REFERENCE_BYTES = 27587892
LAYOUT_AUDIT_SHA = "259ddf5358c69b3d01b05924c17b709764f95a8ceebdfa993f00c39df01d0180"
PREPARED_SHA = "be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7"
STATIC_DIGEST = "b1b4d5b9819ff8845ecb7b0cc68bf641da1c5740effc1c13185f9bde872fe329"
ATOL, RTOL = 1e-8, 1e-10


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "failed-input", "reference-dir", "report", "output-dir", "state-layout-json"):
        result.add_argument("--" + name, type=Path)
    result.add_argument("--lane-chunk", type=int, default=64)
    result.add_argument("--budget-seconds", type=float, default=540)
    result.add_argument("--self-check", action="store_true")
    return result


def scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return value if np.isfinite(value) else ("NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity"))


def peak_rss_bytes():
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def comparison_mask(actual, reference, contract=None):
    """Native dtype/discrete comparison and exact trusted bare-soil NaN mask."""
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("coordinate comparison requires exact native shape/dtype")
    if actual.dtype.kind in "biu":
        return actual != reference, np.zeros(actual.shape, bool), "discrete_exact"
    if actual.dtype.kind != "f":
        raise ValueError(f"unsupported saved native dtype: {actual.dtype}")
    allowed = np.zeros(actual.shape, bool)
    if contract and actual.ndim in (3, 4) and actual.shape[-2:] == tuple(contract["inner_shape"]):
        allowed[..., contract["bare_soil_index"]] = True
    matched = np.isnan(actual) & np.isnan(reference) & allowed
    nonfinite = ((~np.isfinite(actual)) | (~np.isfinite(reference))) & ~matched
    finite = np.isfinite(actual) & np.isfinite(reference)
    with np.errstate(over="ignore", invalid="ignore"):
        violation = finite & (np.abs(actual - reference) > ATOL + RTOL * np.abs(reference))
    return violation | nonfinite, matched, "float_fixed_tolerance_and_nonfinite"


def source_catalog(root, science):
    """Parse exact Git blobs from the execution pin; never import/evaluate them."""
    paths = science.git(root, "ls-tree", "-r", "--name-only", CODE_PIN, "--", "jax_orchidee").decode().splitlines()
    catalog, spans = {}, []
    for path in paths:
        if not path.endswith(".py"):
            continue
        raw = science.git(root, "show", f"{CODE_PIN}:{path}")
        tree = ast.parse(raw.decode("utf-8"), filename=path)
        nodes = list(ast.walk(tree))
        bounds = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        for node in nodes:
            name = node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
            if name is None:
                continue
            owners = [owner for owner in bounds if owner.lineno <= node.lineno <= owner.end_lineno]
            owner = min(owners, key=lambda candidate: candidate.end_lineno - candidate.lineno) if owners else None
            catalog.setdefault(name, []).append(
                {
                    "path": path,
                    "symbol": owner.name if owner else "module",
                    "line": node.lineno,
                    "symbol_start": owner.lineno if owner else node.lineno,
                    "symbol_end": owner.end_lineno if owner else node.end_lineno,
                }
            )
        for node in bounds:
            if node.name in {
                "fast_state_from_previous_packet",
                "_paper_later_day_block_transition_factory",
                "ModeloutResult",
            }:
                spans.append(
                    {
                        "path": path,
                        "symbol": node.name,
                        "start": node.lineno,
                        "end": node.end_lineno,
                        "git_blob_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    batch_path = "scripts/dev/benchmark_global_batch.py"
    batch_raw = science.git(root, "show", f"{CODE_PIN}:{batch_path}")
    node = next(
        node
        for node in ast.parse(batch_raw.decode()).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_two"
    )
    spans.append(
        {
            "path": batch_path,
            "symbol": "run_two",
            "start": node.lineno,
            "end": node.end_lineno,
            "git_blob_sha256": hashlib.sha256(batch_raw).hexdigest(),
        }
    )
    return catalog, spans


def layout_document(path, science):
    if path is None:
        return None, {
            "available": False,
            "reason": "saved runtime report omits initial.spec; field order cannot be inferred from shape or unordered source field sets",
        }
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    layout = document.get("state_layout", document)
    if isinstance(layout, list):
        components, fields = layout
    else:
        components, fields = layout["components"], layout["field_names_by_component"]
    if (
        len(components) != len(fields)
        or len(set(components)) != len(components)
        or any(len(set(names)) != len(names) for names in fields)
    ):
        raise ValueError("saved state layout has duplicate names or wrong component axis")
    return (components, fields), {
        "available": True,
        "path": str(path.resolve()),
        "sha256": science.digest_file(path)[0],
        "document_provenance": document.get("provenance") if isinstance(document, dict) else None,
        "authority": "saved prepared-state insertion order; exact source AST establishes serialization rule",
    }


def meaning(path, layout, catalog):
    match = re.match(r"\[([01])\]\[([01])\](.*)", path)
    if not match:
        raise ValueError(f"undeclared native output root path: {path}")
    block, branch, tail = int(match[1]), int(match[2]), match[3]
    start, end = (2, 8) if block == 0 else (9, 15)
    description = {
        "block_index": block,
        "science_days": [start, end],
        "half_hours_per_day": 48,
        "role": "block_end_state" if branch == 0 else "daily_native_observation",
        "time": f"end of science Day{end}"
        if branch == 0
        else "first inner coordinate is scan day index0..6; science day=start+index",
    }
    if branch == 0:
        indices = re.match(r"\[(\d+)\]\[(\d+)\](.*)", tail)
        if not indices:
            raise ValueError(f"undeclared state tuple path: {path}")
        ci, fi, suffix = int(indices[1]), int(indices[2]), indices[3]
        description.update(component_index=ci, field_index=fi, nested_path=suffix)
        if layout:
            components, fields = layout
            component, name = components[ci], fields[ci][fi]
            if component not in catalog or name not in catalog:
                raise ValueError(f"supplied state layout field lacks exact source evidence: {component}/{name}")
            description.update(
                component=component,
                field=name,
                named_path=component + "." + name + suffix,
                source_field_evidence=catalog[name][:8],
                field_mapping_complete=True,
            )
        else:
            description.update(field_mapping_complete=False, field=f"UNRESOLVED_STATE_TUPLE_{ci}_{fi}")
    else:
        names = re.findall(r"\['([^']+)'\]|\.([A-Za-z_][A-Za-z_0-9]*)", tail)
        name = next((left or right for left, right in reversed(names)), None)
        if name is None or name not in catalog:
            raise ValueError(f"native observation field lacks exact source evidence: {path}")
        description.update(
            field=name, named_path=tail, field_mapping_complete=True, source_field_evidence=catalog[name][:8]
        )
    return description


def write_line(stream, record):
    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def run(args, started):
    required = ("code_root", "failed_input", "reference_dir", "report", "output_dir", "state_layout_json")
    if (
        any(getattr(args, name) is None for name in required)
        or not 1 <= args.lane_chunk <= 2048
        or args.budget_seconds <= 0
    ):
        raise ValueError("all five input/output paths and positive bounded chunk/budget required")
    root = args.code_root.resolve()
    sys.path.insert(0, str(root))
    from scripts.dev import global_lifecycle_science as science
    from scripts.dev import validate_global_gpu_lifecycle as validator

    if "jax" in sys.modules or "jaxlib" in sys.modules or any(name.startswith("jax_orchidee") for name in sys.modules):
        raise ValueError("forbidden JAX/model import before saved-array comparison")
    source = science.source_identity(root, expected_code_sha=CODE_PIN, require_clean=False)
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    identity = report["identity"]
    science.validate_identity(identity, strategy="phased_resident_control", expected_code_sha=CODE_PIN)
    for key, value in source.items():
        if identity.get(key) != value:
            raise ValueError(f"saved failure scientific/code source identity differs: {key}")
    if science.digest_file(args.failed_input) != (FAILED_SHA, FAILED_BYTES):
        raise ValueError("failed initial pickle immutable checksum/size mismatch")
    if report.get("initial_validation_failure", {}).get("sha256") != FAILED_SHA:
        raise ValueError("saved report does not bind the declared failed initial arrays")
    with args.failed_input.open("rb") as stream:
        failed = science.ArrayUnpickler(stream).load()
    if set(failed) != {"identity", "initial_output"} or failed["identity"] != identity:
        raise ValueError("failed initial envelope/identity mismatch")
    actual = failed["initial_output"]
    reference, reference_origin = science.load_comparison_reference(args.reference_dir, identity)
    if (reference_origin["sha256"], reference_origin["bytes"]) != (REFERENCE_SHA, REFERENCE_BYTES):
        raise ValueError("frozen reference checksum/size mismatch")
    left, right = validator.flatten(actual), validator.flatten(reference)
    if len(left) != 760 or len(right) != 760 or validator.structure(actual) != validator.structure(reference):
        raise ValueError("saved native full760 root schema does not match frozen reference")
    catalog, source_spans = source_catalog(root, science)
    layout, layout_origin = layout_document(args.state_layout_json, science)
    provenance = layout_origin["document_provenance"]
    if (
        provenance.get("source_audit_sha256") != LAYOUT_AUDIT_SHA
        or provenance.get("canonical_prepared_pickle_sha256") != PREPARED_SHA
        or provenance.get("all60_bound_layouts_exactly_equal") is not True
        or provenance.get("all60_named_field_bijections_source_bound_indices_verified") is not True
        or provenance.get("actual_full64_reader_screen", {}).get("static_digest") != STATIC_DIGEST
        or report.get("input_compatibility_audit", {}).get("static_digest") != STATIC_DIGEST
        or provenance.get("failed_report_sha256") != science.digest_file(args.report)[0]
    ):
        raise ValueError("canonical saved layout audit/prepared/static-digest/report binding mismatch")
    if layout:
        for block in (0, 1):
            state = actual[block][0]
            if len(state) != len(layout[0]) or any(
                len(values) != len(names) for values, names in zip(state, layout[1], strict=True)
            ):
                raise ValueError("saved state layout component/field lengths differ from actual arrays")
    destination = args.output_dir.resolve()
    for input_directory in (args.failed_input.parent, args.report.parent, args.reference_dir):
        source_directory = input_directory.resolve()
        if (
            destination == source_directory
            or destination.is_relative_to(source_directory)
            or source_directory.is_relative_to(destination)
        ):
            raise ValueError("output directory overlaps a read-only input directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "all-failing-coordinates.jsonl.gz"
    if detail_path.exists() or (args.output_dir / "summary.json").exists():
        raise ValueError("explanation output already exists; preserve earlier receipts")
    origins = identity["lane_origins"]
    all_lanes, failures, sentinel_count, coordinate_count, group_count = set(), [], 0, 0, 0
    with gzip.open(detail_path, "wt", encoding="utf-8", compresslevel=6) as detail:
        for leaf_index, ((path, a), (other_path, r)) in enumerate(zip(left, right, strict=True)):
            if time.monotonic() - started > args.budget_seconds:
                raise TimeoutError("bounded explanation budget exceeded; gzip is partial, no completed summary")
            if path != other_path or a.ndim < 1 or r.ndim < 1 or a.shape[0] != 2048 or r.shape[0] != 64:
                raise ValueError(f"native lane/path schema mismatch: {path}")
            field = meaning(path, layout, catalog)
            leaf = {
                "leaf_index": leaf_index,
                "path": path,
                **field,
                "actual_dtype": str(a.dtype),
                "reference_dtype": str(r.dtype),
                "actual_shape": list(a.shape),
                "reference_shape": list(r.shape),
            }
            if a.shape[1:] != r.shape[1:] or a.dtype != r.dtype:
                all_lanes.update(range(2048))
                record = {
                    **leaf,
                    "reason": "native_shape_or_dtype",
                    "failing_lane_count": 2048,
                    "failing_lanes": list(range(2048)),
                    "failing_unique_points": [lane["landpoint_id"] for lane in identity["lanes"]],
                }
                failures.append(record)
                write_line(detail, {"record_type": "schema_failure", **record})
                continue
            grouped, fail_lanes, fail_points, native_failed_coordinates = {}, [], set(), 0
            for start in range(0, 2048, args.lane_chunk):
                stop = min(start + args.lane_chunk, 2048)
                rr = r[np.arange(start, stop) % 64]
                aa = a[start:stop]
                mask, sentinels, policy = comparison_mask(aa, rr, reference_origin["sentinel_paths"].get(path))
                sentinel_count += int(sentinels.sum())
                for local_lane in np.flatnonzero(mask.reshape(stop - start, -1).any(axis=1)):
                    lane = start + int(local_lane)
                    coords = np.argwhere(mask[local_lane])
                    select = tuple(coords.T)
                    av, rv = (
                        np.asarray(aa[local_lane][select]).reshape(-1),
                        np.asarray(rr[local_lane][select]).reshape(-1),
                    )
                    source_index = lane % 64
                    # Exact bytes, not hashes alone: same coordinates and both
                    # actual/reference failing values must match before grouping.
                    key = (source_index, coords.shape, coords.tobytes(), av.dtype.str, av.tobytes(), rv.tobytes())
                    if key not in grouped:
                        grouped[key] = {"lanes": [], "coords": coords, "actual": av, "reference": rv}
                    grouped[key]["lanes"].append(lane)
                    fail_lanes.append(lane)
                    fail_points.add(origins[lane]["landpoint_id"])
                    native_failed_coordinates += len(coords)
            if not fail_lanes:
                continue
            all_lanes.update(fail_lanes)
            coordinate_count += native_failed_coordinates
            record = {
                **leaf,
                "reason": policy,
                "failing_lane_count": len(fail_lanes),
                "failing_lanes": fail_lanes,
                "failing_unique_points": sorted(fail_points),
                "native_failing_coordinates_including_repeats": native_failed_coordinates,
                "exact_failure_groups": len(grouped),
                "detail_file": detail_path.name,
            }
            failures.append(record)
            write_line(detail, {"record_type": "leaf_failure", **record})
            for group in grouped.values():
                group_count += 1
                group_id = f"leaf{leaf_index}-group{group_count}"
                write_line(
                    detail,
                    {
                        "record_type": "exact_repeat_group",
                        "group_id": group_id,
                        "path": path,
                        "lanes": group["lanes"],
                        "lane_origins": [origins[lane] for lane in group["lanes"]],
                        "coordinate_count": len(group["coords"]),
                        "exact_repeat_equivalence_verified": True,
                        "verification": "bytewise equality of every failing coordinate and both native actual/reference values",
                    },
                )
                for coordinate, av, rv in zip(group["coords"], group["actual"], group["reference"], strict=True):
                    if time.monotonic() - started > args.budget_seconds:
                        raise TimeoutError(
                            "bounded explanation budget exceeded during coordinate writing; output remains partial"
                        )
                    actual_value, reference_value = scalar(av), scalar(rv)
                    signed = (int(av) - int(rv)) if a.dtype.kind in "biu" else float(av) - float(rv)
                    bound = ATOL + RTOL * abs(float(rv))
                    absolute = abs(signed)
                    ratio = absolute / bound
                    time_meaning = (
                        {
                            "science_day": field["science_days"][0] + int(coordinate[0]),
                            "scan_day_index": int(coordinate[0]),
                        }
                        if field["role"] == "daily_native_observation"
                        else {"end_science_day": field["science_days"][1]}
                    )
                    write_line(
                        detail,
                        {
                            "record_type": "failed_coordinate",
                            "group_id": group_id,
                            "inner_coordinate": coordinate.tolist(),
                            **time_meaning,
                            "actual": actual_value,
                            "reference": reference_value,
                            "signed_delta": scalar(signed),
                            "abs_delta": scalar(absolute),
                            "fixed_bound": scalar(bound),
                            "bound_ratio": scalar(ratio),
                            "exact_discrete_gate": a.dtype.kind in "biu",
                            "failure_reason": "discrete"
                            if a.dtype.kind in "biu"
                            else ("nonfinite" if not np.isfinite(av) or not np.isfinite(rv) else "tolerance"),
                        },
                    )
            if leaf_index % 50 == 0:
                print(
                    json.dumps(
                        {
                            "progress_leaves": leaf_index + 1,
                            "failed_leaves": len(failures),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    failing_lanes = sorted(all_lanes)
    if "jax" in sys.modules or "jaxlib" in sys.modules or any(name.startswith("jax_orchidee") for name in sys.modules):
        raise ValueError("forbidden JAX/jaxlib/model import at explanation completion")
    summary = {
        "status": "complete_saved_initial_failure_comparison",
        "all760_leaves_and2048_lanes_checked": True,
        "failed_input": {"path": str(args.failed_input.resolve()), "sha256": FAILED_SHA, "bytes": FAILED_BYTES},
        "report": {"path": str(args.report.resolve()), "sha256": science.digest_file(args.report)[0]},
        "reference": reference_origin,
        "execution_code_pin": CODE_PIN,
        "source_identity": source,
        "state_layout": layout_origin,
        "source_AST_spans": source_spans,
        "field_mapping_complete": all(record["field_mapping_complete"] for record in failures),
        "gate": {
            "atol": ATOL,
            "rtol": RTOL,
            "discrete_shape_dtype": "exact",
            "sentinels": "only trusted frozen bare-soil roughheight matched NaN coordinates",
        },
        "failing_leaf_count": len(failures),
        "failing_leaves": failures,
        "failing_lane_count": len(failing_lanes),
        "failing_lanes": failing_lanes,
        "failing_lane_range": [min(failing_lanes), max(failing_lanes)] if failing_lanes else None,
        "failing_lane_origins": [origins[lane] for lane in failing_lanes],
        "failing_unique_points": sorted({origins[lane]["landpoint_id"] for lane in failing_lanes}),
        "failing_unique_point_count": len({origins[lane]["landpoint_id"] for lane in failing_lanes}),
        "native_failing_coordinates_including_repeats": coordinate_count,
        "exact_failure_group_count": group_count,
        "allowed_sentinel_coordinates": sentinel_count,
        "complete_coordinate_detail": {
            "path": str(detail_path.resolve()),
            "sha256": science.digest_file(detail_path)[0],
            "bytes": detail_path.stat().st_size,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "Python": sys.version,
            "NumPy": np.__version__,
            "command": sys.argv,
            "cwd": str(Path.cwd()),
            "elapsed_seconds": time.monotonic() - started,
            "peak_RSS_bytes": peak_rss_bytes(),
            "JAX_imported": "jax" in sys.modules,
            "jaxlib_imported": "jaxlib" in sys.modules,
            "model_imported": any(name.startswith("jax_orchidee") for name in sys.modules),
        },
        "model_execution": False,
        "JIT_execution": False,
        "GPU_initialization": False,
        "scientific_acceptance": False,
        "claim_boundary": "saved new-science initial arrays versus frozen old-science numerical reference only; no cause attribution, tolerance change, retry or performance acceptance",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "failing_leaf_count",
                    "failing_lane_count",
                    "failing_unique_point_count",
                    "field_mapping_complete",
                )
            }
        ),
        flush=True,
    )


def main():
    args, started = parser().parse_args(), time.monotonic()
    if args.self_check:
        actual, reference = np.zeros((2, 1, 14)), np.zeros((2, 1, 14))
        actual[..., 0] = reference[..., 0] = np.nan
        actual[1, 0, 1] = 1e-6
        failed, sentinel, _ = comparison_mask(actual, reference, {"inner_shape": [1, 14], "bare_soil_index": 0})
        assert np.argwhere(failed).tolist() == [[1, 0, 1]] and int(sentinel.sum()) == 2
        discrete, _, policy = comparison_mask(np.array([True, False]), np.array([True, True]))
        assert discrete.tolist() == [False, True] and policy == "discrete_exact"
        boundary, _, _ = comparison_mask(np.array([ATOL, np.nextafter(ATOL, np.inf)]), np.zeros(2))
        assert boundary.tolist() == [False, True]
        scalar_values = np.array([0.0, 1e-6])
        scalar_mask, _, _ = comparison_mask(scalar_values, np.zeros(2))
        coords = np.argwhere(scalar_mask[1])
        selected = np.asarray(scalar_values[1][tuple(coords.T)]).reshape(-1)
        assert coords.shape == (1, 0) and len(list(zip(coords, selected, strict=True))) == 1
        nonfinite, _, _ = comparison_mask(np.array([np.inf, np.nan]), np.array([np.inf, 0.0]))
        assert nonfinite.tolist() == [True, True]
        assert (
            "jax" not in sys.modules
            and "jaxlib" not in sys.modules
            and not any(name.startswith("jax_orchidee") for name in sys.modules)
        )
        print(
            json.dumps(
                {
                    "status": "passed_small_NumPy_comparator_sanity",
                    "checks": [
                        "declared14PFTNaN",
                        "discreteexact",
                        "toleranceequality",
                        "nextafterviolation",
                        "scalarcoordinate",
                        "unexpectednonfinite",
                    ],
                    "JAX_imported": False,
                    "jaxlib_imported": False,
                    "model_imported": False,
                }
            )
        )
        return 0
    run(args, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
