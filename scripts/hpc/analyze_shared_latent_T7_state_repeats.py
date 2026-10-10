"""CPU-only saved state-gradient extraction and descriptive repeat analysis.

No JAX, model evaluation, precision change or gate replacement is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

RTOL, ATOL = 1e-5, 1e-6
EXPECTED_SHAPE = (4, 3854)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def state_metrics(actual, reference):
    """Complete strict counts plus scale-safe descriptive L2/cosine."""
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
        raise ValueError("identical finite state-array shapes required")
    failed = ~np.isclose(a, r, rtol=RTOL, atol=ATOL)
    scale = float(max(np.max(np.abs(a)), np.max(np.abs(r))))
    if scale == 0:
        relative, cosine = 0.0, 1.0
    else:
        aa, rr = a.reshape(-1) / scale, r.reshape(-1) / scale
        an, rn = float(np.linalg.norm(aa)), float(np.linalg.norm(rr))
        relative = float(np.linalg.norm(aa - rr) / rn) if rn else None
        cosine = float(np.dot(aa / an, rr / rn)) if an and rn else None
    return {"coordinates": int(a.size), "strict_failed": int(np.count_nonzero(failed)),
            "failed_positions": np.argwhere(failed).tolist(), "relative_L2_to_reference": relative,
            "cosine": cosine, "descriptive_metrics_do_not_replace_strict_gate": True}


def scalar_statistics(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("nonempty finite scalar observations required")
    scale = float(np.max(np.abs(x)))
    std = float(np.std(x / scale) * scale) if scale else 0.0
    return {"count": int(x.size), "values": x.tolist(), "minimum": float(np.min(x)),
            "maximum": float(np.max(x)), "mean": float(np.mean(x)),
            "population_std": std, "range_width": float(np.ptp(x))}


def compare_ranges(actual_values, reference_values):
    a, r = scalar_statistics(actual_values), scalar_statistics(reference_values)
    lower, upper = max(a["minimum"], r["minimum"]), min(a["maximum"], r["maximum"])
    direction = "overlap" if lower <= upper else "actual_above" if a["minimum"] > r["maximum"] else "actual_below"
    gap = 0.0 if direction == "overlap" else lower - upper
    return {"actual": a, "reference": r, "observed_ranges_overlap": lower <= upper,
            "direction": direction, "observed_range_gap": gap,
            "range_gap_over_reference_self_width": gap / r["range_width"] if r["range_width"] else None}


def iter_bindings(report):
    comparisons = [row.get("comparison", {}) for row in report.get("numerical_failures", [])]
    comparisons += [row.get("comparison", {}) for row in report.get("repeat_failures", [])]
    for comparison in comparisons:
        yield from comparison.get("state_failure_coordinates", [])


def extract(reports, output_prefix):
    """Read full pickles only on CPU; write just small native state arrays."""
    arrays, metadata, bindings = {}, [], {}
    began = time.monotonic()
    for session, report_path in reports:
        report = read_json(report_path)
        for row in iter_bindings(report):
            bindings[f'{row["sample"]}:{row["coordinate"]}'] = row.get("bindings", [])
        for row in report["results"]:
            path = Path(row["artifact"])
            if not str(path.resolve()).startswith("/WORK/liwei_work/jcwu/"):
                raise ValueError("saved scientific artifact must remain in jcwu tree")
            with path.open("rb") as handle:
                saved = pickle.load(handle)
            states = np.asarray(saved["result"][1][1])
            if states.shape != EXPECTED_SHAPE or not np.all(np.isfinite(states)):
                raise ValueError("native full4x3854 finite state AD required")
            key = f"state{len(metadata):02d}"
            arrays[key] = states.copy()
            mode = row.get("implementation", row.get("mode"))
            metadata.append({"key": key, "session": session, "mode": mode, "label": row["label"],
                "artifact": str(path), "artifact_size_bytes": path.stat().st_size,
                "state_dtype": str(states.dtype), "state_shape": list(states.shape),
                "state_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
                "saved_loss": float(saved["result"][0][0]),
                "report_status": report["status"], "identity_case_sha256": report["mathematical_identity"]["case_sha256"],
                "resource_node": report.get("resources", {}).get("node", report.get("resources", {}).get("hostname"))})
            del states, saved
    output_prefix = Path(output_prefix)
    if output_prefix.with_suffix(".npz").exists() or output_prefix.with_suffix(".metadata.json").exists():
        raise FileExistsError("preserve existing extracted state evidence")
    np.savez_compressed(output_prefix.with_suffix(".npz"), **arrays)
    record = {"schema_version": "latent_T7_saved_state_extract_v1", "records": metadata,
        "reported_coordinate_bindings": bindings, "extraction_seconds": time.monotonic() - began,
        "model_calls": 0, "GPU_calls": 0, "new_model_compiles": 0,
        "native_states_already_scaled_by_sample_mean": True,
        "report_sources": [{"session": session, "path": str(path)} for session, path in reports]}
    output_prefix.with_suffix(".metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def analyze(input_prefix, output):
    prefix = Path(input_prefix)
    metadata = read_json(prefix.with_suffix(".metadata.json"))
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    records = metadata["records"]
    groups = {}
    for row in records:
        groups.setdefault((row["session"], row["mode"]), []).append(row)
    affected, comparisons = set(), []

    def compare(kind, actual, reference):
        result = state_metrics(arrays[actual["key"]], arrays[reference["key"]])
        affected.update(tuple(position) for position in result["failed_positions"])
        comparisons.append({"kind": kind, "session": actual["session"],
            "actual_mode": actual["mode"], "actual_label": actual["label"],
            "reference_mode": reference["mode"], "reference_label": reference["label"], **result})

    # Standard saved report comparisons: each mode against its own cold/warmup.
    for rows in groups.values():
        reference = next(row for row in rows if row["label"] in ("cold-first", "warmup"))
        for actual in rows:
            if actual is not reference:
                compare("self_vs_saved_first", actual, reference)
    for session, a_mode, r_mode in (("paid_B4", "batch", "serial"), ("dispatch_B1", "dispatch_four", "block_each")):
        a_rows, r_rows = groups[(session, a_mode)], groups[(session, r_mode)]
        for index in range(1, 4):
            label = f"pair{index}"
            compare("same_session_paired_cross", next(row for row in a_rows if row["label"] == label),
                    next(row for row in r_rows if row["label"] == label))
    # Include every observed repeat-pair direction as a variability envelope;
    # these exploratory comparisons are distinct from saved report gates.
    envelope_failed = set()
    for rows in groups.values():
        for i, actual in enumerate(rows):
            for reference in rows[:i]:
                for aa, rr in ((actual, reference), (reference, actual)):
                    envelope_failed.update(tuple(position) for position in state_metrics(arrays[aa["key"]], arrays[rr["key"]])["failed_positions"])
    affected |= envelope_failed
    affected |= {(3, 434), (3, 2382), (3, 2621)}
    coordinates = []
    for sample, coordinate in sorted(affected):
        group_stats = {f"{session}/{mode}": {"labels": [row["label"] for row in rows],
            **scalar_statistics([arrays[row["key"]][sample, coordinate] for row in rows])}
            for (session, mode), rows in groups.items()}
        paid_a, paid_r = groups[("paid_B4", "batch")], groups[("paid_B4", "serial")]
        paired_a = [next(row for row in paid_a if row["label"] == f"pair{i}") for i in range(1, 4)]
        paired_r = [next(row for row in paid_r if row["label"] == f"pair{i}") for i in range(1, 4)]
        all_range = compare_ranges([arrays[row["key"]][sample, coordinate] for row in paid_a],
                                   [arrays[row["key"]][sample, coordinate] for row in paid_r])
        pair_range = compare_ranges([arrays[row["key"]][sample, coordinate] for row in paired_a],
                                    [arrays[row["key"]][sample, coordinate] for row in paired_r])
        directions = []
        for aa, rr in zip(paired_a, paired_r):
            a, r = float(arrays[aa["key"]][sample, coordinate]), float(arrays[rr["key"]][sample, coordinate])
            allowance = ATOL + RTOL * abs(r)
            delta = a - r
            width = all_range["reference"]["range_width"]
            directions.append({"pair": int(aa["label"][-1]), "B4": a, "B1": r,
                "signed_B4_minus_B1": delta, "absolute_difference": abs(delta), "strict_allowance": allowance,
                "strict_failed": bool(abs(delta) > allowance), "absolute_difference_over_allowance": abs(delta) / allowance,
                "absolute_difference_exceeds_observed_B1_self_range": bool(abs(delta) > width),
                "absolute_difference_over_B1_self_range": abs(delta) / width if width else None})
        coordinates.append({"sample": sample, "coordinate": coordinate,
            "bindings": metadata["reported_coordinate_bindings"].get(f"{sample}:{coordinate}", []),
            "groups": group_stats, "same_session_paid_pairs_only": pair_range,
            "same_session_paid_cold_and_pairs": all_range, "paired_signed_differences": directions,
            "B4_paired_values_vs_cross_session_dispatch_B1": compare_ranges(
                [arrays[row["key"]][sample, coordinate] for row in paired_a],
                [arrays[row["key"]][sample, coordinate] for mode in ("block_each", "dispatch_four")
                 for row in groups[("dispatch_B1", mode)]])})
    result = {"schema_version": "latent_T7_state_repeat_analysis_v1", "rtol": RTOL, "atol": ATOL,
        "all_states_shape": list(EXPECTED_SHAPE), "record_count": len(records),
        "sessions_kept_separate": True, "GPU_calls": 0, "model_calls": 0,
        "affected_coordinate_count": len(coordinates), "coordinates": coordinates,
        "full_state_comparisons": comparisons,
        "interpretation_limit": "Observed finite repeat ranges, not population bounds, cause, harmlessness, or revised acceptance gates.",
        "source_metadata": str(prefix.with_suffix(".metadata.json"))}
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--report", action="append", nargs=2, metavar=("SESSION", "PATH"), required=True)
    e.add_argument("--output-prefix", type=Path, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--input-prefix", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.stage == "extract":
        result = extract(args.report, args.output_prefix)
        print(json.dumps({"records": len(result["records"]), "seconds": result["extraction_seconds"], "GPU_calls": 0}))
    else:
        result = analyze(args.input_prefix, args.output)
        print(json.dumps({"affected_coordinates": result["affected_coordinate_count"], "records": result["record_count"], "GPU_calls": 0}))


if __name__ == "__main__":
    main()
