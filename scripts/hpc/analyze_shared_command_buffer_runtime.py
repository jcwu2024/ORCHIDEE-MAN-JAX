"""NumPy-only interval evidence from preserved full parameter-VJP repeats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np


def leaves(tree, path=()):
    if isinstance(tree, dict):
        for key in sorted(tree):
            yield from leaves(tree[key], path + (str(key),))
    elif isinstance(tree, (tuple, list)):
        for index, value in enumerate(tree):
            yield from leaves(value, path + (str(index),))
    elif tree is not None:
        yield "/".join(path), np.asarray(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=("while", "vmap", "preserve128"), default="while")
    args = parser.parse_args()
    source = json.loads(args.replay_report.read_text())
    expected = "runtime_component_paired_measured" if args.candidate == "while" else "chunk_vmap_component_measured"
    if source.get("status") != expected:
        raise ValueError("requires completed same-runtime paired component evidence")
    repeated = {}
    for name in ("default", args.candidate):
        repeated[name] = []
        for repeat in range(7):
            path = args.replay_report.with_suffix(f".{name}.repeat{repeat}.parameter_vjp.pkl")
            with path.open("rb") as stream:
                repeated[name].append(dict(leaves(pickle.load(stream))))
    paths = list(repeated["default"][0])
    if any(list(tree) != paths for repeats in repeated.values() for tree in repeats):
        raise ValueError("repeated parameter trees differ")
    rows = []
    for path in paths:
        a, b = (np.stack([row[path] for row in repeated[name]]).astype(np.float64) for name in ("default", args.candidate))
        if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("repeat shapes differ or nonfinite gradients")
        allowed = 1e-6 + 1e-5 * np.abs(a)
        failed_every_pair = np.all(np.abs(b - a) > allowed, axis=0)
        low_a, high_a, low_b, high_b = a.min(0), a.max(0), b.min(0), b.max(0)
        gap = np.maximum(np.maximum(low_b - high_a, low_a - high_b), 0)
        # Compare disjoint repeat ranges against the largest strict allowance
        # among observed reference values, without changing its tolerance.
        max_allowed = 1e-6 + 1e-5 * np.max(np.abs(a), axis=0)
        persistent_gap = failed_every_pair & (gap > max_allowed)
        indices = np.argwhere(persistent_gap)
        examples = []
        for index_array in indices[:5]:
            index = tuple(int(value) for value in index_array)
            examples.append({"index": index, "default_range": [low_a[index], high_a[index]],
                             f"{args.candidate}_range": [low_b[index], high_b[index]], "gap": gap[index],
                             "largest_strict_allowance": max_allowed[index],
                             "default_width": high_a[index] - low_a[index], f"{args.candidate}_width": high_b[index] - low_b[index]})
        rows.append({"path": path, "shape": a.shape[1:], "coordinates": int(a[0].size),
                     "failed_every_pair": int(np.count_nonzero(failed_every_pair)),
                     "disjoint_ranges_beyond_largest_strict_allowance": int(np.count_nonzero(persistent_gap)),
                     "default_max_repeat_width": float(np.max(high_a-low_a, initial=0)),
                     f"{args.candidate}_max_repeat_width": float(np.max(high_b-low_b, initial=0)),
                     "max_range_gap": float(np.max(gap, initial=0)), "examples": examples})
    report = {"schema": "shared_command_buffer_parameter_repeat_intervals_v1", "status": "analyzed",
              "candidate": args.candidate,
              "source_replay_report": str(args.replay_report), "scope": "all_decoder_parameter_vjp_7_same_runtime_pairs",
              "rtol": 1e-5, "atol": 1e-6, "parameter_coordinates": sum(row["coordinates"] for row in rows),
              "failed_every_pair": sum(row["failed_every_pair"] for row in rows),
              "disjoint_ranges_beyond_largest_strict_allowance": sum(row["disjoint_ranges_beyond_largest_strict_allowance"] for row in rows),
              "leaves": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key != "leaves"}))
    print(json.dumps(sorted([{key:value for key,value in row.items() if key not in ("examples",)} for row in rows], key=lambda row: row["failed_every_pair"], reverse=True)[:8]))


if __name__ == "__main__":
    main()
