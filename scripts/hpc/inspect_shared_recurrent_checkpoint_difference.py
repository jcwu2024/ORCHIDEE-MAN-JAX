"""Inspect saved selective-checkpoint initial-state gradient discrepancies."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def inspect(reference_path: Path, candidate_path: Path) -> dict:
    with reference_path.open("rb") as stream:
        reference = pickle.load(stream)
    with candidate_path.open("rb") as stream:
        candidate = pickle.load(stream)
    if reference["identity"] != candidate["identity"]:
        raise ValueError("saved input/model identities differ")
    original = np.asarray(reference["result"][1][1], dtype=np.float64)
    changed = np.asarray(candidate["result"][1][1], dtype=np.float64)
    if original.shape != changed.shape:
        raise ValueError("initial-state gradient shapes differ")
    delta = changed - original
    threshold = 1e-6 + 1e-5 * np.abs(original)
    failed = np.flatnonzero(~np.isfinite(delta) | (np.abs(delta) > threshold))
    return {
        "artifact_id": reference["identity"]["artifact_id"],
        "horizon": reference["identity"]["horizon"],
        "shape": list(original.shape),
        "failed_count": int(failed.size),
        "reference_max_abs": float(np.max(np.abs(original))),
        "candidate_max_abs": float(np.max(np.abs(changed))),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(original)),
        "failed": [
            {"index": int(index), "reference": float(original.flat[index]),
             "candidate": float(changed.flat[index]), "difference": float(delta.flat[index]),
             "allowed": float(threshold.flat[index]),
             "difference_over_allowed": float(np.abs(delta.flat[index]) / threshold.flat[index])}
            for index in failed
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
