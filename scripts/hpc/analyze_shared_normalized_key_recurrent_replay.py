"""Raw saved-state repeat ranges and strict coordinates; host NumPy only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(replay):
    states = {name: np.asarray([row["arms"][name]["state_gradient"] for row in replay["repeats"]], dtype=np.float64)
              for name in ("default", "normalized")}
    if any(value.shape != (7, 3854) or not np.all(np.isfinite(value)) for value in states.values()):
        raise ValueError("requires seven finite complete state vectors per arm")
    paired = ~np.isclose(states["normalized"], states["default"], rtol=1e-5, atol=1e-6)
    self_failed = {name: ~np.isclose(value, value[0], rtol=1e-5, atol=1e-6) for name, value in states.items()}
    indices = np.flatnonzero(np.any(paired, axis=0))
    labels = {}
    for row in replay["repeats"]:
        for leaf in row["paired_normalized_against_default"]:
            for example in leaf["examples"]:
                labels[example["flat_index"]] = {key: value for key, value in example.items() if key.startswith("state_leaf_")}
    rows = []
    for index in indices:
        baseline, candidate = states["default"][:, index], states["normalized"][:, index]
        records = {name: {"min": float(value.min()), "max": float(value.max()), "range": float(np.ptp(value)),
                          "self_failure_repeats": np.flatnonzero(self_failed[name][:, index]).tolist()}
                   for name, value in (("default", baseline), ("normalized", candidate))}
        gap = max(0.0, float(candidate.min() - baseline.max()), float(baseline.min() - candidate.max()))
        rows.append({"flat_index": int(index), **labels.get(int(index), {}), "arms": records,
                     "paired_failure_repeats": np.flatnonzero(paired[:, index]).tolist(),
                     "paired_abs_errors": np.abs(candidate - baseline).tolist(),
                     "paired_allowed_errors": (1e-6 + 1e-5 * np.abs(baseline)).tolist(),
                     "nonoverlapping_observed_range_gap": gap})
    return {"scope": "bounded_seven_saved_binary_state_repeats", "rtol": 1e-5, "atol": 1e-6,
            "new_model_gradient_compilations": 0, "paired_failed_counts": np.sum(paired, axis=1).tolist(),
            "within_arm_failed_counts": {name: np.sum(value, axis=1).tolist() for name, value in self_failed.items()},
            "persistent_cross_arm_indices": np.flatnonzero(np.all(paired, axis=0)).tolist(), "coordinates": rows,
            "interpretation_limit": "Observed ranges are bounded repeat evidence, not a universal floating-point error bound."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = analyze(json.loads(args.replay_report.read_text()))
    report["replay_report"] = str(args.replay_report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
