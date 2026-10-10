"""Exact state-coordinate evidence from saved initial and seven paired VJPs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    daily = source.get("scope") == "original_four_daily_micro4_public_loss_full_parameter_and_initial_state_AD"
    if (not daily and source.get("horizon") != 7) or source.get("rtol") != 1e-5 or source.get("atol") != 1e-6:
        raise ValueError("requires unchanged strict T7 or complete daily-dual gate")
    original = np.asarray(load(args.report.with_suffix(".field-replay.pkl" if daily else ".reference-replay.pkl"))["result"][1][1])
    candidate = np.asarray(load(args.report.with_suffix(".candidate.pkl"))["result"][1][1])
    if daily:
        saved = load(args.report.with_suffix(".state-replays.pkl"))
        repeated = {"default": np.stack(saved["field"]), "candidate": np.stack(saved["candidate"])}
    else:
        repeated = {name: np.stack([load(args.report.with_suffix(f".{name}.repeat{r}.initial_state_vjp.pkl"))
                                   for r in range(7)]) for name in ("default", "candidate")}
    a, b = repeated["default"], repeated["candidate"]
    if (original.size != (15416 if daily else 3854) or candidate.shape != original.shape or a.shape != b.shape
            or a.shape[1:] != original.shape or not all(np.all(np.isfinite(v)) for v in (original,candidate,a,b))):
        raise ValueError("invalid complete finite state evidence")
    initial_failed = ~np.isclose(candidate, original, rtol=1e-5, atol=1e-6)
    paired_failed = ~np.isclose(b, a, rtol=1e-5, atol=1e-6)
    paired_counts = [int(np.count_nonzero(row)) for row in paired_failed]
    recorded = ([row["all_initial_state_gradients"] for row in source["full_derivative_pair_comparisons"]]
                if daily else source["initial_state_pair_replay"])
    if paired_counts != [row["failed"] for row in recorded]:
        raise ValueError("raw state pairs disagree with recorded gate")
    if int(np.count_nonzero(initial_failed)) != source["comparisons"]["all_initial_state_gradients" if daily else "initial_state_gradient"]["failed"]:
        raise ValueError("initial state evidence disagrees with recorded gate")
    self_failed = {name: ~np.isclose(values, initial, rtol=1e-5, atol=1e-6)
                   for name, values, initial in (("default",a,original),("candidate",b,candidate))}
    union = initial_failed | np.any(paired_failed,axis=0)
    low_a, high_a, low_b, high_b = a.min(0), a.max(0), b.min(0), b.max(0)
    gap = np.maximum(np.maximum(low_b-high_a, low_a-high_b),0)
    allowed = 1e-6 + 1e-5*np.max(np.abs(a),axis=0)
    persistent = np.all(paired_failed,axis=0) & (gap > allowed)
    rows = []
    for index_array in np.argwhere(union):
        index = tuple(int(v) for v in index_array)
        values_a, values_b = a[(slice(None),)+index], b[(slice(None),)+index]
        rows.append({"index":index, "initial_default":float(original[index]),
                     "initial_candidate":float(candidate[index]), "initial_failed":bool(initial_failed[index]),
                     "initial_absolute_difference":float(abs(candidate[index]-original[index])),
                     "initial_strict_allowance":float(1e-6+1e-5*abs(original[index])),
                     "default_repeats":values_a.tolist(), "candidate_repeats":values_b.tolist(),
                     "failed_pair_indices":np.flatnonzero(paired_failed[(slice(None),)+index]).tolist(),
                     "default_self_failed_indices":np.flatnonzero(self_failed["default"][(slice(None),)+index]).tolist(),
                     "candidate_self_failed_indices":np.flatnonzero(self_failed["candidate"][(slice(None),)+index]).tolist(),
                     "default_range":[float(low_a[index]),float(high_a[index])],
                     "candidate_range":[float(low_b[index]),float(high_b[index])],
                     "range_gap":float(gap[index]), "largest_strict_allowance":float(allowed[index]),
                     "disjoint_ranges_beyond_allowance":bool(persistent[index])})
    delta = np.abs(candidate-original)
    result = {"schema":"shared_chunk_batch_state_repeat_intervals_v1", "status":"analyzed",
              "source_report":str(args.report), "scope":("all15416_daily_initial_state_coordinates_and_seven_pairs" if daily else "all3854_initial_state_coordinates_and_seven_pairs"),
              "rtol":1e-5,"atol":1e-6,"self_replays_are_diagnostic":True,"adoption":False,
              "initial_failed":int(np.count_nonzero(initial_failed)), "paired_failure_counts":paired_counts,
              "initial_global_max_abs":float(np.max(delta)),
              "initial_failed_coordinates_max_abs":float(np.max(delta[initial_failed],initial=0)),
              "union_failed_coordinates":int(np.count_nonzero(union)),
              "failed_every_pair":int(np.count_nonzero(np.all(paired_failed,axis=0))),
              "disjoint_ranges_beyond_largest_strict_allowance":int(np.count_nonzero(persistent)),
              "self_failure_counts":{name:[int(np.count_nonzero(row)) for row in values] for name,values in self_failed.items()},
              "coordinates":rows}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="coordinates"}))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
