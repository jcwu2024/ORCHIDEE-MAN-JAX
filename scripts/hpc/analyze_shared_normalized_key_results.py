"""Host-side strict leaf diagnosis of saved component/recurrent results."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np


def leaves(value, path=()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key], path + (str(key),))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from leaves(child, path + (str(index),))
    elif value is not None:
        yield "/".join(path), np.asarray(value)


def compare(actual, reference, *, max_examples=5):
    result = []
    for (path, a), (other, b) in zip(leaves(actual), leaves(reference), strict=True):
        if path != other or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("saved result leaf structure/shape/dtype drift")
        failed = ~np.isclose(a, b, rtol=1e-5, atol=1e-6)
        if np.any(failed):
            delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
            indices = np.flatnonzero(failed)
            worst = indices[np.argsort(delta.reshape(-1)[indices])[-max_examples:][::-1]]
            result.append({"path": path, "shape": a.shape, "failed": int(np.count_nonzero(failed)),
                           "max_abs": float(np.max(delta)),
                           "examples": [{"flat_index": int(i), "candidate": float(a.reshape(-1)[i]),
                                         "reference": float(b.reshape(-1)[i]), "abs_error": float(delta.reshape(-1)[i]),
                                         "allowed_error": float(1e-6 + 1e-5 * abs(b.reshape(-1)[i]))} for i in worst]})
    return sorted(result, key=lambda row: -row["failed"])


def annotate_state(rows, state_leaves):
    for row in rows:
        if tuple(row["shape"]) != (3854,):
            raise ValueError("state annotation requires declared complete3854-vector")
        for example in row["examples"]:
            index = example["flat_index"]
            matches = [leaf for leaf in state_leaves if leaf.start is not None and leaf.start <= index < leaf.stop]
            if len(matches) != 1:
                raise ValueError("state coordinate does not map to one declared leaf")
            leaf = matches[0]
            coordinate = tuple(int(i) for i in np.unravel_index(index - leaf.start, leaf.shape))
            example.update(state_leaf_key=leaf.key, state_leaf_coordinate=coordinate,
                           state_leaf_shape=leaf.shape, state_leaf_source_ref=leaf.source_ref)
    return rows


def recurrent(args):
    report = json.loads(args.recurrent_report.read_text())
    if args.saved_reference is None:
        raise ValueError("recurrent diagnosis requires --saved-reference")
    with args.saved_reference.open("rb") as stream:
        saved = pickle.load(stream)
    replay_path = args.recurrent_report.with_suffix(".reference-replay.pkl")
    with replay_path.open("rb") as stream:
        replay = pickle.load(stream)
    base_identity = report["baseline_identity"]
    for artifact in (saved, replay):
        if any(artifact["identity"].get(key) != value for key, value in base_identity.items()):
            raise ValueError("saved recurrent baseline identity drift")
    output = {"recurrent_report": str(args.recurrent_report), "status": report["status"],
              "rtol": 1e-5, "atol": 1e-6, "new_model_gradient_compilations": 0,
              "saved_reference": str(args.saved_reference), "reference_replay": str(replay_path),
              "baseline_state_cross_process_drift": compare(replay["result"][1][1], saved["result"][1][1], max_examples=3854)}
    state_leaves = None
    if args.prepared_context is not None:
        with args.prepared_context.open("rb") as stream:
            state_leaves = pickle.load(stream)["head"].contract.state_leaves
        output["prepared_context"] = str(args.prepared_context)
        annotate_state(output["baseline_state_cross_process_drift"], state_leaves)
    candidate_path = args.recurrent_report.with_suffix(".candidate.pkl")
    if candidate_path.is_file():
        with candidate_path.open("rb") as stream:
            candidate = pickle.load(stream)
        if candidate["identity"] != report["identity"]:
            raise ValueError("saved recurrent candidate identity drift")
        output["candidate_against_same_process_baseline"] = {
            "loss_and_auxiliary": compare(candidate["result"][0], replay["result"][0]),
            "initial_state_gradient": compare(candidate["result"][1][1], replay["result"][1][1], max_examples=3854)}
        if state_leaves is not None:
            annotate_state(output["candidate_against_same_process_baseline"]["initial_state_gradient"], state_leaves)
        if not report.get("comparisons", {}).get("all_parameter_gradients", {}).get("passed", False):
            output["candidate_against_same_process_baseline"]["parameters"] = compare(
                candidate["result"][1][0], replay["result"][1][0])
        output["candidate_artifact"] = str(candidate_path)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--screen-report", type=Path)
    modes.add_argument("--recurrent-report", type=Path)
    parser.add_argument("--saved-reference", type=Path)
    parser.add_argument("--prepared-context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.recurrent_report is not None:
        recurrent(args)
        return
    screen = json.loads(args.screen_report.read_text())
    output = {"screen": str(args.screen_report), "status": screen["status"], "routes": {}}
    for route, record in screen["results"].items():
        variants = record["variants"]
        with Path(variants["reference"]["result_artifact"]).open("rb") as stream:
            reference = pickle.load(stream)
        with Path(variants["normalized"]["result_artifact"]).open("rb") as stream:
            candidate = pickle.load(stream)
        if reference["identity"] != candidate["identity"]:
            raise ValueError("saved results identity drift")
        output["routes"][route] = {"output": compare(candidate["result"][0], reference["result"][0]),
                                    "parameters": compare(candidate["result"][1][0], reference["result"][1][0]),
                                    "inputs": compare(candidate["result"][1][1], reference["result"][1][1])}
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
