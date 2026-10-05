"""Export and admit exact saved NEW60 numerical inputs across host runtimes.

No preparation formulas, Day1 generation or physical transition runs here.
BlockArguments and every nested class retain their original module identity.
Only recorded VEGETATION_FILE path metadata may change on the remote host.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev import benchmark_global_scale as scale
from scripts.dev import prepare_global_scale_inputs as producer

batch, np, jax = producer.batch, producer.np, producer.jax
BLOCK_KEYS = ("first_block_arguments_sha256", "second_block_arguments_sha256")


def host_numeric_tree(value):
    """Preserve nested classes and insertion order, with no dtype conversions."""
    if value is None or isinstance(value, (str, Path)):
        return value
    if isinstance(value, dict):
        return type(value)((key, host_numeric_tree(item)) for key, item in value.items())
    if isinstance(value, list):
        return [host_numeric_tree(item) for item in value]
    if isinstance(value, tuple):
        items = tuple(host_numeric_tree(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: host_numeric_tree(getattr(value, field.name)) for field in fields(value)})
    return np.asarray(jax.device_get(value))


def state_layout(state):
    spec = batch.fast_state_from_previous_packet(state).spec
    return (spec.components, spec.field_names_by_component)


def mapped_static_signature(signature, local_vegetation, remote_vegetation):
    """Map only the exact audited path; retain all other static values."""
    values = dict(signature)
    run_def = dict(values["run_def"])
    if run_def.get("VEGETATION_FILE") != ("str", local_vegetation):
        raise ValueError("frozen static signature vegetation path differs from audited mapping")
    run_def["VEGETATION_FILE"] = ("str", remote_vegetation)
    values["run_def"] = tuple((key, run_def[key]) for key, _ in values["run_def"])
    return tuple((key, values[key]) for key, _ in signature)


def validate_arguments(point, arguments, record, state, layout, templates=None):
    """Require original numeric digests, full schema and cold packet boundary."""
    if state.tstep != 47:
        raise ValueError(f"frozen arguments require the original cold Day1 boundary: {point}")
    if not isinstance(arguments, tuple) or len(arguments) != 2:
        raise ValueError(f"frozen point requires two complete blocks: {point}")
    if layout != state_layout(state):
        raise ValueError(f"frozen cold state layout differs: {point}")
    initial = batch.fast_state_from_previous_packet(state).values_by_component
    for block, (key, value) in enumerate(zip(BLOCK_KEYS, arguments, strict=True)):
        if type(value) is not batch.BlockArguments:
            raise ValueError(f"frozen BlockArguments class identity differs: {point}")
        if templates is not None and batch._schema(value) != batch._schema(templates[block]):
            raise ValueError(f"frozen numerical schema differs from Linux source template: {point}/{block}")
        for leaf in jax.tree_util.tree_leaves(value):
            if type(leaf) is not np.ndarray:
                raise ValueError(f"frozen numerical leaf must be host NumPy: {point}/{block}")
        if producer._tree_digest(value) != record[key]:
            raise ValueError(f"frozen numerical prepared inputs differ from original audit: {point}/{key}")
        if (batch._schema(value.values_by_component) != batch._schema(initial)
                or producer._tree_digest(value.values_by_component) != producer._tree_digest(initial)):
            raise ValueError(f"frozen argument cold state boundary differs: {point}/{block}")
        days = np.arange(2 + block * 7, 9 + block * 7, dtype=np.int32)
        if value.science_day_numbers.dtype != days.dtype or not np.array_equal(value.science_day_numbers, days):
            raise ValueError(f"frozen seven-day boundaries differ: {point}/{block}")
        for leaf in jax.tree_util.tree_leaves(value.forcing_days):
            if leaf.shape[:2] != (7, 48):
                raise ValueError(f"frozen forcing requires seven days by 48 steps: {point}/{block}")
    return arguments


def load_frozen(path, expected_sha256, source_report, source_audit_sha256):
    """Verify the one consolidated transport before accepting its point set."""
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("frozen arguments transport checksum changed")
    payload = scale.PreparedUnpickler(io.BytesIO(raw)).load()
    ids = source_report["selected_points"]
    if (payload.get("schema_version") != 1 or payload.get("selected_points") != ids
            or len(ids) != 64 or len(set(ids)) != 64 or tuple(ids[:4]) != producer.ANCHORS):
        raise ValueError("frozen argument selection/schema differs from exact FULL64 contract")
    if (payload.get("source_audit_sha256") != source_audit_sha256
            or payload.get("config_sha256") != source_report["config_sha256"]
            or payload.get("source_prepared_sha256") != source_report["prepared_pickle_sha256"]):
        raise ValueError("frozen arguments source audit/config/prepared provenance changed")
    points = payload.get("points", [])
    if [item.get("landpoint_id") for item in points] != ids[4:]:
        raise ValueError("frozen arguments ordered point set differs from selected NEW60")
    return {item["landpoint_id"]: item for item in points}, actual_sha256


def bind_frozen(lane, second, frozen, record, remote_vegetation):
    """Bind exact saved values into independently source-rebuilt Linux metadata."""
    point = lane.landpoint_id
    if frozen["landpoint_id"] != point:
        raise ValueError(f"frozen argument wrong point: {point}")
    signature = frozen["static_signature"]
    if hashlib.sha256(repr(signature).encode()).hexdigest() != record["static_signature_sha256"]:
        raise ValueError(f"frozen original static signature changed: {point}")
    mapped = mapped_static_signature(signature,
        record["vegetation_file_path_only_mapping"]["local_value"], remote_vegetation)
    if mapped != lane.static_signature:
        raise ValueError(f"frozen static closure differs from rebuilt Linux context: {point}")
    arguments = validate_arguments(point, frozen["arguments"], record, lane.initial_state,
                                   frozen["state_layout"], (lane.arguments, second))
    return replace(lane, arguments=arguments[0]), arguments[1]


def export(args):
    started = time.perf_counter()
    if args.output.exists() or args.report.exists():
        raise ValueError("frozen export destinations must be new; preserve existing evidence")
    source_report = json.loads(args.preparation_audit.read_text(encoding="utf-8"))
    if source_report.get("status") != "prepared" or source_report.get("prepared_point_count") != 64:
        raise ValueError("frozen export requires the completed original FULL64 audit")
    if hashlib.sha256(args.config.read_bytes()).hexdigest() != source_report["config_sha256"]:
        raise ValueError("frozen export configuration differs from original audit")
    lanes, second = scale.read_prepared(args.prepared_input, args.config)
    ids = [lane.landpoint_id for lane in lanes]
    if ids != source_report["selected_points"]:
        raise ValueError("saved local prepared point order differs from original audit")
    screen = scale.screen_inputs(lanes, second)
    audit_sha256 = producer._sha256_file(args.preparation_audit)
    records = {item["landpoint_id"]: item for item in source_report["new_point_records"]}
    if len(source_report["new_point_records"]) != 60 or set(records) != set(ids[4:]):
        raise ValueError("original audit does not cover exactly NEW60")
    points, evidence = [], []
    for lane, other in zip(lanes[4:], second[4:], strict=True):
        point = lane.landpoint_id
        record = records[point]
        if producer._state_digest(lane.initial_state) != record["cold_state_numeric_sha256"]:
            raise ValueError(f"saved cold state differs from original audit: {point}")
        if hashlib.sha256(repr(lane.static_signature).encode()).hexdigest() != record["static_signature_sha256"]:
            raise ValueError(f"saved static closure differs from original audit: {point}")
        host = host_numeric_tree((lane.arguments, other))
        layout = state_layout(lane.initial_state)
        validate_arguments(point, host, record, lane.initial_state, layout, (lane.arguments, other))
        points.append({"landpoint_id": point, "arguments": host,
                       "static_signature": lane.static_signature, "state_layout": layout})
        evidence.append({"landpoint_id": point, **{key: record[key] for key in BLOCK_KEYS},
                         "numeric_identity_unchanged": True, "cold_state_boundary_verified": True})
    payload = {"schema_version": 1, "selected_points": ids, "source_audit_sha256": audit_sha256,
               "config_sha256": source_report["config_sha256"],
               "source_prepared_sha256": source_report["prepared_pickle_sha256"], "points": points}
    producer._atomic_pickle(args.output, payload)
    payload_sha256 = producer._sha256_file(args.output)
    report = {"status": "exact_saved_arguments_exported", "payload": str(args.output),
              "payload_sha256": payload_sha256, "payload_bytes": args.output.stat().st_size,
              "payload_file_count": 1, "new_point_count": 60, "block_count": 2,
              "source_audit_sha256": audit_sha256, "source_prepared_sha256": source_report["prepared_pickle_sha256"],
              "point_records": evidence, "original_full64_screen": screen,
              "elapsed_seconds": time.perf_counter() - started,
              "scope": "exact saved host NumPy IO; no casts, tolerance waiver, Day1 generation or physical forward"}
    producer._atomic_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("status", "payload", "payload_sha256", "payload_bytes", "elapsed_seconds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-input", "preparation-audit", "config", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
