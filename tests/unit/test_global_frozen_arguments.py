"""Frozen input transport gates; no physical computation or source regeneration."""
from __future__ import annotations

import copy
import hashlib
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev import prepare_global_frozen_arguments as frozen  # noqa: E402


@pytest.fixture
def case(monkeypatch):
    state = SimpleNamespace(tstep=47, values=(np.array([1.0]),), layout=(("state",), (("value",),)))
    monkeypatch.setattr(frozen.batch, "fast_state_from_previous_packet", lambda state: SimpleNamespace(
        values_by_component=state.values, spec=SimpleNamespace(
            components=state.layout[0], field_names_by_component=state.layout[1])))
    args = tuple(frozen.batch.BlockArguments(
        state.values, {"swdown": np.ones((7, 48, 1))}, np.arange(start, start + 7, dtype=np.int32),
        {"param": np.array([1.0])}, {"table": np.array([2.0])},
        {"lat": np.ones((1, 1), dtype=np.float32)}, None, None, None) for start in (2, 9))
    signature = (("run_def", (("STATIC_SWITCH", ("str", "TRUE")),
                              ("VEGETATION_FILE", ("str", "local.nc")))),)
    record = {key: frozen.producer._tree_digest(value) for key, value in zip(frozen.BLOCK_KEYS, args, strict=True)}
    record.update(static_signature_sha256=hashlib.sha256(repr(signature).encode()).hexdigest(),
                  vegetation_file_path_only_mapping={"local_value": "local.nc"})
    lane = frozen.batch.PreparedLane("215.0-115.0", None, state, None, None, args[0],
        frozen.mapped_static_signature(signature, "local.nc", "/remote.nc"))
    item = {"landpoint_id": lane.landpoint_id, "arguments": args, "static_signature": signature,
            "state_layout": state.layout}
    return lane, args[1], item, record


def test_bind_uses_exact_original_values_with_changed_regenerated_low_bits(case):
    lane, other, item, record = case
    regenerated = lane.arguments._replace(hydrol_table_arrays={"table": np.nextafter(np.array([2.0]), np.inf)})
    lane = frozen.replace(lane, arguments=regenerated)
    result, second = frozen.bind_frozen(lane, other, item, record, "/remote.nc")
    assert result.context is lane.context and result.tables is lane.tables
    assert result.arguments is item["arguments"][0] and second is item["arguments"][1]
    assert result.arguments.hydrol_table_arrays["table"].tobytes() == np.array([2.0]).tobytes()


@pytest.mark.parametrize("change,match", [
    ("wrong_point", "wrong point"), ("wrong_signature", "original static signature"),
    ("static_closure", "static closure"), ("layout", "cold state layout"),
    ("value", "original audit"), ("shape", "numerical schema"), ("dtype", "numerical schema"),
    ("cold_state", "cold state boundary"), ("wrong_day", "seven-day boundaries"),
    ("wrong_steps", "seven days by 48 steps"),
])
def test_bind_rejects_tampered_inputs_even_with_self_consistent_transport(case, change, match):
    lane, other, original, record = case
    item = copy.deepcopy(original)
    args = list(item["arguments"])
    if change == "wrong_point":
        item["landpoint_id"] = "wrong"
    elif change == "wrong_signature":
        item["static_signature"] = ()
    elif change == "static_closure":
        lane = frozen.replace(lane, static_signature=())
    elif change == "layout":
        item["state_layout"] = ()
    elif change in {"value", "shape", "dtype"}:
        value = np.array([np.nextafter(1.0, np.inf)])
        if change == "shape":
            value = np.ones((2,))
        elif change == "dtype":
            value = np.ones((1,), dtype=np.float32)
        args[0] = args[0]._replace(stomate_parameters={"param": value})
    elif change == "cold_state":
        args[0] = args[0]._replace(values_by_component=(np.array([3.0]),))
        record = {**record, frozen.BLOCK_KEYS[0]: frozen.producer._tree_digest(args[0])}
    elif change == "wrong_day":
        args[0] = args[0]._replace(science_day_numbers=np.arange(3, 10, dtype=np.int32))
        record = {**record, frozen.BLOCK_KEYS[0]: frozen.producer._tree_digest(args[0])}
    elif change == "wrong_steps":
        args[0] = args[0]._replace(forcing_days={"swdown": np.ones((7, 47, 1))})
        lane = frozen.replace(lane, arguments=args[0])
        record = {**record, frozen.BLOCK_KEYS[0]: frozen.producer._tree_digest(args[0])}
    item["arguments"] = tuple(args)
    with pytest.raises(ValueError, match=match):
        frozen.bind_frozen(lane, other, item, record, "/remote.nc")


def test_host_export_preserves_class_native_dtype_shape_and_nested_insertion_order(case):
    args = case[2]["arguments"][0]._replace(stomate_parameters=OrderedDict((
        ("z", np.array([1.0])), ("a", np.array(2, dtype=np.int32)))))
    actual = frozen.host_numeric_tree(args)
    assert type(actual) is type(args)
    assert type(actual.stomate_parameters) is OrderedDict
    assert tuple(actual.stomate_parameters) == ("z", "a")
    assert actual.landpoint_payload["lat"].dtype == np.dtype("float32")
    assert frozen.batch._schema(actual) == frozen.batch._schema(args)
    assert frozen.producer._tree_digest(actual) == frozen.producer._tree_digest(args)


@pytest.mark.parametrize("change,match", [
    ("checksum", "transport checksum"), ("schema", "selection/schema"),
    ("point_order", "ordered point set"), ("duplicate", "ordered point set"),
    ("audit", "provenance changed"), ("config", "provenance changed"),
])
def test_load_rejects_transport_selection_and_provenance_tampering(tmp_path, change, match):
    ids = [*frozen.producer.ANCHORS, *(f"point-{i}" for i in range(60))]
    report = {"selected_points": ids, "config_sha256": "config", "prepared_pickle_sha256": "prepared"}
    payload = {"schema_version": 1, "selected_points": ids, "source_audit_sha256": "audit",
               "config_sha256": "config", "source_prepared_sha256": "prepared",
               "points": [{"landpoint_id": point} for point in ids[4:]]}
    if change == "schema":
        payload["schema_version"] = 2
    elif change == "point_order":
        payload["points"].reverse()
    elif change == "duplicate":
        payload["points"][-1] = payload["points"][0]
    elif change == "audit":
        payload["source_audit_sha256"] = "other"
    elif change == "config":
        payload["config_sha256"] = "other"
    path = tmp_path / "frozen.pkl"
    raw = pickle.dumps(payload)
    path.write_bytes(raw)
    sha = "wrong" if change == "checksum" else hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match=match):
        frozen.load_frozen(path, sha, report, "audit")


def test_only_exact_recorded_vegetation_metadata_may_be_remapped(case):
    with pytest.raises(ValueError, match="audited mapping"):
        frozen.mapped_static_signature(case[2]["static_signature"], "unrecorded.nc", "/remote.nc")


def test_shifted_packet_is_rejected_as_frozen_cold_source(case):
    lane, other, item, record = case
    lane.initial_state.tstep = 383
    with pytest.raises(ValueError, match="original cold Day1 boundary"):
        frozen.bind_frozen(lane, other, item, record, "/remote.nc")


def test_load_preserves_original_classes_and_all_ordered_points(tmp_path, case):
    ids = [*frozen.producer.ANCHORS, *(f"point-{i}" for i in range(60))]
    report = {"selected_points": ids, "config_sha256": "config", "prepared_pickle_sha256": "prepared"}
    payload = {"schema_version": 1, "selected_points": ids, "source_audit_sha256": "audit",
               "config_sha256": "config", "source_prepared_sha256": "prepared",
               "points": [{**case[2], "landpoint_id": point} for point in ids[4:]]}
    path = tmp_path / "frozen.pkl"
    raw = pickle.dumps(payload)
    path.write_bytes(raw)
    points, actual_sha = frozen.load_frozen(path, hashlib.sha256(raw).hexdigest(), report, "audit")
    assert tuple(points) == tuple(ids[4:])
    assert actual_sha == hashlib.sha256(raw).hexdigest()
    arguments = points[ids[4]]["arguments"]
    assert type(arguments[0]) is frozen.batch.BlockArguments
    assert frozen.producer._tree_digest(arguments[0]) == case[3][frozen.BLOCK_KEYS[0]]
    assert arguments[0].landpoint_payload["lat"].dtype == np.dtype("float32")
