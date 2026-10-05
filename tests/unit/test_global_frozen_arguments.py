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


@pytest.fixture
def ordered_state_case(monkeypatch):
    source = frozen.producer.driver.DriverPreviousStepStatePacket(tstep=47,
        fields_by_component={"component": {"beta": np.array([7.0]), "alpha": np.array([3.0])}},
        provenance_by_component={"component": ("audited source",)})
    target = frozen.producer.driver.DriverPreviousStepStatePacket(tstep=47,
        fields_by_component={"component": {"alpha": np.array([11.0]), "beta": np.array([13.0])}},
        provenance_by_component=source.provenance_by_component)
    def signature(context, state, tables, dispatch):
        return frozen.state_layout(state)
    monkeypatch.setattr(frozen.batch, "static_signature", signature)
    monkeypatch.setattr(frozen.batch, "audit_compatible_lanes", lambda lanes: None)
    initial = frozen.batch.fast_state_from_previous_packet(source).values_by_component
    args = frozen.batch.BlockArguments(initial, {"forcing": np.ones((7, 48, 1))},
        np.arange(2, 9, dtype=np.int32), *({"parameter": np.array([5.0])} for _ in range(6)))
    lane = frozen.batch.PreparedLane("source", None, source, None, None, args, frozen.state_layout(source))
    template = frozen.replace(lane, landpoint_id="anchor", initial_state=target,
                              static_signature=frozen.state_layout(target))
    second = args._replace(science_day_numbers=np.arange(9, 16, dtype=np.int32))
    return lane, second, template


def test_explicit_bijection_preserves_equal_shape_nonzero_fields_by_name(ordered_state_case):
    lane, second, template = ordered_state_case
    result, other, proof = frozen.canonicalize_state_arguments(lane, second, template)
    assert np.array_equal(result.arguments.values_by_component[0][0], [3.0])
    assert np.array_equal(result.arguments.values_by_component[0][1], [7.0])
    assert result.arguments.values_by_component[0][0] is lane.arguments.values_by_component[0][1]
    assert result.arguments.values_by_component[0][1] is lane.arguments.values_by_component[0][0]
    assert tuple(result.initial_state.fields_by_component["component"]) == ("alpha", "beta")
    assert result.static_signature == template.static_signature
    assert frozen.state_layout(result.initial_state) == template.static_signature
    for original, bound, block_proof in zip((lane.arguments, second), (result.arguments, other), proof["blocks"], strict=True):
        assert all(original[index] is bound[index] for index in range(1, 9))
        assert block_proof["original_arguments_sha256"] != block_proof["bound_arguments_sha256"]
        assert block_proof["named_state_unchanged"] and block_proof["cold_packet_named_association_verified"]
    assert proof["named_field_bijection"][0]["fields"][0] == {
        "field": "alpha", "source_field_index": 1, "bound_field_index": 0}


def test_schema_only_equal_shape_swap_is_rejected(ordered_state_case):
    lane, second, template = ordered_state_case
    wrong = lane.arguments._replace(values_by_component=(tuple(reversed(lane.arguments.values_by_component[0])),))
    assert frozen.batch._schema(wrong) == frozen.batch._schema(lane.arguments)
    with pytest.raises(ValueError, match="wrong named cold packet association"):
        frozen.canonicalize_state_arguments(frozen.replace(lane, arguments=wrong), second, template)


def test_migration_that_swaps_equal_shape_named_values_is_rejected(ordered_state_case, monkeypatch):
    lane, second, template = ordered_state_case
    real_restore = frozen.producer._restore_state_layout_order
    def wrong_restore(*args):
        state, record = real_restore(*args)
        state.fields_by_component["component"]["alpha"] = np.array([7.0])
        state.fields_by_component["component"]["beta"] = np.array([3.0])
        return state, record
    monkeypatch.setattr(frozen.producer, "_restore_state_layout_order", wrong_restore)
    with pytest.raises(ValueError, match="changed a named class/shape/dtype/value"):
        frozen.canonicalize_state_arguments(lane, second, template)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_canonical_layout_requires_exact_field_name_set(ordered_state_case, change):
    lane, second, template = ordered_state_case
    if change == "missing":
        del template.initial_state.fields_by_component["component"]["beta"]
    else:
        template.initial_state.fields_by_component["component"]["extra"] = np.array([17.0])
    with pytest.raises(ValueError, match="missing or extra"):
        frozen.canonicalize_state_arguments(lane, second, template)


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "component_axis", "field_axis"])
def test_named_state_bijection_rejects_nonunique_or_wrong_axes(ordered_state_case, change):
    lane, _, template = ordered_state_case
    values, layout = lane.arguments.values_by_component, frozen.state_layout(lane.initial_state)
    if change == "duplicate":
        layout = (("component",), (("alpha", "alpha"),))
    elif change == "missing":
        layout = (("component",), (("alpha",),))
    elif change == "extra":
        layout = (("component",), (("alpha", "beta", "extra"),))
    elif change == "component_axis":
        values = values + values
    elif change == "field_axis":
        values = (values[0] + (np.array([9.0]),),)
    with pytest.raises(ValueError, match="duplicate|wrong.*axis"):
        frozen._named_state(values, layout)
