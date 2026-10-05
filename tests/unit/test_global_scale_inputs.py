from __future__ import annotations

import hashlib
import json
import pickle
import sys
from collections import OrderedDict, namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import global_batch as batch  # noqa: E402
from jax_orchidee.driver import orchestration as driver  # noqa: E402
from jax_orchidee.driver.run_def_materialization import write_materialized_run_def  # noqa: E402
from scripts.dev.prepare_global_scale_inputs import (  # noqa: E402
    ANCHORS,
    _host_numeric_state,
    _prepare_new_point,
    _restore_state_layout_order,
    _save_progress,
    _state_digest,
    _verify_selected_assets,
    select_points,
)


def _fixture():
    points = [*ANCHORS, "005.0-073.0", "007.0-075.0", "009.0-077.0"]
    entries = [
        {"landpoint_id": point, "spatial_split": ("train", "validation", "test")[i % 3]}
        for i, point in enumerate(points)
    ]
    run_defs = {}
    for i, point in enumerate(points):
        lon_token, lat_token = (float(v) for v in point.split("-"))
        lon = (lon_token + 180.0) % 360.0 - 180.0
        lat = lat_token - 90.0
        run_defs[point] = {
            "STATIC_SWITCH": "TRUE",
            "LIMIT_WEST": str(lon - 1.0),
            "LIMIT_EAST": str(lon + 1.0),
            "LIMIT_SOUTH": str(lat - 1.0),
            "LIMIT_NORTH": str(lat + 1.0),
            "FORCING_FILE": "forcing.nc",
            "VCMAX25__00014": str(1.0 + i),
            "ARJV__00014": str(0.5 + i),
        }
    return entries, run_defs


def test_selection_keeps_anchors_first_and_is_deterministic_and_nested():
    entries, run_defs = _fixture()
    selected_small, report = select_points(entries, run_defs, count=6)
    selected_large, _ = select_points(entries, run_defs, count=7)

    assert selected_small[:4] == list(ANCHORS)
    assert selected_large[:6] == selected_small
    assert len(set(selected_large)) == 7
    assert report["anchor_compatible_family_size"] == 7
    assert "VCMAX25__00014" in report["point_varying_allowed_keys"]


def test_unknown_static_run_def_difference_splits_family():
    entries, run_defs = _fixture()
    run_defs["009.0-077.0"]["UNRECOGNIZED_SWITCH"] = "TRUE"

    with pytest.raises(ValueError, match="anchor-compatible inventory has 6"):
        select_points(entries, run_defs, count=7)


def test_anchor_static_difference_is_rejected():
    entries, run_defs = _fixture()
    run_defs["319.0-057.0"]["UNKNOWN"] = "1"

    with pytest.raises(ValueError, match="anchor run.def static-family mismatch"):
        select_points(entries, run_defs, count=6)


def test_asset_manifest_paths_resolve_from_asset_root(tmp_path):
    point = ANCHORS[0]
    asset_root = tmp_path / "assets-v2"
    point_dir = asset_root / "landpoints" / point
    reference_file = point_dir / "reference" / "driver_start.nc"
    reference_file.parent.mkdir(parents=True)
    reference_file.write_bytes(b"reference restart")
    runtime_run_def = point_dir / "runtime_used_run.def"
    runtime_run_def.parent.mkdir(parents=True, exist_ok=True)
    run_def_values = {"STATIC_SWITCH": "TRUE"}
    write_materialized_run_def(run_def_values, runtime_run_def)
    run_def_sha256 = hashlib.sha256(runtime_run_def.read_bytes()).hexdigest()
    reference_sha256 = hashlib.sha256(reference_file.read_bytes()).hexdigest()
    manifest = {
        "landpoints": [
            {
                "landpoint_id": point,
                "files": {
                    "driver_start.nc": {
                        "path": f"landpoints/{point}/reference/driver_start.nc",
                        "sha256": reference_sha256,
                        "bytes": reference_file.stat().st_size,
                    }
                },
            }
        ]
    }

    verified = _verify_selected_assets(
        asset_root,
        [point],
        manifest,
        {point: run_def_values},
        {point: run_def_sha256},
    )

    assert verified[point]["reference_file_sha256"] == {"driver_start.nc": reference_sha256}


def test_asset_manifest_path_cannot_escape_asset_root(tmp_path):
    point = ANCHORS[0]
    asset_root = tmp_path / "assets-v2"
    point_dir = asset_root / "landpoints" / point
    (point_dir / "reference").mkdir(parents=True)
    runtime_run_def = point_dir / "runtime_used_run.def"
    run_def_values = {"STATIC_SWITCH": "TRUE"}
    write_materialized_run_def(run_def_values, runtime_run_def)
    run_def_sha256 = hashlib.sha256(runtime_run_def.read_bytes()).hexdigest()
    manifest = {
        "landpoints": [
            {
                "landpoint_id": point,
                "files": {
                    "driver_start.nc": {
                        "path": f"landpoints/{point}/../../outside.nc",
                        "sha256": "unused",
                    }
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="manifest asset path escapes asset root"):
        _verify_selected_assets(
            asset_root,
            [point],
            manifest,
            {point: run_def_values},
            {point: run_def_sha256},
        )


def test_host_numeric_state_preserves_mapping_order_and_nested_types():
    Fields = namedtuple("Fields", ("second", "first"))
    fields_by_component = OrderedDict(
        (
            ("z_component", OrderedDict((('z_field', Fields(second=np.array([1.0]), first=None)), ('a_field', 2.0)))),
            ("a_component", {"only_field": None}),
        )
    )
    source = driver.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=fields_by_component,
        provenance_by_component={"z_component": ("source",), "a_component": ()},
    )

    converted = _host_numeric_state(source)

    assert type(converted) is type(source)
    assert tuple(converted.fields_by_component) == tuple(fields_by_component)
    assert tuple(converted.fields_by_component["z_component"]) == tuple(fields_by_component["z_component"])
    nested = converted.fields_by_component["z_component"]["z_field"]
    assert type(nested) is Fields
    assert nested.second.dtype.kind == "f"
    assert nested.first is None
    assert type(converted.fields_by_component) is OrderedDict


def test_legacy_cold_state_layout_repair_is_value_preserving():
    template = driver.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=OrderedDict(
            (
                (
                    "first",
                    OrderedDict(
                        (
                            ("z", np.array([1.0, 2.0])),
                            ("a", None),
                            ("daily", OrderedDict((("second", np.array(2.0)), ("first", np.array(1.0))))),
                        )
                    ),
                ),
                ("second", {"q": np.array(3, dtype=np.int32)}),
            )
        ),
        provenance_by_component={"first": (), "second": (), "extra": ("preserve",)},
    )
    legacy = driver.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component={
            "second": {"q": np.array(3, dtype=np.int32)},
            "first": {"daily": {"first": np.array(1.0), "second": np.array(2.0)}, "a": None, "z": np.array([1.0, 2.0])},
        },
        provenance_by_component={"second": (), "first": (), "extra": ("preserve",)},
    )

    restored, migration = _restore_state_layout_order(legacy, template)

    assert tuple(restored.fields_by_component) == tuple(template.fields_by_component)
    assert tuple(restored.fields_by_component["first"]) == tuple(template.fields_by_component["first"])
    assert tuple(restored.fields_by_component["first"]["daily"]) == ("second", "first")
    assert restored.provenance_by_component == legacy.provenance_by_component
    assert migration is not None and migration["numeric_state_unchanged"] is True
    assert _state_digest(restored) == _state_digest(legacy)


def test_progress_cache_keeps_records_atomic_and_public_tuple_shape(tmp_path):
    lanes = ["lane"]
    second = ["second"]
    records = {"point": {"landpoint_id": "point"}}

    _save_progress(tmp_path, lanes, second, records)

    progress = pickle.loads((tmp_path / "prepared_inputs.progress.pkl").read_bytes())
    envelope = pickle.loads((tmp_path / "prepared_inputs.progress.cache.pkl").read_bytes())
    assert progress == (lanes, second)
    assert envelope["new_point_records"] == list(records.values())
    assert envelope["schema_version"] == 1


def test_fresh_cold_state_is_persisted_before_lane_admission(tmp_path, monkeypatch):
    point = "215.0-115.0"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    asset_root = tmp_path / "assets"
    run_def = asset_root / "landpoints" / point / "runtime_used_run.def"
    run_def.parent.mkdir(parents=True)
    run_def.write_text("asset run def", encoding="utf-8")
    data_root = tmp_path / "data"
    vegetation = data_root / "INPUTDIR_ZZ" / "PFT1860_mangr_025deg.nc"
    vegetation.parent.mkdir(parents=True)
    vegetation.write_bytes(b"vegetation")
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")
    args = SimpleNamespace(
        data_root=data_root,
        asset_root=asset_root,
        output_dir=output_dir,
        config=config,
    )
    run_values = {
        "VEGETATION_FILE": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/data/INPUTDIR_ZZ/PFT1860_mangr_025deg.nc"
    }
    packet = driver.DriverPreviousStepStatePacket(
        tstep=47,
        fields_by_component=OrderedDict((('z_component', OrderedDict((('z_field', np.array([3.0])), ('a_field', None)))),)),
        provenance_by_component={"z_component": ("source",)},
    )
    monkeypatch.setattr(driver, "prepare_paper_1961_driver_context", lambda *a, **k: object())
    monkeypatch.setattr(
        driver,
        "paper_1961_driver_cold_start_day_scaffold",
        lambda *a, **k: SimpleNamespace(
            ready_for_first_day_end_state=True,
            first_day_end_state=packet,
            missing_components=(),
        ),
    )

    def fail_at_lane_admission(*args, **kwargs):
        cached_states = pickle.loads((output_dir / "cold_day1_new60.pkl").read_bytes())
        cached_ids = json.loads((output_dir / "cold_day1_new60_identity.json").read_text(encoding="utf-8"))
        cache = pickle.loads((output_dir / "cold_day1_new60.cache.pkl").read_bytes())
        assert point in cached_states and point in cached_ids
        assert cache["states"].keys() == cached_states.keys()
        assert cache["identity_hashes"] == cached_ids
        assert tuple(cached_states[point].fields_by_component) == ("z_component",)
        raise RuntimeError("stop after cache persistence")

    monkeypatch.setattr(batch, "prepare_lane", fail_at_lane_admission)
    with pytest.raises(RuntimeError, match="stop after cache persistence"):
        _prepare_new_point(
            point,
            args=args,
            source_values=run_values,
            reference_dir=tmp_path / "reference",
            cold_states={},
            cold_identity_hashes={},
        )
