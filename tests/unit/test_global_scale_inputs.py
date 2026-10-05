from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.run_def_materialization import write_materialized_run_def  # noqa: E402
from scripts.dev.prepare_global_scale_inputs import (  # noqa: E402
    ANCHORS,
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
