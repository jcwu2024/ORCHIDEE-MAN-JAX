"""Select compatible real landpoints and prepare consolidated scale inputs.

The inventory screen is metadata-only. It preserves every unknown static
``run.def`` difference and excludes only source-audited dynamic/host fields.
Selected lanes are rechecked with ``global_batch.static_signature`` after their
own complete runtime inputs and cold Day1 states are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tarfile
import tempfile
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import global_batch as batch  # noqa: E402
from jax_orchidee.driver import orchestration as driver  # noqa: E402
from jax_orchidee.driver.global_batch import _DOMAIN_RUN_KEYS, _DYNAMIC_RUN_KEYS  # noqa: E402
from jax_orchidee.driver.run_def_materialization import (  # noqa: E402
    read_run_def_values,
    write_materialized_run_def,
)

ANCHORS = ("001.0-071.0", "069.0-119.0", "319.0-057.0", "331.0-057.0")
_COORD = re.compile(r"^(?P<lon>\d+\.\d+)-(?P<lat>\d+\.\d+)$")


def _static_values(values: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (key, value.strip())
            for key, value in values.items()
            if not _is_allowed_dynamic_key(key)
        )
    )


def _is_allowed_dynamic_key(key: str) -> bool:
    return (
        key in _DOMAIN_RUN_KEYS
        or key.split("__", 1)[0] in _DYNAMIC_RUN_KEYS
        or key == "ARJV__00014"
    )


def _read_inventory(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, str]]:
    with tarfile.open(path, mode="r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        manifest_member = members.get("asset_manifest.json")
        if manifest_member is None:
            raise ValueError("inventory has no asset_manifest.json")
        manifest_file = archive.extractfile(manifest_member)
        assert manifest_file is not None
        manifest_bytes = manifest_file.read()
        manifest = json.loads(manifest_bytes)
        entries = manifest.get("landpoints", [])
        ids = [str(item.get("landpoint_id", "")) for item in entries]
        if len(ids) != manifest.get("landpoint_count") or len(set(ids)) != len(ids):
            raise ValueError("manifest point count/identity mismatch")
        expected = {f"landpoints/{point}/runtime_used_run.def" for point in ids}
        present = {name for name in members if name.endswith("/runtime_used_run.def")}
        if present != expected:
            raise ValueError(
                f"runtime run.def inventory mismatch: missing={sorted(expected-present)[:5]} "
                f"extra={sorted(present-expected)[:5]}"
            )
        run_defs: dict[str, dict[str, str]] = {}
        run_def_sha256: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="orcjax-r4-rundef-") as temp_dir:
            scratch = Path(temp_dir) / "runtime_used_run.def"
            for point in ids:
                source = archive.extractfile(members[f"landpoints/{point}/runtime_used_run.def"])
                assert source is not None
                raw = source.read()
                run_def_sha256[point] = hashlib.sha256(raw).hexdigest()
                scratch.write_bytes(raw)
                run_defs[point] = read_run_def_values(scratch)
    manifest["_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest, run_defs, run_def_sha256


def _numeric_feature_map(
    points: list[str], run_defs: dict[str, dict[str, str]], entries: dict[str, dict[str, Any]]
) -> tuple[list[str], dict[str, tuple[float, ...]]]:
    dynamic_candidates = {
        key
        for point in points
        for key in run_defs[point]
        if key.split("__", 1)[0] in _DYNAMIC_RUN_KEYS or key == "ARJV__00014"
    }
    parameter_keys = []
    for key in sorted(dynamic_candidates):
        tokens = [run_defs[point].get(key) for point in points]
        if len(set(tokens)) < 2 or any(value is None for value in tokens):
            continue
        try:
            numbers = [float(value) for value in tokens]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in numbers):
            parameter_keys.append(key)

    split_names = sorted({str(entries[p].get("spatial_split", "unknown")) for p in points})
    feature_labels = ["longitude_sin", "longitude_cos", "latitude"]
    feature_labels.extend(f"spatial_split={name}" for name in split_names[1:])
    feature_labels.extend(parameter_keys)
    coordinates = {point: _point_coordinates(point, run_defs[point]) for point in points}
    lat_values = [v[1] for v in coordinates.values()]
    ranges: list[tuple[float, float]] = [(0.0, 1.0), (0.0, 1.0), (min(lat_values), max(lat_values))]
    for _ in split_names[1:]:
        ranges.append((0.0, 1.0))
    for key in parameter_keys:
        values = [float(run_defs[p][key]) for p in points]
        ranges.append((min(values), max(values)))

    matrix: dict[str, tuple[float, ...]] = {}
    for point in points:
        values: list[float] = []
        longitude, latitude = coordinates[point]
        radians = math.radians(longitude)
        values.extend((math.sin(radians), math.cos(radians)))
        low, high = ranges[2]
        values.append(0.0 if high == low else (latitude - low) / (high - low))
        split = str(entries[point].get("spatial_split", "unknown"))
        values.extend(float(split == name) for name in split_names[1:])
        for key, (low, high) in zip(parameter_keys, ranges[3 + len(split_names) - 1 :], strict=True):
            value = float(run_defs[point][key])
            values.append(0.0 if high == low else (value - low) / (high - low))
        matrix[point] = tuple(values)
    return feature_labels, matrix


def _point_coordinates(point: str, run_def: dict[str, str]) -> tuple[float, float]:
    _point_coordinates_from_id(point)
    required = ("LIMIT_WEST", "LIMIT_EAST", "LIMIT_SOUTH", "LIMIT_NORTH")
    try:
        west, east, south, north = (float(run_def[key]) for key in required)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{point} lacks numeric LIMIT_* domain bounds") from exc
    if not all(math.isfinite(value) for value in (west, east, south, north)):
        raise ValueError(f"{point} has nonfinite LIMIT_* domain bounds")
    if not (-360.0 <= west <= 360.0 and -360.0 <= east <= 360.0 and -90.0 <= south <= north <= 90.0):
        raise ValueError(f"{point} has invalid LIMIT_* bounds")
    width = (east - west) % 360.0
    if width == 0.0:
        raise ValueError(f"{point} has zero-width longitude domain")
    longitude = (west + width / 2.0 + 180.0) % 360.0 - 180.0
    latitude = (south + north) / 2.0
    return longitude, latitude


def _point_coordinates_from_id(point: str) -> tuple[float, float]:
    match = _COORD.fullmatch(point)
    if match is None:
        raise ValueError(f"invalid landpoint identifier: {point}")
    return float(match.group("lon")), float(match.group("lat"))


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def select_points(
    entries: list[dict[str, Any]], run_defs: dict[str, dict[str, str]], *, anchors=ANCHORS, count=64
) -> tuple[list[str], dict[str, Any]]:
    entry_map = {str(item["landpoint_id"]): item for item in entries}
    if any(point not in entry_map or point not in run_defs for point in anchors):
        raise ValueError("one or more mandatory anchor points are missing")
    families: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for point in sorted(entry_map):
        families.setdefault(_static_values(run_defs[point]), []).append(point)
    anchor_family = _static_values(run_defs[anchors[0]])
    anchor_mismatches = [point for point in anchors[1:] if _static_values(run_defs[point]) != anchor_family]
    if anchor_mismatches:
        raise ValueError(f"anchor run.def static-family mismatch: {anchor_mismatches}")
    candidates = families[anchor_family]
    if count < len(anchors) or count > len(candidates):
        raise ValueError(f"requested {count} points; anchor-compatible inventory has {len(candidates)}")

    feature_labels, features = _numeric_feature_map(candidates, run_defs, entry_map)
    selected = list(anchors)
    remaining = set(candidates) - set(selected)
    while len(selected) < count:
        scored = [
            (min(_distance(features[candidate], features[chosen]) for chosen in selected), candidate)
            for candidate in remaining
        ]
        # Lexical id is the stable tie-breaker.
        _, next_point = max(scored, key=lambda item: (item[0], tuple(-ord(c) for c in item[1])))
        selected.append(next_point)
        remaining.remove(next_point)
    family_sizes = sorted((len(members) for members in families.values()), reverse=True)
    dynamic_keys = sorted(
        key
        for key in {key for point in candidates for key in run_defs[point]}
        if any(run_defs[point].get(key) != run_defs[candidates[0]].get(key) for point in candidates)
        and _is_allowed_dynamic_key(key)
    )
    report = {
        "selection_method": "mandatory anchors then greedy maximin over sin/cos of actual LIMIT_* longitude center, normalized LIMIT_* latitude center, manifest spatial-split indicators, and varying numeric source-audited dynamic parameters; lexical point-id tie-break",
        "screen_scope": "metadata preaudit only; exact PreparedLane static signatures and full schema remain required after per-point context preparation",
        "anchor_compatible_family_size": len(candidates),
        "static_family_count": len(families),
        "static_family_sizes_descending": family_sizes,
        "point_varying_allowed_keys": dynamic_keys,
        "diversity_features": feature_labels,
        "selected_count": count,
        "selected_points": selected,
    }
    return selected, report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _tree_digest(tree: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        if array.dtype.kind not in "biufc":
            raise ValueError(f"nonnumeric prepared input leaf: {array.dtype}")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_digest(state: Any) -> str:
    return _tree_digest(state.fields_by_component)


def _host_numeric_state(state: Any) -> Any:
    def host_tree(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return type(value)((key, host_tree(item)) for key, item in value.items())
        if isinstance(value, list):
            return [host_tree(item) for item in value]
        if isinstance(value, tuple):
            items = tuple(host_tree(item) for item in value)
            return type(value)(*items) if hasattr(value, "_fields") else items
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{field.name: host_tree(getattr(value, field.name)) for field in fields(value)})
        return np.asarray(value)

    return driver.DriverPreviousStepStatePacket(
        tstep=state.tstep,
        fields_by_component=host_tree(state.fields_by_component),
        provenance_by_component=state.provenance_by_component,
    )


def _value_shape_dtype(value: Any) -> Any:
    if value is None:
        return ("none",)
    if isinstance(value, dict):
        return ("dict", type(value), tuple((key, _value_shape_dtype(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_value_shape_dtype(item) for item in value))
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((field.name, _value_shape_dtype(getattr(value, field.name))) for field in fields(value)))
    array = np.asarray(value)
    return ("array", array.shape, array.dtype.str)


def _state_values_by_path_sha256(fields_by_component: Any) -> str:
    digest = hashlib.sha256()

    def visit(value: Any, path: tuple[str, ...]) -> None:
        path_bytes = json.dumps(path, separators=(",", ":")).encode("utf-8")
        digest.update(len(path_bytes).to_bytes(4, "little"))
        digest.update(path_bytes)
        digest.update(f"{type(value).__module__}.{type(value).__qualname__}".encode("utf-8"))
        if value is None:
            return
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], (*path, str(key)))
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, (*path, str(index)))
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name), (*path, field.name))
            return
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))

    visit(fields_by_component, ())
    return digest.hexdigest()


def _restore_nested_layout(value: Any, template: Any, path: str, changes: list[str]) -> Any:
    if isinstance(value, dict) and isinstance(template, dict):
        if type(value) is not type(template) or set(value) != set(template):
            raise ValueError(f"cannot restore state layout with different nested keys: {path}")
        if tuple(value) != tuple(template):
            changes.append(path)
        return type(value)(
            (key, _restore_nested_layout(value[key], template[key], f"{path}.{key}", changes))
            for key in template
        )
    if isinstance(value, list) and isinstance(template, list):
        if len(value) != len(template):
            raise ValueError(f"cannot restore state layout with different list lengths: {path}")
        return [
            _restore_nested_layout(item, expected, f"{path}[{index}]", changes)
            for index, (item, expected) in enumerate(zip(value, template, strict=True))
        ]
    if isinstance(value, tuple) and isinstance(template, tuple):
        if len(value) != len(template) or type(value) is not type(template):
            raise ValueError(f"cannot restore state layout with different tuple structure: {path}")
        items = tuple(
            _restore_nested_layout(item, expected, f"{path}[{index}]", changes)
            for index, (item, expected) in enumerate(zip(value, template, strict=True))
        )
        return type(value)(*items) if hasattr(value, "_fields") else items
    if is_dataclass(value) and is_dataclass(template) and not isinstance(value, type) and not isinstance(template, type):
        if type(value) is not type(template):
            raise ValueError(f"cannot restore state layout with different dataclass types: {path}")
        return replace(
            value,
            **{
                field.name: _restore_nested_layout(
                    getattr(value, field.name),
                    getattr(template, field.name),
                    f"{path}.{field.name}",
                    changes,
                )
                for field in fields(value)
            },
        )
    if _value_shape_dtype(value) != _value_shape_dtype(template):
        raise ValueError(f"cannot restore state layout with incompatible shape/dtype: {path}")
    return value


def _restore_state_layout_order(state: Any, template: Any) -> tuple[Any, dict[str, Any] | None]:
    template_fields = template.fields_by_component
    state_fields = state.fields_by_component
    if type(state_fields) is not type(template_fields):
        raise ValueError("cold-state component mapping class differs from the declared anchor layout")
    if set(state_fields) != set(template_fields):
        raise ValueError("cold-state component keys differ from the declared anchor layout")
    for component in template_fields:
        if type(state_fields[component]) is not type(template_fields[component]):
            raise ValueError(f"cold-state field mapping class differs from the declared anchor layout: {component}")
        if set(state_fields[component]) != set(template_fields[component]):
            raise ValueError(f"cold-state field keys differ from the declared anchor layout: {component}")
        for name in template_fields[component]:
            if _value_shape_dtype(state_fields[component][name]) != _value_shape_dtype(template_fields[component][name]):
                raise ValueError(f"cold-state field structure/shape/dtype differs from the declared anchor layout: {component}.{name}")
    before_digest = _state_digest(state)
    component_order = tuple(state_fields) != tuple(template_fields)
    nested_order_changes: list[str] = []
    field_order = {
        component: {"before": tuple(state_fields[component]), "after": tuple(template_fields[component])}
        for component in template_fields
        if tuple(state_fields[component]) != tuple(template_fields[component])
    }
    if not component_order and not field_order:
        for component in template_fields:
            for name in template_fields[component]:
                _restore_nested_layout(
                    state_fields[component][name],
                    template_fields[component][name],
                    f"{component}.{name}",
                    nested_order_changes,
                )
    if not component_order and not field_order and not nested_order_changes:
        return state, None
    nested_order_changes.clear()
    ordered_fields = {
        component: type(state_fields[component])(
            (
                name,
                _restore_nested_layout(
                    state_fields[component][name],
                    template_fields[component][name],
                    f"{component}.{name}",
                    nested_order_changes,
                ),
            )
            for name in template_fields[component]
        )
        for component in template_fields
    }
    restored = driver.DriverPreviousStepStatePacket(
        tstep=state.tstep,
        fields_by_component=type(state_fields)(ordered_fields.items()),
        provenance_by_component=state.provenance_by_component,
    )
    after_digest = _state_digest(restored)
    path_digest_before = _state_values_by_path_sha256(state.fields_by_component)
    path_digest_after = _state_values_by_path_sha256(restored.fields_by_component)
    return restored, {
        "components_reordered": component_order,
        "field_order_changes": field_order,
        "nested_mapping_order_changes": nested_order_changes,
        "numeric_state_sha256_before": before_digest,
        "numeric_state_sha256_after": after_digest,
        "numeric_state_unchanged": path_digest_before == path_digest_after,
        "numeric_state_pytree_digest_equal": before_digest == after_digest,
        "state_values_by_path_sha256_before": path_digest_before,
        "state_values_by_path_sha256_after": path_digest_after,
        "state_values_by_path_unchanged": path_digest_before == path_digest_after,
    }


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _verify_selected_assets(
    asset_root: Path,
    selected: list[str],
    manifest: dict[str, Any],
    inventory_run_defs: dict[str, dict[str, str]],
    inventory_run_def_sha256: dict[str, str],
) -> dict[str, dict[str, Any]]:
    entries = {str(item["landpoint_id"]): item for item in manifest["landpoints"]}
    if set(selected) != {p.name for p in (asset_root / "landpoints").iterdir() if p.is_dir()}:
        raise ValueError("asset root point set does not exactly match selected IDs")
    result = {}
    for point in selected:
        point_dir = asset_root / "landpoints" / point
        run_def = point_dir / "runtime_used_run.def"
        if not run_def.is_file():
            raise FileNotFoundError(run_def)
        raw_sha = _sha256_file(run_def)
        if raw_sha != inventory_run_def_sha256[point]:
            raise ValueError(f"asset run.def differs from audited inventory for {point}")
        if read_run_def_values(run_def) != inventory_run_defs[point]:
            raise ValueError(f"asset run.def parsed values differ for {point}")
        reference_dir = point_dir / "reference"
        if not reference_dir.is_dir():
            raise FileNotFoundError(reference_dir)
        reference_hashes = {}
        for key, spec in entries[point].get("files", {}).items():
            rel = spec.get("path") if isinstance(spec, dict) else None
            expected_sha = spec.get("sha256") if isinstance(spec, dict) else None
            expected_size = spec.get("bytes") if isinstance(spec, dict) else None
            if not rel or not expected_sha:
                raise ValueError(f"manifest file identity is incomplete: {point}:{key}")
            relative_path = Path(rel)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"manifest asset path escapes asset root: {point}:{key}")
            asset_root_resolved = asset_root.resolve()
            local_path = (asset_root_resolved / relative_path).resolve()
            if not local_path.is_relative_to(asset_root_resolved):
                raise ValueError(f"manifest asset path escapes asset root: {point}:{key}")
            if not local_path.is_file():
                # The manifest's used_run.def is a distinct historical runtime
                # artifact; the selected bundle carries runtime_used_run.def.
                if key == "used_run.def":
                    continue
                raise FileNotFoundError(local_path)
            if expected_size is not None and local_path.stat().st_size != int(expected_size):
                raise ValueError(f"asset byte length mismatch: {point}:{key}")
            observed_sha = _sha256_file(local_path)
            if observed_sha != expected_sha:
                raise ValueError(f"asset checksum mismatch: {point}:{key}")
            reference_hashes[key] = observed_sha
        result[point] = {
            "runtime_run_def_path": str(run_def.resolve()),
            "runtime_run_def_sha256": raw_sha,
            "reference_dir": str(reference_dir.resolve()),
            "reference_file_sha256": reference_hashes,
        }
    return result


def _save_progress(
    output_dir: Path,
    lanes: list[Any],
    second: list[Any],
    new_records: dict[str, dict[str, Any]],
) -> None:
    _atomic_pickle(
        output_dir / "prepared_inputs.progress.cache.pkl",
        {
            "schema_version": 1,
            "lanes": lanes,
            "second": second,
            "new_point_records": list(new_records.values()),
        },
    )
    _atomic_pickle(output_dir / "prepared_inputs.progress.pkl", (lanes, second))


def _cold_identity_hash(identity: tuple[Any, ...]) -> str:
    values, config_bytes, reference_dir = identity
    digest = hashlib.sha256()
    digest.update(json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(config_bytes)
    digest.update(reference_dir.encode("utf-8"))
    return digest.hexdigest()


def _validate_resumed_lane(
    point: str,
    lane: Any,
    second: Any,
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
    source_values: dict[str, str],
    source_run_def_sha256: str,
    cold_identity_sha256: str,
) -> None:
    run_values = dict(source_values)
    vegetation_path = (args.data_root / "INPUTDIR_ZZ" / "PFT1860_mangr_025deg.nc").resolve()
    source_vegetation = run_values.get("VEGETATION_FILE")
    expected_source_vegetation = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/data/INPUTDIR_ZZ/PFT1860_mangr_025deg.nc"
    if source_vegetation is None or source_vegetation.replace("\\", "/") != expected_source_vegetation:
        raise ValueError(f"unapproved VEGETATION_FILE path mapping for resumed lane {point}")
    run_values["VEGETATION_FILE"] = str(vegetation_path).replace("\\", "/")
    reference_dir = (args.asset_root / "landpoints" / point / "reference").resolve()
    expected_identity = _cold_identity_hash((run_values, args.config.read_bytes(), str(reference_dir)))
    normalized_run_def = lane.context.run_def_path
    if (
        record.get("cold_identity_sha256") != expected_identity
        or record.get("cold_identity_sha256") != cold_identity_sha256
        or lane.context.run_def_values != run_values
        or read_run_def_values(normalized_run_def) != run_values
        or normalized_run_def.resolve() != Path(record["normalized_local_run_def_path"]).resolve()
        or lane.context.config_path.read_bytes() != args.config.read_bytes()
        or lane.context.config_path.resolve() != args.config.resolve()
        or Path(lane.context.reference_run_dir).resolve() != reference_dir
        or record.get("config_sha256") != hashlib.sha256(args.config.read_bytes()).hexdigest()
        or record.get("runtime_used_run_def_sha256") != source_run_def_sha256
        or record.get("normalized_local_run_def_sha256") != _sha256_file(normalized_run_def)
        or record.get("cold_state_numeric_sha256") != _state_digest(lane.initial_state)
        or record.get("first_block_arguments_sha256") != _tree_digest(lane.arguments)
        or record.get("second_block_arguments_sha256") != _tree_digest(second)
    ):
        raise ValueError(f"resumed lane source identity or numeric digests changed: {point}")
    computed_signature = batch.static_signature(lane.context, lane.initial_state, lane.tables, lane.dispatch)
    signature_sha256 = hashlib.sha256(repr(computed_signature).encode("utf-8")).hexdigest()
    if computed_signature != lane.static_signature or record.get("static_signature_sha256") != signature_sha256:
        raise ValueError(f"resumed lane static signature changed: {point}")
    vegetation_audit = record.get("vegetation_file_path_only_mapping", {})
    if (
        not vegetation_path.is_file()
        or vegetation_audit.get("local_file_sha256") != _sha256_file(vegetation_path)
        or record.get("run_def_values_sha256")
        != hashlib.sha256(json.dumps(run_values, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    ):
        raise ValueError(f"resumed lane vegetation or run.def identity changed: {point}")


def _save_cold_cache(
    output_dir: Path,
    states: dict[str, Any],
    identity_hashes: dict[str, str],
    layout_migrations: dict[str, dict[str, Any]] | None = None,
) -> None:
    _atomic_pickle(
        output_dir / "cold_day1_new60.cache.pkl",
        {
            "schema_version": 1,
            "states": states,
            "identity_hashes": identity_hashes,
            "layout_migrations": layout_migrations or {},
        },
    )
    _atomic_pickle(output_dir / "cold_day1_new60.pkl", states)
    _atomic_json(output_dir / "cold_day1_new60_identity.json", identity_hashes)


def _prepare_new_point(
    point: str,
    *,
    args: argparse.Namespace,
    source_values: dict[str, str],
    reference_dir: Path,
    cold_states: dict[str, Any],
    cold_identity_hashes: dict[str, str],
    cold_state_layout_migrations: dict[str, dict[str, Any]],
    layout_template: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    run_values = dict(source_values)
    vegetation_path = (args.data_root / "INPUTDIR_ZZ" / "PFT1860_mangr_025deg.nc").resolve()
    if "VEGETATION_FILE" not in run_values:
        raise KeyError(f"{point} runtime run.def lacks VEGETATION_FILE")
    source_vegetation = run_values["VEGETATION_FILE"]
    expected_source_vegetation = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/data/INPUTDIR_ZZ/PFT1860_mangr_025deg.nc"
    if source_vegetation.replace("\\", "/") != expected_source_vegetation:
        raise ValueError(f"unapproved VEGETATION_FILE path mapping for {point}: {source_vegetation}")
    if not vegetation_path.is_file():
        raise FileNotFoundError(f"local ORCHIDEE data file is missing: {vegetation_path}")
    run_values["VEGETATION_FILE"] = str(vegetation_path).replace("\\", "/")
    run_def_path = args.output_dir / "run_defs_local" / point / "used_run.def"
    write_materialized_run_def(run_values, run_def_path)
    if read_run_def_values(run_def_path) != run_values:
        raise ValueError(f"normalized local run.def round-trip mismatch: {point}")
    context = driver.prepare_paper_1961_driver_context(
        args.config, used_run_def_path=run_def_path, reference_run_dir=reference_dir
    )
    identity = (run_values, args.config.read_bytes(), str(reference_dir.resolve()))
    identity_sha256 = _cold_identity_hash(identity)
    if point in cold_states:
        cold_state = cold_states[point]
        if cold_identity_hashes.get(point) != identity_sha256:
            raise ValueError(f"cold Day1 cache identity changed for {point}")
        origin = "reused_round4_incremental_cold_state"
        cold_seconds = 0.0
    else:
        start = time.perf_counter()
        first = driver.paper_1961_driver_cold_start_day_scaffold(
            args.config,
            year=1961,
            used_run_def_path=run_def_path,
            reference_run_dir=reference_dir,
            prepared_context=context,
        )
        if not first.ready_for_first_day_end_state or first.first_day_end_state is None:
            raise RuntimeError(f"cold Day1 incomplete for {point}: {first.missing_components}")
        cold_state = _host_numeric_state(first.first_day_end_state)
        for leaf in jax.tree_util.tree_leaves(cold_state.fields_by_component):
            ready = getattr(leaf, "block_until_ready", None)
            if ready is not None:
                ready()
        cold_seconds = time.perf_counter() - start
        origin = "fresh_local_cold_Day1_from_selected_source_assets"
        cold_states[point] = cold_state
        cold_identity_hashes[point] = identity_sha256

    cold_state, layout_migration = _restore_state_layout_order(cold_state, layout_template)
    if layout_migration is not None:
        if not layout_migration["numeric_state_unchanged"] or not layout_migration["state_values_by_path_unchanged"]:
            raise ValueError(f"cold-state layout restoration changed values: {point}")
        cold_state_layout_migrations[point] = layout_migration
        cold_states[point] = cold_state
    if origin == "fresh_local_cold_Day1_from_selected_source_assets" or layout_migration is not None:
        # Persist the verified Day1 state and its canonical layout before lane admission.
        _save_cold_cache(args.output_dir, cold_states, cold_identity_hashes, cold_state_layout_migrations)

    start = time.perf_counter()
    lane = batch.prepare_lane(point, context, cold_state)
    shifted = driver.DriverPreviousStepStatePacket(
        tstep=8 * 48 - 1,
        fields_by_component=lane.initial_state.fields_by_component,
        provenance_by_component=lane.initial_state.provenance_by_component,
    )
    second_lane = batch.prepare_lane(point, context, shifted, day_numbers=range(9, 16))
    lane_seconds = time.perf_counter() - start
    record = {
        "landpoint_id": point,
        "origin": origin,
        "runtime_used_run_def_sha256": _sha256_file(args.asset_root / "landpoints" / point / "runtime_used_run.def"),
        "normalized_local_run_def_path": str(run_def_path.resolve()),
        "normalized_local_run_def_sha256": _sha256_file(run_def_path),
        "run_def_values_sha256": hashlib.sha256(
            json.dumps(run_values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "reference_dir": str(reference_dir.resolve()),
        "cold_identity_sha256": identity_sha256,
        "vegetation_file_path_only_mapping": {
            "source_runtime_used_run_def_value": source_vegetation,
            "local_value": run_values["VEGETATION_FILE"],
            "local_file_sha256": _sha256_file(vegetation_path),
            "numeric_input_changed": False,
        },
        "cold_Day1_seconds": cold_seconds,
        "lane_and_second_forcing_prepare_seconds": lane_seconds,
        "cold_state_numeric_sha256": _state_digest(lane.initial_state),
        "cold_state_layout_migration": layout_migration,
        "first_block_arguments_sha256": _tree_digest(lane.arguments),
        "second_block_arguments_sha256": _tree_digest(second_lane.arguments),
        "first_science_days": lane.arguments.science_day_numbers.tolist(),
        "second_science_days": second_lane.arguments.science_day_numbers.tolist(),
        "static_signature_sha256": hashlib.sha256(repr(lane.static_signature).encode("utf-8")).hexdigest(),
    }
    return lane, second_lane.arguments, cold_state, record


def _prepare_count(args: argparse.Namespace, manifest: dict[str, Any], run_defs: dict[str, dict[str, str]], run_def_shas: dict[str, str]) -> None:
    selection_path = args.output_dir / "selection-preaudit.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    ids = selection["selections"].get(str(args.count))
    if not ids or ids[:4] != list(ANCHORS):
        raise ValueError(f"selection file lacks the required ordered {args.count}-point prefix")
    all_selected = selection["selections"].get("64")
    if not all_selected or all_selected[: len(ids)] != ids:
        raise ValueError("64-point selected list is missing or does not preserve the requested prefix")
    source_hashes = _verify_selected_assets(args.asset_root, all_selected, manifest, run_defs, run_def_shas)
    os.environ["ORCHIDEE_DATA_ROOT"] = str(args.data_root.resolve())
    local_asset_anchor = args.local_anchor_prepared
    with local_asset_anchor.open("rb") as stream:
        lanes, second = pickle.load(stream)
    if [lane.landpoint_id for lane in lanes] != list(ANCHORS) or len(second) != len(ANCHORS):
        raise ValueError("existing local anchor pickle is not the exact four-anchor contract")
    anchor_records = []
    for lane_index, lane in enumerate(lanes):
        computed = batch.static_signature(lane.context, lane.initial_state, lane.tables, lane.dispatch)
        if read_run_def_values(lane.context.run_def_path) != lane.context.run_def_values:
            raise ValueError(f"local anchor run.def identity changed: {lane.landpoint_id}")
        if lane.context.config_path.read_bytes() != args.config.read_bytes():
            raise ValueError(f"local anchor config bytes differ from declared config: {lane.landpoint_id}")
        anchor_records.append(
            {
                "landpoint_id": lane.landpoint_id,
                "old_static_signature_sha256": hashlib.sha256(repr(lane.static_signature).encode("utf-8")).hexdigest(),
                "refreshed_static_signature_sha256": hashlib.sha256(repr(computed).encode("utf-8")).hexdigest(),
                "signature_metadata_changed": computed != lane.static_signature,
                "numeric_state_sha256": _state_digest(lane.initial_state),
                "first_block_arguments_sha256": _tree_digest(lane.arguments),
                "second_block_arguments_sha256": _tree_digest(second[lane_index]),
                "run_def_path": str(lane.context.run_def_path.resolve()),
                "run_def_values_match_on_disk": True,
                "config_path": str(lane.context.config_path.resolve()),
                "config_sha256": hashlib.sha256(lane.context.config_path.read_bytes()).hexdigest(),
            }
        )
        if computed != lane.static_signature:
            lanes[lane_index] = replace(lane, static_signature=computed)
    anchor_static_audit = batch.audit_compatible_lanes(lanes)
    anchor_second_schema = batch._schema(second[0])
    for lane, item in zip(lanes, second, strict=True):
        if batch._schema(item) != anchor_second_schema:
            raise ValueError(f"existing anchor second-block schema mismatch: {lane.landpoint_id}")
    fresh_anchor_lanes = list(lanes)
    fresh_anchor_second = list(second)

    progress_path = args.output_dir / "prepared_inputs.progress.pkl"
    progress_cache_path = args.output_dir / "prepared_inputs.progress.cache.pkl"
    cold_path = args.output_dir / "cold_day1_new60.pkl"
    cold_cache_path = args.output_dir / "cold_day1_new60.cache.pkl"
    report_path = args.output_dir / "preparation-audit.json"
    cold_states: dict[str, Any] = {}
    cold_identity_hashes: dict[str, str] = {}
    cold_state_layout_migrations: dict[str, dict[str, Any]] = {}
    if cold_cache_path.exists():
        with cold_cache_path.open("rb") as stream:
            cold_cache = pickle.load(stream)
        if cold_cache.get("schema_version") != 1:
            raise ValueError("unsupported cold Day1 cache schema")
        cold_states = cold_cache["states"]
        cold_identity_hashes = cold_cache["identity_hashes"]
        cold_state_layout_migrations = cold_cache.get("layout_migrations", {})
    elif cold_path.exists():
        identity_path = args.output_dir / "cold_day1_new60_identity.json"
        if not identity_path.is_file():
            raise ValueError("cold Day1 state exists without identity metadata; refusing an unverifiable cache")
        with cold_path.open("rb") as stream:
            cold_states = pickle.load(stream)
        cold_identity_hashes = json.loads(identity_path.read_text(encoding="utf-8"))
        _save_cold_cache(args.output_dir, cold_states, cold_identity_hashes, cold_state_layout_migrations)

    new_records: dict[str, dict[str, Any]] = {}
    if progress_cache_path.exists():
        with progress_cache_path.open("rb") as stream:
            progress_cache = pickle.load(stream)
        if progress_cache.get("schema_version") != 1:
            raise ValueError("unsupported prepared progress cache schema")
        lanes = progress_cache["lanes"]
        second = progress_cache["second"]
        new_records = {item["landpoint_id"]: item for item in progress_cache["new_point_records"]}
    elif progress_path.exists():
        with progress_path.open("rb") as stream:
            lanes, second = pickle.load(stream)
        if [lane.landpoint_id for lane in lanes] != ids[: len(lanes)]:
            raise ValueError("incremental prepared checkpoint is not an exact selection prefix")
        if len(second) != len(lanes):
            raise ValueError("incremental first/second block counts differ")
        if len(lanes) < len(ANCHORS):
            raise ValueError("incremental prepared checkpoint dropped anchors")
        anchor_static_audit = batch.audit_compatible_lanes(lanes[: len(ANCHORS)])
        existing_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        new_records = {item["landpoint_id"]: item for item in existing_report.get("new_point_records", [])}
    else:
        anchor_static_audit = batch.audit_compatible_lanes(lanes)
        for lane, second_args in zip(lanes, second, strict=True):
            if not np.array_equal(np.asarray(lane.arguments.science_day_numbers), np.arange(2, 9, dtype=np.int32)):
                raise ValueError(f"existing local anchor first block is not days 2-8: {lane.landpoint_id}")
            if not np.array_equal(np.asarray(second_args.science_day_numbers), np.arange(9, 16, dtype=np.int32)):
                raise ValueError(f"existing local anchor second block is not days 9-15: {lane.landpoint_id}")
    if [lane.landpoint_id for lane in lanes] != ids[: len(lanes)]:
        raise ValueError("incremental prepared checkpoint is not an exact selection prefix")
    if len(second) != len(lanes) or len(lanes) < len(ANCHORS):
        raise ValueError("incremental prepared checkpoint has invalid lane/block counts")
    for saved_lane, fresh_lane, saved_second, fresh_second in zip(
        lanes[: len(ANCHORS)],
        fresh_anchor_lanes,
        second[: len(ANCHORS)],
        fresh_anchor_second,
        strict=True,
    ):
        anchor_checks = (
            _state_digest(saved_lane.initial_state) == _state_digest(fresh_lane.initial_state),
            _tree_digest(saved_lane.arguments) == _tree_digest(fresh_lane.arguments),
            _tree_digest(saved_second) == _tree_digest(fresh_second),
            hashlib.sha256(repr(saved_lane.static_signature).encode("utf-8")).hexdigest()
            == hashlib.sha256(repr(fresh_lane.static_signature).encode("utf-8")).hexdigest(),
            saved_lane.context.config_path.read_bytes() == args.config.read_bytes(),
            saved_lane.context.run_def_values == fresh_lane.context.run_def_values,
            read_run_def_values(saved_lane.context.run_def_path) == saved_lane.context.run_def_values,
        )
        if not all(anchor_checks):
            raise ValueError(f"resumed anchor differs from freshly verified source anchor: {saved_lane.landpoint_id}")
    if not progress_cache_path.exists() and cold_states:
        anchor_state = lanes[0].initial_state
        restored_states = {}
        for point, state in cold_states.items():
            restored, migration = _restore_state_layout_order(state, anchor_state)
            restored_states[point] = restored
            if migration is not None:
                if not migration["numeric_state_unchanged"]:
                    raise ValueError(f"cold-state layout restoration changed numeric values: {point}")
                cold_state_layout_migrations[point] = migration
        cold_states = restored_states
        if cold_state_layout_migrations:
            _save_cold_cache(args.output_dir, cold_states, cold_identity_hashes, cold_state_layout_migrations)
    required_record_ids = [lane.landpoint_id for lane in lanes[len(ANCHORS) :]]
    missing_records = [point for point in required_record_ids if point not in new_records]
    for point in missing_records:
        if point not in cold_states or point not in cold_identity_hashes:
            raise ValueError(f"prepared progress lacks its audit record and reusable cold state for {point}")
        reference_dir = (args.asset_root / "landpoints" / point / "reference").resolve()
        rebuilt_lane, rebuilt_second, _, rebuilt_record = _prepare_new_point(
            point,
            args=args,
            source_values=run_defs[point],
            reference_dir=reference_dir,
            cold_states=cold_states,
            cold_identity_hashes=cold_identity_hashes,
            cold_state_layout_migrations=cold_state_layout_migrations,
            layout_template=lanes[0].initial_state,
        )
        saved_lane = lanes[ids.index(point)]
        saved_second = second[ids.index(point)]
        checks = {
            "prepared state": (_state_digest(saved_lane.initial_state), _state_digest(rebuilt_lane.initial_state)),
            "first-block arguments": (_tree_digest(saved_lane.arguments), _tree_digest(rebuilt_lane.arguments)),
            "second-block arguments": (_tree_digest(saved_second), _tree_digest(rebuilt_second)),
            "static signature": (
                hashlib.sha256(repr(saved_lane.static_signature).encode("utf-8")).hexdigest(),
                hashlib.sha256(repr(rebuilt_lane.static_signature).encode("utf-8")).hexdigest(),
            ),
        }
        changed = [label for label, (saved, rebuilt) in checks.items() if saved != rebuilt]
        if changed:
            raise ValueError(f"cannot recover interrupted audit record for {point}; changed {changed}")
        new_records[point] = rebuilt_record
    extra_records = set(new_records) - set(required_record_ids)
    if extra_records:
        raise ValueError(f"prepared progress contains audit records without matching lanes: {sorted(extra_records)}")
    if missing_records or not progress_cache_path.exists() or not progress_path.exists():
        _save_progress(args.output_dir, lanes, second, new_records)
    for index, lane in enumerate(lanes[len(ANCHORS) :], len(ANCHORS)):
        point = lane.landpoint_id
        _validate_resumed_lane(
            point,
            lane,
            second[index],
            new_records[point],
            args=args,
            source_values=run_defs[point],
            source_run_def_sha256=run_def_shas[point],
            cold_identity_sha256=cold_identity_hashes.get(point, ""),
        )

    report = {
        "status": "preparing",
        "requested_count": args.count,
        "selected_points": ids,
        "inventory_manifest_sha256": manifest["_sha256"],
        "asset_root": str(args.asset_root.resolve()),
        "asset_inputs": source_hashes,
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "data_root": str(args.data_root.resolve()),
        "anchors": {
            "source": str(local_asset_anchor.resolve()),
            "origin": "exact old local-audit PreparedLane/BlockArguments/state objects reused without numerical replacement",
            "prepared_pickle_sha256": _sha256_file(local_asset_anchor),
            "stored_static_signatures_recomputed": True,
            "numeric_identity_records": anchor_records,
            "strict_static_compatibility": anchor_static_audit,
        },
        "new_point_records": list(new_records.values()),
        "cold_state_layout_migrations": [
            {"landpoint_id": point, **migration} for point, migration in cold_state_layout_migrations.items()
        ],
        "input_interface": "(list[PreparedLane], list[BlockArguments])",
        "static_schema_audit": None,
    }
    if not progress_path.exists() or not progress_cache_path.exists():
        _save_progress(args.output_dir, lanes, second, new_records)
    for point in ids[len(lanes) :]:
        reference_dir = (args.asset_root / "landpoints" / point / "reference").resolve()
        lane, second_args, cold_state, record = _prepare_new_point(
            point,
            args=args,
            source_values=run_defs[point],
            reference_dir=reference_dir,
            cold_states=cold_states,
            cold_identity_hashes=cold_identity_hashes,
            cold_state_layout_migrations=cold_state_layout_migrations,
            layout_template=lanes[0].initial_state,
        )
        candidate_lanes = [*lanes, lane]
        candidate_second = [*second, second_args]
        report["new_point_records"] = [*new_records.values(), record]
        report["status"] = "preparing"
        # Do not silently skip a candidate if its full prepared closure differs.
        static_audit = batch.audit_compatible_lanes(candidate_lanes)
        second_schema = batch._schema(candidate_second[0])
        for index, item in enumerate(candidate_second[1:], 1):
            if batch._schema(item) != second_schema:
                raise ValueError(f"second-block input schema/dtype mismatch at {candidate_lanes[index].landpoint_id}")
            if not np.array_equal(np.asarray(item.science_day_numbers), np.arange(9, 16, dtype=np.int32)):
                raise ValueError(f"second-block days are not 9-15 at {candidate_lanes[index].landpoint_id}")
        lanes, second = candidate_lanes, candidate_second
        new_records[point] = record
        _save_progress(args.output_dir, lanes, second, new_records)
        _save_cold_cache(args.output_dir, cold_states, cold_identity_hashes, cold_state_layout_migrations)
        report["static_schema_audit"] = static_audit
        report["checkpoint_count"] = len(lanes)
        report["checkpoint_prepared_pickle_sha256"] = _sha256_file(progress_path)
        _atomic_json(report_path, report)

    # Persist an atomic, stable pickle after the requested prefix is complete.
    stable_path = args.output_dir / f"prepared_inputs_local_{args.count}.pkl"
    _atomic_pickle(stable_path, (lanes, second))
    report["status"] = "prepared"
    report["prepared_pickle"] = str(stable_path.resolve())
    report["prepared_pickle_sha256"] = _sha256_file(stable_path)
    report["prepared_point_count"] = len(lanes)
    report["new_point_records"] = list(new_records.values())
    report["static_schema_audit"] = batch.audit_compatible_lanes(lanes)
    report["actual_source_asset_hash_validation"] = "passed for every selected point and manifest-listed reference file"
    report["cold_states_sha256"] = _sha256_file(cold_path)
    report["cold_identities_sha256"] = _sha256_file(args.output_dir / "cold_day1_new60_identity.json")
    report["cold_state_payload_schema"] = {
        "path": str(cold_path.resolve()),
        "identity_path": str((args.output_dir / "cold_day1_new60_identity.json").resolve()),
        "values": "point_id -> DriverPreviousStepStatePacket with NumPy numeric leaves",
        "point_ids": sorted(cold_states),
    }
    _atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "count": len(lanes), "prepared_pickle": str(stable_path), "sha256": report["prepared_pickle_sha256"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-tar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--count", type=int, choices=[16, 64])
    parser.add_argument("--local-anchor-prepared", type=Path, default=ROOT / "outputs/research/global_throughput/local-audit/prepared_inputs.pkl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    manifest, run_defs, run_def_shas = _read_inventory(args.inventory_tar)
    if args.count is not None:
        if args.asset_root is None:
            parser.error("--asset-root is required with --count")
        _prepare_count(args, manifest, run_defs, run_def_shas)
        return
    entries = manifest["landpoints"]
    selected64, report64 = select_points(entries, run_defs, count=max(args.counts))
    if min(args.counts) > len(selected64):
        raise ValueError("small selection cannot exceed large selection")
    output = {
        "inventory_manifest_sha256": manifest["_sha256"],
        "dataset_id": manifest["dataset_id"],
        "landpoint_count": manifest["landpoint_count"],
        "anchor_order": list(ANCHORS),
        "requested_counts": sorted(set(args.counts)),
        "large_selection": report64,
        "selections": {str(count): selected64[:count] for count in sorted(set(args.counts))},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "selection-preaudit.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    ids_path = args.output_dir / "selected-point-ids.txt"
    ids_path.write_text("\n".join(selected64) + "\n", encoding="ascii")
    print(json.dumps({"selection_file": str(target), "points_16": selected64[:16], "points_64": selected64}, indent=2))


if __name__ == "__main__":
    main()
