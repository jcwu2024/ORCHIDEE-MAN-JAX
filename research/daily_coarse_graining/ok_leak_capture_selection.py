"""Deterministic train-only selection for bounded OK_LEAK driver capture."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    MarkovShard,
    daily_markov_contract_from_metadata,
    load_markov_shard,
)

DYNAMIC_SELECTION_METRICS = (
    "soil_wetness",
    "soil_temperature_k",
    "pft14_gpp_daily",
    "soil_carbon_proxy",
    "precip_daily",
    "hydrologic_export",
)
ALL_SELECTION_METRICS = (
    *DYNAMIC_SELECTION_METRICS,
    "peat_hydrology_activity",
    "peat_cover_fraction",
)
PAPER_INACTIVE_DRIVER_FIELDS = ("runoff2peat", "shumdiag_peat")
DEFAULT_SAMPLES_PER_LANDPOINT = 16
MINIMUM_EXTREMA_CAPACITY = 1 + 2 * len(DYNAMIC_SELECTION_METRICS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _leaf(leaves: Sequence[Any], key: str) -> Any:
    matches = tuple(item for item in leaves if item.key == key)
    if len(matches) != 1:
        raise ValueError(f"capture selection requires exactly one {key} leaf")
    return matches[0]


def _leaf_values(matrix: np.ndarray, leaves: Sequence[Any], key: str) -> np.ndarray:
    spec = _leaf(leaves, key)
    return np.asarray(matrix[:, spec.start : spec.stop]).reshape(
        matrix.shape[0], *spec.shape
    )


def _valid_float(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.where(np.isfinite(array) & (np.abs(array) < 1.0e19), array, np.nan)


def _row_mean(value: np.ndarray) -> np.ndarray:
    array = _valid_float(value)
    flat = array.reshape(array.shape[0], -1)
    count = np.sum(np.isfinite(flat), axis=1)
    total = np.nansum(flat, axis=1)
    return np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )


def _row_sum(value: np.ndarray) -> np.ndarray:
    array = _valid_float(value)
    flat = array.reshape(array.shape[0], -1)
    with np.errstate(invalid="ignore"):
        result = np.nansum(flat, axis=1)
    return np.where(np.any(np.isfinite(flat), axis=1), result, np.nan)


def _pft14_values(value: np.ndarray, spec: Any) -> np.ndarray:
    if "nvm" not in spec.axis_names:
        raise ValueError(f"{spec.key} has no PFT axis")
    selected = tuple(spec.selected_pft_indices)
    if 13 not in selected:
        raise ValueError(f"{spec.key} does not retain PFT14")
    compact_axis = spec.axis_names.index("nvm") + 1
    return np.take(value, selected.index(13), axis=compact_axis)


def capture_selection_metrics(
    shard: MarkovShard,
    contract: DailyMarkovContract,
) -> Mapping[str, np.ndarray]:
    """Compute source-backed condition metrics for every day in one shard."""

    states = np.asarray(shard.state_trajectory[:-1])
    targets = np.asarray(shard.fast_day_target)
    state_leaves = contract.state_leaves
    target_leaves = contract.fast_day_target_leaves

    soil_mc = _leaf_values(
        states,
        state_leaves,
        "hydrol_previous_step_state.soil_mc",
    )
    soil_temperature = _leaf_values(
        targets,
        target_leaves,
        "daily_interface.tsoil_daily",
    )
    gpp_spec = _leaf(
        target_leaves,
        "daily_interface.gpp_daily",
    )
    gpp = _leaf_values(targets, target_leaves, gpp_spec.key)
    soil_carbon = _leaf_values(
        states,
        state_leaves,
        "slowproc_stomate_previous_step_state.soilc_total",
    )
    precipitation = _leaf_values(
        targets,
        target_leaves,
        "daily_interface.precip_daily",
    )
    runoff = _leaf_values(
        targets,
        target_leaves,
        "hydrol_previous_step_state.runoff_per_soil",
    )
    drainage = _leaf_values(
        targets,
        target_leaves,
        "hydrol_previous_step_state.drainage_per_soil",
    )
    runoff2peat = _leaf_values(
        targets,
        target_leaves,
        "hydrol_previous_step_state.runoff2peat",
    )
    shumdiag_peat = _leaf_values(
        targets,
        target_leaves,
        "hydrol_previous_step_state.shumdiag_peat",
    )
    peat = _leaf_values(
        states,
        state_leaves,
        "slowproc_stomate_previous_step_state.fpeat",
    )
    metrics = {
        "soil_wetness": _row_mean(soil_mc),
        "soil_temperature_k": _row_mean(soil_temperature),
        "pft14_gpp_daily": _row_sum(_pft14_values(gpp, gpp_spec)),
        "soil_carbon_proxy": _row_sum(soil_carbon),
        "precip_daily": _row_sum(precipitation),
        "hydrologic_export": _row_sum(runoff) + _row_sum(drainage),
        "peat_hydrology_activity": _row_sum(np.abs(runoff2peat))
        + _row_sum(np.abs(shumdiag_peat)),
        "peat_cover_fraction": _row_mean(peat),
    }
    invalid = {
        name: int(np.count_nonzero(~np.isfinite(value)))
        for name, value in metrics.items()
        if not np.all(np.isfinite(value))
    }
    if invalid:
        raise ValueError(f"capture selection metrics contain invalid rows: {invalid}")
    return metrics


def _record_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["landpoint_id"]),
        int(record["year"]),
        int(record["day_index"]),
    )


def _rank_features(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    columns = []
    for name in DYNAMIC_SELECTION_METRICS:
        values = np.asarray([item["metrics"][name] for item in records])
        unique, inverse = np.unique(values, return_inverse=True)
        ranks = (
            np.zeros(values.size, dtype=np.float64)
            if unique.size == 1
            else inverse.astype(np.float64) / (unique.size - 1)
        )
        columns.append(ranks)
    time_order = np.arange(len(records), dtype=np.float64)
    columns.append(time_order / max(1, len(records) - 1))
    return np.column_stack(columns)


def select_bounded_capture_records(
    records: Sequence[Mapping[str, Any]],
    *,
    samples_per_landpoint: int = DEFAULT_SAMPLES_PER_LANDPOINT,
) -> tuple[dict[str, Any], ...]:
    """Select cold anchor, per-metric extrema, then rank-space farthest days."""

    if samples_per_landpoint < MINIMUM_EXTREMA_CAPACITY:
        raise ValueError(
            "samples_per_landpoint must retain the cold anchor and all metric extrema"
        )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["landpoint_id"])].append(record)
    if not grouped:
        raise ValueError("capture selection has no train/train candidate records")

    output = []
    for landpoint_id in sorted(grouped):
        group = sorted(grouped[landpoint_id], key=_record_key)
        if len(group) < samples_per_landpoint:
            raise ValueError(f"{landpoint_id} has too few candidate days")
        selected: dict[int, set[str]] = defaultdict(set)
        selected[0].add("earliest_train_day")
        for name in DYNAMIC_SELECTION_METRICS:
            values = np.asarray([item["metrics"][name] for item in group])
            selected[int(np.argmin(values))].add(f"{name}:minimum")
            selected[int(np.argmax(values))].add(f"{name}:maximum")

        features = _rank_features(group)
        while len(selected) < samples_per_landpoint:
            chosen = np.asarray(sorted(selected), dtype=np.int64)
            distance = np.min(
                np.sum((features[:, None, :] - features[chosen][None, :, :]) ** 2, axis=2),
                axis=1,
            )
            distance[chosen] = -1.0
            next_index = int(np.argmax(distance))
            selected[next_index].add("rank_space_farthest_fill")

        for index in sorted(selected):
            record = dict(group[index])
            record["selection_reasons"] = sorted(selected[index])
            output.append(record)
    return tuple(sorted(output, key=_record_key))


def build_bounded_capture_plan(
    dataset_manifest: Path,
    dataset_root: Path,
    *,
    samples_per_landpoint: int = DEFAULT_SAMPLES_PER_LANDPOINT,
) -> Mapping[str, Any]:
    """Read only accepted train/train shards and build a hash-bound capture plan."""

    manifest_path = dataset_manifest.resolve()
    root = dataset_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("capture selection requires a complete dataset manifest")
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    references = tuple(
        item
        for item in manifest["shards"]
        if item["spatial_split"] == "train" and item["temporal_split"] == "train"
    )
    if not references:
        raise ValueError("dataset has no train/train shards")

    candidates = []
    verified_shards = {}
    for reference in sorted(
        references,
        key=lambda item: (item["landpoint_id"], int(item["year"])),
    ):
        shard_path = (root / reference["shard"]).resolve()
        observed_hash = _sha256_file(shard_path)
        if observed_hash != reference["shard_sha256"]:
            raise ValueError(f"source shard hash mismatch: {reference['shard']}")
        verified_shards[reference["shard"]] = observed_hash
        shard = load_markov_shard(shard_path, contract=contract)
        metrics = capture_selection_metrics(shard, contract)
        for row, day_index in enumerate(np.asarray(shard.day_index)):
            candidates.append(
                {
                    "landpoint_id": reference["landpoint_id"],
                    "year": int(reference["year"]),
                    "day_index": int(day_index),
                    "row": row,
                    "source_shard": reference["shard"],
                    "source_shard_sha256": observed_hash,
                    "metrics": {
                        name: float(values[row]) for name, values in metrics.items()
                    },
                }
            )

    selected = select_bounded_capture_records(
        candidates,
        samples_per_landpoint=samples_per_landpoint,
    )
    shard_cache = {}
    final_records = []
    for record in selected:
        shard_name = record["source_shard"]
        if shard_name not in shard_cache:
            shard_cache[shard_name] = load_markov_shard(
                (root / shard_name).resolve(), contract=contract
            )
        shard = shard_cache[shard_name]
        row = int(record["row"])
        final_records.append(
            record
            | {
                "day_start_state_sha256": _sha256_array(shard.state_trajectory[row]),
                "fast_day_target_sha256": _sha256_array(shard.fast_day_target[row]),
            }
        )

    landpoints = sorted({item["landpoint_id"] for item in final_records})
    plan = {
        "schema_version": "ok_leak_auxiliary_capture_plan_v1",
        "dataset_id": manifest["dataset_id"],
        "teacher_git_head": manifest["teacher_git_head"],
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "markov_contract_sha256": manifest["markov_contract_sha256"],
        "sealed_test_used": False,
        "split_policy": {
            "spatial_split": "train",
            "temporal_split": "train",
        },
        "selection_policy": {
            "method": "per_landpoint_extrema_then_rank_space_farthest_v1",
            "samples_per_landpoint": samples_per_landpoint,
            "dynamic_metrics": list(DYNAMIC_SELECTION_METRICS),
            "condition_metrics": list(ALL_SELECTION_METRICS),
            "mandatory_anchors": ["earliest_train_day", "minimum", "maximum"],
        },
        "execution_contract": {
            "paper_hydrol_soil_peat_hydro": False,
            "inactive_zero_driver_fields": list(PAPER_INACTIVE_DRIVER_FIELDS),
            "provenance": [
                "jax_orchidee/driver/orchestration.py::PAPER_1961_HYDROL_SOIL_PEAT_HYDRO",
                "traces/hydrol_1961_v9: peat_hydro=F, branch_peat=F",
            ],
        },
        "candidate_shard_count": len(references),
        "candidate_day_count": len(candidates),
        "selected_landpoints": landpoints,
        "selected_day_count": len(final_records),
        "estimated_uncompressed_driver_bytes": len(final_records) * 486_912,
        "source_shards": verified_shards,
        "records": final_records,
    }
    return plan | {"plan_sha256": _canonical_sha256(plan)}


def verify_bounded_capture_plan(
    plan_path: Path,
    dataset_manifest: Path,
    dataset_root: Path,
) -> Mapping[str, Any]:
    """Independently verify plan identity, split scope, anchors, and selected rows."""

    plan = json.loads(plan_path.resolve().read_text(encoding="utf-8"))
    manifest_path = dataset_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = dataset_root.resolve()
    checks = []

    def check(check_id: str, passed: bool, detail: Any) -> None:
        checks.append({"id": check_id, "passed": bool(passed), "detail": detail})

    unsigned_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    observed_plan_hash = _canonical_sha256(unsigned_plan)
    check("schema", plan.get("schema_version") == "ok_leak_auxiliary_capture_plan_v1", plan.get("schema_version"))
    check("plan_sha256", plan.get("plan_sha256") == observed_plan_hash, observed_plan_hash)
    check("sealed_test_unused", plan.get("sealed_test_used") is False, plan.get("sealed_test_used"))
    check(
        "split_policy",
        plan.get("split_policy") == {"spatial_split": "train", "temporal_split": "train"},
        plan.get("split_policy"),
    )
    for field in ("dataset_id", "teacher_git_head", "markov_contract_sha256"):
        check(f"manifest_{field}", plan.get(field) == manifest.get(field), plan.get(field))
    check(
        "dataset_manifest_sha256",
        plan.get("dataset_manifest_sha256") == _sha256_file(manifest_path),
        plan.get("dataset_manifest_sha256"),
    )

    references = tuple(
        item
        for item in manifest["shards"]
        if item["spatial_split"] == "train" and item["temporal_split"] == "train"
    )
    expected_sources = {
        item["shard"]: item["shard_sha256"]
        for item in references
    }
    check("source_shard_inventory", plan.get("source_shards") == expected_sources, len(expected_sources))
    check("candidate_shard_count", plan.get("candidate_shard_count") == len(references), len(references))

    records = tuple(plan.get("records", ()))
    policy = plan.get("selection_policy", {})
    samples_per_landpoint = int(policy.get("samples_per_landpoint", -1))
    expected_landpoints = sorted({item["landpoint_id"] for item in references})
    expected_days = len(expected_landpoints) * samples_per_landpoint
    check("selected_landpoints", plan.get("selected_landpoints") == expected_landpoints, expected_landpoints)
    check("selected_day_count", len(records) == plan.get("selected_day_count") == expected_days, len(records))
    check(
        "estimated_driver_bytes",
        plan.get("estimated_uncompressed_driver_bytes") == len(records) * 486_912,
        plan.get("estimated_uncompressed_driver_bytes"),
    )
    identities = tuple(_record_key(item) for item in records)
    check("unique_day_identity", len(identities) == len(set(identities)), len(set(identities)))

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["landpoint_id"])].append(record)
    count_detail = {name: len(grouped.get(name, ())) for name in expected_landpoints}
    check(
        "per_landpoint_count",
        all(value == samples_per_landpoint for value in count_detail.values()),
        count_detail,
    )
    missing_reasons = []
    for landpoint_id in expected_landpoints:
        reasons = {
            reason
            for record in grouped.get(landpoint_id, ())
            for reason in record.get("selection_reasons", ())
        }
        required = {"earliest_train_day"} | {
            f"{name}:{bound}"
            for name in DYNAMIC_SELECTION_METRICS
            for bound in ("minimum", "maximum")
        }
        missing_reasons.extend(
            f"{landpoint_id}:{reason}" for reason in sorted(required - reasons)
        )
    check("mandatory_anchor_reasons", not missing_reasons, missing_reasons)

    metric_errors = []
    for record in records:
        metrics = record.get("metrics", {})
        if tuple(metrics) != ALL_SELECTION_METRICS:
            metric_errors.append(f"{_record_key(record)}:schema")
        elif not all(np.isfinite(float(value)) for value in metrics.values()):
            metric_errors.append(f"{_record_key(record)}:nonfinite")
    check("metric_schema_and_finiteness", not metric_errors, metric_errors)
    peat_activity = np.asarray(
        [item.get("metrics", {}).get("peat_hydrology_activity", np.nan) for item in records],
        dtype=np.float64,
    )
    peat_detail = {
        "minimum": float(np.nanmin(peat_activity)) if peat_activity.size else None,
        "maximum": float(np.nanmax(peat_activity)) if peat_activity.size else None,
    }
    check(
        "inactive_peat_hydrology_zero",
        bool(
            peat_activity.size
            and np.all(np.isfinite(peat_activity))
            and np.all(peat_activity == 0.0)
            and plan.get("execution_contract", {}).get("paper_hydrol_soil_peat_hydro") is False
            and tuple(plan.get("execution_contract", {}).get("inactive_zero_driver_fields", ()))
            == PAPER_INACTIVE_DRIVER_FIELDS
        ),
        peat_detail,
    )

    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    shard_cache = {}
    selected_row_errors = []
    for record in records:
        shard_name = record["source_shard"]
        if shard_name not in expected_sources or record.get("source_shard_sha256") != expected_sources.get(shard_name):
            selected_row_errors.append(f"{_record_key(record)}:source")
            continue
        if shard_name not in shard_cache:
            shard_path = (root / shard_name).resolve()
            if _sha256_file(shard_path) != expected_sources[shard_name]:
                selected_row_errors.append(f"{shard_name}:file_hash")
                continue
            shard_cache[shard_name] = load_markov_shard(shard_path, contract=contract)
        shard = shard_cache.get(shard_name)
        if shard is None:
            continue
        row = int(record["row"])
        if row < 0 or row >= shard.days:
            selected_row_errors.append(f"{_record_key(record)}:row")
            continue
        if int(shard.year) != int(record["year"]) or int(shard.day_index[row]) != int(record["day_index"]):
            selected_row_errors.append(f"{_record_key(record)}:identity")
        if _sha256_array(shard.state_trajectory[row]) != record.get("day_start_state_sha256"):
            selected_row_errors.append(f"{_record_key(record)}:state_hash")
        if _sha256_array(shard.fast_day_target[row]) != record.get("fast_day_target_sha256"):
            selected_row_errors.append(f"{_record_key(record)}:target_hash")
    check("selected_source_rows", not selected_row_errors, selected_row_errors)

    return {
        "schema_version": "ok_leak_auxiliary_capture_plan_verification_v1",
        "passed": all(item["passed"] for item in checks),
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan.get("plan_sha256"),
        "sealed_test_used": False,
        "checks": checks,
    }
