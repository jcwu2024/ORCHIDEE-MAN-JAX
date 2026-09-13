"""Audit archived canopy organic-carbon endpoints without running the model.

The attribution archives store physical carbon inventory start, fast endpoint,
incoming, and outgoing amounts. Gross transfers are not identifiable from two
endpoints, but the training targets use a deterministic net-change encoding.
The optional raw-v6 reader applies the production encoder to the stored fast
endpoint and checks alignment with the archived terminal state.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

CANOPY_FIELD = "interception_storage"
CARBON_FIELDS = ("litter_above", "litter_below", "carbon_32l", "DOC", CANOPY_FIELD)
SOURCE_SUMMATION = (
    "Fortran stomate_soilcarbon.f90 soilcarbon_leak: pool_start lines 1245-1251 "
    "and pool_end lines 2326-2332 add interception_storage directly, unlike "
    "DOC/carbon_32l which multiply veget_max."
)
SOURCE_TRANSITION = (
    "Fortran stomate_soilcarbon.f90 soilcarbon_leak lines 1497-1525: carbon "
    "storage changes by dry_dep_canopy + DOC_precip2canopy - DOC_canopy2ground."
)


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _summary(values: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    selected = values[valid]
    if not selected.size:
        return {"count": 0, "mean": None, "rmse": None, "maximum_absolute": None,
                "p05": None, "p50": None, "p95": None}
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "rmse": float(np.sqrt(np.mean(selected * selected))),
        "maximum_absolute": float(np.max(np.abs(selected))),
        "p05": float(np.quantile(selected, 0.05)),
        "p50": float(np.quantile(selected, 0.50)),
        "p95": float(np.quantile(selected, 0.95)),
    }


def _field_layout(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    matches = {
        row["field"]: row
        for row in report["state_layout"]
        if row["field"] in CARBON_FIELDS
    }
    if set(matches) != set(CARBON_FIELDS):
        raise ValueError(f"state layout lacks carbon fields: {sorted(set(CARBON_FIELDS) - set(matches))}")
    canopy = matches[CANOPY_FIELD]
    if tuple(canopy["shape"]) != (1, 2, 1):
        raise ValueError(f"unexpected canopy state shape: {canopy['shape']}")
    return matches


def _carbon_state_slices(layout: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    for field in CARBON_FIELDS:
        row = layout[field]
        indices.extend(range(int(row["start"]), int(row["stop"])))
    return np.asarray(indices, dtype=np.int64), np.arange(
        int(sum(int(layout[field]["stop"]) - int(layout[field]["start"]) for field in CARBON_FIELDS[:-1])),
        int(sum(int(layout[field]["stop"]) - int(layout[field]["start"]) for field in CARBON_FIELDS)),
        dtype=np.int64,
    )


def _pool(value: np.ndarray, fraction: np.ndarray, *, weighted: bool) -> np.ndarray:
    """Reduce a (day, 2) canopy vector to a daily landpoint amount."""
    if weighted:
        return np.sum(value * fraction, axis=1)
    return np.sum(value, axis=1)


def _archive_path(output_dir: Path, row: dict[str, Any]) -> Path:
    name = f"trained-forward-attribution-v1-{row['artifact_id']}-T{row['length']}.npz"
    path = output_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _state_loss_metrics(
    predicted: np.ndarray,
    expected: np.ndarray,
    predicted_defined: np.ndarray,
    expected_defined: np.ndarray,
    state_scale: np.ndarray,
    carbon_indices: np.ndarray,
    canopy_positions: np.ndarray,
) -> dict[str, Any]:
    prediction = predicted[:, carbon_indices]
    target = expected[:, carbon_indices]
    valid = predicted_defined[:, carbon_indices] & expected_defined[:, carbon_indices]
    scale = state_scale[carbon_indices][None, :]
    residual = (prediction - target) / scale
    valid &= np.isfinite(residual)
    residual = np.where(valid, residual, 0.0)
    count = int(np.count_nonzero(valid))
    if not count:
        raise ValueError("no defined carbon state coordinates in archived window")
    mse = float(np.sum(residual * residual) / count)
    bias = float(np.sum(residual) / count)
    canopy_valid = valid[:, canopy_positions]
    canopy_residual = residual[:, canopy_positions]
    canopy_count = int(np.count_nonzero(canopy_valid))
    canopy_sse = float(np.sum(canopy_residual * canopy_residual))
    all_sse = float(np.sum(residual * residual))
    canopy_mean = float(np.sum(canopy_residual) / canopy_count) if canopy_count else None
    return {
        "carbon_group": {
            "count": count,
            "normalized_mse": mse,
            "normalized_bias_squared": bias * bias,
            "normalized_group_loss": 0.5 * (mse + bias * bias),
        },
        "canopy_coordinates": {
            "count": canopy_count,
            "normalized_mse": canopy_sse / canopy_count if canopy_count else None,
            "normalized_bias_squared": canopy_mean * canopy_mean if canopy_mean is not None else None,
            "normalized_group_loss": (
                0.5 * (canopy_sse / canopy_count + canopy_mean * canopy_mean)
                if canopy_mean is not None else None
            ),
            "carbon_normalized_sse_share": canopy_sse / all_sse if all_sse else None,
        },
    }


def _path_metrics(
    archive: Any,
    *,
    path: str,
    teacher_start: np.ndarray,
    teacher_end: np.ndarray,
    fraction: np.ndarray,
) -> dict[str, Any]:
    start = np.asarray(archive[f"{path}_carbon_start"][:, -2:], dtype=np.float64)
    fast = np.asarray(archive[f"{path}_carbon_fast"][:, -2:], dtype=np.float64)
    incoming = np.asarray(archive[f"{path}_carbon_input"][:, -2:], dtype=np.float64)
    outgoing = np.asarray(archive[f"{path}_carbon_output"][:, -2:], dtype=np.float64)
    endpoint_residual = fast - start - incoming + outgoing
    outgoing_fraction = np.full_like(outgoing, np.nan)
    positive_start = start > 0.0
    np.divide(outgoing, start, out=outgoing_fraction, where=positive_start)

    result: dict[str, Any] = {
        "endpoint_identity_maximum_absolute": float(np.max(np.abs(endpoint_residual))),
        "positive_start_count": int(np.count_nonzero(positive_start)),
        "zero_or_negative_start_count": int(np.size(start) - np.count_nonzero(positive_start)),
        "outgoing_fraction": _summary(outgoing_fraction, positive_start),
        "per_coordinate": [],
        "reductions": {},
    }
    for coordinate, name in enumerate(("bare_soil_slot", "pft14_slot")):
        target_change = teacher_end[:, coordinate] - teacher_start[:, coordinate]
        predicted_change = fast[:, coordinate] - start[:, coordinate]
        result["per_coordinate"].append({
            "coordinate": name,
            "teacher_endpoint_change_gC_m2": _summary(target_change),
            "predicted_endpoint_change_gC_m2": _summary(predicted_change),
            "endpoint_error_gC_m2": _summary(fast[:, coordinate] - teacher_end[:, coordinate]),
            "incoming_amount_gC_m2": _summary(incoming[:, coordinate]),
            "outgoing_amount_gC_m2": _summary(outgoing[:, coordinate]),
            "outgoing_fraction": _summary(outgoing_fraction[:, coordinate], positive_start[:, coordinate]),
        })
    for reduction_name, weighted in (("fortran_direct_sum", False), ("historical_fraction_weighted", True)):
        target_start = _pool(teacher_start, fraction, weighted=weighted)
        target_end = _pool(teacher_end, fraction, weighted=weighted)
        predicted_start = _pool(start, fraction, weighted=weighted)
        predicted_fast = _pool(fast, fraction, weighted=weighted)
        result["reductions"][reduction_name] = {
            "teacher_start_gC_m2": _summary(target_start),
            "teacher_endpoint_change_gC_m2": _summary(target_end - target_start),
            "predicted_start_gC_m2": _summary(predicted_start),
            "predicted_endpoint_change_gC_m2": _summary(predicted_fast - predicted_start),
            "endpoint_error_gC_m2": _summary(predicted_fast - target_end),
            "incoming_amount_gC_m2": _summary(_pool(incoming, fraction, weighted=weighted)),
            "outgoing_amount_gC_m2": _summary(_pool(outgoing, fraction, weighted=weighted)),
        }
    return result


def _bare_slot_applicability(archive: Any, teacher_start: np.ndarray, teacher_end: np.ndarray) -> dict[str, Any]:
    """Report the source-excluded bare slot without treating it as a branch label."""
    bare_teacher_start = teacher_start[:, 0]
    bare_teacher_end = teacher_end[:, 0]
    result = {
        "source_rule": (
            "soilcarbon_leak loops j=2..nvm in the TF-DOC canopy-to-ground and storage "
            "update (Fortran lines 1497-1525), excluding the one-indexed bare-soil slot."
        ),
        "teacher_start_nonzero_count": int(np.count_nonzero(bare_teacher_start != 0.0)),
        "teacher_endpoint_nonzero_count": int(np.count_nonzero(bare_teacher_end != 0.0)),
        "model_paths": {},
    }
    for path in ("free", "reset"):
        start = np.asarray(archive[f"{path}_carbon_start"][:, -2], dtype=np.float64)
        fast = np.asarray(archive[f"{path}_carbon_fast"][:, -2], dtype=np.float64)
        incoming = np.asarray(archive[f"{path}_carbon_input"][:, -2], dtype=np.float64)
        outgoing = np.asarray(archive[f"{path}_carbon_output"][:, -2], dtype=np.float64)
        result["model_paths"][path] = {
            "positive_predicted_incoming_count": int(np.count_nonzero(incoming > 0.0)),
            "incoming_amount_gC_m2": _summary(incoming),
            "positive_predicted_endpoint_count": int(np.count_nonzero(fast > 0.0)),
            "endpoint_value_gC_m2": _summary(fast),
            "endpoint_error_gC_m2": _summary(fast - bare_teacher_end),
            "start_value_gC_m2": _summary(start),
            "outgoing_amount_gC_m2": _summary(outgoing),
        }
    return result


def _load_prepared_artifact(manifest_path: Path, artifact_id: int) -> Any:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError("prepared manifest is not accepted")
    matches = [row for row in manifest["entries"] if int(row["update"]) == artifact_id]
    if len(matches) != 1:
        raise ValueError(f"prepared manifest lacks unique artifact {artifact_id}")
    path = (manifest_path.parent / matches[0]["path"]).resolve()
    path.relative_to(manifest_path.parent.resolve())
    with gzip.open(path, "rb") as handle:
        artifact = pickle.load(handle)
    if int(artifact.schedule_entry["update"]) != artifact_id:
        raise ValueError(f"prepared artifact {artifact_id} identity drift")
    return artifact


def _one_day_rows(artifact: Any) -> dict[tuple[str, int, int], int]:
    rows: dict[tuple[str, int, int], int] = {}
    index = 0
    for record in artifact.schedule_entry["components"]["one_day"]:
        for start in record["start_rows"]:
            key = (str(record["landpoint_id"]), int(record["year"]), int(start))
            if key in rows:
                raise ValueError(f"duplicate one-day prepared key {key}")
            rows[key] = index
            index += 1
    if index != len(artifact.batch.one_day.inputs.continuous_state):
        raise ValueError("one-day schedule and prepared batch lengths differ")
    return rows


def _compact_target(value: Any) -> np.ndarray:
    target = np.asarray(value, dtype=np.float64)
    if target.shape != (1, 2, 1):
        raise ValueError(f"unexpected canopy typed target shape: {target.shape}")
    return target.reshape(2)


def _label_probe(
    *,
    selection_report: dict[str, Any],
    archive_dir: Path,
    layout: dict[str, dict[str, Any]],
    prepared_manifest: Path,
    incoming_scale: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    """Compare matching persisted typed labels, with complete coverage required for a verdict."""
    cached: dict[int, tuple[Any, dict[tuple[str, int, int], int]]] = {}
    values: dict[str, list[np.ndarray]] = {
        "incoming_prediction": [], "incoming_target": [], "incoming_defined": [],
        "outgoing_prediction": [], "outgoing_target": [], "outgoing_defined": [],
    }
    errors = {"archive_start": [], "archive_terminal": [], "fast_reconstruction": []}
    missing: list[tuple[str, int, int]] = []
    matched = 0
    zero_start_outgoing = 0
    canopy_start = int(layout[CANOPY_FIELD]["start"])
    canopy_stop = int(layout[CANOPY_FIELD]["stop"])
    for window in selection_report["windows"]:
        row = window["selection"]
        if int(row["length"]) != horizon:
            continue
        artifact_id = int(row["artifact_id"])
        if artifact_id not in cached:
            cached.clear()
            artifact = _load_prepared_artifact(prepared_manifest, artifact_id)
            cached[artifact_id] = artifact, _one_day_rows(artifact)
        artifact, row_map = cached[artifact_id]
        archive = np.load(_archive_path(archive_dir, row), allow_pickle=False)
        keys = [tuple(key) for key in row["teacher_day_keys"]]
        for day, raw_key in enumerate(keys):
            key = (str(raw_key[0]), int(raw_key[1]), int(raw_key[2]))
            prepared_index = row_map.get(key)
            if prepared_index is None:
                missing.append(key)
                continue
            batch = artifact.batch.one_day
            start = np.asarray(batch.inputs.continuous_state[prepared_index], dtype=np.float64)
            terminal = np.asarray(batch.expected_next_state.other[prepared_index], dtype=np.float64)
            outgoing = batch.process_targets[
                "inventory.carbon.interception_storage.outgoing_fraction"
            ]
            incoming = batch.process_targets[
                "inventory.carbon.interception_storage.incoming_amount"
            ]
            outgoing_target = _compact_target(outgoing.value[prepared_index])
            incoming_target = _compact_target(incoming.value[prepared_index])
            outgoing_defined = _compact_target(outgoing.defined[prepared_index]).astype(bool)
            incoming_defined = _compact_target(incoming.defined[prepared_index]).astype(bool)
            start_canopy = start[canopy_start:canopy_stop]
            terminal_canopy = terminal[canopy_start:canopy_stop]
            shifted = start_canopy + 1.0e-12
            reconstructed_fast = shifted * (1.0 - outgoing_target) + incoming_target - 1.0e-12
            predicted_start = np.asarray(archive["reset_carbon_start"][day, -2:], dtype=np.float64)
            predicted_incoming = np.asarray(archive["reset_carbon_input"][day, -2:], dtype=np.float64)
            predicted_outgoing_amount = np.asarray(archive["reset_carbon_output"][day, -2:], dtype=np.float64)
            positive = predicted_start > 0.0
            predicted_outgoing = np.full(2, np.nan, dtype=np.float64)
            np.divide(predicted_outgoing_amount, predicted_start, out=predicted_outgoing, where=positive)
            archive_terminal = np.asarray(archive["teacher_state"][day, canopy_start:canopy_stop], dtype=np.float64)
            archive_start = np.asarray(archive["teacher_start"][day, canopy_start:canopy_stop], dtype=np.float64)
            errors["archive_start"].append(start_canopy - archive_start)
            errors["archive_terminal"].append(terminal_canopy - archive_terminal)
            errors["fast_reconstruction"].append(reconstructed_fast - terminal_canopy)
            values["incoming_prediction"].append(predicted_incoming)
            values["incoming_target"].append(incoming_target)
            values["incoming_defined"].append(incoming_defined)
            values["outgoing_prediction"].append(predicted_outgoing)
            values["outgoing_target"].append(outgoing_target)
            values["outgoing_defined"].append(outgoing_defined)
            zero_start_outgoing += int(np.count_nonzero(outgoing_defined & ~positive))
            matched += 1
    result: dict[str, Any] = {
        "status": "complete" if not missing else "partial",
        "selected_day_count": sum(int(row["selection"]["length"]) == horizon for row in selection_report["windows"]) * horizon,
        "matched_one_day_label_count": matched,
        "missing_one_day_label_count": len(missing),
        "missing_day_keys": missing,
        "zero_start_outgoing_prediction_not_recoverable_count": zero_start_outgoing,
        "label_rule": "encode_inventory_endpoints uses shifted start/end by1e-12; decrease selects outgoing fraction, otherwise incoming amount",
    }
    if not matched:
        return result
    stacked = {name: np.stack(value) for name, value in values.items()}
    incoming_valid = stacked["incoming_defined"]
    outgoing_valid = stacked["outgoing_defined"] & np.isfinite(stacked["outgoing_prediction"])
    incoming_residual = (stacked["incoming_prediction"] - stacked["incoming_target"]) / incoming_scale[None, :]
    outgoing_residual = stacked["outgoing_prediction"] - stacked["outgoing_target"]
    result["archive_alignment_maximum_absolute"] = {
        name: float(np.max(np.abs(np.stack(value)))) for name, value in errors.items()
    }
    result["incoming_amount_loss"] = {
        "defined_count": int(np.count_nonzero(incoming_valid)),
        "normalized_mse": float(np.mean((incoming_residual[incoming_valid]) ** 2)),
        "per_coordinate": [
            _summary(incoming_residual[:, coordinate], incoming_valid[:, coordinate])
            for coordinate in range(2)
        ],
    }
    result["outgoing_fraction_loss_partial"] = {
        "recoverable_defined_count": int(np.count_nonzero(outgoing_valid)),
        "normalized_mse": float(np.mean((outgoing_residual[outgoing_valid]) ** 2)) if np.any(outgoing_valid) else None,
        "per_coordinate": [
            _summary(outgoing_residual[:, coordinate], outgoing_valid[:, coordinate])
            for coordinate in range(2)
        ],
        "not_exact_reason": "saved external outgoing amount loses the raw fraction whenever day-start stock is zero",
    }
    return result


def _raw_label_probe(report, archive_dir, layout, manifest_path, incoming_scale, horizon):
    from research.daily_coarse_graining.constrained_daily_replay import encode_inventory_endpoints
    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
        load_markov_shard,
    )
    from research.daily_coarse_graining.markov_dataset import load_dataset_index

    metadata = json.loads(manifest_path.read_text())
    if metadata["schema_version"] != "daily_teacher_dataset_manifest_v6":
        raise ValueError("raw label audit requires frozen coherent v6")
    contract = daily_markov_contract_from_metadata(metadata["markov_contract"])
    index = load_dataset_index(manifest_path, verify_files=False, verify_hashes=False)
    references = {(row.landpoint_id, row.year): row for row in index.select(
        spatial_split="train", temporal_split="train")}
    state_leaf = next(leaf for leaf in contract.state_leaves
                      if leaf.key == "slowproc_stomate_previous_step_state.interception_storage")
    fast_leaf = next(leaf for leaf in contract.fast_day_target_leaves
                     if leaf.family == "ok_leak" and leaf.path == (CANOPY_FIELD,))
    if tuple(state_leaf.selected_pft_indices) != tuple(fast_leaf.selected_pft_indices):
        raise ValueError("state and fast canopy PFT layouts differ")
    if state_leaf.stop - state_leaf.start != 2 or fast_leaf.stop - fast_leaf.start != 2:
        raise ValueError("audit requires compact bare/PFT14 carbon-only canopy")
    rows, incoming_errors, outgoing_errors = [], [], []
    target_in, target_out, predicted_out = [], [], []
    all_alignments = []
    for window in report["windows"]:
        selection = window["selection"]
        if selection["length"] != horizon:
            continue
        reference = references[(str(selection["landpoint_id"]), int(selection["year"]))]
        shard = load_markov_shard(reference.path, contract=contract)
        starts = np.asarray([key[2] for key in selection["teacher_day_keys"]], dtype=int)
        start = shard.state_trajectory[starts, state_leaf.start:state_leaf.stop]
        end = shard.state_trajectory[starts + 1, state_leaf.start:state_leaf.stop]
        fast = shard.fast_day_target[starts, fast_leaf.start:fast_leaf.stop]
        transition = encode_inventory_endpoints(start, fast, name="carbon.interception_storage")
        # This is exactly the encoder called by training_data._inventory_supervision.
        inc = np.asarray(transition.incoming_amount)
        out = np.asarray(transition.outgoing_fraction)
        valid = np.asarray(transition.defined)
        with np.load(_archive_path(archive_dir, selection)) as archive:
            a = int(layout[CANOPY_FIELD]["start"])
            b = int(layout[CANOPY_FIELD]["stop"])
            alignment = {
                "start_max_abs": float(np.max(np.abs(start - archive["teacher_start"][:, a:b]))),
                "terminal_max_abs": float(np.max(np.abs(end - archive["teacher_state"][:, a:b]))),
                "fast_terminal_max_abs": float(np.max(np.abs(fast-end))),
            }
            pin = np.asarray(archive["reset_carbon_input"][:, -2:])
            pout_amount = np.asarray(archive["reset_carbon_output"][:, -2:])
            pstart = np.asarray(archive["reset_carbon_start"][:, -2:])
            pout = np.full_like(pout_amount, np.nan)
            np.divide(pout_amount, pstart, out=pout, where=pstart > 0.)
        incoming_errors.append(np.where(valid, (pin-inc)/incoming_scale[None, :], np.nan))
        outgoing_errors.append(np.where(valid & (pstart > 0.), pout-out, np.nan))
        target_in.append(inc)
        target_out.append(out)
        predicted_out.append(pout)
        all_alignments.append(alignment)
        rows.append({"selection": selection, "source": str(reference.path), "alignment": alignment,
                     "teacher_incoming_mean": np.mean(inc, axis=0),
                     "teacher_outgoing_fraction_mean": np.mean(out, axis=0),
                     "bare_incoming_nonzero": int(np.count_nonzero(inc[:, 0])),
                     "bare_outgoing_nonzero": int(np.count_nonzero(out[:, 0]))})
        del shard
    incoming_error, outgoing_error = np.concatenate(incoming_errors), np.concatenate(outgoing_errors)
    targets_in, targets_out, predictions_out = map(np.concatenate, (target_in, target_out, predicted_out))
    aligned = all(all(np.isfinite(value) and value <= 1.e-8 for value in item.values()) for item in all_alignments)
    return {"status": "complete" if aligned else "alignment_discrepancy",
            "selected_day_count": len(incoming_error), "windows": rows,
            "label_source": "production encode_inventory_endpoints(start, stored ok_leak fast endpoint), default atol1e-12",
            "teacher_incoming_gC_m2_day": [_summary(targets_in[:, i]) for i in range(2)],
            "teacher_outgoing_fraction": [_summary(targets_out[:, i]) for i in range(2)],
            "predicted_outgoing_fraction": [_summary(predictions_out[:, i]) for i in range(2)],
            "incoming_normalized_residual": [_summary(incoming_error[:, i]) for i in range(2)],
            "incoming_field_mse": float(np.nanmean(incoming_error**2)),
            "outgoing_normalized_residual_positive_start_only": [_summary(outgoing_error[:, i]) for i in range(2)],
            "outgoing_mse_positive_start_only": float(np.nanmean(outgoing_error**2)),
            "outgoing_unrecoverable_count": int(np.isnan(outgoing_error).sum()),
            "limit": "raw outgoing fraction at zero-start bare slot is not stored; no full process macro claim"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(args.report.read_text(encoding="utf-8"))
    layout = _field_layout(report)
    state_scale = np.load(args.base_statistics, allow_pickle=False)["next_state__scale"]
    if state_scale.shape != (3854,):
        raise ValueError(f"unexpected v6 next-state scale shape: {state_scale.shape}")
    carbon_indices, canopy_positions = _carbon_state_slices(layout)
    process = np.load(args.process_statistics, allow_pickle=False)
    incoming_scale = process[
        "loss_scale__inventory.carbon.interception_storage.incoming_amount"
    ].reshape(2)
    outgoing_scale = process[
        "loss_scale__inventory.carbon.interception_storage.outgoing_fraction"
    ].reshape(2)
    if not np.all(np.isfinite(incoming_scale)) or np.any(incoming_scale <= 0.0):
        raise ValueError("invalid frozen incoming loss scale")
    if not np.allclose(outgoing_scale, 1.0):
        raise ValueError("outgoing fractions must retain unit-range loss scale")

    windows = []
    for row in report["windows"]:
        if int(row["selection"]["length"]) != int(args.horizon):
            continue
        selection = row["selection"]
        archive_path = _archive_path(args.archive_dir, selection)
        archive = np.load(archive_path, allow_pickle=False)
        required = {
            "teacher_start", "teacher_state", "teacher_defined", "teacher_fraction",
            "free_state", "free_defined", "reset_state", "reset_defined",
            "free_carbon_start", "free_carbon_fast", "free_carbon_input", "free_carbon_output",
            "reset_carbon_start", "reset_carbon_fast", "reset_carbon_input", "reset_carbon_output",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{archive_path.name} lacks {sorted(missing)}")
        teacher_start_full = np.asarray(archive["teacher_start"], dtype=np.float64)
        teacher_end_full = np.asarray(archive["teacher_state"], dtype=np.float64)
        start = teacher_start_full[:, int(layout[CANOPY_FIELD]["start"]):int(layout[CANOPY_FIELD]["stop"])].reshape(-1, 2)
        end = teacher_end_full[:, int(layout[CANOPY_FIELD]["start"]):int(layout[CANOPY_FIELD]["stop"])].reshape(-1, 2)
        fraction = np.asarray(archive["teacher_fraction"], dtype=np.float64).reshape(-1, 2)
        if start.shape != end.shape or start.shape != fraction.shape:
            raise ValueError(f"canopy state/fraction shape mismatch in {archive_path.name}")
        state_metrics = {
            name: _state_loss_metrics(
                np.asarray(archive[f"{name}_state"]), teacher_end_full,
                np.asarray(archive[f"{name}_defined"]), np.asarray(archive["teacher_defined"]),
                state_scale, carbon_indices, canopy_positions,
            )
            for name in ("free", "reset")
        }
        windows.append({
            "selection": selection,
            "archive": str(archive_path),
            "teacher_fraction": {
                "pft14_slot_minimum": float(np.min(fraction[:, 1])),
                "pft14_slot_maximum": float(np.max(fraction[:, 1])),
            },
            "source_consistent_canopy_stock": {
                "teacher_start_gC_m2": _summary(_pool(start, fraction, weighted=False)),
                "teacher_end_gC_m2": _summary(_pool(end, fraction, weighted=False)),
            },
            "bare_slot_applicability": _bare_slot_applicability(archive, start, end),
            "paths": {
                name: _path_metrics(archive, path=name, teacher_start=start, teacher_end=end, fraction=fraction)
                for name in ("free", "reset")
            },
            "frozen_scale_audit": {
                "incoming_amount_scale_gC_m2": incoming_scale,
                "outgoing_fraction_scale": outgoing_scale,
                "state_scale_gC_m2": state_scale[
                    int(layout[CANOPY_FIELD]["start"]):int(layout[CANOPY_FIELD]["stop"])
                ],
                "state_anchor": state_metrics,
                "branch_label_loss_reconstructable": False,
                "branch_label_loss_reason": (
                    "The archives retain predicted physical branch amounts and Teacher endpoints, "
                    "but not the stored Teacher fast endpoints needed to validate the deterministic net-change targets."
                ),
            },
        })
    if len(windows) != 8:
        raise ValueError(f"expected eight T{args.horizon} archives, found {len(windows)}")
    result = {
        "status": "completed",
        "scope": "archive_only_canopy_inventory_supervision_audit",
        "checkpoint": report["checkpoint"],
        "source_git_head": report["source_git_head"],
        "archive_count": len(windows),
        "source_provenance": {"summation": SOURCE_SUMMATION, "transition": SOURCE_TRANSITION},
        "semantics": {
            "field": CANOPY_FIELD,
            "units": "gC m-2",
            "carbon_element": "the stored v6 field has nelements=1, so it is the carbon coordinate; no nitrogen coordinate is present",
            "pft_weighting": "Fortran balance directly sums the field; historical fraction-weighting is retained only as a comparison.",
            "model_representation": "archive input/output are physical external amounts; outgoing fraction is output/start only where start > 0.",
            "budget_limit": "Independent incoming/outgoing endpoints do not establish GPP/Ra/Rh ecosystem-budget coupling.",
        },
        "windows": windows,
        "next_required_label_probe": {
            "required_existing_data": "typed v6 Teacher labels for inventory.carbon.interception_storage.{outgoing_fraction,incoming_amount}",
            "calculation": "masked per-coordinate branch MSE using the frozen process scales, followed by the equal-label carbon.ok_leak_inventory_endpoints macro contribution",
            "not_required": "Teacher regeneration or model execution",
        },
    }
    if args.prepared_manifest is not None:
        result["prepared_label_probe"] = _label_probe(
            selection_report=report,
            archive_dir=args.archive_dir,
            layout=layout,
            prepared_manifest=args.prepared_manifest,
            incoming_scale=incoming_scale,
            horizon=args.horizon,
        )
    if args.dataset_manifest is not None:
        result["raw_label_probe"] = _raw_label_probe(
            report, args.archive_dir, layout, args.dataset_manifest, incoming_scale, args.horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--report", type=Path, default=Path("outputs/trained-forward-attribution-v1.json"))
    result.add_argument("--archive-dir", type=Path, default=Path("outputs"))
    result.add_argument("--base-statistics", type=Path, default=Path("outputs/v6-training-statistics.npz"))
    result.add_argument("--process-statistics", type=Path, default=Path("outputs/v6-process-training-statistics.npz"))
    result.add_argument("--horizon", type=int, default=30)
    result.add_argument("--dataset-manifest", type=Path,
                        help="Frozen v6 manifest; read only selected train/train point-years.")
    result.add_argument(
        "--prepared-manifest", type=Path,
        help="Accepted prepared source manifest that owns the selected artifacts; read-only optional label probe.",
    )
    result.add_argument("--output", type=Path, default=Path("outputs/diagnostics/trained-canopy-inventory-audit-v1.json"))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
