"""Select and prepare the held-out panel for the current fresh method.

Selection is metadata-only: it reads the typed-sidecar index and previously
frozen panel metadata, never Teacher target arrays.  Normal preparation reuses
the established one-window-per-artifact CPU preparer after the selection has
been frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.daily_coarse_graining.daily_operator.conditioned_independent_panel import (
    build_panel as build_metadata_panel,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
    load_target_oriented_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_screen import (
    HORIZONS,
    SLICE_SPLITS,
    canonical_sha256,
)
from research.daily_coarse_graining.typed_sidecar import (
    load_coherent_typed_sidecar_index,
)
from scripts.hpc import prepare_target_oriented_validation as prepare
from scripts.hpc.successor_rollout_screen_prepared import atomic_json

ROOT = Path(__file__).resolve().parents[2]
CONSUMED_PANEL_NAMES = (
    "target-oriented-validation-9b82b5a",
    "structured-gpp-fresh-validation-c3ea51a",
    "successor-staged-validation-a2c2638",
    "conditioned-independent-validation-20260909",
)
SELECTION_SEED = 20260921
WINDOWS_PER_CELL = 8
PANEL_SCHEMA = "fresh_method_validation_panel_v1"
METHOD_ID = "mixed_learning_fresh_litter_v1"
MODEL_PROTOCOL = ROOT / "manifests/coarse_graining/mixed_learning_fresh_litter_model_v1.json"
METHOD_CONTRACT = ROOT / "manifests/coarse_graining/mixed_learning_fresh_litter_v1.json"


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _panel_window_count(panel: Mapping[str, Any]) -> int:
    total = 0
    for slice_id in SLICE_SPLITS:
        cell = panel["slices"][slice_id]
        if isinstance(cell.get("horizons"), Mapping):
            total += sum(
                len(cell["horizons"][str(horizon)]) for horizon in HORIZONS
            )
        elif isinstance(cell.get("entries"), list):
            # Older staged panels store both horizons in each entry.
            total += len(cell["entries"])
        else:
            raise ValueError(f"unsupported consumed panel slice schema: {slice_id}")
    return total


def load_consumed_panels(root: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load exactly the four already-consumed panel metadata files."""

    root = Path(root).resolve()
    panels = []
    for name in CONSUMED_PANEL_NAMES:
        path = root / name / "validation_panel.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing consumed panel: {path}")
        panel = _load_json(path)
        if panel.get("sealed_test_used") is not False:
            raise ValueError(f"consumed panel crossed the sealed-test boundary: {name}")
        if panel.get("panel_sha256") != canonical_sha256(panel, hash_key="panel_sha256"):
            raise ValueError(f"consumed panel hash mismatch: {name}")
        if not isinstance(panel.get("slices"), Mapping):
            raise ValueError(f"consumed panel has no slice metadata: {name}")
        panels.append(panel)
    return tuple(panels)


def _method_metadata() -> Mapping[str, Any]:
    return {
        "method_id": METHOD_ID,
        "role": "fresh_method_held_out_validation_selection",
        "initialization": "fresh_model_seed_and_fresh_AdamW; no_exploratory_weights",
        "initialization_seed": 20260911,
        "training_data_split": "train/train",
        "model_protocol_path": MODEL_PROTOCOL.as_posix(),
        "model_protocol_sha256": sha256_file(MODEL_PROTOCOL),
        "method_contract_path": METHOD_CONTRACT.as_posix(),
        "method_contract_sha256": sha256_file(METHOD_CONTRACT),
    }


def build_fresh_method_panel(
    index: Any,
    target_contract: Any,
    consumed_panels: Sequence[Mapping[str, Any]],
    *,
    selection_seed: int = SELECTION_SEED,
    windows_per_cell: int = WINDOWS_PER_CELL,
    method_metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build a deterministic 48-window panel without reading target values."""

    plan = {
        "independent_validation": {
            "selection_seed": int(selection_seed),
            "windows_per_slice_per_horizon": int(windows_per_cell),
        }
    }
    selected = dict(build_metadata_panel(index, target_contract, plan, consumed_panels))
    selected["schema_version"] = PANEL_SCHEMA
    selected["method"] = dict(_method_metadata() if method_metadata is None else method_metadata)
    selected["method"]["selection_seed"] = int(selection_seed)
    selected["method"]["windows_per_slice_per_horizon"] = int(windows_per_cell)
    selected["consumed_panels"] = [
        {
            "directory": name,
            "panel_sha256": panel["panel_sha256"],
            "window_count": _panel_window_count(panel),
        }
        for name, panel in zip(CONSUMED_PANEL_NAMES, consumed_panels, strict=True)
    ]
    selected["consumed_panel_count"] = len(consumed_panels)
    selected["selection_policy"] = {
        "split_cells": {
            name: {"spatial_split": spatial, "temporal_split": temporal}
            for name, (spatial, temporal) in SLICE_SPLITS.items()
        },
        "horizons_days": list(HORIZONS),
        "exclude_union_of_consumed_point_years": True,
        "distinct_landpoints_within_cell": True,
        "target_values_read_during_selection": False,
    }
    selected_count = sum(
        len(selected["slices"][slice_id]["horizons"][str(horizon)])
        for slice_id in SLICE_SPLITS
        for horizon in HORIZONS
    )
    expected_count = len(SLICE_SPLITS) * len(HORIZONS) * int(windows_per_cell)
    if selected_count != expected_count:
        raise ValueError(
            f"fresh-method validation panel has {selected_count} windows; "
            f"expected {expected_count}"
        )
    selected["target_values_read_during_selection"] = False
    selected["sealed_test_used"] = False
    selected["panel_sha256"] = canonical_sha256(selected, hash_key="panel_sha256")
    return selected


def _selection_index(dataset_manifest: Path) -> Any:
    """Load only release and typed-sidecar metadata; no arrays are opened."""

    return load_coherent_typed_sidecar_index(dataset_manifest, verify_hashes=False)


def _write_selection_only(args: argparse.Namespace, panel: Mapping[str, Any]) -> Mapping[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "validation_panel.json"
    if panel_path.exists() and _load_json(panel_path) != panel:
        raise ValueError("existing fresh-method validation panel changed")
    atomic_json(panel_path, panel)
    report = {
        "schema_version": "fresh_method_validation_selection_v1",
        "status": "selected",
        "selection_only": True,
        "validation_panel": panel_path.name,
        "validation_panel_sha256": panel["panel_sha256"],
        "window_count": 48,
        "sealed_test_used": False,
        "target_values_read_during_selection": False,
    }
    atomic_json(output_dir / "selection_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path)
    parser.add_argument("--base-statistics", type=Path)
    parser.add_argument("--process-statistics", type=Path)
    parser.add_argument("--fast-statistics", type=Path)
    parser.add_argument("--training-contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--consumed-panel-root", type=Path, required=True)
    parser.add_argument("--selection-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    consumed = load_consumed_panels(args.consumed_panel_root)
    target_contract = load_target_oriented_contract()
    if args.selection_only:
        index = _selection_index(args.dataset_manifest)
        panel = build_fresh_method_panel(index, target_contract, consumed)
        return _write_selection_only(args, panel)

    required = (
        args.acceptance_report,
        args.base_statistics,
        args.process_statistics,
        args.fast_statistics,
        args.training_contract,
    )
    if any(value is None for value in required):
        raise ValueError("normal preparation requires all runtime statistics and training-contract paths")

    def panel_builder(index: Any, target: Any) -> Mapping[str, Any]:
        return build_fresh_method_panel(index, target, consumed)

    result = prepare.run(args, panel_builder=panel_builder)
    manifest_path = args.output_dir.resolve() / "prepared_manifest.json"
    manifest = dict(_load_json(manifest_path))
    manifest.update(
        {
            "validation_method_id": METHOD_ID,
            "validation_method_contract_sha256": sha256_file(METHOD_CONTRACT),
            "validation_model_protocol_sha256": sha256_file(MODEL_PROTOCOL),
            "consumed_panel_directories": list(CONSUMED_PANEL_NAMES),
            "sealed_test_used": False,
        }
    )
    atomic_json(manifest_path, manifest)
    return result | {"validation_method_id": METHOD_ID}


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") in {"selected", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
