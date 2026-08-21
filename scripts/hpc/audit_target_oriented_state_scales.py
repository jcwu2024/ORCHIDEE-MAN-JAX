"""Audit the frozen target-oriented state-anchor scale amendment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _scale_report(values: np.ndarray) -> dict[str, object]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
    return {
        "size": int(flat.size),
        "quantiles": {
            str(value): float(result) for value, result in zip(quantiles, np.quantile(flat, quantiles), strict=True)
        },
        "count_le_1e_10": int(np.count_nonzero(flat <= 1.0e-10)),
        "count_le_1e_8": int(np.count_nonzero(flat <= 1.0e-8)),
        "count_le_1e_6": int(np.count_nonzero(flat <= 1.0e-6)),
        "count_le_1e_4": int(np.count_nonzero(flat <= 1.0e-4)),
    }


def run(dataset_manifest: Path, base_statistics: Path) -> dict[str, object]:
    from research.daily_coarse_graining.constrained_daily_replay import (
        CARBON_INVENTORY_FIELDS,
    )
    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        TargetOrientedStateAnchorPolicy,
        derive_target_oriented_scales,
        load_target_oriented_state_anchor_policy,
    )
    from research.daily_coarse_graining.markov_dataset import load_training_statistics

    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    statistics = load_training_statistics(base_statistics)
    next_state = np.asarray(statistics.arrays["next_state"].scale, dtype=np.float64)
    dummy_gpp = np.ones((len(contract.active_pft_indices),), dtype=np.float64)
    policy = load_target_oriented_state_anchor_policy()
    raw_anchor = dict(policy.state_anchor)
    raw_anchor["minimum_resolved_empirical_scale"] = 0.0
    raw_payload = dict(policy.payload)
    raw_payload["one_day_state_anchor"] = raw_anchor
    raw_policy = TargetOrientedStateAnchorPolicy(raw_payload, policy.source_path, "diagnostic_raw")
    raw = derive_target_oriented_scales(
        next_state,
        dummy_gpp,
        contract,
        state_anchor_policy=raw_policy,
    )
    amended = derive_target_oriented_scales(
        next_state,
        dummy_gpp,
        contract,
        state_anchor_policy=policy,
    )
    groups = {}
    for name in raw.state._fields:
        before = np.asarray(getattr(raw.state, name), dtype=np.float64)
        after = np.asarray(getattr(amended.state, name), dtype=np.float64)
        eligible = np.asarray(getattr(amended.state_loss_eligible, name), dtype=np.bool_)
        groups[name] = {
            "raw_next_state_scale": _scale_report(before),
            "amended_state_anchor_scale": _scale_report(after),
            "native_unit_fallback_coordinate_count": int(np.count_nonzero(after != before)),
            "loss_eligible_coordinate_count": int(np.count_nonzero(eligible)),
            "loss_excluded_coordinate_count": int(np.count_nonzero(~eligible)),
        }
    low_scale_leaves = []
    for leaf in contract.state_leaves:
        if leaf.start is None or leaf.stop is None:
            continue
        values = next_state[int(leaf.start) : int(leaf.stop)]
        if values.size < 1 or float(np.min(values)) > 1.0e-4:
            continue
        low_scale_leaves.append(
            {
                "key": leaf.key,
                "size": int(values.size),
                "minimum": float(np.min(values)),
                "count_le_1e_10": int(np.count_nonzero(values <= 1.0e-10)),
                "count_le_1e_8": int(np.count_nonzero(values <= 1.0e-8)),
                "count_le_1e_6": int(np.count_nonzero(values <= 1.0e-6)),
                "count_le_1e_4": int(np.count_nonzero(values <= 1.0e-4)),
            }
        )
    carbon_fields = []
    for name in CARBON_INVENTORY_FIELDS:
        matches = tuple(
            leaf
            for leaf in contract.state_leaves
            if leaf.component == "slowproc_stomate_previous_step_state"
            and leaf.path == (name,)
            and leaf.start is not None
            and leaf.stop is not None
        )
        if len(matches) != 1:
            raise ValueError(f"carbon inventory scale owner drift for {name}")
        leaf = matches[0]
        values = next_state[int(leaf.start) : int(leaf.stop)]
        carbon_fields.append({"field": name, **_scale_report(values)})
    return {
        "schema_version": "target_oriented_state_scale_audit_v2",
        "state_anchor_policy_sha256": policy.sha256,
        "policy": {
            "minimum_resolved_empirical_scale": policy.state_anchor[
                "minimum_resolved_empirical_scale"
            ],
            "unresolved_scale_fallback": policy.state_anchor["unresolved_scale_fallback"],
        },
        "groups": groups,
        "low_scale_state_leaves": sorted(low_scale_leaves, key=lambda value: value["minimum"]),
        "carbon_inventory_fields": carbon_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_manifest, args.base_statistics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
