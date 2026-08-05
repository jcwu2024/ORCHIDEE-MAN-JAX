from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.typed_sidecar import (
    load_typed_sidecar_contract,
    write_typed_sidecar_shard,
)
from research.daily_coarse_graining.typed_sidecar_production import (
    DEFAULT_PILOT_PLAN,
    CapabilityMasks,
    PilotEntry,
    PilotPlan,
    _canonical_sha256,
    _layout_metadata,
    _sha256_file,
    aggregate_pilot,
    build_defined_masks,
    capability_masks_from_context,
    load_pilot_plan,
)

EXPECTED_PILOT_ENTRIES = (
    (0, "001.0-071.0", 1961),
    (1, "069.0-119.0", 1961),
    (2, "281.0-095.0", 1961),
    (3, "295.0-113.0", 1961),
    (4, "315.0-097.0", 1961),
    (5, "333.0-057.0", 1961),
)


def _write_plan(path: Path, raw: dict) -> None:
    raw["pilot_sha256"] = _canonical_sha256(raw, omit="pilot_sha256")
    path.write_text(json.dumps(raw), encoding="utf-8")


def _all_values(contract, *, days: int) -> dict[str, np.ndarray]:
    return {
        field.path: np.zeros((days, *field.feature_shape), dtype=np.float64)
        for field in contract.fields
    }


def test_frozen_pilot_plan_has_exact_train_only_inventory():
    plan = load_pilot_plan()

    assert tuple(
        (entry.task_index, entry.landpoint_id, entry.year)
        for entry in plan.entries
    ) == EXPECTED_PILOT_ENTRIES
    assert plan.raw["execution"]["spatial_split"] == "train"
    assert plan.raw["execution"]["temporal_split"] == "train"
    assert plan.raw["execution"]["expected_days_per_entry"] == 364


def test_pilot_plan_rejects_self_hash_and_source_hash_drift(tmp_path):
    raw = json.loads(DEFAULT_PILOT_PLAN.read_text(encoding="utf-8"))
    raw["entries"][0]["year"] = 1962
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash mismatch"):
        load_pilot_plan(stale)

    raw = json.loads(DEFAULT_PILOT_PLAN.read_text(encoding="utf-8"))
    raw["parent"]["production_spec_canonical_sha256"] = "0" * 64
    source_drift = tmp_path / "source-drift.json"
    _write_plan(source_drift, raw)
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_pilot_plan(source_drift)


def test_capability_masks_follow_active_pfts_and_their_soil_tiles():
    active = np.zeros(14, dtype=np.bool_)
    active[[0, 13]] = True
    preference = np.ones(14, dtype=np.int32)
    preference[13] = 4
    entries = [SimpleNamespace(capabilities=("sechiba_vegetation",)) for _ in range(14)]
    entries[0] = SimpleNamespace(capabilities=("bare_soil_surface",))
    entries[13] = SimpleNamespace(
        capabilities=("sechiba_vegetation", "leak_carbon")
    )
    is_peat = np.zeros(14, dtype=np.bool_)
    is_peat[13] = True
    context = SimpleNamespace(
        run_scalars=SimpleNamespace(
            active_pft_mask=active,
            pref_soil_veg=preference,
            nstm=6,
            pft_layout=SimpleNamespace(entries=tuple(entries)),
            is_peat=is_peat,
        ),
        run_def_values={"PERMA_PEAT": "FALSE"},
        ok_explicitsnow=False,
        river_routing=False,
    )

    masks = capability_masks_from_context(context)

    np.testing.assert_array_equal(masks.active_pft, active)
    np.testing.assert_array_equal(masks.vegetation_pft, is_peat)
    np.testing.assert_array_equal(masks.leak_carbon_pft, is_peat)
    np.testing.assert_array_equal(masks.peat_pft, is_peat)
    np.testing.assert_array_equal(
        masks.soil_tile,
        np.asarray([True, False, False, True, False, False]),
    )
    assert not masks.snow
    assert masks.interface
    assert not masks.routing
    assert not masks.peat

    context.run_scalars.pref_soil_veg[13] = 7
    with pytest.raises(ValueError, match="outside the source layout"):
        capability_masks_from_context(context)


def test_defined_masks_broadcast_capabilities_and_finite_status():
    contract = load_typed_sidecar_contract()
    values = {
        field.path: np.ones((2, *field.feature_shape), dtype=np.float64)
        for field in contract.fields
    }
    values["water.precip2canopy"][0, 13] = np.nan
    values["water.rootsink"][0, 0, 3] = 1.0e20
    capability = CapabilityMasks(
        active_pft=np.asarray([True] + [False] * 12 + [True]),
        vegetation_pft=np.asarray([False] * 13 + [True]),
        leak_carbon_pft=np.asarray([False] * 13 + [True]),
        peat_pft=np.asarray([False] * 13 + [True]),
        soil_tile=np.asarray([True, False, False, True, False, False]),
        snow=False,
        interface=True,
        routing=False,
        peat=False,
    )

    masks = build_defined_masks(
        values,
        contract=contract,
        capability=capability,
    )

    np.testing.assert_array_equal(
        masks["water.precip2canopy"][1], capability.vegetation_pft
    )
    assert not masks["water.precip2canopy"][0, 13]
    np.testing.assert_array_equal(
        masks["energy.netrad_pft"][1], capability.active_pft
    )
    np.testing.assert_array_equal(
        masks["ok_leak.litter_respiration"][1, :, 0],
        capability.leak_carbon_pft,
    )
    assert np.all(masks["water.rootsink"][:, :, 0])
    assert not masks["water.rootsink"][0, 0, 3]
    assert np.all(masks["water.rootsink"][1, :, 3])
    assert not np.any(masks["water.rootsink"][:, :, 1])
    assert not np.any(masks["water.snowmelt"])
    assert not np.any(masks["water.returnflow"])
    assert not np.any(masks["ok_leak.perma_peat_carbon_transfer"])
    assert np.all(masks["water.runoff"])


def _synthetic_aggregate(tmp_path: Path):
    contract = load_typed_sidecar_contract()
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    parent_shard = parent_root / "shard.npz"
    np.savez(parent_shard, marker=np.asarray(1, dtype=np.int32))
    parent_shard_hash = _sha256_file(parent_shard)
    parent_manifest = parent_root / "dataset_manifest.json"
    parent_manifest.write_text(
        json.dumps(
            {
                "schema_version": "daily_teacher_dataset_manifest_v4",
                "dataset_id": contract.parent_dataset_id,
                "teacher_git_head": "parent-head",
                "markov_contract_sha256": contract.parent_contract_sha256,
                "shards": [
                    {
                        "landpoint_id": "001.0-071.0",
                        "year": 1961,
                        "spatial_split": "train",
                        "temporal_split": "train",
                        "shard": parent_shard.name,
                        "shard_sha256": parent_shard_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    parent_manifest_hash = _sha256_file(parent_manifest)
    pilot = PilotPlan(
        path=tmp_path / "pilot.json",
        sha256="pilot-hash",
        raw={
            "parent": {
                "dataset_id": contract.parent_dataset_id,
                "dataset_manifest_sha256": parent_manifest_hash,
                "markov_contract_sha256": contract.parent_contract_sha256,
            },
            "sidecar_contract": {"contract_sha256": contract.sha256},
            "execution": {
                "expected_day_start": 2,
                "expected_day_stop": 3,
                "expected_days_per_entry": 2,
                "spatial_split": "train",
                "temporal_split": "train",
            },
        },
        entries=(PilotEntry(0, "001.0-071.0", 1961),),
    )
    output_root = tmp_path / "output"
    task_root = output_root / "tasks" / "001.0-071.0" / "1961"
    values = _all_values(contract, days=2)
    capability = CapabilityMasks(
        active_pft=np.ones(14, dtype=np.bool_),
        vegetation_pft=np.ones(14, dtype=np.bool_),
        leak_carbon_pft=np.ones(14, dtype=np.bool_),
        peat_pft=np.ones(14, dtype=np.bool_),
        soil_tile=np.ones(6, dtype=np.bool_),
        snow=True,
        interface=True,
        routing=True,
        peat=True,
    )
    defined = build_defined_masks(
        values,
        contract=contract,
        capability=capability,
    )
    days = np.asarray([2, 3], dtype=np.int32)
    layouts = {}
    for name, requested in (
        ("dense", {field.path: "dense" for field in contract.fields}),
        ("hybrid_auto", {field.path: "auto" for field in contract.fields}),
    ):
        path = task_root / f"sidecar_{name}.npz"
        write_typed_sidecar_shard(
            path,
            contract=contract,
            day_index=days,
            values=values,
            defined=defined,
            layouts=requested,
        )
        layouts[name] = _layout_metadata(path)
    report = {
        "schema_version": "gate_e2_typed_sidecar_pilot_task_v1",
        "status": "complete",
        "pilot_sha256": pilot.sha256,
        "teacher_git_head": "teacher-head",
        "task_index": 0,
        "landpoint_id": "001.0-071.0",
        "year": 1961,
        "spatial_split": "train",
        "temporal_split": "train",
        "parent_shard_sha256": parent_shard_hash,
        "sidecar_contract_sha256": contract.sha256,
        "day_count": 2,
        "first_day": 2,
        "last_day": 3,
        "state_replay": "exact_every_day_start_and_next_state",
        "capability_masks": capability.metadata(),
        "defined_counts": {
            field.path: int(np.count_nonzero(defined[field.path]))
            for field in contract.fields
        },
        "element_counts": {
            field.path: int(defined[field.path].size) for field in contract.fields
        },
        "layouts": layouts,
        "timing_seconds": {"teacher_capture": 1.0, "total": 2.0},
    }
    report_path = task_root / "task_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return pilot, parent_manifest, output_root, report_path


def test_aggregate_redecodes_layouts_and_rejects_stale_task_identity(tmp_path):
    pilot, parent_manifest, output_root, report_path = _synthetic_aggregate(tmp_path)

    report = aggregate_pilot(
        pilot,
        parent_manifest=parent_manifest,
        output_root=output_root,
    )

    assert report["status"] == "passed"
    assert report["entry_count"] == 1
    assert report["point_days"] == 2
    assert report["accepted_layout"] == min(
        report["layout_bytes"],
        key=lambda name: (report["layout_bytes"][name], name),
    )
    manifest = json.loads((output_root / "dataset_manifest.json").read_text())
    assert manifest["pilot_sha256"] == pilot.sha256
    assert report["sidecar_dataset_manifest_sha256"] == _sha256_file(
        output_root / "dataset_manifest.json"
    )

    stale = json.loads(report_path.read_text(encoding="utf-8"))
    stale["pilot_sha256"] = "stale"
    report_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="task plan drift"):
        aggregate_pilot(
            pilot,
            parent_manifest=parent_manifest,
            output_root=output_root,
        )


def test_aggregate_rejects_modified_sidecar_bytes(tmp_path):
    pilot, parent_manifest, output_root, report_path = _synthetic_aggregate(tmp_path)
    task = json.loads(report_path.read_text(encoding="utf-8"))
    sidecar = report_path.parent / task["layouts"]["dense"]["path"]
    sidecar.write_bytes(sidecar.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        aggregate_pilot(
            pilot,
            parent_manifest=parent_manifest,
            output_root=output_root,
        )
