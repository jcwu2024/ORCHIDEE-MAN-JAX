"""Small asset contract tests; no neural model or gradient compilation."""
import gzip
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import numpy as np
import pytest

from scripts.hpc import shared_latent_T7_batch_assets as assets


class Templates(NamedTuple):
    continuous_state: object
    day_index: object


class Case(NamedTuple):
    initial_continuous: object
    initial_discrete: object
    days: object
    targets: object
    loss_context: object


def case(value):
    return Case(np.full(3854, value, dtype=np.float64),
        {"discrete": np.asarray(value, dtype=np.int32)},
        (Templates(np.full((7, 3854), value, dtype=np.float64), np.arange(1, 8, dtype=np.int32)),
         {"physical_static": np.asarray(value, dtype=np.float64)}),
        {"state": np.full((7, 3), value, dtype=np.float64),
         "mask": np.full((7, 3), value % 2 == 0, dtype=bool)},
        {"initial_mask": np.full(3, value % 2 == 0, dtype=bool)})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from research.daily_coarse_graining.daily_operator import shared_recurrent_training
    monkeypatch.setattr(shared_recurrent_training, "prepare_shared_window",
                        lambda head, window, **kw: window.case)
    cases = [case(value) for value in range(1, 5)]
    dataset = {"schema_version": "daily_teacher_dataset_manifest_v6", "status": "complete", "shards": []}
    entries = []
    for i in range(4):
        dataset["shards"].append({"landpoint_id": str(i), "year": 1990+i,
            "spatial_split": "train", "temporal_split": "train", "day_count": 365,
            "shard_sha256": f"parent{i}", "typed_shard_sha256": f"typed{i}"})
        record = {"landpoint_id": str(i), "year": 1990+i,
            "parent_shard_sha256": f"parent{i}", "typed_shard_sha256": f"typed{i}", "start_rows": [i]}
        artifact = SimpleNamespace(schedule_entry={"update": i+1, "runtime_year": 1990+i,
            "components": {"seven_day": [record]}}, trace_signature="trace",
            training_contract_sha256="contract", batch=SimpleNamespace(seven_day=SimpleNamespace(case=cases[i])))
        path = tmp_path/f"{i+1}.pkl.gz"
        with gzip.open(path, "wb") as stream:
            pickle.dump(artifact, stream)
        entries.append({"update": i+1, "runtime_year": 1990+i, "path": path.name,
                        "trace_signature": "trace", "sha256": assets.sha256(path)})
    dataset_path = tmp_path/"dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    identity = {"dataset_manifest_sha256": assets.sha256(dataset_path), "training_contract_sha256": "contract"}
    method = {"training_plan": {"prepared_manifest_identity": identity, "training_artifact_ids": [1, 2, 3, 4]},
              "configuration": {"recurrent_window_index": 0}}
    manifest = {"schema_version": "target_oriented_prepared_manifest_v1", "status": "passed", "identity": identity,
                "entries": entries}
    manifest_path = tmp_path/"prepared.json"
    manifest_path.write_text(json.dumps(manifest))
    bundle_path = tmp_path/"bundle.pkl"
    bundle = {"method": method, "artifact_ids": [1, 2], "payloads": [{"cases": {7: value}} for value in cases[:2]]}
    with bundle_path.open("wb") as stream:
        pickle.dump(bundle, stream)
    return dict(prepared_manifest=manifest_path, dataset_manifest=dataset_path,
                input_bundle=bundle_path, head=None, method=method)


def test_four_existing_windows_keep_all_inputs_and_cpu_abstract_contract(prepared):
    result = assets.load_training_T7_cases(jax, **prepared)
    assert result["selected_artifact_ids"] == (1, 2, 3, 4)
    assert result["contract"]["light_identity_eval_shape_passed"]
    assert result["contract"]["all_five_dynamic_arguments_stacked"]
    assert len({row["case_sha256"] for row in result["window_metadata"]}) == 4
    for index, single in enumerate(result["cases"]):
        restored = jax.tree_util.tree_map(lambda value: value[index], result["stacked"])
        assert assets.tree_digest(jax, restored) == assets.tree_digest(jax, single)


def test_no_copied_window_padding():
    with pytest.raises(ValueError, match="copied"):
        assets.stack_cases(jax, (case(1), case(2), case(3), case(1)))


@pytest.mark.parametrize("change", ["shape", "dtype", "static"])
def test_B1_contract_mismatch_rejected(change):
    left, right = case(1), case(2)
    if change == "shape":
        right = right._replace(initial_continuous=np.zeros(3853, dtype=np.float64))
    elif change == "dtype":
        right = right._replace(initial_continuous=right[0].astype(np.float32))
    else:
        right = tuple(right)
    with pytest.raises(ValueError, match="contract differs"):
        assets.stack_cases(jax, (left, right))


def test_insufficient_declared_training_inputs_report_actual_count(prepared):
    path = prepared["prepared_manifest"]
    manifest = json.loads(path.read_text())
    manifest["entries"] = manifest["entries"][:3]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="only 3 existing"):
        assets.load_training_T7_cases(jax, **prepared)


def test_prepared_bytes_changed_rejected_before_unpickle(prepared):
    path = prepared["prepared_manifest"].parent/"4.pkl.gz"
    path.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="bytes differ"):
        assets.load_training_T7_cases(jax, **prepared)


def test_held_out_canonical_provenance_rejected():
    dataset = {"shards": [{"landpoint_id": "x", "year": 1999,
        "spatial_split": "validation", "temporal_split": "train"}]}
    with pytest.raises(ValueError, match="held-out access forbidden"):
        assets._training_provenance(dataset, {"landpoint_id": "x", "year": 1999}, 0, 1999)


def test_native_calendar_avoids_gregorian_leap_offset_and_source_row_assumption():
    assert assets.native_window_dates(1964, [59, 60, 365]) == ("1964-02-28", "1964-03-01", "1964-12-31")
    with pytest.raises(ValueError, match="one-based"):
        assets.native_window_dates(1964, [0])


def test_compact_B1_source_must_equal_reloaded_source(prepared):
    path = prepared["input_bundle"]
    with path.open("rb") as stream:
        bundle = pickle.load(stream)
    bundle["payloads"][1]["cases"][7] = case(99)
    with path.open("wb") as stream:
        pickle.dump(bundle, stream)
    with pytest.raises(ValueError, match="differs from its saved B1"):
        assets.load_training_T7_cases(jax, **prepared)
