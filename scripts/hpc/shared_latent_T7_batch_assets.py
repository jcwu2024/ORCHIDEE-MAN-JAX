"""Read four existing TRAIN T7 inputs; no Teacher generation or model tracing.

Every five-part public case is kept intact, including retained arrays, masks,
static physical input arrays and loss context. Parameters are supplied by the
existing pair owner and are never initialized or changed here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(jax, tree):
    digest = hashlib.sha256(str(jax.tree_util.tree_structure(tree)).encode())
    for value in jax.tree_util.tree_leaves(tree):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def case_contract(jax, case):
    """Exact pytree (including static metadata), shapes and dtypes of all inputs."""
    if len(case) != 5:
        raise ValueError("a public recurrent case must have five arguments")
    paths, tree = jax.tree_util.tree_flatten_with_path(case)
    if not paths:
        raise ValueError("recurrent case is empty")
    rows = []
    for path, value in paths:
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise ValueError("recurrent case leaves must be numeric arrays")
        rows.append({"path": jax.tree_util.keystr(path), "shape": list(array.shape),
                     "dtype": str(array.dtype)})
    return tree, rows


def stack_cases(jax, cases, *, reference_case=None):
    """Stack every dynamic leaf while preserving the exact static tree."""
    cases = tuple(cases)
    if not cases:
        raise ValueError("at least one distinct recurrent case required")
    reference = cases[0] if reference_case is None else reference_case
    expected_tree, expected_rows = case_contract(jax, reference)
    fingerprints = []
    for case in cases:
        tree, rows = case_contract(jax, case)
        if tree != expected_tree or rows != expected_rows:
            raise ValueError("recurrent case shape/dtype/static contract differs from B1")
        fingerprints.append(tree_digest(jax, case))
    if len(set(fingerprints)) != len(cases):
        raise ValueError("copied recurrent windows are forbidden")
    # NumPy retains the prepared dtypes even before the GPU runtime enables x64.
    return jax.tree_util.tree_map(lambda *values: np.stack(values, axis=0), *cases)


def _selected_record(artifact, window_index):
    rows = [(record, int(start))
            for record in artifact.schedule_entry["components"]["seven_day"]
            for start in record["start_rows"]]
    if not 0 <= window_index < len(rows):
        raise ValueError("declared window index absent from prepared schedule")
    return rows[window_index]


def _training_provenance(dataset, record, start, runtime_year):
    matches = [row for row in dataset["shards"]
               if str(row["landpoint_id"]) == str(record["landpoint_id"])
               and int(row["year"]) == int(record["year"])]
    if len(matches) != 1:
        raise ValueError("window has no unique canonical v6 shard")
    shard = matches[0]
    if shard["spatial_split"] != "train" or shard["temporal_split"] != "train":
        raise ValueError("T7 batching requires train/train windows; held-out access forbidden")
    if (int(record["year"]) != runtime_year or start < 0
            or start + 7 > int(shard["day_count"])
            or record["parent_shard_sha256"] != shard["shard_sha256"]
            or record["typed_shard_sha256"] != shard["typed_shard_sha256"]):
        raise ValueError("prepared window canonical shard identity/range differs")
    return shard


def native_window_dates(year, day_indices):
    """One-based model days in the fixed 365-day paper noleap calendar.

    Source: daily_markov_contract.py forcing_record_indices_for_day:1637-1647;
    teacher_shards.py load_plan:309-318 enforces the 365-day noleap upper bound.
    ISO labels use model month/day, never Gregorian leap-day arithmetic.
    """
    labels = []
    for value in day_indices:
        day = int(value)
        if day != value or not 1 <= day <= 365:
            raise ValueError("one-based 365-day noleap model day required")
        month_day = date(2001, 1, 1) + timedelta(days=day-1)
        labels.append(f"{int(year):04d}-{month_day.month:02d}-{month_day.day:02d}")
    return tuple(labels)


def load_training_T7_cases(jax, *, prepared_manifest, dataset_manifest,
                           input_bundle, head, method, count=4):
    """Select the first four declared training diagnostics at their window0.

    Existing prepared arrays are adapted by public prepare_shared_window, never
    reconstructed from Teacher files. The two compact-bundle cases must match
    their independently reloaded prepared sources byte for byte.
    """
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import prepare_shared_window

    if count != 4:
        raise ValueError("this bounded diagnostic requires exactly four windows")
    prepared_manifest, dataset_manifest, input_bundle = map(
        Path, (prepared_manifest, dataset_manifest, input_bundle))
    manifest = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    plan = method["training_plan"]
    if (manifest.get("schema_version") != "target_oriented_prepared_manifest_v1"
            or manifest.get("status") != "passed"
            or manifest.get("identity") != plan["prepared_manifest_identity"]
            or sha256(dataset_manifest) != manifest["identity"]["dataset_manifest_sha256"]):
        raise ValueError("prepared training manifest/dataset identity differs")
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != "daily_teacher_dataset_manifest_v6" or dataset.get("status") != "complete":
        raise ValueError("complete canonical v6 manifest required")
    registry = {int(row["update"]): row for row in manifest["entries"]}
    if len(registry) != len(manifest["entries"]):
        raise ValueError("duplicate prepared artifact ids")
    declared = tuple(int(value) for value in plan["training_artifact_ids"])
    if len(set(declared)) != len(declared):
        raise ValueError("duplicate declared training artifact ids")
    available = [identifier for identifier in declared if identifier in registry]
    if len(available) < count:
        raise ValueError(f"only {len(available)} existing declared training artifacts available; four required")
    identifiers = declared[:count]
    if any(identifier not in registry for identifier in identifiers):
        raise ValueError("one of the first four declared training artifacts is absent")
    with input_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    if (bundle.get("method") != method
            or tuple(bundle.get("artifact_ids", ())) != declared[:2]
            or len(bundle.get("payloads", ())) != 2):
        raise ValueError("B1 compact bundle training provenance differs")
    window_index = int(method["configuration"]["recurrent_window_index"])
    if window_index != 0:
        raise ValueError("declared diagnostic window0 required")
    cases, metadata, semantic_days = [], [], set()
    for identifier in identifiers:
        entry = registry[identifier]
        path = (prepared_manifest.parent / entry["path"]).resolve()
        path.relative_to(prepared_manifest.parent.resolve())
        if sha256(path) != entry["sha256"]:
            raise ValueError("prepared artifact bytes differ from immutable manifest")
        with gzip.open(path, "rb") as stream:
            artifact = pickle.load(stream)
        if (int(artifact.schedule_entry["update"]) != identifier
                or int(artifact.schedule_entry["runtime_year"]) != int(entry["runtime_year"])
                or artifact.trace_signature != entry["trace_signature"]
                or artifact.training_contract_sha256 != manifest["identity"]["training_contract_sha256"]):
            raise ValueError("prepared artifact schedule/contract differs")
        record, start = _selected_record(artifact, window_index)
        shard = _training_provenance(dataset, record, start, int(entry["runtime_year"]))
        days = tuple((str(record["landpoint_id"]), int(record["year"]), start + day) for day in range(7))
        if semantic_days.intersection(days):
            raise ValueError("independent training windows overlap in Teacher-day identity")
        semantic_days.update(days)
        case = jax.device_get(prepare_shared_window(
            head, artifact.batch.seven_day, window_index=window_index, horizon=7))
        if np.shape(case[0]) != (3854,) or np.shape(case.days[0].continuous_state)[0] != 7 or np.shape(case.targets["state"])[0] != 7:
            raise ValueError("exact seven-day native public case required")
        indices = np.asarray(case.days[0].day_index).reshape(-1)
        if indices.size != 7 or not np.all(np.diff(indices) == 1):
            raise ValueError("actual prepared day indices are not seven consecutive days")
        native_dates = native_window_dates(record["year"], indices)
        fingerprint = tree_digest(jax, case)
        if identifier in bundle["artifact_ids"]:
            previous = bundle["payloads"][bundle["artifact_ids"].index(identifier)]["cases"][7]
            if fingerprint != tree_digest(jax, previous):
                raise ValueError("public prepared window differs from its saved B1 case")
        metadata.append({"artifact_id": identifier, "window_index": window_index,
            "horizon": 7, "landpoint_id": str(record["landpoint_id"]),
            "year": int(record["year"]), "start_row": start, "stop_row_exclusive": start+7,
            "day_indices": indices.astype(int).tolist(), "teacher_day_keys": days,
            "calendar": "365_day_noleap", "day_index_base": 1,
            "model_date_labels": native_dates, "model_start_date": native_dates[0],
            "model_end_date": native_dates[-1],
            "spatial_split": "train", "temporal_split": "train",
            "prepared_artifact": str(path), "prepared_artifact_sha256": entry["sha256"],
            "parent_shard_sha256": shard["shard_sha256"],
            "typed_shard_sha256": shard["typed_shard_sha256"],
            "initial_state_source": "prepared seven_day.inputs.continuous_state[window0,day0]",
            "case_sha256": fingerprint})
        cases.append(case)
    reference = bundle["payloads"][0]["cases"][7]
    stacked = stack_cases(jax, cases, reference_case=reference)
    # Small identity-only abstract trace checks the complete B4 tree; no model.
    abstract = jax.tree_util.tree_map(lambda value: jax.ShapeDtypeStruct(np.shape(value), np.asarray(value).dtype), stacked)
    checked = jax.eval_shape(lambda value: value, abstract)
    for expected, actual in zip(jax.tree_util.tree_leaves(abstract), jax.tree_util.tree_leaves(checked), strict=True):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ValueError("CPU abstract contract changed an input dtype/shape; enable JAX x64")
    _, rows = case_contract(jax, reference)
    return {"cases": tuple(cases), "stacked": stacked, "window_metadata": tuple(metadata),
            "contract": {"B1_leaves": rows, "B4_leaf_count": len(jax.tree_util.tree_leaves(checked)),
                         "batch_size": count, "all_five_dynamic_arguments_stacked": True,
                         "static_pytree_equal_to_B1": True, "light_identity_eval_shape_passed": True},
            "selected_artifact_ids": identifiers, "available_declared_training_artifacts": len(available),
            "prepared_manifest_sha256": sha256(prepared_manifest),
            "dataset_manifest_sha256": sha256(dataset_manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-manifest", "dataset-manifest", "input-bundle", "prepared-context", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    root = os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, root)
    import jax
    if any(device.platform != "cpu" for device in jax.local_devices()):
        raise ValueError("asset inspector requires JAX_PLATFORMS=cpu")
    jax.config.update("jax_enable_x64", True)
    with args.prepared_context.open("rb") as stream:
        context = pickle.load(stream)
    with args.input_bundle.open("rb") as stream:
        bundle = pickle.load(stream)
    assets = load_training_T7_cases(jax, prepared_manifest=args.prepared_manifest,
        dataset_manifest=args.dataset_manifest, input_bundle=args.input_bundle,
        head=context["head"], method=bundle["method"])
    report = {key: value for key, value in assets.items() if key not in ("cases", "stacked")}
    report.update(status="passed", scope="CPU_existing_training_input_contract_only",
                  gpu_executed=False, model_traced=False, scientific_acceptance=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "artifact_ids": assets["selected_artifact_ids"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
