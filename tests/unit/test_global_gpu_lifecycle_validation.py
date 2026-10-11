"""Small synthetic sealed trees exercise actual760/2051/2048 array admission."""

import hashlib
import json
import pickle
from copy import deepcopy

import numpy as np
import pytest
from test_global_gpu_lifecycle_science import fake_identity, fake_source

from scripts.dev import benchmark_global_gpu_capacity as capacity
from scripts.dev import global_lifecycle_science as science
from scripts.dev import validate_global_gpu_lifecycle as validator
from scripts.dev.global_batch_stage_store import StageStore


def output_tree(count):
    values = (np.arange(count) % 64).astype(np.float64)[:, None]
    rough = np.ones((count, 1, 14))
    rough[..., 0] = np.nan
    outputs = {f"v{i:03d}": values for i in range(376)}
    for index in range(4):
        del outputs[f"v{index:03d}"]
    outputs["modelout"] = science.ModeloutResult(*(values for _ in range(4)))
    outputs["bool_mask"] = values % 2 == 0
    outputs["native_float32"] = values.astype(np.float32)
    return (((values, rough), outputs), ((values + 1, rough), outputs))


def sentinel_contract():
    return {
        f"[{block}][0][1]": {
            "field": "roughheight_pft",
            "component": component,
            "bare_soil_index": 0,
            "inner_shape": [1, 14],
            "provenance": "declared synthetic test reference",
        }
        for block, component in enumerate(("diffuco_previous_step_state", "sechiba_finalize_state"))
    }


def make_reference(directory, identity, monkeypatch):
    directory.mkdir()
    output, sentinels = output_tree(64), sentinel_contract()
    old = {**identity, "science_sha256": science.OLD_SCIENCE_SHA256, "validation_mode": "strict"}
    proof = {"own_inputs": True, "point_ids": [lane["landpoint_id"] for lane in identity["lanes"]]}
    gates = [
        {
            "passed": True,
            "allowed_sentinels": [{"path": path, "contract": contract} for path, contract in sentinels.items()],
        }
    ]
    record = {"provenance": proof, "gates": gates, "evidence_class": "accepted"}
    envelope = {
        "identity": old,
        "name": science.REFERENCE_STAGE,
        "status": "complete",
        "record": record,
        "value": output,
    }
    data = pickle.dumps(envelope)
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(science, "REFERENCE_SHA256", sha)
    identity["comparison_reference"] = science.comparison_admission()
    path = directory / f"stage_{science.REFERENCE_STAGE}.pkl"
    path.write_bytes(data)
    (directory / "stage_manifest.json").write_text(
        json.dumps(
            {
                "identity": old,
                "stages": {science.REFERENCE_STAGE: {"status": "complete", "path": path.name, "sha256": sha, **record}},
            }
        ),
        encoding="utf-8",
    )
    return output, sentinels


def make_stage(directory, reference_dir, monkeypatch, strategy="phased_resident_control"):
    identity = fake_identity(strategy)
    reference, sentinels = make_reference(reference_dir, identity, monkeypatch)
    monkeypatch.setattr(science, "source_identity", lambda *a, **k: fake_source())
    initial = output_tree(2048)
    reset = initial
    permutation = validator.map_arrays(lambda a: a[np.arange(1023, -1, -1)], initial)
    replacement = validator.map_arrays(lambda a: a[np.array(list(range(1023)) + [0])], initial)
    handoffs = [
        {
            "first_output_state": validator.map_arrays(lambda a: a[start : start + 1024], initial[0][0]),
            "second_input_state": validator.map_arrays(lambda a: a[start : start + 1024], initial[0][0]),
        }
        for start in (0, 1024)
    ]
    names = [f"capacity_B1024_lane_{i}_source_{i % 64}_GPU_reference" for i in range(2048)]
    gates = [
        {"name": name, **gate}
        for name, gate in zip(names, validator.compare_reference_lanes(initial, reference, sentinels), strict=True)
    ]
    for suffix, actual, expected in (
        ("frozen_reset_repeat", reset, initial),
        ("permutation", permutation, permutation),
        ("replacement_isolation", replacement, replacement),
    ):
        gates.append(
            {"name": f"capacity_B1024_{suffix}", **validator.compare_tree(actual, expected, sentinel_paths=sentinels)}
        )
    measurements = {key: capacity.lane_metrics([1.0, 2.0, 3.0, 4.0, 5.0], 2048) for key in ("core", "wrapper")}
    physical = {**identity["run_binding"]["physical_gpu"], "free_bytes": 30 * 1024**3, "applicable": True}
    environment = {
        **identity["runtime"],
        "git_head": science.SCIENCE_PIN,
        "env": identity["compiler_environment"],
        "physical_gpu": physical,
    }
    actual, previous = {"available": True, "temp_size_in_bytes": 100}, {"available": True, "temp_size_in_bytes": 200}
    resident = 1024
    output_bytes = sum(a.nbytes for _, a in validator.flatten(initial))
    estimated = 2 * resident + 3 * output_bytes + 200
    event_order = [("core_roots_release_before", 1), ("core_roots_release_after", 1)]
    event_order += [(f"wrapper_sample_{event}", index) for index in range(1, 6) for event in ("start", "end")]
    observations = [
        {
            "event": event,
            "sample_index": sample,
            "logical_root_owned": strategy != "phased_release" or index == 0,
            "logical_root_bytes": resident if strategy != "phased_release" or index == 0 else 0,
            "host_RSS_bytes": 1024,
            "physical_gpu": physical,
            "peak_unavailable": True,
        }
        for index, (event, sample) in enumerate(event_order)
    ]
    proof = {
        "total_lanes": 2048,
        "lane_days": 28672,
        "unique_real_points": 64,
        "chunk_size": 1024,
        "chunks_per_collection": 2,
        "blocks": [[2, 8], [9, 15]],
        "half_hours_per_day": 48,
        "full_outputs": True,
        "full_two_block_workload": True,
        "process_strategy": science.PROCESS_STRATEGY,
        **science.measurement_fields(strategy),
        "full_output_native_schema": validator.native_schema(initial, reference, 2048),
        "continuous_state_binding": science.run_two_binding(validator.ROOT),
        "executable_memory_analysis": actual,
        "admission_compiler_memory_records": {"1024": previous},
        "memory_preflight": {
            "resident_2048_input_bytes": resident,
            "full_2048_output_bytes": output_bytes,
            "physical_gpu": physical,
        },
        "postcompile_memory_budget_validation": {
            "original_preallocation_physical_gpu_budget": physical,
            "postcompile_physical_gpu_snapshot": physical,
            "estimated_device_bytes": estimated,
            "permitted": True,
            "required_physical_slack_bytes": 2 * 1024**3,
            "resident_2048_input_bytes": resident,
            "full_2048_output_bytes": output_bytes,
            "actual_executable_memory_analysis": actual,
        },
        "server_resources": identity["run_binding"]["server_resources"],
        "memory_observations": observations,
    }
    value = {
        "outputs": reset,
        "measurements": measurements,
        "validation_arrays": {
            "initial_output": initial,
            "frozen_reset_output": reset,
            "permutation_output": permutation,
            "replacement_output": replacement,
            "handoffs": handoffs,
        },
    }
    record = {
        "measurement_status": "complete",
        "strict_numerical_passed": True,
        "gates": gates,
        "timing": measurements,
        "provenance": proof,
        "execution_origin": {"available": True, "pid": 1, "process_start_epoch": 2.0, "environment": environment},
    }
    phases = [
        {"phase": f"capacity_B1024_{boundary}_{suffix}", "sample_index": sample}
        for boundary in ("core", "wrapper")
        for sample in range(1, 6)
        for suffix in ("start", "complete")
    ]
    store = StageStore(directory, identity)
    store.save(science.STAGE_NAME, value, record)
    report = {
        "status": "completed_lifecycle_diagnostic",
        "completed_shapes": [1024],
        "identity": identity,
        "environment": environment,
        "server_resources": identity["run_binding"]["server_resources"],
        "stages": store.manifest["stages"],
        "phases": phases,
        "lifecycle_memory_observations": observations,
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return store, value, record, report


def resave(store, value, record, report):
    store.save(science.STAGE_NAME, value, record)
    report["stages"] = store.manifest["stages"]
    (store.directory / "report.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize("strategy", science.STRATEGIES)
def test_actual_array_admission_complete760_2051_2048(tmp_path, monkeypatch, strategy):
    directory, refdir = tmp_path / "stage", tmp_path / "reference"
    make_stage(directory, refdir, monkeypatch, strategy)
    receipt = validator.validate_stage_receipt(
        directory, strategy=strategy, expected_code_sha=science.SCIENCE_PIN, reference_dir=refdir
    )
    acceptance = json.loads((directory / "array_acceptance.json").read_text())
    assert acceptance["native_leaves"] == 760
    assert acceptance["gate_count"] == 2051
    assert acceptance["FULL64_lane_comparisons"] == 2048
    assert acceptance["continuous_handoffs"] == 2
    assert receipt["array_acceptance_sha256"] == science.digest_file(directory / "array_acceptance.json")[0]


@pytest.mark.parametrize(
    "mutation", ["finite", "mask", "sentinel", "reset", "permutation", "replacement", "handoff", "dtype"]
)
def test_candidate_passed_booleans_cannot_hide_actual_array_failures(tmp_path, monkeypatch, mutation):
    directory, refdir = tmp_path / "stage", tmp_path / "reference"
    store, value, record, report = make_stage(directory, refdir, monkeypatch)
    arrays = value["validation_arrays"]
    if mutation == "finite":
        arrays["initial_output"][0][0][0][0, 0] += 1.0
    elif mutation == "mask":
        arrays["initial_output"][0][1]["bool_mask"][0, 0] = False
    elif mutation == "sentinel":
        arrays["initial_output"][0][0][1][0, 0, 1] = np.nan
        report["allowed_sentinel_contract"] = {"everything": "allow"}
    elif mutation == "dtype":
        initial = arrays["initial_output"]
        arrays["initial_output"] = (
            (initial[0][0], {**initial[0][1], "native_float32": initial[0][1]["native_float32"].astype(np.float64)}),
            initial[1],
        )
    elif mutation == "handoff":
        arrays["handoffs"][0]["second_input_state"] = validator.map_arrays(
            lambda a: a.copy(), arrays["handoffs"][0]["second_input_state"]
        )
        arrays["handoffs"][0]["second_input_state"][0][0, 0] += 1.0
    else:
        key = {
            "reset": "frozen_reset_output",
            "permutation": "permutation_output",
            "replacement": "replacement_output",
        }[mutation]
        arrays[key] = validator.map_arrays(lambda a: a.copy(), arrays[key])
        arrays[key][0][0][0][0, 0] += 1.0
    resave(store, value, record, report)
    with pytest.raises(ValueError):
        validator.validate_stage_receipt(
            directory, strategy="phased_resident_control", expected_code_sha=science.SCIENCE_PIN, reference_dir=refdir
        )


@pytest.mark.parametrize(
    "mutation",
    ["source", "runtime", "order", "gate_order", "root_policy", "uuid", "guard", "report", "checksum", "binding"],
)
def test_sealed_metadata_and_run_binding_mismatches_fail(tmp_path, monkeypatch, mutation):
    directory, refdir = tmp_path / "stage", tmp_path / "reference"
    store, value, record, report = make_stage(directory, refdir, monkeypatch)
    if mutation == "source":
        store.identity["actual_science_source_sha256"] = "b" * 64
    elif mutation == "runtime":
        record["execution_origin"]["environment"]["numpy"] = "wrong actual runtime"
    elif mutation == "order":
        report["phases"][0], report["phases"][2] = report["phases"][2], report["phases"][0]
    elif mutation == "gate_order":
        record["gates"][0], record["gates"][1] = record["gates"][1], record["gates"][0]
    elif mutation == "root_policy":
        record["provenance"]["memory_observations"][1]["logical_root_owned"] = False
    elif mutation == "uuid":
        record["provenance"]["memory_observations"][0]["physical_gpu"] = {"uuid": "other", "total_bytes": 32 * 1024**3}
    elif mutation == "guard":
        record["provenance"]["postcompile_memory_budget_validation"]["estimated_device_bytes"] = 1
    elif mutation == "binding":
        record["provenance"]["continuous_state_binding"]["second_block_state_argument"] = "unbound state"
    elif mutation == "report":
        report["status"] = "failed"
    resave(store, value, record, report)
    if mutation == "checksum":
        with store.checkpoint_path(science.STAGE_NAME).open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        validator.validate_stage_receipt(
            directory, strategy="phased_resident_control", expected_code_sha=science.SCIENCE_PIN, reference_dir=refdir
        )


def test_common_pair_mismatch_rejected_before_arrays(tmp_path, monkeypatch):
    directory, refdir = tmp_path / "stage", tmp_path / "reference"
    store, _, _, _ = make_stage(directory, refdir, monkeypatch, "phased_release")
    common = deepcopy(science.common_identity(store.identity))
    common["sampling_order"] = "interleaved_core_wrapper"
    with pytest.raises(ValueError, match="common runtime/source/order"):
        validator.validate_stage_receipt(
            directory,
            strategy="phased_release",
            expected_code_sha=science.SCIENCE_PIN,
            reference_dir=refdir,
            common_identity=common,
        )


def test_mixed_lane_sentinel_records_match_existing_per_lane_comparator():
    from scripts.dev.benchmark_global_batch import compare

    reference = {"rough": np.zeros((64, 1, 14))}
    reference["rough"][1::2, 0, 0] = np.nan
    outputs = {"rough": reference["rough"][np.arange(2048) % 64]}
    sentinels = {"['rough']": {"inner_shape": [1, 14], "bare_soil_index": 0}}
    records = validator.compare_reference_lanes(outputs, reference, sentinels)
    for index in (0, 1, 63, 64, 2046, 2047):
        expected = compare(
            {"rough": outputs["rough"][index]}, {"rough": reference["rough"][index % 64]}, sentinel_paths=sentinels
        )
        assert records[index] == expected


def test_inert_modelout_dto_keeps_field_schema_and_numpy_only_roundtrip():
    result = science.ModeloutResult(*(np.zeros((64, 1)) for _ in range(4)))
    decoded = science.load_array_pickle(pickle.dumps(result))
    assert validator.structure(decoded) == validator.structure(result)
    assert [path for path, _ in validator.flatten(decoded)] == [".AGB_model", ".BGB_model", ".GPP_model", ".NPP_model"]
    mapped = validator.map_arrays(lambda a: a[0], decoded)
    assert mapped.AGB_model.shape == (1,)
