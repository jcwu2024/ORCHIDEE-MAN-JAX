"""Saved metadata reader regressions using synthetic arrays, no model execution."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.dev.global_batch_stage_store import StageStore
from scripts.dev.global_server_reference import load_reference


@pytest.fixture
def scalar_snapshot(tmp_path, request):
    size = getattr(request, "param", 16)
    runtime = {"jax": "0.4.38", "jaxlib": "0.4.38", "x64_enabled": True,
               "device_details": [{"platform": "cpu"}]}
    points = [f"point{index}" for index in range(size)]
    identity = {"contract": "server-CPU-same-workload-v1", "mode": "scalar", "CPU_budget": 1,
        "validation_mode": "exploratory", "cpu_reference_role": "cpu-internal", "runtime": runtime,
        "baseline_ref": "baseline", "science_sha256": "science", "numerical_reference_sha256": "sealed",
        "precision_policy": "native", "gate": {"atol": 1e-8, "rtol": 1e-10},
        "lanes": [{"landpoint_id": point, "inputs_sha256": f"native{index}"} for index, point in enumerate(points)]}
    metric = {"samples_seconds": [1.0] * 5, "valid_point_days": size * 14, "median_seconds": 1.0}
    timing = {key: copy.deepcopy(metric) for key in (f"B{size}_warm_frozen_state_core",
        f"B{size}_preprepared_wrapper_stack_transfer_materialize")}
    gates = [{"name": f"B{size}_scalar_lane_{index}_{point}_GPU_reference", "passed": True,
              "atol": 1e-8, "rtol": 1e-10} for index, point in enumerate(points)]
    gates[6].update(passed=False, compared_leaves=4, max_abs=2e-8, max_bound_ratio=1.2,
        failures=[{"path": "[0][0][0]", "reason": "tolerance", "max_bound_ratio": 1.2}])
    gates.append({"name": f"B{size}_scalar_frozen_reset_repeat", "passed": True,
                  "atol": 1e-8, "rtol": 1e-10, "compared_leaves": 4})
    record = {"measurement_status": "complete", "evidence_class": "exploratory_performance",
        "strict_numerical_passed": False, "gates": gates, "timing": timing,
        "provenance": {"full_two_block_workload": True, "full_outputs": True, "blocks": [[2, 8], [9, 15]],
            "half_hours_per_day": 48, "point_ids": points, "unique_real_points": size,
            "frozen_reset_repeat": True, "five_or_more_synchronized_repeats": True, "full_two_block_parity": False},
        "execution_origin": {"available": True, "environment": {**runtime, "loadavg": (0.14, 0.06, 0.06)}}}
    output = tuple((({"x": np.ones((size, 1))},), {"daily": np.ones(size)}) for _ in range(2))
    StageStore(tmp_path, identity).save(f"B{size}_server_scalar_v1", {"outputs": output, "measurements": timing}, record)
    consumer = copy.deepcopy(identity)
    consumer.update(mode="pool", CPU_budget=4)
    return tmp_path, consumer


def read_internal(snapshot):
    directory, identity = snapshot
    return load_reference(directory, identity, name="B16_server_scalar_v1", points=16, cpu_reference_role="cpu-internal")


def test_saved_loadavg_tuple_and_JSON_list_admit_internal_without_promoting_science(scalar_snapshot):
    output, origin = read_internal(scalar_snapshot)
    assert len(output) == 2 and output[0][0][0]["x"].shape == (16, 1)
    assert origin["role"] == "cpu_internal_comparison" and origin["native_input_identity_verified"] is True
    with pytest.raises(ValueError, match="accepted reference"):
        load_reference(scalar_snapshot[0], scalar_snapshot[1], name="B16_server_scalar_v1", points=16)


@pytest.mark.parametrize("field,expected", [
    ("loadavg", "record.execution_origin.environment.loadavg[0]"),
    ("loadavg_length", "record.execution_origin.environment.loadavg"),
    ("loadavg_nonfinite", "record.execution_origin.environment.loadavg"),
    ("gates", "record.gates[0].passed"),
    ("timing", "record.timing.B16_warm_frozen_state_core.samples_seconds[0]"),
    ("provenance", "record.provenance.blocks[0][0]"),
    ("execution", "record.execution_origin.environment.jax"),
])
def test_other_evidence_changes_still_reject_at_exact_field(scalar_snapshot, field, expected):
    path = scalar_snapshot[0] / "stage_manifest.json"
    manifest = json.loads(path.read_text())
    record = manifest["stages"]["B16_server_scalar_v1"]
    if field == "loadavg":
        record["execution_origin"]["environment"]["loadavg"][0] = 0.15
    elif field == "loadavg_length":
        record["execution_origin"]["environment"]["loadavg"].pop()
    elif field == "loadavg_nonfinite":
        record["execution_origin"]["environment"]["loadavg"][0] = float("nan")
    elif field == "gates":
        record["gates"][0]["passed"] = False
    elif field == "timing":
        record["timing"]["B16_warm_frozen_state_core"]["samples_seconds"][0] = 2.0
    elif field == "provenance":
        record["provenance"]["blocks"][0][0] = 3
    else:
        record["execution_origin"]["environment"]["jax"] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError) as failure:
        read_internal(scalar_snapshot)
    assert str(failure.value).endswith(f"changed at {expected}")


def test_tuple_list_normalization_is_limited_to_loadavg(scalar_snapshot):
    directory, _ = scalar_snapshot
    manifest = json.loads((directory / "stage_manifest.json").read_text())
    store = StageStore(directory, manifest["identity"])
    value, record = store.load("B16_server_scalar_v1")
    record["provenance"]["blocks"][0] = (2, 8)
    store.save("B16_server_scalar_v1", value, record)
    with pytest.raises(ValueError, match=r"changed at record\.provenance\.blocks\[0\]$"):
        read_internal(scalar_snapshot)


@pytest.mark.parametrize("field", ["science_sha256", "lanes", "validation_mode"])
def test_source_identity_and_internal_consumer_role_remain_strict(scalar_snapshot, field):
    directory, original = scalar_snapshot
    identity = copy.deepcopy(original)
    if field == "lanes":
        identity["lanes"][0]["inputs_sha256"] = "different-native-input"
    elif field == "science_sha256":
        identity[field] = "different-science"
    else:
        identity[field] = "strict"
    with pytest.raises(ValueError):
        read_internal((directory, identity))


@pytest.mark.parametrize("scalar_snapshot", [64], indirect=True)
@pytest.mark.parametrize("corrupt", [None, "discount", "TRES", "step"])
def test_FULL64_scalar_parent48_reference_admits_batch48_without_discounting_parent(scalar_snapshot, corrupt):
    directory, consumer = scalar_snapshot
    original = json.loads((directory / "stage_manifest.json").read_text())["identity"]
    value, record = StageStore(directory, original).load("B64_server_scalar_v1")
    origin = {**original, "parent_allocated_CPU_request_count": 48, "billing_CPU_request_count": 48}
    resources = {"step_CPU_budget": 1, "allocated_CPU_request_count": 1,
        "parent_allocated_CPU_request_count": 48, "billing_CPU_request_count": 48,
        "Slurm": {"ORCHJAX_PARENT_CPU_COUNT": "48", "SLURM_CPUS_PER_TASK": "1",
                  "ORCHJAX_SLURM_ALLOC_TRES": "cpu=48,mem=1M"}}
    if corrupt == "discount":
        resources["billing_CPU_request_count"] = 1
    elif corrupt == "TRES":
        resources["Slurm"]["ORCHJAX_SLURM_ALLOC_TRES"] = "cpu=1,mem=1M"
    elif corrupt == "step":
        resources["step_CPU_budget"] = 48
    record["provenance"]["server_resources"] = resources
    destination = directory / "parent48"
    StageStore(destination, origin).save("B64_server_scalar_v1", value, record)
    consumer.update(mode="batch", CPU_budget=48, parent_allocated_CPU_request_count=48,
                    billing_CPU_request_count=48)
    if corrupt is not None:
        with pytest.raises(ValueError, match="CPU scalar reference"):
            load_reference(destination, consumer, name="B64_server_scalar_v1", points=64, cpu_reference_role="cpu-internal")
    else:
        output, proof = load_reference(destination, consumer, name="B64_server_scalar_v1", points=64, cpu_reference_role="cpu-internal")
        assert output[0][0][0]["x"].shape == (64, 1)
        assert proof["source_identity"]["CPU_budget"] == 1
        assert proof["source_identity"]["billing_CPU_request_count"] == 48
        assert proof["role"] == "cpu_internal_comparison"


@pytest.mark.parametrize(("parent", "declared", "tres"), [(1, None, ""), (1, None, "cpu=1"), (48, None, "cpu=48")])
def test_ordinary_scalar_metadata_can_derive_parent_without_new_launcher_env(parent, declared, tres):
    from scripts.dev.global_server_reference import _validate_scalar_parent_allocation

    origin = {"CPU_budget": 1, "parent_allocated_CPU_request_count": parent, "billing_CPU_request_count": parent}
    proof = {"server_resources": {"step_CPU_budget": 1, "allocated_CPU_request_count": 1,
        "parent_allocated_CPU_request_count": parent, "billing_CPU_request_count": parent,
        "Slurm": {"ORCHJAX_PARENT_CPU_COUNT": declared, "SLURM_CPUS_PER_TASK": "1",
                  "ORCHJAX_SLURM_ALLOC_TRES": tres}}}
    _validate_scalar_parent_allocation(origin, proof)
