"""Saved-result collector regressions with tiny arrays; no model or JIT calls."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.dev import benchmark_global_batch as base
from scripts.dev.global_server_reference import finite_tolerance_failure

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load("continuation_collector", HERE / "collect_continuation_result.py")
old_tests = load(
    "continuation_old_test_helpers", HERE.parent / "gpu-sweetspot-preparation-20261006/test_collect_capacity_result.py"
)
original_fixture = old_tests.fixture
old_tests.collector = collector


def fixture(directory, tolerance=False, size=256):
    identity, _, envelope, reference = original_fixture(directory, tolerance)
    name = f"capacity_B{size}_fixed512_v1"
    envelope["name"] = name
    proof = envelope["record"]["provenance"]
    proof.update(chunk_size=size, chunks_per_collection=512 // size)
    for gate in envelope["record"]["gates"]:
        gate["name"] = gate["name"].replace("capacity_B128", f"capacity_B{size}")
    return identity, name, envelope, reference


old_tests.fixture = fixture
# Exercise all existing scientific/schema/native/daily/gate/SHA/sample/sentinel
# and narrow JSON-record checks through the continuation wrapper, not shortcuts.
for _name in dir(old_tests):
    if _name.startswith("test_") and _name != "test_terminal_cost_counts_parent_only_and_never_running_job":
        globals()[_name] = getattr(old_tests, _name)


def identity_fixture():
    source = {
        "lanes": [{"landpoint_id": f"point{i}", "inputs_sha256": str(i)} for i in range(64)],
        "science_sha256": "science",
        "baseline_ref": "fortran",
        "precision_policy": "native",
        "gate": collector.GATE,
    }
    identity = {
        **source,
        "contract": collector.CONTRACT,
        "validation_mode": "exploratory",
        "workload_kind": "capacity_screen_repeated_FULL64",
        "total_lanes": 512,
        "unique_real_points": 64,
        "candidate_shapes": [256, 512],
        "continuation_B128_sha256": collector.B128_SHA,
        "conditional_512": True,
        "optional_B64_baseline": False,
        "warm_repeats": 5,
        "conditional_policy": "scale_curve_30min_B256compiler_memory_2GiB_GPU_8GiB_host_slack",
        "numerical_reference_sha256": "reference",
    }
    identity["lane_origins"] = [
        {
            "lane_index": i,
            "source_index": i % 64,
            "landpoint_id": source["lanes"][i % 64]["landpoint_id"],
            "repeat_index": i // 64,
        }
        for i in range(512)
    ]
    admission = {"source_identity": source, "FULL64_reference_origin": {"sha256": "reference"}}
    return identity, admission


@pytest.mark.parametrize(
    "key,value",
    [
        ("contract", collector.OLD_CONTRACT),
        ("candidate_shapes", [128, 256]),
        ("candidate_shapes", [512, 256]),
        ("candidate_shapes", [256, 512, 64]),
        ("continuation_B128_sha256", "wrong"),
        ("optional_B64_baseline", True),
        ("conditional_512", False),
        ("conditional_policy", "speedup5percent"),
        ("science_sha256", "changed"),
        ("warm_repeats", 4),
    ],
)
def test_exact_continuation_identity_rejects_changes(key, value):
    identity, admission = identity_fixture()
    collector.identity_checks(identity, admission)
    identity[key] = value
    with pytest.raises(ValueError):
        collector.identity_checks(identity, admission)


@pytest.mark.parametrize("size", [256, 512])
def test_both_candidate_stage_shapes_use_full_existing_checks(tmp_path, size):
    identity, name, envelope, reference = fixture(tmp_path, size=size)
    proof = envelope["record"]["provenance"]
    proof["server_resources"].update(hostname=collector.NODE)
    proof["server_resources"]["Slurm"].update(SLURM_JOB_NODELIST=collector.NODE, SLURM_JOB_GPUS="4")
    envelope["record"]["execution_origin"]["environment"].update(
        hostname=collector.NODE, physical_gpu={"uuid": collector.GPU_UUID, "driver_version": "525.85.12"}
    )
    record = old_tests.save(tmp_path, envelope)
    result = collector.validate_stage(
        tmp_path, name, identity, record, reference, base, finite_tolerance_failure, leaf_count=4
    )
    assert result["batch_shape"] == size and len(result["actual_saved_output_reference_comparisons"]) == 512
    assert result["numerical_validation_class"] == "strict_saved_numerical_pass"


@pytest.mark.parametrize("fault", ["job", "code", "contract", "node", "gpu", "B128_stage"])
def test_new_stage_origin_rejects_wrong_job_code_contract_node_gpu_or_relabel(tmp_path, fault):
    identity, name, envelope, reference = fixture(tmp_path)
    if fault == "job":
        envelope["record"]["provenance"]["server_resources"]["Slurm"]["SLURM_JOB_ID"] = collector.OLD_JOB
    elif fault == "code":
        envelope["record"]["execution_origin"]["environment"]["git_head"] = collector.OLD_CODE_SHA
    elif fault == "contract":
        identity["contract"] = collector.OLD_CONTRACT
    elif fault == "node":
        envelope["record"]["execution_origin"]["environment"]["hostname"] = "other-node"
    elif fault == "gpu":
        envelope["record"]["execution_origin"]["environment"]["physical_gpu"] = {"uuid": "other"}
    else:
        name = "capacity_B128_fixed512_v1"
        envelope["name"] = name
    record = old_tests.save(tmp_path, envelope)
    with pytest.raises(ValueError):
        collector.validate_stage(
            tmp_path, name, identity, record, reference, base, finite_tolerance_failure, leaf_count=4
        )


def baseline_fixture(identity):
    original_identity = {
        **identity,
        "contract": collector.OLD_CONTRACT,
        "candidate_shapes": [128, 256],
        "conditional_policy": "core_and_wrapper_5percent_30min_2GiBslack",
    }
    original_identity.pop("continuation_B128_sha256")
    return {
        "path": str(collector.BASELINE_FILE),
        "manifest": str(collector.BASELINE_DIR / "stage_manifest.json"),
        "sha256": collector.B128_SHA,
        "bytes": 220429366,
        "source_stage": "capacity_B128_fixed512_v1",
        "manifest_sha256": "sealedmanifest",
        "source_identity": original_identity,
        "execution_origin": {"environment": {"git_head": collector.OLD_CODE_SHA, "loadavg": (0.1, 0.2, 0.3)}},
        "timing": {"core": {"samples_seconds": [1] * 5}, "wrapper": {"samples_seconds": [2] * 5}},
        "provenance": {"server_resources": {"Slurm": {"SLURM_JOB_ID": collector.OLD_JOB}}},
        "role": "imported historical baseline; no new B128 measurement, executable, or destination stage",
    }


@pytest.mark.parametrize(
    "fault", [None, "SHA", "bytes", "source_identity", "oldjob", "oldcode", "role", "timing", "path"]
)
def test_sealed_B128_origin_is_imported_without_new_job_relabel(fault):
    identity, _ = identity_fixture()
    validated = baseline_fixture(identity)
    origin = copy.deepcopy(validated)
    origin["execution_origin"]["environment"]["loadavg"] = [0.1, 0.2, 0.3]
    calls = []

    def importer(directory, expected, destination, jax):
        calls.append((directory, expected, destination))
        assert expected == validated["source_identity"]
        return validated["timing"], validated

    capacity = SimpleNamespace(import_b128_baseline=importer)
    if fault == "SHA":
        origin["sha256"] = "wrong"
    elif fault == "bytes":
        origin["bytes"] -= 1
    elif fault == "source_identity":
        origin["source_identity"]["science_sha256"] = "changed"
    elif fault == "oldjob":
        origin["provenance"]["server_resources"]["Slurm"]["SLURM_JOB_ID"] = collector.JOB
    elif fault == "oldcode":
        origin["execution_origin"]["environment"]["git_head"] = collector.CODE_SHA
    elif fault == "role":
        origin["role"] = "newB128stage"
    elif fault == "timing":
        origin["timing"]["core"]["samples_seconds"][0] = 99
    elif fault == "path":
        origin["path"] = "other.pkl"
    if fault is None:
        result = collector.validate_imported_baseline(origin, identity, capacity, base.jax)
        assert result["execution_origin"]["environment"]["git_head"] == collector.OLD_CODE_SHA
        assert result["provenance"]["server_resources"]["Slurm"]["SLURM_JOB_ID"] == collector.OLD_JOB
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError):
            collector.validate_imported_baseline(origin, identity, capacity, base.jax)


def test_parent_accounting_uses_exact_new_parent_allocation_node_and_terminal_only():
    parent = f"{collector.JOB}|COMPLETED|3600|4|cpu=4,gres/gpu=1|0:0|{collector.NODE}\n"
    child = f"{collector.JOB}.batch|COMPLETED|3600|4|cpu=4|0:0|{collector.NODE}\n"
    result = collector.parent_accounting(parent + child)
    assert result["whole_job_cost"]["primary"]["CNY"] == pytest.approx(2.4)
    assert result["whole_job_cost"]["conservative"]["CNY"] == pytest.approx(2.48)
    assert collector.parent_accounting(parent.replace("COMPLETED", "RUNNING"))["whole_job_cost"] is None
    for bad in (
        parent + parent,
        parent.replace(collector.JOB, collector.OLD_JOB),
        parent.replace(collector.NODE, "other-node"),
        parent.replace("cpu=4", "cpu=8"),
    ):
        with pytest.raises(ValueError):
            collector.parent_accounting(bad)


def test_legacy_source_loaded_in_isolated_module_without_changing_original():
    original = load("untouched_old_collector", collector.legacy_path)
    assert original.JOB == collector.OLD_JOB and original.CODE_SHA == collector.OLD_CODE_SHA
    assert original.CONTRACT == collector.OLD_CONTRACT and collector.legacy.JOB == collector.JOB
    assert collector.LEGACY_SHA == collector.sha(collector.legacy_path)
