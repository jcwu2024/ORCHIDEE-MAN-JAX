"""Runner wiring and strict initial rejection, using only small fake executables."""

import json
import subprocess
import sys
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_global_gpu_lifecycle_observer import make_measure_inputs
from test_global_gpu_lifecycle_science import fake_identity, fake_source
from test_global_gpu_lifecycle_validation import make_stage

from scripts.dev import benchmark_global_batch as actual_base
from scripts.dev import benchmark_global_gpu_capacity as capacity
from scripts.dev import benchmark_global_gpu_lifecycle as runner
from scripts.dev import global_lifecycle_science as science
from scripts.dev import validate_global_gpu_lifecycle as validator


def test_import_and_cli_parser_do_not_import_jax_or_accept_other_shapes():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from scripts.dev import benchmark_global_gpu_lifecycle; import sys; print('jax' in sys.modules)",
        ],
        cwd=runner.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
    common = [
        "--prepared-input",
        "input",
        "--full-reference-dir",
        "reference",
        "--memory-baseline-dir",
        "memory",
        "--b1024-evidence-dir",
        "history",
        "--output-dir",
        "destination",
        "--expected-code-sha",
        science.SCIENCE_PIN,
        "--measurement-strategy",
        "phased_release",
    ]
    for shape in (1280, 1536, 2048):
        with pytest.raises(SystemExit):
            runner.parser().parse_args([*common, "--stage-shape", str(shape)])
    with pytest.raises(SystemExit):
        runner.parser().parse_args(common)


def test_validation_capture_has_no_hot_transfer_and_clears_device_roots():
    transfers = []
    capture = runner.ValidationCapture(
        SimpleNamespace(device_get=lambda value: transfers.append(value) or value.copy())
    )
    state = np.array([1.0])
    first = (state, np.array([2.0]))
    checked = capture.executable_factory(lambda *arguments: first)
    for _ in range(2):
        checked(np.array([0.0]))
        checked(state)
    assert transfers == []
    capture.sink("initial_output", "host-output")
    assert len(transfers) == 1
    assert capture.pending == [] and capture.first_state is None and not capture.active
    for _ in range(4):
        checked(state)
    assert capture.pending == []


def test_finite_initial_reference_mismatch_stops_before_warm_or_seal(tmp_path, monkeypatch):
    trace, args, recorder, expanded, others, reference, scale, memories = make_measure_inputs(monkeypatch, tmp_path)
    # The native float64 field changes only; all shapes/discrete fields remain
    # matched. The legacy cross-backend gate permits this exploratory failure.
    reference[0][1]["native_float64"][0, 0] += 1.0
    legacy = SimpleNamespace(result={"gates": []}, sentinel_paths={})
    original = dict(reference[0][1])
    original["native_float64"] = reference[0][1]["native_float64"].copy()
    original["native_float64"][0, 0] -= 1.0
    gate = capacity.gate(
        legacy, "legacy-finite-only", (reference[0][0], original), reference[0], scale.base.compare, cross_backend=True
    )
    assert not gate["passed"] and gate["exploratory_continuation"]

    identity = fake_identity("phased_release")
    identity["run_binding"]["window_start_epoch"] = time.time()
    args.output_dir = tmp_path / "stage"
    args.output_dir.mkdir()
    args.budget_seconds = 7200
    args.expected_code_sha = science.SCIENCE_PIN
    args.full_reference_dir = tmp_path / "reference"
    recorder.result["environment"] = {"hostname": "fixture-node"}
    recorder.flush = lambda: None
    scale.base.roughheight_sentinel_contract = lambda lane: {}
    monkeypatch.setattr(runner, "runtime_admission", lambda *a: identity["run_binding"]["physical_gpu"])
    monkeypatch.setattr(runner, "build_identity", lambda *a: identity)
    monkeypatch.setattr(science, "load_comparison_reference", lambda *a: (reference, {"sentinel_paths": {}}))
    monkeypatch.setattr(runner, "historical_memory_admission", lambda *a: (memories[1024], {}))
    # The thin fake helper has ten native leaves, so760 schema admission is
    # independently tested with complete synthetic760 trees in validator tests.
    monkeypatch.setattr(runner, "native_schema", lambda *a: [])
    monkeypatch.setenv("ORCHJAX_GLOBAL_BATCH_PROCESS_START_EPOCH", str(time.time()))
    with pytest.raises(ValueError, match="FULL64 tolerance"):
        runner.execute_stage(args, recorder, expanded[:64], others[:64], scale, fake_source())
    assert not any(
        phase["phase"].startswith(
            (
                "capacity_B1024_core_",
                "capacity_B1024_wrapper_",
                "capacity_B1024_permutation",
                "capacity_B1024_replacement",
            )
        )
        for phase in recorder.result["phases"]
    )
    assert recorder.result["gates"] == []
    manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
    assert manifest["stages"][science.STAGE_NAME]["status"] == "failed"
    assert not (args.output_dir / f"stage_{science.STAGE_NAME}.pkl").exists()
    assert (args.output_dir / "failed_initial_arrays.pkl").exists()


def test_same_uuid_memory_observer_fails_closed_before_publication(tmp_path):
    selected = {"uuid": "expected", "total_bytes": 32 * 1024**3}
    recorder = SimpleNamespace(result={}, phase=lambda *a, **k: None)
    scale = SimpleNamespace(
        visible_gpu_memory=lambda: {"uuid": "wrong", "total_bytes": 32 * 1024**3, "applicable": True},
        base=SimpleNamespace(rss_bytes=lambda: 1),
    )
    with pytest.raises(ValueError, match="UUID"):
        runner.memory_observer(recorder, scale, selected)("wrapper_sample_start", {"logical_root_owned": False})
    assert "lifecycle_memory_observations" not in recorder.result


@pytest.mark.parametrize("strategy", science.STRATEGIES)
def test_runner_seals_and_independently_validates_complete_fake_stage(tmp_path, monkeypatch, strategy):
    """Exercise real runner/store/validator interfaces without physical model calls."""
    refdir = tmp_path / "reference"
    template, value, record, report = make_stage(tmp_path / "template", refdir, monkeypatch, strategy)
    identity = template.identity
    identity["run_binding"]["window_start_epoch"] = time.time()
    record["startup"] = {}
    args = SimpleNamespace(
        output_dir=tmp_path / "runner-stage",
        full_reference_dir=refdir,
        measurement_strategy=strategy,
        expected_code_sha=science.SCIENCE_PIN,
        budget_seconds=7200,
    )
    recorder = SimpleNamespace(
        result={
            "environment": report["environment"],
            "server_resources": report["server_resources"],
            "phases": [],
        }
    )
    recorder.phase = lambda name, **fields: recorder.result["phases"].append({"phase": name, **fields})
    recorder.flush = lambda: (args.output_dir / "report.json").write_text(json.dumps(recorder.result), encoding="utf-8")
    physical = report["environment"]["physical_gpu"]
    sentinels = record["provenance"]["continuous_state_binding"]
    # Native reference admission and actual saved-array validation remain real.
    _, reference_origin = science.load_comparison_reference(refdir, identity)
    scale = SimpleNamespace(
        jax=SimpleNamespace(device_get=deepcopy),
        visible_gpu_memory=lambda: physical,
        base=SimpleNamespace(
            rss_bytes=lambda: 1024, roughheight_sentinel_contract=lambda lane: reference_origin["sentinel_paths"]
        ),
    )
    monkeypatch.setattr(runner, "runtime_admission", lambda *a: physical)
    monkeypatch.setattr(runner, "build_identity", lambda *a: identity)
    monkeypatch.setattr(
        runner,
        "historical_memory_admission",
        lambda *a: (
            record["provenance"]["admission_compiler_memory_records"]["1024"],
            {},
        ),
    )
    monkeypatch.setenv("ORCHJAX_GLOBAL_BATCH_PROCESS_START_EPOCH", str(time.time()))
    arrays = value["validation_arrays"]
    for key in ("initial_output", "frozen_reset_output", "permutation_output", "replacement_output"):
        arrays[key] = runner.restore_runtime_reference_containers(arrays[key])
    value["outputs"] = arrays["frozen_reset_output"]

    def fake_measure(*positional, memory_observer, validation_sink, validation_executable_factory):
        assert positional[0] == 1024 and positional[-1] == 2048
        runtime_reference = positional[5]
        from jax_orchidee.stomate.modelout import ModeloutResult

        assert type(runtime_reference[0][1]["modelout"]) is ModeloutResult
        assert type(arrays["initial_output"][0][1]["modelout"]) is ModeloutResult
        # This is the actual capacity/base initial-gate comparator and JAX
        # treedef, including the production NamedTuple class, not a stub.
        for index in (0, 1, 63, 64, 2047):
            comparison = actual_base.compare(
                validator.map_arrays(lambda a: a[index], arrays["initial_output"]),
                validator.map_arrays(lambda a: a[index % 64], runtime_reference),
                sentinel_paths=reference_origin["sentinel_paths"],
            )
            assert comparison["passed"], comparison
        checked = validation_executable_factory(lambda state: (state, None))
        for handoff in arrays["handoffs"]:
            checked(handoff["first_output_state"])
            checked(handoff["second_input_state"])
        for event in ("initial_output", "frozen_reset_output", "permutation_output", "replacement_output"):
            validation_sink(event, arrays[event])
        for event in record["provenance"]["memory_observations"]:
            memory_observer(
                event["event"],
                {key: event[key] for key in ("sample_index", "logical_root_owned", "logical_root_bytes")},
            )
        for phase in report["phases"]:
            recorder.phase(phase["phase"], sample_index=phase["sample_index"])
        return value, record

    monkeypatch.setattr(capacity, "measure_shape", fake_measure)
    lanes = [SimpleNamespace(landpoint_id=lane["landpoint_id"]) for lane in identity["lanes"]]
    receipt = runner.execute_stage(args, recorder, lanes, lanes, scale, fake_source())
    acceptance = json.loads((args.output_dir / "array_acceptance.json").read_text())
    assert acceptance["native_leaves"] == 760 and acceptance["gate_count"] == 2051
    assert acceptance["FULL64_lane_comparisons"] == 2048 and acceptance["continuous_handoffs"] == 2
    assert record["provenance"]["continuous_state_binding"] == sentinels
    assert receipt["sha256"] == science.digest_file(args.output_dir / f"stage_{science.STAGE_NAME}.pkl")[0]
    # Repeat the public validator at the final report state, exercising receipt hashes.
    final_receipt = validator.validate_stage_receipt(
        args.output_dir, strategy=strategy, expected_code_sha=science.SCIENCE_PIN, reference_dir=refdir
    )
    assert final_receipt["report_sha256"] == science.digest_file(args.output_dir / "report.json")[0]


def test_runtime_reference_restores_exact_native_treedef_without_array_changes():
    from collections import namedtuple

    from jax_orchidee.stomate.modelout import ModeloutResult

    array = np.zeros((1, 14), np.float64)
    inert = science.ModeloutResult(*(array for _ in range(4)))
    native = ModeloutResult(*(array for _ in range(4)))
    assert not actual_base.compare(native, inert)["passed"]  # Reproduce original class-identity refusal.
    restored = runner.restore_runtime_reference_containers({"nested": [inert]})["nested"][0]
    assert type(restored) is ModeloutResult and all(a is b for a, b in zip(restored, inert, strict=True))
    assert actual_base.compare(native, restored)["passed"]
    WrongResult = namedtuple("WrongResult", ModeloutResult._fields)
    with pytest.raises(ValueError, match="undeclared NamedTuple"):
        runner.restore_runtime_reference_containers(WrongResult(*inert))
