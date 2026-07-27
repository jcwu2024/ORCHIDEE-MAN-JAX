from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver.paper_binding import expected_paper_domain_limits
from research.daily_coarse_graining import supervised_learnability_pilot as capture
from research.daily_coarse_graining import teacher_shards as shards


def _write_plan(tmp_path: Path, entries: list[dict], **overrides) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text("test: true\n", encoding="utf-8")
    payload = {
        "schema_version": shards.SCHEMA_VERSION,
        "dataset_id": "test-dataset",
        "teacher_config": str(config),
        "output_root": str(tmp_path / "dataset"),
        "block_size": 28,
        "entries": entries,
        **overrides,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(tmp_path: Path, landpoint: str, year: int, **overrides) -> dict:
    run_def = tmp_path / f"{landpoint}-{year}.def"
    limits = expected_paper_domain_limits(landpoint)
    run_def.write_text(
        "".join(f"{name}={value}\n" for name, value in limits.items()),
        encoding="utf-8",
    )
    reference = tmp_path / f"reference-{landpoint}"
    reference.mkdir(exist_ok=True)
    for name in shards.REFERENCE_INPUT_NAMES:
        (reference / name).write_bytes(name.encode())
    cache = tmp_path / f"cache-{landpoint}-{year}.pkl"
    cache.write_bytes(b"cache")
    return {
        "landpoint_id": landpoint,
        "year": year,
        "spatial_split": "train",
        "temporal_split": "train",
        "run_def": str(run_def),
        "reference_run_dir": str(reference),
        "state_cache": str(cache),
        **overrides,
    }


def test_plan_freezes_spatial_and_temporal_splits(tmp_path):
    first = _entry(tmp_path, "001.0-071.0", 1961)
    second = _entry(
        tmp_path,
        "001.0-071.0",
        1962,
        spatial_split="validation",
        temporal_split="validation",
    )
    with pytest.raises(ValueError, match="leaks across spatial splits"):
        shards.load_plan(_write_plan(tmp_path, [first, second]))

    second = _entry(
        tmp_path,
        "003.0-071.0",
        1961,
        spatial_split="validation",
        temporal_split="test",
    )
    with pytest.raises(ValueError, match="leaks across temporal splits"):
        shards.load_plan(_write_plan(tmp_path, [first, second]))


def test_plan_enforces_paper_noleap_calendar_in_gregorian_leap_year(tmp_path):
    entry = _entry(tmp_path, "001.0-071.0", 1964)
    plan = shards.load_plan(_write_plan(tmp_path, [entry]))
    assert plan.entries[0].days == 365

    with pytest.raises(ValueError, match="fixed 365-day noleap calendar"):
        shards.load_plan(_write_plan(tmp_path, [dict(entry, days=366)]))


def test_plan_requires_explicit_state_for_each_new_or_gapped_chain(tmp_path):
    first = _entry(tmp_path, "001.0-071.0", 1961)
    first.pop("state_cache")
    with pytest.raises(ValueError, match="requires state_cache"):
        shards.load_plan(_write_plan(tmp_path, [first]))

    first = _entry(tmp_path, "001.0-071.0", 1961)
    third = _entry(tmp_path, "001.0-071.0", 1963)
    third.pop("state_cache")
    with pytest.raises(ValueError, match="does not follow"):
        shards.load_plan(_write_plan(tmp_path, [first, third]))


def test_plan_accepts_1961_cold_start_only_at_chain_start(tmp_path):
    cold = _entry(
        tmp_path,
        "001.0-071.0",
        1961,
        initialization_mode=shards.COLD_START_BOOTSTRAP,
    )
    cold.pop("state_cache")
    acceptance = tmp_path / "accepted-1961.pkl"
    acceptance.write_bytes(b"accepted")
    cold["acceptance_checkpoint"] = str(acceptance)
    next_year = _entry(tmp_path, "001.0-071.0", 1962)
    next_year.pop("state_cache")

    plan = shards.load_plan(_write_plan(tmp_path, [cold, next_year]), require_inputs=True)
    assert plan.entries[0].initialization_mode == shards.COLD_START_BOOTSTRAP
    assert plan.entries[0].state_cache is None
    assert plan.entries[0].acceptance_checkpoint == acceptance
    assert plan.entries[1].initialization_mode == shards.YEAR_START_CHECKPOINT

    invalid = dict(cold, year=1962)
    with pytest.raises(ValueError, match="supported only for 1961"):
        shards.load_plan(_write_plan(tmp_path, [invalid]))


def test_plan_rejects_run_def_bound_to_another_landpoint(tmp_path):
    entry = _entry(tmp_path, "001.0-071.0", 1961)
    Path(entry["run_def"]).write_text(
        "LIMIT_WEST=108\nLIMIT_EAST=110\n"
        "LIMIT_SOUTH=20\nLIMIT_NORTH=22\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run_def domain does not match landpoint ID"):
        shards.load_plan(_write_plan(tmp_path, [entry]), require_inputs=True)


def test_plan_requires_cold_start_reference_inventory(tmp_path):
    entry = _entry(tmp_path, "001.0-071.0", 1961)
    (Path(entry["reference_run_dir"]) / "stomate_history_1961.nc").unlink()

    with pytest.raises(FileNotFoundError, match="stomate_history_1961.nc"):
        shards.load_plan(_write_plan(tmp_path, [entry]), require_inputs=True)


def test_worker_assignment_keeps_complete_landpoint_chain_together(tmp_path):
    second_year = _entry(tmp_path, "001.0-071.0", 1962)
    second_year.pop("state_cache")
    entries = [
        _entry(tmp_path, "001.0-071.0", 1961),
        second_year,
        _entry(tmp_path, "003.0-071.0", 1961),
    ]
    plan = shards.load_plan(_write_plan(tmp_path, entries))
    selected = [
        shards.assigned_entries(plan, index, 4)
        for index in range(4)
    ]
    owners = {
        entry.landpoint_id: index
        for index, values in enumerate(selected)
        for entry in values
    }
    assignment = shards.worker_assignment(plan, 4)
    assert sum(len(values) for values in selected) == 3
    assert owners == assignment
    assert [
        entry.year
        for entry in selected[owners["001.0-071.0"]]
        if entry.landpoint_id == "001.0-071.0"
    ] == [1961, 1962]


def test_worker_assignment_balances_complete_landpoint_chains(tmp_path):
    entries = [
        _entry(tmp_path, f"{index:03d}.0-071.0", 1961)
        for index in range(12)
    ]
    plan = shards.load_plan(_write_plan(tmp_path, entries))

    for worker_count, expected_loads in (
        (4, [3, 3, 3, 3]),
        (8, [2, 2, 2, 2, 1, 1, 1, 1]),
        (12, [1] * 12),
    ):
        selected = [
            shards.assigned_entries(plan, index, worker_count)
            for index in range(worker_count)
        ]
        assert [len(values) for values in selected] == expected_loads
        assert {
            entry.landpoint_id
            for values in selected
            for entry in values
        } == {entry["landpoint_id"] for entry in entries}


def test_max_new_entries_advances_across_clean_worker_invocations(
    monkeypatch,
    tmp_path,
):
    entries = tuple(
        shards.PlanEntry(
            landpoint_id="001.0-071.0",
            year=year,
            days=365,
            spatial_split="train",
            temporal_split="train",
            run_def=tmp_path / "run.def",
            reference_run_dir=tmp_path,
            state_cache=None,
            initialization_mode=shards.YEAR_START_CHECKPOINT,
            acceptance_checkpoint=None,
        )
        for year in range(1961, 1966)
    )
    plan = shards.GenerationPlan(
        path=tmp_path / "plan.json",
        raw={},
        plan_sha256="plan",
        dataset_id="dataset",
        teacher_config=tmp_path / "config.yaml",
        block_size=7,
        output_root=tmp_path / "output",
        entries=entries,
    )
    completed = {}

    monkeypatch.setattr(shards, "_clean_git_head", lambda: "head")
    monkeypatch.setattr(
        shards,
        "_completed_metadata",
        lambda _plan, _root, entry, **_kwargs: completed.get(entry.key),
    )
    monkeypatch.setattr(
        shards,
        "_load_checkpoint",
        lambda _root, metadata: metadata["state"],
    )

    def write(_plan, entry, _root, **_kwargs):
        metadata = {
            "checkpoint_sha256": entry.key,
            "state": entry.year,
            "timing_seconds": {"total_entry_before_metadata_write": 1.0},
            "shard_bytes": 1,
        }
        completed[entry.key] = metadata
        return metadata, entry.year

    monkeypatch.setattr(shards, "_write_entry", write)
    monkeypatch.setattr(
        shards,
        "_compact_worker_shard_record",
        lambda _root, entry, _metadata: {
            "landpoint_id": entry.landpoint_id,
            "year": entry.year,
        },
    )

    first = shards.generate_worker(
        plan,
        output_root=plan.output_root,
        worker_index=0,
        worker_count=1,
        max_new_entries=2,
    )
    assert first["completed_entries"] == [
        "001.0-071.0:1961",
        "001.0-071.0:1962",
    ]
    assert not first["complete"]

    second = shards.generate_worker(
        plan,
        output_root=plan.output_root,
        worker_index=0,
        worker_count=1,
        max_new_entries=2,
    )
    assert second["completed_entries"] == [
        "001.0-071.0:1961",
        "001.0-071.0:1962",
        "001.0-071.0:1963",
        "001.0-071.0:1964",
    ]
    assert not second["complete"]

    third = shards.generate_worker(
        plan,
        output_root=plan.output_root,
        worker_index=0,
        worker_count=1,
        max_new_entries=2,
    )
    assert third["completed_entries"] == [entry.key for entry in entries]
    assert third["complete"]


def test_bounded_generate_cli_succeeds_with_an_incomplete_resumable_worker(
    monkeypatch,
    tmp_path,
):
    plan = SimpleNamespace(output_root=tmp_path / "output")
    monkeypatch.setattr(shards, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        shards,
        "generate_worker",
        lambda *_args, **_kwargs: {
            "complete": False,
            "completed_entries": ["001.0-071.0:1961"],
        },
    )

    exit_code = shards.main(
        [
            "generate",
            "--plan",
            str(tmp_path / "plan.json"),
            "--worker-index",
            "0",
            "--worker-count",
            "5",
            "--max-new-entries",
            "1",
        ]
    )

    assert exit_code == 0


def test_generation_rejects_an_uncommitted_teacher_identity(monkeypatch):
    def output(command, **_kwargs):
        return "abc123\n" if command[1:3] == ["rev-parse", "HEAD"] else " M teacher.py\n"

    monkeypatch.setattr(shards.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="clean committed worktree"):
        shards._clean_git_head()


def test_process_memory_bytes_parses_linux_status(tmp_path):
    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython\nVmHWM:\t  4321 kB\nVmRSS:\t  1234 kB\n",
        encoding="utf-8",
    )
    assert shards._process_memory_bytes(status) == {
        "current_rss_bytes": 1234 * 1024,
        "peak_rss_bytes": 4321 * 1024,
    }


def test_compiled_cache_entries_reports_both_production_caches(monkeypatch):
    monkeypatch.setattr(shards.teacher, "_COMPILED_LATER_DAY_BLOCK_CACHE", {1: "a"})
    monkeypatch.setattr(
        shards.teacher,
        "_COMPILED_SECHIBA_SCAN_CACHE",
        {1: "a", 2: "b"},
    )
    assert shards._compiled_cache_entries() == {
        "later_day_block": 1,
        "sechiba_scan": 2,
    }


def test_stale_worker_lock_recovery_requires_exact_owner(tmp_path):
    plan = SimpleNamespace(entries=())
    worker_root = tmp_path / "workers" / "worker-000-of-001"
    worker_root.mkdir(parents=True)
    lock = worker_root / "generation.lock"
    owner = {
        "pid": 42,
        "host": "compute-01",
        "slurm_job_id": "12345",
        "created_unix": 1.0,
    }
    lock.write_text(json.dumps(owner), encoding="utf-8")
    with pytest.raises(ValueError, match="owner mismatch"):
        shards.recover_stale_worker_lock(
            plan,
            output_root=tmp_path,
            worker_index=0,
            worker_count=1,
            expected_host="compute-02",
            expected_pid=42,
        )
    assert lock.is_file()
    recovered = shards.recover_stale_worker_lock(
        plan,
        output_root=tmp_path,
        worker_index=0,
        worker_count=1,
        expected_host="compute-01",
        expected_pid=42,
        expected_slurm_job_id="12345",
    )
    assert recovered["status"] == "recovered"
    assert not lock.exists()
    assert Path(recovered["preserved_lock"]).is_file()


def test_initial_state_rejects_incomplete_year_handoff_before_compilation(monkeypatch):
    state = SimpleNamespace(
        fields_by_component={
            "slowproc_stomate_previous_step_state": {"t2m_month": np.ones(1)}
        }
    )
    entry = SimpleNamespace(key="point:1962", state_cache=Path("state.pkl"))
    monkeypatch.setattr(shards, "_load_state_cache", lambda _path: {"state": state})
    monkeypatch.setattr(shards.teacher, "driver_year_handoff_state_gaps", lambda _state: ())
    monkeypatch.setattr(
        shards.teacher,
        "STOMATE_DAY_SEASON_STATE_FIELDS",
        ("t2m_month", "date"),
    )
    with pytest.raises(ValueError, match="slowproc_stomate_previous_step_state.date"):
        shards._initial_state(entry, None)

    state.fields_by_component["slowproc_stomate_previous_step_state"]["date"] = 365
    assert shards._initial_state(entry, None) is state


def test_year_end_acceptance_compares_complete_scientific_state_exactly():
    expected = SimpleNamespace(
        tstep=17519,
        fields_by_component={
            "hydrol_previous_step_state": {
                "mc": np.asarray([[0.2, np.nan]], dtype=np.float64),
                "mask": np.asarray([True, False]),
            },
            "slowproc_stomate_previous_step_state": {
                "nested": (np.asarray([1], dtype=np.int32), "source-state")
            },
        },
    )
    actual = SimpleNamespace(
        tstep=expected.tstep,
        fields_by_component={
            component: {
                name: value.copy() if isinstance(value, np.ndarray) else value
                for name, value in fields.items()
            }
            for component, fields in expected.fields_by_component.items()
        },
    )
    report = shards._compare_driver_state(actual, expected)
    assert report["status"] == "exact"
    assert report["compared_state_leaves"] == 4

    actual.fields_by_component["hydrol_previous_step_state"]["mc"][0, 0] += 1.0e-15
    report = shards._compare_driver_state(actual, expected)
    assert report["status"] == "numeric_close"
    assert report["exact_mismatch_leaves"] == 1

    actual.fields_by_component["hydrol_previous_step_state"]["mc"][0, 0] += 1.0e-6
    with pytest.raises(ValueError, match="hydrol_previous_step_state.mc"):
        shards._compare_driver_state(actual, expected)


def test_cold_start_capture_uses_day1_end_without_year_rebase(monkeypatch):
    plan = SimpleNamespace(teacher_config=Path("teacher.yaml"))
    entry = SimpleNamespace(
        key="point:1961",
        initialization_mode=shards.COLD_START_BOOTSTRAP,
        year=1961,
        days=365,
        run_def=Path("used_run.def"),
        reference_run_dir=Path("reference"),
    )
    bootstrap = SimpleNamespace(
        ready_for_first_day_end_state=True,
        first_day_end_state="canonical-S1",
        state_gaps=(),
    )
    calls = []
    monkeypatch.setattr(
        shards.teacher,
        "paper_1961_driver_cold_start_day_scaffold",
        lambda *args, **kwargs: calls.append((args, kwargs)) or bootstrap,
    )
    monkeypatch.setattr(
        shards.teacher,
        "rebase_driver_state_for_year_start",
        lambda _state: pytest.fail("cold-start canonical S1 must not be rebased"),
    )

    result = shards._entry_capture_start(plan, entry, "context", None)

    assert result == ("canonical-S1", 2, 364, 1)
    assert calls[0][1]["prepared_context"] == "context"


def test_aggregate_requires_complete_hash_verified_worker_outputs(tmp_path):
    entry = _entry(tmp_path, "001.0-071.0", 1961)
    plan = shards.load_plan(_write_plan(tmp_path, [entry]))
    output_root = tmp_path / "dataset"
    worker_root = output_root / "workers" / "worker-000-of-001"
    shard_path = worker_root / "shards" / "sample.npz"
    checkpoint_path = worker_root / "checkpoints" / "sample.pkl"
    shards._atomic_npz(shard_path, {"value": np.asarray([1.0])})
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    contract = {"schema_version": "test_contract_v1", "state_width": 1}
    contract_hash = hashlib.sha256(shards._canonical_json(contract)).hexdigest()
    _, metadata_path, _ = shards._entry_paths(worker_root, plan.entries[0])
    metadata = {
        "landpoint_id": "001.0-071.0",
        "year": 1961,
        "spatial_split": "train",
        "temporal_split": "train",
        "markov_contract": contract,
        "markov_contract_sha256": contract_hash,
        "preceding_checkpoint_sha256": None,
        "input_hashes": shards._input_hashes(plan, plan.entries[0]),
        "shard": "shards/sample.npz",
        "shard_sha256": shards._sha256_file(shard_path),
        "checkpoint": "checkpoints/sample.pkl",
        "checkpoint_sha256": shards._sha256_file(checkpoint_path),
    }
    shards._atomic_json(metadata_path, metadata)
    shard = {
        name: metadata[name]
        for name in (
            "landpoint_id",
            "year",
            "spatial_split",
            "temporal_split",
            "markov_contract_sha256",
            "preceding_checkpoint_sha256",
            "input_hashes",
            "shard",
            "shard_sha256",
            "checkpoint",
            "checkpoint_sha256",
        )
    }
    shard["metadata"] = shards._relative(metadata_path, worker_root)
    shard["metadata_sha256"] = shards._sha256_file(metadata_path)
    shards._atomic_json(
        worker_root / "manifest.json",
        {
            "schema_version": shards.MANIFEST_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "teacher_git_head": "teacher",
            "worker_index": 0,
            "worker_count": 1,
            "worker_assignment_strategy": shards.WORKER_ASSIGNMENT_STRATEGY,
            "shards": [shard],
        },
    )
    result = shards.aggregate_workers(plan, output_root=output_root, worker_count=1)
    assert result["status"] == "complete"
    assert result["shard_count"] == 1
    assert result["markov_contract"] == contract
    assert "markov_contract" not in result["shards"][0]

    second_entry = _entry(tmp_path, "003.0-071.0", 1961)
    subset_plan = shards.load_plan(_write_plan(tmp_path, [entry, second_entry]))
    worker_manifest_path = worker_root / "manifest.json"
    worker_manifest = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
    worker_manifest["plan_sha256"] = subset_plan.plan_sha256
    shards._atomic_json(worker_manifest_path, worker_manifest)
    subset = shards.aggregate_workers(
        subset_plan,
        output_root=output_root,
        worker_count=1,
        included_landpoints=frozenset({"001.0-071.0"}),
        dataset_id="test-dataset-nine-point-provisional",
        manifest_name="dataset_manifest_subset.json",
    )
    assert subset["status"] == "complete"
    assert subset["dataset_id"] == "test-dataset-nine-point-provisional"
    assert subset["landpoint_count"] == 1
    assert subset["derived_subset"]["excluded_landpoints"] == ["003.0-071.0"]
    assert (output_root / "dataset_manifest_subset.json").is_file()
    with pytest.raises(ValueError, match="dataset is incomplete"):
        shards.aggregate_workers(
            subset_plan,
            output_root=output_root,
            worker_count=1,
        )

    worker_manifest["plan_sha256"] = plan.plan_sha256
    shards._atomic_json(worker_manifest_path, worker_manifest)

    shards._atomic_json(
        worker_root / "manifest.json",
        {
            "schema_version": shards.LEGACY_MANIFEST_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "teacher_git_head": "teacher",
            "worker_index": 0,
            "worker_count": 1,
            "worker_assignment_strategy": shards.WORKER_ASSIGNMENT_STRATEGY,
            "shards": [metadata],
        },
    )
    legacy_result = shards.aggregate_workers(
        plan,
        output_root=output_root,
        worker_count=1,
    )
    assert legacy_result["schema_version"] == shards.DATASET_SCHEMA_VERSION

    shard_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        shards.aggregate_workers(plan, output_root=output_root, worker_count=1)


def test_compiled_capture_uses_a_short_tail_block(monkeypatch):
    audited_calls = []
    first_record = SimpleNamespace(
        half_hour_transition=SimpleNamespace(current_state="boundary-state")
    )

    def capture_first_day(**kwargs):
        audited_calls.append(kwargs)
        day = kwargs["start_day"]
        return ((f"start-{day}",), (f"forcing-{day}",), (first_record,), f"state-{day}")

    monkeypatch.setattr(
        capture,
        "_capture_days",
        capture_first_day,
    )
    monkeypatch.setattr(capture.jax, "device_get", lambda value: value)
    monkeypatch.setattr(
        capture,
        "_indexed_tree",
        lambda tree, index: tree[index],
    )
    monkeypatch.setattr(
        capture,
        "_compiled_training_record",
        lambda **kwargs: SimpleNamespace(
            day_index=kwargs["day_index"],
            expected_result=SimpleNamespace(day_end_state=kwargs["day_end_state"]),
        ),
    )
    monkeypatch.setattr(
        capture.teacher,
        "fast_state_from_previous_packet",
        lambda value: SimpleNamespace(spec="state-spec", values_by_component=f"values:{value}"),
    )
    monkeypatch.setattr(
        capture.teacher,
        "previous_packet_from_fast_state",
        lambda value: value.values_by_component,
    )
    monkeypatch.setattr(
        capture.teacher,
        "DriverFastStateBundle",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        capture.teacher,
        "_paper_1961_later_day_transition_inputs",
        lambda **_kwargs: SimpleNamespace(
            hydrol_runtime_static_tables="tables",
            compiled_base_payload_template="payload",
        ),
    )
    for name, value in {
        "_paper_daily_carbon_static_dispatch": "daily",
        "_compiled_stomate_parameter_values": "stomate-parameters",
        "_compiled_landpoint_payload": "landpoint",
        "_compiled_diffuco_parameter_values": "diffuco-parameters",
        "_compiled_hydrol_table_arrays": "hydrol-arrays",
    }.items():
        monkeypatch.setattr(capture.teacher, name, lambda *_args, _value=value, **_kwargs: _value)
    monkeypatch.setattr(
        capture.teacher,
        "read_stomate_restart_season_state",
        lambda *_args, **_kwargs: SimpleNamespace(_asdict=lambda: {"provenance": "test"}),
    )
    monkeypatch.setattr(
        capture.teacher,
        "_paper_compiled_forcing_day",
        lambda _context, **kwargs: np.asarray(kwargs["start_tstep"]),
    )
    compiled_sizes = []

    def compile_block(_config, **kwargs):
        size = int(np.asarray(kwargs["block_day_numbers"]).size)
        compiled_sizes.append(size)

        def executable(_initial, block_forcing, block_days, *_args):
            count = int(np.asarray(block_days).size)
            return (
                f"final-{int(block_days[-1])}",
                ([f"boundary-{index}" for index in range(count)], [f"state-{index}" for index in range(count)]),
            )

        return executable, "state-spec"

    monkeypatch.setattr(
        capture.teacher,
        "_paper_compiled_later_day_block_executable",
        compile_block,
    )
    context = SimpleNamespace(
        runtime=SimpleNamespace(dt_stomate=86400.0, dt_sechiba=1800.0),
        first_step_restart_state=SimpleNamespace(stomate="stomate", stomate_input="input"),
    )
    blocks = list(
        capture._iter_capture_days_compiled_blocks(
            config_path=Path("config"),
            context=context,
            previous_state="initial",
            year=1964,
            start_day=2,
            days=5,
            block_size=3,
        )
    )
    assert [(call["start_day"], call["days"]) for call in audited_calls] == [(2, 1)]
    assert compiled_sizes == [3, 1]
    assert [len(block[0]) for block in blocks] == [1, 3, 1]
    states = tuple(state for block in blocks for state in block[0])
    forcings = tuple(forcing for block in blocks for forcing in block[1])
    records = tuple(record for block in blocks for record in block[2])
    final_state = blocks[-1][3]
    assert len(states) == len(forcings) == len(records) == 5
    assert final_state == "final-6"
