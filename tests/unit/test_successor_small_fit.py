from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import research.daily_coarse_graining.daily_operator.successor_fast_owner_training as fast_owner_training
import research.daily_coarse_graining.daily_operator.successor_fit_checkpoint as fit_checkpoint
import research.daily_coarse_graining.daily_operator.successor_frozen_coefficients as frozen_coefficients
import research.daily_coarse_graining.daily_operator.successor_small_fit as small_fit
import scripts.hpc.run_successor_fast_owner_gpu_probe as fast_probe
import scripts.hpc.run_successor_small_fit as production_fit
from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
    DAY_START_DELTA,
    DEFINED_LOGIT,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    SuccessorCheckpointHandoff,
    admit_prepared_manifest_for_checkpoint_fit,
    load_successor_checkpoint_handoff,
    load_successor_checkpoint_set,
    load_successor_recovery,
    write_successor_checkpoint_set,
    write_successor_recovery,
)
from research.daily_coarse_graining.daily_operator.successor_small_fit import (
    build_successor_small_fit_schedule,
    initialize_successor_adamw,
    load_successor_small_fit_contract,
    successor_adamw_update,
    successor_learning_rate,
    successor_small_fit_loss,
    successor_teacher_physical_outputs,
    verify_successor_small_fit_schedule,
)
from research.daily_coarse_graining.daily_operator.successor_training import (
    LONG_HORIZON_COMPONENTS,
)
from scripts.hpc.prepare_successor_small_fit_batches import (
    _finalize_worker_progress,
    _worker_updates,
)
from scripts.hpc.run_successor_small_fit import (
    _retained_tail_trace_signature,
    _teacher_replay_training_policy,
    _validate_update_count,
)
from scripts.hpc.run_successor_small_fit_probe import _leaf_failures
from scripts.hpc.successor_small_fit_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedUpdateArtifact,
    atomic_json,
    load_prepared_artifact,
    load_prepared_manifest,
    sha256_file,
    write_prepared_artifact,
)


@dataclass(frozen=True)
class _Parent:
    landpoint_id: str
    year: int
    spatial_split: str = "train"
    temporal_split: str = "train"
    sha256: str = "a" * 64


@dataclass(frozen=True)
class _Reference:
    parent: _Parent
    day_count: int = 365
    sha256: str = "b" * 64


def _references() -> tuple[_Reference, ...]:
    return tuple(
        _Reference(
            _Parent(f"{landpoint:03d}.0-071.0", year),
            sha256=f"{landpoint + year:064x}"[-64:],
        )
        for landpoint in range(1, 5)
        for year in range(1961, 1964)
    )


def test_small_fit_schedule_is_deterministic_matched_and_excludes_first_row():
    contract = load_successor_small_fit_contract()
    references = _references()
    left = build_successor_small_fit_schedule(references, contract, updates=8)
    right = build_successor_small_fit_schedule(tuple(reversed(references)), contract, updates=8)

    assert left == right
    assert len(left["entries"]) == 8
    assert len({entry["landpoint_id"] for entry in left["entries"][:4]}) == 4
    assert all(min(entry["anchor_starts"]) >= 1 for entry in left["entries"])
    assert all(min(entry["rollout_starts"]) >= 1 for entry in left["entries"])
    assert all(len(set(entry["anchor_starts"])) == 16 for entry in left["entries"])
    assert all(len(set(entry["rollout_starts"])) == 2 for entry in left["entries"])
    verify_successor_small_fit_schedule(left, references, contract)


def test_small_fit_schedule_rejects_non_train_reference():
    contract = load_successor_small_fit_contract()
    bad = _Reference(_Parent("001.0-071.0", 1961, temporal_split="validation"))
    with pytest.raises(ValueError, match="non-train"):
        build_successor_small_fit_schedule((bad,), contract, updates=1)


def test_successor_adamw_synthetic_step_is_finite_clipped_and_bounded():
    contract = load_successor_small_fit_contract()
    parameters = {"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float64)}
    state = initialize_successor_adamw(parameters)

    def loss(values):
        return jnp.sum((values["weight"] - 4.0) ** 2)

    loss_value, gradients = jax.value_and_grad(loss)(parameters)
    updated, next_state, norm = successor_adamw_update(
        parameters,
        gradients,
        state,
        successor_learning_rate(contract, 1),
        contract,
    )

    assert np.isfinite(float(loss_value))
    assert float(norm) > contract.optimizer["global_gradient_norm_clip"]
    assert int(next_state.step) == 1
    assert np.all(np.isfinite(np.asarray(updated["weight"])))
    assert successor_learning_rate(contract, 50) == pytest.approx(3.0e-4)
    assert successor_learning_rate(contract, 500) == pytest.approx(3.0e-5)
    with pytest.raises(ValueError, match="outside"):
        successor_learning_rate(contract, 501)


def test_successor_combined_objective_supports_one_synthetic_update(monkeypatch):
    parameters = {"weight": jnp.asarray(2.0, dtype=jnp.float64)}
    coefficients = {
        name: value
        for name, value in zip(
            LONG_HORIZON_COMPONENTS,
            (0.1, 0.2, 0.3, 0.4),
            strict=True,
        )
    }
    monkeypatch.setattr(
        small_fit,
        "successor_anchor_batch_loss",
        lambda values, *_args, **_kwargs: values["weight"] ** 2,
    )
    monkeypatch.setattr(
        small_fit,
        "successor_rollout_component_losses",
        lambda values, *_args, **_kwargs: (
            values["weight"] * jnp.arange(1.0, 5.0),
            None,
        ),
    )

    def objective(values):
        return successor_small_fit_loss(
            values,
            SimpleNamespace(anchor_inputs=None, anchor_targets=None),
            head=None,
            schema=None,
            loss_scales={},
            transition=None,
            markov_contract=None,
            flux_scales={},
            stock_scales={},
            coefficients=coefficients,
        )

    (loss, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(parameters)
    assert float(loss) == pytest.approx(10.0)
    assert float(auxiliary[0]) == pytest.approx(4.0)
    assert np.asarray(auxiliary[1]).tolist() == pytest.approx([2.0, 4.0, 6.0, 8.0])
    assert float(gradient["weight"]) == pytest.approx(7.0)


def test_teacher_replay_inverts_delta_and_defined_logit_representations():
    fields = (
        SimpleNamespace(
            name="direct",
            value_representation="direct",
            shape=(1,),
            day_start_state_start=None,
            day_start_state_stop=None,
        ),
        SimpleNamespace(
            name="delta",
            value_representation=DAY_START_DELTA,
            shape=(1,),
            day_start_state_start=1,
            day_start_state_stop=2,
        ),
        SimpleNamespace(
            name="status",
            value_representation=DEFINED_LOGIT,
            shape=(2,),
            day_start_state_start=None,
            day_start_state_stop=None,
        ),
    )
    schema = SimpleNamespace(fields=fields)
    targets = {
        "direct": SimpleNamespace(value=np.asarray([3.0])),
        "delta": SimpleNamespace(value=np.asarray([0.5])),
        "status": SimpleNamespace(value=np.asarray([1.0, 0.0])),
    }
    inputs = SimpleNamespace(continuous_state=np.asarray([1.0, 2.0]))
    outputs = successor_teacher_physical_outputs(targets, inputs, schema)

    assert np.asarray(outputs["direct"]).tolist() == [3.0]
    assert np.asarray(outputs["delta"]).tolist() == [2.5]
    assert np.asarray(outputs["status"]).tolist() == [20.0, -20.0]


def test_replay_leaf_diagnostic_reports_undefined_representation_mismatch():
    leaves = (
        SimpleNamespace(key="first", start=0, stop=2),
        SimpleNamespace(key="second", start=2, stop=3),
    )
    actual = np.asarray([1.0, 1.0e20, np.nan])
    expected = np.asarray([1.0, np.nan, np.nan])
    defined = np.asarray([True, False, False])
    failures = _leaf_failures(
        leaves,
        actual,
        expected,
        defined,
        defined,
        np.full(3, 1.0e-8),
    )

    assert failures == [
        {
            "key": "first",
            "mask_mismatch_count": 0,
            "numerical_failure_count": 0,
            "undefined_representation_mismatch_count": 1,
            "actual_defined_count": 1,
            "expected_defined_count": 1,
            "maximum_common_absolute_error": 0.0,
        }
    ]


def test_gpu_launcher_registers_cpu_for_jax_scan_partial_evaluation():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_successor_small_fit_probe.sh").read_text()

    assert "SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \\" in launcher


def test_production_runner_trace_signature_excludes_dynamic_landpoint_year_arrays():
    def adapter(*, runtime_year, day_inputs, nstm):
        owner = SimpleNamespace(
            config_path=Path("config.json"),
            contract=SimpleNamespace(sha256="a" * 64),
            mineral_imin=1,
            mineral_imax=11,
            daily_carbon_dispatch={"owner": np.asarray([1.0])},
            metadata=SimpleNamespace(
                nstm=nstm,
                is_tree=np.asarray([False, True]),
                is_peat=np.asarray([False, False]),
                npts=1,
            ),
            season_provenance=("source",),
            runtime_year=runtime_year,
            lifecycle_start_tstep=48,
            result_day_index=2,
            include_modelout_diagnostics=False,
            context=object(),
        )
        return SimpleNamespace(
            static_owner=owner,
            day_inputs=day_inputs,
            fast_owner_materializer=SimpleNamespace(),
        )

    first = adapter(runtime_year=1961, day_inputs=np.asarray([1.0]), nstm=1)
    switched = adapter(runtime_year=1999, day_inputs=np.asarray([9.0]), nstm=1)
    control_drift = adapter(runtime_year=1999, day_inputs=np.asarray([9.0]), nstm=2)

    assert _retained_tail_trace_signature(first) == _retained_tail_trace_signature(switched)
    assert _retained_tail_trace_signature(first) != _retained_tail_trace_signature(control_drift)


def test_production_runner_enforces_bounded_update_budget():
    assert _validate_update_count(2, 500) == 2
    assert _validate_update_count(500, 500) == 500
    with pytest.raises(ValueError, match=r"\[2, 500\]"):
        _validate_update_count(1, 500)
    with pytest.raises(ValueError, match=r"\[2, 500\]"):
        _validate_update_count(501, 500)


def test_nonfinite_anchor_breakdown_resolves_samples_and_process_labels(monkeypatch):
    class Inputs(tuple):
        __slots__ = ()
        _fields = ("continuous_state", "feature")

        def __new__(cls, continuous_state, feature):
            return tuple.__new__(cls, (continuous_state, feature))

        continuous_state = property(lambda self: self[0])
        feature = property(lambda self: self[1])

    head = SimpleNamespace(
        input_assembler=SimpleNamespace(assemble=lambda value: value),
        assemble_context=lambda _parameters, _inputs: None,
        named_outputs=lambda parameters, _context, inputs: {
            "prediction": parameters["decoder"]["weight"] * inputs.feature
        },
    )

    def fake_loss(outputs, target, **_kwargs):
        loss = (outputs["prediction"] - target) ** 2
        return SimpleNamespace(total=loss, label_loss={"process.alpha": loss})

    monkeypatch.setattr(production_fit, "_head", lambda _runtime, _arm: head)
    monkeypatch.setattr(fast_owner_training, "successor_fast_owner_loss", fake_loss)
    batch = SimpleNamespace(
        anchor_inputs=Inputs(
            continuous_state=jnp.zeros((2, 1)),
            feature=jnp.asarray([1.0, 3.0]),
        ),
        anchor_targets=jnp.asarray([0.0, 1.0]),
    )
    report = production_fit._diagnose_anchor_breakdown(
        {"jax": jax, "schema": SimpleNamespace(labels=("process.alpha",)), "loss_scales": {}},
        "attention",
        {"decoder": {"weight": jnp.asarray(2.0)}},
        batch,
        (7, 19),
    )

    assert [item["anchor_start"] for item in report["anchor_samples"]] == [7, 19]
    assert [item["loss"] for item in report["anchor_samples"]] == [4.0, 25.0]
    assert report["process_labels"]["process.alpha"]["loss"] == 14.5
    assert report["process_labels"]["process.alpha"]["gradient"]["all_finite"]


def test_rollout_prefix_diagnostic_slices_window_and_horizon():
    values = np.arange(2 * 30 * 3).reshape(2, 30, 3)
    batch = small_fit.PreparedSuccessorSmallFitBatch(
        anchor_inputs=None,
        anchor_targets=None,
        rollout_inputs={"values": values},
        rollout_targets={"values": values + 1},
        expected_next_state=values + 2,
        expected_next_state_defined=np.ones_like(values, dtype=np.bool_),
        expected_fast_gpp=values + 3,
        expected_fast_gpp_defined=np.ones_like(values, dtype=np.bool_),
        expected_fast_day_target=values + 4,
        expected_next_discrete_state={"counter": values.astype(np.int32)},
    )

    sliced = production_fit._slice_rollout_batch(jax, batch, 7, 1)

    assert sliced.rollout_inputs["values"].shape == (1, 7, 3)
    np.testing.assert_array_equal(sliced.rollout_inputs["values"], values[1:2, :7])
    assert sliced.expected_next_state.shape == (1, 7, 3)
    assert sliced.expected_next_discrete_state["counter"].shape == (1, 7, 3)


def test_production_launcher_preserves_contract_and_dual_backend_boundary():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_successor_small_fit.sh").read_text()

    assert "SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \\" in launcher
    assert "SINGULARITYENV_PYTHONHASHSEED=0 \\" in launcher
    assert '--updates "$UPDATES" \\' in launcher
    assert "requires exactly 500 updates" in launcher
    assert "successor_small_fit_runner_v1.json" in launcher
    assert "successor_small_fit_checkpoint_handoff_v1.json" in launcher
    assert "PREPARED_MANIFEST=${PREPARED_MANIFEST:?" in launcher
    assert '--prepared-manifest "$PREPARED_MANIFEST" \\' in launcher
    assert "successor_small_fit_frozen_coefficients_v1.json" in launcher
    assert '--coefficient-contract "$COEFFICIENT_CONTRACT" \\' in launcher
    assert '--coefficient-source-report "$COEFFICIENT_SOURCE_REPORT" \\' in launcher
    assert '--checkpoint-dir "$CHECKPOINT_DIR" \\' in launcher
    assert '--recovery-path "$RECOVERY_PATH" \\' in launcher
    assert "--recovery-interval 25" in launcher
    assert 'EXTRA_ARGS="--replay-only-update $REPLAY_ONLY_UPDATE"' in launcher
    assert "--diagnose-update $DIAGNOSE_UPDATE" in launcher
    assert "--recovery-identity-git-head $RECOVERY_IDENTITY_GIT_HEAD" in launcher
    assert "--diagnose-horizon $DIAGNOSE_HORIZON" in launcher
    assert "--diagnose-window $DIAGNOSE_WINDOW" in launcher
    assert "$EXTRA_ARGS" in launcher
    assert 'test -f "$CHECKPOINT_DIR/checkpoint_manifest.json"' in launcher


def test_rejected_paid_launcher_is_removed():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "scripts" / "hpc" / "slurm_successor_small_fit.sh").exists()


def test_two_stage_production_launchers_are_bounded_and_unsubmitted():
    root = Path(__file__).resolve().parents[2]
    cpu = (root / "scripts" / "hpc" / "slurm_prepare_successor_small_fit_500.sh").read_text()
    gpu = (root / "scripts" / "hpc" / "slurm_successor_small_fit_500.sh").read_text()

    assert "#SBATCH --partition=cnmix" in cpu
    assert "#SBATCH --ntasks=4" in cpu
    assert "#SBATCH --time=04:00:00" in cpu
    assert "WORKER_COUNT=4" in cpu
    assert "FINALIZE=true" in cpu
    assert 'test ! -e "$OUTPUT_DIR/prepared_manifest.json"' in cpu
    assert "CNY 1.12 worst case" in cpu

    assert "#SBATCH --partition=gnall" in gpu
    assert "#SBATCH --cpus-per-task=16" in gpu
    assert "#SBATCH --gres=gpu:1" in gpu
    assert "#SBATCH --time=02:00:00" in gpu
    assert "PREPARED_MANIFEST=${PREPARED_MANIFEST:-" in gpu
    assert "UPDATES=500" in gpu
    assert 'test ! -e "$CHECKPOINT_DIR"' in gpu
    assert "RECOVERY_PATH=" in gpu
    assert "CNY 6.64 worst case" in gpu


def test_checkpoint_gpu_probe_is_serialization_only_and_removes_temporary_set():
    root = Path(__file__).resolve().parents[2]
    probe = (root / "scripts" / "hpc" / "probe_successor_checkpoint_handoff.py").read_text()

    assert '"training_updates": 0' in probe
    assert '"teacher_data_read": False' in probe
    assert '"teacher_data_regenerated": False' in probe
    assert '"historical_v5_used": False' in probe
    assert "TemporaryDirectory" in probe
    assert '"temporary_checkpoint_removed": True' in probe


def test_checkpoint_handoff_contract_is_frozen_and_keeps_old_producer_identity():
    handoff = load_successor_checkpoint_handoff()

    assert handoff.producer["producer_git_head"] == "d4936996635b02fcf04cd5f8c4926be3cc010124"
    assert handoff.producer["prepared_manifest_sha256"] == (
        "0182b7f7a6d5eae09025a336f26bc67ba5fb54706c3529a846d3885c16d80462"
    )
    assert handoff.checkpoint["checkpoint_on_failed_gate"] is False
    assert handoff.checkpoint["checkpoint_on_submaximal_probe"] is False


def test_checkpoint_handoff_separates_producer_and_consumer_git_heads(tmp_path):
    manifest_path = tmp_path / "prepared_manifest.json"
    manifest = {
        "identity": {
            "source_git_head": "a" * 40,
            "runner_contract_sha256": "runner",
            "schedule_sha256": "schedule",
            "dataset_manifest_sha256": "dataset",
            "successor_candidate_contract_sha256": "candidate",
            "fast_owner_contract_sha256": "owner",
        },
        "updates": 2,
        "entries": [{}, {}],
        "total_size_bytes": 12,
        "trace_signatures": ["trace"],
    }
    atomic_json(manifest_path, manifest)
    producer = {
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "producer_git_head": "a" * 40,
        "producer_runner_contract_sha256": "runner",
        "schedule_sha256": "schedule",
        "dataset_manifest_sha256": "dataset",
        "successor_candidate_contract_sha256": "candidate",
        "fast_owner_contract_sha256": "owner",
        "updates": 2,
        "artifact_count": 2,
        "total_size_bytes": 12,
        "trace_signature": "trace",
    }
    handoff = SuccessorCheckpointHandoff(
        payload={"producer_artifacts": producer},
        source_path=tmp_path / "handoff.json",
        sha256="handoff",
    )

    admitted = admit_prepared_manifest_for_checkpoint_fit(
        handoff,
        manifest_path,
        manifest,
        runner_contract_sha256="runner",
        schedule_sha256="schedule",
        consumer_git_head="b" * 40,
    )
    assert admitted["producer_git_head"] == "a" * 40
    assert admitted["consumer_git_head"] == "b" * 40
    with pytest.raises(ValueError, match="consumer git identity"):
        admit_prepared_manifest_for_checkpoint_fit(
            handoff,
            manifest_path,
            manifest,
            runner_contract_sha256="runner",
            schedule_sha256="schedule",
            consumer_git_head="a" * 40,
        )


def _checkpoint_fixture():
    arms = (
        "source_routed_process_attention_v2",
        "source_routed_graph_mlp_control_v2",
    )
    parameters = {
        arm: {"weight": np.asarray([index + 1.0, index + 2.0])}
        for index, arm in enumerate(arms)
    }
    optimizers = {
        arm: {
            "step": np.asarray(500, dtype=np.int32),
            "first": {"weight": np.asarray([0.1, 0.2])},
            "second": {"weight": np.asarray([0.01, 0.04])},
        }
        for arm in arms
    }
    return arms, parameters, optimizers


def test_atomic_matched_checkpoint_set_round_trip_and_tree_schema(tmp_path):
    arms, parameters, optimizers = _checkpoint_fixture()
    identity = {"consumer_git_head": "b" * 40, "prepared_manifest_sha256": "manifest"}
    checkpoint_dir = tmp_path / "checkpoints"
    result = write_successor_checkpoint_set(
        checkpoint_dir,
        identity=identity,
        parameters=parameters,
        optimizers=optimizers,
        completed_updates={arm: 500 for arm in arms},
        parameter_counts={arm: 2 for arm in arms},
        final_metrics={arm: {"loss": 1.0} for arm in arms},
        loss_reductions={arm: {"one_day": 0.2} for arm in arms},
        expected_arms=arms,
        required_updates=500,
        promotion_passed=True,
    )

    assert result["status"] == "passed"
    assert set(path.name for path in checkpoint_dir.iterdir()) == {
        "checkpoint_manifest.json",
        *(f"{arm}.pkl" for arm in arms),
    }
    loaded = load_successor_checkpoint_set(
        checkpoint_dir,
        expected_manifest_sha256=result["manifest_sha256"],
        identity=identity,
        expected_arms=arms,
        required_updates=500,
    )
    for arm in arms:
        np.testing.assert_array_equal(loaded[arm]["parameters"]["weight"], parameters[arm]["weight"])
        assert loaded[arm]["parameter_tree_schema"] == [
            {"path": "['weight']", "shape": [2], "dtype": "float64"}
        ]

    manifest = json.loads((checkpoint_dir / "checkpoint_manifest.json").read_text())
    arm = arms[0]
    path = checkpoint_dir / manifest["checkpoints"][arm]["path"]
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["parameters"]["weight"] = np.asarray([[1.0, 2.0]])
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    manifest["checkpoints"][arm]["sha256"] = sha256_file(path)
    atomic_json(checkpoint_dir / "checkpoint_manifest.json", manifest)
    with pytest.raises(ValueError, match="tree-schema drift"):
        load_successor_checkpoint_set(
            checkpoint_dir,
            expected_manifest_sha256=sha256_file(checkpoint_dir / "checkpoint_manifest.json"),
            identity=identity,
            expected_arms=arms,
            required_updates=500,
        )


def test_checkpoint_set_refuses_failed_or_incomplete_fit_without_publication(tmp_path, monkeypatch):
    arms, parameters, optimizers = _checkpoint_fixture()
    common = {
        "identity": {"consumer_git_head": "b" * 40},
        "parameters": parameters,
        "optimizers": optimizers,
        "completed_updates": {arm: 500 for arm in arms},
        "parameter_counts": {arm: 2 for arm in arms},
        "final_metrics": {arm: {} for arm in arms},
        "loss_reductions": {arm: {} for arm in arms},
        "expected_arms": arms,
        "required_updates": 500,
    }
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="passing promotion gates"):
        write_successor_checkpoint_set(failed, promotion_passed=False, **common)
    assert not failed.exists()

    incomplete = tmp_path / "incomplete"
    incomplete_updates = dict(common["completed_updates"])
    incomplete_updates[arms[1]] = 499
    with pytest.raises(ValueError, match="update count is incomplete"):
        write_successor_checkpoint_set(
            incomplete,
            promotion_passed=True,
            **(common | {"completed_updates": incomplete_updates}),
        )
    assert not incomplete.exists()

    original = fit_checkpoint._write_pickle
    calls = 0

    def fail_second(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-arm write failure")
        original(path, payload)

    monkeypatch.setattr(fit_checkpoint, "_write_pickle", fail_second)
    interrupted = tmp_path / "interrupted"
    with pytest.raises(OSError, match="second-arm"):
        write_successor_checkpoint_set(interrupted, promotion_passed=True, **common)
    assert not interrupted.exists()
    assert not list(tmp_path.glob(".interrupted.*.tmp"))


def test_recovery_round_trip_is_resumable_but_not_promotion_eligible(tmp_path):
    arms, parameters, optimizers = _checkpoint_fixture()
    identity = {"consumer_git_head": "c" * 40, "schedule_sha256": "schedule"}
    recovery_path = tmp_path / "recovery_state.pkl"
    written = write_successor_recovery(
        recovery_path,
        identity=identity,
        parameters=parameters,
        optimizers=optimizers,
        completed_updates={arm: 25 for arm in arms},
        parameter_counts={arm: 2 for arm in arms},
        initial_metrics={arm: {"loss": 2.0} for arm in arms},
        completed_update_records=[{"update": 25}],
        expected_arms=arms,
    )

    assert written["promotion_eligible"] is False
    loaded = load_successor_recovery(
        recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts={arm: 2 for arm in arms},
        maximum_updates=500,
    )
    assert loaded is not None
    assert loaded["promotion_eligible"] is False
    assert loaded["completed_updates"] == {arm: 25 for arm in arms}
    for arm in arms:
        np.testing.assert_array_equal(loaded["parameters"][arm]["weight"], parameters[arm]["weight"])

    assert (
        load_successor_recovery(
            tmp_path / "missing.pkl",
            identity=identity,
            expected_arms=arms,
            parameter_counts={arm: 2 for arm in arms},
            maximum_updates=500,
        )
        is None
    )
    with pytest.raises(ValueError, match="identity drift"):
        load_successor_recovery(
            recovery_path,
            identity=identity | {"schedule_sha256": "different"},
            expected_arms=arms,
            parameter_counts={arm: 2 for arm in arms},
            maximum_updates=500,
        )


def test_training_policy_logs_continuous_replay_drift_but_keeps_hard_gates():
    replay = {
        "status": "failed",
        "cases": [
            {
                "batch_index": 0,
                "fast_target": {
                    "defined_mask_exact": True,
                    "maximum_defined_tolerance_ratio": 1.0302695733977536,
                    "leaf_failures": [{"numerical_failure_count": 1}],
                },
                "defined_mask_exact": True,
                "defined_values_finite": True,
                "undefined_representation_exact": True,
                "maximum_defined_tolerance_ratio": 1.0302695733977536,
                "state_leaf_failures": [{"numerical_failure_count": 1}],
                "discrete_state_exact": True,
                "water_budget_residual": 0.0,
                "carbon_budget_residual": 0.0,
            }
        ],
    }
    policy = _teacher_replay_training_policy(replay)
    assert policy["training_admitted"] is True
    assert policy["continuous_tolerance_exceeded"] is True
    assert policy["continuous_numerical_failure_count"] == 2
    assert policy["maximum_continuous_tolerance_ratio"] == pytest.approx(1.0302695733977536)

    replay["cases"][0]["defined_values_finite"] = False
    rejected = _teacher_replay_training_policy(replay)
    assert rejected["training_admitted"] is False
    assert rejected["hard_failures"] == ["batch 0: nonfinite defined state"]


def test_prepared_update_round_trip_and_hash_drift(tmp_path):
    entry = {"landpoint_id": "001.0-071.0", "year": 1961, "anchor_starts": [1]}
    artifact = PreparedUpdateArtifact(
        schema_version=PREPARED_ARTIFACT_SCHEMA,
        schedule_entry=entry,
        trace_signature="trace",
        batch={"state": np.asarray([1.0, 2.0])},
        day_inputs=(np.asarray([3.0]),),
    )
    artifact_path = tmp_path / "update-0001.pkl"
    record = write_prepared_artifact(artifact_path, artifact) | {
        "update": 1,
        "trace_signature": "trace",
    }
    manifest_path = tmp_path / "prepared_manifest.json"
    loaded = load_prepared_artifact(manifest_path, record, schedule_entry=entry)

    assert np.asarray(loaded.batch["state"]).tolist() == [1.0, 2.0]
    artifact_path.write_bytes(artifact_path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="hash drift"):
        load_prepared_artifact(manifest_path, record, schedule_entry=entry)


def test_prepared_manifest_binds_source_schedule_and_update_coverage(tmp_path):
    path = tmp_path / "prepared_manifest.json"
    payload = {
        "schema_version": PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "identity": {
            "runner_contract_sha256": "runner",
            "schedule_sha256": "schedule",
            "source_git_head": "head",
        },
        "updates": 2,
        "entries": [{"update": 1}, {"update": 2}],
    }
    atomic_json(path, payload)

    assert (
        load_prepared_manifest(
            path,
            runner_contract_sha256="runner",
            schedule_sha256="schedule",
            updates=2,
            source_git_head="head",
        )["status"]
        == "passed"
    )
    with pytest.raises(ValueError, match="identity drift"):
        load_prepared_manifest(
            path,
            runner_contract_sha256="runner",
            schedule_sha256="schedule",
            updates=2,
            source_git_head="new-head",
        )

    payload["entries"] = [{"update": 1}, {"update": 3}]
    atomic_json(path, payload)
    with pytest.raises(ValueError, match="coverage drift"):
        load_prepared_manifest(
            path,
            runner_contract_sha256="runner",
            schedule_sha256="schedule",
            updates=2,
            source_git_head="head",
        )


def test_partitioned_preparation_finalizer_checks_and_merges_exact_coverage(tmp_path):
    identity = {"source_git_head": "head", "schedule_sha256": "schedule"}
    schedule = {"entries": [{"landpoint_id": f"{update:03d}.0-071.0", "year": 1960 + update} for update in range(1, 5)]}
    assert _worker_updates(4, 0, 2) == (1, 3)
    assert _worker_updates(4, 1, 2) == (2, 4)

    partials = tmp_path / "partials"
    for worker_index in range(2):
        records = []
        for update in _worker_updates(4, worker_index, 2):
            entry = schedule["entries"][update - 1]
            artifact = PreparedUpdateArtifact(
                schema_version=PREPARED_ARTIFACT_SCHEMA,
                schedule_entry=entry,
                trace_signature="trace",
                batch={"state": np.asarray([update])},
                day_inputs=(np.asarray([update]),),
            )
            record = write_prepared_artifact(tmp_path / f"update-{update:04d}.pkl", artifact) | {
                "update": update,
                "landpoint_id": entry["landpoint_id"],
                "year": entry["year"],
                "trace_signature": "trace",
            }
            records.append(record)
        atomic_json(
            partials / f"worker-{worker_index:04d}.json",
            {
                "schema_version": PREPARED_MANIFEST_SCHEMA,
                "status": "passed",
                "identity": identity,
                "updates": 4,
                "worker": {"index": worker_index, "count": 2},
                "entries": records,
                "teacher_data_regenerated": False,
                "historical_v5_used": False,
            },
        )

    merged = _finalize_worker_progress(tmp_path, identity, schedule, 4, 2)
    assert merged["status"] == "passed"
    assert merged["worker_count"] == 2
    assert [record["update"] for record in merged["entries"]] == [1, 2, 3, 4]
    assert sha256_file(tmp_path / "prepared_manifest.json") == merged["manifest_sha256"]

    partial = json.loads((partials / "worker-0001.json").read_text())
    partial["entries"] = list(reversed(partial["entries"]))
    atomic_json(partials / "worker-0001.json", partial)
    with pytest.raises(ValueError, match="coverage or identity drift"):
        _finalize_worker_progress(tmp_path, identity, schedule, 4, 2)


def test_cpu_preparer_launcher_is_bounded_and_workspace_clean():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "prepare_successor_small_fit_batches.sh").read_text()

    assert "$ROOT/.venvs/orcjax_cpu/bin/python" in launcher
    assert "JAX_PLATFORMS=cpu" in launcher
    assert "PYTHONHASHSEED=0" in launcher
    assert "ORCHIDEE_DATA_ROOT=$ROOT/runtime/data" in launcher
    assert "ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets" in launcher
    assert 'test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"' in launcher
    assert "prepare_successor_small_fit_batches" in launcher


def test_extended_runtime_forwards_explicit_cpu_backend(monkeypatch):
    marker = object()

    class BackendObserved(RuntimeError):
        pass

    def backend_initializer():
        return marker

    def fake_load_runtime(_args, *, backend_initializer):
        assert backend_initializer() is marker
        raise BackendObserved

    monkeypatch.setattr(fast_probe, "_load_runtime", fake_load_runtime)
    with pytest.raises(BackendObserved):
        fast_probe._load_extended_runtime(
            SimpleNamespace(),
            backend_initializer=backend_initializer,
        )


def test_frozen_coefficients_bind_source_report_and_first_artifact(tmp_path):
    components = tuple(LONG_HORIZON_COMPONENTS)
    targets = dict(zip(components, (0.1, 0.15, 0.1, 0.15), strict=True))
    resolved = dict(zip(components, (0.02, 0.03, 0.004, 0.006), strict=True))
    preflight = {
        "anchor_gradient_norm": 10.0,
        "component_gradient_norms": {name: 2.0 + index for index, name in enumerate(components)},
        "component_losses": {name: 3.0 + index for index, name in enumerate(components)},
        "resolved_coefficients": resolved,
        "split_label": "train/train",
        "target_gradient_ratios": targets,
    }
    report_path = tmp_path / "runner_report.json"
    atomic_json(
        report_path,
        {
            "status": "passed",
            "run_kind": "bounded_reuse_probe",
            "identity": {
                "git_head": "source-head",
                "runner_contract_sha256": "runner",
                "schedule_sha256": "source-schedule",
                "successor_candidate_contract_sha256": "candidate",
                "fast_owner_contract_sha256": "owner",
                "prepared_manifest_sha256": "prepared-manifest",
            },
            "training_updates_per_arm": 2,
            "coefficient_preflight": preflight,
        },
    )
    payload = {
        "schema_version": frozen_coefficients.SCHEMA_VERSION,
        "status": "accepted",
        "identity": {
            "runner_contract_sha256": "runner",
            "schedule_sha256": "source-schedule",
            "successor_candidate_contract_sha256": "candidate",
            "fast_owner_contract_sha256": "owner",
            "first_prepared_artifact_sha256": "artifact",
        },
        "source_evidence": {
            "git_head": "source-head",
            "runner_report_sha256": sha256_file(report_path),
            "prepared_manifest_sha256": "prepared-manifest",
            "training_updates_per_arm": 2,
        },
        "objective": {
            "components": list(components),
            "split_label": "train/train",
            "gradient_ratio_targets": targets,
            "coefficient_minimum": 0.001,
            "coefficient_maximum": 10.0,
            "resolved_coefficients": resolved,
            "same_coefficients_for_both_arms": True,
            "recalibrate_during_fit": False,
            "validation_weight_tuning_allowed": False,
        },
    }
    payload["contract_sha256"] = frozen_coefficients._canonical_sha256(payload)
    contract_path = tmp_path / "coefficients.json"
    atomic_json(contract_path, payload)

    loaded = frozen_coefficients.load_frozen_successor_coefficients(
        contract_path,
        source_report_path=report_path,
        runner_contract_sha256="runner",
        schedule_sha256="production-schedule",
        successor_contract_sha256="candidate",
        fast_owner_contract_sha256="owner",
        first_prepared_artifact_sha256="artifact",
        components=components,
        gradient_ratio_targets=targets,
        coefficient_minimum=0.001,
        coefficient_maximum=10.0,
    )
    assert loaded.coefficients == resolved

    # The frozen coefficients are derived from the exact first artifact, not
    # from the later unused entries in the source probe's two-update schedule.
    assert payload["identity"]["schedule_sha256"] != "production-schedule"

    report_path.write_text(report_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source report hash drift"):
        frozen_coefficients.load_frozen_successor_coefficients(
            contract_path,
            source_report_path=report_path,
            runner_contract_sha256="runner",
            schedule_sha256="production-schedule",
            successor_contract_sha256="candidate",
            fast_owner_contract_sha256="owner",
            first_prepared_artifact_sha256="artifact",
            components=components,
            gradient_ratio_targets=targets,
            coefficient_minimum=0.001,
            coefficient_maximum=10.0,
        )


def test_final_evaluation_reuses_compiled_train_step():
    source = Path(__file__).resolve().parents[2] / "scripts" / "hpc" / "run_successor_small_fit.py"
    runner = source.read_text(encoding="utf-8")

    assert '"final_evaluation_reuses_train_step": True' in runner
    assert '"optimizer_result_discarded": True' in runner
    assert "np.asarray(0.0, dtype=np.float64)" in runner


def test_production_asset_admission_probe_is_cpu_only_and_zero_update():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "hpc" / "probe_successor_production_admission.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts" / "hpc" / "probe_successor_production_admission.sh").read_text(
        encoding="utf-8"
    )

    assert "backend_initializer=_cpu_backend" in source
    assert "admit_prepared_manifest_for_checkpoint_fit(" in source
    assert "load_frozen_successor_coefficients(" in source
    assert '"training_updates_executed": 0' in source
    assert '"checkpoint_written": False' in source
    assert "test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04" in launcher
    assert "JAX_PLATFORMS=cpu" in launcher
    assert "sbatch" not in launcher
