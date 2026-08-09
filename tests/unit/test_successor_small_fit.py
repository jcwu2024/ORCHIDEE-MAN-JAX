from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import research.daily_coarse_graining.daily_operator.successor_small_fit as small_fit
import scripts.hpc.run_successor_fast_owner_gpu_probe as fast_probe
from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
    DAY_START_DELTA,
    DEFINED_LOGIT,
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
from scripts.hpc.run_successor_small_fit import (
    _retained_tail_trace_signature,
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


def test_production_launcher_preserves_contract_and_dual_backend_boundary():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_successor_small_fit.sh").read_text()

    assert "SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \\" in launcher
    assert "SINGULARITYENV_PYTHONHASHSEED=0 \\" in launcher
    assert '--updates "$UPDATES" \\' in launcher
    assert "UPDATES must lie in [2, 500]" in launcher
    assert "successor_small_fit_runner_v1.json" in launcher
    assert "PREPARED_MANIFEST=${PREPARED_MANIFEST:?" in launcher
    assert '--prepared-manifest "$PREPARED_MANIFEST" \\' in launcher


def test_rejected_paid_launcher_is_removed():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "scripts" / "hpc" / "slurm_successor_small_fit.sh").exists()


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
