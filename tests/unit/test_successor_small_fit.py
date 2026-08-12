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
from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
    MaskedTrajectory,
)
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
    _batched_rollout_objective_components,
    build_successor_small_fit_schedule,
    build_successor_validation_panel,
    initialize_successor_adamw,
    load_successor_small_fit_contract,
    successor_adamw_update,
    successor_learning_rate,
    successor_small_fit_loss,
    successor_teacher_physical_outputs,
    verify_successor_small_fit_schedule,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
    write_staged_best_checkpoint,
    write_staged_recovery,
)
from research.daily_coarse_graining.daily_operator.successor_staged_training import (
    advance_staged_validation_state,
    validation_primary_composite,
)
from research.daily_coarse_graining.daily_operator.successor_training import (
    LONG_HORIZON_COMPONENTS,
    BiomassProductivityTrajectories,
    GradientRatioPreflightError,
    calibrate_gradient_ratio_coefficients,
)
from research.daily_coarse_graining.daily_operator.successor_training_assets import (
    load_successor_training_assets,
)
from scripts.dev.audit_successor_multibatch_activity import _finite_summary
from scripts.dev.freeze_successor_training_assets import _freeze_coefficients
from scripts.hpc.prepare_successor_small_fit_batches import (
    _finalize_worker_progress,
    _worker_updates,
)
from scripts.hpc.probe_successor_multibatch_coefficients import (
    _aggregate_coefficient_records,
)
from scripts.hpc.run_successor_small_fit import (
    _compile_reuse_passed,
    _publication_identity_for_recovery,
    _retained_tail_trace_signature,
    _teacher_replay_training_policy,
    _validate_update_count,
)
from scripts.hpc.run_successor_small_fit_probe import _leaf_failures
from scripts.hpc.successor_rollout_screen_prepared import (
    PREPARED_ARTIFACT_SCHEMA as ROLLOUT_PREPARED_ARTIFACT_SCHEMA,
)
from scripts.hpc.successor_rollout_screen_prepared import (
    PreparedRolloutScreenArtifact,
    write_prepared_rollout_artifact,
)
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
from scripts.hpc.successor_staged_prepared import (
    STAGED_PREPARED_MANIFEST_SCHEMA,
    admit_staged_prepared_artifacts,
    load_staged_prepared_manifest,
    stage_update_range,
    staged_worker_updates,
)
from scripts.hpc.successor_staged_validation_prepared import (
    STAGED_VALIDATION_PREPARED_SCHEMA,
    load_staged_validation_prepared,
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


def test_staged_schedule_is_prefix_stable_and_ranges_are_incremental():
    root = Path(__file__).resolve().parents[2]
    contract = load_successor_small_fit_contract(
        root / "manifests" / "coarse_graining" / "successor_training_staged_v3.json"
    )
    references = tuple(
        _Reference(
            _Parent(f"{landpoint:03d}.0-071.0", year),
            sha256=f"{landpoint + year:064x}"[-64:],
        )
        for landpoint in range(1, 601)
        for year in range(1961, 1963)
    )
    first = build_successor_small_fit_schedule(references, contract, updates=500)
    second = build_successor_small_fit_schedule(references, contract, updates=1000)

    assert first["entries"] == second["entries"][:500]
    assert stage_update_range(contract.sampling["stage_milestones"], 500) == (1, 500)
    assert stage_update_range(contract.sampling["stage_milestones"], 1000) == (501, 1000)
    assert stage_update_range(contract.sampling["stage_milestones"], 10000) == (5001, 10000)
    assert staged_worker_updates(501, 506, 0, 2) == (501, 503, 505)
    assert staged_worker_updates(501, 506, 1, 2) == (502, 504, 506)


def test_training_sufficiency_contract_expands_data_exposure_without_changing_architecture():
    root = Path(__file__).resolve().parents[2]
    contract = load_successor_small_fit_contract(
        root / "manifests" / "coarse_graining" / "successor_training_sufficiency_v2.json"
    )

    assert contract.sampling["maximum_updates_per_arm"] == 10000
    assert contract.sampling["one_day_batch_size"] == 32
    assert contract.sampling["rollout_batch_size"] == 4
    assert contract.payload["matched_arms"] == [
        "source_routed_process_attention_v2",
        "source_routed_graph_mlp_control_v2",
    ]
    assert contract.payload["objective"]["bias_reduction_axis"] == ("joint_across_rollout_windows")
    assert contract.payload["objective"]["coefficient_aggregation"] == (
        "shared_median_gradient_ratio"
    )
    assert contract.payload["gates"]["budget_absolute_tolerance"] == 2e-12


def test_staged_validation_panel_is_deterministic_and_excludes_formal_windows():
    root = Path(__file__).resolve().parents[2]
    contract = load_successor_small_fit_contract(
        root / "manifests" / "coarse_graining" / "successor_training_staged_v3.json"
    )
    split_references = []
    for slice_id, (spatial, temporal) in {
        "temporal": ("train", "validation"),
        "spatial": ("validation", "train"),
        "joint": ("validation", "validation"),
    }.items():
        split_references.extend(
            _Reference(
                _Parent(
                    f"{slice_id}-{ordinal:02d}",
                    2005 + ordinal % 3,
                    spatial_split=spatial,
                    temporal_split=temporal,
                ),
                sha256=f"{slice_id}-{ordinal}",
            )
            for ordinal in range(10)
        )
    excluded = {
        "plan_sha256": contract.payload["validation"][
            "excluded_formal_window_plan_sha256"
        ],
        "slices": {
            slice_id: {
                "entries": [
                    {
                        "landpoint_id": f"{slice_id}-00",
                        "year": 2005,
                        "start_rows": [1, 31],
                    }
                ]
            }
            for slice_id in ("temporal", "spatial", "joint")
        },
    }
    first = build_successor_validation_panel(
        split_references,
        contract,
        excluded_window_plan=excluded,
    )
    second = build_successor_validation_panel(
        tuple(reversed(split_references)),
        contract,
        excluded_window_plan=excluded,
    )

    assert first == second
    assert first["sealed_test_used"] is False
    assert first["milestones"] == [500, 1000, 2000, 5000, 10000]
    for slice_id, cell in first["slices"].items():
        assert len(cell["entries"]) == 8
        assert len({entry["landpoint_id"] for entry in cell["entries"]}) == 8
        assert all(len(entry["start_rows"]) == 2 for entry in cell["entries"])
        assert all(
            not (
                entry["landpoint_id"] == f"{slice_id}-00"
                and entry["year"] == 2005
                and start in (1, 31)
            )
            for entry in cell["entries"]
            for start in entry["start_rows"]
        )


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


def test_rollout_bias_is_computed_across_windows_instead_of_duplicating_mse():
    predicted = BiomassProductivityTrajectories(
        fluxes={
            "gpp": MaskedTrajectory(
                value=jnp.asarray([[1.0], [-1.0]], dtype=jnp.float64),
                defined=jnp.ones((2, 1), dtype=jnp.bool_),
            )
        },
        stocks={
            "live": MaskedTrajectory(
                value=jnp.asarray([[1.0], [-1.0]], dtype=jnp.float64),
                defined=jnp.ones((2, 1), dtype=jnp.bool_),
            )
        },
    )
    expected = BiomassProductivityTrajectories(
        fluxes={
            "gpp": MaskedTrajectory(
                value=jnp.zeros((2, 1), dtype=jnp.float64),
                defined=jnp.ones((2, 1), dtype=jnp.bool_),
            )
        },
        stocks={
            "live": MaskedTrajectory(
                value=jnp.zeros((2, 1), dtype=jnp.float64),
                defined=jnp.ones((2, 1), dtype=jnp.bool_),
            )
        },
    )

    components = _batched_rollout_objective_components(
        predicted,
        expected,
        flux_scales={"gpp": 1.0},
        stock_scales={"live": 1.0},
        horizons=(1,),
    )

    np.testing.assert_allclose(np.asarray(components), [1.0, 0.0, 1.0, 0.0])


def test_gradient_ratio_interval_failure_reports_the_unclamped_coefficient():
    parameters = {"weight": jnp.asarray(1.0, dtype=jnp.float64)}
    components = {name: (lambda values: 1000.0 * values["weight"]) for name in LONG_HORIZON_COMPONENTS}

    with pytest.raises(GradientRatioPreflightError) as captured:
        calibrate_gradient_ratio_coefficients(
            parameters,
            lambda values: values["weight"] ** 2,
            components,
            target_gradient_ratios={name: 0.1 for name in LONG_HORIZON_COMPONENTS},
            coefficient_minimum=0.001,
            coefficient_maximum=10.0,
            split_label="train/train",
        )

    assert captured.value.diagnostic["resolved_coefficient"] == pytest.approx(0.0002)
    assert captured.value.diagnostic["coefficient_minimum"] == 0.001


def test_multibatch_coefficients_use_one_shared_lower_quantile_across_arms_and_batches():
    records = [
        {
            "anchor_gradient_norm": 10.0,
            "component_gradient_norms": {name: value for name in LONG_HORIZON_COMPONENTS},
        }
        for value in (10.0, 20.0, 30.0, 1000.0)
    ]
    result = _aggregate_coefficient_records(
        records,
        component_names=LONG_HORIZON_COMPONENTS,
        targets={name: 0.1 for name in LONG_HORIZON_COMPONENTS},
        coefficient_minimum=1e-6,
        coefficient_maximum=10.0,
        coefficient_aggregation="shared_lower_quantile_gradient_ratio",
        coefficient_quantile=0.10,
    )

    expected = np.quantile(
        [0.1, 0.05, 1.0 / 30.0, 0.001],
        0.10,
        method="lower",
    )
    assert result["record_count"] == 4
    assert result["coefficient_quantile"] == 0.10
    assert result["quantile_method"] == "lower"
    assert result["coefficient_aggregation"] == ("shared_lower_quantile_gradient_ratio")
    assert all(value == pytest.approx(expected) for value in result["resolved_coefficients"].values())


def test_multibatch_v2_coefficients_remain_median_without_quantile_field():
    records = [
        {
            "anchor_gradient_norm": 10.0,
            "component_gradient_norms": {name: value for name in LONG_HORIZON_COMPONENTS},
        }
        for value in (10.0, 20.0, 30.0, 1000.0)
    ]
    result = _aggregate_coefficient_records(
        records,
        component_names=LONG_HORIZON_COMPONENTS,
        targets={name: 0.1 for name in LONG_HORIZON_COMPONENTS},
        coefficient_minimum=1e-6,
        coefficient_maximum=10.0,
        coefficient_aggregation="shared_median_gradient_ratio",
    )

    expected = np.median([0.1, 0.05, 1.0 / 30.0, 0.001])
    assert "coefficient_quantile" not in result
    assert all(value == pytest.approx(expected) for value in result["resolved_coefficients"].values())


def test_multibatch_coefficient_wrapper_is_free_gpu_and_writes_no_checkpoint():
    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "scripts" / "hpc" / "run_successor_multibatch_coefficient_probe.sh").read_text()

    assert "PREPARED_MANIFEST=${PREPARED_MANIFEST:?" in wrapper
    assert "SINGULARITYENV_JAX_PLATFORMS=cuda,cpu" in wrapper
    assert "sbatch" not in wrapper
    assert "checkpoint" not in wrapper.lower()


def test_multibatch_activity_summary_respects_the_defined_mask():
    summary = _finite_summary(
        np.asarray([1.0, 1.0e20, np.nan, -3.0]),
        np.asarray([True, False, True, True]),
    )

    assert summary == {
        "count": 2,
        "nonzero_count": 2,
        "absolute_mean": 2.0,
        "rms": pytest.approx(np.sqrt(5.0)),
        "absolute_maximum": 3.0,
    }


def test_low_gradient_path_wrapper_is_free_gpu_and_zero_update():
    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "scripts" / "hpc" / "run_successor_low_gradient_path_probe.sh").read_text()
    probe = (root / "scripts" / "hpc" / "probe_successor_low_gradient_path.py").read_text()

    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in wrapper
    assert 'UPDATES=${UPDATES:-"12 15"}' in wrapper
    assert "sbatch" not in wrapper
    assert '"training_updates": 0' in probe
    assert '"checkpoint_written": False' in probe


def test_multibatch_coefficient_probe_reuses_rollout_for_hard_gates():
    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "scripts" / "hpc" / "run_successor_multibatch_coefficient_probe.sh").read_text()
    probe = (root / "scripts" / "hpc" / "probe_successor_multibatch_coefficients.py").read_text()

    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in wrapper
    assert "has_aux=True" in probe
    assert '"hard_gate_counts": hard_counts_host' in probe
    assert '"hard_failure_count": hard_failure_count' in probe


def test_staged_training_assets_rederive_coefficients_and_are_hash_bound(tmp_path):
    root = Path(__file__).resolve().parents[2]
    contract = load_successor_small_fit_contract(
        root / "manifests" / "coarse_graining" / "successor_training_staged_v3.json"
    )
    components = tuple(LONG_HORIZON_COMPONENTS)
    measurements = [
        {
            "anchor_gradient_norm": 10.0,
            "component_gradient_norms": {
                name: float(record + index + 1)
                for index, name in enumerate(components)
            },
        }
        for record in range(32)
    ]
    resolved = _aggregate_coefficient_records(
        measurements,
        component_names=components,
        targets=contract.payload["objective"]["gradient_ratio_targets"],
        coefficient_minimum=contract.payload["objective"]["coefficient_minimum"],
        coefficient_maximum=contract.payload["objective"]["coefficient_maximum"],
        coefficient_aggregation="shared_lower_quantile_gradient_ratio",
        coefficient_quantile=0.10,
    )["resolved_coefficients"]
    prepared_path = tmp_path / "prepared.json"
    atomic_json(prepared_path, {"identity": {"schedule_sha256": "schedule"}})
    report_path = tmp_path / "report.json"
    atomic_json(
        report_path,
        {
            "status": "passed",
            "git_head": "a" * 40,
            "runner_contract_sha256": contract.payload["identity"][
                "coefficient_probe_runner_contract_sha256"
            ],
            "prepared_manifest_sha256": sha256_file(prepared_path),
            "training_updates": 0,
            "checkpoint_written": False,
            "sealed_test_used": False,
            "historical_v5_used": False,
            "teacher_data_regenerated": False,
            "hard_failure_count": 0,
            "measurement_record_count": 32,
            "coefficient_aggregation": "shared_median_gradient_ratio",
            "measurements": measurements,
            "aggregation": {
                "record_count": 32,
                "resolved_coefficients": {name: 9.0 for name in components},
            },
        },
    )
    coefficient_payload = _freeze_coefficients(
        json.loads(report_path.read_text()),
        report_path=report_path,
        contract=contract,
        prepared_manifest_path=prepared_path,
    )
    coefficient_path = tmp_path / "coefficients.json"
    atomic_json(coefficient_path, coefficient_payload)

    panel_payload = {
        "schema_version": "successor_training_validation_panel_v1",
        "runner_contract_sha256": contract.sha256,
        "reported_horizons_days": [7, 30],
        "milestones": [500, 1000, 2000, 5000, 10000],
        "excluded_formal_window_plan_file_sha256": contract.payload["validation"][
            "excluded_formal_window_plan_file_sha256"
        ],
        "excluded_formal_window_plan_sha256": contract.payload["validation"][
            "excluded_formal_window_plan_sha256"
        ],
        "slices": {name: {} for name in ("temporal", "spatial", "joint")},
        "sealed_test_used": False,
    }
    panel_payload["panel_sha256"] = small_fit._canonical_sha256(panel_payload)
    panel_path = tmp_path / "panel.json"
    atomic_json(panel_path, panel_payload)

    assets = load_successor_training_assets(
        coefficient_path,
        panel_path,
        coefficient_report_path=report_path,
        prepared_manifest_path=prepared_path,
        runner_contract=contract,
        components=components,
    )
    assert assets.coefficients == resolved


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
    assert "RUNNER_CONTRACT=${RUNNER_CONTRACT:-" in launcher


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


def test_compile_reuse_accepts_one_post_recovery_array_specialization():
    arms = ("attention", "control")

    def record(update, before, after, new):
        return {
            "update": update,
            "trace_signature": "trace",
            "teacher_replay": {
                "jit_cache_size_before": 1,
                "jit_cache_size_after": 1,
                "new_jit_executables": 0,
            },
            "arms": {
                arm: {
                    "jit_cache_size_before": before,
                    "jit_cache_size_after": after,
                    "new_jit_executables": new,
                }
                for arm in arms
            },
        }

    records = [
        record(1, 0, 1, 1),
        record(2, 1, 1, 0),
        record(3, 0, 1, 1),
        record(4, 1, 2, 1),
        record(5, 2, 2, 0),
    ]
    assert _compile_reuse_passed(records, trace_signature="trace", arms=arms)

    records.append(record(6, 2, 3, 1))
    assert not _compile_reuse_passed(records, trace_signature="trace", arms=arms)


def test_completed_cross_commit_recovery_keeps_training_identity_for_publication():
    execution = {"git_head": "e" * 40, "schedule_sha256": "schedule"}
    recovery = {"git_head": "r" * 40, "schedule_sha256": "schedule"}

    assert (
        _publication_identity_for_recovery(
            execution,
            recovery,
            resume_update=500,
            requested_updates=500,
            diagnostic=False,
        )
        == recovery
    )
    with pytest.raises(ValueError, match="may only diagnose or finalize"):
        _publication_identity_for_recovery(
            execution,
            recovery,
            resume_update=155,
            requested_updates=500,
            diagnostic=False,
        )
    assert (
        _publication_identity_for_recovery(
            execution,
            recovery,
            resume_update=155,
            requested_updates=500,
            diagnostic=True,
        )
        == execution
    )


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
    assert "--diagnose-boundary-window $DIAGNOSE_BOUNDARY_WINDOW" in launcher
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


def test_prepared_batch_wrapper_exposes_bounded_artifact_compression():
    root = Path(__file__).resolve().parents[2]
    wrapper = (root / "scripts" / "hpc" / "prepare_successor_small_fit_batches.sh").read_text()

    assert "ARTIFACT_COMPRESSION=${ARTIFACT_COMPRESSION:-none}" in wrapper
    assert "none|gzip)" in wrapper
    assert 'EXTRA_ARGS=(--artifact-compression "$ARTIFACT_COMPRESSION")' in wrapper


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
    parameters = {arm: {"weight": np.asarray([index + 1.0, index + 2.0])} for index, arm in enumerate(arms)}
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
        assert loaded[arm]["parameter_tree_schema"] == [{"path": "['weight']", "shape": [2], "dtype": "float64"}]

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


def test_staged_prepared_manifest_admits_global_update_interval(tmp_path):
    schedule_entries = [
        {
            "update": update,
            "landpoint_id": f"{update:03d}.0-071.0",
            "year": 1960 + update,
        }
        for update in range(1, 5)
    ]
    records = []
    manifest_path = tmp_path / "prepared_manifest.json"
    for update in (3, 4):
        entry = schedule_entries[update - 1]
        artifact = PreparedUpdateArtifact(
            PREPARED_ARTIFACT_SCHEMA,
            entry,
            "trace",
            {"state": np.asarray([update])},
            (np.asarray([update]),),
        )
        records.append(
            {
                "update": update,
                "trace_signature": "trace",
                **write_prepared_artifact(
                    tmp_path / f"update-{update:05d}.pkl.gz",
                    artifact,
                    compression="gzip",
                ),
            }
        )
    payload = {
        "schema_version": STAGED_PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "identity": {
            "staged_training_contract_sha256": "contract",
            "full_schedule_sha256": "schedule",
        },
        "start_update": 3,
        "end_update": 4,
        "artifact_count": 2,
        "entries": records,
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
    }
    atomic_json(manifest_path, payload)

    loaded = load_staged_prepared_manifest(
        manifest_path,
        staged_training_contract_sha256="contract",
        full_schedule_sha256="schedule",
        start_update=3,
        end_update=4,
    )
    admit_staged_prepared_artifacts(
        manifest_path,
        loaded,
        full_schedule={"entries": schedule_entries},
    )
    assert loaded["manifest_sha256"] == sha256_file(manifest_path)

    payload["entries"] = list(reversed(records))
    atomic_json(manifest_path, payload)
    with pytest.raises(ValueError, match="identity drift"):
        load_staged_prepared_manifest(
            manifest_path,
            staged_training_contract_sha256="contract",
            full_schedule_sha256="schedule",
            start_update=3,
            end_update=4,
        )


def test_staged_validation_prepared_admits_fixed_panel_and_artifacts(tmp_path):
    slices = {}
    records = []
    ordinal = 0
    for slice_id in ("temporal", "spatial", "joint"):
        entry = {
            "landpoint_id": slice_id,
            "year": 2006,
            "parent_shard_sha256": "a" * 64,
            "typed_shard_sha256": "b" * 64,
            "day_count": 365,
            "start_rows": [1, 31],
        }
        slices[slice_id] = {"entries": [entry]}
        ordinal += 1
        artifact = PreparedRolloutScreenArtifact(
            ROLLOUT_PREPARED_ARTIFACT_SCHEMA,
            slice_id,
            entry,
            "trace",
            {"state": np.asarray([ordinal])},
            (np.asarray([ordinal]),),
        )
        records.append(
            {
                "ordinal": ordinal,
                "slice_id": slice_id,
                "window_entry": entry,
                "trace_signature": "trace",
                **write_prepared_rollout_artifact(
                    tmp_path / f"validation-{ordinal:03d}.pkl",
                    artifact,
                ),
            }
        )
    panel = {
        "schema_version": "successor_training_validation_panel_v1",
        "runner_contract_sha256": "contract",
        "slices": slices,
        "sealed_test_used": False,
    }
    panel["panel_sha256"] = small_fit._canonical_sha256(panel)
    panel_path = tmp_path / "validation_panel.json"
    atomic_json(panel_path, panel)
    manifest = {
        "schema_version": STAGED_VALIDATION_PREPARED_SCHEMA,
        "status": "passed",
        "staged_training_contract_sha256": "contract",
        "validation_panel": panel_path.name,
        "validation_panel_sha256": panel["panel_sha256"],
        "validation_panel_file_sha256": sha256_file(panel_path),
        "artifact_count": 3,
        "entries": records,
        "sealed_test_used": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
    }
    manifest_path = tmp_path / "prepared_manifest.json"
    atomic_json(manifest_path, manifest)

    loaded, loaded_panel = load_staged_validation_prepared(
        manifest_path,
        staged_training_contract_sha256="contract",
        validation_panel_sha256=panel["panel_sha256"],
    )
    assert loaded["manifest_sha256"] == sha256_file(manifest_path)
    assert loaded_panel == panel

    records[0]["slice_id"] = "joint"
    atomic_json(manifest_path, manifest | {"entries": records})
    with pytest.raises(ValueError, match="order drift"):
        load_staged_validation_prepared(
            manifest_path,
            staged_training_contract_sha256="contract",
            validation_panel_sha256=panel["panel_sha256"],
        )


def _staged_validation_report(score, *, hard_failures=0):
    keys = (
        "cumulative.gpp",
        "cumulative.npp",
        "cumulative.maintenance_respiration",
        "terminal.aboveground_biomass_carbon",
        "terminal.belowground_biomass_carbon",
        "terminal.total_live_biomass_carbon",
    )
    return {
        "slices": {
            slice_id: {
                "horizons": {
                    str(horizon): {
                        "keys": {
                            key: {"normalized_rmse": float(score)} for key in keys
                        }
                    }
                    for horizon in (7, 30)
                },
                "hard_counts": {"budget_residual_failure": int(hard_failures)},
            }
            for slice_id in ("temporal", "spatial", "joint")
        }
    }


def test_staged_validation_state_stops_only_after_both_arms_plateau_twice():
    root = Path(__file__).resolve().parents[2]
    contract = load_successor_small_fit_contract(
        root / "manifests" / "coarse_graining" / "successor_training_staged_v3.json"
    )
    attention, control = contract.payload["matched_arms"]
    state = advance_staged_validation_state(
        None,
        milestone=500,
        arm_reports={
            attention: _staged_validation_report(1.0),
            control: _staged_validation_report(1.0),
        },
        contract=contract,
    )
    assert validation_primary_composite(
        _staged_validation_report(1.0),
        primary_outputs=contract.payload["validation"]["primary_outputs"],
    ) == pytest.approx(1.0)
    assert state["stop_training"] is False

    state = advance_staged_validation_state(
        state,
        milestone=1000,
        arm_reports={
            attention: _staged_validation_report(0.98),
            control: _staged_validation_report(0.995),
        },
        contract=contract,
    )
    assert state["arms"][attention]["best_milestone"] == 1000
    assert state["arms"][control]["best_milestone"] == 500
    assert state["stop_training"] is False

    state = advance_staged_validation_state(
        state,
        milestone=2000,
        arm_reports={
            attention: _staged_validation_report(0.975),
            control: _staged_validation_report(1.01),
        },
        contract=contract,
    )
    assert state["stop_training"] is False
    state = advance_staged_validation_state(
        state,
        milestone=5000,
        arm_reports={
            attention: _staged_validation_report(0.98),
            control: _staged_validation_report(1.02),
        },
        contract=contract,
    )
    assert state["stop_training"] is True
    assert state["stop_reason"] == "both_arms_plateaued_twice"
    assert state["arms"][attention]["best_milestone"] == 1000


def test_staged_validation_hard_failure_stops_without_selecting_checkpoint():
    root = Path(__file__).resolve().parents[2]
    contract = load_successor_small_fit_contract(
        root / "manifests" / "coarse_graining" / "successor_training_staged_v3.json"
    )
    reports = {
        arm: _staged_validation_report(1.0, hard_failures=int(index == 0))
        for index, arm in enumerate(contract.payload["matched_arms"])
    }
    state = advance_staged_validation_state(
        None,
        milestone=500,
        arm_reports=reports,
        contract=contract,
    )

    assert state["stop_training"] is True
    assert state["stop_reason"] == "validation_hard_failure"
    assert state["arms"][contract.payload["matched_arms"][0]]["best_milestone"] is None


def test_staged_recovery_and_best_checkpoint_round_trip(tmp_path):
    arms, parameters, optimizers = _checkpoint_fixture()
    identity = {"full_schedule_sha256": "schedule", "validation_panel_sha256": "panel"}
    validation_state = {
        "completed_milestones": [500],
        "arms": {
            arm: {
                "best_milestone": 500,
                "best_primary_composite": 1.0 + index,
            }
            for index, arm in enumerate(arms)
        },
    }
    recovery_path = tmp_path / "recovery.pkl"
    write_staged_recovery(
        recovery_path,
        identity=identity,
        parameters=parameters,
        optimizers=optimizers,
        completed_updates=743,
        parameter_counts={arm: 2 for arm in arms},
        validation_state=validation_state,
        best_parameters=parameters,
        expected_arms=arms,
    )
    loaded = load_staged_recovery(
        recovery_path,
        identity=identity,
        expected_arms=arms,
        parameter_counts={arm: 2 for arm in arms},
        minimum_completed_updates=500,
        maximum_completed_updates=1000,
    )
    assert loaded["completed_updates"] == 743
    assert loaded["validation_state"] == validation_state
    with pytest.raises(ValueError, match="identity drift"):
        load_staged_recovery(
            recovery_path,
            identity=identity,
            expected_arms=arms,
            parameter_counts={arm: 2 for arm in arms},
            minimum_completed_updates=744,
            maximum_completed_updates=1000,
        )

    best = write_staged_best_checkpoint(
        tmp_path / "best.pkl",
        identity=identity,
        parameters=parameters,
        parameter_counts={arm: 2 for arm in arms},
        validation_state=validation_state,
        expected_arms=arms,
    )
    assert best["best_milestones"] == {arm: 500 for arm in arms}
    assert best["promotion_eligible"] is False


def test_staged_runner_uses_prevalidated_artifacts_without_per_update_teacher_replay():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts" / "hpc" / "run_successor_staged_training.py").read_text()

    assert "_teacher_replay_gate_with_transition" not in runner
    assert "checkpoint_handoff" not in runner
    assert "load_successor_training_assets(" in runner
    assert "load_staged_validation_prepared(" in runner
    assert "advance_staged_validation_state(" in runner
    assert '"reconstructs_48_forcing_records": False' in runner
    assert '"executes_48_step_state_scan": False' in runner


def test_gzip_prepared_update_is_deterministic_and_round_trips(tmp_path):
    entry = {"landpoint_id": "001.0-071.0", "year": 1961, "anchor_starts": [1]}
    artifact = PreparedUpdateArtifact(
        schema_version=PREPARED_ARTIFACT_SCHEMA,
        schedule_entry=entry,
        trace_signature="trace",
        batch={"state": np.zeros((128, 128), dtype=np.float64)},
        day_inputs=(np.asarray([3.0]),),
    )
    paths = (tmp_path / "left.pkl.gz", tmp_path / "right.pkl.gz")
    records = [
        write_prepared_artifact(path, artifact, compression="gzip") | {"update": 1, "trace_signature": "trace"}
        for path in paths
    ]

    assert records[0]["compression"] == "gzip"
    assert records[0]["sha256"] == records[1]["sha256"]
    assert records[0]["size_bytes"] < artifact.batch["state"].nbytes // 10
    loaded = load_prepared_artifact(
        tmp_path / "prepared_manifest.json",
        records[0],
        schedule_entry=entry,
    )
    np.testing.assert_array_equal(loaded.batch["state"], artifact.batch["state"])


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

    assert '"final_evaluation_reuses_train_step": resume_update < requested_updates' in runner
    assert '"final_evaluation_from_completed_recovery": resume_update == requested_updates' in runner
    assert '"optimizer_result_discarded": True' in runner
    assert "np.asarray(0.0, dtype=np.float64)" in runner


def test_production_asset_admission_probe_is_cpu_only_and_zero_update():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "hpc" / "probe_successor_production_admission.py").read_text(encoding="utf-8")
    launcher = (root / "scripts" / "hpc" / "probe_successor_production_admission.sh").read_text(encoding="utf-8")

    assert "backend_initializer=_cpu_backend" in source
    assert "admit_prepared_manifest_for_checkpoint_fit(" in source
    assert "load_frozen_successor_coefficients(" in source
    assert '"training_updates_executed": 0' in source
    assert '"checkpoint_written": False' in source
    assert "test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04" in launcher
    assert "JAX_PLATFORMS=cpu" in launcher
    assert "sbatch" not in launcher
