from __future__ import annotations

import copy
import hashlib
import json
import pickle
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining import rollout_stability_run
from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalPrediction,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
)
from research.daily_coarse_graining.canonical_training_run import (
    CHECKPOINT_SCHEMA_VERSION,
)
from research.daily_coarse_graining.daily_model_architecture import (
    AXIS_PROCESS_COUPLED_V1,
)
from research.daily_coarse_graining.markov_dataset import (
    MarkovDatasetIndex,
    MarkovShardRef,
)
from research.daily_coarse_graining.rollout_stability_protocol import (
    RolloutStabilityProtocol,
    load_rollout_stability_protocol,
)
from research.daily_coarse_graining.rollout_stability_run import (
    ARM_CHECKPOINT_SCHEMA_VERSION,
    CALIBRATION_SCHEMA_VERSION,
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    MIN_STOMATE_DAILY_CARBON_OWNERS,
    PREFLIGHT_SCHEMA_VERSION,
    _bind_min_stomate_variant_transition,
    _boundary_variant_metrics,
    _daily_carbon_min_stomate_variant,
    _host_hard_constraint_counts,
    _leaf_for_compact_index,
    _state_variant_metrics,
    _write_coefficient_calibration_failure,
    broadcast_retained_tail_day_inputs,
    build_arm_checkpoint,
    build_sampling_schedule,
    calibrate_gradient_coefficients,
    completed_horizon_counts,
    create_execution_manifest,
    make_detached_next_day_gradient_diagnostic,
    make_retained_tail_ad_boundary_jvp_diagnostic,
    retained_tail_trace_signature,
    sample_start_indices,
    verify_arm_checkpoint,
    verify_parent_architecture_assets,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "manifests"
    / "coarse_graining"
    / "canonical_669_rollout_stability_protocol.json"
)
ValueTuple = namedtuple("ValueTuple", ("value",))
DailyCarbonBundles = namedtuple(
    "DailyCarbonBundles",
    ("prescribe_inputs", "alloc_inputs", "post_npp_inputs"),
)
DiagnosticSequence = namedtuple(
    "DiagnosticSequence",
    ("retained_tail_inputs", "year", "day_index"),
)


class _FakeShard:
    def __init__(self, *, year: int):
        self.days = 10
        self.year = year

    def window(self, start: int, horizon: int):
        values = np.arange(start, start + horizon, dtype=np.float64)
        return {
            "initial_state": np.asarray([float(start)]),
            "state_trajectory": np.arange(
                start,
                start + horizon + 1,
                dtype=np.float64,
            )[:, None],
            "teacher_next_state": np.arange(
                start + 1,
                start + horizon + 1,
                dtype=np.float64,
            )[:, None],
            "teacher_fast_day_target": values[:, None],
            "forcing_native": values[:, None, None],
            "parameters": np.ones((horizon, 1)),
            "landpoint_static": np.ones((horizon, 1)),
            "annual_conditions": np.ones((horizon, 1)),
            "year": np.full((horizon,), self.year, dtype=np.int32),
            "day_index": np.arange(start + 1, start + horizon + 1),
            "initial_discrete_state": {
                "flag": np.asarray([start % 2 == 0]),
            },
            "discrete_trajectory": {
                "flag": np.full((horizon + 1, 1), start % 2 == 0),
            },
            "teacher_next_discrete_state": {
                "flag": np.full((horizon, 1), start % 2 == 0),
            },
        }


def test_daily_carbon_min_stomate_variant_is_owner_explicit_and_non_mutating():
    bundles = DailyCarbonBundles(
        prescribe_inputs={"value": 1, "min_stomate": 7.0},
        alloc_inputs={"value": 2},
        post_npp_inputs={"value": 3, "min_stomate": 9.0},
    )

    variant = _daily_carbon_min_stomate_variant(
        bundles,
        ("allocation",),
    )

    assert variant.prescribe_inputs == {"value": 1, "min_stomate": 0.0}
    assert variant.alloc_inputs == {"value": 2, "min_stomate": 1.0e-8}
    assert variant.post_npp_inputs == {"value": 3, "min_stomate": 0.0}
    assert bundles.prescribe_inputs["min_stomate"] == 7.0
    assert bundles.post_npp_inputs["min_stomate"] == 9.0
    assert MIN_STOMATE_DAILY_CARBON_OWNERS == (
        "prescribe",
        "allocation",
        "post_npp_chain",
    )
    with pytest.raises(ValueError, match="unknown min_stomate"):
        _daily_carbon_min_stomate_variant(bundles, ("unknown",))


def test_min_stomate_variant_transition_replaces_builder_only_during_call(
    monkeypatch,
):
    bundles = DailyCarbonBundles(
        prescribe_inputs={},
        alloc_inputs={},
        post_npp_inputs={},
    )
    observed = []

    def original_builder():
        return bundles

    def base_transition(*args, **kwargs):
        del args, kwargs
        result = rollout_stability_run.teacher.stomate_restart_input_bundles()
        observed.append(result)
        return "transition-result"

    monkeypatch.setattr(
        rollout_stability_run.teacher,
        "stomate_restart_input_bundles",
        original_builder,
    )
    monkeypatch.setattr(
        rollout_stability_run,
        "_bind_dynamic_transition",
        lambda resources, runtime: base_transition,
    )

    transition = _bind_min_stomate_variant_transition(
        SimpleNamespace(),
        SimpleNamespace(),
        ("prescribe", "post_npp_chain"),
    )

    assert transition("unused") == "transition-result"
    assert observed[0].prescribe_inputs["min_stomate"] == 1.0e-8
    assert observed[0].alloc_inputs["min_stomate"] == 0.0
    assert observed[0].post_npp_inputs["min_stomate"] == 1.0e-8
    assert (
        rollout_stability_run.teacher.stomate_restart_input_bundles
        is original_builder
    )


def test_state_variant_metrics_reports_only_changed_owner_leaves():
    leaves = (
        SimpleNamespace(
            discrete=False,
            start=0,
            stop=2,
            key="slow.biomass",
            component="slow",
            source_ref="fortran:1-2",
        ),
        SimpleNamespace(
            discrete=False,
            start=2,
            stop=3,
            key="slow.npp",
            component="slow",
            source_ref="fortran:3",
        ),
        SimpleNamespace(
            discrete=True,
            start=None,
            stop=None,
            key="slow.present",
            component="slow",
            source_ref="fortran:4",
        ),
    )
    resources = SimpleNamespace(
        contract=SimpleNamespace(state_leaves=leaves),
        statistics=SimpleNamespace(
            arrays={
                "state": SimpleNamespace(
                    scale=np.asarray([1.0, 2.0, 4.0])
                )
            }
        ),
        process_weighting=SimpleNamespace(
            weights=np.asarray([0.25, 0.25, 0.5])
        ),
    )

    metrics = _state_variant_metrics(
        np.asarray([1.0, 4.0, 8.0]),
        np.asarray([1.0, 0.0, 4.0]),
        resources=resources,
    )

    assert metrics["changed_leaf_count"] == 2
    assert {
        item["key"] for item in metrics["changed_leaves"]
    } == {"slow.biomass", "slow.npp"}
    assert metrics["defined_status_mismatches"] == 0
    assert metrics["maximum_absolute_physical_difference"] == 4.0
    assert metrics["weighted_huber_state_loss"] == pytest.approx(0.625)


def test_boundary_variant_metrics_orders_first_changed_process_boundary():
    baseline = {
        "allocation.biomass": np.asarray([1.0, 2.0]),
        "npp.biomass": np.asarray([3.0, 4.0]),
    }
    values = {
        "allocation.biomass": np.asarray([1.0, 2.0]),
        "npp.biomass": np.asarray([3.0, 14.0]),
    }

    metrics = _boundary_variant_metrics(values, baseline)

    assert metrics["largest_differences"][0]["name"] == "npp.biomass"
    assert (
        metrics["largest_differences"][0]["maximum_absolute_difference"]
        == 10.0
    )
    allocation = next(
        record
        for record in metrics["boundaries"]
        if record["name"] == "allocation.biomass"
    )
    assert allocation["changed_common_values"] == 0


def _canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_retained_tail_ad_boundary_jvp_reports_named_process_tangents():
    def transition(
        initial_state,
        initial_discrete_state,
        target,
        retained_tail_inputs,
        year,
        day_index,
    ):
        del (
            initial_state,
            initial_discrete_state,
            retained_tail_inputs,
            year,
            day_index,
        )
        return target, {
            "allocation.f_alloc": 2.0 * target,
            "npp.bm_alloc": target[:2] ** 2,
        }

    diagnostic = make_retained_tail_ad_boundary_jvp_diagnostic(
        retained_tail_ad_boundary_transition=transition,
    )
    target = jnp.asarray([1.0, 3.0, 5.0])
    values, tangents = diagnostic(
        jnp.zeros(1),
        {"flag": jnp.asarray(False)},
        target,
        DiagnosticSequence(
            retained_tail_inputs=jnp.zeros((1, 1)),
            year=jnp.asarray([1973]),
            day_index=jnp.asarray([7]),
        ),
        jnp.asarray(1, dtype=jnp.int32),
    )

    np.testing.assert_allclose(
        np.asarray(values["allocation.f_alloc"]),
        [2.0, 6.0, 10.0],
    )
    np.testing.assert_allclose(
        np.asarray(tangents["allocation.f_alloc"]),
        [0.0, 2.0, 0.0],
    )
    np.testing.assert_allclose(
        np.asarray(tangents["npp.bm_alloc"]),
        [0.0, 6.0],
    )


def test_detached_next_day_gradient_excludes_prefix_parameter_path():
    statistics = SimpleNamespace(
        arrays={
            name: SimpleNamespace(mean=np.zeros(shape), scale=np.ones(shape))
            for name, shape in {
                "state": (1,),
                "fast_day_target": (1,),
                "forcing_native": (1, 1),
                "parameters": (1,),
                "landpoint_static": (1,),
                "annual_conditions": (1,),
            }.items()
        }
    )
    representation = SimpleNamespace(
        state_indices=(0,),
        dynamic_undefined_indices=(0,),
        dynamic_undefined_fill_values=(np.nan,),
        nonnegative_indices=(),
    )

    def model_apply(parameters, batch):
        batch_size = batch.state.shape[0]
        return CanonicalPrediction(
            normalized_fast_day_target=jnp.broadcast_to(
                parameters["weight"],
                (batch_size, 1),
            ),
            dynamic_undefined_flip_logits=jnp.full((batch_size, 1), -1.0),
        )

    def transition(state, discrete, target, *_):
        return state + target, discrete

    diagnostic = make_detached_next_day_gradient_diagnostic(
        statistics=statistics,
        representation=representation,
        fast_day_weights=jnp.ones(1),
        retained_tail_transition=transition,
        state_weights=jnp.ones(1),
        undefined_loss_weight=0.1,
        model_apply=model_apply,
    )
    sequence = CanonicalMultistepSequence(
        forcing_native=jnp.zeros((2, 1, 1)),
        parameters=jnp.zeros((2, 1)),
        landpoint_static=jnp.zeros((2, 1)),
        annual_conditions=jnp.zeros((2, 1)),
        year=jnp.asarray([1973, 1973]),
        day_index=jnp.asarray([7, 8]),
        teacher_state=jnp.zeros((2, 1)),
        teacher_fast_day_target=jnp.zeros((2, 1)),
        teacher_next_state=jnp.zeros((2, 1)),
        retained_tail_inputs=jnp.zeros((2, 1)),
    )

    (
        value,
        state,
        _,
        parameter_gradient,
        state_gradient,
        retained_tail_state_gradient,
        model_state_gradient,
    ) = diagnostic(
        {"weight": jnp.asarray(0.2)},
        jnp.asarray([0.2]),
        {"flag": jnp.asarray(False)},
        sequence,
    )

    np.testing.assert_allclose(np.asarray(state), [0.4], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(value), 0.18, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(parameter_gradient["weight"]),
        0.6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(state_gradient), [0.6], rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(retained_tail_state_gradient),
        [0.6],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(model_state_gradient),
        [0.0],
        atol=1e-7,
    )


def _small_protocol(*, updates: int = 10) -> RolloutStabilityProtocol:
    loaded = load_rollout_stability_protocol(PROTOCOL_PATH)
    raw = copy.deepcopy(loaded.raw)
    raw["optimization"]["screening_updates"] = updates
    return RolloutStabilityProtocol(
        path=loaded.path,
        raw=raw,
        sha256=f"test-{updates}",
    )


def _index(tmp_path: Path) -> MarkovDatasetIndex:
    references = []
    for landpoint in ("001.0-001.0", "002.0-002.0", "003.0-003.0"):
        for year in range(1961, 1965):
            references.append(
                MarkovShardRef(
                    landpoint_id=landpoint,
                    year=year,
                    spatial_split="train",
                    temporal_split="train",
                    path=tmp_path / f"{landpoint}-{year}.npz",
                    sha256=f"{landpoint}-{year}",
                    contract_sha256="contract",
                )
            )
    references.append(
        MarkovShardRef(
            landpoint_id="test",
            year=2008,
            spatial_split="test",
            temporal_split="test",
            path=tmp_path / "sealed.npz",
            sha256="sealed",
            contract_sha256="contract",
        )
    )
    return MarkovDatasetIndex("dataset", "teacher", "contract", tuple(references))


def test_sampling_schedule_is_balanced_deterministic_and_exact_horizon_mix(tmp_path):
    protocol = _small_protocol(updates=10)
    first = build_sampling_schedule(_index(tmp_path), protocol)
    second = build_sampling_schedule(_index(tmp_path), protocol)

    assert first == second
    assert first["horizon_counts"] == {"1": 4, "3": 3, "7": 2, "30": 1}
    assert first["landpoint_update_count_min"] == 3
    assert first["landpoint_update_count_max"] == 4
    assert all(
        item["landpoint_id"] != "test" for item in first["reference_inventory"]
    )
    assert len(first["entries"]) == 10


def test_sample_start_indices_are_restart_reconstructible_and_without_replacement():
    first = sample_start_indices(
        shard_days=365,
        update=17,
        horizon=30,
        anchor_batch_size=256,
        rollout_batch_size=8,
        seed=20260728,
    )
    second = sample_start_indices(
        shard_days=365,
        update=17,
        horizon=30,
        anchor_batch_size=256,
        rollout_batch_size=8,
        seed=20260728,
    )

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert np.unique(first[0]).size == 256
    assert np.unique(first[1]).size == 8
    assert np.max(first[1]) < 365 - 30 + 1


def test_rollout_preparation_loads_once_and_reuses_only_matching_landpoint_runtime(
    monkeypatch,
    tmp_path,
):
    references = (
        MarkovShardRef(
            landpoint_id="001.0-001.0",
            year=1961,
            spatial_split="train",
            temporal_split="train",
            path=tmp_path / "first.npz",
            sha256="first",
            contract_sha256="contract",
        ),
        MarkovShardRef(
            landpoint_id="001.0-001.0",
            year=1962,
            spatial_split="train",
            temporal_split="train",
            path=tmp_path / "second.npz",
            sha256="second",
            contract_sha256="contract",
        ),
        MarkovShardRef(
            landpoint_id="002.0-002.0",
            year=1961,
            spatial_split="train",
            temporal_split="train",
            path=tmp_path / "third.npz",
            sha256="third",
            contract_sha256="contract",
        ),
    )
    shard_years = {
        reference.path: reference.year
        for reference in references
    }
    calls = {"load": 0, "make_runtime": 0, "compile_forcing": 0}

    def load(path):
        calls["load"] += 1
        return _FakeShard(year=shard_years[path])

    def make_runtime(*, landpoint_id, batch, **_):
        calls["make_runtime"] += 1
        runtime = rollout_stability_run.LandpointRuntime(
            context=f"context-{landpoint_id}",
            transition=f"transition-{landpoint_id}",
            static={"landpoint_id": landpoint_id},
        )
        return runtime, {"source": "runtime", "batch": batch["forcing_native"]}

    def compile_forcing(batch, _contract, context):
        calls["compile_forcing"] += 1
        return {
            "source": "cached_runtime",
            "context": context,
            "batch": batch["forcing_native"],
        }

    monkeypatch.setattr(rollout_stability_run, "load_markov_shard", load)
    monkeypatch.setattr(
        rollout_stability_run,
        "sample_start_indices",
        lambda **_: (np.asarray([0, 1]), np.asarray([2, 3])),
    )
    monkeypatch.setattr(
        rollout_stability_run,
        "_make_runtime_and_compiled_forcing",
        make_runtime,
    )
    monkeypatch.setattr(
        rollout_stability_run,
        "_compiled_forcing_batch",
        compile_forcing,
    )
    monkeypatch.setattr(
        rollout_stability_run,
        "broadcast_retained_tail_day_inputs",
        lambda forcing, *_args, **_kwargs: forcing,
    )
    monkeypatch.setattr(
        rollout_stability_run,
        "_anchor_training_batch",
        lambda batch, _resources: batch,
    )
    monkeypatch.setattr(
        rollout_stability_run,
        "_sequence",
        lambda _batch, forcing: forcing,
    )
    monkeypatch.setattr(
        rollout_stability_run,
        "retained_tail_trace_signature",
        lambda static: f"trace-{static['landpoint_id']}",
    )
    resources = SimpleNamespace(
        plan_entries={
            reference.landpoint_id: {}
            for reference in references
        },
        config_path=tmp_path / "config.json",
        contract=object(),
        statistics=object(),
        representation=object(),
    )
    runtimes = {}

    first = rollout_stability_run.prepare_rollout_update(
        reference=references[0],
        update=0,
        horizon=2,
        anchor_batch_size=2,
        rollout_batch_size=2,
        seed=7,
        resources=resources,
        runtime_cache=runtimes,
    )
    second = rollout_stability_run.prepare_rollout_update(
        reference=references[1],
        update=1,
        horizon=2,
        anchor_batch_size=2,
        rollout_batch_size=2,
        seed=7,
        resources=resources,
        runtime_cache=runtimes,
    )
    third = rollout_stability_run.prepare_rollout_update(
        reference=references[2],
        update=2,
        horizon=2,
        anchor_batch_size=2,
        rollout_batch_size=2,
        seed=7,
        resources=resources,
        runtime_cache=runtimes,
    )

    assert calls == {"load": 3, "make_runtime": 2, "compile_forcing": 1}
    assert first.runtime is second.runtime
    assert first.runtime is not third.runtime
    assert set(runtimes) == {"001.0-001.0", "002.0-002.0"}
    assert first.sequence["source"] == "runtime"
    assert second.sequence["source"] == "cached_runtime"
    assert third.sequence["source"] == "runtime"
    assert np.array_equal(first.anchor_starts, np.asarray([0, 1]))
    assert np.array_equal(first.rollout_starts, np.asarray([2, 3]))
    assert first.trace_signature == "trace-001.0-001.0"
    assert third.trace_signature == "trace-002.0-002.0"


def test_gradient_calibration_uses_paired_median_ratios_and_skips_zero_rollout():
    protocol = _small_protocol()
    records = []
    for horizon in (1, 3, 7, 30):
        for batch in range(32):
            records.append(
                {
                    "horizon": horizon,
                    "batch": batch,
                    "gradient_norms": {
                        "L_fast": 2.0,
                        "L_next": 4.0,
                        "L_rollout": 0.0 if horizon == 1 else 1.0,
                        "L_bias": 0.5,
                        "L_science": 2.0,
                    },
                }
            )

    calibration = calibrate_gradient_coefficients(records, protocol)

    assert calibration["resolved_coefficients"] == {
        "L_next": 0.5,
        "L_rollout": 2.0,
        "L_bias": 1.0,
        "L_science": 0.5,
    }
    assert calibration["component_audit"]["L_rollout"][
        "positive_ratio_records"
    ] == 96
    assert len(calibration["canonical_sha256"]) == 64

    with pytest.raises(ValueError, match="horizon 30 count drift"):
        calibrate_gradient_coefficients(records[:-1], protocol)


def test_calibration_hard_constraint_failure_records_exact_sample(tmp_path):
    protocol = _small_protocol()
    components = rollout_stability_run.RolloutStabilityComponents(
        L_fast=np.asarray(0.1),
        L_next=np.asarray(0.2),
        L_rollout=np.asarray(0.3),
        L_bias=np.asarray(0.4),
        L_science=np.asarray(0.5),
        unexpected_defined_status_mismatches=np.asarray(3, dtype=np.int32),
        declared_dynamic_status_mismatches=np.asarray(11, dtype=np.int32),
        discrete_state_mismatches=np.asarray(0, dtype=np.int32),
        nonfinite_defined_values=np.asarray(2, dtype=np.int32),
        negative_source_nonnegative_carbon_stocks=np.asarray(7, dtype=np.int32),
    )
    counts = _host_hard_constraint_counts(components)
    reference = MarkovShardRef(
        landpoint_id="301.0-089.0",
        year=1964,
        spatial_split="train",
        temporal_split="train",
        path=tmp_path / "sample.npz",
        sha256="sample-sha",
        contract_sha256="contract",
    )

    failure_path = _write_coefficient_calibration_failure(
        output_root=tmp_path,
        preflight={"training_git_head": "a" * 40},
        protocol=protocol,
        reference=reference,
        ordinal=3,
        total_batches=128,
        horizon=1,
        calibration_seed=20260728,
        trace_signature="trace",
        hard_constraint_counts=counts,
        completed_records=3,
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))

    assert counts == {
        "unexpected_defined_status_mismatches": 3,
        "discrete_state_mismatches": 0,
        "nonfinite_defined_values": 2,
        "negative_source_nonnegative_carbon_stocks": 7,
    }
    assert failure["status"] == "failed_hard_constraint"
    assert failure["batch"] == 4
    assert failure["landpoint_id"] == "301.0-089.0"
    assert failure["year"] == 1964
    assert failure["hard_constraint_counts"] == counts
    assert failure["completed_records"] == 3
    assert failure["sealed_test_used"] is False


def test_compact_index_attribution_requires_one_exact_owner():
    leaves = (
        SimpleNamespace(
            start=0,
            stop=2,
            key="component.first",
            shape=(2,),
            axis_names=("nvm",),
            selected_pft_indices=(0, 13),
        ),
        SimpleNamespace(
            start=2,
            stop=3,
            key="component.second",
            shape=(1,),
            axis_names=(),
            selected_pft_indices=(),
        ),
    )

    assert _leaf_for_compact_index(leaves, 1) == {
        "key": "component.first",
        "offset": 1,
        "shape": [2],
        "axis_names": ["nvm"],
        "selected_pft_indices": [0, 13],
    }
    with pytest.raises(ValueError, match="compact index 3 has 0 owner leaves"):
        _leaf_for_compact_index(leaves, 3)


def _retained_static():
    return {
        "metadata": SimpleNamespace(
            pref_soil_veg=np.asarray([1.0, 2.0]),
            lalo=np.asarray([[3.0, 4.0]]),
            nstm=11,
            is_tree=np.asarray([False, True]),
            is_peat=np.asarray([False, False]),
            npts=1,
        ),
        "stomate_parameter_values": {"value": np.asarray([2.0])},
        "hydrol_table_arrays": ValueTuple(value=np.asarray([3.0])),
        "landpoint_payload": {"value": np.asarray([4.0])},
        "stomate_restart_template": ValueTuple(value=np.asarray([5.0])),
        "stomate_season_values": {"value": np.asarray([6.0])},
        "diffuco_parameter_values": ValueTuple(value=np.asarray([7.0])),
        "mineral_imin": 1,
        "mineral_imax": 11,
        "daily_carbon_dispatch": {"enabled": True, "mode": "compiled"},
        "season": SimpleNamespace(provenance=("restart",)),
    }


def test_dynamic_retained_tail_inputs_broadcast_without_landpoint_static_closure():
    forcing = {"air_temperature": np.ones((3, 7, 48, 1))}
    inputs = broadcast_retained_tail_day_inputs(
        forcing,
        _retained_static(),
        batch_size=3,
        horizon=7,
    )

    assert inputs.compiled_forcing is forcing
    assert inputs.metadata.pref_soil_veg.shape == (3, 7, 2)
    assert inputs.metadata.lalo.shape == (3, 7, 1, 2)
    assert inputs.stomate_parameter_values["value"].shape == (3, 7, 1)
    assert np.all(np.asarray(inputs.metadata.pref_soil_veg[:, :, 0]) == 1.0)

    with pytest.raises(ValueError, match="batch and horizon axes"):
        broadcast_retained_tail_day_inputs(
            {"forcing": np.ones((2, 7, 1))},
            _retained_static(),
            batch_size=3,
            horizon=7,
        )


def test_trace_signature_changes_only_when_bound_dispatch_changes():
    first = _retained_static()
    second = copy.deepcopy(first)
    second["metadata"].pref_soil_veg[:] = 99.0
    second["metadata"].lalo[:] = -10.0
    second["stomate_parameter_values"]["value"][:] = 8.0

    assert retained_tail_trace_signature(first) == retained_tail_trace_signature(
        second
    )
    second["daily_carbon_dispatch"]["mode"] = "different"
    assert retained_tail_trace_signature(first) != retained_tail_trace_signature(
        second
    )


def test_arm_checkpoint_binds_exact_schedule_progress_and_rejects_drift(tmp_path):
    schedule = build_sampling_schedule(_index(tmp_path), _small_protocol(updates=10))
    identity = {"protocol_sha256": "protocol", "calibration_sha256": "calibration"}
    checkpoint = build_arm_checkpoint(
        arm_id="mixed_horizon_stability_v1",
        identity=identity,
        parameters={"weight": np.asarray([1.0])},
        optimizer={"step": np.asarray(4)},
        next_update=4,
        schedule=schedule,
        history=[{"update": 4}],
    )

    assert checkpoint["schema_version"] == ARM_CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["completed_horizon_counts"] == completed_horizon_counts(
        schedule,
        next_update=4,
    )
    verify_arm_checkpoint(
        checkpoint,
        arm_id="mixed_horizon_stability_v1",
        identity=identity,
        schedule=schedule,
    )

    changed = copy.deepcopy(checkpoint)
    changed["next_update"] = 5
    with pytest.raises(ValueError, match="horizon-count drift"):
        verify_arm_checkpoint(
            changed,
            arm_id="mixed_horizon_stability_v1",
            identity=identity,
            schedule=schedule,
        )
    changed = copy.deepcopy(checkpoint)
    changed["schedule_canonical_sha256"] = "drift"
    with pytest.raises(ValueError, match="schedule drift"):
        verify_arm_checkpoint(
            changed,
            arm_id="mixed_horizon_stability_v1",
            identity=identity,
            schedule=schedule,
        )


def test_execution_manifest_binds_calibration_parent_schedule_and_environment(
    tmp_path,
):
    protocol = load_rollout_stability_protocol(PROTOCOL_PATH)
    schedule = {
        "canonical_sha256": "schedule-canonical",
        "entries": [],
        "horizon_counts": {"1": 0, "3": 0, "7": 0, "30": 0},
    }
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    records_path = tmp_path / "records.json"
    records_path.write_text("{}", encoding="utf-8")
    parent_sha = "parent-best"
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "awaiting_coefficient_calibration",
        "training_git_head": rollout_stability_run._current_git_head(),
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": protocol.sha256},
        "parent_architecture": {
            "report": {"path": "report", "sha256": "report"},
            "best_checkpoint": {"path": "best", "sha256": parent_sha},
            "optimizer_checkpoint": {"path": "optimizer", "sha256": "optimizer"},
            "model_architecture": {"id": AXIS_PROCESS_COUPLED_V1},
        },
        "artifacts": {
            "dataset_manifest": {"path": "dataset", "sha256": "dataset"},
            "training_statistics": {"path": "statistics", "sha256": "statistics"},
            "acceptance_report": {"path": "acceptance", "sha256": "acceptance"},
            "training_protocol": {"path": "training", "sha256": "training"},
            "environment_lock": {"path": "lock", "sha256": "lock"},
            "sampling_schedule": {
                "path": str(schedule_path),
                "sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
                "canonical_sha256": schedule["canonical_sha256"],
            },
        },
        "accepted_identity": {},
        "optimizer_update_budget": 8192,
        "screening_seed": 20260728,
        "output_root": str(tmp_path),
        "sealed_test_used": False,
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "passed",
        "protocol_sha256": protocol.sha256,
        "training_git_head": preflight["training_git_head"],
        "train_only": True,
        "resolved_coefficients": {
            "L_next": 1.0,
            "L_rollout": 2.0,
            "L_bias": 0.25,
            "L_science": 0.5,
        },
        "records": {
            "path": str(records_path),
            "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        },
        "parent_best_checkpoint_sha256": parent_sha,
    }
    calibration["canonical_sha256"] = _canonical_digest(calibration)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    execution_path = create_execution_manifest(
        preflight_path=preflight_path,
        calibration_path=calibration_path,
        protocol_path=PROTOCOL_PATH,
        output_root=tmp_path,
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))

    assert execution["schema_version"] == EXECUTION_MANIFEST_SCHEMA_VERSION
    assert execution["status"] == "ready_for_matched_arm_training"
    assert execution["training_git_head"] == preflight["training_git_head"]
    assert execution["artifacts"]["environment_lock"]["sha256"] == "lock"
    assert execution["same_anchor_batches"] is True
    assert execution["sealed_test_used"] is False
    assert execution["coefficient_calibration"]["resolved_coefficients"] == (
        calibration["resolved_coefficients"]
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            rollout_stability_run,
            "_current_git_head",
            lambda: "0" * 40,
        )
        with pytest.raises(ValueError, match="training commit drift"):
            rollout_stability_run._load_verified_execution_manifest(
                execution_path,
                protocol,
            )
        diagnostic_execution = (
            rollout_stability_run._load_verified_execution_manifest(
                execution_path,
                protocol,
                require_current_training_head=False,
            )
        )
        assert diagnostic_execution["training_git_head"] == preflight[
            "training_git_head"
        ]


def test_screening_prefix_gate_cli_requires_an_explicit_stop():
    parser = rollout_stability_run._parser()
    args = parser.parse_args(
        [
            "--phase",
            "screening-prefix-gate",
            "--protocol",
            "protocol.json",
            "--dataset",
            "dataset.json",
            "--statistics",
            "statistics.json",
            "--output-root",
            "output",
            "--execution",
            "execution.json",
            "--stop-after-updates",
            "128",
        ]
    )

    assert args.phase == "screening-prefix-gate"
    assert args.stop_after_updates == 128


def test_screening_update_diagnostic_cli_accepts_candidate_checkpoint():
    parser = rollout_stability_run._parser()
    args = parser.parse_args(
        [
            "--phase",
            "diagnose-screening-update",
            "--protocol",
            "protocol.json",
            "--dataset",
            "dataset.json",
            "--statistics",
            "statistics.json",
            "--output-root",
            "output",
            "--execution",
            "execution.json",
            "--screening-update",
            "302",
            "--parameter-checkpoint",
            "checkpoint.pkl",
        ]
    )

    assert args.screening_update == 302
    assert args.parameter_checkpoint == "checkpoint.pkl"


def _write_parent_assets(tmp_path: Path):
    identity = {"model_architecture": {"id": AXIS_PROCESS_COUPLED_V1}}
    parameters = {"weight": np.asarray([1.0, 2.0], dtype=np.float32)}
    best = tmp_path / "best.pkl"
    best.write_bytes(
        pickle.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": identity,
                "parameters": parameters,
                "epoch": 1,
            }
        )
    )
    optimizer = tmp_path / "checkpoint.pkl"
    optimizer.write_bytes(
        pickle.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": identity,
                "parameters": parameters,
                "optimizer": {"step": np.asarray(1)},
                "completed_epochs": 1,
            }
        )
    )
    return best, optimizer


def test_parent_assets_require_passed_report_matching_best_parameters_and_optimizer(
    tmp_path,
):
    protocol = load_rollout_stability_protocol(PROTOCOL_PATH)
    best, optimizer = _write_parent_assets(tmp_path)
    best_sha = hashlib.sha256(best.read_bytes()).hexdigest()
    report = {
        "schema_version": "canonical_architecture_ab_report_v1",
        "status": "completed",
        "experiment": {
            "sha256": protocol.raw["parent_architecture_screen"][
                "experiment_canonical_sha256"
            ]
        },
        "classification": {
            "status": "passed",
            "decision": "advance_axis_process_to_rollout_stability_experiment",
        },
        "arms": {
            "axis_process": {
                "model_architecture": {"id": AXIS_PROCESS_COUPLED_V1},
                "test_split_evaluated": False,
                "best_checkpoint_sha256": best_sha,
                "best_epoch": 1,
            }
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    accepted = verify_parent_architecture_assets(
        protocol=protocol,
        report_path=report_path,
        best_checkpoint_path=best,
        optimizer_checkpoint_path=optimizer,
    )

    assert accepted["report"]["decision"] == (
        "advance_axis_process_to_rollout_stability_experiment"
    )
    assert accepted["best_checkpoint"]["sha256"] == best_sha

    changed = pickle.loads(optimizer.read_bytes())
    changed["parameters"] = {"weight": np.asarray([9.0, 2.0], dtype=np.float32)}
    optimizer.write_bytes(pickle.dumps(changed))
    with pytest.raises(ValueError, match="best parameters differ"):
        verify_parent_architecture_assets(
            protocol=protocol,
            report_path=report_path,
            best_checkpoint_path=best,
            optimizer_checkpoint_path=optimizer,
        )
