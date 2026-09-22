from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import (
    FLUX_KEYS,
    STOCK_KEYS,
    SharedComponentTrajectories,
)
from scripts.hpc.shared_learning_monitor import (
    FIT_ROLE,
    SharedFitPool,
    build_shared_learning_monitor,
)


class PFT(NamedTuple):
    pft_fraction: object
    active_pft_mask: object
    pft_traits: object


class Inputs(NamedTuple):
    continuous_state: object
    discrete_state: object
    day_index: object
    pft: PFT


def test_shared_fit_pool_uses_fixed_window_zero_and_caches_only_one_artifact(monkeypatch, tmp_path):
    import research.daily_coarse_graining.daily_operator.shared_recurrent_training as recurrent
    import scripts.hpc.shared_transformer_runtime as shared_runtime

    loaded = []
    validated = []

    def load(_manifest, identifiers):
        identifier = int(identifiers[0])
        loaded.append(identifier)
        artifact = SimpleNamespace(
            batch=SimpleNamespace(
                seven_day=(identifier, "seven_day"),
                thirty_day=(identifier, "thirty_day"),
            )
        )
        return (artifact,), {"status": "passed"}

    def validate(index, identifiers, artifacts):
        assert index == "canonical-index" and len(artifacts) == 1
        validated.append(int(identifiers[0]))
        return {"split": "train/train"}

    def prepare(_head, batch, *, window_index, horizon):
        assert window_index == 0
        identifier, component = batch
        assert component == {7: "seven_day", 30: "thirty_day"}[horizon]
        return SimpleNamespace(
            initial_continuous=(identifier, horizon),
            initial_discrete={"flag": identifier},
            days=(component, horizon),
            targets={"state": identifier},
            loss_context=SimpleNamespace(expected_discrete={"flag": horizon}),
        )

    monkeypatch.setattr(shared_runtime, "_load_prepared_artifacts", load)
    monkeypatch.setattr(shared_runtime, "validate_shared_artifact_metadata", validate)
    monkeypatch.setattr(recurrent, "prepare_shared_window", prepare)

    pool = SharedFitPool(
        {"index": "canonical-index"},
        object(),
        tmp_path / "prepared.json",
        range(100, 108),
    )
    assert len(pool.rows) == 16
    assert {row["length"] for row in pool.rows} == {7, 30}
    assert all(row["window_index"] == 0 and row["slice_id"] == FIT_ROLE for row in pool.rows)

    first = pool.case(jax, pool.rows[0])
    second = pool.case(jax, pool.rows[1])
    third = pool.case(jax, pool.rows[2])
    assert first[0] == (100, 7)
    assert second[0] == (100, 30)
    assert third[0] == (101, 7)
    assert loaded == [100, 101]
    assert validated == [100, 101]
    assert pool._cached_id == 101


class _SyntheticPool:
    def __init__(self):
        self.rows = [
            {
                "artifact_id": artifact,
                "component": {7: "seven_day", 30: "thirty_day"}[horizon],
                "length": horizon,
                "window_index": 0,
                "slice_id": FIT_ROLE,
            }
            for artifact in range(8)
            for horizon in (7, 30)
        ]

    def case(self, _jax, row):
        horizon = int(row["length"])
        initial = jnp.asarray([2.0, 1.0, 0.5, *([10.0] * 7)], dtype=jnp.float64)
        pft = PFT(
            pft_fraction=jnp.ones((horizon, 1), dtype=jnp.float64),
            active_pft_mask=jnp.ones((horizon, 1), dtype=jnp.bool_),
            pft_traits=jnp.zeros((horizon, 1, 1), dtype=jnp.float64),
        )
        templates = Inputs(
            continuous_state=jnp.broadcast_to(initial, (horizon, initial.size)),
            discrete_state={"flag": jnp.zeros((horizon,), dtype=jnp.int32)},
            day_index=jnp.arange(100, 100 + horizon, dtype=jnp.int32),
            pft=pft,
        )
        targets = {
            "state": templates.continuous_state,
            "defined": jnp.ones_like(templates.continuous_state, dtype=jnp.bool_),
            "gpp": jnp.full((horizon, 1), 10.0, dtype=jnp.float64),
            "gpp_defined": jnp.ones((horizon, 1), dtype=jnp.bool_),
            "fraction": pft.pft_fraction,
            "active": pft.active_pft_mask,
        }
        return (
            initial,
            {"flag": jnp.asarray(0, dtype=jnp.int32)},
            (templates, jnp.zeros((horizon,), dtype=jnp.float64)),
            targets,
            {"flag": jnp.zeros((horizon,), dtype=jnp.int32)},
        )


def _install_tiny_monitor_dependencies(monkeypatch):
    import research.daily_coarse_graining.daily_operator.canonical_rollout as canonical
    import research.daily_coarse_graining.daily_operator.recurrent_validation as validation
    import research.daily_coarse_graining.daily_operator.shared_trajectory_loss as shared
    import research.daily_coarse_graining.daily_operator.successor_training as successor

    def step_factory(_head, _template):
        def step(parameter, state, discrete, day):
            del day
            stock_delta = jnp.concatenate(
                (jnp.zeros((3,), dtype=state.dtype), jnp.full((7,), parameter, dtype=state.dtype))
            )
            next_state = state + stock_delta
            observations = {
                "state": next_state,
                "defined": jnp.ones_like(next_state, dtype=jnp.bool_),
                "gpp": jnp.asarray([10.0], dtype=jnp.float64),
                "gpp_defined": jnp.asarray([True]),
                "fraction": jnp.asarray([1.0], dtype=jnp.float64),
                "active": jnp.asarray([True]),
                "discrete": discrete,
                "budgets": jnp.zeros((2,), dtype=jnp.float64),
                "learned_output_nonfinite": jnp.asarray(0, dtype=jnp.int32),
            }
            return next_state, discrete, observations

        return step

    def masked(value, defined):
        return MaskedTrajectory(jnp.asarray(value), jnp.asarray(defined, dtype=jnp.bool_))

    def shared_extract(observation, _contract):
        state = jnp.asarray(observation["state"])
        defined = jnp.asarray(observation["defined"], dtype=jnp.bool_)
        gpp = jnp.asarray(observation["gpp"])
        gpp_defined = jnp.asarray(observation["gpp_defined"], dtype=jnp.bool_)
        fluxes = {
            "gpp": masked(gpp, gpp_defined),
            "maintenance_respiration": masked(state[:, 0:1], defined[:, 0:1]),
            "growth_respiration": masked(state[:, 1:2], defined[:, 1:2]),
            "heterotrophic_respiration": masked(state[:, 2:3], defined[:, 2:3]),
        }
        stocks = {
            key: masked(state[:, 3 + index : 4 + index], defined[:, 3 + index : 4 + index])
            for index, key in enumerate(STOCK_KEYS)
        }
        return SharedComponentTrajectories(fluxes=fluxes, stocks=stocks)

    def source_extract(
        state,
        defined,
        *,
        daily_fast_gpp,
        daily_fast_gpp_defined,
        **_kwargs,
    ):
        state = jnp.asarray(state)
        defined = jnp.asarray(defined, dtype=jnp.bool_)
        gpp = jnp.asarray(daily_fast_gpp)
        gpp_defined = jnp.asarray(daily_fast_gpp_defined, dtype=jnp.bool_)
        npp = gpp - state[:, 0:1] - state[:, 1:2]
        npp_defined = gpp_defined & defined[:, 0:1] & defined[:, 1:2]
        return SimpleNamespace(fluxes={"npp": masked(npp, npp_defined)}, stocks={})

    monkeypatch.setattr(canonical, "make_canonical_step", step_factory)
    monkeypatch.setattr(shared, "shared_component_trajectories", shared_extract)
    monkeypatch.setattr(successor, "extract_biomass_productivity_trajectories", source_extract)
    monkeypatch.setattr(
        validation,
        "physical_counts",
        lambda *_args: {
            "state_nonfinite": jnp.asarray(0, dtype=jnp.int32),
            "budget_failure": jnp.asarray(0, dtype=jnp.int32),
        },
    )


def test_monitor_reports_source_npp_units_masks_common_initial_and_negative_stock_gate(monkeypatch):
    _install_tiny_monitor_dependencies(monkeypatch)

    class Assembler:
        def assemble(self, first):
            return SimpleNamespace(
                state_defined=SimpleNamespace(
                    other=jnp.ones_like(first.continuous_state, dtype=jnp.bool_)
                )
            )

    head = SimpleNamespace(contract=object(), input_assembler=Assembler())
    runtime = {"jax": jax}
    pool = _SyntheticPool()
    monitor = build_shared_learning_monitor(
        runtime,
        head,
        object(),
        object(),
        object(),
        memory_limit_bytes=2**30,
    )
    report = monitor(
        jnp.asarray(0.0, dtype=jnp.float64),
        pool,
        pool.rows,
        panel_name="fit_at_update_0",
        split_role=FIT_ROLE,
    )

    assert report["physical_gate_passed"] is True
    assert report["split_role"] == FIT_ROLE
    assert report["scientific_acceptance"] is False
    assert report["accuracy_gates_applied"] is False
    assert report["metadata"]["flux_units"] == "gC m-2 day-1"
    assert report["metadata"]["stock_units"] == "gC m-2"
    assert tuple(report["metadata"]["independent_training_flux_keys"]) == FLUX_KEYS
    assert set(report["metadata"]["stock_keys"]) == set(STOCK_KEYS)

    window = report["windows"][0]
    curves = window["physical_curves"]
    assert curves["fluxes"]["gpp"]["predicted"][0][0] == 10.0
    assert curves["fluxes"]["npp"]["predicted"][0][0] == 7.0
    assert curves["fluxes"]["maintenance_respiration"]["predicted"][0][0] == 2.0
    assert curves["fluxes"]["growth_respiration"]["predicted"][0][0] == 1.0
    assert curves["fluxes"]["autotrophic_respiration"]["predicted"][0][0] == 3.0
    assert curves["fluxes"]["heterotrophic_respiration"]["predicted"][0][0] == 0.5
    assert all(curves["fluxes"]["npp"]["defined"][day][0] for day in range(7))
    assert window["source_npp_identity"]["prediction"]["nonzero_residual_count"] == 0
    assert window["source_npp_identity"]["teacher"]["nonzero_residual_count"] == 0

    stock = curves["stocks"]["aboveground_biomass_carbon"]
    assert stock["initial"] == [10.0]
    assert stock["initial_defined"] == [True]
    assert stock["predicted_change_from_common_initial"][0] == [0.0]
    assert stock["teacher_change_from_common_initial"][0] == [0.0]
    assert report["horizons"]["7"]["window_count"] == 8
    assert report["horizons"]["30"]["window_count"] == 8
    assert (
        report["horizons"]["7"]["physical"]["stocks"]["aboveground_biomass_carbon"]
        ["trajectory_change_gC_m2"]["count"]
        == 56
    )
    assert "state" not in window and "observations" not in window

    rejected = monitor(
        # Reusing the monitor with a different parameter dtype must compile a
        # distinct executable rather than dispatching through the T7/T30 key.
        jnp.asarray(-20.0, dtype=jnp.float32),
        pool,
        pool.rows,
        panel_name="fit_negative_stock",
        split_role=FIT_ROLE,
    )
    assert rejected["physical_gate_passed"] is False
    assert rejected["windows"][0]["hard"]["shared_stock_negative"] > 0
