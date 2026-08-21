from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    build_target_oriented_training_schedule,
    derive_target_oriented_scales,
    initialize_target_oriented_adamw,
    load_target_oriented_state_anchor_policy,
    load_target_oriented_training_contract,
    target_oriented_adamw_update,
    target_oriented_learning_rate,
)


@dataclass(frozen=True)
class Parent:
    landpoint_id: str
    year: int
    sha256: str


@dataclass(frozen=True)
class Reference:
    parent: Parent
    sha256: str
    day_count: int = 364


def _references(count: int = 32, years: tuple[int, ...] = (1961, 1962, 1963)):
    return tuple(
        Reference(
            parent=Parent(
                landpoint_id=f"point-{index:03d}",
                year=year,
                sha256=f"parent-{index:03d}-{year}",
            ),
            sha256=f"typed-{index:03d}-{year}",
        )
        for year in years
        for index in range(count)
    )


def test_target_oriented_training_contract_freezes_diverse_effective_batch():
    contract = load_target_oriented_training_contract()
    assert contract.data["teacher_days_per_update"] == 208
    assert contract.data["distinct_point_years_per_update"] == 28
    assert contract.data["distinct_landpoints_per_update"] == 28
    assert contract.data["common_runtime_year_per_update"] is True
    assert contract.payload["microbatch"] == {
        "one_day_size": 4,
        "seven_day_window_size": 1,
        "thirty_day_window_size": 1,
        "gradient_accumulation_dtype": "float32",
        "loss_reduction_dtype": "float64",
        "optimizer_updates_per_effective_batch": 1,
        "global_bias_reduction": "exact_sufficient_statistics_across_the_full_effective_batch",
        "rollout_rematerialization": True,
        "multi_gpu_required": False,
    }
    assert contract.optimizer["stage_milestones"] == [250, 1000, 3000]
    assert contract.payload["validation"]["promotion"] == ("absolute_target_oriented_7_30_day_gates_only")


def test_target_oriented_adamw_schedule_and_clipping():
    contract = load_target_oriented_training_contract()
    assert target_oriented_learning_rate(contract, 1) == pytest.approx(4.0e-7)
    assert target_oriented_learning_rate(contract, 250) == pytest.approx(1.0e-4)
    assert target_oriented_learning_rate(contract, 3000) == pytest.approx(1.0e-5)

    parameters = {"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}
    gradients = {"weight": jnp.asarray([3.0, 4.0], dtype=jnp.float32)}
    state = initialize_target_oriented_adamw(parameters)
    updated, next_state, norm, scale = target_oriented_adamw_update(
        parameters,
        gradients,
        state,
        learning_rate=target_oriented_learning_rate(contract, 1),
        clip_norm=1.0,
        contract=contract,
    )
    assert float(norm) == pytest.approx(5.0)
    assert float(scale) == pytest.approx(0.2)
    assert int(next_state.step) == 1
    assert np.all(np.isfinite(np.asarray(updated["weight"])))
    assert not np.array_equal(np.asarray(updated["weight"]), np.asarray(parameters["weight"]))


def test_target_oriented_schedule_is_deterministic_diverse_and_complete():
    contract = load_target_oriented_training_contract()
    references = _references()
    first = build_target_oriented_training_schedule(references, contract, updates=2)
    second = build_target_oriented_training_schedule(references, contract, updates=2)
    assert first == second
    assert first["schedule_sha256"] == second["schedule_sha256"]

    for entry in first["entries"]:
        components = entry["components"]
        assert {name: len(values) for name, values in components.items()} == {
            "one_day": 16,
            "seven_day": 8,
            "thirty_day": 4,
        }
        all_records = sum(components.values(), [])
        assert {item["year"] for item in all_records} == {entry["runtime_year"]}
        assert len({(item["landpoint_id"], item["year"]) for item in all_records}) == 28
        assert len({item["landpoint_id"] for item in all_records}) == 28
        assert sum(len(item["start_rows"]) for item in components["one_day"]) == 32
        assert sum(len(item["start_rows"]) for item in components["seven_day"]) == 8
        assert sum(len(item["start_rows"]) for item in components["thirty_day"]) == 4
        assert all(min(item["start_rows"]) >= 1 for item in all_records)
        assert all(max(item["start_rows"]) <= 334 for item in components["thirty_day"])
    assert first["entries"][0]["runtime_year"] != first["entries"][1]["runtime_year"]


def test_target_oriented_state_anchor_amendment_is_frozen_and_reuses_prepared_data():
    policy = load_target_oriented_state_anchor_policy()

    assert policy.sha256 == policy.payload["contract_sha256"]
    assert policy.state_anchor["minimum_resolved_empirical_scale"] == 1.0e-6
    assert policy.state_anchor["unresolved_scale_fallback"] == 1.0
    assert policy.payload["data_and_execution"]["prepared_batch_reuse_allowed"] is True
    assert policy.payload["data_and_execution"]["network_architecture_changed"] is False


def test_target_oriented_schedule_fails_when_landpoint_diversity_is_impossible():
    contract = load_target_oriented_training_contract()
    repeated = tuple(
        Reference(Parent("same-point", 1961 + index, f"parent-{index}"), f"typed-{index}") for index in range(28)
    )
    with pytest.raises(ValueError, match="28 distinct landpoints"):
        build_target_oriented_training_schedule(repeated, contract, updates=1)


def test_target_oriented_thermal_scales_inflate_selected_pft_axis(monkeypatch):
    fields = {
        ("hydrol_previous_step_state", "qsintveg"): ((1, 2), (1, 14), ("npts", "nvm"), (1, 13)),
        ("hydrol_previous_step_state", "mc"): ((3,), (3,), ("soil",), ()),
        ("hydrol_previous_step_state", "snow"): ((1,), (1,), ("snow",), ()),
        ("hydrol_previous_step_state", "snow_nobio"): ((1,), (1,), ("snow",), ()),
        ("hydrol_previous_step_state", "snowliq"): ((1,), (1,), ("snow",), ()),
        ("hydrol_previous_step_state", "flood_res"): ((1,), (1,), ("npts",), ()),
        ("slowproc_stomate_previous_step_state", "litter_above"): (
            (1, 1, 2),
            (1, 1, 14),
            ("npts", "pool", "nvm"),
            (1, 13),
        ),
        ("enerbil_previous_step_state", "temp_sol"): ((1,), (1,), ("npts",), ()),
        ("thermosoil_previous_step_state", "ptn"): (
            (1, 2, 2),
            (1, 2, 14),
            ("npts", "soil", "nvm"),
            (1, 13),
        ),
        ("thermosoil_previous_step_state", "stempdiag"): ((1,), (1,), ("npts",), ()),
        ("hydrol_previous_step_state", "snowtemp"): ((1,), (1,), ("snow",), ()),
        ("hydrol_previous_step_state", "snowheat"): ((1,), (1,), ("snow",), ()),
    }
    leaves = []
    cursor = 0
    for (component, name), (shape, full_shape, axis_names, selected) in fields.items():
        size = int(np.prod(shape))
        leaves.append(
            SimpleNamespace(
                component=component,
                path=(name,),
                shape=shape,
                full_shape=full_shape,
                axis_names=axis_names,
                selected_pft_indices=selected,
                start=cursor,
                stop=cursor + size,
                key=f"{component}.{name}",
            )
        )
        cursor += size
    contract = SimpleNamespace(
        continuous_state_width=cursor,
        state_leaves=tuple(leaves),
    )
    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.target_oriented_training.WATER_INVENTORY_FIELDS",
        ("qsintveg", "mc", "snow", "snow_nobio", "snowliq", "flood_res"),
    )
    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.target_oriented_training.CARBON_INVENTORY_FIELDS",
        ("litter_above",),
    )
    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.target_oriented_training.derive_long_horizon_scales",
        lambda *_args: SimpleNamespace(
            fluxes={
                "gpp": 2.0,
                "maintenance_respiration": 3.0,
                "growth_respiration": 4.0,
                "heterotrophic_respiration": 5.0,
            },
            stocks={
                "aboveground_biomass_carbon": 6.0,
                "belowground_biomass_carbon": 7.0,
                "ecosystem_carbon": 8.0,
                "total_live_biomass_carbon": 9.0,
            },
        ),
    )
    raw = np.arange(1, cursor + 1, dtype=np.float64)
    litter = next(leaf for leaf in leaves if leaf.path == ("litter_above",))
    raw[litter.start] = 1.0e-9
    policy = SimpleNamespace(
        state_anchor={
            "minimum_resolved_empirical_scale": 1.0e-6,
            "unresolved_scale_fallback": 1.0,
            "other_excluded_state_leaves": [
                leaf.key for leaf in leaves if leaf.path != ("snowheat",)
            ],
        },
        payload={"defined_status": {"declared_dynamic_state_leaves": []}},
    )

    scales = derive_target_oriented_scales(
        raw,
        np.asarray(2.0),
        contract,
        state_anchor_policy=policy,
    )

    assert scales.state.thermal.shape == (32,)
    assert scales.state.carbon[0] == 1.0
    assert np.count_nonzero(scales.state_loss_eligible.other) == 1
    assert not np.any(scales.state_declared_dynamic.other)
    ptn_leaf = next(leaf for leaf in leaves if leaf.path == ("ptn",))
    compact = raw[ptn_leaf.start : ptn_leaf.stop].reshape(ptn_leaf.shape)
    inflated = scales.state.thermal[1:29].reshape(ptn_leaf.full_shape)
    np.testing.assert_array_equal(inflated[:, :, 1], compact[:, :, 0])
    np.testing.assert_array_equal(inflated[:, :, 13], compact[:, :, 1])
    inactive = np.delete(inflated, (1, 13), axis=2)
    np.testing.assert_array_equal(inactive, np.ones_like(inactive))
